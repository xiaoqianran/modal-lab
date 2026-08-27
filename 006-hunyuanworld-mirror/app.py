# -*- coding: utf-8 -*-
"""
006-hunyuanworld-mirror — HunyuanWorld-Mirror 最低成本 Modal 复现。

成本策略：
  - 默认 GPU: L4（24GB · 约 $0.000222/s）— 比 A100/H100/PRO6000 便宜数倍
  - 权重下载走 CPU（不计 GPU 费），落在 Volume 复用
  - smoke 仅 2 张示例图，跳过 3DGS 渲染视频 / COLMAP / sky mask
  - 无 keep_warm / 无并行容器

输出只写远程 Volume:
  modal-lab-hunyuanworld-mirror-outputs/runs/<name>/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-hunyuanworld-mirror"
HF_REPO = "tencent/HunyuanWorld-Mirror"
UPSTREAM = "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror"
UPSTREAM_COMMIT = "main"  # shallow clone; pin if needed

DEFAULT_GPU = "L4"  # cheapest reliable fit for ~5GB weights + few views
REPO_DIR = Path("/opt/HunyuanWorld-Mirror")
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
WEIGHTS_CACHE = Path(WEIGHTS_MOUNT) / "huggingface"
CKPT_DIR = Path(WEIGHTS_MOUNT) / "ckpts"
VOLUME_WEIGHTS = "modal-lab-hunyuanworld-mirror-weights"
VOLUME_OUTPUTS = "modal-lab-hunyuanworld-mirror-outputs"

# Modal public GPU $/s (approx, 2026-07 lab notes + pricing page)
GPU_PRICE_PER_SEC = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100": 0.000694,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H100!": 0.001097,
    "RTX-PRO-6000": 0.000842,
}

DOWNLOAD_TIMEOUT = 2 * 60 * 60
INFER_TIMEOUT = 45 * 60

EXP_DIR = Path(__file__).resolve().parent

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

# CPU image: download only
download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(
        {
            "HF_HOME": str(WEIGHTS_CACHE),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# GPU image: torch 2.4 + cu124 + gsplat wheel (official recipe, python 3.10)
inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "wget",
        "curl",
        "ca-certificates",
    )
    .uv_pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # ninja must come from PyPI; gsplat wheel index does not host it
    .uv_pip_install("ninja")
    .uv_pip_install(
        "gsplat",
        index_url="https://docs.gsplat.studio/whl/pt24cu124",
        extra_index_url="https://pypi.org/simple",
    )
    .uv_pip_install(
        # inference-only subset (skip lightning / training stack extras)
        "numpy<2.0.0",
        "einops",
        "trimesh",
        "roma",
        "opencv-python-headless",
        "Pillow",
        "pillow_heif",
        "scipy",
        "matplotlib",
        "tqdm",
        "huggingface_hub[hf_transfer,torch]>=0.26.0",
        "safetensors",
        "plyfile",
        "jaxtyping",
        "tyro",
        "rich",
        "onnxruntime",
        "torchmetrics",
        "colorspacious",
        "moviepy==1.0.3",
        "lpips",
        # RankedLogger import path needs these soft deps at import-time
        "rootutils",
    )
    .run_commands(
        f"git clone --depth 1 {UPSTREAM}.git {REPO_DIR}",
        # keep training/ (vision_transformer imports RankedLogger from it)
        f"rm -rf {REPO_DIR}/.git || true",
        "python -c \"import torch; import gsplat; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gsplat', gsplat.__version__)\"",
    )
    .env(
        {
            "HF_HOME": str(WEIGHTS_CACHE),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCH_HOME": str(Path(WEIGHTS_MOUNT) / "torch"),
            "PYTHONPATH": str(REPO_DIR),
        }
    )
    .add_local_dir(
        str(EXP_DIR / "examples"),
        remote_path="/root/examples",
    )
)

app = modal.App(APP_NAME)


def _dir_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "files": 0, "size_gb": 0.0}
    files = [p for p in path.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {
        "exists": True,
        "path": str(path),
        "files": len(files),
        "size_gb": round(total / 1e9, 3),
    }


def _nvidia_smi() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 4:
        return {"raw": out}
    return {
        "name": parts[0],
        "mem_used_mib": float(parts[1]),
        "mem_total_mib": float(parts[2]),
        "util_gpu_pct": float(parts[3]),
    }


class VramSampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_mib = 0.0
        self.samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        total = self.samples[-1]["mem_total_mib"] if self.samples else None
        name = self.samples[-1]["name"] if self.samples else None
        return {
            "gpu_name_smi": name,
            "peak_mem_used_mib": round(self.peak_used_mib, 1),
            "peak_mem_used_gb": round(self.peak_used_mib / 1024.0, 2)
            if self.peak_used_mib
            else None,
            "mem_total_mib": total,
            "n_samples": len(self.samples),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            q = _nvidia_smi()
            if q and "mem_used_mib" in q:
                self.samples.append(q)
                self.peak_used_mib = max(self.peak_used_mib, q["mem_used_mib"])
            self._stop.wait(self.interval_s)


def _save_scene_ply(path: Path, pts, colors) -> None:
    """Minimal PLY writer (xyz + rgb)."""
    pts_np = pts.detach().float().cpu().numpy() if hasattr(pts, "detach") else pts
    col_np = colors.detach().cpu().numpy() if hasattr(colors, "detach") else colors
    n = pts_np.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts_np[i]
            r, g, b = col_np[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def _list_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    return files


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    """CPU-only HF download → Volume (no GPU cost)."""
    from huggingface_hub import snapshot_download

    marker = CKPT_DIR / "model.safetensors"
    if marker.is_file() and not force:
        info = _dir_info(CKPT_DIR)
        info.update({"skipped": True, "repo": HF_REPO})
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(CKPT_DIR),
        local_dir_use_symlinks=False,
    )
    weights_vol.commit()
    info = _dir_info(CKPT_DIR)
    info.update(
        {
            "skipped": False,
            "repo": HF_REPO,
            "local_dir": path,
            "seconds": round(time.time() - t0, 1),
        }
    )
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


@app.function(
    image=inference_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=INFER_TIMEOUT,
    cpu=4,
    memory=16384,
    gpu=DEFAULT_GPU,
    scaledown_window=30,  # drop GPU ASAP after idle
)
def infer(
    example: str = "Bright_Room",
    max_images: int = 2,
    target_size: int = 518,
    conf_percentile: float = 10.0,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    save_gs: bool = False,
) -> dict[str, Any]:
    """
    Lowest-cost inference path:
      - few views (default 2)
      - no sky mask / COLMAP / interpolated render video
      - optional Gaussian PLY (off by default — saves time & peak VRAM)
    """
    import sys

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)

    # Prefer volume snapshot of weights (offline-friendly after first download)
    weights_vol.reload()
    os.environ["HF_HOME"] = str(WEIGHTS_CACHE)
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

    from src.models.models.worldmirror import WorldMirror
    from src.utils.inference_utils import prepare_images_to_tensor

    # Resolve input images: local examples first, then upstream examples/
    candidates = [
        Path("/root/examples") / example,
        REPO_DIR / "examples" / "realistic" / example,
        REPO_DIR / "examples" / "stylistic" / example,
        Path(example),
    ]
    img_dir = next((p for p in candidates if p.is_dir()), None)
    if img_dir is None:
        raise FileNotFoundError(
            f"example not found: {example!r}; tried {[str(c) for c in candidates]}"
        )

    images = _list_images(img_dir)[: max(1, max_images)]
    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = run_name.strip() or f"{example}_{stamp}"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe)
    outdir = Path(OUTPUTS_MOUNT) / "runs" / safe
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    depth_dir = outdir / "depth"
    normal_dir = outdir / "normal"
    depth_dir.mkdir(exist_ok=True)
    normal_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    price = GPU_PRICE_PER_SEC.get(gpu_label)

    t_all = time.time()
    sampler = VramSampler()
    sampler.start()

    # Load from local ckpt if present, else HF hub (still cached under HF_HOME on volume)
    t_load = time.time()
    if (CKPT_DIR / "model.safetensors").is_file() or (CKPT_DIR / "config.json").is_file():
        model = WorldMirror.from_pretrained(str(CKPT_DIR)).to(device)
        load_src = str(CKPT_DIR)
    else:
        model = WorldMirror.from_pretrained(HF_REPO).to(device)
        load_src = HF_REPO
    model.eval()
    load_s = time.time() - t_load

    # Prepare batch [1, N, 3, H, W]
    t_prep = time.time()
    try:
        imgs = prepare_images_to_tensor(
            [str(p) for p in images], target_size=target_size
        )
    except TypeError:
        from src.utils.inference_utils import load_and_preprocess_images

        imgs = load_and_preprocess_images(
            [str(p) for p in images], output_size=target_size
        )
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)
    imgs = imgs.to(device)
    prep_s = time.time() - t_prep

    inputs: dict[str, Any] = {"img": imgs}
    cond_flags = [0, 0, 0]  # no pose/depth/intrinsics priors

    t_fwd = time.time()
    with torch.no_grad():
        predictions = model(views=inputs, cond_flags=cond_flags)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fwd_s = time.time() - t_fwd

    # --- save geometry ---
    S = imgs.shape[1]
    H, W = imgs.shape[-2], imgs.shape[-1]
    n_pts_saved = 0

    if "pts3d" in predictions:
        pts3d = predictions["pts3d"][0]  # [S,H,W,3]
        conf = predictions["pts3d_conf"][0]  # [S,H,W]
        thr = torch.quantile(conf.reshape(-1).float(), conf_percentile / 100.0)
        mask = conf >= thr
        pts_list = []
        col_list = []
        for i in range(S):
            m = mask[i]
            pts = pts3d[i][m]
            col = (imgs[0, i].permute(1, 2, 0)[m] * 255).to(torch.uint8)
            pts_list.append(pts.reshape(-1, 3))
            col_list.append(col.reshape(-1, 3))
        all_pts = torch.cat(pts_list, dim=0)
        all_cols = torch.cat(col_list, dim=0)
        # Cap PLY size for cheap transfer (keep top conf if huge)
        max_pts = 500_000
        if all_pts.shape[0] > max_pts:
            conf_flat = conf[mask].reshape(-1)
            topk = torch.topk(conf_flat, k=max_pts).indices
            all_pts = all_pts[topk]
            all_cols = all_cols[topk]
        _save_scene_ply(outdir / "pts_from_pointmap.ply", all_pts, all_cols)
        n_pts_saved = int(all_pts.shape[0])

    if "depth" in predictions:
        for i in range(S):
            d = predictions["depth"][0, i, :, :, 0].detach().float().cpu().numpy()
            np.save(depth_dir / f"depth_{i:04d}.npy", d)
            dmin, dmax = float(np.nanpercentile(d, 2)), float(np.nanpercentile(d, 98))
            dn = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
            Image.fromarray((dn * 255).astype(np.uint8)).save(
                depth_dir / f"depth_{i:04d}.png"
            )

    if "normals" in predictions:
        for i in range(S):
            n = predictions["normals"][0, i].detach().float().cpu().numpy()
            vis = ((n * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(vis).save(normal_dir / f"normal_{i:04d}.png")

    cameras = {}
    if "camera_poses" in predictions:
        cameras["c2w"] = predictions["camera_poses"][0].detach().float().cpu().tolist()
    if "camera_intrs" in predictions:
        cameras["K"] = predictions["camera_intrs"][0].detach().float().cpu().tolist()
    if cameras:
        (outdir / "cameras.json").write_text(
            json.dumps(cameras, indent=2), encoding="utf-8"
        )

    if save_gs and "splats" in predictions:
        try:
            from src.utils.save_utils import save_gs_ply

            means = predictions["splats"]["means"][0].reshape(-1, 3)
            scales = predictions["splats"]["scales"][0].reshape(-1, 3)
            quats = predictions["splats"]["quats"][0].reshape(-1, 4)
            colors = (
                predictions["splats"]["sh"][0]
                if "sh" in predictions["splats"]
                else predictions["splats"]["colors"][0]
            ).reshape(-1, 3)
            opacities = predictions["splats"]["opacities"][0].reshape(-1)
            save_gs_ply(
                outdir / "gaussians.ply",
                means,
                scales,
                quats,
                colors,
                opacities,
            )
        except Exception as e:  # noqa: BLE001
            (outdir / "gaussians_error.txt").write_text(repr(e), encoding="utf-8")

    src_out = outdir / "images"
    src_out.mkdir(exist_ok=True)
    for p in images:
        shutil.copy2(p, src_out / p.name)

    vram = sampler.stop()
    total_s = time.time() - t_all
    cost = round(total_s * price, 4) if price else None

    meta = {
        "ok": True,
        "app": APP_NAME,
        "upstream": UPSTREAM,
        "hf_repo": HF_REPO,
        "load_src": load_src,
        "example": example,
        "image_dir": str(img_dir),
        "n_images": len(images),
        "image_names": [p.name for p in images],
        "target_size": target_size,
        "resolution": [H, W],
        "n_points_saved": n_pts_saved,
        "save_gs": save_gs,
        "gpu_request": gpu_label,
        "gpu_actual": actual_gpu,
        "seconds": {
            "load": round(load_s, 2),
            "prep": round(prep_s, 2),
            "forward": round(fwd_s, 2),
            "total": round(total_s, 2),
        },
        "est_cost_usd": cost,
        "price_per_sec": price,
        "vram": vram,
        "volume": VOLUME_OUTPUTS,
        "run_dir": f"runs/{safe}",
        "outputs": sorted(
            str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()
        ),
    }
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest = Path(OUTPUTS_MOUNT) / "runs" / "latest.json"
    latest.write_text(
        json.dumps(
            {"run_dir": f"runs/{safe}", "meta": meta},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs_vol.commit()
    weights_vol.commit()

    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return meta


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=120,
    cpu=1,
    memory=2048,
)
def status() -> dict[str, Any]:
    weights_vol.reload()
    outputs_vol.reload()
    runs = []
    runs_root = Path(OUTPUTS_MOUNT) / "runs"
    if runs_root.is_dir():
        for d in sorted(runs_root.iterdir()):
            if d.is_dir():
                meta = d / "meta.json"
                runs.append(
                    {
                        "name": d.name,
                        "has_meta": meta.is_file(),
                        "files": sum(1 for _ in d.rglob("*") if _.is_file()),
                    }
                )
    out = {
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "hf_repo": HF_REPO,
        "volumes": {
            "weights": VOLUME_WEIGHTS,
            "outputs": VOLUME_OUTPUTS,
        },
        "weights": _dir_info(CKPT_DIR),
        "runs": runs[-20:],
        "cost_note": (
            f"default {DEFAULT_GPU} @ ${GPU_PRICE_PER_SEC.get(DEFAULT_GPU)}/s; "
            "download is CPU-only; smoke uses 2 images, no GS video"
        ),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="006 HunyuanWorld-Mirror on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs Volume")

    download = sub.add_parser("download", help="CPU 下载 HF 权重到 Volume")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="最低成本 Bright_Room·2图")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--max-images", type=int, default=2)

    infer_cmd = sub.add_parser("infer", help="对示例目录做推理")
    infer_cmd.add_argument("--dry-run", action="store_true")
    infer_cmd.add_argument("--example", default="Bright_Room")
    infer_cmd.add_argument("--max-images", type=int, default=2)
    infer_cmd.add_argument("--target-size", type=int, default=518)
    infer_cmd.add_argument("--run-name", default="")
    infer_cmd.add_argument("--gpu", default=DEFAULT_GPU)
    infer_cmd.add_argument("--save-gs", action="store_true")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "006-hunyuanworld-mirror",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "hf_repo": HF_REPO,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "smoke": {"example": "Bright_Room", "max_images": 2, "target_size": 518, "save_gs": False},
    }


def inference_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "smoke":
        return {
            "action": "smoke",
            "example": "Bright_Room",
            "max_images": min(max(1, args.max_images), 2),
            "target_size": 518,
            "run_name": "smoke_bright_room",
            "gpu": args.gpu,
            "save_gs": False,
        }
    if args.max_images <= 0:
        raise ValueError("--max-images 必须 > 0")
    if args.target_size <= 0:
        raise ValueError("--target-size 必须 > 0")
    return {
        "action": "infer",
        "example": args.example,
        "max_images": args.max_images,
        "target_size": args.target_size,
        "run_name": args.run_name,
        "gpu": args.gpu,
        "save_gs": args.save_gs,
    }


@app.local_entrypoint()
def main(*argv: str) -> None:
    args = parse_cli(argv)
    if args.command == "status":
        print(json.dumps(local_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "check":
        print(json.dumps(status.remote(), ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        plan = {"action": "download", "force": args.force}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(download_weights.remote(force=args.force), ensure_ascii=False, indent=2))
        return

    try:
        plan = inference_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    print(
        infer.with_options(gpu=plan["gpu"]).remote(
            example=plan["example"],
            max_images=plan["max_images"],
            target_size=plan["target_size"],
            run_name=plan["run_name"],
            gpu_label=plan["gpu"],
            save_gs=plan["save_gs"],
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
