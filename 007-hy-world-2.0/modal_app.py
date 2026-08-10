# -*- coding: utf-8 -*-
"""
007-hy-world-2.0 — HY-World 2.0 / WorldMirror 2.0 最低成本 Modal 复现。

成本策略（只跑 worldrecon，不跑 80B panogen / 17B worldstereo）：
  - 默认 GPU: T4（16GB · ~$0.000164/s）— smoke 实测 peak ~5GB 够用
  - 备选 L4（24GB · ~$0.000222/s）多视图/更高 res 时更稳
  - 权重 CPU 下载到 Volume（HY-WorldMirror-2.0 子目录，~5GB）
  - smoke: 2 张 Desk 图 · target_size=518 · bf16 · 关 sky/COLMAP/渲染视频
  - 无 keep_warm，scaledown_window=30s
  - 不装 flash-attn；patch attention → SDPA
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-hy-world-2"
HF_REPO = "tencent/HY-World-2.0"
HF_SUBFOLDER = "HY-WorldMirror-2.0"
UPSTREAM = "https://github.com/Tencent-Hunyuan/HY-World-2.0"

DEFAULT_GPU = "T4"
REPO_DIR = Path("/opt/HY-World-2.0")
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
WEIGHTS_CACHE = Path(WEIGHTS_MOUNT) / "huggingface"
CKPT_DIR = Path(WEIGHTS_MOUNT) / "HY-WorldMirror-2.0"
VOLUME_WEIGHTS = "modal-lab-hy-world-2-weights"
VOLUME_OUTPUTS = "modal-lab-hy-world-2-outputs"

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

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(
        {
            "HF_HOME": str(WEIGHTS_CACHE),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
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
        "build-essential",
        "ninja-build",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("ninja", "packaging", "wheel", "setuptools")
    .run_commands(
        "python -m pip install --no-cache-dir 'gsplat' || "
        "python -m pip install --no-cache-dir --no-build-isolation "
        "'git+https://github.com/nerfstudio-project/gsplat.git'",
    )
    .pip_install(
        "numpy==1.26.4",
        "omegaconf",
        "einops",
        "safetensors",
        "huggingface_hub[hf_transfer]>=0.26.0",
        "opencv-python-headless==4.10.0.84",
        "Pillow",
        "scipy==1.14.1",
        "matplotlib==3.10.3",
        "tqdm",
        "loguru==0.7.3",
        "tyro==1.0.8",
        "plyfile",
        "trimesh",
        "imageio[ffmpeg]",
        "kornia",
        "timm==1.0.11",
        "torchmetrics",
        "jaxtyping",
        "rich",
        "easydict",
        "imagesize",
        "scikit-image==0.25.2",
        "requests",
        "onnxruntime",
        "pyyaml",
        "filelock",
        "decord",
    )
    .add_local_file(
        str(EXP_DIR / "patch_attention.py"),
        remote_path="/tmp/patch_attention.py",
        copy=True,
    )
    .run_commands(
        f"git clone --depth 1 {UPSTREAM}.git {REPO_DIR}",
        f"rm -rf {REPO_DIR}/hyworld2/panogen {REPO_DIR}/hyworld2/worldgen "
        f"{REPO_DIR}/assets {REPO_DIR}/.git || true",
        "python /tmp/patch_attention.py",
        "python -c \"import torch, gsplat; print('torch', torch.__version__, "
        "'cuda', torch.version.cuda, 'gsplat', getattr(gsplat, '__version__', '?'))\"",
    )
    .env(
        {
            "HF_HOME": str(WEIGHTS_CACHE),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCH_HOME": str(Path(WEIGHTS_MOUNT) / "torch"),
            "PYTHONPATH": str(REPO_DIR),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(str(EXP_DIR / "examples"), remote_path="/root/examples")
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


def _list_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    marker = CKPT_DIR / "model.safetensors"
    if marker.is_file() and not force:
        info = _dir_info(CKPT_DIR)
        info.update({"skipped": True, "repo": HF_REPO, "subfolder": HF_SUBFOLDER})
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{HF_SUBFOLDER}/*"],
        local_dir=str(Path(WEIGHTS_MOUNT) / "hf_repo"),
        local_dir_use_symlinks=False,
    )
    src = Path(path) / HF_SUBFOLDER
    if src.is_dir():
        if CKPT_DIR.exists():
            shutil.rmtree(CKPT_DIR)
        shutil.copytree(src, CKPT_DIR)
    weights_vol.commit()
    info = _dir_info(CKPT_DIR)
    info.update(
        {
            "skipped": False,
            "repo": HF_REPO,
            "subfolder": HF_SUBFOLDER,
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
    scaledown_window=30,
)
def infer(
    example: str = "Desk",
    max_images: int = 2,
    target_size: int = 518,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    enable_bf16: bool = True,
    save_gs: bool = False,
) -> dict[str, Any]:
    import sys

    import torch

    sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)
    weights_vol.reload()
    os.environ["HF_HOME"] = str(WEIGHTS_CACHE)

    candidates = [
        Path("/root/examples") / example,
        REPO_DIR / "examples" / "worldrecon" / "realistic" / example,
        REPO_DIR / "examples" / "worldrecon" / "stylistic" / example,
        Path(example),
    ]
    img_dir = next((p for p in candidates if p.is_dir()), None)
    if img_dir is None:
        raise FileNotFoundError(f"example not found: {example!r}")

    images = _list_images(img_dir)[: max(1, max_images)]
    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")

    stage = Path("/tmp/hw2_input")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for p in images:
        shutil.copy2(p, stage / p.name)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = run_name.strip() or f"{example}_{stamp}"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe)
    outdir = Path(OUTPUTS_MOUNT) / "runs" / safe
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    price = GPU_PRICE_PER_SEC.get(gpu_label)

    t_all = time.time()
    sampler = VramSampler()
    sampler.start()

    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    t_load = time.time()
    if (CKPT_DIR / "model.safetensors").is_file():
        pipeline = WorldMirrorPipeline.from_pretrained(
            str(CKPT_DIR),
            subfolder="_unused_",
            enable_bf16=enable_bf16,
        )
        load_src = str(CKPT_DIR)
    else:
        pipeline = WorldMirrorPipeline.from_pretrained(
            HF_REPO,
            subfolder=HF_SUBFOLDER,
            enable_bf16=enable_bf16,
        )
        load_src = f"{HF_REPO}/{HF_SUBFOLDER}"
    load_s = time.time() - t_load

    t_fwd = time.time()
    result_dir = pipeline(
        str(stage),
        output_path=str(outdir),
        strict_output_path=str(outdir),
        target_size=target_size,
        fps=1,
        video_max_frames=max_images,
        save_depth=True,
        save_normal=True,
        save_gs=save_gs,
        save_camera=True,
        save_points=True,
        save_colmap=False,
        save_conf=False,
        apply_sky_mask=False,
        apply_edge_mask=True,
        apply_confidence_mask=False,
        save_rendered=False,
        log_time=True,
        compress_pts=True,
        compress_pts_max_points=500_000,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    fwd_s = time.time() - t_fwd

    vram = sampler.stop()
    total_s = time.time() - t_all
    cost = round(total_s * price, 4) if price else None

    outputs = sorted(
        str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()
    )
    n_pts = None
    for ply in outdir.rglob("*.ply"):
        try:
            text = ply.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines()[:40]:
                if line.startswith("element vertex"):
                    n_pts = int(line.split()[-1])
                    break
        except Exception:  # noqa: BLE001
            pass
        if n_pts is not None:
            break

    meta = {
        "ok": True,
        "app": APP_NAME,
        "upstream": UPSTREAM,
        "hf_repo": HF_REPO,
        "hf_subfolder": HF_SUBFOLDER,
        "component": "WorldMirror-2.0 (worldrecon only)",
        "load_src": load_src,
        "example": example,
        "image_dir": str(img_dir),
        "n_images": len(images),
        "image_names": [p.name for p in images],
        "target_size": target_size,
        "enable_bf16": enable_bf16,
        "save_gs": save_gs,
        "n_points_saved": n_pts,
        "pipeline_result_dir": result_dir,
        "gpu_request": gpu_label,
        "gpu_actual": actual_gpu,
        "seconds": {
            "load": round(load_s, 2),
            "forward_and_save": round(fwd_s, 2),
            "total": round(total_s, 2),
        },
        "est_cost_usd": cost,
        "price_per_sec": price,
        "vram": vram,
        "volume": VOLUME_OUTPUTS,
        "run_dir": f"runs/{safe}",
        "outputs": outputs,
        "cost_note": (
            "Default path is WorldMirror 2.0 recon only. "
            "Full HY-World generation (Pano 80B + Stereo 17B) is NOT run."
        ),
    }
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (Path(OUTPUTS_MOUNT) / "runs" / "latest.json").write_text(
        json.dumps(
            {"run_dir": f"runs/{safe}", "meta": meta}, ensure_ascii=False, indent=2
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
                runs.append(
                    {
                        "name": d.name,
                        "has_meta": (d / "meta.json").is_file(),
                        "files": sum(1 for _ in d.rglob("*") if _.is_file()),
                    }
                )
    out = {
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "hf_repo": HF_REPO,
        "component": "WorldMirror-2.0 only (cheapest)",
        "volumes": {"weights": VOLUME_WEIGHTS, "outputs": VOLUME_OUTPUTS},
        "weights": _dir_info(CKPT_DIR),
        "runs": runs[-20:],
        "cost_note": (
            f"default {DEFAULT_GPU} @ ${GPU_PRICE_PER_SEC.get(DEFAULT_GPU)}/s; "
            "smoke=2 imgs@518 bf16 SDPA; full worldgen not included"
        ),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(
    action: str = "status",
    example: str = "Desk",
    max_images: int = 2,
    target_size: int = 518,
    run_name: str = "",
    gpu: str = DEFAULT_GPU,
    force_download: bool = False,
    enable_bf16: bool = True,
    save_gs: bool = False,
) -> None:
    if action == "status":
        print(status.remote())
        return
    if action == "download":
        print(download_weights.remote(force=force_download))
        return
    if action in {"smoke", "infer"}:
        if action == "smoke":
            example = "Desk"
            max_images = min(max_images, 2)
            target_size = min(target_size, 518)
            run_name = run_name or "smoke_desk"
            save_gs = False
            enable_bf16 = True
        print(
            infer.with_options(gpu=gpu).remote(
                example=example,
                max_images=max_images,
                target_size=target_size,
                run_name=run_name,
                gpu_label=gpu,
                enable_bf16=enable_bf16,
                save_gs=save_gs,
            )
        )
        return
    raise SystemExit(f"unknown action={action!r}")
