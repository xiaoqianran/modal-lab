# -*- coding: utf-8 -*-
"""
008-hy-pano — HY-World 2.0 / HY-Pano 2.0 全景生成（Modal）

默认：轻量 Backend B（Qwen-Image-Edit + LoRA）。
全量 ~80B 需显式 --backend full（默认禁用）。

详见 PLAN.md。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-hy-pano"
UPSTREAM = "https://github.com/Tencent-Hunyuan/HY-World-2.0"
HF_REPO = "tencent/HY-World-2.0"
HF_PANO_SUBFOLDER = "HY-Pano-2.0"
QWEN_BASE = "Qwen/Qwen-Image-Edit-2509"
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

DEFAULT_BACKEND = "qwen"
DEFAULT_GPU_QWEN = "RTX-PRO-6000"
DEFAULT_GPU_FULL = "H100:4"

REPO_DIR = Path("/opt/HY-World-2.0")
PANO_DIR = REPO_DIR / "hyworld2" / "panogen"
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
WEIGHTS_CACHE = Path(WEIGHTS_MOUNT) / "huggingface"
FULL_CKPT = Path(WEIGHTS_MOUNT) / "HY-Pano-2.0"
QWEN_CKPT = Path(WEIGHTS_MOUNT) / "Qwen-Image-Edit-2509"
LORA_DIR = Path(WEIGHTS_MOUNT) / "HY-Pano-2.0-lora"

VOLUME_WEIGHTS = "modal-lab-hy-pano-weights"
VOLUME_OUTPUTS = "modal-lab-hy-pano-outputs"

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
    "H200": 0.001536,
    "RTX-PRO-6000": 0.000842,
    "H100:2": 0.001097 * 2,
    "H100:3": 0.001097 * 3,
    "H100:4": 0.001097 * 4,
    "A100-80GB:2": 0.000694 * 2,
    "A100-80GB:3": 0.000694 * 3,
    "A100-80GB:4": 0.000694 * 4,
}

DOWNLOAD_TIMEOUT = 4 * 60 * 60
INFER_TIMEOUT = 2 * 60 * 60

EXP_DIR = Path(__file__).resolve().parent

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

app = modal.App(APP_NAME)

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
        "libgomp1",
        "wget",
        "curl",
        "ca-certificates",
        "build-essential",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "einops==0.8.1",
        "numpy==2.2.0",
        "pillow",
        "diffusers==0.36.0",
        "safetensors==0.7.0",
        "tokenizers==0.22.0",
        "transformers[accelerate,tiktoken]==4.57.1",
        "huggingface_hub[hf_transfer,cli]>=0.26.0",
        "loguru>=0.7.3",
        "accelerate",
        "peft",  # required by diffusers load_lora_weights
        "sentencepiece",
        "protobuf",
        "opencv-python-headless",
        "tqdm",
        "bitsandbytes",
    )
    .run_commands(
        f"git clone --depth 1 {UPSTREAM} {REPO_DIR} && "
        f"rm -rf {REPO_DIR}/hyworld2/worldgen {REPO_DIR}/hyworld2/worldrecon "
        f"{REPO_DIR}/examples/worldrecon || true",
    )
    .env(
        {
            "HF_HOME": str(WEIGHTS_CACHE),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(PANO_DIR),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(str(EXP_DIR / "examples"), remote_path="/root/examples")
)


def _price_key(gpu: str) -> str:
    return gpu if gpu in GPU_PRICE_PER_SEC else gpu.split(":")[0]


def _est_cost(gpu: str, seconds: float) -> float:
    unit = GPU_PRICE_PER_SEC.get(gpu) or GPU_PRICE_PER_SEC.get(_price_key(gpu), 0.0)
    return round(unit * seconds, 4)


def _vram_sampler(stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            for i, line in enumerate(out.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    samples.append(
                        {
                            "gpu_index": i,
                            "name": parts[0],
                            "used_mib": float(parts[1]),
                            "total_mib": float(parts[2]),
                            "t": time.time(),
                        }
                    )
        except Exception:
            pass
        stop.wait(0.5)


def _peak_vram(samples: list[dict]) -> dict[str, Any]:
    if not samples:
        return {}
    by: dict[int, list] = {}
    for s in samples:
        by.setdefault(s["gpu_index"], []).append(s)
    peaks = []
    for i, rows in sorted(by.items()):
        peak = max(rows, key=lambda r: r["used_mib"])
        peaks.append(
            {
                "gpu_index": i,
                "name": peak["name"],
                "peak_mem_used_mib": peak["used_mib"],
                "peak_mem_used_gb": round(peak["used_mib"] / 1024, 2),
                "mem_total_mib": peak["total_mib"],
            }
        )
    return {
        "per_gpu": peaks,
        "peak_mem_used_gb_max": max(p["peak_mem_used_gb"] for p in peaks),
        "n_samples": len(samples),
    }


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=16384,
)
def download_weights(backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    backend = backend.lower().strip()
    t0 = time.time()
    result: dict[str, Any] = {"ok": True, "backend": backend, "paths": {}}

    if backend in ("qwen", "both"):
        print(f"[download] Qwen base → {QWEN_CKPT}")
        snapshot_download(
            QWEN_BASE,
            local_dir=str(QWEN_CKPT),
            local_dir_use_symlinks=False,
        )
        print(f"[download] LoRA → {LORA_DIR}")
        LORA_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            HF_REPO,
            allow_patterns=[f"{HF_PANO_SUBFOLDER}/{LORA_WEIGHT_NAME}"],
            local_dir=str(Path(WEIGHTS_MOUNT) / "_hf_pano_lora"),
            local_dir_use_symlinks=False,
        )
        src = Path(WEIGHTS_MOUNT) / "_hf_pano_lora" / HF_PANO_SUBFOLDER / LORA_WEIGHT_NAME
        dst = LORA_DIR / LORA_WEIGHT_NAME
        if src.is_file():
            shutil.copy2(src, dst)
        result["paths"]["qwen_base"] = str(QWEN_CKPT)
        result["paths"]["lora"] = str(dst)

    if backend in ("full", "both"):
        print(f"[download] Full HY-Pano-2.0 (~169GB) → {FULL_CKPT}")
        snapshot_download(
            HF_REPO,
            allow_patterns=[f"{HF_PANO_SUBFOLDER}/*"],
            local_dir=str(Path(WEIGHTS_MOUNT) / "_hf_pano_full"),
            local_dir_use_symlinks=False,
        )
        src_root = Path(WEIGHTS_MOUNT) / "_hf_pano_full" / HF_PANO_SUBFOLDER
        if FULL_CKPT.exists():
            shutil.rmtree(FULL_CKPT)
        shutil.copytree(src_root, FULL_CKPT)
        result["paths"]["full"] = str(FULL_CKPT)

    weights_vol.commit()
    result["seconds"] = round(time.time() - t0, 2)
    result["volume"] = VOLUME_WEIGHTS
    return result


def _resolve_example(image_name: str) -> Path:
    candidates = [
        Path("/root/examples") / image_name,
        Path(image_name),
    ]
    for c in candidates:
        if c.is_file():
            return c
    for ext in (".jpg", ".png", ".jpeg", ".webp"):
        p = Path("/root/examples") / (
            image_name if image_name.endswith(ext) else f"{image_name}{ext}"
        )
        if p.is_file():
            return p
    raise FileNotFoundError(f"image not found: {image_name}")


def _run_qwen(
    image_path: Path,
    out_path: Path,
    prompt: str,
    seed: int | None,
    height: int,
    width: int,
    num_inference_steps: int,
    load_mode: str,
) -> dict[str, Any]:
    """load_mode: gpu | cpu_offload | sequential_offload"""
    import sys

    import torch

    sys.path.insert(0, str(PANO_DIR))
    from pipeline_with_qwen_image import HunyuanPanoPipeline  # type: ignore
    from qwen_image import PanoDiffusionPipeline  # type: ignore

    print(f"[qwen] load_mode={load_mode} dtype=bf16")
    t_load = time.time()
    pipe = PanoDiffusionPipeline.from_pretrained(
        str(QWEN_CKPT),
        torch_dtype=torch.bfloat16,
    )
    lora_file = LORA_DIR / LORA_WEIGHT_NAME
    if not lora_file.is_file():
        found = list(LORA_DIR.rglob(LORA_WEIGHT_NAME))
        if not found:
            raise FileNotFoundError(f"LoRA missing under {LORA_DIR}")
        lora_file = found[0]
    print(f"[qwen] loading LoRA {lora_file}")
    pipe.load_lora_weights(
        str(lora_file.parent),
        weight_name=lora_file.name,
        torch_dtype=torch.bfloat16,
    )

    if load_mode == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
    elif load_mode == "cpu_offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    load_s = time.time() - t_load
    print(f"[qwen] loaded in {load_s:.1f}s")

    wrapper = HunyuanPanoPipeline(pipe)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
    }
    if seed is not None:
        kwargs["seed"] = seed

    t_fwd = time.time()
    output = wrapper(str(image_path), **kwargs)
    output.save(str(out_path))
    fwd_s = time.time() - t_fwd
    return {"load_s": round(load_s, 2), "forward_s": round(fwd_s, 2), "load_mode": load_mode}


def _run_full(
    image_path: Path,
    out_path: Path,
    prompt: str,
    seed: int | None,
    height: int,
    width: int,
    diff_infer_steps: int,
    use_taylor_cache: bool,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(PANO_DIR))
    from pipeline import HunyuanPanoPipeline  # type: ignore

    t0 = time.time()
    pipe = HunyuanPanoPipeline.from_pretrained(
        str(FULL_CKPT),
        subfolder="",
        attn_impl="sdpa",
        moe_impl="eager",
    )
    load_s = time.time() - t0
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "diff_infer_steps": diff_infer_steps,
        "use_taylor_cache": use_taylor_cache,
    }
    if seed is not None:
        kwargs["seed"] = seed
    t1 = time.time()
    output = pipe(str(image_path), **kwargs)
    output.save(str(out_path))
    return {"load_s": round(load_s, 2), "forward_s": round(time.time() - t1, 2)}


@app.function(
    image=inference_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=INFER_TIMEOUT,
    gpu=DEFAULT_GPU_QWEN,
    memory=65536,
)
def infer_pano(
    backend: str = DEFAULT_BACKEND,
    image_name: str = "desk.jpg",
    prompt: str = "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style.",
    seed: int | None = 42,
    height: int = 960,
    width: int = 1952,
    diff_infer_steps: int = 40,
    use_taylor_cache: bool = False,
    load_mode: str = "gpu",
    run_name: str | None = None,
    gpu_label: str = DEFAULT_GPU_QWEN,
) -> dict[str, Any]:
    backend = backend.lower().strip()
    if backend not in ("qwen", "full"):
        raise ValueError("backend must be 'qwen' or 'full'")
    load_mode = load_mode.lower().strip()
    if load_mode not in ("gpu", "cpu_offload", "sequential_offload"):
        raise ValueError("load_mode must be gpu|cpu_offload|sequential_offload")

    if backend == "qwen" and load_mode == "gpu":
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True,
            )
            total_mib = float(out.strip().splitlines()[0].strip())
            if total_mib < 70_000:
                load_mode = "cpu_offload"
                print(f"[auto] GPU {total_mib} MiB < 70GiB → load_mode=cpu_offload")
        except Exception as e:
            print(f"[auto] nvidia-smi probe failed: {e}")

    image_path = _resolve_example(image_name)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = run_name or f"{backend}_{Path(image_name).stem}_{ts}"
    run_dir = Path(OUTPUTS_MOUNT) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "panorama.png"
    shutil.copy2(image_path, run_dir / f"input{image_path.suffix}")

    stop = threading.Event()
    samples: list[dict] = []
    th = threading.Thread(target=_vram_sampler, args=(stop, samples), daemon=True)
    th.start()

    t0 = time.time()
    timing: dict[str, Any] = {}
    try:
        if backend == "qwen":
            if not QWEN_CKPT.exists() or not any(QWEN_CKPT.iterdir()):
                raise FileNotFoundError(
                    f"Qwen weights missing at {QWEN_CKPT}; run download --backend qwen"
                )
            timing = _run_qwen(
                image_path,
                out_path,
                prompt,
                seed,
                height,
                width,
                diff_infer_steps,
                load_mode,
            )
        else:
            if not FULL_CKPT.is_dir() or not any(FULL_CKPT.glob("*.safetensors")):
                raise FileNotFoundError(
                    f"Full HY-Pano weights missing at {FULL_CKPT}; run download --backend full"
                )
            timing = _run_full(
                image_path,
                out_path,
                prompt,
                seed,
                height,
                width,
                diff_infer_steps,
                use_taylor_cache,
            )
    finally:
        stop.set()
        th.join(timeout=2)

    total = time.time() - t0
    vram = _peak_vram(samples)
    meta = {
        "ok": True,
        "app": APP_NAME,
        "upstream": UPSTREAM,
        "component": "HY-Pano 2.0",
        "backend": backend,
        "image": str(image_path),
        "image_name": image_name,
        "prompt": prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_inference_steps": diff_infer_steps,
        "load_mode": load_mode if backend == "qwen" else None,
        "use_taylor_cache": use_taylor_cache if backend == "full" else None,
        "gpu_request": gpu_label,
        "seconds": {
            "total": round(total, 2),
            "load": timing.get("load_s"),
            "forward": timing.get("forward_s"),
        },
        "est_cost_usd": _est_cost(gpu_label, total),
        "price_per_sec": GPU_PRICE_PER_SEC.get(gpu_label)
        or GPU_PRICE_PER_SEC.get(_price_key(gpu_label)),
        "vram": vram,
        "volume": VOLUME_OUTPUTS,
        "run_dir": f"runs/{run_name}",
        "outputs": ["panorama.png", f"input{image_path.suffix}", "meta.json"],
        "cost_note": "Lightweight path = Qwen+LoRA only. Full 80B disabled unless --backend full.",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    outputs_vol.commit()
    latest = Path(OUTPUTS_MOUNT) / "runs" / "latest.json"
    latest.write_text(
        json.dumps({"run_dir": f"runs/{run_name}", "backend": backend}, indent=2)
    )
    outputs_vol.commit()
    return meta


BACKEND_CHOICES = ("qwen", "full")
DOWNLOAD_BACKEND_CHOICES = ("qwen", "full", "both")
LOAD_MODE_CHOICES = ("gpu", "cpu_offload", "sequential_offload")
DEFAULT_PROMPT = "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="008 HY-Pano 2.0 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")

    download = sub.add_parser("download", help="下载 Qwen+LoRA / full 权重")
    download.add_argument("--backend", choices=DOWNLOAD_BACKEND_CHOICES, default=DEFAULT_BACKEND)
    download.add_argument("--dry-run", action="store_true")

    for name in ("smoke", "infer"):
        cmd = sub.add_parser(name, help="样例全景" if name == "smoke" else "自定义全景推理")
        cmd.add_argument("--dry-run", action="store_true")
        cmd.add_argument("--backend", choices=BACKEND_CHOICES, default=DEFAULT_BACKEND)
        cmd.add_argument("--gpu", default="", help="为空时按 backend 选择默认 GPU")
        cmd.add_argument("--image", default="desk.jpg")
        cmd.add_argument("--prompt", default=DEFAULT_PROMPT)
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--height", type=int, default=960)
        cmd.add_argument("--width", type=int, default=1952)
        cmd.add_argument("--steps", type=int, default=40)
        cmd.add_argument("--load-mode", choices=LOAD_MODE_CHOICES, default="gpu")
        cmd.add_argument("--use-taylor-cache", action="store_true")
        cmd.add_argument("--run-name", default="")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def default_gpu_for(backend: str) -> str:
    return DEFAULT_GPU_FULL if backend == "full" else DEFAULT_GPU_QWEN


def local_status() -> dict[str, Any]:
    return {
        "experiment": "008-hy-pano",
        "app": APP_NAME,
        "upstream": UPSTREAM,
        "default_backend": DEFAULT_BACKEND,
        "default_gpu_qwen": DEFAULT_GPU_QWEN,
        "default_gpu_full": DEFAULT_GPU_FULL,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "note": "Qwen+LoRA 默认；full 约 80B / 多 GPU，显式 --backend full",
    }


def inference_plan(args: argparse.Namespace) -> dict[str, Any]:
    gpu = args.gpu or default_gpu_for(args.backend)
    run_name = args.run_name or (f"smoke_{args.backend}" if args.command == "smoke" else "")
    return {
        "action": args.command,
        "backend": args.backend,
        "gpu": gpu,
        "image": args.image,
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "load_mode": args.load_mode,
        "use_taylor_cache": args.use_taylor_cache,
        "run_name": run_name,
    }


@app.local_entrypoint()
def main(*argv: str) -> None:
    args = parse_cli(argv)
    if args.command == "status":
        print(json.dumps(local_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        plan = {"action": "download", "backend": args.backend}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(download_weights.remote(backend=args.backend), ensure_ascii=False, indent=2))
        return

    plan = inference_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if plan["backend"] == "full":
        print("[warn] full backend is expensive (multi-GPU + ~169GB). Continuing…", flush=True)
    fn = infer_pano.with_options(gpu=plan["gpu"])
    meta = fn.remote(
        backend=plan["backend"],
        image_name=plan["image"],
        prompt=plan["prompt"],
        seed=plan["seed"],
        height=plan["height"],
        width=plan["width"],
        diff_infer_steps=plan["steps"],
        use_taylor_cache=plan["use_taylor_cache"],
        load_mode=plan["load_mode"],
        run_name=plan["run_name"] or None,
        gpu_label=plan["gpu"],
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
