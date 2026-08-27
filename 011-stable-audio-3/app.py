# -*- coding: utf-8 -*-
"""
011-stable-audio-3 — Stability AI Stable Audio 3 Medium（Modal）

默认策略（省钱）：
  - 模型: stabilityai/stable-audio-3-medium（1.4B DiT · 最长 ~380s · 44.1kHz 立体声）
  - GPU: L4（24GB · ~$0.000222/s）— medium 需 FlashAttn(Ampere+)；T4 不可用；峰值显存 ~5–6.5GB
  - 权重: CPU 下载到 Volume（不计 GPU 费）
  - smoke: 20s · 8 steps · seed 42 · chunked decode
  - 无 keep_warm

上游: https://github.com/Stability-AI/stable-audio-3
权重: https://huggingface.co/stabilityai/stable-audio-3-medium
许可: Stability AI Community License（门禁 · 需 HF_TOKEN + 同意协议）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-stable-audio-3"
UPSTREAM = "https://github.com/Stability-AI/stable-audio-3"
UPSTREAM_COMMIT = "a0b57f5483c4588f827f3552b7d5c6ca2a9687be"

HF_REPO = "stabilityai/stable-audio-3-medium"
DEFAULT_MODEL = "medium"
DEFAULT_GPU = "L4"

REPO_DIR = Path("/opt/stable-audio-3")
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
VOLUME_WEIGHTS = "modal-lab-stable-audio-3-weights"
VOLUME_OUTPUTS = "modal-lab-stable-audio-3-outputs"

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
INFER_TIMEOUT = 30 * 60
SMOKE_TIMEOUT = 20 * 60

FLASH_ATTN_WHL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
    "v0.7.16/flash_attn-2.6.3%2Bcu126torch2.7-cp310-cp310-linux_x86_64.whl"
)

app = modal.App(APP_NAME)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

_HF_ENV = {
    "HF_HOME": str(HF_HOME),
    "HF_HUB_CACHE": str(HF_HOME / "hub"),
    "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
    "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTHONUNBUFFERED": "1",
}

download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(_HF_ENV)
)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
        "libsndfile1-dev",
        "curl",
        "ca-certificates",
        "build-essential",
        "libgomp1",
    )
    .run_commands(
        "python -c \"import sys; print('base', sys.version); assert sys.version_info[:2]==(3,10)\"",
        "pip install -U pip uv",
        f"git clone {UPSTREAM}.git {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {UPSTREAM_COMMIT}",
        f"cd {REPO_DIR} && uv venv --python $(which python) .venv",
        f"cd {REPO_DIR} && .venv/bin/python -c "
        "\"import sys; print('venv', sys.version); assert sys.version_info[:2]==(3,10)\"",
        f"cd {REPO_DIR} && UV_PROJECT_ENVIRONMENT={REPO_DIR}/.venv "
        "uv pip install torch==2.7.1 torchaudio==2.7.1 "
        "--index-url https://download.pytorch.org/whl/cu126",
        f"cd {REPO_DIR} && UV_PROJECT_ENVIRONMENT={REPO_DIR}/.venv "
        "uv sync --inexact --no-dev "
        "--no-install-package torch --no-install-package torchaudio",
        f"cd {REPO_DIR} && UV_PROJECT_ENVIRONMENT={REPO_DIR}/.venv "
        f"uv pip install --no-deps '{FLASH_ATTN_WHL}'",
        f"cd {REPO_DIR} && .venv/bin/python -c "
        "\"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); "
        "import flash_attn; print('flash_attn', flash_attn.__version__); "
        "from stable_audio_3 import StableAudioModel; print('stable_audio_3 ok')\"",
    )
    .env({**_HF_ENV, "SA3_REPO_DIR": str(REPO_DIR)})
)


def _estimate_cost(gpu: str, seconds: float) -> float | None:
    rate = GPU_PRICE_PER_SEC.get(gpu)
    if rate is None:
        return None
    return round(rate * seconds, 4)


def _dir_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    n_files = 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            n_files += 1
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return {
        "path": str(path),
        "exists": True,
        "n_files": n_files,
        "size_gb": round(total / 1e9, 3),
    }


def _weights_ready() -> bool:
    hub = HF_HOME / "hub"
    if not hub.is_dir():
        return False
    for p in hub.rglob("model.safetensors"):
        try:
            if p.stat().st_size > 5_000_000_000:
                return True
        except OSError:
            continue
    return False


def _inject_venv_path() -> None:
    """Modal runtime uses system Python; torch lives in project .venv."""
    import sys

    sites = list((REPO_DIR / ".venv" / "lib").glob("python*/site-packages"))
    for s in sites:
        sp = str(s)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))


def _save_audio(tensor, sample_rate: int, path: Path) -> dict[str, Any]:
    import torch
    import torchaudio

    if tensor.ndim == 3:
        audio = tensor[0].detach().cpu().float()
    elif tensor.ndim == 2:
        audio = tensor.detach().cpu().float()
    else:
        raise ValueError(f"unexpected audio shape {tuple(tensor.shape)}")
    peak = audio.abs().max().clamp(min=1e-8)
    if float(peak) > 1.0:
        audio = audio / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, sample_rate)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]),
        "samples": int(audio.shape[-1]),
        "duration_s": round(audio.shape[-1] / sample_rate, 3),
        "shape": [int(x) for x in tensor.shape],
    }


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    """CPU-only: HF hub cache → Volume（from_pretrained 依赖此 cache）。"""
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    (HF_HOME / "hub").mkdir(parents=True, exist_ok=True)

    local_dir = Path(WEIGHTS_MOUNT) / "models" / "stable-audio-3-medium"
    local_dir.mkdir(parents=True, exist_ok=True)

    if _weights_ready() and not force:
        info = {
            "skipped": True,
            "repo": HF_REPO,
            "ready": True,
            "local": _dir_info(local_dir),
            "hf_home": _dir_info(HF_HOME),
        }
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info

    t0 = time.time()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Downloading {HF_REPO} → hub cache {HF_HOME / 'hub'}", flush=True)

    snapshot_download(
        repo_id=HF_REPO,
        cache_dir=str(HF_HOME / "hub"),
        token=token,
    )
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(local_dir),
        token=token,
    )

    weights_vol.commit()
    info = {
        "skipped": False,
        "repo": HF_REPO,
        "elapsed_s": round(time.time() - t0, 1),
        "ready": _weights_ready(),
        "local": _dir_info(local_dir),
        "hf_home": _dir_info(HF_HOME),
    }
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=10 * 60,
    cpu=1,
    memory=2048,
)
def status_fn() -> dict[str, Any]:
    runs_root = Path(OUTPUTS_MOUNT) / "runs"
    recent = []
    if runs_root.is_dir():
        for p in sorted(
            runs_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        )[:10]:
            if p.is_dir():
                recent.append(p.name)
    out = {
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "hf_repo": HF_REPO,
        "upstream": UPSTREAM,
        "upstream_commit": UPSTREAM_COMMIT,
        "ready": _weights_ready(),
        "weights_local": _dir_info(
            Path(WEIGHTS_MOUNT) / "models" / "stable-audio-3-medium"
        ),
        "hf_home": _dir_info(HF_HOME),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "price_note": "L4 cheapest Ampere+; T4 unsupported (FlashAttn); medium VRAM ~5–6.5GB",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _run_generate(
    *,
    prompt: str,
    duration: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    model_name: str,
    run_name: str,
    audio_format: str,
    gpu_label: str,
    chunked_decode: bool,
    negative_prompt: str,
) -> dict[str, Any]:
    # Modal system Python ≠ project venv — inject site-packages BEFORE torch import
    _inject_venv_path()
    import torch

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")

    if not prompt.strip():
        return {"success": False, "error": "empty prompt"}

    if not _weights_ready():
        return {
            "success": False,
            "error": "weights not on volume — run download first",
            "hf_home": _dir_info(HF_HOME),
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2a_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    t_load0 = time.time()
    from stable_audio_3 import StableAudioModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableAudioModel.from_pretrained(model_name, device=device, model_half=True)
    load_s = time.time() - t_load0

    t_gen0 = time.time()
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "duration": float(duration),
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "seed": int(seed),
        "chunked_decode": bool(chunked_decode),
        "batch_size": 1,
    }
    if negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt
    audio = model.generate(**kwargs)
    gen_s = time.time() - t_gen0

    sr = int(model.model.sample_rate)
    ext = "flac" if audio_format.lower() == "flac" else "wav"
    out_path = save_dir / f"audio.{ext}"
    audio_info = _save_audio(audio, sr, out_path)

    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    wall_s = time.time() - t0
    result = {
        "success": True,
        "error": None,
        "load_s": round(load_s, 2),
        "generate_s": round(gen_s, 2),
        "wall_s": round(wall_s, 2),
        "sample_rate": sr,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "model_name": model_name,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    meta = {
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": round(wall_s, 2),
        "est_gpu_usd": _estimate_cost(gpu_label, wall_s),
        "payload": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "duration": duration,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "model_name": model_name,
            "chunked_decode": chunked_decode,
            "audio_format": audio_format,
        },
        "result": result,
        "created_utc": ts,
        "upstream_commit": UPSTREAM_COMMIT,
        "hf_repo": HF_REPO,
    }
    (save_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs_vol.commit()
    # pure JSON types only (Modal pickle back to local must not require torch)
    meta = json.loads(json.dumps(meta, ensure_ascii=False, default=str))
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return meta


@app.function(
    image=inference_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    secrets=[hf_secret],
    timeout=INFER_TIMEOUT,
    cpu=4,
    memory=16384,
    gpu=DEFAULT_GPU,
    scaledown_window=30,
)
def generate_fn(
    prompt: str = "",
    duration: float = 20.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = 42,
    model_name: str = DEFAULT_MODEL,
    run_name: str = "",
    audio_format: str = "flac",
    gpu_label: str = DEFAULT_GPU,
    chunked_decode: bool = True,
    negative_prompt: str = "",
) -> dict[str, Any]:
    return _run_generate(
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        model_name=model_name,
        run_name=run_name,
        audio_format=audio_format,
        gpu_label=gpu_label,
        chunked_decode=chunked_decode,
        negative_prompt=negative_prompt,
    )


@app.function(
    image=inference_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    secrets=[hf_secret],
    timeout=SMOKE_TIMEOUT,
    cpu=4,
    memory=16384,
    gpu=DEFAULT_GPU,
    scaledown_window=30,
)
def smoke_fn(
    gpu_label: str = DEFAULT_GPU,
    duration: float = 20.0,
    seed: int = 42,
    run_name: str = "smoke_house",
) -> dict[str, Any]:
    return _run_generate(
        prompt=(
            "Uplifting house music instrumental, sunny festival energy, "
            "punchy four-on-the-floor kick, warm pads, 124 BPM"
        ),
        duration=duration,
        steps=8,
        cfg_scale=1.0,
        seed=seed,
        model_name=DEFAULT_MODEL,
        run_name=run_name,
        audio_format="flac",
        gpu_label=gpu_label,
        chunked_decode=True,
        negative_prompt="",
    )


@app.function(
    image=download_image,
    volumes={OUTPUTS_MOUNT: outputs_vol},
    timeout=5 * 60,
    cpu=1,
    memory=2048,
)
def list_outputs_fn() -> dict[str, Any]:
    runs_root = Path(OUTPUTS_MOUNT) / "runs"
    items = []
    if runs_root.is_dir():
        for p in sorted(
            runs_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        ):
            if not p.is_dir():
                continue
            meta_p = p / "meta.json"
            meta = None
            if meta_p.is_file():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                except Exception:
                    meta = None
            items.append(
                {
                    "name": p.name,
                    "meta": {
                        "gpu": (meta or {}).get("gpu_requested"),
                        "wall_s": (meta or {}).get("wall_s"),
                        "est_gpu_usd": (meta or {}).get("est_gpu_usd"),
                        "success": ((meta or {}).get("result") or {}).get("success"),
                    }
                    if meta
                    else None,
                }
            )
    out = {"runs": items, "count": len(items)}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


SMOKE_PROMPT = (
    "Uplifting house music instrumental, sunny festival energy, "
    "punchy four-on-the-floor kick, warm pads, 124 BPM"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="011 Stable Audio 3 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs Volume")
    sub.add_parser("list-outputs", help="结构化汇总远程 run meta")

    download = sub.add_parser("download", help="下载 gated medium 权重")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="固定 20s house · 8 steps benchmark")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--duration", type=float, default=20.0)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--run-name", default="smoke_house")

    t2a = sub.add_parser("t2a", help="Text-to-Audio")
    t2a.add_argument("--dry-run", action="store_true")
    t2a.add_argument("--gpu", default=DEFAULT_GPU)
    t2a.add_argument("--prompt", required=True)
    t2a.add_argument("--negative-prompt", default="")
    t2a.add_argument("--duration", type=float, default=30.0)
    t2a.add_argument("--steps", type=int, default=8)
    t2a.add_argument("--cfg-scale", type=float, default=1.0)
    t2a.add_argument("--seed", type=int, default=42)
    t2a.add_argument("--model", default=DEFAULT_MODEL)
    t2a.add_argument("--format", dest="audio_format", choices=["flac", "wav"], default="flac")
    t2a.add_argument("--run-name", default="")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "011-stable-audio-3",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "hf_repo": HF_REPO,
        "upstream": UPSTREAM,
        "upstream_commit": UPSTREAM_COMMIT,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "gpu_note": "medium requires FlashAttention 2 / Ampere+; T4 unsupported",
    }


def generation_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.duration <= 0:
        raise ValueError("--duration 必须 > 0")
    if args.command == "smoke":
        return {
            "action": "smoke",
            "gpu": args.gpu,
            "prompt": SMOKE_PROMPT,
            "negative_prompt": "",
            "duration": args.duration,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": args.seed,
            "model": DEFAULT_MODEL,
            "audio_format": "flac",
            "run_name": args.run_name,
        }
    prompt = args.prompt.strip()
    if not prompt:
        raise ValueError("prompt 不能为空")
    if args.steps <= 0:
        raise ValueError("--steps 必须 > 0")
    return {
        "action": "t2a",
        "gpu": args.gpu,
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "duration": args.duration,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "model": args.model,
        "audio_format": args.audio_format,
        "run_name": args.run_name,
    }


@app.local_entrypoint()
def main(*argv: str) -> None:
    args = parse_cli(argv)
    if args.command == "status":
        print(json.dumps(local_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "check":
        print(json.dumps(status_fn.remote(), ensure_ascii=False, indent=2))
        return
    if args.command == "list-outputs":
        print(json.dumps(list_outputs_fn.remote(), ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        plan = {"action": "download", "force": args.force}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(download_weights.remote(force=args.force), ensure_ascii=False, indent=2))
        return

    try:
        plan = generation_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if args.command == "smoke":
        out = smoke_fn.with_options(gpu=plan["gpu"]).remote(
            gpu_label=plan["gpu"],
            duration=plan["duration"],
            seed=plan["seed"],
            run_name=plan["run_name"],
        )
    else:
        out = generate_fn.with_options(gpu=plan["gpu"]).remote(
            prompt=plan["prompt"],
            duration=plan["duration"],
            steps=plan["steps"],
            cfg_scale=plan["cfg_scale"],
            seed=plan["seed"],
            model_name=plan["model"],
            run_name=plan["run_name"],
            audio_format=plan["audio_format"],
            gpu_label=plan["gpu"],
            negative_prompt=plan["negative_prompt"],
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
