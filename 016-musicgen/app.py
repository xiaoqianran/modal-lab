# -*- coding: utf-8 -*-
"""
016-musicgen — Meta MusicGen (transformers) on Modal

默认策略（最便宜基线）：
  - 模型: facebook/musicgen-small（300M · text→music · 32kHz）
  - GPU: T4（$0.000164/s）— small 短音频够用
  - 可选: medium @ L4
  - 权重 CPU 下载到 Volume
  - smoke: 15s · seed 42 · 器乐 prompt

上游: https://github.com/facebookresearch/audiocraft
推理: HuggingFace transformers MusicgenForConditionalGeneration
许可: CC-BY-NC 4.0（非商用）
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

APP_NAME = "modal-lab-musicgen"
HF_REPOS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}
DEFAULT_MODEL = "small"
DEFAULT_GPU = "T4"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
VOLUME_WEIGHTS = "modal-lab-musicgen-weights"
VOLUME_OUTPUTS = "modal-lab-musicgen-outputs"

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
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(_HF_ENV)
)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ffmpeg", "libsndfile1", "git", "ca-certificates")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.39.0,<4.48.0",
        "accelerate>=0.28.0",
        "scipy",
        "soundfile",
        "numpy",
        "huggingface_hub>=0.26.0",
    )
    .env(_HF_ENV)
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


def _norm_model(name: str) -> str:
    n = (name or DEFAULT_MODEL).strip().lower()
    if n in HF_REPOS:
        return n
    for k, repo in HF_REPOS.items():
        if n == repo or n.endswith(k):
            return k
    raise ValueError(f"unknown model {name!r}; use {list(HF_REPOS)}")


def _weights_ready(model_key: str) -> bool:
    repo = HF_REPOS[model_key]
    local = Path(WEIGHTS_MOUNT) / "models" / model_key
    if (local / "config.json").is_file():
        return True
    hub = HF_HOME / "hub"
    if not hub.is_dir():
        return False
    slug = "models--" + repo.replace("/", "--")
    return any(hub.glob(f"{slug}/**/config.json"))


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    key = _norm_model(model)
    repo = HF_REPOS[key]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    local = Path(WEIGHTS_MOUNT) / "models" / key
    local.mkdir(parents=True, exist_ok=True)

    if _weights_ready(key) and not force:
        info = {
            "skipped": True,
            "model": key,
            "repo": repo,
            "ready": True,
            "local": _dir_info(local),
            "hf_home": _dir_info(HF_HOME),
        }
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return info

    t0 = time.time()
    print(f"Downloading {repo} → {local} + hub cache", flush=True)
    snapshot_download(
        repo_id=repo,
        cache_dir=str(HF_HOME / "hub"),
        token=token,
    )
    snapshot_download(
        repo_id=repo,
        local_dir=str(local),
        token=token,
    )
    weights_vol.commit()
    info = {
        "skipped": False,
        "model": key,
        "repo": repo,
        "elapsed_s": round(time.time() - t0, 1),
        "ready": _weights_ready(key),
        "local": _dir_info(local),
        "hf_home": _dir_info(HF_HOME),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
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
        "slot": "016-musicgen",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "license": "CC-BY-NC 4.0 (non-commercial)",
        "models": {
            k: {
                "repo": r,
                "ready": _weights_ready(k),
                **_dir_info(Path(WEIGHTS_MOUNT) / "models" / k),
            }
            for k, r in HF_REPOS.items()
            if k in ("small", "medium")
        },
        "hf_home": _dir_info(HF_HOME),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "gpu_note": "T4 cheapest for small; medium prefers L4",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _run_generate(
    *,
    prompt: str,
    duration: float,
    seed: int,
    model: str,
    run_name: str,
    gpu_label: str,
    guidance_scale: float,
    temperature: float,
) -> dict[str, Any]:
    import torch
    import torchaudio
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")

    key = _norm_model(model)
    repo = HF_REPOS[key]
    if not prompt.strip():
        return {"success": False, "error": "empty prompt"}
    if not _weights_ready(key):
        return {"success": False, "error": f"weights missing for {key} — run download"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2a_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    local = Path(WEIGHTS_MOUNT) / "models" / key
    load_id = str(local) if (local / "config.json").is_file() else repo

    t0 = time.time()
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(load_id)
    model_obj = MusicgenForConditionalGeneration.from_pretrained(load_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_obj = model_obj.to(device)
    model_obj.eval()
    load_s = time.time() - t_load

    max_new_tokens = max(32, int(float(duration) * 50))
    if seed is not None and seed >= 0:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t_gen = time.time()
    with torch.no_grad():
        audio_values = model_obj.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=float(guidance_scale),
            temperature=float(temperature),
        )
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    wav = audio_values[0].detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    sr = int(model_obj.config.audio_encoder.sampling_rate)
    out_path = save_dir / "audio.wav"
    torchaudio.save(str(out_path), wav, sr)

    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    audio_info = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sample_rate": sr,
        "channels": int(wav.shape[0]),
        "samples": int(wav.shape[-1]),
        "duration_s": round(wav.shape[-1] / sr, 3),
    }
    result = {
        "success": True,
        "error": None,
        "wall_s": round(wall, 2),
        "load_s": round(load_s, 2),
        "generate_s": round(gen_s, 2),
        "est_gpu_usd": _estimate_cost(gpu_label, wall),
        "gpu": gpu_label,
        "model": key,
        "hf_repo": repo,
        "max_new_tokens": max_new_tokens,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "prompt": prompt,
        "seed": seed,
        "duration_requested": duration,
        "guidance_scale": guidance_scale,
        "temperature": temperature,
    }
    meta = {
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "prompt": prompt,
            "duration": duration,
            "seed": seed,
            "model": key,
            "guidance_scale": guidance_scale,
            "temperature": temperature,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "CC-BY-NC 4.0 — non-commercial",
    }
    (save_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs_vol.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=inference_image,
    gpu=DEFAULT_GPU,
    timeout=INFER_TIMEOUT,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    secrets=[hf_secret],
    memory=16384,
)
def generate_fn(
    prompt: str,
    duration: float = 15.0,
    seed: int = 42,
    model: str = DEFAULT_MODEL,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    guidance_scale: float = 3.0,
    temperature: float = 1.0,
) -> dict[str, Any]:
    return _run_generate(
        prompt=prompt,
        duration=duration,
        seed=seed,
        model=model,
        run_name=run_name,
        gpu_label=gpu_label,
        guidance_scale=guidance_scale,
        temperature=temperature,
    )


SMOKE_PROMPT = (
    "lo-fi hip hop instrumental, soft piano, dusty drums, warm vinyl crackle, chill study beat"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="016 MusicGen on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs Volume")

    download = sub.add_parser("download", help="下载指定 MusicGen 模型")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--model", default=DEFAULT_MODEL)

    smoke = sub.add_parser("smoke", help="固定 lo-fi benchmark")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--model", default=DEFAULT_MODEL)
    smoke.add_argument("--duration", type=float, default=15.0)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--run-name", default="smoke_lofi")
    smoke.add_argument("--guidance-scale", type=float, default=3.0)
    smoke.add_argument("--temperature", type=float, default=1.0)

    t2a = sub.add_parser("t2a", help="Text-to-Audio")
    t2a.add_argument("--dry-run", action="store_true")
    t2a.add_argument("--gpu", default=DEFAULT_GPU)
    t2a.add_argument("--model", default=DEFAULT_MODEL)
    t2a.add_argument("--prompt", required=True)
    t2a.add_argument("--duration", type=float, default=15.0)
    t2a.add_argument("--seed", type=int, default=42)
    t2a.add_argument("--run-name", default="")
    t2a.add_argument("--guidance-scale", type=float, default=3.0)
    t2a.add_argument("--temperature", type=float, default=1.0)
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "016-musicgen",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "models": HF_REPOS,
        "license": "CC-BY-NC 4.0",
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
    }


def generation_plan(args: argparse.Namespace) -> dict[str, Any]:
    model = _norm_model(args.model)
    prompt = SMOKE_PROMPT if args.command == "smoke" else args.prompt.strip()
    if not prompt:
        raise ValueError("prompt 不能为空")
    return {
        "action": args.command,
        "gpu": args.gpu,
        "model": model,
        "prompt": prompt,
        "duration": args.duration,
        "seed": args.seed,
        "run_name": args.run_name,
        "guidance_scale": args.guidance_scale,
        "temperature": args.temperature,
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

    try:
        if args.command == "download":
            model = _norm_model(args.model)
            plan = {"action": "download", "model": model, "force": args.force}
            if args.dry_run:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
                return
            print(json.dumps(download_weights.remote(force=args.force, model=model), ensure_ascii=False, indent=2))
            return
        plan = generation_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    download_weights.remote(force=False, model=plan["model"])
    out = generate_fn.with_options(gpu=args.gpu).remote(
        prompt=plan["prompt"],
        duration=plan["duration"],
        seed=plan["seed"],
        model=plan["model"],
        run_name=plan["run_name"],
        gpu_label=args.gpu,
        guidance_scale=plan["guidance_scale"],
        temperature=plan["temperature"],
    )
    label = "SMOKE_RESULT" if args.command == "smoke" else "T2A_RESULT"
    print(label, json.dumps(out, ensure_ascii=False), flush=True)
    if not out.get("success"):
        raise SystemExit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
