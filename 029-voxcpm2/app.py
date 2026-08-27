# -*- coding: utf-8 -*-
"""
029-voxcpm2 — OpenBMB VoxCPM2 TTS on Modal

真实用量榜 Tier A1：HF ~643k · GH VoxCPM ~35k · Apache-2.0 · 速度/延迟向
默认：openbmb/VoxCPM2 · GPU L4
能力：多语 TTS · Voice Design · Controllable/Ultimate Cloning · 48kHz

上游: https://github.com/OpenBMB/VoxCPM
权重: https://huggingface.co/openbmb/VoxCPM2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-voxcpm2"

HF_REPO = "openbmb/VoxCPM2"
DEFAULT_MODEL = "voxcpm2"
DEFAULT_GPU = "L4"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
LOCAL_MODEL = Path(WEIGHTS_MOUNT) / "models" / "VoxCPM2"
VOLUME_WEIGHTS = "modal-lab-voxcpm2-weights"
VOLUME_OUTPUTS = "modal-lab-voxcpm2-outputs"
VOLUME_PROMPTS = "modal-lab-voxcpm2-prompts"

# Official example reference (in-repo)
REF_URL = (
    "https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/examples/reference_speaker.wav"
)

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

app = modal.App(APP_NAME)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
prompts_vol = modal.Volume.from_name(VOLUME_PROMPTS, create_if_missing=True)

_HF_ENV = {
    "HF_HOME": str(HF_HOME),
    "HF_HUB_CACHE": str(HF_HOME / "hub"),
    "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONUNBUFFERED": "1",
}

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0,<1.0")
    .env(_HF_ENV)
)

# torch 2.5+cu124 · voxcpm 2.0.x · skip denoiser (funasr) at runtime
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "git",
        "ca-certificates",
        "curl",
        "build-essential",
    )
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "voxcpm==2.0.3",
        "soundfile",
        "numpy",
        "librosa",
        "scipy",
        "einops",
        "safetensors",
        "transformers>=4.36.2",
        "huggingface_hub[hf_transfer]>=0.26.0,<1.0",
        "pydantic",
        "tqdm",
        "inflect",
        "addict",
        "wetext",
        "simplejson",
        "sortedcontainers",
        "argbind",
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


def _weights_ready() -> bool:
    if (LOCAL_MODEL / "model.safetensors").is_file() and (
        LOCAL_MODEL / "config.json"
    ).is_file():
        return True
    hub = HF_HOME / "hub"
    if hub.is_dir() and any(hub.glob("models--openbmb--VoxCPM2/**/model.safetensors")):
        return True
    return False


def _resolve_model_path() -> str:
    if (LOCAL_MODEL / "model.safetensors").is_file():
        return str(LOCAL_MODEL)
    return HF_REPO


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, PROMPTS_MOUNT: prompts_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    LOCAL_MODEL.mkdir(parents=True, exist_ok=True)
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"model": DEFAULT_MODEL, "repos": []}
    t0 = time.time()
    if _weights_ready() and not force:
        results["repos"].append(
            {"repo": HF_REPO, "skipped": True, **_dir_info(LOCAL_MODEL)}
        )
    else:
        print(f"Downloading {HF_REPO} → {LOCAL_MODEL}…", flush=True)
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(LOCAL_MODEL),
            token=token,
        )
        results["repos"].append(
            {
                "repo": HF_REPO,
                "skipped": False,
                "elapsed_s": round(time.time() - t0, 1),
                **_dir_info(LOCAL_MODEL),
            }
        )

    # also stash official reference speaker for clone smoke
    ref_dest = Path(PROMPTS_MOUNT) / "reference_speaker.wav"
    if not ref_dest.is_file():
        try:
            print(f"Fetching reference → {ref_dest}", flush=True)
            urllib.request.urlretrieve(REF_URL, ref_dest)
        except Exception as e:
            results["ref_error"] = repr(e)

    n_wav = len(list(Path(PROMPTS_MOUNT).glob("**/*.wav")))
    results["prompts"] = {"n_wav": n_wav, "path": PROMPTS_MOUNT}
    results["hf_home"] = _dir_info(HF_HOME)
    results["ready"] = _weights_ready()
    weights_vol.commit()
    prompts_vol.commit()
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


@app.function(
    image=download_image,
    volumes={
        WEIGHTS_MOUNT: weights_vol,
        OUTPUTS_MOUNT: outputs_vol,
        PROMPTS_MOUNT: prompts_vol,
    },
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
    prompts = sorted(Path(PROMPTS_MOUNT).glob("**/*.wav"))
    out = {
        "app": APP_NAME,
        "slot": "029-voxcpm2",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "license": "Apache-2.0",
        "hf_repo": HF_REPO,
        "weights_ready": _weights_ready(),
        "model_dir": _dir_info(LOCAL_MODEL),
        "prompts": {"n": len(prompts), "names": [p.stem for p in prompts[:30]]},
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "HF ~643k · GH ~35k · Apache · Tier A1 speed/latency",
        "gpu_note": "L4 default; 2B bf16 ~5GB weights; RTF ~0.3 on 4090",
        "modes": ["tts", "design", "clone", "ultimate_clone"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _resolve_prompt_path(voice: str) -> Path | None:
    if not voice:
        return None
    v = voice.strip()
    root = Path(PROMPTS_MOUNT)
    if not root.is_dir():
        return None
    for p in root.rglob("*.wav"):
        if p.stem.lower() == v.lower() or p.name.lower() == v.lower():
            return p
    c = root / f"{v}.wav"
    return c if c.is_file() else None


def _run_tts(
    *,
    text: str,
    run_name: str,
    gpu_label: str,
    reference_wav: str = "",
    prompt_wav: str = "",
    prompt_text: str = "",
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    seed: int = 42,
    optimize: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}
    if not _weights_ready():
        return {"success": False, "error": "weights missing; run download first"}

    # resolve refs
    ref_path = None
    if reference_wav.strip():
        p = Path(reference_wav.strip())
        if p.is_file():
            ref_path = str(p)
        elif reference_wav.startswith("http"):
            tmp = Path("/tmp/ref.wav")
            urllib.request.urlretrieve(reference_wav, tmp)
            ref_path = str(tmp)
        else:
            found = _resolve_prompt_path(reference_wav)
            if found:
                ref_path = str(found)
    prompt_path = None
    if prompt_wav.strip():
        p = Path(prompt_wav.strip())
        if p.is_file():
            prompt_path = str(p)
        else:
            found = _resolve_prompt_path(prompt_wav)
            if found:
                prompt_path = str(found)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from voxcpm import VoxCPM

    model_path = _resolve_model_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    t_load = time.time()
    print(f"Loading VoxCPM2 from {model_path} on {device} (optimize={optimize})…", flush=True)
    model = VoxCPM.from_pretrained(
        model_path,
        load_denoiser=False,
        optimize=bool(optimize),
        device=device,
        local_files_only=True if model_path == str(LOCAL_MODEL) else False,
    )
    load_s = time.time() - t_load

    gen_kwargs: dict[str, Any] = {
        "text": text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
    }
    if seed is not None and seed >= 0:
        gen_kwargs["seed"] = int(seed)
    if ref_path:
        gen_kwargs["reference_wav_path"] = ref_path
    if prompt_path and prompt_text.strip():
        gen_kwargs["prompt_wav_path"] = prompt_path
        gen_kwargs["prompt_text"] = prompt_text.strip()

    # PyPI voxcpm may lag GitHub API (e.g. no seed kwarg)
    import inspect
    try:
        sig = inspect.signature(model._generate)
        accepted = set(sig.parameters)
        filtered = {k: v for k, v in gen_kwargs.items() if k in accepted}
        dropped = sorted(set(gen_kwargs) - set(filtered))
        if dropped:
            print(f"dropped gen kwargs unsupported by package: {dropped}", flush=True)
        gen_kwargs = filtered
    except Exception:
        pass

    if seed is not None and seed >= 0 and "seed" not in gen_kwargs:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    t_gen = time.time()
    wav = model.generate(**gen_kwargs)
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", 48000) or 48000)

    out_path = save_dir / "audio.wav"
    sf.write(str(out_path), arr, sr)

    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    audio_info = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sample_rate": sr,
        "channels": 1,
        "samples": int(arr.shape[0]),
        "duration_s": round(float(arr.shape[0]) / float(sr), 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
    }
    result = {
        "success": True,
        "error": None,
        "wall_s": round(wall, 2),
        "load_s": round(load_s, 2),
        "generate_s": round(gen_s, 2),
        "est_gpu_usd": _estimate_cost(gpu_label, wall),
        "gpu": gpu_label,
        "device": device,
        "model": DEFAULT_MODEL,
        "repo_id": HF_REPO,
        "reference_wav": ref_path,
        "prompt_wav": prompt_path,
        "gen_kwargs": {
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "seed": seed,
            "optimize": optimize,
            "has_reference": bool(ref_path),
            "has_prompt": bool(prompt_path),
        },
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:800],
        "text_len": len(text),
        "license_note": "Apache-2.0 — OpenBMB VoxCPM2",
    }
    meta = {
        "experiment": "029-voxcpm2",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "model": DEFAULT_MODEL,
            "reference_wav": ref_path,
            "prompt_wav": prompt_path,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Apache-2.0 — OpenBMB VoxCPM2",
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
    volumes={
        WEIGHTS_MOUNT: weights_vol,
        OUTPUTS_MOUNT: outputs_vol,
        PROMPTS_MOUNT: prompts_vol,
    },
    memory=24576,
    scaledown_window=60,
)
def generate_fn(
    text: str,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    reference_wav: str = "",
    prompt_wav: str = "",
    prompt_text: str = "",
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    seed: int = 42,
    optimize: bool = False,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        run_name=run_name,
        gpu_label=gpu_label,
        reference_wav=reference_wav,
        prompt_wav=prompt_wav,
        prompt_text=prompt_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        seed=seed,
        optimize=optimize,
    )


SMOKE_EN = (
    "VoxCPM2 is running on Modal. This is a tokenizer-free multilingual "
    "text to speech model optimized for speed and natural prosody."
)
SMOKE_ZH = (
    "你好，这是 modal-lab 第零二九号 VoxCPM2 实验。"
    "今天天气真不错，希望你有一个愉快的周末。"
)
SMOKE_DESIGN = (
    "(A young woman, gentle and sweet voice, warm midrange)"
    "Hello, welcome to VoxCPM2 voice design on Modal lab."
)
SMOKE_CLONE = (
    "This is a controllable voice clone generated by VoxCPM2 on Modal."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="029 VoxCPM2 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查模型 / prompts / outputs Volume")

    download = sub.add_parser("download", help="下载模型与官方 clone reference")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="固定 EN / ZH / design / clone smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--kind", default="en", choices=["en", "zh", "design", "clone"])
    smoke.add_argument("--run-name", default="")
    smoke.add_argument("--reference-wav", default="")
    smoke.add_argument("--prompt-wav", default="")
    smoke.add_argument("--prompt-text", default="")
    smoke.add_argument("--cfg-value", type=float, default=2.0)
    smoke.add_argument("--timesteps", type=int, default=10)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--optimize", action="store_true")

    t2s = sub.add_parser("t2s", help="通用 VoxCPM2 TTS")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--reference-wav", default="")
    t2s.add_argument("--prompt-wav", default="")
    t2s.add_argument("--prompt-text", default="")
    t2s.add_argument("--cfg-value", type=float, default=2.0)
    t2s.add_argument("--timesteps", type=int, default=10)
    t2s.add_argument("--seed", type=int, default=42)
    t2s.add_argument("--run-name", default="")
    t2s.add_argument("--optimize", action="store_true")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "029-voxcpm2",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "hf_repo": HF_REPO,
        "reference_url": REF_URL,
        "default_clone_reference": "reference_speaker",
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "prompts_volume": VOLUME_PROMPTS,
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    ref = args.reference_wav
    if args.kind == "zh":
        text, run_name = SMOKE_ZH, args.run_name or "smoke_zh"
    elif args.kind == "design":
        text, run_name = SMOKE_DESIGN, args.run_name or "smoke_design"
    elif args.kind == "clone":
        text, run_name = SMOKE_CLONE, args.run_name or "smoke_clone"
        if not ref:
            ref = "reference_speaker"
    else:
        text, run_name = SMOKE_EN, args.run_name or "smoke_en"
    return {
        "action": "smoke",
        "gpu": args.gpu,
        "kind": args.kind,
        "text": text,
        "run_name": run_name,
        "reference_wav": ref,
        "prompt_wav": args.prompt_wav,
        "prompt_text": args.prompt_text,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.timesteps,
        "seed": args.seed,
        "optimize": args.optimize,
    }


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "t2s",
        "gpu": args.gpu,
        "text": args.text.strip(),
        "run_name": args.run_name,
        "reference_wav": args.reference_wav,
        "prompt_wav": args.prompt_wav,
        "prompt_text": args.prompt_text,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.timesteps,
        "seed": args.seed,
        "optimize": args.optimize,
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
    if args.command == "download":
        plan = {"action": "download", "force": args.force}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(download_weights.remote(force=args.force), ensure_ascii=False, indent=2))
        return

    plan = smoke_plan(args) if args.command == "smoke" else t2s_plan(args)
    if not plan["text"]:
        raise SystemExit("t2s requires non-empty --text")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    download_weights.remote(force=False)
    out = generate_fn.with_options(gpu=args.gpu).remote(
        text=plan["text"],
        run_name=plan["run_name"],
        gpu_label=args.gpu,
        reference_wav=plan["reference_wav"],
        prompt_wav=plan["prompt_wav"],
        prompt_text=plan["prompt_text"],
        cfg_value=plan["cfg_value"],
        inference_timesteps=plan["inference_timesteps"],
        seed=plan["seed"],
        optimize=plan["optimize"],
    )
    label = "SMOKE_RESULT" if args.command == "smoke" else "T2S_RESULT"
    print(label, json.dumps(out, ensure_ascii=False), flush=True)
    if not out.get("success"):
        raise SystemExit(2)
    if args.command == "smoke":
        if (out.get("audio") or {}).get("duration_s", 0) < 0.5:
            raise SystemExit("smoke audio too short")
        if (out.get("audio") or {}).get("rms", 0) < 1e-4:
            raise SystemExit("smoke audio near silent")


if __name__ == "__main__":
    main(*sys.argv[1:])
