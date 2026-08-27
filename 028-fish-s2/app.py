# -*- coding: utf-8 -*-
"""
028-fish-s2 — Fish Audio S2 Pro TTS on Modal

真实用量榜 Tier S4：AA Elo 开源 #1 (1121) · GH ~32k · HF s2-pro ~428k
默认：S2-Pro 4B · GPU L40S（官方建议 ≥24GB VRAM）
许可：Fish Audio Research License（研究/非商用；商用需单独授权）

上游: https://github.com/fishaudio/fish-speech
权重: https://huggingface.co/fishaudio/s2-pro
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

APP_NAME = "modal-lab-fish-s2"

HF_REPO = "fishaudio/s2-pro"
DEFAULT_MODEL = "s2-pro"
DEFAULT_GPU = "L40S"
FISH_REPO = "https://github.com/fishaudio/fish-speech.git"
# pin for reproducible image builds (main @ 2026-06 AMD ROCm)
FISH_COMMIT = "e5e292632cb1"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
CKPT_ROOT = Path(WEIGHTS_MOUNT) / "checkpoints" / "s2-pro"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
VOLUME_WEIGHTS = "modal-lab-fish-s2-weights"
VOLUME_OUTPUTS = "modal-lab-fish-s2-outputs"
VOLUME_PROMPTS = "modal-lab-fish-s2-prompts"
FISH_SRC = Path("/opt/fish-speech")

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

# Official-style clone demo audio (same as Qwen3-TTS docs — public)
DEFAULT_CLONE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
)
DEFAULT_CLONE_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)

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
    "EINX_FILTER_TRACEBACK": "false",
}

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0,<1.0")
    .env(_HF_ENV)
)

# torch 2.8 + cu126 matches fish-speech pyproject; install package from git pin
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "libsox-dev",
        "sox",
        "git",
        "portaudio19-dev",
        "ca-certificates",
        "build-essential",
        "curl",
    )
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu126",
    )
    .run_commands(
        # shallow clone main; FISH_COMMIT documented for reference only
        f"git clone --depth 1 {FISH_REPO} {FISH_SRC}"
    )
    .pip_install(
        # minimal inference runtime (skip train/web: wandb, tensorboard, gradio, lightning, datasets)
        "numpy",
        "transformers==4.57.3",
        "accelerate>=0.33.0",
        "einops>=0.7.0",
        "einx[torch]==0.2.2",
        "librosa>=0.10.1",
        "soundfile",
        "scipy",
        "resampy>=0.4.3",
        "pydub",
        "loguru>=0.6.0",
        "rich>=13.5.3",
        "pydantic==2.9.2",
        "ormsgpack",
        "tiktoken>=0.8.0",
        "safetensors",
        "hydra-core>=1.3.2",
        "omegaconf",
        "pyrootutils>=1.0.4",
        "natsort>=8.4.0",
        "loralib>=0.1.2",
        "zstandard>=0.22.0",
        "cachetools",
        "opencc-python-reimplemented==0.1.7",
        "silero-vad",
        "huggingface_hub[hf_transfer]>=0.26.0,<1.0",
        "setuptools",
        "wheel",
        "tensorboard==2.18.0",
        "lightning>=2.1.0",
        "protobuf>=3.20.0,<6.0.0",
        "click",
        "ffmpy",
        "flatten-dict",
        "importlib-resources",
        "julius",
        "markdown2",
        "matplotlib",
        "pyloudnorm",
        "pystoi",
        "randomname",
        "argbind",
    )
    # descript-audio-codec pulls protobuf<3.20; fish-speech needs >=3.20 → install no-deps then force protobuf
    .run_commands(
        "pip install --no-deps descript-audio-codec descript-audiotools",
        "pip install 'protobuf>=3.20.0,<6.0.0'",
        f"cd {FISH_SRC} && pip install -e . --no-deps",
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
    codec = CKPT_ROOT / "codec.pth"
    # either sharded safetensors or single
    shards = list(CKPT_ROOT.glob("model*.safetensors")) + list(
        CKPT_ROOT.glob("*.safetensors")
    )
    cfg = CKPT_ROOT / "config.json"
    return codec.is_file() and cfg.is_file() and len(shards) >= 1


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, PROMPTS_MOUNT: prompts_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"model": model or DEFAULT_MODEL, "repos": []}
    t0 = time.time()
    if _weights_ready() and not force:
        results["repos"].append(
            {
                "repo": HF_REPO,
                "skipped": True,
                **_dir_info(CKPT_ROOT),
            }
        )
    else:
        print(f"Downloading {HF_REPO} → {CKPT_ROOT}…", flush=True)
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(CKPT_ROOT),
            token=token,
        )
        results["repos"].append(
            {
                "repo": HF_REPO,
                "skipped": False,
                "elapsed_s": round(time.time() - t0, 1),
                **_dir_info(CKPT_ROOT),
            }
        )

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
        "slot": "028-fish-s2",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "license": "Fish Audio Research License (non-commercial / research)",
        "hf_repo": HF_REPO,
        "elo": 1121,
        "weights_ready": _weights_ready(),
        "checkpoint": _dir_info(CKPT_ROOT),
        "prompts": {"n": len(prompts), "names": [p.stem for p in prompts[:30]]},
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "AA Elo open-weights #1 · GH fish-speech ~32k · HF s2-pro ~428k",
        "gpu_note": "L40S default (≥24GB VRAM recommended); L4 may OOM on long text",
        "tags_note": "Inline [excited] [whisper] [laughing] free-form tags supported",
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


def _ensure_fish_path() -> None:
    if str(FISH_SRC) not in sys.path:
        sys.path.insert(0, str(FISH_SRC))
    # pyrootutils / relative imports
    os.chdir(str(FISH_SRC))


def _run_tts(
    *,
    text: str,
    run_name: str,
    gpu_label: str,
    ref_audio_path: str = "",
    ref_text: str = "",
    voice: str = "",
    temperature: float = 0.8,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 1024,
    chunk_length: int = 200,
    seed: int | None = 42,
    compile_model: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    _ensure_fish_path()

    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}

    if not _weights_ready():
        return {
            "success": False,
            "error": f"weights missing under {CKPT_ROOT}; run download first",
        }

    # resolve reference
    ref_path: Path | None = None
    ref_bytes: bytes | None = None
    ref_used = ""
    if ref_audio_path.strip():
        p = Path(ref_audio_path.strip())
        if p.is_file():
            ref_path = p
        elif ref_audio_path.startswith("http"):
            import urllib.request

            tmp = Path("/tmp/ref_prompt.wav")
            urllib.request.urlretrieve(ref_audio_path, tmp)
            ref_path = tmp
            ref_used = ref_audio_path
        else:
            found = _resolve_prompt_path(ref_audio_path)
            if found:
                ref_path = found
    if ref_path is None and voice:
        found = _resolve_prompt_path(voice)
        if found:
            ref_path = found
    if ref_path is not None and ref_path.is_file():
        ref_bytes = ref_path.read_bytes()
        if not ref_used:
            ref_used = str(ref_path)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.time()
    t_load = time.time()
    print(f"Loading S2-Pro from {CKPT_ROOT} on {device}…", flush=True)
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(CKPT_ROOT),
        device=device,
        precision=precision,
        compile=bool(compile_model),
    )
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(CKPT_ROOT / "codec.pth"),
        device=device,
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=bool(compile_model),
    )
    load_s = time.time() - t_load

    references = []
    if ref_bytes is not None:
        if not (ref_text or "").strip():
            return {
                "success": False,
                "error": "clone requires --ref-text matching the reference audio",
            }
        references = [
            ServeReferenceAudio(audio=ref_bytes, text=ref_text.strip())
        ]

    req = ServeTTSRequest(
        text=text,
        references=references,
        reference_id=None,
        max_new_tokens=int(max_new_tokens),
        chunk_length=int(chunk_length),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        temperature=float(temperature),
        seed=seed,
        format="wav",
        normalize=True,
        use_memory_cache="off",
    )

    t_gen = time.time()
    audio_tuple = None
    err = None
    for result in engine.inference(req):
        if result.code == "final":
            audio_tuple = result.audio
            break
        if result.code == "error":
            err = result.error
            break
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    if err is not None:
        return {
            "success": False,
            "error": str(err),
            "wall_s": round(wall, 2),
            "load_s": round(load_s, 2),
            "generate_s": round(gen_s, 2),
        }
    if audio_tuple is None:
        return {
            "success": False,
            "error": "no audio generated",
            "wall_s": round(wall, 2),
            "load_s": round(load_s, 2),
            "generate_s": round(gen_s, 2),
        }

    if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
        sr, arr = audio_tuple
        arr = np.asarray(arr, dtype=np.float32)
    else:
        arr = np.asarray(audio_tuple, dtype=np.float32)
        sr = 44100
        if hasattr(decoder_model, "sample_rate"):
            sr = int(decoder_model.sample_rate)
        elif hasattr(decoder_model, "spec_transform"):
            sr = int(decoder_model.spec_transform.sample_rate)

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    out_path = save_dir / "audio.wav"
    sf.write(str(out_path), arr, int(sr))

    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    audio_info = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sample_rate": int(sr),
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
        "ref_audio": ref_used or None,
        "ref_text": (ref_text or None) if ref_bytes else None,
        "gen_kwargs": {
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "seed": seed,
            "compile": compile_model,
            "n_references": len(references),
        },
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:800],
        "text_len": len(text),
        "license_note": "Fish Audio Research License — research/non-commercial",
    }
    meta = {
        "experiment": "028-fish-s2",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "model": DEFAULT_MODEL,
            "ref_audio": ref_used or None,
            "ref_text": (ref_text or None) if ref_bytes else None,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Fish Audio Research License — research/non-commercial",
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
    memory=32768,
    scaledown_window=60,
)
def generate_fn(
    text: str,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    ref_audio_path: str = "",
    ref_text: str = "",
    voice: str = "",
    temperature: float = 0.8,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 1024,
    chunk_length: int = 200,
    seed: int = 42,
    compile_model: bool = False,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        run_name=run_name,
        gpu_label=gpu_label,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        seed=seed if seed >= 0 else None,
        compile_model=compile_model,
    )


SMOKE_EN = (
    "Fish Audio S2 Pro is running on Modal. "
    "This is the open-weights quality leader — natural multilingual speech."
)
SMOKE_ZH = (
    "你好，这是 modal-lab 第零二八号 Fish Audio S2 Pro 实验。"
    "今天天气真不错，希望你有一个愉快的周末。"
)
SMOKE_TAGS = (
    "Hi there [excited], welcome to the demo! "
    "[chuckle] Can you believe this voice has over fifteen thousand control tags? "
    "[whisper] Soft parts work too."
)
SMOKE_CLONE = (
    "I am solving the equation on the board. "
    "Nobody said it would be this hard, but we are almost there!"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="028 Fish Audio S2 Pro on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / prompts / outputs Volume")

    download = sub.add_parser("download", help="下载 S2-Pro 权重")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="固定 EN / ZH / tags / clone smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--kind", default="en", choices=["en", "zh", "tags", "clone"])
    smoke.add_argument("--run-name", default="")
    smoke.add_argument("--ref-audio", default="")
    smoke.add_argument("--ref-text", default="")
    smoke.add_argument("--voice", default="")
    smoke.add_argument("--temperature", type=float, default=0.8)
    smoke.add_argument("--top-p", type=float, default=0.8)
    smoke.add_argument("--repetition-penalty", type=float, default=1.1)
    smoke.add_argument("--max-new-tokens", type=int, default=1024)
    smoke.add_argument("--chunk-length", type=int, default=200)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--compile", action="store_true")

    t2s = sub.add_parser("t2s", help="通用 Fish S2 TTS")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--ref-audio", default="")
    t2s.add_argument("--ref-text", default="")
    t2s.add_argument("--voice", default="")
    t2s.add_argument("--temperature", type=float, default=0.8)
    t2s.add_argument("--top-p", type=float, default=0.8)
    t2s.add_argument("--repetition-penalty", type=float, default=1.1)
    t2s.add_argument("--max-new-tokens", type=int, default=1024)
    t2s.add_argument("--chunk-length", type=int, default=200)
    t2s.add_argument("--seed", type=int, default=42)
    t2s.add_argument("--run-name", default="")
    t2s.add_argument("--compile", action="store_true")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "028-fish-s2",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "hf_repo": HF_REPO,
        "upstream": FISH_REPO,
        "documented_upstream_commit": FISH_COMMIT,
        "license": "Fish Audio Research License",
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "prompts_volume": VOLUME_PROMPTS,
    }


def _generation_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "gpu": args.gpu,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "voice": args.voice,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "seed": args.seed,
        "compile": args.compile,
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    fields = _generation_fields(args)
    if args.kind == "zh":
        text, run_name = SMOKE_ZH, args.run_name or "smoke_zh"
    elif args.kind == "tags":
        text, run_name = SMOKE_TAGS, args.run_name or "smoke_tags"
    elif args.kind == "clone":
        text, run_name = SMOKE_CLONE, args.run_name or "smoke_clone_en"
        if not fields["ref_audio"]:
            fields["ref_audio"] = DEFAULT_CLONE_URL
        if not fields["ref_text"]:
            fields["ref_text"] = DEFAULT_CLONE_TEXT
    else:
        text, run_name = SMOKE_EN, args.run_name or "smoke_en"
    return {"action": "smoke", "kind": args.kind, "text": text, "run_name": run_name, **fields}


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "t2s",
        "text": args.text.strip(),
        "run_name": args.run_name,
        **_generation_fields(args),
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
        ref_audio_path=plan["ref_audio"],
        ref_text=plan["ref_text"],
        voice=plan["voice"],
        temperature=plan["temperature"],
        top_p=plan["top_p"],
        repetition_penalty=plan["repetition_penalty"],
        max_new_tokens=plan["max_new_tokens"],
        chunk_length=plan["chunk_length"],
        seed=plan["seed"],
        compile_model=plan["compile"],
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
