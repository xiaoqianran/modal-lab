# -*- coding: utf-8 -*-
"""
030-vibevoice — Microsoft VibeVoice-Realtime-0.5B on Modal

真实用量榜 Tier A2：GH VibeVoice ~52k stars #1 · Realtime HF ~594k
默认：microsoft/VibeVoice-Realtime-0.5B · GPU L4 · MIT
说明：官方已撤长文 TTS 推理码；本槽用仍开放的 Realtime 流式 TTS。

上游: https://github.com/microsoft/VibeVoice
权重: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-vibevoice"

HF_REPO = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_MODEL = "realtime_0.5b"
DEFAULT_GPU = "L4"
DEFAULT_SPEAKER = "Carter"
SAMPLE_RATE = 24000

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
LOCAL_MODEL = Path(WEIGHTS_MOUNT) / "models" / "VibeVoice-Realtime-0.5B"
VOICES_DIR = Path(PROMPTS_MOUNT) / "streaming_model"
VOLUME_WEIGHTS = "modal-lab-vibevoice-weights"
VOLUME_OUTPUTS = "modal-lab-vibevoice-outputs"
VOLUME_PROMPTS = "modal-lab-vibevoice-prompts"

VOICE_PRESETS = {
    "Carter": "en-Carter_man.pt",
    "Emma": "en-Emma_woman.pt",
    "Mike": "en-Mike_man.pt",
    "Grace": "en-Grace_woman.pt",
    "Davis": "en-Davis_man.pt",
    "Frank": "en-Frank_man.pt",
}

VOICE_RAW_BASE = (
    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/"
    "demo/voices/streaming_model"
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

# Clone VibeVoice (streamingtts) + torch 2.5 cu124 · SDPA (no flash-attn required)
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
        "transformers==4.51.3",
        "accelerate",
        "diffusers",
        "tqdm",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "ml-collections",
        "absl-py",
        "av",
        "pydub",
        "requests",
        "huggingface_hub[hf_transfer]>=0.26.0,<1.0",
        "llvmlite",
        "numba",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/microsoft/VibeVoice.git /opt/VibeVoice",
        "cd /opt/VibeVoice && pip install -e . --no-deps",
    )
    .env({**_HF_ENV, "PYTHONPATH": "/opt/VibeVoice"})
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
    if (LOCAL_MODEL / "model.safetensors").is_file():
        return True
    hub = HF_HOME / "hub"
    if hub.is_dir() and any(
        hub.glob("models--microsoft--VibeVoice-Realtime-0.5B/**/model.safetensors")
    ):
        return True
    return False


def _resolve_model_path() -> str:
    if (LOCAL_MODEL / "model.safetensors").is_file():
        return str(LOCAL_MODEL)
    return HF_REPO


def _ensure_voices() -> list[str]:
    import urllib.request

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    got = []
    for name, fname in VOICE_PRESETS.items():
        dest = VOICES_DIR / fname
        if not dest.is_file():
            url = f"{VOICE_RAW_BASE}/{fname}"
            try:
                print(f"Fetching voice {fname}…", flush=True)
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"voice fetch fail {fname}: {e}", flush=True)
                continue
        got.append(name)
    return got


def _resolve_voice_path(speaker: str) -> Path | None:
    s = (speaker or DEFAULT_SPEAKER).strip()
    # direct path
    p = Path(s)
    if p.is_file():
        return p
    # preset name
    fname = VOICE_PRESETS.get(s) or VOICE_PRESETS.get(s.title())
    if fname:
        cand = VOICES_DIR / fname
        if cand.is_file():
            return cand
    # stem match
    for f in VOICES_DIR.glob("*.pt"):
        if s.lower() in f.stem.lower():
            return f
    # default Carter
    d = VOICES_DIR / VOICE_PRESETS[DEFAULT_SPEAKER]
    return d if d.is_file() else None


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

    voices = _ensure_voices()
    results["voices"] = voices
    results["prompts"] = _dir_info(VOICES_DIR)
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
    voices = sorted(VOICES_DIR.glob("*.pt")) if VOICES_DIR.is_dir() else []
    out = {
        "app": APP_NAME,
        "slot": "030-vibevoice",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "license": "MIT",
        "hf_repo": HF_REPO,
        "weights_ready": _weights_ready(),
        "model_dir": _dir_info(LOCAL_MODEL),
        "voices": [p.stem for p in voices],
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "GH ~52k stars #1 · Realtime HF ~594k · long-form/streaming",
        "gpu_note": "L4 default; 0.5B ~2GB weights; T4 also viable",
        "modes": ["tts_en", "tts_long", "speaker_swap"],
        "note": "Official long-form multi-speaker TTS code removed; using Realtime-0.5B",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _run_tts(
    *,
    text: str,
    run_name: str,
    gpu_label: str,
    speaker: str = DEFAULT_SPEAKER,
    cfg_scale: float = 1.5,
    ddpm_steps: int = 5,
) -> dict[str, Any]:
    import copy
    import sys

    import numpy as np
    import soundfile as sf
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if "/opt/VibeVoice" not in sys.path:
        sys.path.insert(0, "/opt/VibeVoice")

    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}
    if not _weights_ready():
        return {"success": False, "error": "weights missing; run download first"}

    _ensure_voices()
    voice_path = _resolve_voice_path(speaker)
    if voice_path is None or not voice_path.is_file():
        return {"success": False, "error": f"voice not found: {speaker}"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    model_path = _resolve_model_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.time()
    t_load = time.time()
    print(f"Loading VibeVoice Realtime from {model_path} on {device}…", flush=True)
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map="cuda" if device == "cuda" else "cpu",
            attn_implementation="sdpa",
        )
    except Exception as e:
        print(f"primary load failed ({e}); retry sdpa float32…", flush=True)
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.float32 if device != "cuda" else load_dtype,
            device_map="cuda" if device == "cuda" else "cpu",
            attn_implementation="sdpa",
        )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=int(ddpm_steps))
    load_s = time.time() - t_load

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        all_prefilled_outputs = torch.load(
            str(voice_path), map_location=device, weights_only=True
        )

    full_script = (
        text.replace("’", "'").replace("“", '"').replace("”", '"')
    )
    inputs = processor.process_input_with_cached_prompt(
        text=full_script,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    t_gen = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=float(cfg_scale),
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
    )
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        return {"success": False, "error": "no speech_outputs", "wall_s": round(wall, 2)}

    speech = outputs.speech_outputs[0]
    if torch.is_tensor(speech):
        arr = speech.detach().float().cpu().numpy()
    else:
        arr = np.asarray(speech, dtype=np.float32)
    arr = arr.reshape(-1).astype(np.float32)
    # soft peak limit — realtime decoder occasionally exceeds ±1.0
    peak_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak_abs > 1.0:
        arr = arr * (0.99 / peak_abs)
    sr = SAMPLE_RATE

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
        "speaker": speaker,
        "voice_path": str(voice_path),
        "gen_kwargs": {
            "cfg_scale": cfg_scale,
            "ddpm_steps": ddpm_steps,
            "attn": "sdpa",
        },
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:800],
        "text_len": len(text),
        "license_note": "MIT — Microsoft VibeVoice-Realtime-0.5B (research use)",
    }
    meta = {
        "experiment": "030-vibevoice",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "model": DEFAULT_MODEL,
            "speaker": speaker,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "MIT — Microsoft VibeVoice-Realtime-0.5B",
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
    speaker: str = DEFAULT_SPEAKER,
    cfg_scale: float = 1.5,
    ddpm_steps: int = 5,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        run_name=run_name,
        gpu_label=gpu_label,
        speaker=speaker,
        cfg_scale=cfg_scale,
        ddpm_steps=ddpm_steps,
    )


SMOKE_EN = (
    "VibeVoice Realtime is running on Modal. "
    "This lightweight model supports streaming text input "
    "and robust long-form speech generation."
)
SMOKE_LONG = (
    "Welcome to modal-lab experiment zero three zero. "
    "VibeVoice was designed for long conversational audio and podcast narration. "
    "Even this realtime half billion parameter variant can keep going for minutes "
    "while remaining lightweight enough for a single L4 GPU. "
    "Today we measure cold-start cost, VRAM, and natural prosody on English text."
)
SMOKE_EMMA = (
    "Hello, this is Emma speaking through VibeVoice Realtime on Modal. "
    "Switching speakers only requires a different voice preset cache."
)


@app.local_entrypoint()
def main(
    action: str = "status",
    gpu: str = DEFAULT_GPU,
    text: str = "",
    run_name: str = "",
    force_download: bool = False,
    smoke_kind: str = "en",
    speaker: str = DEFAULT_SPEAKER,
    cfg_scale: float = 1.5,
    ddpm_steps: int = 5,
):
    if action == "status":
        status_fn.remote()
        return
    if action == "download":
        download_weights.remote(force=force_download)
        return
    if action == "smoke":
        kind = smoke_kind.lower().strip()
        spk = speaker
        if kind in ("long", "en_long"):
            text_use, run = SMOKE_LONG, run_name or "smoke_long"
            if not spk or spk == DEFAULT_SPEAKER:
                spk = "Emma"
        elif kind in ("emma", "woman"):
            text_use, run = SMOKE_EMMA, run_name or "smoke_emma"
            spk = "Emma"
        else:
            text_use, run = SMOKE_EN, run_name or "smoke_en"
            if not spk:
                spk = "Carter"

        download_weights.remote(force=False)
        out = generate_fn.with_options(gpu=gpu).remote(
            text=text_use,
            run_name=run,
            gpu_label=gpu,
            speaker=spk,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
        )
        print("SMOKE_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        if (out.get("audio") or {}).get("duration_s", 0) < 0.5:
            raise SystemExit("smoke audio too short")
        if (out.get("audio") or {}).get("rms", 0) < 1e-4:
            raise SystemExit("smoke audio near silent")
        return
    if action == "t2s":
        if not text.strip():
            raise SystemExit("t2s requires --text")
        download_weights.remote(force=False)
        out = generate_fn.with_options(gpu=gpu).remote(
            text=text,
            run_name=run_name,
            gpu_label=gpu,
            speaker=speaker,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
        )
        print("T2S_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        return
    raise SystemExit(f"unknown action {action!r}")
