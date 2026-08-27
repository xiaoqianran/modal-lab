# -*- coding: utf-8 -*-
"""
025-kokoro — hexgrad Kokoro-82M TTS on Modal

真实用量榜 #1（HF ~11.5M dl）· Elo 开源前五 · Apache-2.0
默认：v1 + af_heart · GPU T4（最便宜）
可选：v1.1-zh（更好中文 100 speaker）

上游: https://github.com/hexgrad/kokoro
权重: https://huggingface.co/hexgrad/Kokoro-82M
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

APP_NAME = "modal-lab-kokoro"
HF_REPOS = {
    "v1": "hexgrad/Kokoro-82M",
    "v1.1-zh": "hexgrad/Kokoro-82M-v1.1-zh",
}
DEFAULT_MODEL = "v1"
DEFAULT_GPU = "T4"
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
VOLUME_WEIGHTS = "modal-lab-kokoro-weights"
VOLUME_OUTPUTS = "modal-lab-kokoro-outputs"

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

# voice prefix → KPipeline lang_code
VOICE_LANG = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Brazilian Portuguese
    "z": "z",  # Mandarin Chinese
}

app = modal.App(APP_NAME)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

_HF_ENV = {
    "HF_HOME": str(HF_HOME),
    "HF_HUB_CACHE": str(HF_HOME / "hub"),
    "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
    "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
    "HF_XET_HIGH_PERFORMANCE": "1",
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
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "espeak-ng",
        "git",
        "ca-certificates",
    )
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "kokoro>=0.9.4",
        "misaki[en,zh]>=0.9.0",
        "soundfile",
        "numpy",
        "scipy",
        "huggingface_hub[hf_transfer]>=0.26.0",
        "tqdm",
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
    aliases = {
        "v1": "v1",
        "v1.0": "v1",
        "1.0": "v1",
        "default": "v1",
        "kokoro": "v1",
        "kokoro-82m": "v1",
        "v1.1-zh": "v1.1-zh",
        "v1.1": "v1.1-zh",
        "zh": "v1.1-zh",
        "chinese": "v1.1-zh",
    }
    if n in aliases:
        return aliases[n]
    if n in HF_REPOS:
        return n
    for k, repo in HF_REPOS.items():
        if n == repo.lower() or n.endswith(k):
            return k
    raise ValueError(f"unknown model {name!r}; use {list(HF_REPOS)}")


def _lang_from_voice(voice: str, lang: str | None = None) -> str:
    if lang:
        return lang
    v = (voice or DEFAULT_VOICE).strip().lower()
    if len(v) >= 1 and v[0] in VOICE_LANG:
        return VOICE_LANG[v[0]]
    return "a"


def _weights_ready(model_key: str) -> bool:
    repo = HF_REPOS[model_key]
    local = Path(WEIGHTS_MOUNT) / "models" / model_key
    markers = list(local.glob("kokoro*.pth")) + list(local.glob("config.json"))
    if markers:
        return True
    hub = HF_HOME / "hub"
    if not hub.is_dir():
        return False
    slug = "models--" + repo.replace("/", "--")
    return any(hub.glob(f"{slug}/**/kokoro*.pth")) or any(
        hub.glob(f"{slug}/**/config.json")
    )


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
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
    kwargs: dict[str, Any] = {
        "repo_id": repo,
        "token": token,
        "ignore_patterns": ["samples/*", "eval/*", "*.jpeg", "*.jpg", "*.png"],
    }
    snapshot_download(cache_dir=str(HF_HOME / "hub"), **kwargs)
    snapshot_download(local_dir=str(local), **kwargs)
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
        "slot": "025-kokoro",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "default_voice": DEFAULT_VOICE,
        "license": "Apache-2.0",
        "sample_rate": SAMPLE_RATE,
        "models": {
            k: {
                "repo": r,
                "ready": _weights_ready(k),
                **_dir_info(Path(WEIGHTS_MOUNT) / "models" / k),
            }
            for k, r in HF_REPOS.items()
        },
        "hf_home": _dir_info(HF_HOME),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "gpu_note": "T4 cheapest; model is 82M — L4/H100 waste money",
        "ranking_note": "HF TTS downloads #1 (~11.5M) · AA open-weights Elo ~1056",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _list_voices(model_key: str) -> list[str]:
    local = Path(WEIGHTS_MOUNT) / "models" / model_key / "voices"
    if not local.is_dir():
        return []
    return sorted(p.stem for p in local.glob("*.pt"))


def _run_tts(
    *,
    text: str,
    voice: str,
    model: str,
    lang: str | None,
    speed: float,
    run_name: str,
    gpu_label: str,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline

    # Force hub cache onto Volume (KModel only accepts HF repo ids, not local paths)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    HF_HOME.mkdir(parents=True, exist_ok=True)

    key = _norm_model(model)
    repo = HF_REPOS[key]
    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}
    if not _weights_ready(key):
        return {"success": False, "error": f"weights missing for {key} — run download"}

    voice = (voice or DEFAULT_VOICE).strip()
    lang_code = _lang_from_voice(voice, lang)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prefer loading voice tensor from local Volume snapshot (offline-friendly)
    voice_path = Path(WEIGHTS_MOUNT) / "models" / key / "voices" / f"{voice}.pt"
    voice_arg: Any = voice
    if voice_path.is_file():
        voice_arg = torch.load(str(voice_path), weights_only=True, map_location="cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    t_load = time.time()
    # repo_id must be namespace/name — files come from HF_HOME cache on Volume
    kmodel = KModel(repo_id=repo).to(device).eval()
    pipeline = KPipeline(lang_code=lang_code, repo_id=repo, model=kmodel)
    load_s = time.time() - t_load

    t_gen = time.time()
    chunks: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    generator = pipeline(
        text,
        voice=voice_arg,
        speed=float(speed),
        split_pattern=r"\n+",
    )
    for i, item in enumerate(generator):
        if hasattr(item, "audio"):
            audio = item.audio
            gs = getattr(item, "graphemes", None) or getattr(item, "text", "")
            ps = getattr(item, "phonemes", "")
        else:
            gs, ps, audio = item
        if audio is None:
            continue
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        chunks.append(arr)
        segments.append(
            {
                "i": i,
                "graphemes": str(gs)[:200] if gs is not None else "",
                "phonemes": str(ps)[:200] if ps is not None else "",
                "samples": int(arr.shape[0]),
            }
        )
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    if not chunks:
        return {
            "success": False,
            "error": "no audio chunks generated",
            "voice": voice,
            "lang": lang_code,
            "model": key,
        }

    wav = np.concatenate(chunks)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1e-6 and peak > 0.99:
        wav = wav * (0.99 / peak)

    out_path = save_dir / "audio.wav"
    sf.write(str(out_path), wav, SAMPLE_RATE)

    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    audio_info = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "samples": int(wav.shape[0]),
        "duration_s": round(float(wav.shape[0]) / SAMPLE_RATE, 3),
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
        "model": key,
        "hf_repo": repo,
        "voice": voice,
        "voice_from_volume": voice_path.is_file(),
        "lang": lang_code,
        "speed": float(speed),
        "vram_peak_gb": vram_gb,
        "n_segments": len(segments),
        "segments": segments[:20],
        "audio": audio_info,
        "text": text[:500],
        "text_len": len(text),
    }
    meta = {
        "experiment": "025-kokoro",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "voice": voice,
            "lang": lang_code,
            "speed": float(speed),
            "model": key,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Apache-2.0 — hexgrad/Kokoro-82M",
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
    memory=8192,
    scaledown_window=30,
)
def generate_fn(
    text: str,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    lang: str = "",
    speed: float = 1.0,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        voice=voice,
        model=model,
        lang=lang or None,
        speed=speed,
        run_name=run_name,
        gpu_label=gpu_label,
    )


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=5 * 60,
    cpu=1,
    memory=2048,
)
def list_voices_fn(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    key = _norm_model(model)
    voices = _list_voices(key)
    out = {"model": key, "repo": HF_REPOS[key], "n": len(voices), "voices": voices}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


SMOKE_EN = (
    "Kokoro is an open-weight text-to-speech model with eighty-two million parameters. "
    "Despite its lightweight architecture, it delivers strong quality while staying fast and cheap."
)
SMOKE_ZH = (
    "你好，这是 modal-lab 第零二五号实验。Kokoro 是一款参数量仅八千两百万的开源语音合成模型，"
    "体积小、速度快，适合作为成本基线。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="025 Kokoro-82M on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs Volume")

    download = sub.add_parser("download", help="下载指定 Kokoro 模型")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--model", default=DEFAULT_MODEL)

    voices = sub.add_parser("voices", help="列出指定模型 voice")
    voices.add_argument("--model", default=DEFAULT_MODEL)
    voices.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="固定 EN / ZH smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--model", default=DEFAULT_MODEL)
    smoke.add_argument("--voice", default=DEFAULT_VOICE)
    smoke.add_argument("--lang", default="en", help="en | zh")
    smoke.add_argument("--speed", type=float, default=1.0)
    smoke.add_argument("--run-name", default="")

    t2s = sub.add_parser("t2s", help="Text-to-Speech")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--model", default=DEFAULT_MODEL)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--voice", default=DEFAULT_VOICE)
    t2s.add_argument("--lang", default="", help="override Kokoro lang_code")
    t2s.add_argument("--speed", type=float, default=1.0)
    t2s.add_argument("--run-name", default="")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "025-kokoro",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "default_voice": DEFAULT_VOICE,
        "models": HF_REPOS,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "sample_rate": 24000,
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    zh = args.lang.lower().startswith("zh") or args.lang.lower() == "z"
    if zh:
        model = "v1.1-zh"
        voice = "zf_001" if args.voice == DEFAULT_VOICE else args.voice
        text = SMOKE_ZH
        run_name = args.run_name or "smoke_zh"
        lang = "z"
    else:
        model = _norm_model(args.model)
        voice = args.voice or DEFAULT_VOICE
        text = SMOKE_EN
        run_name = args.run_name or "smoke_en_heart"
        lang = args.lang
    return {
        "action": "smoke",
        "gpu": args.gpu,
        "model": model,
        "text": text,
        "voice": voice,
        "lang": lang,
        "speed": args.speed,
        "run_name": run_name,
    }


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "t2s",
        "gpu": args.gpu,
        "model": _norm_model(args.model),
        "text": args.text.strip(),
        "voice": args.voice,
        "lang": args.lang,
        "speed": args.speed,
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

    try:
        if args.command == "download":
            model = _norm_model(args.model)
            plan = {"action": "download", "model": model, "force": args.force}
            if args.dry_run:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
                return
            print(json.dumps(download_weights.remote(force=args.force, model=model), ensure_ascii=False, indent=2))
            return
        if args.command == "voices":
            model = _norm_model(args.model)
            plan = {"action": "voices", "model": model}
            if args.dry_run:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
                return
            print(json.dumps(list_voices_fn.remote(model=model), ensure_ascii=False, indent=2))
            return
        plan = smoke_plan(args) if args.command == "smoke" else t2s_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    if not plan["text"]:
        raise SystemExit("t2s requires non-empty --text")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    download_weights.remote(force=False, model=plan["model"])
    out = generate_fn.with_options(gpu=args.gpu).remote(
        text=plan["text"],
        voice=plan["voice"],
        model=plan["model"],
        lang=plan["lang"],
        speed=plan["speed"],
        run_name=plan["run_name"],
        gpu_label=args.gpu,
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
