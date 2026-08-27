# -*- coding: utf-8 -*-
"""
033-f5tts — F5-TTS zero-shot clone on Modal

真实用量榜 Tier A5：HF ~740k · GH 15k · 零样本扩散克隆
默认：F5TTS_v1_Base · GPU L4 · Code MIT / Model CC-BY-NC

上游: https://github.com/SWivid/F5-TTS
权重: hf://SWivid/F5-TTS (auto via f5-tts package)
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

APP_NAME = "modal-lab-f5tts"

DEFAULT_MODEL = "F5TTS_v1_Base"
DEFAULT_GPU = "L4"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
VOLUME_WEIGHTS = "modal-lab-f5tts-weights"
VOLUME_OUTPUTS = "modal-lab-f5tts-outputs"
VOLUME_PROMPTS = "modal-lab-f5tts-prompts"

REF_EN = "basic_ref_en.wav"
REF_ZH = "basic_ref_zh.wav"
REF_EN_URL = (
    "https://raw.githubusercontent.com/SWivid/F5-TTS/main/"
    "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
)
REF_ZH_URL = (
    "https://raw.githubusercontent.com/SWivid/F5-TTS/main/"
    "src/f5_tts/infer/examples/basic/basic_ref_zh.wav"
)
REF_EN_TEXT = "Some call me nature, others call me mother nature."
REF_ZH_TEXT = "对，这就是我，万人敬仰的太乙真人。"

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
    .uv_pip_install("huggingface_hub[hf_transfer]>=0.26.0,<1.0")
    .env(_HF_ENV)
)

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
    .uv_pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .uv_pip_install(
        "f5-tts",
        "soundfile",
        "numpy",
        "librosa",
        "scipy",
        "huggingface_hub[hf_transfer]>=0.26.0,<1.0",
        "transformers",
        "accelerate",
        "vocos",
        "cached_path",
        "tomli",
        "tqdm",
        "einops",
        "ema_pytorch",
        "x_transformers",
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


def _ensure_refs() -> dict[str, str]:
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in ((REF_EN, REF_EN_URL), (REF_ZH, REF_ZH_URL)):
        dest = Path(PROMPTS_MOUNT) / name
        if not dest.is_file() or dest.stat().st_size < 1000:
            print(f"Fetching {name}…", flush=True)
            urllib.request.urlretrieve(url, dest)
        out[name] = str(dest)
    return out


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, PROMPTS_MOUNT: prompts_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    """Prefetch F5-TTS checkpoint + vocoder into HF cache on volume."""
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"model": DEFAULT_MODEL, "repos": []}
    t0 = time.time()

    # Main F5TTS_v1_Base weights
    try:
        path = hf_hub_download(
            repo_id="SWivid/F5-TTS",
            filename="F5TTS_v1_Base/model_1250000.safetensors",
            token=token,
        )
        results["repos"].append(
            {
                "repo": "SWivid/F5-TTS",
                "file": "F5TTS_v1_Base/model_1250000.safetensors",
                "path": path,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
    except Exception as e:
        # fallback snapshot
        print(f"hf_hub_download failed ({e!r}); snapshot…", flush=True)
        d = snapshot_download(
            repo_id="SWivid/F5-TTS",
            allow_patterns=["F5TTS_v1_Base/*", "vocab.txt"],
            token=token,
        )
        results["repos"].append(
            {
                "repo": "SWivid/F5-TTS",
                "local": d,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )

    # Vocos
    try:
        t1 = time.time()
        v = snapshot_download(repo_id="charactr/vocos-mel-24khz", token=token)
        results["repos"].append(
            {"repo": "charactr/vocos-mel-24khz", "local": v, "elapsed_s": round(time.time() - t1, 1)}
        )
    except Exception as e:
        results["vocos_error"] = repr(e)

    try:
        results["refs"] = _ensure_refs()
    except Exception as e:
        results["ref_error"] = repr(e)

    results["hf_home"] = _dir_info(HF_HOME)
    results["ready"] = True
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
    out = {
        "app": APP_NAME,
        "slot": "033-f5tts",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "license": "Code MIT · Model CC-BY-NC",
        "hf_repo": "SWivid/F5-TTS",
        "weights": _dir_info(HF_HOME),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "HF ~740k · GH 15k · 零样本 · Tier A5",
        "modes": ["clone_en", "clone_zh"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _run_tts(
    *,
    text: str,
    run_name: str,
    gpu_label: str,
    lang: str = "en",
    ref_audio: str = "",
    ref_text: str = "",
    nfe_step: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from f5_tts.api import F5TTS

    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}

    refs = _ensure_refs()
    lang = (lang or "en").lower()
    if ref_audio.strip() and Path(ref_audio.strip()).is_file():
        ref_file = ref_audio.strip()
        ref_tx = ref_text or (REF_EN_TEXT if lang.startswith("en") else REF_ZH_TEXT)
    elif lang.startswith("zh"):
        ref_file = refs[REF_ZH]
        ref_tx = ref_text or REF_ZH_TEXT
    else:
        ref_file = refs[REF_EN]
        ref_tx = ref_text or REF_EN_TEXT

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "audio.wav"

    t0 = time.time()
    t_load = time.time()
    print(f"Loading {DEFAULT_MODEL}…", flush=True)
    f5 = F5TTS(model=DEFAULT_MODEL, hf_cache_dir=str(HF_HOME / "hub"))
    load_s = time.time() - t_load

    t_gen = time.time()
    wav, sr, _spec = f5.infer(
        ref_file=ref_file,
        ref_text=ref_tx,
        gen_text=text,
        file_wave=str(out_path),
        nfe_step=nfe_step,
        seed=seed,
    )
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not out_path.is_file():
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
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": DEFAULT_MODEL,
        "repo_id": "SWivid/F5-TTS",
        "lang": lang,
        "ref_audio": ref_file,
        "ref_text": ref_tx,
        "nfe_step": nfe_step,
        "seed": seed,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:800],
        "text_len": len(text),
        "license_note": "Code MIT · Model CC-BY-NC (Emilia)",
    }
    meta = {
        "experiment": "033-f5tts",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "lang": lang,
            "model": DEFAULT_MODEL,
            "nfe_step": nfe_step,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": result["license_note"],
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
    lang: str = "en",
    ref_audio: str = "",
    ref_text: str = "",
    nfe_step: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        run_name=run_name,
        gpu_label=gpu_label,
        lang=lang,
        ref_audio=ref_audio,
        ref_text=ref_text,
        nfe_step=nfe_step,
        seed=seed,
    )


SMOKE_EN = (
    "Hello from F5-TTS on Modal lab. "
    "This is a zero-shot voice clone demo with flow matching."
)
SMOKE_ZH = "你好，这是 modal-lab 第零三三号 F5-TTS 实验。零样本克隆，中英皆可。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="033 F5-TTS on Modal")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs / prompts Volume")

    download = sub.add_parser("download", help="下载权重与参考音频")
    download.add_argument("--force", action="store_true")

    smoke = sub.add_parser("smoke", help="运行固定 EN / ZH clone smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--kind", default="en", choices=["en", "zh"])
    smoke.add_argument("--run-name", default="")
    smoke.add_argument("--nfe-step", type=int, default=32)

    t2s = sub.add_parser("t2s", help="zero-shot Text-to-Speech")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--lang", default="en")
    t2s.add_argument("--run-name", default="")
    t2s.add_argument("--ref-audio", default="")
    t2s.add_argument("--ref-text", default="")
    t2s.add_argument("--nfe-step", type=int, default=32)
    t2s.add_argument("--seed", type=int, default=42)
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "033-f5tts",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "prompts_volume": VOLUME_PROMPTS,
        "refs": {
            "en": REF_EN,
            "zh": REF_ZH,
        },
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "zh":
        text = SMOKE_ZH
        run_name = args.run_name or "smoke_zh"
        lang = "zh"
    else:
        text = SMOKE_EN
        run_name = args.run_name or "smoke_en"
        lang = "en"
    return {
        "action": "smoke",
        "gpu": args.gpu,
        "kind": args.kind,
        "text": text,
        "run_name": run_name,
        "lang": lang,
        "nfe_step": args.nfe_step,
    }


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "t2s",
        "gpu": args.gpu,
        "text": args.text.strip(),
        "run_name": args.run_name,
        "lang": args.lang,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "nfe_step": args.nfe_step,
        "seed": args.seed,
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
        print(json.dumps(download_weights.remote(force=args.force), ensure_ascii=False, indent=2))
        return

    if args.command == "smoke":
        plan = smoke_plan(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        download_weights.remote(force=False)
        out = generate_fn.with_options(gpu=args.gpu).remote(
            text=plan["text"],
            run_name=plan["run_name"],
            gpu_label=args.gpu,
            lang=plan["lang"],
            nfe_step=args.nfe_step,
        )
        print("SMOKE_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        if (out.get("audio") or {}).get("duration_s", 0) < 0.5:
            raise SystemExit("smoke audio too short")
        if (out.get("audio") or {}).get("rms", 0) < 1e-4:
            raise SystemExit("smoke audio near silent")
        return

    plan = t2s_plan(args)
    if not plan["text"]:
        raise SystemExit("t2s requires non-empty --text")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    download_weights.remote(force=False)
    out = generate_fn.with_options(gpu=args.gpu).remote(
        text=plan["text"],
        run_name=args.run_name,
        gpu_label=args.gpu,
        lang=args.lang,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        nfe_step=args.nfe_step,
        seed=args.seed,
    )
    print("T2S_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
    if not out.get("success"):
        raise SystemExit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
