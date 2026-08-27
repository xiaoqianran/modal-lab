# -*- coding: utf-8 -*-
"""
034-higgs — Higgs Audio v2 3B TTS on Modal (lab TTS 收官号)

真实用量榜 Tier A6：AA Elo #7 · HF ~400k · GH 8k · 表情/多说话人
默认：bosonai/higgs-audio-v2-generation-3B-base @ github-compatible rev
GPU L40S

上游: https://github.com/boson-ai/higgs-audio
权重 pin（兼容 github loader，非 main 的 transformers-native flat config）：
  model @ 10840182ca4a · tokenizer @ 9d4988fbd4ad
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

APP_NAME = "modal-lab-higgs"

MODEL_REPO = "bosonai/higgs-audio-v2-generation-3B-base"
MODEL_REVISION = "10840182ca4a"
TOKENIZER_REPO = "bosonai/higgs-audio-v2-tokenizer"
TOKENIZER_REVISION = "9d4988fbd4ad"
DEFAULT_MODEL = "higgs_v2_3b"
DEFAULT_GPU = "L40S"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
LOCAL_MODEL = Path(WEIGHTS_MOUNT) / "models" / "higgs-audio-v2-generation-3B-base"
LOCAL_TOKENIZER = Path(WEIGHTS_MOUNT) / "models" / "higgs-audio-v2-tokenizer"
VOLUME_WEIGHTS = "modal-lab-higgs-weights"
VOLUME_OUTPUTS = "modal-lab-higgs-outputs"
VOLUME_PROMPTS = "modal-lab-higgs-prompts"

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

DOWNLOAD_TIMEOUT = 3 * 60 * 60
INFER_TIMEOUT = 60 * 60

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
        "g++",
    )
    .uv_pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "torchvision==0.20.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .uv_pip_install(
        "transformers>=4.45.1,<4.47.0",
        "accelerate>=0.26.0",
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "dacite",
        "json_repair",
        "pandas",
        "pydantic",
        "vector_quantize_pytorch",
        "loguru",
        "pydub",
        "omegaconf",
        "click",
        "langid",
        "jieba",
        "einops",
        "huggingface_hub[hf_transfer]>=0.26.0,<1.0",
        "descript-audio-codec",
        "tqdm",
        "safetensors",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/boson-ai/higgs-audio.git /opt/higgs-audio",
        "cd /opt/higgs-audio && pip install -e . --no-deps",
    )
    .env({
        **_HF_ENV,
        "PYTHONPATH": "/opt/higgs-audio",
    })
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


def _model_cfg_ok() -> bool:
    cfg = LOCAL_MODEL / "config.json"
    if not cfg.is_file():
        return False
    try:
        data = json.loads(cfg.read_text())
    except Exception:
        return False
    return "text_config" in data


def _tok_cfg_ok() -> bool:
    cfg = LOCAL_TOKENIZER / "config.json"
    pth = LOCAL_TOKENIZER / "model.pth"
    if not cfg.is_file() or not pth.is_file():
        return False
    try:
        data = json.loads(cfg.read_text())
    except Exception:
        return False
    # old github loader expects flat n_filters/D style, not acoustic_model_config
    return "n_filters" in data and "acoustic_model_config" not in data


def _weights_ready() -> bool:
    has_weights = any(LOCAL_MODEL.glob("*.safetensors")) or any(
        LOCAL_MODEL.glob("model-*.safetensors")
    )
    return _model_cfg_ok() and _tok_cfg_ok() and has_weights


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
    LOCAL_TOKENIZER.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "repos": [],
    }

    # Re-download if incompatible configs present
    if not _model_cfg_ok() and (LOCAL_MODEL / "config.json").is_file():
        print("Incompatible model config — wiping model dir…", flush=True)
        shutil.rmtree(LOCAL_MODEL)
        LOCAL_MODEL.mkdir(parents=True, exist_ok=True)
        force = True
    if not _tok_cfg_ok() and (LOCAL_TOKENIZER / "config.json").is_file():
        print("Incompatible tokenizer config — wiping tokenizer dir…", flush=True)
        shutil.rmtree(LOCAL_TOKENIZER)
        LOCAL_TOKENIZER.mkdir(parents=True, exist_ok=True)
        force = True

    for repo, dest, rev in (
        (MODEL_REPO, LOCAL_MODEL, MODEL_REVISION),
        (TOKENIZER_REPO, LOCAL_TOKENIZER, TOKENIZER_REVISION),
    ):
        t0 = time.time()
        if dest == LOCAL_MODEL and _model_cfg_ok() and any(LOCAL_MODEL.glob("*.safetensors")) and not force:
            results["repos"].append(
                {"repo": repo, "revision": rev, "skipped": True, **_dir_info(dest)}
            )
            continue
        if dest == LOCAL_TOKENIZER and _tok_cfg_ok() and not force:
            results["repos"].append(
                {"repo": repo, "revision": rev, "skipped": True, **_dir_info(dest)}
            )
            continue
        print(f"Downloading {repo}@{rev} → {dest}…", flush=True)
        snapshot_download(
            repo_id=repo,
            local_dir=str(dest),
            token=token,
            revision=rev,
        )
        results["repos"].append(
            {
                "repo": repo,
                "revision": rev,
                "skipped": False,
                "elapsed_s": round(time.time() - t0, 1),
                **_dir_info(dest),
            }
        )

    results["hf_home"] = _dir_info(HF_HOME)
    results["ready"] = _weights_ready()
    results["model_cfg_ok"] = _model_cfg_ok()
    results["tok_cfg_ok"] = _tok_cfg_ok()
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
        "slot": "034-higgs",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "license": "see boson-ai/higgs-audio",
        "model_repo": MODEL_REPO,
        "tokenizer_repo": TOKENIZER_REPO,
        "weights_ready": _weights_ready(),
        "model_dir": _dir_info(LOCAL_MODEL),
        "tokenizer_dir": _dir_info(LOCAL_TOKENIZER),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "AA Elo #7 · HF ~400k · Tier A6 · lab 收官",
        "modes": ["single_en", "scene_desc"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _run_tts(
    *,
    text: str,
    run_name: str,
    gpu_label: str,
    scene: str = "",
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torchaudio

    if "/opt/higgs-audio" not in sys.path:
        sys.path.insert(0, "/opt/higgs-audio")

    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}
    if not _weights_ready():
        return {
            "success": False,
            "error": "weights missing or incompatible; run download (force if needed)",
            "model_cfg_ok": _model_cfg_ok(),
            "tok_cfg_ok": _tok_cfg_ok(),
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "audio.wav"

    from boson_multimodal.serve.serve_engine import (
        HiggsAudioResponse,
        HiggsAudioServeEngine,
    )
    from boson_multimodal.data_types import ChatMLSample, Message

    scene_desc = scene.strip() or "Audio is recorded from a quiet room."
    system_prompt = (
        "Generate audio following instruction.\n\n"
        f"<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=text),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    t_load = time.time()
    print(
        f"Loading Higgs model@{MODEL_REVISION} tok@{TOKENIZER_REVISION}…",
        flush=True,
    )
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    serve_engine = HiggsAudioServeEngine(
        str(LOCAL_MODEL),
        str(LOCAL_TOKENIZER),
        device=device,
    )
    load_s = time.time() - t_load

    t_gen = time.time()
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    audio = np.asarray(output.audio, dtype=np.float32).reshape(-1)
    sr = int(getattr(output, "sampling_rate", 24000) or 24000)
    torchaudio.save(str(out_path), torch.from_numpy(audio)[None, :], sr)

    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    audio_info = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sample_rate": sr,
        "channels": 1,
        "samples": int(audio.shape[0]),
        "duration_s": round(float(audio.shape[0]) / float(sr), 3),
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
        "repo_id": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "scene": scene_desc,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:800],
        "text_len": len(text),
        "license_note": "Higgs Audio v2 — see boson-ai/higgs-audio",
    }
    meta = {
        "experiment": "034-higgs",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "scene": scene_desc,
            "model": DEFAULT_MODEL,
            "temperature": temperature,
            "revision": MODEL_REVISION,
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
    memory=49152,
    scaledown_window=60,
)
def generate_fn(
    text: str,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    scene: str = "",
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        run_name=run_name,
        gpu_label=gpu_label,
        scene=scene,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )


SMOKE_EN = (
    "The sun rises in the east and sets in the west. "
    "This simple fact has been observed by humans for thousands of years."
)
SMOKE_EXPRESSIVE = (
    "Wow! Can you believe it? We finally shipped the last TTS experiment — "
    "Higgs Audio on Modal lab. What a journey!"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="034 Higgs Audio v2 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs Volume 状态")

    download = sub.add_parser("download", help="下载 pinned model/tokenizer 到 Volume")
    download.add_argument("--force", action="store_true")

    smoke = sub.add_parser("smoke", help="运行固定文本 smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--kind", default="en", choices=["en", "expressive"])
    smoke.add_argument("--run-name", default="")
    smoke.add_argument("--scene", default="")
    smoke.add_argument("--temperature", type=float, default=0.3)
    smoke.add_argument("--max-new-tokens", type=int, default=1024)

    t2s = sub.add_parser("t2s", help="Text-to-Speech")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--scene", default="")
    t2s.add_argument("--run-name", default="")
    t2s.add_argument("--temperature", type=float, default=0.3)
    t2s.add_argument("--max-new-tokens", type=int, default=1024)
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "034-higgs",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "prompts_volume": VOLUME_PROMPTS,
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "expressive":
        text = SMOKE_EXPRESSIVE
        run_name = args.run_name or "smoke_expressive"
        scene = args.scene or "A lively studio with an excited narrator."
    else:
        text = SMOKE_EN
        run_name = args.run_name or "smoke_en"
        scene = args.scene or "Audio is recorded from a quiet room."
    return {
        "action": "smoke",
        "gpu": args.gpu,
        "kind": args.kind,
        "text": text,
        "run_name": run_name,
        "scene": scene,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "t2s",
        "gpu": args.gpu,
        "text": args.text.strip(),
        "run_name": args.run_name,
        "scene": args.scene,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
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
            scene=plan["scene"],
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
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
        scene=args.scene,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    print("T2S_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
    if not out.get("success"):
        raise SystemExit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
