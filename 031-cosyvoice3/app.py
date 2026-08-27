# -*- coding: utf-8 -*-
"""
031-cosyvoice3 — Fun-CosyVoice3-0.5B on Modal

真实用量榜 Tier A3：GH CosyVoice ~22.7k · 中文方言 SOTA
默认：FunAudioLLM/Fun-CosyVoice3-0.5B-2512 · GPU L4 · Apache-2.0

上游: https://github.com/FunAudioLLM/CosyVoice
权重: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
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

APP_NAME = "modal-lab-cosyvoice3"

HF_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_MODEL = "cosyvoice3_0.5b"
DEFAULT_GPU = "L4"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
LOCAL_MODEL = Path(WEIGHTS_MOUNT) / "models" / "Fun-CosyVoice3-0.5B"
VOLUME_WEIGHTS = "modal-lab-cosyvoice3-weights"
VOLUME_OUTPUTS = "modal-lab-cosyvoice3-outputs"
VOLUME_PROMPTS = "modal-lab-cosyvoice3-prompts"

PROMPT_WAV_NAME = "zero_shot_prompt.wav"
PROMPT_WAV_URL = (
    "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"
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

# Slim CosyVoice deps (skip tensorrt/deepspeed/gradio stack)
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "sox",
        "libsox-dev",
        "git",
        "ca-certificates",
        "curl",
        "build-essential",
        "g++",
    )
    .uv_pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .uv_pip_install("setuptools", "wheel", "packaging")
    .uv_pip_install("openai-whisper")
    .uv_pip_install(
        "transformers==4.51.3",
        "accelerate",
        "numpy==1.26.4",
        "scipy",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "onnxruntime-gpu==1.18.0",
        "onnx==1.16.0",
        "hyperpyyaml==1.2.3",
        "omegaconf==2.3.0",
        "hydra-core==1.3.2",
        "inflect==7.3.1",
        "wetext==0.0.4",
        "conformer==0.3.2",
        "diffusers==0.29.0",
        "modelscope==1.20.0",
        "huggingface_hub[hf_transfer]>=0.26.0,<1.0",
        "tqdm",
        "einops",
        "pyarrow",
        "protobuf==4.25",
        "networkx",
        "matplotlib",
        "x-transformers==2.11.24",
        "gdown",
        "wget",
        "lightning==2.2.4",
        "pydantic==2.7.0",
        "pyworld==0.3.4",
        "rich",
        "tiktoken",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice",
        "cd /opt/CosyVoice && git submodule update --init --recursive || true",
        # Matcha-TTS is required on path
        "test -d /opt/CosyVoice/third_party/Matcha-TTS || "
        "(cd /opt/CosyVoice && git submodule update --init --recursive third_party/Matcha-TTS || "
        "git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git third_party/Matcha-TTS)",
    )
    .env({
        **_HF_ENV,
        "PYTHONPATH": "/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS",
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


def _weights_ready() -> bool:
    # CosyVoice3 needs llm.pt + flow.pt at minimum
    if (LOCAL_MODEL / "llm.pt").is_file() and (LOCAL_MODEL / "flow.pt").is_file():
        return True
    return False


def _ensure_prompt_wav() -> Path:
    dest = Path(PROMPTS_MOUNT) / PROMPT_WAV_NAME
    if not dest.is_file():
        Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)
        print(f"Fetching prompt wav → {dest}", flush=True)
        urllib.request.urlretrieve(PROMPT_WAV_URL, dest)
    return dest


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

    try:
        pw = _ensure_prompt_wav()
        results["prompt_wav"] = str(pw)
    except Exception as e:
        results["prompt_error"] = repr(e)

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
    out = {
        "app": APP_NAME,
        "slot": "031-cosyvoice3",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "license": "Apache-2.0",
        "hf_repo": HF_REPO,
        "weights_ready": _weights_ready(),
        "model_dir": _dir_info(LOCAL_MODEL),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "GH CosyVoice ~22.7k · 中文方言 SOTA · Tier A3",
        "modes": ["zero_shot_zh", "instruct_dialect", "zero_shot_en"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _run_tts(
    *,
    text: str,
    run_name: str,
    gpu_label: str,
    mode: str = "zero_shot",
    prompt_text: str = "",
    instruct: str = "",
    prompt_wav: str = "",
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    for p in ("/opt/CosyVoice", "/opt/CosyVoice/third_party/Matcha-TTS"):
        if p not in sys.path:
            sys.path.insert(0, p)

    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}
    if not _weights_ready():
        return {"success": False, "error": "weights missing; run download first"}

    try:
        prompt_path = _ensure_prompt_wav()
    except Exception:
        prompt_path = Path(PROMPTS_MOUNT) / PROMPT_WAV_NAME
    if prompt_wav.strip():
        cand = Path(prompt_wav.strip())
        if cand.is_file():
            prompt_path = cand

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from cosyvoice.cli.cosyvoice import AutoModel

    t0 = time.time()
    t_load = time.time()
    print(f"Loading CosyVoice3 from {LOCAL_MODEL}…", flush=True)
    cosyvoice = AutoModel(model_dir=str(LOCAL_MODEL))
    load_s = time.time() - t_load

    mode = (mode or "zero_shot").lower().strip()
    prompt_text = prompt_text or "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    chunks = []
    t_gen = time.time()
    if mode in ("instruct", "instruct2", "dialect"):
        ins = instruct or "You are a helpful assistant. 请用四川话说这句话。<|endofprompt|>"
        for _, j in enumerate(
            cosyvoice.inference_instruct2(
                text, ins, str(prompt_path), stream=False
            )
        ):
            chunks.append(j["tts_speech"])
    elif mode in ("en", "cross_lingual"):
        for _, j in enumerate(
            cosyvoice.inference_cross_lingual(
                f"You are a helpful assistant.<|endofprompt|>{text}",
                str(prompt_path),
                stream=False,
            )
        ):
            chunks.append(j["tts_speech"])
    else:
        # zero_shot default
        for _, j in enumerate(
            cosyvoice.inference_zero_shot(
                text, prompt_text, str(prompt_path), stream=False
            )
        ):
            chunks.append(j["tts_speech"])
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    if not chunks:
        return {"success": False, "error": "no audio chunks", "wall_s": round(wall, 2)}

    wav = torch.cat(chunks, dim=-1)
    arr = wav.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    sr = int(getattr(cosyvoice, "sample_rate", 24000) or 24000)

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
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": DEFAULT_MODEL,
        "repo_id": HF_REPO,
        "mode": mode,
        "prompt_wav": str(prompt_path),
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:800],
        "text_len": len(text),
        "license_note": "Apache-2.0 — Fun-CosyVoice3-0.5B",
    }
    meta = {
        "experiment": "031-cosyvoice3",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {"text": text, "mode": mode, "model": DEFAULT_MODEL},
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Apache-2.0 — Fun-CosyVoice3",
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
    mode: str = "zero_shot",
    prompt_text: str = "",
    instruct: str = "",
    prompt_wav: str = "",
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        run_name=run_name,
        gpu_label=gpu_label,
        mode=mode,
        prompt_text=prompt_text,
        instruct=instruct,
        prompt_wav=prompt_wav,
    )


SMOKE_ZH = "你好，这是 modal-lab 第零三一号 CosyVoice3 实验。今天天气真不错，希望你有一个愉快的周末。"
SMOKE_TONGUE = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
SMOKE_DIALECT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。"
SMOKE_EN = "Hello from CosyVoice3 on Modal lab. This is a zero-shot multilingual speech synthesis demo."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="031 CosyVoice3 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs / prompts Volume")

    download = sub.add_parser("download", help="下载权重与默认 prompt wav")
    download.add_argument("--force", action="store_true")

    smoke = sub.add_parser("smoke", help="运行固定文本 smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--kind", default="zh", choices=["zh", "tongue", "dialect", "en"])
    smoke.add_argument("--run-name", default="")
    smoke.add_argument("--instruct", default="")
    smoke.add_argument("--prompt-text", default="")
    smoke.add_argument("--prompt-wav", default="")

    t2s = sub.add_parser("t2s", help="Text-to-Speech")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--mode", default="zero_shot")
    t2s.add_argument("--instruct", default="")
    t2s.add_argument("--prompt-text", default="")
    t2s.add_argument("--prompt-wav", default="")
    t2s.add_argument("--run-name", default="")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "031-cosyvoice3",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "hf_repo": HF_REPO,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "prompts_volume": VOLUME_PROMPTS,
        "prompt_wav": PROMPT_WAV_NAME,
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "dialect":
        text = SMOKE_DIALECT
        run_name = args.run_name or "smoke_dialect"
        mode = "instruct"
        instruct = args.instruct or "You are a helpful assistant. 请用四川话说这句话。<|endofprompt|>"
    elif args.kind == "en":
        text = SMOKE_EN
        run_name = args.run_name or "smoke_en"
        mode = "en"
        instruct = ""
    elif args.kind == "tongue":
        text = SMOKE_TONGUE
        run_name = args.run_name or "smoke_tongue"
        mode = "zero_shot"
        instruct = ""
    else:
        text = SMOKE_ZH
        run_name = args.run_name or "smoke_zh"
        mode = "zero_shot"
        instruct = ""
    return {
        "action": "smoke",
        "gpu": args.gpu,
        "kind": args.kind,
        "text": text,
        "run_name": run_name,
        "mode": mode,
        "instruct": instruct,
        "prompt_text": args.prompt_text,
        "prompt_wav": args.prompt_wav,
    }


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "t2s",
        "gpu": args.gpu,
        "text": args.text.strip(),
        "run_name": args.run_name,
        "mode": args.mode,
        "instruct": args.instruct,
        "prompt_text": args.prompt_text,
        "prompt_wav": args.prompt_wav,
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
            mode=plan["mode"],
            instruct=plan["instruct"],
            prompt_text=args.prompt_text,
            prompt_wav=args.prompt_wav,
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
        mode=args.mode,
        instruct=args.instruct,
        prompt_text=args.prompt_text,
        prompt_wav=args.prompt_wav,
    )
    print("T2S_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
    if not out.get("success"):
        raise SystemExit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
