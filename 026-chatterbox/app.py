# -*- coding: utf-8 -*-
"""
026-chatterbox — Resemble AI Chatterbox TTS on Modal

真实用量榜 Tier S2：HF ~2.1M dl · GH ~26k stars · AA Elo ~1014 · MIT
默认：multilingual（23 语含中文）· GPU L4
可选：turbo（350M 英文 · 克隆 + [chuckle]）· original

上游: https://github.com/resemble-ai/chatterbox
Modal 官方例: https://modal.com/docs/examples/chatterbox_tts
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-chatterbox"

HF_REPOS = {
    "turbo": "ResembleAI/chatterbox-turbo",
    "original": "ResembleAI/chatterbox",
    "multilingual": "ResembleAI/chatterbox",
}
DEFAULT_MODEL = "multilingual"
DEFAULT_GPU = "L4"
DEFAULT_VOICE = "Lucy"
DEFAULT_LANG = "en"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
VOLUME_WEIGHTS = "modal-lab-chatterbox-weights"
VOLUME_OUTPUTS = "modal-lab-chatterbox-outputs"
VOLUME_PROMPTS = "modal-lab-chatterbox-prompts"

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
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(_HF_ENV)
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git", "ca-certificates")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "chatterbox-tts==0.1.7",
        "peft>=0.18.0",
        "soundfile",
        "numpy",
        "scipy",
        "huggingface_hub[hf_transfer]>=0.26.0",
        "librosa",
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
        "multilingual": "multilingual",
        "mtl": "multilingual",
        "v3": "multilingual",
        "turbo": "turbo",
        "nano": "turbo",
        "original": "original",
        "en": "original",
        "english": "original",
        "default": "multilingual",
    }
    if n in aliases:
        return aliases[n]
    if n in HF_REPOS:
        return n
    raise ValueError(f"unknown model {name!r}; use multilingual|turbo|original")


def _weights_ready(model_key: str) -> bool:
    hub = HF_HOME / "hub"
    if not hub.is_dir():
        return False
    if model_key == "turbo":
        return any(hub.glob("models--ResembleAI--chatterbox-turbo/**/*"))
    return any(hub.glob("models--ResembleAI--chatterbox/**/*"))


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, PROMPTS_MOUNT: prompts_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(
    force: bool = False,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    key = _norm_model(model)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"model": key, "repos": []}
    if key == "turbo":
        repos_to_get = [HF_REPOS["turbo"]]
    elif key == "original":
        repos_to_get = [HF_REPOS["original"]]
    else:
        repos_to_get = [HF_REPOS["multilingual"], HF_REPOS["turbo"]]

    for repo in repos_to_get:
        t0 = time.time()
        slug = "models--" + repo.replace("/", "--")
        already = any((HF_HOME / "hub").glob(f"{slug}/**/*")) if not force else False
        if already and not force:
            results["repos"].append(
                {"repo": repo, "skipped": True, **_dir_info(HF_HOME / "hub" / slug)}
            )
            continue
        print(f"Downloading {repo}…", flush=True)
        snapshot_download(
            repo_id=repo,
            cache_dir=str(HF_HOME / "hub"),
            token=token,
        )
        results["repos"].append(
            {
                "repo": repo,
                "skipped": False,
                "elapsed_s": round(time.time() - t0, 1),
                **_dir_info(HF_HOME / "hub" / slug),
            }
        )

    prompts_root = Path(PROMPTS_MOUNT)
    n_wav = len(list(prompts_root.glob("**/*.wav")))
    results["prompts"] = {"n_wav": n_wav, "path": str(prompts_root)}
    results["hf_home"] = _dir_info(HF_HOME)
    results["ready"] = _weights_ready(key)
    weights_vol.commit()
    prompts_vol.commit()
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


@app.function(
    image=download_image,
    volumes={PROMPTS_MOUNT: prompts_vol},
    timeout=15 * 60,
    cpu=2,
    memory=4096,
)
def upload_prompts(files: dict[str, bytes]) -> dict[str, Any]:
    root = Path(PROMPTS_MOUNT)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        written.append({"path": str(p), "size": len(data)})
    prompts_vol.commit()
    out = {"n": len(written), "files": written[:50]}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


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
        "slot": "026-chatterbox",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "default_voice": DEFAULT_VOICE,
        "license": "MIT",
        "models_ready": {
            "turbo": _weights_ready("turbo"),
            "original": _weights_ready("original"),
            "multilingual": _weights_ready("multilingual"),
        },
        "hf_home": _dir_info(HF_HOME),
        "prompts": {
            "n": len(prompts),
            "names": [p.stem for p in prompts[:30]],
        },
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "HF ~2.1M · GH 26k · AA Elo ~1014 · Modal official example",
        "gpu_note": "L4 default; turbo may fit T4; original/mtl prefer L4+",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _resolve_prompt_path(voice: str) -> str | None:
    if not voice:
        return None
    v = voice.strip()
    root = Path(PROMPTS_MOUNT)
    if root.is_dir():
        for p in root.rglob("*.wav"):
            if p.stem.lower() == v.lower() or p.name.lower() == v.lower():
                return str(p)
    for c in [
        root / f"{v}.wav",
        root / v,
        root / "prompts" / f"{v}.wav",
    ]:
        if c.is_file():
            return str(c)
    return None


def _call_from_pretrained(cls, device: str, **kwargs):
    """Pass only kwargs accepted by this package version."""
    sig = inspect.signature(cls.from_pretrained)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls.from_pretrained(device=device, **filtered)


def _run_tts(
    *,
    text: str,
    model: str,
    lang: str,
    voice: str,
    audio_prompt_path: str,
    exaggeration: float,
    cfg_weight: float,
    run_name: str,
    gpu_label: str,
    nano: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio as ta

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    # drop deprecated TRANSFORMERS_CACHE to silence warnings
    os.environ.pop("TRANSFORMERS_CACHE", None)
    HF_HOME.mkdir(parents=True, exist_ok=True)

    key = _norm_model(model)
    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}

    ref = audio_prompt_path.strip() if audio_prompt_path else ""
    if not ref and voice:
        found = _resolve_prompt_path(voice)
        if found:
            ref = found
    if key == "turbo" and not ref:
        ref = _resolve_prompt_path(DEFAULT_VOICE) or ""
        if not ref:
            return {
                "success": False,
                "error": "turbo requires voice prompt — run upload-prompts",
            }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    t_load = time.time()
    gen_kwargs_used: dict[str, Any] = {}

    if key == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model_obj = _call_from_pretrained(ChatterboxTurboTTS, device, nano=nano)
        load_s = time.time() - t_load
        t_gen = time.time()
        wav = model_obj.generate(text, audio_prompt_path=ref)
        gen_s = time.time() - t_gen
        gen_kwargs_used = {"audio_prompt_path": ref}
        sr = int(model_obj.sr)
    elif key == "original":
        from chatterbox.tts import ChatterboxTTS

        model_obj = _call_from_pretrained(ChatterboxTTS, device)
        load_s = time.time() - t_load
        t_gen = time.time()
        kwargs: dict[str, Any] = {
            "exaggeration": float(exaggeration),
            "cfg_weight": float(cfg_weight),
        }
        if ref:
            kwargs["audio_prompt_path"] = ref
        # filter generate kwargs if needed
        gen_sig = inspect.signature(model_obj.generate)
        kwargs = {k: v for k, v in kwargs.items() if k in gen_sig.parameters}
        wav = model_obj.generate(text, **kwargs)
        gen_s = time.time() - t_gen
        gen_kwargs_used = kwargs
        sr = int(model_obj.sr)
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        model_obj = _call_from_pretrained(
            ChatterboxMultilingualTTS, device, t3_model="v3"
        )
        load_s = time.time() - t_load
        t_gen = time.time()
        lang_id = (lang or "en").strip().lower()
        kwargs = {
            "language_id": lang_id,
            "exaggeration": float(exaggeration),
            "cfg_weight": float(cfg_weight),
        }
        if ref:
            kwargs["audio_prompt_path"] = ref
        gen_sig = inspect.signature(model_obj.generate)
        kwargs = {k: v for k, v in kwargs.items() if k in gen_sig.parameters}
        wav = model_obj.generate(text, **kwargs)
        gen_s = time.time() - t_gen
        gen_kwargs_used = kwargs
        sr = int(model_obj.sr)

    wall = time.time() - t0

    if isinstance(wav, torch.Tensor):
        w = wav.detach().cpu().float()
        if w.ndim == 1:
            w = w.unsqueeze(0)
        out_path = save_dir / "audio.wav"
        ta.save(str(out_path), w, sr)
        arr = w.numpy().reshape(-1)
    else:
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
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
        "duration_s": round(float(arr.shape[0]) / sr, 3),
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
        "nano": nano if key == "turbo" else False,
        "lang": lang,
        "voice": voice,
        "audio_prompt_path": ref or None,
        "gen_kwargs": gen_kwargs_used,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:500],
        "text_len": len(text),
    }
    meta = {
        "experiment": "026-chatterbox",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "model": key,
            "lang": lang,
            "voice": voice,
            "audio_prompt_path": ref or None,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "MIT — Resemble AI Chatterbox",
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
    memory=16384,
    scaledown_window=60,
)
def generate_fn(
    text: str,
    model: str = DEFAULT_MODEL,
    lang: str = DEFAULT_LANG,
    voice: str = DEFAULT_VOICE,
    audio_prompt_path: str = "",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    nano: bool = False,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        model=model,
        lang=lang,
        voice=voice,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        run_name=run_name,
        gpu_label=gpu_label,
        nano=nano,
    )


SMOKE_EN = (
    "Chatterbox is running on Modal. This is the multilingual model — "
    "natural speech across many languages."
)
SMOKE_ZH = "你好，今天天气真不错，希望你有一个愉快的周末。这是 modal-lab 第零二六号 Chatterbox 实验。"
SMOKE_TURBO = (
    "Hi there, Sarah here from MochaFone calling you back [chuckle], "
    "have you got one minute to chat about the billing issue?"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="026 Chatterbox on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查模型 / prompts / outputs Volume")

    download = sub.add_parser("download", help="下载指定 Chatterbox 模型")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--model", default=DEFAULT_MODEL)

    smoke = sub.add_parser("smoke", help="固定 multilingual EN/ZH 或 Turbo smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--kind", default="mtl_en", choices=["mtl_en", "mtl_zh", "turbo"])
    smoke.add_argument("--voice", default=DEFAULT_VOICE)
    smoke.add_argument("--audio-prompt", default="")
    smoke.add_argument("--exaggeration", type=float, default=0.5)
    smoke.add_argument("--cfg-weight", type=float, default=0.5)
    smoke.add_argument("--run-name", default="")
    smoke.add_argument("--nano", action="store_true")

    t2s = sub.add_parser("t2s", help="通用 Chatterbox TTS")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--model", default=DEFAULT_MODEL)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--lang", default=DEFAULT_LANG)
    t2s.add_argument("--voice", default="")
    t2s.add_argument("--audio-prompt", default="", help="prompts Volume 内远程 wav 路径/voice 名")
    t2s.add_argument("--exaggeration", type=float, default=0.5)
    t2s.add_argument("--cfg-weight", type=float, default=0.5)
    t2s.add_argument("--run-name", default="")
    t2s.add_argument("--nano", action="store_true")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "026-chatterbox",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "default_voice": DEFAULT_VOICE,
        "models": HF_REPOS,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "prompts_volume": VOLUME_PROMPTS,
        "prompt_note": "本地 inputs/voices/*.wav 使用 modal volume put 上传，不由推理命令隐式修改 Volume",
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "mtl_zh":
        model, text, lang = "multilingual", SMOKE_ZH, "zh"
        run_name, voice = args.run_name or "smoke_mtl_zh", ""
    elif args.kind == "turbo":
        model, text, lang = "turbo", SMOKE_TURBO, "en"
        run_name, voice = args.run_name or "smoke_turbo_lucy", args.voice or DEFAULT_VOICE
    else:
        model, text, lang = "multilingual", SMOKE_EN, "en"
        run_name, voice = args.run_name or "smoke_mtl_en", ""
    return {
        "action": "smoke",
        "gpu": args.gpu,
        "kind": args.kind,
        "model": model,
        "text": text,
        "lang": lang,
        "voice": voice,
        "audio_prompt": args.audio_prompt,
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "run_name": run_name,
        "nano": args.nano if model == "turbo" else False,
    }


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    model = _norm_model(args.model)
    return {
        "action": "t2s",
        "gpu": args.gpu,
        "model": model,
        "text": args.text.strip(),
        "lang": args.lang,
        "voice": args.voice,
        "audio_prompt": args.audio_prompt,
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "run_name": args.run_name,
        "nano": args.nano if model == "turbo" else False,
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
        try:
            model = _norm_model(args.model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        plan = {"action": "download", "model": model, "force": args.force}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(download_weights.remote(force=args.force, model=model), ensure_ascii=False, indent=2))
        return

    try:
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
        model=plan["model"],
        lang=plan["lang"],
        voice=plan["voice"],
        audio_prompt_path=plan["audio_prompt"],
        exaggeration=plan["exaggeration"],
        cfg_weight=plan["cfg_weight"],
        run_name=plan["run_name"],
        gpu_label=args.gpu,
        nano=plan["nano"],
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
