# -*- coding: utf-8 -*-
"""
027-qwen3-tts — Qwen3-TTS family on Modal

真实用量榜 Tier S3：家族合计 ~6.6M+ HF dl · GH ~13k · Apache-2.0
默认：1.7B-CustomVoice（预设音色 + instruct）· GPU L4
可选：0.6B-CustomVoice · 1.7B-Base（3s 克隆）· 1.7B-VoiceDesign

上游: https://github.com/QwenLM/Qwen3-TTS
PyPI: qwen-tts
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

APP_NAME = "modal-lab-qwen3-tts"

HF_REPOS = {
    "custom_0.6": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "custom_1.7": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base_1.7": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "design_1.7": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
}
# aliases → key
MODEL_ALIASES = {
    "custom": "custom_1.7",
    "custom_1.7": "custom_1.7",
    "customvoice": "custom_1.7",
    "1.7b": "custom_1.7",
    "custom_0.6": "custom_0.6",
    "0.6": "custom_0.6",
    "0.6b": "custom_0.6",
    "base": "base_1.7",
    "base_1.7": "base_1.7",
    "clone": "base_1.7",
    "design": "design_1.7",
    "design_1.7": "design_1.7",
    "voicedesign": "design_1.7",
    "default": "custom_1.7",
}

DEFAULT_MODEL = "custom_1.7"
DEFAULT_GPU = "L4"
DEFAULT_SPEAKER = "Vivian"
DEFAULT_LANG = "Chinese"

# Official demo ref for clone smoke (no local upload required)
DEFAULT_CLONE_REF_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
)
DEFAULT_CLONE_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
VOLUME_WEIGHTS = "modal-lab-qwen3-tts-weights"
VOLUME_OUTPUTS = "modal-lab-qwen3-tts-outputs"
VOLUME_PROMPTS = "modal-lab-qwen3-tts-prompts"

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
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(_HF_ENV)
)

# qwen-tts pins transformers==4.57.3 / accelerate==1.12.0
# Prefer SDPA (no flash-attn wheel build); try flash_attention_2 at runtime if present.
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "sox",
        "libsox-dev",
        "git",
        "ca-certificates",
    )
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "qwen-tts==0.1.1",
        "soundfile",
        "numpy",
        "scipy",
        "librosa",
        "huggingface_hub[hf_transfer]>=0.26.0",
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
    n = (name or DEFAULT_MODEL).strip().lower().replace("-", "_").replace(" ", "")
    if n in MODEL_ALIASES:
        return MODEL_ALIASES[n]
    if n in HF_REPOS and n != "tokenizer":
        return n
    raise ValueError(
        f"unknown model {name!r}; use custom_1.7|custom_0.6|base_1.7|design_1.7"
    )


def _repo_slug(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _weights_ready(model_key: str) -> bool:
    hub = HF_HOME / "hub"
    if not hub.is_dir():
        return False
    repo = HF_REPOS[model_key]
    return any(hub.glob(f"{_repo_slug(repo)}/**/*"))


def _lang_label(lang: str) -> str:
    """Map short codes → Qwen3-TTS language names."""
    raw = (lang or "Chinese").strip()
    low = raw.lower()
    table = {
        "zh": "Chinese",
        "cn": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
        "ja": "Japanese",
        "jp": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "kr": "Korean",
        "korean": "Korean",
        "de": "German",
        "german": "German",
        "fr": "French",
        "french": "French",
        "ru": "Russian",
        "russian": "Russian",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "es": "Spanish",
        "spanish": "Spanish",
        "it": "Italian",
        "italian": "Italian",
        "auto": "Auto",
    }
    if low in table:
        return table[low]
    # already full name?
    for v in table.values():
        if raw == v:
            return v
    return raw


def _resolve_prompt_path(voice: str) -> str | None:
    if not voice:
        return None
    v = voice.strip()
    # URL / absolute remote already
    if v.startswith("http://") or v.startswith("https://"):
        return v
    root = Path(PROMPTS_MOUNT)
    if root.is_dir():
        for p in root.rglob("*.wav"):
            if p.stem.lower() == v.lower() or p.name.lower() == v.lower():
                return str(p)
    for c in [root / f"{v}.wav", root / v]:
        if c.is_file():
            return str(c)
    return None


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

    raw_model = (model or DEFAULT_MODEL).strip()
    want_all = raw_model.lower() in ("all", "family")
    key = DEFAULT_MODEL if want_all else _norm_model(raw_model)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"model": "all" if want_all else key, "repos": []}
    if want_all:
        repos_to_get = list(dict.fromkeys(HF_REPOS.values()))
    else:
        repos_to_get = [HF_REPOS["tokenizer"], HF_REPOS[key]]

    for repo in repos_to_get:
        t0 = time.time()
        slug = _repo_slug(repo)
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

    results["prompts"] = {
        "n_wav": len(list(Path(PROMPTS_MOUNT).glob("**/*.wav"))),
        "path": PROMPTS_MOUNT,
    }
    results["hf_home"] = _dir_info(HF_HOME)
    results["ready"] = True if want_all else _weights_ready(key)
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
        "slot": "027-qwen3-tts",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "default_speaker": DEFAULT_SPEAKER,
        "license": "Apache-2.0",
        "models_ready": {k: _weights_ready(k) for k in HF_REPOS if k != "tokenizer"},
        "tokenizer_ready": any(
            (HF_HOME / "hub").glob(f"{_repo_slug(HF_REPOS['tokenizer'])}/**/*")
        )
        if HF_HOME.is_dir()
        else False,
        "hf_home": _dir_info(HF_HOME),
        "prompts": {"n": len(prompts), "names": [p.stem for p in prompts[:30]]},
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "HF family ~6.6M+ · GH ~13k · Apache · Tier S3",
        "gpu_note": "L4 default; 0.6B may fit T4; 1.7B prefer L4+",
        "speakers": [
            "Vivian",
            "Serena",
            "Uncle_Fu",
            "Dylan",
            "Eric",
            "Ryan",
            "Aiden",
            "Ono_Anna",
            "Sohee",
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _pick_attn() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _run_tts(
    *,
    text: str,
    model: str,
    lang: str,
    speaker: str,
    instruct: str,
    ref_audio: str,
    ref_text: str,
    run_name: str,
    gpu_label: str,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    HF_HOME.mkdir(parents=True, exist_ok=True)

    key = _norm_model(model)
    repo_id = HF_REPOS[key]
    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}

    lang_use = _lang_label(lang)
    speaker_use = (speaker or DEFAULT_SPEAKER).strip()
    instruct_use = (instruct or "").strip()
    ref = (ref_audio or "").strip()
    if not ref:
        found = _resolve_prompt_path(speaker_use) if key == "base_1.7" else None
        if found:
            ref = found
    if key == "base_1.7" and not ref:
        ref = DEFAULT_CLONE_REF_URL
    ref_text_use = (ref_text or "").strip()
    if key == "base_1.7" and not ref_text_use and ref == DEFAULT_CLONE_REF_URL:
        ref_text_use = DEFAULT_CLONE_REF_TEXT

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    attn = _pick_attn()
    t0 = time.time()
    t_load = time.time()

    load_kwargs: dict[str, Any] = {
        "device_map": device if device != "cpu" else "cpu",
        "dtype": dtype,
        "attn_implementation": attn,
    }
    try:
        model_obj = Qwen3TTSModel.from_pretrained(repo_id, **load_kwargs)
    except Exception as e:
        # retry without flash / with sdpa
        if attn != "sdpa":
            print(f"load failed with {attn}: {e!r}; retry sdpa", flush=True)
            load_kwargs["attn_implementation"] = "sdpa"
            attn = "sdpa"
            model_obj = Qwen3TTSModel.from_pretrained(repo_id, **load_kwargs)
        else:
            raise
    load_s = time.time() - t_load

    t_gen = time.time()
    gen_meta: dict[str, Any] = {"mode": key, "attn": attn}

    if key in ("custom_1.7", "custom_0.6"):
        kwargs: dict[str, Any] = {
            "text": text,
            "language": lang_use,
            "speaker": speaker_use,
        }
        if instruct_use:
            kwargs["instruct"] = instruct_use
        wavs, sr = model_obj.generate_custom_voice(**kwargs)
        gen_meta.update(
            {
                "api": "generate_custom_voice",
                "speaker": speaker_use,
                "instruct": instruct_use or None,
                "language": lang_use,
            }
        )
    elif key == "design_1.7":
        if not instruct_use:
            return {
                "success": False,
                "error": "design model requires --instruct description",
            }
        wavs, sr = model_obj.generate_voice_design(
            text=text,
            language=lang_use,
            instruct=instruct_use,
        )
        gen_meta.update(
            {
                "api": "generate_voice_design",
                "instruct": instruct_use,
                "language": lang_use,
            }
        )
    else:  # base_1.7 clone
        if not ref:
            return {"success": False, "error": "clone/base requires ref_audio"}
        clone_kwargs: dict[str, Any] = {
            "text": text,
            "language": lang_use,
            "ref_audio": ref,
        }
        if ref_text_use:
            clone_kwargs["ref_text"] = ref_text_use
        else:
            # x-vector only fallback
            clone_kwargs["x_vector_only_mode"] = True
        wavs, sr = model_obj.generate_voice_clone(**clone_kwargs)
        gen_meta.update(
            {
                "api": "generate_voice_clone",
                "ref_audio": ref,
                "ref_text": ref_text_use or None,
                "language": lang_use,
            }
        )

    gen_s = time.time() - t_gen
    wall = time.time() - t0

    # wavs is list of arrays
    if isinstance(wavs, (list, tuple)):
        arr = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    else:
        arr = np.asarray(wavs, dtype=np.float32).reshape(-1)
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
        "model": key,
        "repo_id": repo_id,
        "lang": lang_use,
        "speaker": speaker_use if key.startswith("custom") else None,
        "instruct": instruct_use or None,
        "ref_audio": ref if key == "base_1.7" else None,
        "gen_kwargs": gen_meta,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:500],
        "text_len": len(text),
    }
    meta = {
        "experiment": "027-qwen3-tts",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "text": text,
            "model": key,
            "lang": lang_use,
            "speaker": speaker_use,
            "instruct": instruct_use or None,
            "ref_audio": ref if key == "base_1.7" else None,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Apache-2.0 — Qwen3-TTS",
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
    model: str = DEFAULT_MODEL,
    lang: str = DEFAULT_LANG,
    speaker: str = DEFAULT_SPEAKER,
    instruct: str = "",
    ref_audio: str = "",
    ref_text: str = "",
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        model=model,
        lang=lang,
        speaker=speaker,
        instruct=instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
        run_name=run_name,
        gpu_label=gpu_label,
    )


SMOKE_CUSTOM_ZH = (
    "你好，这是 modal-lab 第零二七号 Qwen3-TTS 实验。今天天气真不错，希望你有一个愉快的周末。"
)
SMOKE_CUSTOM_EN = (
    "Hello from Modal lab experiment zero two seven. "
    "Qwen3 text to speech is running with a custom voice."
)
SMOKE_DESIGN_ZH = "哥哥，你回来啦，人家等了你好久好久了，要抱抱！"
SMOKE_DESIGN_INSTRUCT = (
    "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。"
)
SMOKE_CLONE_EN = (
    "I am solving the equation on the board. Nobody said it would be this hard, "
    "but we are almost there!"
)


@app.local_entrypoint()
def main(
    action: str = "status",
    gpu: str = DEFAULT_GPU,
    model: str = DEFAULT_MODEL,
    text: str = "",
    lang: str = DEFAULT_LANG,
    speaker: str = DEFAULT_SPEAKER,
    instruct: str = "",
    ref_audio: str = "",
    ref_text: str = "",
    run_name: str = "",
    force_download: bool = False,
    smoke_kind: str = "custom_zh",
):
    if action == "status":
        status_fn.remote()
        return
    if action == "download":
        download_weights.remote(force=force_download, model=model)
        return
    if action == "smoke":
        kind = smoke_kind.lower().strip()
        if kind in ("custom_en", "en"):
            model_use = "custom_1.7"
            text_use = SMOKE_CUSTOM_EN
            lang_use = "English"
            speaker_use = "Ryan"
            instruct_use = "Speak cheerfully and clearly."
            run = run_name or "smoke_custom_en_ryan"
            ref_a, ref_t = "", ""
        elif kind in ("design", "design_zh", "voice_design"):
            model_use = "design_1.7"
            text_use = SMOKE_DESIGN_ZH
            lang_use = "Chinese"
            speaker_use = ""
            instruct_use = SMOKE_DESIGN_INSTRUCT
            run = run_name or "smoke_design_zh"
            ref_a, ref_t = "", ""
        elif kind in ("clone", "clone_en", "base"):
            model_use = "base_1.7"
            text_use = SMOKE_CLONE_EN
            lang_use = "English"
            speaker_use = ""
            instruct_use = ""
            run = run_name or "smoke_clone_en"
            ref_a, ref_t = DEFAULT_CLONE_REF_URL, DEFAULT_CLONE_REF_TEXT
        else:
            # custom_zh default
            model_use = "custom_1.7"
            text_use = SMOKE_CUSTOM_ZH
            lang_use = "Chinese"
            speaker_use = "Vivian"
            instruct_use = "用自然、温和的语气说"
            run = run_name or "smoke_custom_zh_vivian"
            ref_a, ref_t = "", ""

        download_weights.remote(force=False, model=model_use)
        out = generate_fn.with_options(gpu=gpu).remote(
            text=text_use,
            model=model_use,
            lang=lang_use,
            speaker=speaker_use,
            instruct=instruct_use,
            ref_audio=ref_a,
            ref_text=ref_t,
            run_name=run,
            gpu_label=gpu,
        )
        print("SMOKE_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        if (out.get("audio") or {}).get("duration_s", 0) < 0.5:
            raise SystemExit("smoke audio too short")
        if (out.get("audio") or {}).get("rms", 0) < 1e-4:
            raise SystemExit("smoke audio near silent")
        return
    if action in ("t2s", "custom", "design", "clone"):
        if not text.strip():
            raise SystemExit("t2s requires --text")
        model_use = model
        if action == "design":
            model_use = "design_1.7"
        elif action == "clone":
            model_use = "base_1.7"
        elif action == "custom":
            model_use = model if _norm_model(model).startswith("custom") else "custom_1.7"
        download_weights.remote(force=False, model=model_use)
        out = generate_fn.with_options(gpu=gpu).remote(
            text=text,
            model=model_use,
            lang=lang,
            speaker=speaker,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            run_name=run_name,
            gpu_label=gpu,
        )
        print("T2S_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        return
    raise SystemExit(f"unknown action {action!r}")
