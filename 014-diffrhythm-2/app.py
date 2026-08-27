# -*- coding: utf-8 -*-
"""
014-diffrhythm-2 — ASLP-lab DiffRhythm 2 on Modal

默认策略（性价比）：
  - 模型: ASLP-lab/DiffRhythm2 + OpenMuQ/MuQ-MuLan-large
  - GPU: L4（24GB · $0.000222/s）— 扩散非 AR，远快于 YuE；L40S 备用
  - smoke: 60s · 16 steps · text style prompt

上游: https://github.com/ASLP-lab/DiffRhythm2
权重: https://huggingface.co/ASLP-lab/DiffRhythm2
许可: Apache 2.0
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

APP_NAME = "modal-lab-diffrhythm-2"
UPSTREAM = "https://github.com/ASLP-lab/DiffRhythm2"
UPSTREAM_COMMIT = "7804f821b797b4f276090e1a9dcd37e97d9915d5"

REPO_ID = "ASLP-lab/DiffRhythm2"
MULAN_ID = "OpenMuQ/MuQ-MuLan-large"
MUQ_AUDIO_ID = "OpenMuQ/MuQ-large-msd-iter"
TEXT_MODEL_ID = "FacebookAI/xlm-roberta-base"

DEFAULT_GPU = "L4"

REPO_DIR = Path("/opt/DiffRhythm2")
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
CKPT_DIR = Path(WEIGHTS_MOUNT) / "ckpt"
MULAN_DIR = Path(WEIGHTS_MOUNT) / "mulan"
VOLUME_WEIGHTS = "modal-lab-diffrhythm-2-weights"
VOLUME_OUTPUTS = "modal-lab-diffrhythm-2-outputs"

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
hf_secret = modal.Secret.from_name("huggingface")

_ENV = {
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": str(HF_HOME),
    "HF_HUB_CACHE": str(HF_HOME / "hub"),
    "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
    "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_HUB_DISABLE_XET": "1",
    "USER": "root",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PHONEMIZER_ESPEAK_LIBRARY": "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
}

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub>=0.26.0,<0.30")
    .env(_ENV)
)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "espeak-ng",
        "libespeak-ng1",
        "libsndfile1",
        "build-essential",
        "cmake",
        "g++",
        "curl",
        "ca-certificates",
        "libgomp1",
    )
    .run_commands(
        "pip install -U pip setuptools wheel",
        f"git clone {UPSTREAM}.git {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {UPSTREAM_COMMIT}",
        "pip install torch==2.6.0 torchaudio==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu124",
        "pip install "
        "torchdiffeq==0.2.5 transformers==4.47.1 accelerate "
        "cn2an==0.5.23 einops==0.8.1 'huggingface-hub>=0.26.0,<0.30' "
        "jieba==0.42.1 Jinja2 librosa==0.9.2 muq==0.1.0 "
        "numpy==1.26.4 onnx==1.17.0 onnxruntime "
        "pykakasi==2.3.0 pypinyin==0.54.0 PyYAML safetensors "
        "scipy pedalboard unidecode phonemizer py3langid "
        "tokenizers inflect tqdm soundfile",
    )
    .env(_ENV)
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


def _hub_has(repo_id: str) -> bool:
    hub = HF_HOME / "hub"
    if not hub.is_dir():
        return False
    slug = "models--" + repo_id.replace("/", "--")
    # also accept short names linked as xlm-roberta-base
    if any(hub.glob(f"{slug}/**/config.json")):
        return True
    # FacebookAI/xlm-roberta-base vs xlm-roberta-base
    alt = "models--" + repo_id.split("/")[-1]
    return any(hub.glob(f"{alt}/**/config.json")) or any(
        hub.glob("models--FacebookAI--xlm-roberta-base/**/config.json")
    )


def _weights_ready() -> bool:
    need = [
        CKPT_DIR / "model.safetensors",
        CKPT_DIR / "config.json",
        CKPT_DIR / "decoder.bin",
        CKPT_DIR / "decoder.json",
    ]
    if not all(p.is_file() for p in need):
        return False
    if not (MULAN_DIR / "config.json").is_file() and not _hub_has(MULAN_ID):
        return False
    if not _hub_has(MUQ_AUDIO_ID):
        return False
    if not (_hub_has(TEXT_MODEL_ID) or _hub_has("xlm-roberta-base")):
        return False
    return True


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    from huggingface_hub import snapshot_download, hf_hub_download

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    MULAN_DIR.mkdir(parents=True, exist_ok=True)
    hub_cache = str(HF_HOME / "hub")

    if _weights_ready() and not force:
        info = {
            "skipped": True,
            "ready": True,
            "ckpt": _dir_info(CKPT_DIR),
            "mulan": _dir_info(MULAN_DIR),
            "hf_home": _dir_info(HF_HOME),
        }
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return info

    t0 = time.time()
    results = {}

    print(f"Downloading {REPO_ID} → {CKPT_DIR}", flush=True)
    t1 = time.time()
    for fname in ("model.safetensors", "config.json", "decoder.bin", "decoder.json", "model.json"):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                local_dir=str(CKPT_DIR),
                token=token,
            )
        except Exception as e:
            print(f"skip {fname}: {e}", flush=True)
    results["diffrhythm2"] = {
        "elapsed_s": round(time.time() - t1, 1),
        **_dir_info(CKPT_DIR),
    }
    weights_vol.commit()

    for label, repo, local in (
        ("mulan", MULAN_ID, str(MULAN_DIR)),
        ("muq_audio", MUQ_AUDIO_ID, None),
        ("xlm_roberta", TEXT_MODEL_ID, None),
    ):
        print(f"Downloading {repo}", flush=True)
        t = time.time()
        try:
            if local:
                snapshot_download(repo_id=repo, local_dir=local, token=token)
            snapshot_download(repo_id=repo, cache_dir=hub_cache, token=token)
            err = None
        except Exception as e:
            err = str(e)
            # fallback alias for text model
            if label == "xlm_roberta":
                try:
                    snapshot_download(
                        repo_id="xlm-roberta-base",
                        cache_dir=hub_cache,
                        token=token,
                        endpoint="https://huggingface.co",
                    )
                    err = None
                except Exception as e2:
                    err = f"{e} | fallback: {e2}"
        results[label] = {
            "repo": repo,
            "elapsed_s": round(time.time() - t, 1),
            "hub_ok": _hub_has(repo) or (label == "xlm_roberta" and _hub_has("xlm-roberta-base")),
            "error": err,
        }
        if local:
            results[label]["local"] = _dir_info(Path(local))
        weights_vol.commit()

    info = {
        "skipped": False,
        "ready": _weights_ready(),
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
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
    runs = Path(OUTPUTS_MOUNT) / "runs"
    recent = []
    if runs.is_dir():
        for p in sorted(runs.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            if p.is_dir():
                recent.append(p.name)
    out = {
        "app": APP_NAME,
        "slot": "014-diffrhythm-2",
        "default_gpu": DEFAULT_GPU,
        "repo": REPO_ID,
        "ready": _weights_ready(),
        "ckpt": _dir_info(CKPT_DIR),
        "mulan": _dir_info(MULAN_DIR),
        "deps": {
            MUQ_AUDIO_ID: _hub_has(MUQ_AUDIO_ID),
            TEXT_MODEL_ID: _hub_has(TEXT_MODEL_ID) or _hub_has("xlm-roberta-base"),
        },
        "recent_runs": recent,
        "license": "Apache 2.0",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


SMOKE_STYLE = "Pop, Piano, Bass, Drums, Happy, uplifting female vocal"
SMOKE_LYRICS = """[start]
[intro]
[verse]
Staring at the sunset colors paint the sky
Thoughts of you keep swirling I cannot deny
I know I let you down I made mistakes
But I am here to mend the heart I did not break
[chorus]
Every road you take I will be one step behind
Every dream you chase I am reaching for the light
You cannot fight this feeling now
I will not back down
[end]
"""


def _ensure_pyopenjtalk_stub() -> None:
    try:
        import pyopenjtalk  # noqa: F401
        return
    except Exception:
        pass
    import types
    import sys

    m = types.ModuleType("pyopenjtalk")
    m.run_frontend = lambda *_a, **_k: []
    m.estimate_accent = lambda x: x
    m.make_label = lambda _x: []
    sys.modules["pyopenjtalk"] = m


def _run_generate(
    *,
    lyrics: str,
    style_prompt: str,
    run_name: str,
    gpu_label: str,
    max_secs: float,
    steps: int,
    cfg_strength: float,
    seed: int,
) -> dict[str, Any]:
    if not _weights_ready():
        return {"success": False, "error": "weights not ready — run download"}

    import random
    import sys

    import numpy as np
    import torch
    import torchaudio
    from safetensors.torch import load_file

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "hub")

    _ensure_pyopenjtalk_stub()

    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)

    from muq import MuQMuLan
    from diffrhythm2.cfm import CFM
    from diffrhythm2.backbones.dit import DiT
    from bigvgan.model import Generator
    import inference as inference_mod
    from inference import CNENTokenizer, parse_lyrics, inference as dr_inference

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"dr2_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    work = Path("/tmp/dr2_work") / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_ckpt = REPO_DIR / "ckpt"
    if repo_ckpt.is_symlink():
        repo_ckpt.unlink()
    elif repo_ckpt.is_dir():
        shutil.rmtree(repo_ckpt)
    repo_ckpt.symlink_to(CKPT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None and seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    t_load = time.time()

    with open(CKPT_DIR / "config.json") as f:
        model_config = json.load(f)
    model_config["use_flex_attn"] = False
    model = CFM(
        transformer=DiT(**model_config),
        num_channels=model_config["mel_dim"],
        block_size=model_config["block_size"],
    )
    ckpt = load_file(str(CKPT_DIR / "model.safetensors"))
    model.load_state_dict(ckpt)
    model = model.to(device)

    mulan_src = MULAN_ID
    # Prefer hub cache (handles pytorch_model.bin); local_dir may miss safetensors
    mulan = MuQMuLan.from_pretrained(
        mulan_src,
        cache_dir=str(HF_HOME / "hub"),
    ).to(device)

    lrc_tokenizer = CNENTokenizer()
    inference_mod.lrc_tokenizer = lrc_tokenizer

    decoder = Generator(str(CKPT_DIR / "decoder.json"), str(CKPT_DIR / "decoder.bin"))
    decoder = decoder.to(device)
    load_s = time.time() - t_load

    lyrics_token = parse_lyrics(lyrics.strip() + "\n")
    lyrics_token = torch.tensor(sum(lyrics_token, []), dtype=torch.long, device=device)

    style = style_prompt.strip()
    if os.path.isfile(style):
        prompt_wav, sr = torchaudio.load(style)
        prompt_wav = torchaudio.functional.resample(prompt_wav.to(device), sr, 24000)
        if prompt_wav.shape[1] > 24000 * 10:
            start = random.randint(0, prompt_wav.shape[1] - 24000 * 10)
            prompt_wav = prompt_wav[:, start : start + 24000 * 10]
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
        with torch.no_grad():
            style_embed = mulan(wavs=prompt_wav)
    else:
        with torch.no_grad():
            style_embed = mulan(texts=[style])
    style_embed = style_embed.to(device).squeeze(0)

    if device.type != "cpu":
        model = model.half()
        decoder = decoder.half()
        style_embed = style_embed.half()

    t_gen = time.time()
    dr_inference(
        model=model,
        decoder=decoder,
        text=lyrics_token,
        style_prompt=style_embed,
        duration=float(max_secs),
        output_dir=str(out_dir),
        song_name="song",
        sample_steps=int(steps),
        cfg_strength=float(cfg_strength),
        fake_stereo=True,
        process_bar=True,
    )
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    mp3s = list(out_dir.glob("*.mp3"))
    primary = None
    if mp3s:
        dest = save_dir / "song.mp3"
        shutil.copy2(mp3s[0], dest)
        primary = {
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "name": dest.name,
        }
        shutil.copytree(out_dir, save_dir / "raw", dirs_exist_ok=True)

    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    ok = primary is not None
    result = {
        "success": ok,
        "error": None if ok else "no mp3 output",
        "wall_s": round(wall, 2),
        "load_s": round(load_s, 2),
        "generate_s": round(gen_s, 2),
        "est_gpu_usd": _estimate_cost(gpu_label, wall),
        "gpu": gpu_label,
        "model": REPO_ID,
        "max_secs": max_secs,
        "steps": steps,
        "cfg_strength": cfg_strength,
        "seed": seed,
        "vram_peak_gb": vram_gb,
        "primary_audio": primary,
        "style_prompt": style if not os.path.isfile(style) else f"wav:{style}",
        "lyrics_preview": lyrics[:200],
    }
    meta = {
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "lyrics": lyrics,
            "style_prompt": style_prompt,
            "max_secs": max_secs,
            "steps": steps,
            "cfg_strength": cfg_strength,
            "seed": seed,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Apache 2.0 — DiffRhythm 2 by ASLP-lab",
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
    memory=24576,
)
def generate_fn(
    lyrics: str = SMOKE_LYRICS,
    style_prompt: str = SMOKE_STYLE,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    max_secs: float = 60.0,
    steps: int = 16,
    cfg_strength: float = 2.0,
    seed: int = 42,
) -> dict[str, Any]:
    return _run_generate(
        lyrics=lyrics,
        style_prompt=style_prompt,
        run_name=run_name,
        gpu_label=gpu_label,
        max_secs=max_secs,
        steps=steps,
        cfg_strength=cfg_strength,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="014 DiffRhythm 2 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs Volume")

    download = sub.add_parser("download", help="下载 DiffRhythm2 + MuQ 权重")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="固定 English 60s benchmark")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--style", default=SMOKE_STYLE)
    smoke.add_argument("--max-secs", type=float, default=60.0)
    smoke.add_argument("--steps", type=int, default=16)
    smoke.add_argument("--cfg-strength", type=float, default=2.0)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--run-name", default="smoke_en60")

    generate = sub.add_parser("generate", help="lyrics + style -> full song")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--gpu", default=DEFAULT_GPU)
    lyrics = generate.add_mutually_exclusive_group(required=True)
    lyrics.add_argument("--lyrics-file", type=Path)
    lyrics.add_argument("--lyrics")
    generate.add_argument("--style", required=True)
    generate.add_argument("--max-secs", type=float, default=120.0)
    generate.add_argument("--steps", type=int, default=16)
    generate.add_argument("--cfg-strength", type=float, default=2.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--run-name", default="")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "014-diffrhythm-2",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "model": REPO_ID,
        "license": "Apache-2.0",
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "smoke": {"max_secs": 60.0, "steps": 16, "cfg_strength": 2.0},
    }


def _lyrics_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "lyrics_file", None) is not None:
        try:
            return args.lyrics_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"无法读取 lyrics file: {args.lyrics_file}: {exc}") from None
    return (getattr(args, "lyrics", "") or "").replace("\\n", "\n").strip()


def generation_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "smoke":
        lyrics = SMOKE_LYRICS
        style = args.style
    else:
        lyrics = _lyrics_from_args(args)
        style = args.style.strip()
    if not lyrics:
        raise ValueError("lyrics 不能为空")
    if not style:
        raise ValueError("style 不能为空")
    return {
        "action": args.command,
        "gpu": args.gpu,
        "lyrics": lyrics,
        "style_prompt": style,
        "run_name": args.run_name,
        "max_secs": args.max_secs,
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
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
        plan = {"action": "download", "force": args.force}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(download_weights.remote(force=args.force), ensure_ascii=False, indent=2))
        return

    try:
        plan = generation_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    download_weights.remote(force=False)
    out = generate_fn.with_options(gpu=args.gpu).remote(
        lyrics=plan["lyrics"],
        style_prompt=plan["style_prompt"],
        run_name=plan["run_name"],
        gpu_label=args.gpu,
        max_secs=plan["max_secs"],
        steps=plan["steps"],
        cfg_strength=plan["cfg_strength"],
        seed=plan["seed"],
    )
    print("RESULT", json.dumps(out, ensure_ascii=False), flush=True)
    if not out.get("success"):
        raise SystemExit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
