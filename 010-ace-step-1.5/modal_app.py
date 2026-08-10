# -*- coding: utf-8 -*-
"""
010-ace-step-1.5 — ACE-Step 1.5 音乐生成（Modal）。

默认策略：
  - GPU: L4（24GB · 约 $0.000222/s）— turbo DiT <4GB；主包 LM 1.7B 也够
  - 权重 CPU 下载到 Volume（不计 GPU 费）
  - smoke: 20s 器乐 · thinking=False · 8 steps · seed 42
  - 无 keep_warm

上游: https://github.com/ace-step/ACE-Step-1.5
权重: https://huggingface.co/ACE-Step/Ace-Step1.5
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-ace-step-1.5"
UPSTREAM = "https://github.com/ace-step/ACE-Step-1.5"
UPSTREAM_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
HF_MAIN_REPO = "ACE-Step/Ace-Step1.5"

DEFAULT_GPU = "L4"
DEFAULT_DIT = "acestep-v15-turbo"
DEFAULT_LM = "acestep-5Hz-lm-1.7B"

REPO_DIR = Path("/opt/ACE-Step-1.5")
VENV_PY = REPO_DIR / ".venv" / "bin" / "python"
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
CHECKPOINTS = Path(WEIGHTS_MOUNT) / "checkpoints"
VOLUME_WEIGHTS = "modal-lab-ace-step-1.5-weights"
VOLUME_OUTPUTS = "modal-lab-ace-step-1.5-outputs"

MAIN_COMPONENTS = (
    "acestep-v15-turbo",
    "vae",
    "Qwen3-Embedding-0.6B",
    "acestep-5Hz-lm-1.7B",
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
SMOKE_TIMEOUT = 30 * 60

EXP_DIR = Path(__file__).resolve().parent

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# CUDA 12.8 runtime + uv sync into project .venv (do NOT put venv on PATH —
# that shadows Modal's runtime deps and crash-loops the container).
inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
        "libsndfile1-dev",
        "curl",
        "ca-certificates",
        "build-essential",
        "libgomp1",
        "libffi-dev",
        "libssl-dev",
    )
    .run_commands(
        "pip install -U pip uv",
        f"git clone --depth 1 {UPSTREAM}.git {REPO_DIR}",
        f"cd {REPO_DIR} && (git fetch --depth 1 origin {UPSTREAM_COMMIT} && "
        f"git checkout {UPSTREAM_COMMIT} || git rev-parse HEAD)",
        f"cd {REPO_DIR} && uv sync --frozen --no-dev --python 3.11",
        f"cd {REPO_DIR} && .venv/bin/python -c "
        "\"import torch; print('torch', torch.__version__); "
        "import acestep; print('acestep ok')\"",
    )
    .env(
        {
            # Keep Modal system python as default; worker uses VENV_PY explicitly.
            "ACESTEP_REPO_DIR": str(REPO_DIR),
            "ACESTEP_CHECKPOINTS_DIR": str(CHECKPOINTS),
            "ACESTEP_CONFIG_PATH": DEFAULT_DIT,
            "ACESTEP_LM_MODEL_PATH": DEFAULT_LM,
            "ACESTEP_LM_BACKEND": "pt",
            "ACESTEP_INIT_LLM": "false",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": f"{WEIGHTS_MOUNT}/hf_home",
        }
    )
    .add_local_file(
        str(EXP_DIR / "remote_job.py"),
        remote_path="/root/remote_job.py",
    )
    .add_local_dir(
        str(EXP_DIR / "examples"),
        remote_path="/root/examples",
    )
)

app = modal.App(APP_NAME)


def _dir_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "files": 0, "size_gb": 0.0}
    files = [p for p in path.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {
        "exists": True,
        "path": str(path),
        "files": len(files),
        "size_gb": round(total / 1e9, 3),
    }


def _main_model_ready(checkpoints: Path) -> bool:
    for name in MAIN_COMPONENTS:
        p = checkpoints / name
        if not p.is_dir():
            return False
        weights = [
            f
            for f in p.rglob("*")
            if f.is_file()
            and f.suffix.lower() in {".safetensors", ".bin", ".pt", ".ckpt"}
        ]
        if not weights:
            return False
    return True


def _estimate_cost(gpu: str, seconds: float) -> float | None:
    rate = GPU_PRICE_PER_SEC.get(gpu) or GPU_PRICE_PER_SEC.get(gpu.replace("!", ""))
    if rate is None:
        return None
    return round(rate * seconds, 4)


def _run_remote_job(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload_path = Path("/tmp/ace_payload.json")
    result_path = Path("/tmp/ace_result.json")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()

    if not VENV_PY.is_file():
        return {"success": False, "error": f"venv python missing: {VENV_PY}"}

    env = os.environ.copy()
    env["ACESTEP_CHECKPOINTS_DIR"] = str(CHECKPOINTS)
    env["ACESTEP_REPO_DIR"] = str(REPO_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure child uses pure venv site-packages (don't inherit Modal VIRTUAL_ENV)
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = f"{REPO_DIR / '.venv' / 'bin'}:/usr/local/bin:/usr/bin:/bin"

    cmd = [
        str(VENV_PY),
        "/root/remote_job.py",
        "--action",
        action,
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
    ]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=False)
    if not result_path.is_file():
        return {
            "success": False,
            "error": f"remote_job produced no result (exit={proc.returncode})",
        }
    out = json.loads(result_path.read_text(encoding="utf-8"))
    out["remote_exit_code"] = proc.returncode
    return out


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    """CPU-only HF snapshot → Volume (no GPU cost)."""
    from huggingface_hub import snapshot_download

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if _main_model_ready(CHECKPOINTS) and not force:
        info = _dir_info(CHECKPOINTS)
        info.update({"skipped": True, "repo": HF_MAIN_REPO, "force": force})
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info

    t0 = time.time()
    print(f"Downloading {HF_MAIN_REPO} → {CHECKPOINTS}", flush=True)
    snapshot_download(
        repo_id=HF_MAIN_REPO,
        local_dir=str(CHECKPOINTS),
        local_dir_use_symlinks=False,
    )
    weights_vol.commit()
    info = _dir_info(CHECKPOINTS)
    info.update(
        {
            "skipped": False,
            "repo": HF_MAIN_REPO,
            "elapsed_s": round(time.time() - t0, 1),
            "ready": _main_model_ready(CHECKPOINTS),
            "components": {
                name: _dir_info(CHECKPOINTS / name) for name in MAIN_COMPONENTS
            },
        }
    )
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=10 * 60,
    cpu=1,
    memory=2048,
)
def status_fn() -> dict[str, Any]:
    """CPU-only status (no GPU)."""
    runs_root = Path(OUTPUTS_MOUNT) / "runs"
    recent = []
    if runs_root.is_dir():
        for p in sorted(
            runs_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        )[:10]:
            if p.is_dir():
                recent.append(p.name)
    out: dict[str, Any] = {
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "upstream": UPSTREAM,
        "upstream_commit": UPSTREAM_COMMIT,
        "hf_repo": HF_MAIN_REPO,
        "weights": _dir_info(CHECKPOINTS),
        "ready": _main_model_ready(CHECKPOINTS),
        "components": {
            name: _dir_info(CHECKPOINTS / name) for name in MAIN_COMPONENTS
        },
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


@app.function(
    image=inference_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=INFER_TIMEOUT,
    cpu=4,
    memory=32768,
    gpu=DEFAULT_GPU,
    scaledown_window=30,
)
def generate_music_fn(
    run_name: str = "",
    example: str = "smoke_lofi",
    caption: str = "",
    lyrics: str = "",
    duration: float = 20.0,
    bpm: int = 0,
    seed: int = 42,
    thinking: bool = False,
    init_lm: bool = False,
    instrumental: bool = True,
    inference_steps: int = 8,
    audio_format: str = "flac",
    dit_model: str = DEFAULT_DIT,
    lm_model: str = DEFAULT_LM,
    lm_backend: str = "pt",
    gpu_label: str = DEFAULT_GPU,
) -> dict[str, Any]:
    """Text2Music on Modal GPU; audio → outputs volume only."""
    if not _main_model_ready(CHECKPOINTS):
        return {
            "success": False,
            "error": "main model not on volume — run download first",
            "weights": _dir_info(CHECKPOINTS),
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"{'smoke' if example == 'smoke_lofi' else 't2m'}_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    example_path = None
    cand = Path("/root/examples") / f"{example}.json"
    if cand.is_file():
        example_path = str(cand)

    bpm_val: int | None = bpm if bpm and bpm > 0 else None

    payload: dict[str, Any] = {
        "example_path": example_path,
        "save_dir": str(save_dir),
        "caption": caption,
        "lyrics": lyrics,
        "duration": duration,
        "bpm": bpm_val,
        "seed": seed,
        "thinking": thinking,
        "init_lm": init_lm or thinking,
        "instrumental": instrumental,
        "inference_steps": inference_steps,
        "audio_format": audio_format,
        "dit_model": dit_model,
        "lm_model": lm_model,
        "lm_backend": lm_backend,
        "device": "cuda",
        "batch_size": 1,
    }

    t0 = time.time()
    result = _run_remote_job("generate", payload)
    wall_s = time.time() - t0

    meta = {
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": round(wall_s, 2),
        "est_gpu_usd": _estimate_cost(gpu_label, wall_s),
        "payload": {
            k: payload[k]
            for k in (
                "caption",
                "lyrics",
                "duration",
                "bpm",
                "seed",
                "thinking",
                "init_lm",
                "instrumental",
                "inference_steps",
                "dit_model",
                "lm_model",
                "audio_format",
            )
        },
        "result": result,
        "created_utc": ts,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    (save_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs_vol.commit()

    audio_files = sorted(
        str(p.relative_to(save_dir))
        for p in save_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".flac", ".wav", ".mp3", ".opus", ".aac"}
    )
    summary = {
        "success": bool(result.get("success")),
        "run_name": name,
        "remote_path": f"runs/{name}",
        "audio_files": audio_files,
        "wall_s": round(wall_s, 2),
        "est_gpu_usd": meta["est_gpu_usd"],
        "gpu_requested": gpu_label,
        "generate_s": result.get("generate_s"),
        "dit_init_s": result.get("dit_init_s"),
        "error": result.get("error"),
        "status_message": result.get("status_message"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


@app.function(
    image=download_image,
    volumes={OUTPUTS_MOUNT: outputs_vol},
    timeout=10 * 60,
    cpu=1,
    memory=2048,
)
def list_outputs_fn(prefix: str = "runs") -> dict[str, Any]:
    root = Path(OUTPUTS_MOUNT) / prefix
    if not root.exists():
        return {"exists": False, "path": str(root), "runs": []}
    runs = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        audios = [
            str(f.relative_to(p))
            for f in p.rglob("*")
            if f.is_file()
            and f.suffix.lower() in {".flac", ".wav", ".mp3", ".opus", ".aac"}
        ]
        meta_path = p / "meta.json"
        meta = None
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = None
        runs.append(
            {
                "name": p.name,
                "audios": audios,
                "size_mb": round(
                    sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6,
                    2,
                ),
                "success": (meta or {}).get("result", {}).get("success")
                if meta
                else None,
                "wall_s": (meta or {}).get("wall_s"),
            }
        )
    out = {"exists": True, "path": str(root), "runs": runs}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(
    action: str = "status",
    force_download: bool = False,
    gpu: str = DEFAULT_GPU,
    run_name: str = "",
    example: str = "smoke_lofi",
    caption: str = "",
    lyrics: str = "",
    duration: float = 20.0,
    bpm: int = 0,
    seed: int = 42,
    thinking: bool = False,
    init_lm: bool = False,
    instrumental: bool = True,
    inference_steps: int = 8,
    audio_format: str = "flac",
    dit_model: str = DEFAULT_DIT,
    lm_model: str = DEFAULT_LM,
) -> None:
    """
    modal run modal_app.py --action status|download|smoke|t2m|list-outputs
    """
    action = action.strip().lower()
    if action == "status":
        print(status_fn.remote())
        return
    if action == "download":
        print(download_weights.remote(force=force_download))
        return
    if action == "list-outputs":
        print(list_outputs_fn.remote())
        return
    if action in {"smoke", "t2m", "generate"}:
        if action == "smoke":
            example = example or "smoke_lofi"
            if duration <= 0:
                duration = 20.0
            thinking = False
            init_lm = False
            instrumental = True
            if not run_name:
                run_name = "smoke_lofi"
        fn = generate_music_fn
        if gpu and gpu != DEFAULT_GPU:
            fn = generate_music_fn.with_options(gpu=gpu)
        result = fn.remote(
            run_name=run_name,
            example=example,
            caption=caption,
            lyrics=lyrics,
            duration=duration,
            bpm=bpm,
            seed=seed,
            thinking=thinking,
            init_lm=init_lm,
            instrumental=instrumental,
            inference_steps=inference_steps,
            audio_format=audio_format,
            dit_model=dit_model,
            lm_model=lm_model,
            gpu_label=gpu,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result.get("success"):
            raise SystemExit(1)
        return
    raise SystemExit(
        f"unknown action={action!r}; use status|download|smoke|t2m|list-outputs"
    )
