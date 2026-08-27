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

import argparse
import json
import os
import shutil
import subprocess
import sys
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="010 ACE-Step 1.5 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / outputs Volume")
    sub.add_parser("list-outputs", help="结构化汇总远程 run meta")

    download = sub.add_parser("download", help="CPU 下载 ACE-Step 主包")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="固定 lo-fi instrumental benchmark")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--duration", type=float, default=20.0)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--run-name", default="smoke_lofi")

    t2m = sub.add_parser("t2m", help="Text-to-Music")
    t2m.add_argument("--dry-run", action="store_true")
    t2m.add_argument("--gpu", default=DEFAULT_GPU)
    t2m.add_argument("--example", default="smoke_lofi")
    t2m.add_argument("--caption", default="")
    t2m.add_argument("--lyrics", default="")
    t2m.add_argument("--duration", type=float, default=30.0)
    t2m.add_argument("--bpm", type=int, default=0)
    t2m.add_argument("--seed", type=int, default=42)
    t2m.add_argument("--thinking", action="store_true")
    t2m.add_argument("--init-lm", action="store_true")
    t2m.add_argument("--vocal", action="store_true")
    t2m.add_argument("--steps", type=int, default=8)
    t2m.add_argument("--format", dest="audio_format", choices=["flac", "wav"], default="flac")
    t2m.add_argument("--run-name", default="")
    t2m.add_argument("--dit", default=DEFAULT_DIT)
    t2m.add_argument("--lm", default=DEFAULT_LM)
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "010-ace-step-1.5",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_dit": DEFAULT_DIT,
        "default_lm": DEFAULT_LM,
        "hf_repo": HF_MAIN_REPO,
        "upstream": UPSTREAM,
        "upstream_commit": UPSTREAM_COMMIT,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "smoke": {"duration": 20.0, "thinking": False, "init_lm": False, "instrumental": True, "steps": 8},
    }


def generation_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "smoke":
        duration = args.duration if args.duration > 0 else 20.0
        return {
            "action": "smoke",
            "gpu": args.gpu,
            "run_name": args.run_name or "smoke_lofi",
            "example": "smoke_lofi",
            "caption": "",
            "lyrics": "",
            "duration": duration,
            "bpm": 0,
            "seed": args.seed,
            "thinking": False,
            "init_lm": False,
            "instrumental": True,
            "inference_steps": 8,
            "audio_format": "flac",
            "dit_model": DEFAULT_DIT,
            "lm_model": DEFAULT_LM,
        }
    if args.duration <= 0:
        raise ValueError("--duration 必须 > 0")
    if args.steps <= 0:
        raise ValueError("--steps 必须 > 0")
    if args.bpm < 0:
        raise ValueError("--bpm 必须 >= 0")
    return {
        "action": "t2m",
        "gpu": args.gpu,
        "run_name": args.run_name,
        "example": args.example,
        "caption": args.caption,
        "lyrics": args.lyrics.replace("\\n", "\n"),
        "duration": args.duration,
        "bpm": args.bpm,
        "seed": args.seed,
        "thinking": args.thinking,
        "init_lm": args.init_lm,
        "instrumental": not args.vocal,
        "inference_steps": args.steps,
        "audio_format": args.audio_format,
        "dit_model": args.dit,
        "lm_model": args.lm,
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
    if args.command == "list-outputs":
        print(json.dumps(list_outputs_fn.remote(), ensure_ascii=False, indent=2))
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

    fn = generate_music_fn if plan["gpu"] == DEFAULT_GPU else generate_music_fn.with_options(gpu=plan["gpu"])
    result = fn.remote(
        run_name=plan["run_name"],
        example=plan["example"],
        caption=plan["caption"],
        lyrics=plan["lyrics"],
        duration=plan["duration"],
        bpm=plan["bpm"],
        seed=plan["seed"],
        thinking=plan["thinking"],
        init_lm=plan["init_lm"],
        instrumental=plan["instrumental"],
        inference_steps=plan["inference_steps"],
        audio_format=plan["audio_format"],
        dit_model=plan["dit_model"],
        lm_model=plan["lm_model"],
        gpu_label=plan["gpu"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])
