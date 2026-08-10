# -*- coding: utf-8 -*-
"""
013-yue — M-A-P YuE full-song lyrics→music on Modal

默认策略（性价比）：
  - Stage1: m-a-p/YuE-s1-7B-anneal-en-cot（~12.5GB）
  - Stage2: m-a-p/YuE-s2-1B-general（~4GB）
  - Codec:  m-a-p/xcodec_mini_infer（~2GB）
  - GPU: L40S（48GB · $0.000542/s）— 够 2-segment smoke；全曲可加 PRO 6000 / A100-80
  - smoke: 2 segments · max_new_tokens 3000 · stage2_batch_size 2

上游: https://github.com/multimodal-art-projection/YuE
许可: Apache 2.0
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

APP_NAME = "modal-lab-yue"
UPSTREAM = "https://github.com/multimodal-art-projection/YuE"
UPSTREAM_COMMIT = "9f1394bae1d8d218fea750c1413c2d9d731c7310"

STAGE1_REPO = "m-a-p/YuE-s1-7B-anneal-en-cot"
STAGE2_REPO = "m-a-p/YuE-s2-1B-general"
XCODEC_REPO = "m-a-p/xcodec_mini_infer"

DEFAULT_GPU = "L40S"
DEFAULT_STAGE1 = "en-cot"

STAGE1_VARIANTS = {
    "en-cot": "m-a-p/YuE-s1-7B-anneal-en-cot",
    "en-icl": "m-a-p/YuE-s1-7B-anneal-en-icl",
    "zh-cot": "m-a-p/YuE-s1-7B-anneal-zh-cot",
}

REPO_DIR = Path("/opt/YuE")
INFER_DIR = REPO_DIR / "inference"
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
MODELS_DIR = Path(WEIGHTS_MOUNT) / "models"
XCODEC_DIR = Path(WEIGHTS_MOUNT) / "xcodec_mini_infer"
VOLUME_WEIGHTS = "modal-lab-yue-weights"
VOLUME_OUTPUTS = "modal-lab-yue-outputs"

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
INFER_TIMEOUT = 90 * 60
SMOKE_TIMEOUT = 60 * 60

# torch 2.5.1 + cu124 + cp310 prebuilt flash-attn
FLASH_ATTN_WHL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

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
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "USER": "root",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(_ENV)
)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
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
        "ninja-build",
        "libgomp1",
    )
    .run_commands(
        "pip install -U pip setuptools wheel",
        f"git clone {UPSTREAM}.git {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {UPSTREAM_COMMIT}",
        "pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cu124",
        # core deps (explicit versions for reproducibility)
        "pip install "
        "transformers==4.46.3 accelerate==1.1.1 sentencepiece protobuf "
        "omegaconf einops numpy scipy tqdm soundfile "
        "descript-audiotools>=0.7.2 descript-audio-codec "
        "tensorboard huggingface_hub>=0.26.0",
        # flash-attn required by YuE attn_implementation=
        f"pip install {FLASH_ATTN_WHL} || "
        f"pip install flash-attn==2.7.0.post2 --no-build-isolation || true",
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


def _stage1_id(name: str) -> str:
    n = (name or DEFAULT_STAGE1).strip().lower()
    if n in STAGE1_VARIANTS:
        return STAGE1_VARIANTS[n]
    if n.startswith("m-a-p/"):
        return n
    if n in STAGE1_VARIANTS.values():
        return n
    raise ValueError(f"unknown stage1 {name!r}; use {list(STAGE1_VARIANTS)}")


def _local_model_path(repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    return MODELS_DIR / safe


def _weights_ready(stage1: str = DEFAULT_STAGE1) -> bool:
    s1 = _local_model_path(_stage1_id(stage1))
    s2 = _local_model_path(STAGE2_REPO)
    xc = XCODEC_DIR
    return (
        (s1 / "config.json").is_file()
        and (s2 / "config.json").is_file()
        and (xc / "final_ckpt" / "ckpt_00360000.pth").is_file()
    )


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=16384,
)
def download_weights(force: bool = False, stage1: str = DEFAULT_STAGE1) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    s1_repo = _stage1_id(stage1)
    targets = [
        ("stage1", s1_repo, _local_model_path(s1_repo)),
        ("stage2", STAGE2_REPO, _local_model_path(STAGE2_REPO)),
        ("xcodec", XCODEC_REPO, XCODEC_DIR),
    ]

    if _weights_ready(stage1) and not force:
        info = {
            "skipped": True,
            "ready": True,
            "stage1": s1_repo,
            "paths": {k: _dir_info(p) for k, _r, p in targets},
        }
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return info

    t0 = time.time()
    results = {}
    for key, repo, dest in targets:
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / "config.json"
        if key == "xcodec":
            marker = dest / "final_ckpt" / "ckpt_00360000.pth"
        if marker.is_file() and not force:
            results[key] = {"skipped": True, **_dir_info(dest)}
            continue
        print(f"Downloading {repo} → {dest}", flush=True)
        t1 = time.time()
        snapshot_download(
            repo_id=repo,
            local_dir=str(dest),
            token=token,
            max_workers=8,
        )
        results[key] = {
            "skipped": False,
            "repo": repo,
            "elapsed_s": round(time.time() - t1, 1),
            **_dir_info(dest),
        }
        weights_vol.commit()

    info = {
        "skipped": False,
        "ready": _weights_ready(stage1),
        "stage1": s1_repo,
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
        "slot": "013-yue",
        "default_gpu": DEFAULT_GPU,
        "stage1": STAGE1_REPO,
        "stage2": STAGE2_REPO,
        "ready": _weights_ready(),
        "models": {
            "stage1": _dir_info(_local_model_path(STAGE1_REPO)),
            "stage2": _dir_info(_local_model_path(STAGE2_REPO)),
            "xcodec": _dir_info(XCODEC_DIR),
        },
        "recent_runs": recent,
        "license": "Apache 2.0",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


SMOKE_GENRE = "inspiring female uplifting pop airy vocal electronic bright vocal vocal"
SMOKE_LYRICS = """[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny
I know I let you down, I made mistakes
But I'm here to mend the heart I didn't break

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
"""


def _prepare_xcodec_link() -> None:
    """Point inference/xcodec_mini_infer at volume weights."""
    target = INFER_DIR / "xcodec_mini_infer"
    if target.is_symlink() or target.exists():
        if target.is_symlink() or target.is_dir():
            # replace empty dir from clone
            if target.is_dir() and not target.is_symlink():
                # only remove if empty-ish / missing checkpoint
                ckpt = target / "final_ckpt" / "ckpt_00360000.pth"
                if not ckpt.is_file():
                    shutil.rmtree(target)
                else:
                    return
            elif target.is_symlink():
                target.unlink()
    if not XCODEC_DIR.is_dir():
        raise FileNotFoundError(f"xcodec missing at {XCODEC_DIR}")
    target.symlink_to(XCODEC_DIR)


def _find_audio_outputs(run_dir: Path) -> list[dict[str, Any]]:
    found = []
    for pattern in (
        "**/vocoder/mix/*",
        "**/recons/mix/*",
        "**/*.mp3",
        "**/*.wav",
        "**/*.flac",
    ):
        for p in run_dir.glob(pattern):
            if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".flac"}:
                found.append(
                    {
                        "path": str(p),
                        "name": p.name,
                        "size_bytes": p.stat().st_size,
                        "rel": str(p.relative_to(run_dir)),
                    }
                )
    # dedupe by path
    seen = set()
    uniq = []
    for f in found:
        if f["path"] not in seen:
            seen.add(f["path"])
            uniq.append(f)
    return uniq


def _run_infer(
    *,
    genre: str,
    lyrics: str,
    run_name: str,
    gpu_label: str,
    stage1: str,
    run_n_segments: int,
    max_new_tokens: int,
    stage2_batch_size: int,
    seed: int,
    repetition_penalty: float,
) -> dict[str, Any]:
    if not _weights_ready(stage1):
        return {"success": False, "error": "weights not ready — run download"}

    s1_path = _local_model_path(_stage1_id(stage1))
    s2_path = _local_model_path(STAGE2_REPO)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"yue_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    work = Path("/tmp/yue_work") / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    genre_path = work / "genre.txt"
    lyrics_path = work / "lyrics.txt"
    genre_path.write_text(genre.strip() + "\n", encoding="utf-8")
    lyrics_path.write_text(lyrics.strip() + "\n", encoding="utf-8")
    out_dir = work / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    _prepare_xcodec_link()

    # Prefer local snapshot paths (offline-friendly)
    cmd = [
        "python",
        "infer.py",
        "--cuda_idx",
        "0",
        "--stage1_model",
        str(s1_path),
        "--stage2_model",
        str(s2_path),
        "--genre_txt",
        str(genre_path),
        "--lyrics_txt",
        str(lyrics_path),
        "--run_n_segments",
        str(int(run_n_segments)),
        "--stage2_batch_size",
        str(int(stage2_batch_size)),
        "--output_dir",
        str(out_dir),
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--repetition_penalty",
        str(float(repetition_penalty)),
        "--seed",
        str(int(seed)),
    ]

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HOME / "hub"),
            "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    t0 = time.time()
    print("CMD:", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(INFER_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=INFER_TIMEOUT - 120,
    )
    wall = time.time() - t0

    log_path = save_dir / "infer.log"
    log_path.write_text(
        f"CMD: {' '.join(cmd)}\n\n=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}\n",
        encoding="utf-8",
    )

    # copy outputs
    if out_dir.exists():
        shutil.copytree(out_dir, save_dir / "output", dirs_exist_ok=True)

    audios = _find_audio_outputs(save_dir)
    # promote best mix to top-level
    primary = None
    for a in audios:
        if "vocoder/mix" in a["rel"] or "vocoder" in a["rel"]:
            primary = a
            break
    if primary is None and audios:
        primary = audios[0]
    if primary:
        dest = save_dir / f"mix{Path(primary['path']).suffix}"
        shutil.copy2(primary["path"], dest)
        primary = {**primary, "promoted": str(dest)}

    vram_gb = None
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass

    ok = proc.returncode == 0 and primary is not None
    result = {
        "success": ok,
        "error": None if ok else (f"exit={proc.returncode}; no audio" if proc.returncode == 0 else f"exit={proc.returncode}"),
        "returncode": proc.returncode,
        "wall_s": round(wall, 2),
        "est_gpu_usd": _estimate_cost(gpu_label, wall),
        "gpu": gpu_label,
        "stage1": _stage1_id(stage1),
        "stage2": STAGE2_REPO,
        "run_n_segments": run_n_segments,
        "max_new_tokens": max_new_tokens,
        "stage2_batch_size": stage2_batch_size,
        "seed": seed,
        "vram_peak_gb": vram_gb,
        "primary_audio": primary,
        "audios": audios[:20],
        "stdout_tail": (proc.stdout or "")[-3000:],
        "stderr_tail": (proc.stderr or "")[-3000:],
    }
    meta = {
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "genre": genre,
            "lyrics": lyrics,
            "stage1": stage1,
            "run_n_segments": run_n_segments,
            "max_new_tokens": max_new_tokens,
            "stage2_batch_size": stage2_batch_size,
            "seed": seed,
            "repetition_penalty": repetition_penalty,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Apache 2.0 — YuE by HKUST/M-A-P",
    }
    (save_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs_vol.commit()
    print(json.dumps({k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")}, ensure_ascii=False, indent=2), flush=True)
    if not ok:
        print("STDERR_TAIL:\n", result["stderr_tail"], flush=True)
        print("STDOUT_TAIL:\n", result["stdout_tail"], flush=True)
    return result


@app.function(
    image=inference_image,
    gpu=DEFAULT_GPU,
    timeout=INFER_TIMEOUT,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    secrets=[hf_secret],
    memory=32768,
)
def generate_fn(
    genre: str = SMOKE_GENRE,
    lyrics: str = SMOKE_LYRICS,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    stage1: str = DEFAULT_STAGE1,
    run_n_segments: int = 2,
    max_new_tokens: int = 3000,
    stage2_batch_size: int = 2,
    seed: int = 42,
    repetition_penalty: float = 1.1,
) -> dict[str, Any]:
    return _run_infer(
        genre=genre,
        lyrics=lyrics,
        run_name=run_name,
        gpu_label=gpu_label,
        stage1=stage1,
        run_n_segments=run_n_segments,
        max_new_tokens=max_new_tokens,
        stage2_batch_size=stage2_batch_size,
        seed=seed,
        repetition_penalty=repetition_penalty,
    )


@app.local_entrypoint()
def main(
    action: str = "status",
    gpu: str = DEFAULT_GPU,
    stage1: str = DEFAULT_STAGE1,
    genre: str = "",
    lyrics: str = "",
    run_name: str = "",
    run_n_segments: int = 2,
    max_new_tokens: int = 3000,
    stage2_batch_size: int = 2,
    seed: int = 42,
    force_download: bool = False,
    repetition_penalty: float = 1.1,
):
    if action == "status":
        status_fn.remote()
        return
    if action == "download":
        download_weights.remote(force=force_download, stage1=stage1)
        return
    if action in ("smoke", "generate"):
        download_weights.remote(force=False, stage1=stage1)
        g = genre.strip() or SMOKE_GENRE
        ly = lyrics.strip() or SMOKE_LYRICS
        # allow \n escapes from CLI
        ly = ly.replace("\\n", "\n")
        rn = run_name or ("smoke_en" if action == "smoke" else "")
        out = generate_fn.with_options(gpu=gpu).remote(
            genre=g,
            lyrics=ly,
            run_name=rn,
            gpu_label=gpu,
            stage1=stage1,
            run_n_segments=int(run_n_segments),
            max_new_tokens=int(max_new_tokens),
            stage2_batch_size=int(stage2_batch_size),
            seed=int(seed),
            repetition_penalty=float(repetition_penalty),
        )
        print("RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        return
    raise SystemExit(f"unknown action {action}")
