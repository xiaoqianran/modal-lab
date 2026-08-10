# -*- coding: utf-8 -*-
"""
012-levo-2 — LeVo 2 / SongGeneration v2 on Modal

默认策略（性价比）：
  - 模型: SongGeneration-v2-medium（LeVo 2 线 · 12G/18G）
  - GPU: L40S（48GB · $0.000542/s）— 比 A100-40 便宜且够 large；PRO 6000 更贵
  - 可选: v2-large（22G/28G）同一卡；--low-mem 降显存
  - 权重 CPU 下载到 Volume
  - smoke: 短结构英文歌词 + text description

代码: https://github.com/6Morpheus6/songgeneration-tencent （含 version=v2）
权重: lglg666/SongGeneration-Runtime + SongGeneration-v2-{medium,large}
许可: Tencent SongGeneration — 仅学术/研究/教育，禁止商用
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

APP_NAME = "modal-lab-levo-2"
UPSTREAM = "https://github.com/6Morpheus6/songgeneration-tencent"
UPSTREAM_COMMIT = "b1b03ec93a38ae4a566ad8a22010f51dd49cb803"

HF_RUNTIME = "lglg666/SongGeneration-Runtime"
HF_MODELS = {
    "v2-medium": "lglg666/SongGeneration-v2-medium",
    "v2-large": "lglg666/SongGeneration-v2-large",
    "medium": "lglg666/SongGeneration-v2-medium",
    "large": "lglg666/SongGeneration-v2-large",
}

DEFAULT_GPU = "L40S"
DEFAULT_MODEL = "v2-medium"

REPO_DIR = Path("/opt/LeVo")
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
RUNTIME_DIR = Path(WEIGHTS_MOUNT) / "runtime"
MODELS_DIR = Path(WEIGHTS_MOUNT) / "models"
VOLUME_WEIGHTS = "modal-lab-levo-2-weights"
VOLUME_OUTPUTS = "modal-lab-levo-2-outputs"

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
SMOKE_TIMEOUT = 45 * 60

FLASH_ATTN_WHL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

app = modal.App(APP_NAME)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

_ENV = {
    "PYTHONUNBUFFERED": "1",
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
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
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
        "libgomp1",
    )
    .run_commands(
        "pip install -U pip",
        f"git clone {UPSTREAM}.git {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {UPSTREAM_COMMIT}",
        "pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 "
        "--index-url https://download.pytorch.org/whl/cu124",
        f"cd {REPO_DIR} && pip install -r requirements.txt --no-deps || true",
        f"cd {REPO_DIR} && pip install "
        "alias-free-torch>=0.0.6 descript-audio-codec>=1.0.0 "
        "diffusers==0.27.2 einops>=0.8.1 einops-exts==0.0.4 flashy>=0.0.2 "
        "huggingface-hub==0.25.2 julius>=0.2.7 k-diffusion==0.1.1 "
        "kaldiio>=2.18.1 lameenc>=1.8.1 librosa>=0.11.0 lightning>=2.5.2 "
        "ninja>=1.11.1.4 nnAudio>=0.3.3 openunmix>=1.3.0 peft==0.10.0 "
        "transformers==4.37.2 vector-quantize-pytorch>=1.22.17 "
        "wheel>=0.45.1 x-transformers>=2.3.25",
        f"cd {REPO_DIR} && pip install -r requirements_nodeps.txt --no-deps",
        f"pip install --no-deps '{FLASH_ATTN_WHL}' || echo 'flash_attn wheel optional'",
        "python -c \"import torch; print('torch', torch.__version__, torch.version.cuda); "
        "import transformers; print('transformers', transformers.__version__)\"",
    )
    .env({**_ENV, "LEVO_REPO": str(REPO_DIR)})
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
    if n in ("medium", "v2-medium", "v2_medium", "songgeneration-v2-medium"):
        return "v2-medium"
    if n in ("large", "v2-large", "v2_large", "songgeneration-v2-large"):
        return "v2-large"
    raise ValueError(f"unknown model {name!r}; use v2-medium | v2-large")


def _model_dir(key: str) -> Path:
    k = key.replace("-", "_")
    if k in ("v2_medium", "medium"):
        return MODELS_DIR / "songgeneration_v2_medium"
    if k in ("v2_large", "large"):
        return MODELS_DIR / "songgeneration_v2_large"
    return MODELS_DIR / k


def _runtime_ready() -> bool:
    return (RUNTIME_DIR / "ckpt").is_dir() and (RUNTIME_DIR / "third_party").is_dir()


def _model_ready(key: str) -> bool:
    d = _model_dir(key)
    return (d / "model.pt").is_file() and (d / "config.yaml").is_file()


def _link_runtime_into_repo() -> None:
    for name in ("ckpt", "third_party"):
        src = RUNTIME_DIR / name
        dst = REPO_DIR / name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        dst.symlink_to(src, target_is_directory=True)


def _patch_repo_for_runtime() -> None:
    """Mirror ships LFS pointer for auto-prompt; disable load (we use text descriptions)."""
    gen = REPO_DIR / "generate.py"
    text = gen.read_text(encoding="utf-8")
    text2 = text.replace(
        "auto_prompt = torch.load('tools/new_auto_prompt.pt')",
        "auto_prompt = {}  # LFS unavailable in mirror; descriptions-only OK",
    ).replace(
        "auto_prompt = torch.load('tools/new_prompt.pt')",
        "auto_prompt = {}",
    )
    if text2 != text:
        gen.write_text(text2, encoding="utf-8")
        print("patched generate.py auto_prompt load", flush=True)


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(
    force: bool = False,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    key = _norm_model(model)
    repo_model = HF_MODELS[key]
    t0 = time.time()
    results: dict[str, Any] = {"model_key": key, "repos": {}}

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if force or not _runtime_ready():
        print(f"Downloading {HF_RUNTIME} → {RUNTIME_DIR}", flush=True)
        snapshot_download(
            repo_id=HF_RUNTIME,
            local_dir=str(RUNTIME_DIR),
            token=token,
        )
        results["repos"][HF_RUNTIME] = _dir_info(RUNTIME_DIR)
    else:
        results["repos"][HF_RUNTIME] = {"skipped": True, **_dir_info(RUNTIME_DIR)}

    mdir = _model_dir(key)
    if force or not _model_ready(key):
        print(f"Downloading {repo_model} → {mdir}", flush=True)
        mdir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_model,
            local_dir=str(mdir),
            token=token,
        )
        results["repos"][repo_model] = _dir_info(mdir)
    else:
        results["repos"][repo_model] = {"skipped": True, **_dir_info(mdir)}

    weights_vol.commit()
    results["elapsed_s"] = round(time.time() - t0, 1)
    results["runtime_ready"] = _runtime_ready()
    results["model_ready"] = _model_ready(key)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


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
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "upstream": UPSTREAM,
        "upstream_commit": UPSTREAM_COMMIT,
        "license": "Tencent SongGeneration — research/academic only, NO commercial use",
        "runtime_ready": _runtime_ready(),
        "models": {
            k: {"ready": _model_ready(k), **_dir_info(_model_dir(k))}
            for k in ("v2-medium", "v2-large")
        },
        "runtime": _dir_info(RUNTIME_DIR),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "gpu_note": (
            "Default L40S ($0.000542/s): cheaper than A100-40, 48GB fits v2-large. "
            "PRO-6000 ~1.55× L40S price. Avoid A100 for cost."
        ),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _default_smoke_item() -> dict[str, Any]:
    return {
        "idx": "smoke_en",
        "descriptions": "female, indie pop, warm, acoustic guitar and soft drums, hopeful, bpm 100",
        "gt_lyric": (
            "[intro-short] ; "
            "[verse] City lights are humming low. We walk where quiet rivers flow. "
            "Every promise soft and slow. Finding home in afterglow. ; "
            "[chorus] Hold the night a little longer. Let the melody grow stronger. "
            "We are writing every line. In this borrowed summer time. ; "
            "[outro-short]"
        ),
    }


def _run_generate(
    *,
    lyrics_item: dict[str, Any],
    model: str,
    run_name: str,
    gpu_label: str,
    use_flash_attn: bool,
    low_mem: bool,
    generate_type: str,
) -> dict[str, Any]:
    key = _norm_model(model)
    if not _runtime_ready():
        return {"success": False, "error": "runtime weights missing — run download"}
    if not _model_ready(key):
        return {
            "success": False,
            "error": f"model {key} missing — run download --model {key}",
            "model_dir": _dir_info(_model_dir(key)),
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2a_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    work = Path("/tmp/levo_work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    jsonl_path = work / "input.jsonl"
    item = dict(lyrics_item)
    if "idx" not in item:
        item["idx"] = name
    # auto_prompt library not available — strip if present
    item.pop("auto_prompt_audio_type", None)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    _link_runtime_into_repo()
    _patch_repo_for_runtime()
    ckpt_path = _model_dir(key)
    print(f"ckpt_path {ckpt_path}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "USER": "root",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRANSFORMERS_CACHE": str(REPO_DIR / "third_party" / "hub"),
            "HF_HOME": str(REPO_DIR / "third_party" / "hub"),
            "PYTHONPATH": ":".join(
                [
                    str(REPO_DIR / "codeclm" / "tokenizer"),
                    str(REPO_DIR),
                    str(REPO_DIR / "codeclm" / "tokenizer" / "Flow1dVAE"),
                ]
            ),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_LAUNCH_BLOCKING": "0",
        }
    )

    out_tmp = work / "out"
    out_tmp.mkdir(parents=True, exist_ok=True)

    gen_args = [
        "--ckpt_path",
        str(ckpt_path),
        "--input_jsonl",
        str(jsonl_path),
        "--save_dir",
        str(out_tmp),
        "--generate_type",
        generate_type,
    ]
    if use_flash_attn:
        gen_args.append("--use_flash_attn")
    if low_mem:
        gen_args.append("--low_mem")

    launcher = work / "run_generate.py"
    launcher.write_text(
        "import torch\n"
        "_orig = torch.load\n"
        "def _load(*a, **k):\n"
        "    k.setdefault('weights_only', False)\n"
        "    return _orig(*a, **k)\n"
        "torch.load = _load\n"
        "import runpy, sys\n"
        f"sys.argv = {['generate.py', *gen_args]!r}\n"
        f"runpy.run_path({str(REPO_DIR / 'generate.py')!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    cmd = ["python", str(launcher)]

    t0 = time.time()
    print("cwd", REPO_DIR, "cmd", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.time() - t0

    log_path = save_dir / "generate.log"
    log_path.write_text(
        f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
        encoding="utf-8",
    )

    if proc.returncode != 0:
        return {
            "success": False,
            "error": f"generate.py exit {proc.returncode}",
            "wall_s": round(wall, 2),
            "stderr_tail": (proc.stderr or "")[-2500:],
            "stdout_tail": (proc.stdout or "")[-1500:],
            "log": str(log_path),
        }

    audio_src = None
    audio_root = out_tmp / "audio"
    search_roots = [audio_root] if audio_root.exists() else []
    search_roots.append(out_tmp)
    for root in search_roots:
        for p in root.rglob("*"):
            if p.suffix.lower() in {".flac", ".wav", ".mp3"}:
                audio_src = p
                break
        if audio_src:
            break

    audio_info = None
    if audio_src is not None:
        dest = save_dir / f"audio{audio_src.suffix.lower()}"
        shutil.copy2(audio_src, dest)
        audio_info = {
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "name": dest.name,
        }

    for p in out_tmp.rglob("*.jsonl"):
        shutil.copy2(p, save_dir / p.name)

    vram_gb = None
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass

    result = {
        "success": audio_info is not None,
        "error": None if audio_info else "no audio file produced",
        "wall_s": round(wall, 2),
        "est_gpu_usd": _estimate_cost(gpu_label, wall),
        "gpu": gpu_label,
        "model": key,
        "hf_repo": HF_MODELS[key],
        "use_flash_attn": use_flash_attn,
        "low_mem": low_mem,
        "generate_type": generate_type,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "item": item,
        "log": str(log_path),
    }
    meta = {
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {
            "model": key,
            "use_flash_attn": use_flash_attn,
            "low_mem": low_mem,
            "generate_type": generate_type,
            "item": item,
        },
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": UPSTREAM_COMMIT,
        "license_note": "research/academic only — Tencent SongGeneration terms",
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
    memory=32768,
)
def generate_fn(
    lyrics_item: dict[str, Any],
    model: str = DEFAULT_MODEL,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    use_flash_attn: bool = True,
    low_mem: bool = False,
    generate_type: str = "mixed",
) -> dict[str, Any]:
    return _run_generate(
        lyrics_item=lyrics_item,
        model=model,
        run_name=run_name,
        gpu_label=gpu_label,
        use_flash_attn=use_flash_attn,
        low_mem=low_mem,
        generate_type=generate_type,
    )


@app.local_entrypoint()
def main(
    action: str = "status",
    gpu: str = DEFAULT_GPU,
    model: str = DEFAULT_MODEL,
    run_name: str = "",
    force_download: bool = False,
    low_mem: bool = False,
    no_flash: bool = False,
    generate_type: str = "mixed",
    lyrics: str = "",
    descriptions: str = "",
    idx: str = "gen",
):
    if action == "status":
        status_fn.remote()
        return

    if action == "download":
        download_weights.remote(force=force_download, model=model)
        return

    if action == "smoke":
        download_weights.remote(force=False, model=model)
        item = _default_smoke_item()
        rn = run_name or "smoke_en"
        use_low = low_mem or (
            gpu in ("L4", "T4", "A10") and _norm_model(model) == "v2-large"
        )
        out = generate_fn.with_options(gpu=gpu).remote(
            lyrics_item=item,
            model=model,
            run_name=rn,
            gpu_label=gpu,
            use_flash_attn=not no_flash,
            low_mem=use_low,
            generate_type=generate_type,
        )
        print("SMOKE_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        return

    if action == "t2a":
        download_weights.remote(force=False, model=model)
        if lyrics.strip():
            item = {"idx": idx or "gen", "gt_lyric": lyrics}
            if descriptions.strip():
                item["descriptions"] = descriptions
        else:
            item = _default_smoke_item()
            item["idx"] = idx or "gen"
            if descriptions.strip():
                item["descriptions"] = descriptions
        rn = run_name or f"t2a_{idx or 'gen'}"
        out = generate_fn.with_options(gpu=gpu).remote(
            lyrics_item=item,
            model=model,
            run_name=rn,
            gpu_label=gpu,
            use_flash_attn=not no_flash,
            low_mem=low_mem,
            generate_type=generate_type,
        )
        print("T2A_RESULT", json.dumps(out, ensure_ascii=False), flush=True)
        if not out.get("success"):
            raise SystemExit(2)
        return

    raise SystemExit(f"unknown action {action}")
