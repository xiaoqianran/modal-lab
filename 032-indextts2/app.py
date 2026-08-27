# -*- coding: utf-8 -*-
"""
032-indextts2 — IndexTTS-2 on Modal

真实用量榜 Tier A4：GH index-tts ~22.5k · 中文配音/时长控制
默认：IndexTeam/IndexTTS-2 · GPU L4 · Bilibili IndexTTS license

上游: https://github.com/index-tts/index-tts
权重: https://huggingface.co/IndexTeam/IndexTTS-2
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

APP_NAME = "modal-lab-indextts2"

HF_REPO = "IndexTeam/IndexTTS-2"
DEFAULT_MODEL = "indextts2"
DEFAULT_GPU = "L4"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PROMPTS_MOUNT = "/prompts"
HF_HOME = Path(WEIGHTS_MOUNT) / "hf"
LOCAL_MODEL = Path(WEIGHTS_MOUNT) / "models" / "IndexTTS-2"
VOLUME_WEIGHTS = "modal-lab-indextts2-weights"
VOLUME_OUTPUTS = "modal-lab-indextts2-outputs"
VOLUME_PROMPTS = "modal-lab-indextts2-prompts"

PROMPT_WAV_NAME = "ref.wav"

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
    "MODELSCOPE_CACHE": str(Path(WEIGHTS_MOUNT) / "modelscope"),
}

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0,<1.0")
    .env(_HF_ENV)
)

# torch 2.8 + cu126 · IndexTTS-2 deps · force wetext (no WeTextProcessing)
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "git",
        "git-lfs",
        "ca-certificates",
        "curl",
        "build-essential",
        "g++",
        "sox",
        "libsox-dev",
    )
    .run_commands("git lfs install")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "accelerate==1.8.1",
        "cn2an==0.5.22",
        "cython==3.0.7",
        "descript-audiotools==0.7.2",
        "einops>=0.8.1",
        "ffmpeg-python==0.2.0",
        "g2p-en==2.1.0",
        "jieba==0.42.1",
        "json5==0.10.0",
        "librosa==0.10.2.post1",
        "matplotlib==3.10.0",
        "modelscope==1.27.0",
        "munch==4.0.0",
        "numpy==2.2.6",
        "omegaconf>=2.3.0",
        "opencv-python-headless==4.9.0.80",
        "pandas==2.3.2",
        "safetensors==0.5.2",
        "sentencepiece>=0.2.1",
        "textstat>=0.7.10",
        "tokenizers==0.21.0",
        "transformers==4.52.1",
        "wetext>=0.0.9",
        "soundfile",
        "tqdm",
        "requests",
        "huggingface_hub[hf_transfer]>=0.26.0,<1.0",
        "scipy",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/index-tts/index-tts.git /opt/index-tts",
        # Force wetext on Linux — sed is reliable inside image build
        "sed -i 's/if platform.system() != \"Linux\":  # Mac and Windows/if True:  # modal-lab force wetext/' "
        "/opt/index-tts/indextts/utils/front.py",
        "grep -n 'force wetext\\|platform.system' /opt/index-tts/indextts/utils/front.py | head -5",
    )
    .env({
        **_HF_ENV,
        "PYTHONPATH": "/opt/index-tts",
        "INDEXTTS_MODEL_DIR": str(LOCAL_MODEL),
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
    need = ["gpt.pth", "s2mel.pth", "config.yaml", "bpe.model"]
    return all((LOCAL_MODEL / n).is_file() for n in need)


def _ensure_prompt_wav() -> Path:
    dest = Path(PROMPTS_MOUNT) / PROMPT_WAV_NAME
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    Path(PROMPTS_MOUNT).mkdir(parents=True, exist_ok=True)
    for cand in (
        Path(PROMPTS_MOUNT) / "zero_shot_prompt.wav",
        Path("/prompts") / "ref.wav",
    ):
        if cand.is_file() and cand.stat().st_size > 1000:
            shutil.copy2(cand, dest)
            return dest
    raise FileNotFoundError(
        f"missing prompt wav at {dest}; upload ref.wav to prompts volume"
    )


def _patch_front_runtime() -> None:
    """Runtime safety net if image sed patch missed."""
    p = Path("/opt/index-tts/indextts/utils/front.py")
    if not p.is_file():
        return
    t = p.read_text(encoding="utf-8")
    needle = 'if platform.system() != "Linux":  # Mac and Windows'
    if needle in t:
        p.write_text(
            t.replace(needle, "if True:  # modal-lab force wetext", 1),
            encoding="utf-8",
        )
        print("runtime-patched front.py → wetext", flush=True)


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
        "slot": "032-indextts2",
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "license": "Bilibili-IndexTTS (register for commercial)",
        "hf_repo": HF_REPO,
        "weights_ready": _weights_ready(),
        "model_dir": _dir_info(LOCAL_MODEL),
        "outputs": _dir_info(runs_root),
        "recent_runs": recent,
        "ranking_note": "GH index-tts ~22.5k · 时长+情感控制 · Tier A4",
        "modes": ["zero_shot_zh", "zero_shot_en", "emo_text"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def _run_tts(
    *,
    text: str,
    run_name: str,
    gpu_label: str,
    spk_audio: str = "",
    emo_text: str = "",
    use_emo_text: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch

    if "/opt/index-tts" not in sys.path:
        sys.path.insert(0, "/opt/index-tts")

    _patch_front_runtime()

    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "empty text"}
    if not _weights_ready():
        return {"success": False, "error": "weights missing; run download first"}

    if spk_audio.strip() and Path(spk_audio.strip()).is_file():
        prompt_path = Path(spk_audio.strip())
    else:
        try:
            prompt_path = _ensure_prompt_wav()
        except Exception as e:
            return {"success": False, "error": f"prompt wav: {e!r}"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"t2s_{ts}"
    save_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "audio.wav"

    from indextts.infer_v2 import IndexTTS2

    t0 = time.time()
    t_load = time.time()
    print(f"Loading IndexTTS2 from {LOCAL_MODEL}…", flush=True)
    tts = IndexTTS2(
        cfg_path=str(LOCAL_MODEL / "config.yaml"),
        model_dir=str(LOCAL_MODEL),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
    )
    load_s = time.time() - t_load

    t_gen = time.time()
    kwargs: dict[str, Any] = {
        "spk_audio_prompt": str(prompt_path),
        "text": text,
        "output_path": str(out_path),
        "verbose": True,
    }
    if use_emo_text or emo_text:
        kwargs["use_emo_text"] = True
        if emo_text:
            kwargs["emo_text"] = emo_text
    tts.infer(**kwargs)
    gen_s = time.time() - t_gen
    wall = time.time() - t0

    if not out_path.is_file() or out_path.stat().st_size < 1000:
        return {
            "success": False,
            "error": "output wav missing or tiny",
            "wall_s": round(wall, 2),
            "load_s": round(load_s, 2),
            "generate_s": round(gen_s, 2),
        }

    arr, sr = sf.read(str(out_path), always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)

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
        "repo_id": HF_REPO,
        "spk_audio": str(prompt_path),
        "use_emo_text": bool(use_emo_text or emo_text),
        "emo_text": emo_text or None,
        "vram_peak_gb": vram_gb,
        "audio": audio_info,
        "text": text[:800],
        "text_len": len(text),
        "license_note": "Bilibili IndexTTS — register for commercial use",
    }
    meta = {
        "experiment": "032-indextts2",
        "run_name": name,
        "gpu_requested": gpu_label,
        "wall_s": result["wall_s"],
        "est_gpu_usd": result["est_gpu_usd"],
        "payload": {"text": text, "model": DEFAULT_MODEL, "emo_text": emo_text},
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
    memory=32768,
    scaledown_window=60,
)
def generate_fn(
    text: str,
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    spk_audio: str = "",
    emo_text: str = "",
    use_emo_text: bool = False,
) -> dict[str, Any]:
    return _run_tts(
        text=text,
        run_name=run_name,
        gpu_label=gpu_label,
        spk_audio=spk_audio,
        emo_text=emo_text,
        use_emo_text=use_emo_text,
    )


SMOKE_ZH = "你好，这是 modal-lab 第零三二号 IndexTTS-2 实验。精确时长与情感控制，是中文配音的硬通货。"
SMOKE_EN = "Hello from IndexTTS-2 on Modal lab. Industrial zero-shot speech with duration and emotion control."
SMOKE_EMO = "这些年的时光终究是错付了……我只希望你能过得比我好一点。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="032 IndexTTS-2 on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / prompt / outputs Volume")

    download = sub.add_parser("download", help="下载 IndexTTS-2 权重")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="固定 ZH / EN / emotion smoke")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--kind", default="zh", choices=["zh", "en", "emo"])
    smoke.add_argument("--run-name", default="")
    smoke.add_argument("--emo-text", default="")
    smoke.add_argument("--spk-audio", default="", help="prompts Volume 内远程 wav 路径")

    t2s = sub.add_parser("t2s", help="zero-shot Text-to-Speech")
    t2s.add_argument("--dry-run", action="store_true")
    t2s.add_argument("--gpu", default=DEFAULT_GPU)
    t2s.add_argument("--text", required=True)
    t2s.add_argument("--emo-text", default="")
    t2s.add_argument("--spk-audio", default="", help="prompts Volume 内远程 wav 路径")
    t2s.add_argument("--run-name", default="")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "032-indextts2",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "default_model": DEFAULT_MODEL,
        "hf_repo": HF_REPO,
        "prompt_wav": PROMPT_WAV_NAME,
        "prompt_note": "默认需要 prompts Volume /ref.wav；上传使用 modal volume put",
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "prompts_volume": VOLUME_PROMPTS,
    }


def smoke_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "en":
        text, run_name, emo_text, use_emo = SMOKE_EN, args.run_name or "smoke_en", "", False
    elif args.kind == "emo":
        text = SMOKE_EMO
        run_name = args.run_name or "smoke_emo"
        emo_text = args.emo_text or "极度悲伤"
        use_emo = True
    else:
        text, run_name, emo_text, use_emo = SMOKE_ZH, args.run_name or "smoke_zh", "", False
    return {
        "action": "smoke",
        "gpu": args.gpu,
        "kind": args.kind,
        "text": text,
        "run_name": run_name,
        "spk_audio": args.spk_audio,
        "emo_text": emo_text,
        "use_emo_text": use_emo,
    }


def t2s_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "t2s",
        "gpu": args.gpu,
        "text": args.text.strip(),
        "run_name": args.run_name,
        "spk_audio": args.spk_audio,
        "emo_text": args.emo_text,
        "use_emo_text": bool(args.emo_text),
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

    plan = smoke_plan(args) if args.command == "smoke" else t2s_plan(args)
    if not plan["text"]:
        raise SystemExit("t2s requires non-empty --text")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    download_weights.remote(force=False)
    out = generate_fn.with_options(gpu=args.gpu).remote(
        text=plan["text"],
        run_name=plan["run_name"],
        gpu_label=args.gpu,
        spk_audio=plan["spk_audio"],
        emo_text=plan["emo_text"],
        use_emo_text=plan["use_emo_text"],
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
