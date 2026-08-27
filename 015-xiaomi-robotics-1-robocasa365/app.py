# -*- coding: utf-8 -*-
"""
015-xiaomi-robotics-1-robocasa365 — Xiaomi-Robotics-1 (MiBoT) VLA
RoboCasa365 checkpoint on Modal.

Scope:
  - CPU download of HF weights → Volume
  - GPU smoke / infer: load processor+model, feed synthetic multi-view frames
    + language instruction + proprio state → 16-step action chunk
  - Full RoboCasa365 MuJoCo evaluation is NOT in this experiment (needs the
    separate robocasa_365 sim env + multi-GPU client/server launcher).

Upstream checkpoint:
  https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365
Code / eval:
  https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-xr1-robocasa365"
HF_REPO = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
UPSTREAM_CODE = "https://github.com/XiaomiRobotics/Xiaomi-Robotics-1"
DEFAULT_GPU = "A100-40GB"
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60
ACTION_DIM = 12  # first 12 dims used by RoboCasa365 EE+gripper packing
OBS_HISTORY = 4
NUM_DENOISE_STEPS = 5

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
MODEL_DIR = Path(WEIGHTS_MOUNT) / "Xiaomi-Robotics-1-RoboCasa365"
VOLUME_WEIGHTS = "modal-lab-xr1-robocasa365-weights"
VOLUME_OUTPUTS = "modal-lab-xr1-robocasa365-outputs"

DOWNLOAD_TIMEOUT = 2 * 60 * 60
INFER_TIMEOUT = 45 * 60

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

EXP_DIR = Path(__file__).resolve().parent

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# CUDA 12.8 + torch 2.8 + transformers 4.57.1 (official reference stack).
# Flash-Attn optional: smoke defaults to sdpa for easier cold start.
inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "ca-certificates",
    )
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "transformers==4.57.1",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "huggingface_hub[hf_transfer]>=0.26.0",
        "Pillow",
        "numpy",
        "einops",
        "scipy",
        "tqdm",
        "qwen-vl-utils",
        "av",
    )
    .env(
        {
            "HF_HOME": f"{WEIGHTS_MOUNT}/hf_home",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
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


def _estimate_cost(gpu: str, seconds: float) -> float | None:
    rate = GPU_PRICE_PER_SEC.get(gpu) or GPU_PRICE_PER_SEC.get(gpu.replace("!", ""))
    if rate is None:
        return None
    return round(rate * seconds, 4)


def _model_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    needed = [
        "config.json",
        "modeling_mibot.py",
        "processing_mibot.py",
        "model.safetensors.index.json",
    ]
    if not all((model_dir / n).is_file() for n in needed):
        return False
    shards = list(model_dir.glob("model-*.safetensors"))
    return len(shards) >= 3


def _nvidia_smi() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 4:
        return {"raw": out}
    return {
        "name": parts[0],
        "mem_used_mib": float(parts[1]),
        "mem_total_mib": float(parts[2]),
        "util_gpu_pct": float(parts[3]),
    }


def _make_synthetic_views(
    instruction: str,
    history: int = OBS_HISTORY,
    size: tuple[int, int] = (320, 256),  # H,W must be multiples of patch*merge=32
) -> dict[str, list]:
    """Build fake left/right/wrist camera frame lists (PIL RGB)."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    palettes = {
        "left": (40, 90, 140),
        "right": (40, 120, 80),
        "wrist": (120, 70, 40),
    }
    labels = {
        "left": "Left agentview",
        "right": "Right agentview",
        "wrist": "Eye-in-hand",
    }
    keys = {
        "left": "video.robot0_agentview_left",
        "right": "video.robot0_agentview_right",
        "wrist": "video.robot0_eye_in_hand",
    }
    out: dict[str, list] = {v: [] for v in keys.values()}
    for t in range(history):
        for short, key in keys.items():
            img = Image.new("RGB", (w, h), palettes[short])
            draw = ImageDraw.Draw(img)
            # simple fake counter + object as rectangles
            draw.rectangle([20, h - 70, w - 20, h - 20], fill=(180, 160, 130))
            ox = 60 + t * 12 + (10 if short != "wrist" else 0)
            draw.ellipse([ox, 80, ox + 50, 130], fill=(200, 60, 50))
            draw.rectangle([w // 2 - 15, 40, w // 2 + 15, 100], fill=(90, 90, 95))
            try:
                font = ImageFont.load_default()
            except Exception:  # noqa: BLE001
                font = None
            draw.text((8, 8), f"{labels[short]} t={t}", fill=(255, 255, 255), font=font)
            draw.text((8, 22), instruction[:48], fill=(230, 230, 200), font=font)
            out[key].append(img)
    return out


def _build_messages(views: dict[str, list], instruction: str) -> list[dict[str, Any]]:
    cam_left = "video.robot0_agentview_left"
    cam_right = "video.robot0_agentview_right"
    cam_wrist = "video.robot0_eye_in_hand"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Left camera: "},
                {"type": "video", "video": views[cam_left]},
                {"type": "text", "text": "\nRight camera: "},
                {"type": "video", "video": views[cam_right]},
                {"type": "text", "text": "\nWrist camera: "},
                {"type": "video", "video": views[cam_wrist]},
                {
                    "type": "text",
                    "text": (
                        f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<cot></cot>"}],
        },
    ]


def _load_model(model_dir: Path, attn_implementation: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=False,
    )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    # Prefer requested attn; fall back to sdpa if flash_attn missing.
    tried = []
    model = None
    last_err: Exception | None = None
    for attn in [attn_implementation, "sdpa", "eager"]:
        if attn in tried:
            continue
        tried.append(attn)
        try:
            model = AutoModel.from_pretrained(
                str(model_dir),
                attn_implementation=attn,
                **load_kwargs,
            )
            attn_implementation = attn
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            model = None
    if model is None:
        raise RuntimeError(f"failed to load model; last error: {last_err!r}")
    model = model.cuda().to(torch.bfloat16).eval()
    return processor, model, attn_implementation


def _run_infer(
    *,
    instruction: str,
    run_name: str,
    gpu_label: str,
    attn_implementation: str,
    num_steps: int,
    obs_history: int,
) -> dict[str, Any]:
    import torch
    import numpy as np

    t0 = time.time()
    if not _model_ready(MODEL_DIR):
        return {
            "success": False,
            "error": f"weights not ready at {MODEL_DIR}; run download first",
            "weights": _dir_info(MODEL_DIR),
        }

    smi_before = _nvidia_smi()
    load_t0 = time.time()
    processor, model, attn_used = _load_model(MODEL_DIR, attn_implementation)
    load_s = round(time.time() - load_t0, 2)

    robot_types = processor.list_robot_types()
    if ROBOT_TYPE not in robot_types:
        return {
            "success": False,
            "error": f"robot_type {ROBOT_TYPE!r} not in {robot_types}",
            "robot_types": robot_types,
        }

    views = _make_synthetic_views(instruction, history=obs_history)
    # state: (1, T, 60) — first 14 dims are real RoboCasa365 EE packing; rest pad
    state = np.zeros((1, obs_history, STATE_DIM), dtype=np.float32)
    # mild non-zero proprio so projector sees a signal
    for t in range(obs_history):
        state[0, t, 0:3] = [0.45, 0.02 * t, 0.95]
        state[0, t, 6] = 0.04  # gripper open-ish

    prep_t0 = time.time()
    inputs = processor.apply_chat_template(
        _build_messages(views, instruction),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,  # we already pass exact history frames
        state=state,
        robot_type=ROBOT_TYPE,
    )
    prep_s = round(time.time() - prep_t0, 3)

    batch: dict[str, Any] = {}
    for k, v in dict(inputs).items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                batch[k] = v.to(device=model.device, dtype=model.dtype)
            else:
                batch[k] = v.to(device=model.device)
        else:
            batch[k] = v

    # Drop task_id-like non-tensor keys the model doesn't take
    batch.pop("task_id", None)

    torch.cuda.synchronize()
    infer_t0 = time.time()
    with torch.inference_mode():
        outputs = model(**batch, num_steps=num_steps)
    torch.cuda.synchronize()
    infer_s = round(time.time() - infer_t0, 3)

    actions_norm = outputs.actions  # (1, 16, 60) normalized
    actions = processor.decode_action(actions_norm, robot_type=ROBOT_TYPE)
    actions_np = actions[0].float().cpu().numpy()
    actions_ee = actions_np[:, :ACTION_DIM]  # RoboCasa365 uses first 12 dims

    smi_after = _nvidia_smi()
    peak = None
    if torch.cuda.is_available():
        peak = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

    # persist run
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"run_{stamp}"
    out_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "actions_full.npy", actions_np)
    np.save(out_dir / "actions_ee12.npy", actions_ee)
    np.save(out_dir / "actions_normalized.npy", actions_norm[0].float().cpu().numpy())

    # save a contact sheet of synthetic inputs
    try:
        from PIL import Image

        row = []
        for key in (
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        ):
            row.append(views[key][-1].resize((160, 120)))
        sheet = Image.new("RGB", (160 * 3, 120))
        for i, im in enumerate(row):
            sheet.paste(im, (i * 160, 0))
        sheet.save(out_dir / "input_views.jpg", quality=90)
    except Exception as e:  # noqa: BLE001
        sheet_err = repr(e)
    else:
        sheet_err = None

    meta = {
        "success": True,
        "run_name": name,
        "hf_repo": HF_REPO,
        "robot_type": ROBOT_TYPE,
        "instruction": instruction,
        "obs_history": obs_history,
        "num_denoise_steps": num_steps,
        "attn_implementation": attn_used,
        "gpu": gpu_label,
        "timings_s": {
            "load": load_s,
            "prep": prep_s,
            "infer": infer_s,
            "wall": round(time.time() - t0, 2),
        },
        "cost_est_usd": _estimate_cost(gpu_label, time.time() - t0),
        "vram": {
            "before": smi_before,
            "after": smi_after,
            "peak_alloc_gb": peak,
        },
        "shapes": {
            "actions_full": list(actions_np.shape),
            "actions_ee12": list(actions_ee.shape),
            "action_mask": list(batch["action_mask"].shape)
            if "action_mask" in batch
            else None,
            "state": list(batch["state"].shape) if "state" in batch else None,
            "input_ids": list(batch["input_ids"].shape)
            if "input_ids" in batch
            else None,
        },
        "action_ee12_stats": {
            "mean": actions_ee.mean(axis=0).round(5).tolist(),
            "std": actions_ee.std(axis=0).round(5).tolist(),
            "first_step": actions_ee[0].round(5).tolist(),
        },
        "robot_types": robot_types,
        "sheet_error": sheet_err,
        "weights": _dir_info(MODEL_DIR),
        "utc": stamp,
        "note": (
            "Synthetic multi-view smoke only — not a RoboCasa365 success-rate eval. "
            "Full benchmark needs MuJoCo + robocasa_365 client/server (see UPSTREAM)."
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs_vol.commit()
    return meta


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=8192,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    if _model_ready(MODEL_DIR) and not force:
        info = _dir_info(MODEL_DIR)
        info.update({"skipped": True, "reason": "already present", "wall_s": 0})
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return info

    if force and MODEL_DIR.exists():
        import shutil

        shutil.rmtree(MODEL_DIR)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    weights_vol.commit()
    info = _dir_info(MODEL_DIR)
    info.update(
        {
            "skipped": False,
            "snapshot": path,
            "wall_s": round(time.time() - t0, 2),
            "ready": _model_ready(MODEL_DIR),
        }
    )
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return info


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=120,
    cpu=1,
    memory=2048,
)
def status_fn() -> dict[str, Any]:
    runs_root = Path(OUTPUTS_MOUNT) / "runs"
    runs = []
    if runs_root.is_dir():
        for p in sorted(runs_root.iterdir()):
            if not p.is_dir():
                continue
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
                    "success": (meta or {}).get("success"),
                    "instruction": (meta or {}).get("instruction"),
                    "infer_s": ((meta or {}).get("timings_s") or {}).get("infer"),
                    "gpu": (meta or {}).get("gpu"),
                }
            )
    out = {
        "app": APP_NAME,
        "hf_repo": HF_REPO,
        "upstream_code": UPSTREAM_CODE,
        "robot_type": ROBOT_TYPE,
        "default_gpu": DEFAULT_GPU,
        "weights": _dir_info(MODEL_DIR),
        "weights_ready": _model_ready(MODEL_DIR),
        "outputs": _dir_info(Path(OUTPUTS_MOUNT)),
        "runs": runs[-20:],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


@app.function(
    image=download_image,
    volumes={OUTPUTS_MOUNT: outputs_vol},
    timeout=120,
    cpu=1,
    memory=2048,
)
def list_outputs_fn() -> dict[str, Any]:
    root = Path(OUTPUTS_MOUNT) / "runs"
    if not root.exists():
        out = {"exists": False, "path": str(root), "runs": []}
        print(json.dumps(out, indent=2), flush=True)
        return out
    runs = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        files = [f.name for f in p.iterdir() if f.is_file()]
        runs.append({"name": p.name, "files": files})
    out = {"exists": True, "path": str(root), "runs": runs}
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(
    image=inference_image,
    gpu=DEFAULT_GPU,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=INFER_TIMEOUT,
    memory=32768,
    cpu=4,
)
def infer_fn(
    instruction: str = "close the blender lid",
    run_name: str = "",
    gpu_label: str = DEFAULT_GPU,
    attn_implementation: str = "sdpa",
    num_steps: int = NUM_DENOISE_STEPS,
    obs_history: int = OBS_HISTORY,
) -> dict[str, Any]:
    result = _run_infer(
        instruction=instruction,
        run_name=run_name,
        gpu_label=gpu_label,
        attn_implementation=attn_implementation,
        num_steps=num_steps,
        obs_history=obs_history,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


ATTN_CHOICES = ("sdpa", "flash_attention_2", "eager")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="015 Xiaomi-Robotics-1 RoboCasa365 VLA")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("check", help="远程检查权重 / 最近 runs")
    sub.add_parser("list-outputs", help="结构化列出远程 run 文件")

    download = sub.add_parser("download", help="CPU 下载 HF 权重到 Volume")
    download.add_argument("--force", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="合成三视角 + 固定指令动作生成")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gpu", default=DEFAULT_GPU)
    smoke.add_argument("--instruction", default="close the blender lid")
    smoke.add_argument("--run-name", default="smoke_close_blender_lid")
    smoke.add_argument("--attn", choices=ATTN_CHOICES, default="sdpa")
    smoke.add_argument("--num-steps", type=int, default=NUM_DENOISE_STEPS)
    smoke.add_argument("--obs-history", type=int, default=OBS_HISTORY)

    infer_cmd = sub.add_parser("infer", help="自定义指令动作生成")
    infer_cmd.add_argument("--dry-run", action="store_true")
    infer_cmd.add_argument("--gpu", default=DEFAULT_GPU)
    infer_cmd.add_argument("--instruction", required=True)
    infer_cmd.add_argument("--run-name", default="")
    infer_cmd.add_argument("--attn", choices=ATTN_CHOICES, default="sdpa")
    infer_cmd.add_argument("--num-steps", type=int, default=NUM_DENOISE_STEPS)
    infer_cmd.add_argument("--obs-history", type=int, default=OBS_HISTORY)
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "015-xiaomi-robotics-1-robocasa365",
        "app": APP_NAME,
        "hf_repo": HF_REPO,
        "upstream_code": UPSTREAM_CODE,
        "robot_type": ROBOT_TYPE,
        "default_gpu": DEFAULT_GPU,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "obs_history": OBS_HISTORY,
        "num_denoise_steps": NUM_DENOISE_STEPS,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "scope": "synthetic multi-view VLA smoke/infer; not full RoboCasa365 simulation eval",
    }


def inference_plan(args: argparse.Namespace) -> dict[str, Any]:
    instruction = args.instruction.strip()
    if not instruction:
        raise ValueError("instruction 不能为空")
    if args.num_steps <= 0:
        raise ValueError("--num-steps 必须 > 0")
    if args.obs_history <= 0:
        raise ValueError("--obs-history 必须 > 0")
    return {
        "action": args.command,
        "gpu": args.gpu,
        "instruction": instruction,
        "run_name": args.run_name,
        "attn": args.attn,
        "num_steps": args.num_steps,
        "obs_history": args.obs_history,
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
        plan = inference_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    fn = infer_fn if plan["gpu"] == DEFAULT_GPU else infer_fn.with_options(gpu=plan["gpu"])
    result = fn.remote(
        instruction=plan["instruction"],
        run_name=plan["run_name"],
        gpu_label=plan["gpu"],
        attn_implementation=plan["attn"],
        num_steps=plan["num_steps"],
        obs_history=plan["obs_history"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])
