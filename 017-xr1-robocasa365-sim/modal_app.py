# -*- coding: utf-8 -*-
"""
017-xr1-robocasa365-sim — RoboCasa365 simulation smoke on Modal.

Goals:
  1) download kitchen assets (~10GB) to a Volume
  2) random-policy rollout → episode .mp4 (proves sim + video)
  3) optional XR-1 closed-loop smoke (1 task × 1 episode, short horizon)

Full 2500-episode eval is intentionally out of scope (multi-hour / multi-$$$).
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

APP_NAME = "modal-lab-xr1-robocasa365-sim"
HF_REPO = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
ROBOSUITE = "https://github.com/ARISE-Initiative/robosuite"
ROBOCASA = "https://github.com/robocasa/robocasa"
XR1_CODE = "https://github.com/XiaomiRobotics/Xiaomi-Robotics-1"

DEFAULT_GPU = "A100-40GB"
DEFAULT_TASK = "CloseBlenderLid"
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60
ACTION_DIM = 12
OBS_HISTORY = 4
OBS_INTERVAL = 2
REPLAN_STEPS = 16
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)

WEIGHTS_MOUNT = "/weights"
ASSETS_MOUNT = "/assets"
OUTPUTS_MOUNT = "/outputs"
MODEL_DIR = Path(WEIGHTS_MOUNT) / "Xiaomi-Robotics-1-RoboCasa365"
# Kitchen assets live under the installed robocasa package; we also keep a
# Volume mirror so restarts do not re-download 10GB every cold start.
ASSETS_VOL_DIR = Path(ASSETS_MOUNT) / "kitchen_assets_marker"
VOLUME_WEIGHTS = "modal-lab-xr1-robocasa365-weights"  # shared with 015
VOLUME_ASSETS = "modal-lab-xr1-robocasa365-sim-assets"
VOLUME_OUTPUTS = "modal-lab-xr1-robocasa365-sim-outputs"

DOWNLOAD_TIMEOUT = 3 * 60 * 60
SIM_TIMEOUT = 2 * 60 * 60

GPU_PRICE_PER_SEC = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H100!": 0.001097,
}

EXP_DIR = Path(__file__).resolve().parent

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
assets_vol = modal.Volume.from_name(VOLUME_ASSETS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

# ---------- images ----------
# Slim CPU image for volume listing / status
status_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.26.0")
    .env({"PYTHONUNBUFFERED": "1"})
)

# Weights download (reuse 015 path)
download_weights_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)

# Simulation stack: CUDA runtime + EGL + robosuite/robocasa + torch for policy
sim_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "wget",
        "curl",
        "ca-certificates",
        "build-essential",
        "g++",
        "gcc",
        "make",
        "python3-dev",
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libgles2",
        "libosmesa6",
        "libosmesa6-dev",
        "libglfw3",
        "libgomp1",
        "libx11-6",
        "libxrandr2",
        "libxss1",
        "libxcursor1",
        "libxcomposite1",
        "libasound2",
        "libxi6",
        "libxtst6",
        "libxinerama1",
        "libxkbcommon0",
        "mesa-utils",
        "linux-libc-dev",
    )
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # Headless eval stack only — skip pynput/hidapi/evdev teleop deps.
        "transformers==4.57.1",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]>=0.26.0",
        "numpy<2.2",
        "scipy",
        "Pillow",
        "tqdm",
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "opencv-python-headless",
        "gymnasium",
        "mujoco",
        "numba",
        "h5py",
        "lxml",
        "pyyaml",
        "termcolor",
        "einops",
        "av",
        "qwen-vl-utils",
        "pyopengl",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/ARISE-Initiative/robosuite.git /opt/robosuite",
        "cd /opt/robosuite && pip install -e . --no-cache-dir",
        "git clone --depth 1 https://github.com/robocasa/robocasa.git /opt/robocasa",
        # skip shadowed pins; deps already installed above
        "cd /opt/robocasa && pip install -e . --no-deps --no-cache-dir",
        # non-interactive private macros
        "python -c \"import shutil, robocasa, os; "
        "bp=robocasa.__path__[0]; "
        "src=os.path.join(bp,'macros.py'); "
        "dst=os.path.join(bp,'macros_private.py'); "
        "shutil.copyfile(src,dst); print('macros_private ok', dst)\"",
        "python -c \"import robosuite, robocasa, gymnasium; print('robosuite', robosuite.__version__ if hasattr(robosuite,'__version__') else 'ok'); print('robocasa', robocasa.__path__)\"",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": f"{WEIGHTS_MOUNT}/hf_home",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
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
    need = ["config.json", "modeling_mibot.py", "model.safetensors.index.json"]
    if not all((model_dir / n).is_file() for n in need):
        return False
    return len(list(model_dir.glob("model-*.safetensors"))) >= 3


def _assets_ready() -> bool:
    # Heuristic: textures + fixtures folders exist and are non-empty under robocasa package
    try:
        import robocasa

        base = Path(robocasa.__path__[0]) / "models" / "assets"
        tex = base / "textures"
        fix = base / "fixtures"
        if not tex.is_dir() or not fix.is_dir():
            return False
        # any files?
        return any(tex.rglob("*")) and any(fix.rglob("*"))
    except Exception:
        return False


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


# ---------- remote helpers ----------
def _download_assets_impl(force: bool = False) -> dict[str, Any]:
    t0 = time.time()
    import robocasa
    from robocasa.scripts.download_kitchen_assets import (
        DOWNLOAD_ASSET_REGISTRY,
        download_and_extract_zip,
    )

    base = Path(robocasa.__path__[0]) / "models" / "assets"
    if _assets_ready() and not force:
        info = {
            "skipped": True,
            "reason": "assets already present",
            "assets": _dir_info(base),
            "wall_s": 0.0,
            "ready": True,
        }
        print(json.dumps(info, indent=2), flush=True)
        return info

    results = {}
    for name, config in DOWNLOAD_ASSET_REGISTRY.items():
        cfg = dict(config)
        # disable interactive prompts
        cfg["prompt_before_download"] = False
        cfg["check_folder_exists"] = False
        cfg["delete_old_folder"] = bool(force)
        print(f"==> downloading asset pack: {name}", flush=True)
        try:
            download_and_extract_zip(**cfg)
            results[name] = "ok"
        except TypeError:
            # older signature without prompt kwargs — monkey-patch input
            import builtins

            real_input = builtins.input
            builtins.input = lambda *a, **k: "y"  # noqa: E731
            try:
                download_and_extract_zip(**config)
                results[name] = "ok"
            except Exception as e:  # noqa: BLE001
                results[name] = repr(e)
            finally:
                builtins.input = real_input
        except Exception as e:  # noqa: BLE001
            results[name] = repr(e)

    # marker on assets volume for status_fn without importing robocasa
    ASSETS_VOL_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_VOL_DIR / "ready.json").write_text(
        json.dumps({"ready": _assets_ready(), "results": results}, indent=2),
        encoding="utf-8",
    )
    assets_vol.commit()

    info = {
        "skipped": False,
        "results": results,
        "assets": _dir_info(base),
        "ready": _assets_ready(),
        "wall_s": round(time.time() - t0, 2),
    }
    print(json.dumps(info, indent=2, default=str), flush=True)
    return info


def _make_video_frame(observation: dict[str, Any]):
    import numpy as np

    frames = []
    for key in CAMERA_KEYS:
        if key in observation:
            frames.append(np.asarray(observation[key], dtype=np.uint8))
    if not frames:
        # fallback: any video.* keys
        for k, v in observation.items():
            if str(k).startswith("video.") and hasattr(v, "shape"):
                frames.append(np.asarray(v, dtype=np.uint8))
    if not frames:
        raise RuntimeError(f"no camera frames in obs keys={list(observation)[:30]}")
    return np.ascontiguousarray(np.concatenate(frames, axis=1))


def _quat_xyzw_to_axis_angle(quaternion):
    import numpy as np

    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def _observation_to_state(observation: dict[str, Any]):
    import numpy as np

    state = np.concatenate(
        [
            np.asarray(
                observation["state.end_effector_position_relative"], dtype=np.float32
            ).reshape(-1),
            _quat_xyzw_to_axis_angle(observation["state.end_effector_rotation_relative"]),
            np.asarray(observation["state.gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(observation["state.base_position"], dtype=np.float32).reshape(-1),
            _quat_xyzw_to_axis_angle(observation["state.base_rotation"]),
        ]
    ).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected 14D state, got {state.shape}")
    return state


def _sample_history(history, length: int, interval: int):
    import numpy as np

    items = list(history)
    if not items:
        raise ValueError("empty history")
    indices = [
        max(0, len(items) - 1 - (length - 1 - index) * interval)
        for index in range(length)
    ]
    return np.ascontiguousarray(np.stack([items[i] for i in indices], axis=0))


def _center_crop(image, crop_ratio: float):
    from PIL import Image
    import numpy as np

    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil
    w, h = pil.size
    cw = max(1, int(w * crop_ratio))
    ch = max(1, int(h * crop_ratio))
    left = (w - cw) // 2
    top = (h - ch) // 2
    return pil.crop((left, top, left + cw, top + ch))


def _build_messages(image_history: dict, instruction: str, crop_ratio: float):
    videos = {
        key: [_center_crop(frame, crop_ratio) for frame in image_history[key]]
        for key in CAMERA_KEYS
    }
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Left camera: "},
                {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                {"type": "text", "text": "\nRight camera: "},
                {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                {"type": "text", "text": "\nWrist camera: "},
                {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                {
                    "type": "text",
                    "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def _load_policy(model_dir: Path, attn: str = "sdpa"):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, use_fast=False
    )
    last_err = None
    model = None
    used = attn
    for a in [attn, "sdpa", "eager"]:
        try:
            model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                attn_implementation=a,
                dtype=torch.bfloat16,
            )
            used = a
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            model = None
    if model is None:
        raise RuntimeError(f"model load failed: {last_err!r}")
    model = model.cuda().to(torch.bfloat16).eval()
    return processor, model, used


def _infer_actions(processor, model, state_history, image_history, instruction, crop_ratio, num_steps=5):
    import torch
    import numpy as np

    state_history = np.asarray(state_history, dtype=np.float32)
    state = np.zeros((1, state_history.shape[0], STATE_DIM), dtype=np.float32)
    state[0, :, : state_history.shape[-1]] = state_history

    inputs = processor.apply_chat_template(
        _build_messages(image_history, instruction, crop_ratio),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
        state=state,
        robot_type=ROBOT_TYPE,
    )
    batch = {}
    for k, v in dict(inputs).items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                batch[k] = v.to(device=model.device, dtype=model.dtype)
            else:
                batch[k] = v.to(device=model.device)
        else:
            batch[k] = v
    batch.pop("task_id", None)
    with torch.inference_mode():
        out = model(**batch, num_steps=num_steps)
    actions = processor.decode_action(out.actions, robot_type=ROBOT_TYPE)
    return actions[0, :, :ACTION_DIM].float().cpu().numpy().astype(np.float32)


# ---------- Modal functions ----------
@app.function(
    image=download_weights_image,
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
        info.update({"skipped": True, "ready": True, "wall_s": 0})
        print(json.dumps(info, indent=2), flush=True)
        return info
    if force and MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, local_dir=str(MODEL_DIR))
    weights_vol.commit()
    info = _dir_info(MODEL_DIR)
    info.update(
        {
            "skipped": False,
            "ready": _model_ready(MODEL_DIR),
            "wall_s": round(time.time() - t0, 2),
        }
    )
    print(json.dumps(info, indent=2), flush=True)
    return info


@app.function(
    image=sim_image,
    volumes={ASSETS_MOUNT: assets_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=16384,
)
def download_assets(force: bool = False) -> dict[str, Any]:
    return _download_assets_impl(force=force)


@app.function(
    image=status_image,
    volumes={
        WEIGHTS_MOUNT: weights_vol,
        ASSETS_MOUNT: assets_vol,
        OUTPUTS_MOUNT: outputs_vol,
    },
    timeout=120,
    cpu=1,
    memory=2048,
)
def status_fn() -> dict[str, Any]:
    runs_root = Path(OUTPUTS_MOUNT) / "runs"
    runs = []
    if runs_root.is_dir():
        for p in sorted(runs_root.iterdir()):
            if p.is_dir():
                meta = p / "meta.json"
                m = None
                if meta.is_file():
                    try:
                        m = json.loads(meta.read_text())
                    except Exception:
                        m = None
                vids = [f.name for f in p.rglob("*.mp4")]
                runs.append(
                    {
                        "name": p.name,
                        "success": (m or {}).get("success"),
                        "mode": (m or {}).get("mode"),
                        "videos": vids,
                        "task": (m or {}).get("task"),
                    }
                )
    marker = ASSETS_VOL_DIR / "ready.json"
    assets_marker = None
    if marker.is_file():
        try:
            assets_marker = json.loads(marker.read_text())
        except Exception:
            assets_marker = {"raw": marker.read_text()[:200]}
    out = {
        "app": APP_NAME,
        "note": "016 is MusicGen; this sim experiment is 017",
        "hf_repo": HF_REPO,
        "default_gpu": DEFAULT_GPU,
        "default_task": DEFAULT_TASK,
        "weights": _dir_info(MODEL_DIR),
        "weights_ready": _model_ready(MODEL_DIR),
        "assets_marker": assets_marker,
        "outputs": _dir_info(Path(OUTPUTS_MOUNT)),
        "runs": runs[-30:],
        "cost_guide": {
            "download_assets_cpu": "~$0.02–0.10 (one-time ~10GB)",
            "smoke_random_1ep": "~$0.05–0.30 on L4/A100 (minutes)",
            "smoke_policy_horizon20": "~$0.10–0.60 on A100-40GB",
            "full_2500_eps": "hours–days, often $50–500+ (not this exp)",
        },
    }
    print(json.dumps(out, indent=2, default=str), flush=True)
    return out


@app.function(
    image=sim_image,
    gpu=DEFAULT_GPU,
    volumes={ASSETS_MOUNT: assets_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=SIM_TIMEOUT,
    memory=32768,
    cpu=4,
)
def smoke_random_fn(
    task: str = DEFAULT_TASK,
    steps: int = 80,
    seed: int = 7,
    split: str = "pretrain",
    gpu_label: str = DEFAULT_GPU,
    run_name: str = "",
) -> dict[str, Any]:
    """Random-policy rollout; always tries to write an mp4."""
    import numpy as np
    import imageio.v2 as imageio
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import run_random_rollouts

    t0 = time.time()
    if not _assets_ready():
        print("assets missing — downloading first…", flush=True)
        _download_assets_impl(force=False)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"random_{task}_{stamp}"
    out_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{task}_random.mp4"

    # Prefer EGL; fall back to osmesa if needed
    os.environ.setdefault("MUJOCO_GL", "egl")
    smi = _nvidia_smi()

    env = None
    err = None
    info: dict[str, Any] = {}
    try:
        env = gym.make(f"robocasa/{task}", split=split, seed=seed)
        info = run_random_rollouts(
            env,
            num_rollouts=1,
            num_steps=steps,
            video_path=str(video_path),
            camera_name="robot0_agentview_left",
        )
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        # fallback: manual loop with concatenated cameras if available
        try:
            if env is None:
                os.environ["MUJOCO_GL"] = "osmesa"
                env = gym.make(f"robocasa/{task}", split=split, seed=seed)
            obs, _ = env.reset(seed=seed)
            frames = []
            success = False
            for i in range(steps):
                action = env.action_space.sample()
                if "action.base_motion" in action:
                    action["action.base_motion"][:] = 0.0
                obs, reward, term, trunc, step_info = env.step(action)
                try:
                    frames.append(_make_video_frame(obs))
                except Exception:
                    # render via sim if gym obs lacks cameras
                    img = env.unwrapped.sim.render(
                        height=256, width=256, camera_name="robot0_agentview_left"
                    )[::-1]
                    frames.append(np.asarray(img, dtype=np.uint8))
                success = bool(step_info.get("success", False))
                if success or term or trunc:
                    break
            if frames:
                imageio.mimsave(str(video_path), frames, fps=20)
            info = {
                "num_success_rollouts": int(success),
                "fallback": "manual",
                "frames": len(frames),
                "first_error": err,
            }
            err = None
        except Exception as e2:  # noqa: BLE001
            err = f"{err} | fallback={e2!r}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    wall = round(time.time() - t0, 2)
    meta = {
        "success": err is None and video_path.is_file(),
        "mode": "random",
        "task": task,
        "steps": steps,
        "seed": seed,
        "split": split,
        "run_name": name,
        "video": str(video_path.relative_to(OUTPUTS_MOUNT)) if video_path.is_file() else None,
        "video_bytes": video_path.stat().st_size if video_path.is_file() else 0,
        "info": info,
        "error": err,
        "gpu": gpu_label,
        "vram": smi,
        "wall_s": wall,
        "cost_est_usd": _estimate_cost(gpu_label, wall),
        "utc": stamp,
        "note": "Random actions — not XR-1. Proves MuJoCo/RoboCasa can render a video on Modal.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    outputs_vol.commit()
    print(json.dumps(meta, indent=2, default=str), flush=True)
    return meta


@app.function(
    image=sim_image,
    gpu=DEFAULT_GPU,
    volumes={
        WEIGHTS_MOUNT: weights_vol,
        ASSETS_MOUNT: assets_vol,
        OUTPUTS_MOUNT: outputs_vol,
    },
    timeout=SIM_TIMEOUT,
    memory=49152,
    cpu=4,
)
def smoke_policy_fn(
    task: str = DEFAULT_TASK,
    horizon: int = 20,
    seed: int = 7,
    split: str = "pretrain",
    crop_ratio: float = 0.95,
    gpu_label: str = DEFAULT_GPU,
    run_name: str = "",
    attn: str = "sdpa",
    num_denoise_steps: int = 5,
) -> dict[str, Any]:
    """One-episode XR-1 closed-loop smoke with video."""
    import collections
    import numpy as np
    import imageio.v2 as imageio
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import convert_action

    t0 = time.time()
    if not _assets_ready():
        _download_assets_impl(force=False)
    if not _model_ready(MODEL_DIR):
        return {
            "success": False,
            "error": "weights missing — run download-weights first (or 015 download)",
            "weights": _dir_info(MODEL_DIR),
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"policy_{task}_h{horizon}_{stamp}"
    out_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    smi_before = _nvidia_smi()

    load_t0 = time.time()
    processor, model, attn_used = _load_policy(MODEL_DIR, attn=attn)
    load_s = round(time.time() - load_t0, 2)

    env = gym.make(f"robocasa/{task}", split=split, seed=seed)
    episode_seed = seed
    observation, _ = env.reset(seed=episode_seed)
    instruction = observation.get(
        "annotation.human.task_description",
        f"perform the task {task}",
    )

    queue_length = (OBS_HISTORY - 1) * OBS_INTERVAL + 1
    image_queues = {
        key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS
    }
    state_queue: collections.deque = collections.deque(maxlen=queue_length)

    def _push_obs(obs):
        for key in CAMERA_KEYS:
            if key in obs:
                image_queues[key].append(np.ascontiguousarray(obs[key], dtype=np.uint8))
        state_queue.append(_observation_to_state(obs))

    _push_obs(observation)
    video_frames = [_make_video_frame(observation)]
    action_plan: collections.deque = collections.deque()
    success = False
    steps = 0
    infer_times = []
    err = None

    try:
        while steps < horizon:
            if not action_plan:
                # pad histories if short
                for key in CAMERA_KEYS:
                    while len(image_queues[key]) < queue_length:
                        image_queues[key].appendleft(image_queues[key][0])
                while len(state_queue) < queue_length:
                    state_queue.appendleft(state_queue[0])

                states = _sample_history(state_queue, OBS_HISTORY, OBS_INTERVAL)
                images = {
                    key: _sample_history(image_queues[key], OBS_HISTORY, OBS_INTERVAL)
                    for key in CAMERA_KEYS
                }
                it0 = time.time()
                chunk = _infer_actions(
                    processor,
                    model,
                    states,
                    images,
                    instruction,
                    crop_ratio,
                    num_steps=num_denoise_steps,
                )
                infer_times.append(round(time.time() - it0, 3))
                for a in chunk[:REPLAN_STEPS]:
                    action_plan.append(a)

            policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
            # pad to 12 dims if needed
            if policy_action.shape[0] < 12:
                pad = np.zeros(12, dtype=np.float32)
                pad[: policy_action.shape[0]] = policy_action
                policy_action = pad
            observation, _, done, truncated, info = env.step(
                convert_action(policy_action[:12])
            )
            steps += 1
            _push_obs(observation)
            success = bool(info.get("success", False))
            video_frames.append(_make_video_frame(observation))
            if success or done or truncated:
                break
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    finally:
        try:
            env.close()
        except Exception:
            pass

    status = "success" if success else "failure"
    video_path = task_dir / f"episode_000_seed_{episode_seed}_{status}.mp4"
    if video_frames:
        imageio.mimsave(str(video_path), video_frames, fps=20)

    stats = {
        "env_name": task,
        "split": split,
        "num_episodes": 1,
        "successes": int(success),
        "success_rate": float(success),
        "horizon": horizon,
        "episodes": [
            {
                "episode": 0,
                "seed": episode_seed,
                "success": success,
                "steps": steps,
            }
        ],
    }
    (task_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    wall = round(time.time() - t0, 2)
    meta = {
        "success": err is None and video_path.is_file(),
        "mode": "policy",
        "task": task,
        "instruction": instruction if isinstance(instruction, str) else str(instruction),
        "horizon": horizon,
        "steps_run": steps,
        "episode_success": success,
        "seed": episode_seed,
        "split": split,
        "run_name": name,
        "attn": attn_used,
        "load_s": load_s,
        "infer_times_s": infer_times,
        "video": str(video_path.relative_to(OUTPUTS_MOUNT)) if video_path.is_file() else None,
        "video_frames": len(video_frames),
        "video_bytes": video_path.stat().st_size if video_path.is_file() else 0,
        "gpu": gpu_label,
        "vram_before": smi_before,
        "vram_after": _nvidia_smi(),
        "wall_s": wall,
        "cost_est_usd": _estimate_cost(gpu_label, wall),
        "error": err,
        "utc": stamp,
        "note": (
            "Single-episode closed-loop smoke with XR-1. "
            "horizon=20 is official smoke length — may be too short to finish the task. "
            "Video is the form of the result (plus stats.json)."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    outputs_vol.commit()
    print(json.dumps(meta, indent=2, default=str), flush=True)
    return meta


@app.local_entrypoint()
def main(
    action: str = "status",
    force: bool = False,
    gpu: str = DEFAULT_GPU,
    task: str = DEFAULT_TASK,
    steps: int = 80,
    horizon: int = 20,
    seed: int = 7,
    split: str = "pretrain",
    run_name: str = "",
    attn: str = "sdpa",
) -> None:
    action = action.strip().lower()
    if action == "status":
        print(status_fn.remote())
        return
    if action in {"download-weights", "download_weights"}:
        print(download_weights.remote(force=force))
        return
    if action in {"download-assets", "download_assets"}:
        print(download_assets.remote(force=force))
        return
    if action in {"smoke-random", "random"}:
        fn = smoke_random_fn
        if gpu != DEFAULT_GPU:
            fn = smoke_random_fn.with_options(gpu=gpu)
        r = fn.remote(
            task=task,
            steps=steps,
            seed=seed,
            split=split,
            gpu_label=gpu,
            run_name=run_name,
        )
        print(json.dumps(r, indent=2, default=str))
        if not r.get("success"):
            raise SystemExit(1)
        return
    if action in {"smoke-policy", "policy", "smoke"}:
        fn = smoke_policy_fn
        if gpu != DEFAULT_GPU:
            fn = smoke_policy_fn.with_options(gpu=gpu)
        r = fn.remote(
            task=task,
            horizon=horizon,
            seed=seed,
            split=split,
            gpu_label=gpu,
            run_name=run_name,
            attn=attn,
        )
        print(json.dumps(r, indent=2, default=str))
        if not r.get("success"):
            raise SystemExit(1)
        return
    raise SystemExit(
        "unknown action; use status|download-weights|download-assets|smoke-random|smoke-policy"
    )
