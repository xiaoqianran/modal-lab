# -*- coding: utf-8 -*-
"""
017-xr1-robocasa365-sim — RoboCasa365 simulation smoke on Modal.

Goals:
  1) kitchen assets (download once per container; optional volume mirror)
  2) random-policy rollout → episode .mp4
  3) XR-1 closed-loop episode (default horizon=100)

Full 2500-episode eval is intentionally out of scope.
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

DEFAULT_GPU = "L40S"
DEFAULT_TASK = "CloseBlenderLid"
# Longer than official smoke (20) so the arm has a real chance to finish.
DEFAULT_POLICY_HORIZON = 100
# Official target50 often uses task horizons 450–900+; CBL successes need ~236+ steps.
DEFAULT_EVAL_HORIZON = 200
DEFAULT_EVAL_LONG_HORIZON = 500
# 5 tasks × 5 seeds mini-eval (mix of easy + CBL)
MINI_EVAL_TASKS = (
    "OpenStandMixerHead",      # validated 5/5 @ h=200
    "TurnOnElectricKettle",    # validated 3/5
    "OpenDrawer",              # high official SR
    "TurnOnMicrowave",         # mid official SR
    "CloseBlenderLid",         # hard; long track
)
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60
ACTION_DIM = 12
OBS_HISTORY = 4
OBS_INTERVAL = 2
REPLAN_STEPS = 16
POLICY_IMAGE_SIZE = (320, 256)  # W,H multiples of 32
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)

WEIGHTS_MOUNT = "/weights"
ASSETS_MOUNT = "/assets"
OUTPUTS_MOUNT = "/outputs"
MODEL_DIR = Path(WEIGHTS_MOUNT) / "Xiaomi-Robotics-1-RoboCasa365"
# Volume keeps a marker + optional partial cache. Full ~20GB tree copy is
# optional (CACHE_FULL_ASSETS) because shutil.copytree of that size is very slow.
ASSETS_CACHE_DIR = Path(ASSETS_MOUNT) / "models_assets"
ASSETS_VOL_DIR = Path(ASSETS_MOUNT) / "kitchen_assets_marker"
VOLUME_WEIGHTS = "modal-lab-xr1-robocasa365-weights"
VOLUME_ASSETS = "modal-lab-xr1-robocasa365-sim-assets"
VOLUME_OUTPUTS = "modal-lab-xr1-robocasa365-sim-outputs"
# Set True only when you intentionally want multi-GB volume mirror.
CACHE_FULL_ASSETS = os.environ.get("CACHE_FULL_ASSETS", "0") == "1"

DOWNLOAD_TIMEOUT = 3 * 60 * 60
SIM_TIMEOUT = 5 * 60 * 60

GPU_PRICE_PER_SEC = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H100!": 0.001097,
    "RTX-PRO-6000": 0.000842,
}

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
assets_vol = modal.Volume.from_name(VOLUME_ASSETS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

status_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.26.0")
    .env({"PYTHONUNBUFFERED": "1"})
)

download_weights_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.26.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)

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
        "transformers==4.57.1",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]>=0.26.0",
        "numpy==2.2.5",
        "scipy",
        "Pillow",
        "tqdm",
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "opencv-python-headless",
        "gymnasium",
        "mujoco==3.3.1",
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
        "cd /opt/robosuite && pip install -e . --no-deps --no-cache-dir",
        "pip install --no-cache-dir 'numpy==2.2.5' 'mujoco==3.3.1' 'scipy' 'Pillow' 'opencv-python-headless' 'numba' 'termcolor' 'gymnasium'",
        "git clone --depth 1 https://github.com/robocasa/robocasa.git /opt/robocasa",
        "cd /opt/robocasa && pip install -e . --no-deps --no-cache-dir",
        "python -c \"import shutil, os; "
        "bp='/opt/robocasa/robocasa'; "
        "src=os.path.join(bp,'macros.py'); "
        "dst=os.path.join(bp,'macros_private.py'); "
        "shutil.copyfile(src,dst); print('macros_private ok', dst)\"",
        "python -c \"import mujoco; assert mujoco.__version__=='3.3.1', mujoco.__version__; "
        "import robosuite, robocasa, gymnasium; print('mujoco', mujoco.__version__); "
        "print('robocasa', robocasa.__path__)\"",
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
    # Fast path via du when available (full rglob on 20GB is painful).
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True, timeout=120)
        total = int(out.split()[0])
        return {
            "exists": True,
            "path": str(path),
            "size_gb": round(total / 1e9, 3),
            "via": "du",
        }
    except Exception:
        pass
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


def _assets_tree_ready(base: Path) -> bool:
    try:
        tex = base / "textures"
        fix = base / "fixtures"
        if not tex.is_dir() or not fix.is_dir():
            return False
        # cheap non-empty check
        return any(tex.iterdir()) and any(fix.iterdir())
    except Exception:
        return False


def _robocasa_assets_base() -> Path:
    import robocasa

    return Path(robocasa.__path__[0]) / "models" / "assets"


def _link_package_assets_to_cache() -> str | None:
    cache = ASSETS_CACHE_DIR
    if not _assets_tree_ready(cache):
        return None
    base = _robocasa_assets_base()
    parent = base.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        if base.is_symlink():
            if base.resolve() == cache.resolve():
                return "volume_symlink"
            base.unlink()
        elif base.exists():
            if _assets_tree_ready(base):
                return "package"
            shutil.rmtree(base)
        base.symlink_to(cache)
        if _assets_tree_ready(base):
            print(f"linked package assets → {cache}", flush=True)
            return "volume_symlink"
    except Exception as e:  # noqa: BLE001
        print(f"symlink assets failed: {e!r}", flush=True)
    return None


def _write_assets_marker(extra: dict[str, Any] | None = None) -> None:
    ASSETS_VOL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ready_package_hint": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "full_cache": CACHE_FULL_ASSETS,
    }
    if extra:
        payload.update(extra)
    (ASSETS_VOL_DIR / "ready.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    assets_vol.commit()


def _cache_package_assets_to_volume() -> dict[str, Any]:
    """Optional full mirror. Default OFF — 20GB copytree was hanging cold starts."""
    if not CACHE_FULL_ASSETS:
        _write_assets_marker({"note": "full cache disabled; package assets used in-container"})
        return {"ok": True, "skipped": True, "reason": "CACHE_FULL_ASSETS=0"}

    base = _robocasa_assets_base()
    cache = ASSETS_CACHE_DIR
    t0 = time.time()
    if not _assets_tree_ready(base):
        return {"ok": False, "error": "package assets not ready"}
    print("caching full assets tree to volume (slow)...", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        shutil.rmtree(cache)
    # cp -a is usually faster / more robust than pure shutil for huge trees
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", "-a", f"{base}/.", str(cache)])
    info = {
        "ok": True,
        "wall_s": round(time.time() - t0, 2),
        "cache": _dir_info(cache),
    }
    _write_assets_marker({"cache": info})
    print(f"cache done in {info['wall_s']}s", flush=True)
    return info


def _assets_ready() -> bool:
    try:
        if _assets_tree_ready(_robocasa_assets_base()):
            return True
    except Exception:
        pass
    return _link_package_assets_to_cache() is not None


def _nvidia_smi() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).strip()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 4:
        return {"raw": out}
    return {
        "name": parts[0],
        "mem_used_mib": float(parts[1]),
        "mem_total_mib": float(parts[2]),
        "util_gpu_pct": float(parts[3]),
    }


def _download_assets_impl(force: bool = False) -> dict[str, Any]:
    t0 = time.time()
    import robocasa  # noqa: F401
    from robocasa.scripts.download_kitchen_assets import (
        DOWNLOAD_ASSET_REGISTRY,
        download_and_extract_zip,
    )

    base = _robocasa_assets_base()

    if not force:
        linked = _link_package_assets_to_cache()
        if linked or _assets_tree_ready(base):
            info = {
                "skipped": True,
                "reason": f"assets ready via {linked or 'package'}",
                "source": linked or "package",
                "assets": _dir_info(base),
                "wall_s": round(time.time() - t0, 2),
                "ready": True,
            }
            print(json.dumps(info, indent=2, default=str), flush=True)
            return info

    results: dict[str, Any] = {}
    for name, config in DOWNLOAD_ASSET_REGISTRY.items():
        cfg = dict(config)
        cfg["prompt_before_download"] = False
        cfg["check_folder_exists"] = False
        cfg["delete_old_folder"] = bool(force)
        print(f"==> downloading asset pack: {name}", flush=True)
        try:
            download_and_extract_zip(**cfg)
            results[name] = "ok"
        except TypeError:
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

    print("downloads finished; writing marker (full volume mirror optional)", flush=True)
    cache_info = _cache_package_assets_to_volume()
    info = {
        "skipped": False,
        "results": results,
        "assets": _dir_info(base),
        "cache_write": cache_info,
        "ready": _assets_tree_ready(base),
        "wall_s": round(time.time() - t0, 2),
        "source": "download",
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


def _prepare_frame(image, crop_ratio: float = 1.0, size: tuple[int, int] = POLICY_IMAGE_SIZE):
    from PIL import Image
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if crop_ratio < 1.0:
        w, h = pil.size
        cw = max(1, int(w * crop_ratio))
        ch = max(1, int(h * crop_ratio))
        left = (w - cw) // 2
        top = (h - ch) // 2
        pil = pil.crop((left, top, left + cw, top + ch))
    tw, th = size
    if pil.size != (tw, th):
        pil = pil.resize((tw, th), Image.Resampling.BICUBIC)
    return pil


def _build_messages(image_history: dict, instruction: str, crop_ratio: float):
    videos = {
        key: [_prepare_frame(frame, crop_ratio) for frame in image_history[key]]
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
                runs.append({"name": p.name, "meta": m})
    info = {
        "app": APP_NAME,
        "model_ready": _model_ready(MODEL_DIR),
        "weights": _dir_info(MODEL_DIR),
        "assets_cache_ready": _assets_tree_ready(ASSETS_CACHE_DIR),
        "runs": runs[-20:],
        "default_gpu": DEFAULT_GPU,
        "default_task": DEFAULT_TASK,
        "default_policy_horizon": DEFAULT_POLICY_HORIZON,
        "policy_image_size_wh": list(POLICY_IMAGE_SIZE),
    }
    print(json.dumps(info, indent=2, default=str), flush=True)
    return info


@app.function(
    image=sim_image,
    gpu=DEFAULT_GPU,
    volumes={
        ASSETS_MOUNT: assets_vol,
        OUTPUTS_MOUNT: outputs_vol,
    },
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
    import numpy as np
    import imageio.v2 as imageio
    import gymnasium as gym
    import robocasa  # noqa: F401

    t0 = time.time()
    assets_info = _download_assets_impl(force=False)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"random_{task}_{stamp}"
    out_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    env = gym.make(f"robocasa/{task}", split=split, seed=seed)
    obs, _ = env.reset(seed=seed)
    video_path = out_dir / f"{task}_random.mp4"
    info: dict[str, Any] = {"num_success_rollouts": 0}
    err = None
    try:
        frames = []
        for _i in range(steps):
            action = env.action_space.sample()
            obs, _, done, truncated, step_info = env.step(action)
            try:
                frames.append(_make_video_frame(obs))
            except Exception:
                try:
                    unwrapped = env
                    while hasattr(unwrapped, "env"):
                        unwrapped = unwrapped.env
                    sim = getattr(unwrapped, "sim", None)
                    if sim is not None:
                        img = sim.render(
                            height=256, width=256, camera_name="robot0_agentview_left"
                        )
                        frames.append(np.asarray(img, dtype=np.uint8))
                except Exception:
                    pass
            if step_info.get("success"):
                info["num_success_rollouts"] = 1
            if done or truncated:
                break
        if frames:
            imageio.mimsave(str(video_path), frames, fps=20)
        info.update({"fallback": "manual", "frames": len(frames)})
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    finally:
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
        "assets": assets_info.get("source") or assets_info.get("reason"),
        "error": err,
        "gpu": gpu_label,
        "vram": _nvidia_smi(),
        "wall_s": wall,
        "cost_est_usd": _estimate_cost(gpu_label, wall),
        "utc": stamp,
        "note": "Random actions — not XR-1.",
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
    horizon: int = DEFAULT_POLICY_HORIZON,
    seed: int = 7,
    split: str = "pretrain",
    crop_ratio: float = 0.95,
    gpu_label: str = DEFAULT_GPU,
    run_name: str = "",
    attn: str = "sdpa",
    num_denoise_steps: int = 5,
) -> dict[str, Any]:
    """XR-1 closed-loop episode with video (default horizon=100)."""
    import collections
    import numpy as np
    import imageio.v2 as imageio
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import convert_action

    t0 = time.time()
    print(f"==> ensuring assets (horizon={horizon})", flush=True)
    assets_info = _download_assets_impl(force=False)
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

    print("==> loading XR-1 policy", flush=True)
    load_t0 = time.time()
    processor, model, attn_used = _load_policy(MODEL_DIR, attn=attn)
    load_s = round(time.time() - load_t0, 2)
    print(f"policy loaded in {load_s}s attn={attn_used}", flush=True)

    env = gym.make(f"robocasa/{task}", split=split, seed=seed)
    episode_seed = seed
    observation, _ = env.reset(seed=episode_seed)
    instruction = observation.get(
        "annotation.human.task_description",
        f"perform the task {task}",
    )
    print(f"task instruction: {instruction}", flush=True)
    cam_shapes = {
        k: list(np.asarray(observation[k]).shape)
        for k in CAMERA_KEYS
        if k in observation
    }

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
    first_action_preview = None
    success_step = None

    try:
        while steps < horizon:
            if not action_plan:
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
                dt = round(time.time() - it0, 3)
                infer_times.append(dt)
                print(f"step {steps}: replan infer {dt}s plan={len(chunk[:REPLAN_STEPS])}", flush=True)
                if first_action_preview is None:
                    first_action_preview = chunk[0].tolist()
                for a in chunk[:REPLAN_STEPS]:
                    action_plan.append(a)

            policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
            if policy_action.shape[0] < 12:
                pad = np.zeros(12, dtype=np.float32)
                pad[: policy_action.shape[0]] = policy_action
                policy_action = pad
            observation, _, done, truncated, info = env.step(
                convert_action(policy_action[:12])
            )
            steps += 1
            _push_obs(observation)
            if bool(info.get("success", False)):
                success = True
                success_step = steps
                print(f"SUCCESS at step {steps}", flush=True)
            video_frames.append(_make_video_frame(observation))
            if steps % 10 == 0:
                print(f"progress {steps}/{horizon} success={success}", flush=True)
            if success or done or truncated:
                break
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        print(f"loop error: {err}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    status = "success" if success else "failure"
    video_path = task_dir / f"episode_000_seed_{episode_seed}_{status}.mp4"
    if video_frames:
        print(f"writing video frames={len(video_frames)} → {video_path}", flush=True)
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
                "success_step": success_step,
            }
        ],
    }
    (task_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    wall = round(time.time() - t0, 2)
    ran_ok = err is None and video_path.is_file() and len(video_frames) > 1
    infer_sum = round(sum(infer_times), 3) if infer_times else 0.0
    meta = {
        "success": ran_ok,
        "mode": "policy",
        "task": task,
        "instruction": instruction if isinstance(instruction, str) else str(instruction),
        "horizon": horizon,
        "steps_run": steps,
        "episode_success": success,
        "success_step": success_step,
        "seed": episode_seed,
        "split": split,
        "run_name": name,
        "attn": attn_used,
        "policy_image_size_wh": list(POLICY_IMAGE_SIZE),
        "cam_shapes_raw": cam_shapes,
        "assets_source": assets_info.get("source") or assets_info.get("reason"),
        "assets_skipped": assets_info.get("skipped"),
        "load_s": load_s,
        "infer_times_s": infer_times,
        "infer_total_s": infer_sum,
        "infer_count": len(infer_times),
        "first_action_preview": first_action_preview,
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
            f"Proper closed-loop demo: horizon={horizon} on {gpu_label}. "
            "Official smoke is 20; longer horizon gives the arm time to act. "
            "Frames resized to 320×256 after crop."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    outputs_vol.commit()
    print(json.dumps(meta, indent=2, default=str), flush=True)
    return meta



def _run_policy_episode(
    *,
    env,
    processor,
    model,
    task: str,
    instruction: str | None,
    horizon: int,
    episode_seed: int,
    crop_ratio: float,
    num_denoise_steps: int,
    save_video_path: Path | None,
    replan_steps: int = REPLAN_STEPS,
) -> dict[str, Any]:
    """One closed-loop episode. Caller owns env create/close and model load."""
    import collections
    import numpy as np
    import imageio.v2 as imageio

    observation, _ = env.reset(seed=episode_seed)
    if instruction is None:
        instruction = observation.get(
            "annotation.human.task_description",
            f"perform the task {task}",
        )
    if not isinstance(instruction, str):
        instruction = str(instruction)

    queue_length = (OBS_HISTORY - 1) * OBS_INTERVAL + 1
    image_queues = {key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS}
    state_queue: collections.deque = collections.deque(maxlen=queue_length)

    def _push_obs(obs):
        for key in CAMERA_KEYS:
            if key in obs:
                image_queues[key].append(np.ascontiguousarray(obs[key], dtype=np.uint8))
        state_queue.append(_observation_to_state(obs))

    _push_obs(observation)
    video_frames = [_make_video_frame(observation)] if save_video_path is not None else []
    action_plan: collections.deque = collections.deque()
    success = False
    steps = 0
    infer_times: list[float] = []
    err = None
    success_step = None

    try:
        while steps < horizon:
            if not action_plan:
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
                for a in chunk[:replan_steps]:
                    action_plan.append(a)

            policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
            if policy_action.shape[0] < 12:
                pad = np.zeros(12, dtype=np.float32)
                pad[: policy_action.shape[0]] = policy_action
                policy_action = pad

            from robocasa.utils.env_utils import convert_action

            observation, _, done, truncated, info = env.step(
                convert_action(policy_action[:12])
            )
            steps += 1
            _push_obs(observation)
            if bool(info.get("success", False)):
                success = True
                success_step = steps
            if save_video_path is not None:
                video_frames.append(_make_video_frame(observation))
            if success or done or truncated:
                break
    except Exception as e:  # noqa: BLE001
        err = repr(e)

    video_bytes = 0
    if save_video_path is not None and video_frames:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(save_video_path), video_frames, fps=20)
        video_bytes = save_video_path.stat().st_size if save_video_path.is_file() else 0

    return {
        "task": task,
        "seed": episode_seed,
        "instruction": instruction,
        "horizon": horizon,
        "steps": steps,
        "success": success,
        "success_step": success_step,
        "error": err,
        "infer_times_s": infer_times,
        "infer_total_s": round(sum(infer_times), 3) if infer_times else 0.0,
        "infer_count": len(infer_times),
        "video_frames": len(video_frames),
        "video_bytes": video_bytes,
        "video": str(save_video_path.relative_to(OUTPUTS_MOUNT))
        if save_video_path is not None and save_video_path.is_file()
        else None,
        "pipeline_ok": err is None,
    }



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
def eval_mini_fn(
    tasks_csv: str = "",
    num_seeds: int = 5,
    base_seed: int = 7,
    horizon: int = DEFAULT_EVAL_HORIZON,
    long_horizon: int = DEFAULT_EVAL_LONG_HORIZON,
    long_task: str = "CloseBlenderLid",
    run_long_track: bool = True,
    split: str = "pretrain",
    crop_ratio: float = 0.95,
    gpu_label: str = DEFAULT_GPU,
    run_name: str = "",
    attn: str = "sdpa",
    num_denoise_steps: int = 5,
    save_every_video: bool = True,
) -> dict[str, Any]:
    """5×N mini-eval + optional long-horizon track for task success.

    1) grid: tasks × seeds @ horizon (default 200)
    2) long: long_task × seeds @ long_horizon (default CBL×5 @ 500)
       Official CBL successes need ~236–870 steps; 200 is often too short.
    """
    import gymnasium as gym
    import robocasa  # noqa: F401

    t0 = time.time()
    tasks = [t.strip() for t in (tasks_csv or ",".join(MINI_EVAL_TASKS)).split(",") if t.strip()]
    seeds = [base_seed + i for i in range(num_seeds)]

    print(f"==> eval_mini tasks={tasks} seeds={seeds} h={horizon} long_h={long_horizon}", flush=True)
    assets_info = _download_assets_impl(force=False)
    if not _model_ready(MODEL_DIR):
        return {"success": False, "error": "weights missing"}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"eval_mini_h{horizon}_{stamp}"
    out_dir = Path(OUTPUTS_MOUNT) / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    smi_before = _nvidia_smi()
    processor, model, attn_used = _load_policy(MODEL_DIR, attn=attn)
    load_s = round(time.time() - t0, 2)  # includes assets; refined below
    # re-time load only approximately
    load_s = round(time.time() - t0, 2)

    episodes: list[dict[str, Any]] = []
    task_stats: dict[str, Any] = {}

    def _run_grid(task_list: list[str], h: int, track: str):
        nonlocal episodes
        for task in task_list:
            task_dir = out_dir / track / task
            task_dir.mkdir(parents=True, exist_ok=True)
            successes = 0
            task_eps = []
            for ep_i, seed in enumerate(seeds):
                print(f"==> [{track}] {task} seed={seed} h={h}", flush=True)
                env = None
                try:
                    env = gym.make(f"robocasa/{task}", split=split, seed=seed)
                    vid = None
                    if save_every_video or ep_i == 0:
                        vid = task_dir / f"episode_{ep_i:03d}_seed_{seed}.mp4"
                    ep = _run_policy_episode(
                        env=env,
                        processor=processor,
                        model=model,
                        task=task,
                        instruction=None,
                        horizon=h,
                        episode_seed=seed,
                        crop_ratio=crop_ratio,
                        num_denoise_steps=num_denoise_steps,
                        save_video_path=vid,
                    )
                    ep["track"] = track
                    ep["episode_index"] = ep_i
                    if ep.get("success"):
                        successes += 1
                        print(f"   SUCCESS step={ep.get('success_step')}", flush=True)
                    else:
                        print(
                            f"   fail steps={ep.get('steps')} err={ep.get('error')}",
                            flush=True,
                        )
                    task_eps.append(ep)
                    episodes.append(ep)
                    # intermediate commit so partial results survive
                    if (len(episodes) % 3) == 0:
                        outputs_vol.commit()
                except Exception as e:  # noqa: BLE001
                    ep = {
                        "task": task,
                        "seed": seed,
                        "track": track,
                        "horizon": h,
                        "success": False,
                        "pipeline_ok": False,
                        "error": repr(e),
                        "steps": 0,
                    }
                    task_eps.append(ep)
                    episodes.append(ep)
                    print(f"   EXCEPTION {e!r}", flush=True)
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
            n = len(task_eps)
            task_stats[f"{track}/{task}"] = {
                "task": task,
                "track": track,
                "horizon": h,
                "num_episodes": n,
                "successes": successes,
                "success_rate": round(successes / n, 4) if n else 0.0,
                "episodes": task_eps,
            }

    _run_grid(tasks, horizon, "grid_h" + str(horizon))
    if run_long_track and long_task:
        _run_grid([long_task], long_horizon, "long_h" + str(long_horizon))

    # aggregate
    grid_eps = [e for e in episodes if str(e.get("track", "")).startswith("grid_")]
    long_eps = [e for e in episodes if str(e.get("track", "")).startswith("long_")]
    def _sr(eps):
        ok = sum(1 for e in eps if e.get("success"))
        n = len(eps)
        return ok, n, round(ok / n, 4) if n else 0.0

    g_ok, g_n, g_sr = _sr(grid_eps)
    l_ok, l_n, l_sr = _sr(long_eps)
    a_ok, a_n, a_sr = _sr(episodes)

    wall = round(time.time() - t0, 2)
    summary = {
        "success": True,
        "mode": "eval_mini",
        "run_name": name,
        "gpu": gpu_label,
        "attn": attn_used,
        "split": split,
        "tasks": tasks,
        "seeds": seeds,
        "horizon_grid": horizon,
        "horizon_long": long_horizon if run_long_track else None,
        "long_task": long_task if run_long_track else None,
        "assets_source": assets_info.get("source") or assets_info.get("reason"),
        "grid": {"successes": g_ok, "episodes": g_n, "success_rate": g_sr},
        "long": {"successes": l_ok, "episodes": l_n, "success_rate": l_sr},
        "overall": {"successes": a_ok, "episodes": a_n, "success_rate": a_sr},
        "per_task": task_stats,
        "episodes": episodes,
        "vram_before": smi_before,
        "vram_after": _nvidia_smi(),
        "wall_s": wall,
        "cost_est_usd": _estimate_cost(gpu_label, wall),
        "utc": stamp,
        "note": (
            f"Mini-eval {len(tasks)}×{num_seeds} @ h={horizon}; "
            f"long track {long_task}×{num_seeds} @ h={long_horizon}. "
            "Official CBL horizon=900; successes typically ≥236 steps."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    # compact table for README
    lines = [
        f"# eval_mini {name}",
        f"GPU={gpu_label} wall={wall}s cost≈${summary['cost_est_usd']}",
        f"GRID h={horizon}: {g_ok}/{g_n} = {g_sr:.1%}",
        f"LONG h={long_horizon}: {l_ok}/{l_n} = {l_sr:.1%}",
        "",
        "| track/task | h | succ | n | SR |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in task_stats.items():
        lines.append(
            f"| {k} | {v['horizon']} | {v['successes']} | {v['num_episodes']} | {v['success_rate']:.1%} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs_vol.commit()
    print(json.dumps({k: summary[k] for k in summary if k not in ('episodes','per_task')}, indent=2), flush=True)
    print((out_dir / "SUMMARY.md").read_text(), flush=True)
    return summary



@app.local_entrypoint()
def main(
    action: str = "status",
    force: bool = False,
    gpu: str = DEFAULT_GPU,
    task: str = DEFAULT_TASK,
    steps: int = 80,
    horizon: int = DEFAULT_POLICY_HORIZON,
    seed: int = 7,
    split: str = "pretrain",
    run_name: str = "",
    attn: str = "sdpa",
    tasks_csv: str = "",
    num_seeds: int = 5,
    long_horizon: int = DEFAULT_EVAL_LONG_HORIZON,
    long_task: str = "CloseBlenderLid",
    run_long_track: bool = True,
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
    if action in {"eval-mini", "eval_mini", "mini"}:
        fn = eval_mini_fn
        if gpu != DEFAULT_GPU:
            fn = eval_mini_fn.with_options(gpu=gpu)
        r = fn.remote(
            tasks_csv=tasks_csv,
            num_seeds=num_seeds,
            base_seed=seed,
            horizon=horizon,
            long_horizon=long_horizon,
            long_task=long_task,
            run_long_track=run_long_track,
            split=split,
            gpu_label=gpu,
            run_name=run_name,
            attn=attn,
        )
        print(json.dumps({k: r.get(k) for k in ("success","grid","long","overall","wall_s","cost_est_usd","run_name") if isinstance(r, dict)}, indent=2, default=str))
        if not (isinstance(r, dict) and r.get("success")):
            raise SystemExit(1)
        return
    raise SystemExit(
        "unknown action; use status|download-weights|download-assets|smoke-random|smoke-policy|eval-mini"
    )
