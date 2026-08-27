# -*- coding: utf-8 -*-
"""
009-hy-worldgen — single-GPU (RTX-PRO-6000) World Generation pipeline. v8.2.2

Stage 1–2 use official Qwen3-VL-8B via in-container vLLM (share-GPU or split).
Stage 3–5 unchanged (WorldStereo-dmd → GS).
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

APP_NAME = "modal-lab-hy-worldgen"
UPSTREAM = "https://github.com/Tencent-Hunyuan/HY-World-2.0"
WORLDSTEREO_HF = "hanshanxue/WorldStereo"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VLM_PORT = 8000
VLM_HOST = "127.0.0.1"
VLM_DEFAULT_MEM_UTIL = 0.38  # leave room for MoGe/SAM on same 96GB card
VLM_MAX_MODEL_LEN = 8192

VOLUME_WEIGHTS = "modal-lab-hy-worldgen-weights"
VOLUME_OUTPUTS = "modal-lab-hy-worldgen-outputs"
VOLUME_PANO_OUT = "modal-lab-hy-pano-outputs"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PANO_MOUNT = "/pano_out"

DEFAULT_GPU = "RTX-PRO-6000"
GPU_PRICE = {
    "RTX-PRO-6000": 0.000842,
    "H100": 0.001097,
    "A100-80GB": 0.000694,
    "L40S": 0.000542,
}

REPO_DIR = Path("/opt/HY-World-2.0")
WORLDGEN = REPO_DIR / "hyworld2" / "worldgen"
HF_HOME = Path(WEIGHTS_MOUNT) / "huggingface"

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
pano_vol = modal.Volume.from_name(VOLUME_PANO_OUT, create_if_missing=True)

app = modal.App(APP_NAME)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.34.0", "pillow")
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

worldgen_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git", "git-lfs", "ffmpeg", "libgl1", "libglib2.0-0", "libgomp1",
        "libsm6", "libxext6", "libxrender1", "wget", "curl", "ca-certificates",
        "build-essential", "cmake", "ninja-build", "python3-dev",
    )
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        # skip full pytorch3d CUDA compile (20–40min). Stage1/2 use pure-torch look_at stub.
        "pip install iopath",
    )
    .pip_install(
        "numpy==1.26.4", "pillow", "opencv-python==4.10.0.84", "imageio[ffmpeg]",
        "decord", "imagesize", "matplotlib==3.10.3", "scikit-image==0.25.2",
        "scipy==1.14.1", "tqdm", "loguru==0.7.3", "einops", "omegaconf",
        "easydict", "kornia", "timm==1.0.11", "ftfy", "regex", "openai",
        "trimesh", "plyfile", "safetensors", "accelerate", "peft==0.18.1",
        "diffusers==0.36.0", "transformers==5.2.0",
        "huggingface_hub[hf_transfer]>=0.34.0", "sentencepiece", "protobuf",
        "torchmetrics", "tyro==0.9.14", "splines", "viser", "cupy-cuda12x==13.6.0",
    )
    .pip_install("open3d==0.18.0")
    .run_commands(
        "pip install --no-deps 'git+https://github.com/EasternJournalist/utils3d.git@v0.0.2' || "
        "pip install --no-deps 'git+https://github.com/EasternJournalist/utils3d.git' || true",
        "pip install --no-build-isolation 'git+https://github.com/microsoft/MoGe.git@0286b495230a074aadf1c76cc5c679e943e5d1c6'",
    )
    .run_commands(
        f"git clone --depth 1 {UPSTREAM} {REPO_DIR} && "
        f"cd {REPO_DIR} && git submodule update --init --recursive || true",
    )
    .run_commands(
        "pip install pybind11 scikit-build-core nanobind",
        f"cd {REPO_DIR}/hyworld2/worldgen/third_party && "
        f"(test -d recastnavigation || git clone --depth 1 https://github.com/recastnavigation/recastnavigation.git) && "
        f"export RECAST_PATH={REPO_DIR}/hyworld2/worldgen/third_party/recastnavigation && "
        f"export CXX=g++ && export CC=gcc && "
        f"cd navmesh && pip install . --no-build-isolation",
    )
    .run_commands(
        f"cd {REPO_DIR}/hyworld2/worldgen/third_party/gsplat_maskgaussian && "
        f"(pip install -e . --no-build-isolation || pip install gsplat || true)",
        "pip install zim_anything || true",
        "pip install 'transformers==5.2.0' 'huggingface_hub[hf_transfer]>=0.34.0' --upgrade",
        "pip install openai fastapi uvicorn pillow",
        # Re-pin torch after any package that may have upgraded it (vLLM etc.)
        "pip install --force-reinstall 'numpy==1.26.4' "
        "'torch==2.7.1' 'torchvision==0.22.1' "
        "--index-url https://download.pytorch.org/whl/cu128 "
        "|| pip install 'numpy==1.26.4'",
    )
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "HUGGINGFACE_HUB_CACHE": str(HF_HOME),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{WORLDGEN}:{REPO_DIR}",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_HOME": str(Path(WEIGHTS_MOUNT) / "torch"),
        }
    )
)


def _price(gpu: str, seconds: float) -> float:
    rate = GPU_PRICE.get(gpu, GPU_PRICE[DEFAULT_GPU])
    return round(rate * seconds, 4)


def _scene_dir(scene: str) -> Path:
    return Path(OUTPUTS_MOUNT) / "scenes" / scene


def _write_meta(scene: str, payload: dict) -> None:
    d = _scene_dir(scene)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "run_log.jsonl").open("a") as f:
        f.write(json.dumps(payload) + "\n")
    stage = payload.get("stage")
    if stage is not None:
        (d / f"stage{stage}_meta.json").write_text(json.dumps(payload, indent=2) + "\n")


def _seed_minimal_scene(scene: str, from_008: str = "smoke_qwen") -> Path:
    """Seed scene dir with panorama from 008 volume, existing scene, or official case000."""
    scene_path = _scene_dir(scene)
    scene_path.mkdir(parents=True, exist_ok=True)
    pano = scene_path / "panorama.png"
    if not pano.is_file():
        candidates = [
            Path(PANO_MOUNT) / "runs" / from_008 / "panorama.png",
            Path(PANO_MOUNT) / "runs" / from_008 / from_008 / "panorama.png",
        ]
        src = next((p for p in candidates if p.is_file()), None)
        if src is not None:
            shutil.copy2(src, pano)
            print(f"[seed] panorama from 008 {src}")
        else:
            # Fallback: official HY-World-2.0 example (no 008 dependency)
            url = (
                "https://raw.githubusercontent.com/Tencent-Hunyuan/HY-World-2.0/"
                "main/examples/worldgen/case000/panorama.png"
            )
            print(f"[seed] 008 run '{from_008}' missing; downloading official case000…")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, pano)
            except Exception as e:
                runs = Path(PANO_MOUNT) / "runs"
                avail = sorted(p.name for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
                raise FileNotFoundError(
                    f"no panorama for {from_008}; available={avail}; official download failed: {e}"
                ) from e
            if not pano.is_file() or pano.stat().st_size < 1000:
                raise FileNotFoundError(f"official panorama download looks empty: {pano}")
            print(f"[seed] official case000 → {pano} ({pano.stat().st_size} bytes)")
    meta = scene_path / "meta_info.json"
    if not meta.is_file():
        meta.write_text(json.dumps({"scene_type": "indoor", "scene": scene}, indent=2))
    return scene_path


def _ensure_vllm() -> None:
    """Install vLLM at runtime if the image layer missed it."""
    import sys
    try:
        import vllm  # noqa: F401
        print(f"[vlm] vllm ok")
        return
    except Exception as e:
        print(f"[vlm] importing vllm failed ({e}); pip install…")
    env = os.environ.copy()
    env["CXX"] = "g++"
    env["CC"] = "gcc"
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "vllm", "openai"],
        env=env,
    )
    print("[vlm] vllm installed")



def _write_hf_vlm_server(path: Path) -> None:
    """Minimal OpenAI-compatible server for Qwen3-VL (chat.completions + image_url)."""
    path.write_text(r"""#!/usr/bin/env python3
import base64, io, os, traceback
from typing import Any
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

MODEL_ID = os.environ.get("VLM_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
SERVED = os.environ.get("VLM_SERVED_NAME", "Qwen/Qwen3-VL-8B-Instruct")
PORT = int(os.environ.get("VLM_PORT", "8000"))
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"[hf-vlm] loading {MODEL_ID} dtype={DTYPE}", flush=True)
processor = None
model = None

def load():
    global processor, model
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as M
    except Exception:
        from transformers import AutoModelForImageTextToText as M
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = M.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print("[hf-vlm] ready", flush=True)

load()
app = FastAPI()

class ChatReq(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    temperature: float = 0.0
    seed: int | None = None

def _to_qwen_messages(messages):
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    parts.append({"type": "text", "text": c.get("text", "")})
                elif c.get("type") == "image_url":
                    url = (c.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        b64 = url.split(",", 1)[-1]
                        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                        parts.append({"type": "image", "image": img})
                    else:
                        parts.append({"type": "image", "image": url})
        out.append({"role": role, "content": parts or [{"type": "text", "text": ""}]})
    return out

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED, "object": "model", "owned_by": "modal-lab"}]}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    try:
        if req.seed is not None:
            torch.manual_seed(int(req.seed))
        qmsgs = _to_qwen_messages(req.messages)
        # processor.apply_chat_template path
        text = processor.apply_chat_template(qmsgs, tokenize=False, add_generation_prompt=True)
        images = []
        for m in qmsgs:
            for c in m.get("content", []):
                if c.get("type") == "image" and hasattr(c.get("image"), "size"):
                    images.append(c["image"])
        inputs = processor(text=[text], images=images or None, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=int(req.max_tokens), do_sample=False)
        # decode only new tokens
        trim = out[:, inputs["input_ids"].shape[-1]:]
        content = processor.batch_decode(trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {
            "id": "chatcmpl-modal",
            "object": "chat.completion",
            "model": SERVED,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "id": "chatcmpl-err",
            "object": "chat.completion",
            "model": SERVED,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"ERROR: {e}"}, "finish_reason": "stop"}],
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
""")
    print(f"[vlm] wrote HF server → {path}")


def _vlm_model_path() -> str:
    """Prefer volume snapshot; fall back to HF id."""
    local = Path(WEIGHTS_MOUNT) / "Qwen3-VL-8B-Instruct"
    if local.is_dir() and any(local.iterdir()):
        # accept either flat or nested hub layout
        if (local / "config.json").is_file():
            return str(local)
        for p in local.rglob("config.json"):
            return str(p.parent)
    return VLM_MODEL


def _vlm_ready(timeout_s: float = 5.0) -> bool:
    import urllib.request
    url = f"http://{VLM_HOST}:{VLM_PORT}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def _wait_vlm(
    timeout_s: float = 900.0,
    log_path: Path | None = None,
    proc: subprocess.Popen | None = None,
) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log_path and log_path.is_file():
                tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"vLLM exited early code={proc.returncode}\n{tail}"
            )
        if _vlm_ready(timeout_s=3.0):
            print(f"[vlm] ready in {time.time() - t0:.1f}s → http://{VLM_HOST}:{VLM_PORT}")
            return
        if log_path and log_path.is_file() and int(time.time() - t0) % 30 < 5:
            try:
                tail = log_path.read_text(errors="replace")[-400:]
                print(f"[vlm] still starting… log tail:\n{tail}", flush=True)
            except Exception:
                pass
        time.sleep(4)
    tail = log_path.read_text(errors="replace")[-2000:] if log_path and log_path.is_file() else ""
    raise TimeoutError(
        f"vLLM not ready after {timeout_s}s — see /tmp/vllm_vlm.log\n{tail}"
    )


def _start_vlm(
    *,
    gpu_mem_util: float = VLM_DEFAULT_MEM_UTIL,
    max_model_len: int = VLM_MAX_MODEL_LEN,
    cuda_devices: str | None = None,
    backend: str = "hf",  # hf | vllm
) -> subprocess.Popen:
    """Launch official Qwen3-VL-8B. Default: HF transformers server (stable)."""
    if _vlm_ready():
        print("[vlm] already serving — reuse")
        return None  # type: ignore

    model = _vlm_model_path()
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env.setdefault("HF_HOME", str(HF_HOME))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME))
    log_path = Path("/tmp/vllm_vlm.log")

    if backend == "vllm":
        _ensure_vllm()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_USE_V1"] = "0"
        cmd = [
            "vllm", "serve", model,
            "--served-model-name", VLM_MODEL,
            "--host", "0.0.0.0",
            "--port", str(VLM_PORT),
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(gpu_mem_util),
            "--max-model-len", str(max_model_len),
            "--enforce-eager",
        ]
    else:
        # HF transformers + FastAPI (OpenAI-compatible). Avoids vLLM engine core crashes.
        import sys
        server = Path("/tmp/hf_vlm_server.py")
        _write_hf_vlm_server(server)
        env["VLM_MODEL_PATH"] = model
        env["VLM_SERVED_NAME"] = VLM_MODEL
        env["VLM_PORT"] = str(VLM_PORT)
        # ensure fastapi/uvicorn
        try:
            import fastapi, uvicorn  # noqa: F401
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "pillow"])
        cmd = [sys.executable, str(server)]

    print("[vlm] starting:", " ".join(cmd), flush=True)
    print(f"[vlm] backend={backend} model={model}", flush=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        _wait_vlm(timeout_s=900.0, log_path=log_path, proc=proc)
    except Exception:
        _stop_vlm(proc)
        raise
    return proc


def _stop_vlm(proc: subprocess.Popen | None) -> None:
    if proc is None:
        # best-effort kill any leftover vllm on our port
        try:
            out = subprocess.check_output(["pgrep", "-f", f"vllm serve.*{VLM_PORT}"], text=True)
            for pid in out.split():
                try:
                    os.kill(int(pid), 15)
                except Exception:
                    pass
        except Exception:
            pass
        return
    if proc.poll() is not None:
        return
    print(f"[vlm] stopping pid={proc.pid}", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)
    # free CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_vlm_cuda(gpu_label: str, vlm_mode: str) -> str | None:
    """share → same GPU (None); split → last device if multi-GPU string."""
    if vlm_mode == "share":
        return None
    # e.g. "H100:2" or "RTX-PRO-6000:2"
    if ":" in gpu_label:
        try:
            n = int(gpu_label.split(":")[-1])
            if n >= 2:
                return str(n - 1)  # VLM on last card
        except ValueError:
            pass
    # single GPU forced split → still share
    print("[vlm] split requested but single GPU — falling back to share")
    return None



def _patch_pointcloud_pure_torch() -> None:
    """Replace pytorch3d point renderer with pure-torch projection splat (smoke quality)."""
    path = WORLDGEN / "src" / "pointcloud.py"
    if not path.is_file():
        print("[patch] pointcloud.py missing")
        return
    text = path.read_text()
    if "MODAL_LAB_PURE_TORCH_PCD_v2" in text:
        print("[patch] pointcloud already pure-torch")
        return
    # Full rewrite of module with compatible API
    path.write_text('''# MODAL_LAB_PURE_TORCH_PCD_v2 — pure torch point splat (no pytorch3d)
import contextlib
import os
import sys

import einops
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms

from .general_utils import split_n_into_d_parts


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


@contextlib.contextmanager
def suppress_stdout_stderr():
    yield


def point_rendering(K, w2cs, points, colors, device, h, w, background_color=[0, 0, 0],
                    render_radius=0.008, points_per_pixel=8, return_depth=False):
    """Pure-torch approximate point splat. API-compatible smoke fallback."""
    nframe = w2cs.shape[0]
    K = K.to(device=device, dtype=torch.float32)
    w2cs = w2cs.to(device=device, dtype=torch.float32)
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    else:
        points = points.to(device=device, dtype=torch.float32)
    if not torch.is_tensor(colors):
        colors = torch.tensor(colors, dtype=torch.float32, device=device)
    else:
        colors = colors.to(device=device, dtype=torch.float32)

    N = points.shape[0]
    ones = torch.ones((N, 1), device=device, dtype=torch.float32)
    pts_h = torch.cat([points, ones], dim=-1)  # [N,4]

    bg = torch.tensor(background_color, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    rgbs = bg.expand(nframe, 3, h, w).clone()
    depths = torch.full((nframe, 1, h, w), -1.0, device=device)
    masks = torch.ones((nframe, 1, h, w), device=device)

    # chunk frames for memory
    for fi in range(nframe):
        w2c = w2cs[fi]
        Ki = K[fi] if K.ndim == 3 else K
        cam = (w2c @ pts_h.T).T  # [N,4]
        z = cam[:, 2]
        valid = z > 1e-4
        if valid.sum() == 0:
            continue
        cam_v = cam[valid]
        col_v = colors[valid]
        z_v = cam_v[:, 2]
        # project
        u = Ki[0, 0] * (cam_v[:, 0] / z_v) + Ki[0, 2]
        v = Ki[1, 1] * (cam_v[:, 1] / z_v) + Ki[1, 2]
        ui = u.round().long()
        vi = v.round().long()
        inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if inb.sum() == 0:
            continue
        ui, vi, z_v, col_v = ui[inb], vi[inb], z_v[inb], col_v[inb]
        # depth sort far→near so near overwrites
        order = torch.argsort(z_v, descending=True)
        ui, vi, z_v, col_v = ui[order], vi[order], z_v[order], col_v[order]
        # splat 1px (+ optional neighbor for radius)
        rgbs[fi, :, vi, ui] = col_v.T
        depths[fi, 0, vi, ui] = z_v
        masks[fi, 0, vi, ui] = 0.0
        # small cross splat
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u2, v2 = ui + du, vi + dv
            ok = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)
            if ok.any():
                rgbs[fi, :, v2[ok], u2[ok]] = col_v[ok].T
                depths[fi, 0, v2[ok], u2[ok]] = z_v[ok]
                masks[fi, 0, v2[ok], u2[ok]] = 0.0

    if not return_depth:
        return rgbs, masks
    # Match pytorch3d return_depth=True: colors [F,H,W,3], depth [F,1,H,W]
    # (non-depth path returns CHW for multi_gpu_point_rendering)
    return rgbs.permute(0, 2, 3, 1).contiguous(), depths


def multi_gpu_point_rendering(image, Ks, w2cs, render_points, render_colors, image_h, image_w, device, device_num,
                              render_radius=0.008, points_per_pixel=20, slice_size=4, local_rank=0, replace_first_frame=True):
    image_tensor = (transforms.ToTensor()(image) * 2 - 1)[None]

    if type(Ks) != torch.Tensor:
        Ks_tensor = torch.tensor(Ks).float()
    else:
        Ks_tensor = Ks

    if type(w2cs) != torch.Tensor:
        w2cs_tensor = torch.tensor(w2cs).float()
    else:
        w2cs_tensor = w2cs

    pcd_renders, pcd_mask = [], []
    n_per_gpu_list = split_n_into_d_parts(Ks_tensor.shape[0], device_num)
    cumsum_gpu_list = np.cumsum(n_per_gpu_list)

    if local_rank == 0:
        Ks_tensor = Ks_tensor[:cumsum_gpu_list[0]]
        w2cs_tensor = w2cs_tensor[:cumsum_gpu_list[0]]
    else:
        Ks_tensor = Ks_tensor[cumsum_gpu_list[local_rank - 1]:cumsum_gpu_list[local_rank]]
        w2cs_tensor = w2cs_tensor[cumsum_gpu_list[local_rank - 1]:cumsum_gpu_list[local_rank]]

    gather_pcd_renders_r = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_renders_g = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_renders_b = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_mask = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]

    slice_times = w2cs_tensor.shape[0] // slice_size
    if w2cs_tensor.shape[0] % slice_size != 0:
        slice_times += 1

    for si in range(slice_times):
        pcd_renders_, pcd_mask_ = point_rendering(K=Ks_tensor[si * slice_size:(si + 1) * slice_size],
                                                  w2cs=w2cs_tensor[si * slice_size:(si + 1) * slice_size],
                                                  points=render_points, colors=render_colors,
                                                  h=image_h, w=image_w, render_radius=render_radius, points_per_pixel=points_per_pixel,
                                                  device=device, background_color=[0, 0, 0])
        pcd_renders.append(pcd_renders_)
        pcd_mask.append(pcd_mask_)

    pcd_renders = torch.cat(pcd_renders, dim=0).to(torch.float32)
    pcd_mask = torch.cat(pcd_mask, dim=0).to(torch.float32)

    dist.barrier()
    dist.all_gather(gather_pcd_renders_r, pcd_renders[:, 0:1].contiguous())
    dist.all_gather(gather_pcd_renders_g, pcd_renders[:, 1:2].contiguous())
    dist.all_gather(gather_pcd_renders_b, pcd_renders[:, 2:3].contiguous())
    dist.all_gather(gather_pcd_mask, pcd_mask)
    dist.barrier()

    gather_pcd_renders_r = torch.cat(gather_pcd_renders_r, dim=0)
    gather_pcd_renders_g = torch.cat(gather_pcd_renders_g, dim=0)
    gather_pcd_renders_b = torch.cat(gather_pcd_renders_b, dim=0)
    gather_pcd_renders = torch.cat([gather_pcd_renders_r, gather_pcd_renders_g, gather_pcd_renders_b], dim=1)
    gather_pcd_mask = torch.cat(gather_pcd_mask, dim=0)

    if replace_first_frame:
        gather_pcd_renders[0:1] = image_tensor
        gather_pcd_mask[0:1] = 0
    return gather_pcd_renders, gather_pcd_mask


def depth2pcd(w2c, K, points2d, depth, colors, mask):
    points3d = w2c.inverse() @ points_padding((K.inverse() @ points2d.T).T * depth.reshape(-1, 1)).T
    points3d = points3d.T[:, :3]
    points3d = points3d[mask.reshape(-1)]
    colors = colors[mask.reshape(-1)]
    return points3d, colors
''')
    print("[patch] pointcloud -> pure-torch splat")


def _patch_traj_generate_lazy() -> None:
    path = WORLDGEN / "traj_generate.py"
    text = path.read_text()
    if "from transformers import Sam3Processor, Sam3Model" in text and "Sam3Processor = Sam3Model = None" not in text:
        text = text.replace(
            "from transformers import Sam3Processor, Sam3Model\n",
            "try:\n"
            "    from transformers import Sam3Processor, Sam3Model\n"
            "except Exception as _e:\n"
            "    print('[modal-lab] Sam3 import skipped:', _e)\n"
            "    Sam3Processor = Sam3Model = None\n",
            1,
        )
    if "MODAL_LAB_PATCH" not in text:
        needle = (
            '    zim_predictor = build_zim_model("vit_l", resolve_zim_checkpoint(), device=device)\n'
            "    gd_processor, gd_model = build_gd_model(resolve_gd_checkpoint(), device=device)\n\n"
            "    depth_model = MoGeModel.from_pretrained(MOGE_ID).to(device).eval()\n\n"
            "    # VLM & SAM3\n"
            '    client = OpenAI(api_key="EMPTY", base_url=f"http://{LLM_ADDR}:{LLM_PORT}/v1")\n'
            '    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)\n'
            '    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")\n'
        )
        repl = (
            "    # MODAL_LAB_PATCH: only load what smoke needs\n"
            "    zim_predictor = gd_processor = gd_model = None\n"
            "    sam3_model = sam3_processor = None\n"
            "    if args.apply_nav_traj:\n"
            '        zim_predictor = build_zim_model("vit_l", resolve_zim_checkpoint(), device=device)\n'
            "        gd_processor, gd_model = build_gd_model(resolve_gd_checkpoint(), device=device)\n"
            '        sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)\n'
            '        sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")\n'
            "    depth_model = MoGeModel.from_pretrained(MOGE_ID).to(device).eval()\n\n"
            "    # VLM\n"
            '    client = OpenAI(api_key="EMPTY", base_url=f"http://{LLM_ADDR}:{LLM_PORT}/v1")\n'
        )
        if needle in text:
            text = text.replace(needle, repl, 1)
            print("[patch] traj_generate.py patched")
        else:
            print("[patch] traj_generate pattern missing")
    path.write_text(text)


def _patch_pytorch3d_stub() -> None:
    cands = list(WORLDGEN.rglob("camera_utils.py"))
    if not cands:
        print("[patch] camera_utils look_at already/missing")
        return
    path = cands[0]
    text = path.read_text()
    if "MODAL_LAB_LOOKAT" in text:
        return
    if "from pytorch3d.renderer.cameras import look_at_rotation" in text:
        stub = (
            "# MODAL_LAB_LOOKAT: pure-torch look_at_rotation stub\n"
            "import torch as _torch\n\n"
            "def look_at_rotation(camera_position, at, up=((0, 1, 0),), device=\"cpu\"):\n"
            "    if not _torch.is_tensor(camera_position):\n"
            "        camera_position = _torch.tensor(camera_position, dtype=_torch.float32, device=device)\n"
            "    else:\n"
            "        camera_position = camera_position.to(device=device, dtype=_torch.float32)\n"
            "    if camera_position.ndim == 1:\n"
            "        camera_position = camera_position.unsqueeze(0)\n"
            "    if not _torch.is_tensor(at):\n"
            "        at = _torch.tensor(at, dtype=_torch.float32, device=camera_position.device)\n"
            "    else:\n"
            "        at = at.to(device=camera_position.device, dtype=_torch.float32)\n"
            "    if at.ndim == 1:\n"
            "        at = at.unsqueeze(0)\n"
            "    if not _torch.is_tensor(up):\n"
            "        up = _torch.tensor(up, dtype=_torch.float32, device=camera_position.device)\n"
            "    else:\n"
            "        up = up.to(device=camera_position.device, dtype=_torch.float32)\n"
            "    if up.ndim == 1:\n"
            "        up = up.unsqueeze(0)\n"
            "    z = camera_position - at\n"
            "    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)\n"
            "    x = _torch.cross(up.expand_as(z), z, dim=-1)\n"
            "    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)\n"
            "    y = _torch.cross(z, x, dim=-1)\n"
            "    return _torch.stack([x, y, z], dim=-1)\n\n"
        )
        text = text.replace("from pytorch3d.renderer.cameras import look_at_rotation\n", stub, 1)
        path.write_text(text)
        print("[patch] camera_utils look_at stub")
    else:
        print("[patch] camera_utils look_at already/missing")


def _patch_traj_render_captions() -> None:
    path = WORLDGEN / "traj_render.py"
    if not path.is_file():
        return
    text = path.read_text()
    if "MODAL_LAB_PATCH" in text:
        return
    needle = "traj_caption = get_traj_caption(llm_addr, llm_port, model_name, render_path)"
    if needle in text:
        text = text.replace(
            needle,
            (
                "try:\n"
                "            traj_caption = get_traj_caption(llm_addr, llm_port, model_name, render_path)\n"
                "        except Exception as _cap_e:\n"
                '            print(f"[modal-lab] caption fallback for {render_path}: {_cap_e}")\n'
                '            traj_caption = "A continuous indoor camera trajectory through a room."  # MODAL_LAB_PATCH'
            ),
            1,
        )
        path.write_text(text)
        print("[patch] traj_render caption fallback")


def _ensure_sam3_transformers() -> None:
    import sys
    try:
        from transformers import Sam3VideoModel  # noqa: F401
        print("[deps] Sam3VideoModel available")
        return
    except Exception as e:
        print(f"[deps] Sam3 missing ({e}); upgrading transformers==5.2.0")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "transformers==5.2.0", "huggingface_hub>=0.34.0"]
    )
    from transformers import Sam3VideoModel  # noqa: F401
    print("[deps] transformers upgraded OK")


def _patch_worldmirror_attention() -> None:
    """No flash-attn on Modal image — fall back to PyTorch SDPA (same as 007)."""
    path = REPO_DIR / "hyworld2" / "worldrecon" / "hyworldmirror" / "models" / "layers" / "attention.py"
    if not path.is_file():
        print("[patch] worldmirror attention.py missing")
        return
    text = path.read_text()
    if "MODAL_LAB_SDPA" in text:
        print("[patch] worldmirror attention already SDPA")
        return
    old = (
        "try:\n"
        "    from flash_attn_interface import flash_attn_func as flash_attn_func_v3\n"
        "    _USE_FLASH_ATTN_V3 = True\n"
        "except ImportError:\n"
        "    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2\n"
        "    _USE_FLASH_ATTN_V3 = False\n"
    )
    new = (
        "try:  # MODAL_LAB_SDPA\n"
        "    from flash_attn_interface import flash_attn_func as flash_attn_func_v3\n"
        "    _USE_FLASH_ATTN_V3 = True\n"
        "    _HAS_FLASH_ATTN = True\n"
        "except ImportError:\n"
        "    try:\n"
        "        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2\n"
        "        _USE_FLASH_ATTN_V3 = False\n"
        "        _HAS_FLASH_ATTN = True\n"
        "    except ImportError:\n"
        "        flash_attn_func_v3 = None\n"
        "        flash_attn_func_v2 = None\n"
        "        _USE_FLASH_ATTN_V3 = False\n"
        "        _HAS_FLASH_ATTN = False\n"
        '        print("[modal-lab] flash_attn missing — WorldMirror uses SDPA")\n'
    )
    if old not in text:
        print("[patch] worldmirror attention import block missing")
        return
    text = text.replace(old, new, 1)
    old2 = "if q.dtype==torch.bfloat16 or q.dtype==torch.float16:"
    new2 = "if _HAS_FLASH_ATTN and (q.dtype==torch.bfloat16 or q.dtype==torch.float16):"
    if old2 not in text:
        print("[patch] worldmirror attention dtype branch missing")
        return
    text = text.replace(old2, new2, 1)
    path.write_text(text)
    print("[patch] worldmirror attention → SDPA")


def _patch_worldmirror_local_weights() -> None:
    """Point apply_worldmirror at a valid local path or HF repo id.

    Never inject a bare absolute path that does not contain model files —
    HuggingFace treats unknown paths as repo ids and raises HFValidationError.
    """
    path = WORLDGEN / "src" / "retrieval_wm.py"
    if not path.is_file():
        return
    text = path.read_text()
    # Always re-resolve so a later download is picked up after re-patch
    if "MODAL_LAB_WM_LOCAL" in text:
        # strip previous injection so we can re-evaluate
        import re as _re
        text = _re.sub(
            r',\n\s*"--pretrained_model_name_or_path", "[^"]*",  # MODAL_LAB_WM_LOCAL',
            "",
            text,
            count=1,
        )
    needle = (
        '"--disable_heads", "normal", "points", "gs"\n'
        "                ]"
    )
    wm_flat = Path(WEIGHTS_MOUNT) / "HY-WorldMirror-2.0"
    wm_nested = Path(WEIGHTS_MOUNT) / "hf_hy_world_2"
    if (wm_flat / "model.safetensors").is_file() and (
        (wm_flat / "config.yaml").is_file() or (wm_flat / "config.json").is_file()
    ):
        pretrained = str(wm_flat)
    elif (wm_nested / "HY-WorldMirror-2.0" / "model.safetensors").is_file():
        # repo-root layout; pipeline default subfolder=HY-WorldMirror-2.0
        pretrained = str(wm_nested)
    else:
        # Valid HF id — will download into HF_HOME on first use
        pretrained = "tencent/HY-World-2.0"
    repl = (
        '"--disable_heads", "normal", "points", "gs",\n'
        f'                    "--pretrained_model_name_or_path", "{pretrained}",  # MODAL_LAB_WM_LOCAL\n'
        "                ]"
    )
    if needle not in text:
        print("[patch] worldmirror wm_cmd pattern missing")
        return
    path.write_text(text.replace(needle, repl, 1))
    print(f"[patch] worldmirror local weights → {pretrained}")


def _patch_gs_distloss() -> None:
    """Stock gsplat has no distloss kwarg — strip unsupported kwargs at runtime."""
    path = WORLDGEN / "world_gs_trainer.py"
    text = path.read_text()
    if "MODAL_LAB_DISTLOSS_WRAP" in text:
        return
    old = "from gsplat.rendering import rasterization"
    if old not in text:
        print("[patch] rasterization import not found")
        return
    # Use triple-quoted replacement written as a list of lines to avoid nesting hell
    lines = [

        "from gsplat.rendering import rasterization as _rasterization_raw",
        "import inspect as _inspect",
        "def rasterization(*args, **kwargs):  # MODAL_LAB_DISTLOSS_WRAP",
        "    sig = _inspect.signature(_rasterization_raw)",
        "    if not any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):",
        "        for k in list(kwargs):",
        "            if k not in sig.parameters:",
        "                kwargs.pop(k, None)",
        "    return _rasterization_raw(*args, **kwargs)",
    ]
    text = text.replace(old, "\n".join(lines), 1)
    path.write_text(text)
    print("[patch] gsplat distloss filter")


def _ensure_gs_trainer_deps() -> None:
    """Install 3DGS trainer extras; fallback pure-torch fused_ssim if CUDA ext fails."""
    import sys

    def _pip(*pkgs: str) -> None:
        env = os.environ.copy()
        env["CXX"] = "g++"
        env["CC"] = "gcc"
        env["CUDAHOSTCXX"] = "g++"
        print(f"[deps] pip install --no-build-isolation CXX=g++ {pkgs}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", *pkgs],
            env=env,
        )

    try:
        import fused_ssim  # noqa: F401
    except Exception:
        try:
            _pip("git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5")
        except Exception as e:
            print(f"[deps] fused-ssim build failed ({e}); pure-torch stub")
            stub_dir = Path("/tmp/fused_ssim_stub")
            (stub_dir / "fused_ssim").mkdir(parents=True, exist_ok=True)
            (stub_dir / "fused_ssim" / "__init__.py").write_text(
                "import torch.nn.functional as F\n\n"
                "def fused_ssim(img1, img2, padding='same', train=True):\n"
                "    C1, C2 = 0.01 ** 2, 0.03 ** 2\n"
                "    pad = 5 if padding == 'same' else 0\n"
                "    mu1 = F.avg_pool2d(img1, 11, 1, pad)\n"
                "    mu2 = F.avg_pool2d(img2, 11, 1, pad)\n"
                "    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2\n"
                "    sigma1_sq = F.avg_pool2d(img1 * img1, 11, 1, pad) - mu1_sq\n"
                "    sigma2_sq = F.avg_pool2d(img2 * img2, 11, 1, pad) - mu2_sq\n"
                "    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, pad) - mu1_mu2\n"
                "    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / "
                "((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))\n"
                "    return ssim_map.mean()\n"
            )
            (stub_dir / "setup.py").write_text(
                "from setuptools import setup, find_packages\n"
                "setup(name='fused_ssim', version='0.0.0', packages=find_packages())\n"
            )
            _pip(str(stub_dir))
    try:
        import nerfview  # noqa: F401
    except Exception:
        _pip("git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787")
    # Prefer custom gsplat_maskgaussian; fall back to stock gsplat
    gs_path = REPO_DIR / "hyworld2" / "worldgen" / "third_party" / "gsplat_maskgaussian"
    if gs_path.is_dir():
        # glm submodule often missing — fetch if needed
        glm = gs_path / "gsplat" / "cuda" / "csrc" / "third_party" / "glm"
        if not (glm / "gtc" / "type_ptr.hpp").is_file():
            print("[deps] cloning glm into gsplat third_party")
            glm.parent.mkdir(parents=True, exist_ok=True)
            subprocess.call(
                ["git", "clone", "--depth", "1", "https://github.com/g-truc/glm.git", str(glm)]
            )
        try:
            env = os.environ.copy()
            env["CXX"] = "g++"
            env["CC"] = "gcc"
            env["CUDAHOSTCXX"] = "g++"
            env["TORCH_CUDA_ARCH_LIST"] = "8.0;8.9;9.0;12.0"
            print(f"[deps] install gsplat_maskgaussian from {gs_path}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", str(gs_path)],
                env=env,
            )
        except Exception as e:
            print(f"[deps] custom gsplat build failed ({e}); keep stock gsplat")
            try:
                import gsplat  # noqa: F401
            except Exception:
                _pip("gsplat")
    else:
        try:
            import gsplat  # noqa: F401
        except Exception:
            _pip("gsplat")
    for extra in ("tensorboard", "lpips", "jaxtyping", "pytorch_msssim"):
        try:
            __import__(extra if extra != "pytorch_msssim" else "pytorch_msssim")
        except Exception:
            try:
                _pip(extra)
            except Exception as _e:
                print(f"[deps] optional {extra} skip: {_e}")
    print("[deps] GS trainer deps ready")


def _patch_video_gen_local() -> None:
    path = WORLDGEN / "video_gen.py"
    text = path.read_text()
    if "MODAL_LAB_VIDEO_PATCH" not in text:
        old = (
            "worldstereo = WorldStereo.from_pretrained(\n"
            '        "hanshanxue/WorldStereo",\n'
            "        subfolder=args.model_type,\n"
            "        local_files_only=args.local_files_only,\n"
        )
        new = (
            "# MODAL_LAB_VIDEO_PATCH\n"
            '    _ws_root = os.environ.get("WORLDSTEREO_LOCAL", "hanshanxue/WorldStereo")\n'
            "    _local_only = bool(args.local_files_only)\n"
            '    print(f"[modal-lab] WorldStereo root={_ws_root} local_only={_local_only}")\n'
            "    worldstereo = WorldStereo.from_pretrained(\n"
            "        _ws_root,\n"
            "        subfolder=args.model_type,\n"
            "        local_files_only=_local_only,\n"
        )
        if old in text:
            text = text.replace(old, new, 1)
            print("[patch] video_gen local WorldStereo")
        else:
            print("[patch] video_gen WorldStereo pattern missing")
    if "MODAL_LAB_SKIP_SAM3" not in text:
        old_s = (
            "    sam3_model = Sam3VideoModel.from_pretrained(SAM3_REPO_ID).to(device, dtype=torch.bfloat16)\n"
            "    sam3_processor = Sam3VideoProcessor.from_pretrained(SAM3_REPO_ID)\n"
        )
        new_s = (
            "    # MODAL_LAB_SKIP_SAM3\n"
            '    if os.environ.get("SKIP_SAM3", "1") == "1":\n'
            '        print("[modal-lab] SKIP_SAM3=1 — not loading facebook/sam3")\n'
            "        sam3_model = None\n"
            "        sam3_processor = None\n"
            "    else:\n"
            "        sam3_model = Sam3VideoModel.from_pretrained(SAM3_REPO_ID).to(device, dtype=torch.bfloat16)\n"
            "        sam3_processor = Sam3VideoProcessor.from_pretrained(SAM3_REPO_ID)\n"
        )
        if old_s in text:
            text = text.replace(old_s, new_s, 1)
            print("[patch] video_gen skip SAM3")
        else:
            print("[patch] video_gen SAM3 load pattern missing")
    path.write_text(text)

    rpath = WORLDGEN / "src" / "retrieval_wm.py"
    rtext = rpath.read_text()
    if "MODAL_LAB_SKIP_SAM3" not in rtext:
        old_r = (
            '        rank0_log(f"Initializing SAM3 Model...")\n'
            "        if sam3_model is None or sam3_processor is None:\n"
            "            from transformers import Sam3VideoModel, Sam3VideoProcessor\n"
            '            self.sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)\n'
            '            self.sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")\n'
            "        else:\n"
            "            self.sam3_model = sam3_model\n"
            "            self.sam3_processor = sam3_processor\n"
        )
        new_r = (
            '        rank0_log(f"Initializing SAM3 Model...")\n'
            "        # MODAL_LAB_SKIP_SAM3\n"
            "        if sam3_model is None or sam3_processor is None:\n"
            '            if os.environ.get("SKIP_SAM3", "1") == "1":\n'
            '                rank0_log("SKIP_SAM3=1 — sam3 disabled")\n'
            "                self.sam3_model = None\n"
            "                self.sam3_processor = None\n"
            "            else:\n"
            "                from transformers import Sam3VideoModel, Sam3VideoProcessor\n"
            '                self.sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)\n'
            '                self.sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")\n'
            "        else:\n"
            "            self.sam3_model = sam3_model\n"
            "            self.sam3_processor = sam3_processor\n"
        )
        if old_r in rtext:
            rpath.write_text(rtext.replace(old_r, new_r, 1))
            print("[patch] retrieval_wm skip SAM3")
        else:
            print("[patch] retrieval_wm SAM3 pattern missing")
    rtext = rpath.read_text()
    needle = 'if self.meta_info["scene_type"] == "outdoor":'
    guard = 'if self.meta_info["scene_type"] == "outdoor" and self.sam3_model is not None and self.sam3_processor is not None:'
    if needle in rtext and guard not in rtext:
        rpath.write_text(rtext.replace(needle, guard, 1))
        print("[patch] outdoor SAM3 guard")
    vtext = path.read_text()
    if "MODAL_LAB_SOFT_WM" not in vtext and "memory_bank.apply_worldmirror(skip_exist=True)" in vtext:
        vtext = vtext.replace(
            "memory_bank.apply_worldmirror(skip_exist=True)",
            (
                "try:\n"
                "                        memory_bank.apply_worldmirror(skip_exist=True)  # MODAL_LAB_SOFT_WM\n"
                "                    except Exception as _wm_e:\n"
                '                        print(f"[modal-lab] apply_worldmirror soft-fail: {_wm_e}")\n'
            ),
            1,
        )
        path.write_text(vtext)
        print("[patch] apply_worldmirror soft-fail")


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol, PANO_MOUNT: pano_vol},
    timeout=4 * 60 * 60,
    cpu=4,
    memory=16384,
)
def prepare_scene(
    from_008_run: str = "smoke_qwen",
    scene_name: str = "scene_from_008",
) -> dict[str, Any]:
    scene = _seed_minimal_scene(scene_name, from_008_run)
    outputs_vol.commit()
    return {"ok": True, "scene": scene_name, "panorama": str(scene / "panorama.png")}


@app.function(
    image=download_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol, PANO_MOUNT: pano_vol},
    timeout=4 * 60 * 60,
    cpu=4,
    memory=16384,
)
def download_weights(which: str = "worldstereo-dmd") -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    if which in ("vlm", "qwen3-vl", "qwen", "all"):
        dest = Path(WEIGHTS_MOUNT) / "Qwen3-VL-8B-Instruct"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[dl] {VLM_MODEL} → {dest}")
        snapshot_download(
            VLM_MODEL,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        weights_vol.commit()
        if which != "all":
            return {"ok": True, "which": which, "path": str(dest), "model": VLM_MODEL}

    if which in ("worldmirror", "wm", "all"):
        dest = Path(WEIGHTS_MOUNT) / "HY-WorldMirror-2.0"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[dl] WorldMirror → {dest}")
        snapshot_download(
            "tencent/HY-World-2.0",
            local_dir=str(Path(WEIGHTS_MOUNT) / "hf_hy_world_2"),
            allow_patterns=["HY-WorldMirror-2.0/*"],
            local_dir_use_symlinks=False,
        )
        src = Path(WEIGHTS_MOUNT) / "hf_hy_world_2" / "HY-WorldMirror-2.0"
        if src.is_dir():
            for p in src.iterdir():
                target = dest / p.name
                if p.is_file() and not target.exists():
                    shutil.copy2(p, target)
        weights_vol.commit()
        if which != "all":
            return {"ok": True, "which": which, "path": str(dest)}

    dest = Path(WEIGHTS_MOUNT) / "WorldStereo"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[dl] WorldStereo → {dest}")
    allow = None
    if which in ("worldstereo-dmd", "all"):
        allow = ["worldstereo-memory-dmd/*", "*.json", "*.txt", "*.md"]
    if which in ("worldstereo-dmd", "all", "worldstereo"):
        snapshot_download(
            WORLDSTEREO_HF,
            local_dir=str(dest),
            allow_patterns=allow,
            local_dir_use_symlinks=False,
        )
    weights_vol.commit()
    return {"ok": True, "which": which, "path": str(dest)}



@app.function(
    image=worldgen_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol, PANO_MOUNT: pano_vol},
    timeout=3 * 60 * 60,
    gpu=DEFAULT_GPU,
    memory=131072,
    cpu=8,
)
def run_stage(
    stage: int,
    scene: str = "scene_from_008",
    from_008: str = "smoke_qwen",
    gpu_label: str = DEFAULT_GPU,
    split_view_num: int = 1,
    nframe: int = 16,
    wonder_topk: int = 1,
    recon_topk: int = 0,
    max_steps: int = 4000,
    # Stage1/2 VLM (official Qwen3-VL-8B)
    force_vlm: bool = True,
    apply_nav_traj: bool = True,
    apply_up_route: bool = True,
    apply_recon_iteration: bool = False,
    vlm_mode: str = "share",  # share | split
    vlm_mem_util: float = VLM_DEFAULT_MEM_UTIL,
    vlm_max_model_len: int = VLM_MAX_MODEL_LEN,
    keep_vlm: bool = False,
) -> dict[str, Any]:
    """Run one pipeline stage. Stages 1–2 start official Qwen3-VL-8B via vLLM."""
    import sys

    sys.path.insert(0, str(WORLDGEN))
    sys.path.insert(0, str(REPO_DIR))

    scene_path = _seed_minimal_scene(scene, from_008)
    os.chdir(WORLDGEN)
    _patch_pointcloud_pure_torch()
    _patch_traj_generate_lazy()
    _patch_traj_render_captions()
    _patch_pytorch3d_stub()
    if stage >= 3:
        _ensure_sam3_transformers()
        _patch_video_gen_local()
        _patch_worldmirror_attention()
        _patch_worldmirror_local_weights()
    if stage >= 5:
        _ensure_gs_trainer_deps()
        _patch_gs_distloss()

    t0 = time.time()
    log: dict[str, Any] = {
        "stage": stage,
        "scene": scene,
        "gpu": gpu_label,
        "vlm_model": VLM_MODEL if stage in (1, 2) else None,
        "force_vlm": force_vlm if stage == 1 else None,
        "apply_nav_traj": apply_nav_traj if stage == 1 else None,
        "vlm_mode": vlm_mode if stage in (1, 2) else None,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    vlm_proc = None
    need_vlm = stage in (1, 2)
    try:
        if need_vlm:
            cuda_dev = _resolve_vlm_cuda(gpu_label, vlm_mode)
            mem = vlm_mem_util
            # Stage1 without nav: skip VLM (meta pre-seeded) to avoid MoGe SIGSEGV
            # Stage1 with nav: start VLM (risky on 1×GPU)
            # Stage2: start VLM for captions
            start_vlm = True
            if stage == 1 and not apply_nav_traj:
                start_vlm = False
                print("[vlm] stage1 no-nav: skip VLM (meta seeded; free GPU for MoGe)", flush=True)
            if start_vlm:
                if apply_nav_traj and stage == 1 and vlm_mode == "share":
                    mem = min(mem, 0.35)
                vlm_proc = _start_vlm(
                    gpu_mem_util=mem if stage == 1 else 0.50,
                    max_model_len=vlm_max_model_len,
                    cuda_devices=cuda_dev,
                )
                log["vlm_mem_util"] = mem
                log["vlm_cuda"] = cuda_dev
            else:
                log["vlm_mem_util"] = None
                log["vlm_cuda"] = None

        if stage == 1:
            cmd = [
                "python", "traj_generate.py",
                "--target_path", str(scene_path),
                "--llm_addr", VLM_HOST,
                "--llm_port", str(VLM_PORT),
                "--llm_name", VLM_MODEL,
                "--split_view_num", str(split_view_num),
                "--nframe", str(nframe),
                "--splitted_resolution", "480",
                "--wonder_topk", str(wonder_topk),
                "--recon_topk", str(max(recon_topk, 0)),
            ]
            if force_vlm:
                cmd.append("--force_vlm")
            if apply_nav_traj:
                cmd.append("--apply_nav_traj")
            if apply_up_route:
                cmd.append("--apply_up_route")
            if apply_recon_iteration:
                cmd.append("--apply_recon_iteration")
            print("+", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(WORLDGEN))
        elif stage == 2:
            cmd = [
                "torchrun", "--nproc_per_node=1", "traj_render.py",
                "--target_path", str(scene_path),
                "--llm_addr", VLM_HOST,
                "--llm_port", str(VLM_PORT),
                "--llm_name", VLM_MODEL,
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(WORLDGEN))
        elif stage == 3:
            env = os.environ.copy()
            env["HF_HOME"] = str(HF_HOME)
            env["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME)
            env["SKIP_SAM3"] = "1"
            ws_local = Path(WEIGHTS_MOUNT) / "WorldStereo"
            if (ws_local / "worldstereo-memory-dmd").is_dir():
                env["WORLDSTEREO_LOCAL"] = str(ws_local)
            cmd = [
                "torchrun", "--nproc_per_node=1", "video_gen.py",
                "--target_path", str(scene_path),
                "--model_type", "worldstereo-memory-dmd",
                "--max_reference", "2",
                "--align_nframe", "4",
                "--skip_exist",
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(WORLDGEN), env=env)
        elif stage == 4:
            cmd = [
                "torchrun", "--nproc_per_node=1", "gen_gs_data.py",
                "--root_path", str(scene_path),
                "--save_normal", "--split_sky", "--interval", "2",
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(WORLDGEN))
        elif stage == 5:
            result_dir = scene_path / "gs_result"
            result_dir.mkdir(parents=True, exist_ok=True)
            data_dir = scene_path / "gs_data"
            if not data_dir.is_dir():
                alts = list(scene_path.glob("**/gs_data"))
                if alts:
                    data_dir = alts[0]
            cmd = [
                "python", "-m", "world_gs_trainer", "default",
                "--data_dir", str(data_dir),
                "--result_dir", str(result_dir),
                "--max_steps", str(max_steps),
                "--save_steps", str(max_steps),
                "--eval_steps", str(max_steps),
                "--ply_steps", str(max_steps),
                "--save_ply", "--disable_video", "--disable_viewer",
                "--use_scale_regularization", "--antialiased",
                "--depth_loss", "--normal_loss",
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(WORLDGEN))
        else:
            raise ValueError(f"unknown stage {stage}")
        log["ok"] = True
    except subprocess.CalledProcessError as e:
        log["ok"] = False
        log["error"] = f"exit {e.returncode}"
        log["cmd"] = list(e.cmd) if e.cmd else None
        raise
    finally:
        if need_vlm and not keep_vlm:
            _stop_vlm(vlm_proc)
        total = time.time() - t0
        log["seconds"] = round(total, 2)
        log["est_cost_usd"] = _price(gpu_label, total)
        arts = []
        for p in sorted(scene_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".mp4", ".ply", ".json", ".png", ".spz", ".glb"}:
                arts.append(str(p.relative_to(scene_path)))
            if len(arts) >= 24:
                break
        log["artifacts_sample"] = arts
        _write_meta(scene, log)
        outputs_vol.commit()
        weights_vol.commit()
    return log


@app.function(
    image=worldgen_image,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol, PANO_MOUNT: pano_vol},
    timeout=4 * 60 * 60,
    gpu=DEFAULT_GPU,
    memory=131072,
    cpu=8,
)
def run_stage12(
    scene: str = "scene_from_008",
    from_008: str = "smoke_qwen",
    gpu_label: str = DEFAULT_GPU,
    split_view_num: int = 1,
    nframe: int = 16,
    wonder_topk: int = 1,
    recon_topk: int = 0,
    force_vlm: bool = True,
    apply_nav_traj: bool = True,
    apply_up_route: bool = True,
    apply_recon_iteration: bool = False,
    vlm_mode: str = "share",
    vlm_mem_util: float = VLM_DEFAULT_MEM_UTIL,
    vlm_max_model_len: int = VLM_MAX_MODEL_LEN,
) -> dict[str, Any]:
    """Stage1 + Stage2 with one VLM lifecycle (official Qwen3-VL-8B)."""
    import sys

    sys.path.insert(0, str(WORLDGEN))
    sys.path.insert(0, str(REPO_DIR))

    scene_path = _seed_minimal_scene(scene, from_008)
    os.chdir(WORLDGEN)
    _patch_pointcloud_pure_torch()
    _patch_traj_generate_lazy()
    _patch_traj_render_captions()
    _patch_pytorch3d_stub()

    t0 = time.time()
    results: list[dict[str, Any]] = []
    vlm_proc = None
    cuda_dev = _resolve_vlm_cuda(gpu_label, vlm_mode)
    mem = min(vlm_mem_util, 0.35) if apply_nav_traj and vlm_mode == "share" else vlm_mem_util

    try:
        # ----- Stage 1 -----
        # CRITICAL: do NOT keep VLM on GPU while MoGe runs (same-card OOM/SIGSEGV).
        # Pre-seeded meta_info skips VLM for env_cls. apply_nav_traj needs VLM mid-run —
        # for that path we briefly start VLM only if objects.json missing, then kill before MoGe.
        # Smoke default: no nav → stage1 pure vision, VLM only in stage2 captions.
        s1_t0 = time.time()
        need_vlm_s1 = bool(apply_nav_traj) or not (scene_path / "meta_info.json").is_file()
        if need_vlm_s1 and apply_nav_traj:
            print("[vlm] stage1 needs nav VLM briefly — start then MoGe shares risk; prefer --no-apply-nav-traj on 1×GPU", flush=True)
            vlm_proc = _start_vlm(gpu_mem_util=mem, max_model_len=vlm_max_model_len, cuda_devices=cuda_dev)
        elif need_vlm_s1:
            vlm_proc = _start_vlm(gpu_mem_util=0.45, max_model_len=vlm_max_model_len, cuda_devices=cuda_dev)

        cmd1 = [
            "python", "traj_generate.py",
            "--target_path", str(scene_path),
            "--llm_addr", VLM_HOST,
            "--llm_port", str(VLM_PORT),
            "--llm_name", VLM_MODEL,
            "--split_view_num", str(split_view_num),
            "--nframe", str(nframe),
            "--splitted_resolution", "480",
            "--wonder_topk", str(wonder_topk),
            "--recon_topk", str(max(recon_topk, 0)),
        ]
        if force_vlm and need_vlm_s1:
            cmd1.append("--force_vlm")
        if apply_nav_traj:
            cmd1.append("--apply_nav_traj")
        if apply_up_route:
            cmd1.append("--apply_up_route")
        if apply_recon_iteration:
            cmd1.append("--apply_recon_iteration")
        print("======== STAGE 1 ========", flush=True)
        print("+", " ".join(cmd1), flush=True)
        # Free VLM before MoGe if we only needed it for meta (already written) — hard with concurrent design.
        # When meta exists and no nav: never started VLM here.
        if vlm_proc is not None and not apply_nav_traj:
            # meta labeling happens first inside traj_generate; keep VLM only if meta missing.
            # If meta already present, stop VLM before heavy MoGe by not having started it.
            pass
        # Always kill VLM before long MoGe if nav is off
        if vlm_proc is not None and not apply_nav_traj:
            _stop_vlm(vlm_proc)
            vlm_proc = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        subprocess.check_call(cmd1, cwd=str(WORLDGEN))
        # After stage1, free any leftover VLM before stage2 restart cleanly
        if vlm_proc is not None:
            _stop_vlm(vlm_proc)
            vlm_proc = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        s1 = {
            "stage": 1, "ok": True,
            "seconds": round(time.time() - s1_t0, 2),
            "est_cost_usd": _price(gpu_label, time.time() - s1_t0),
            "cmd": cmd1,
            "vlm_during_stage1": need_vlm_s1,
        }
        _write_meta(scene, {**s1, "ts": datetime.now(timezone.utc).isoformat(), "gpu": gpu_label})
        results.append(s1)

        # ----- Stage 2 -----
        # Start VLM for captions only (after stage1 freed GPU)
        s2_t0 = time.time()
        vlm_proc = _start_vlm(
            gpu_mem_util=0.50,
            max_model_len=vlm_max_model_len,
            cuda_devices=cuda_dev,
        )
        cmd2 = [
            "torchrun", "--nproc_per_node=1", "traj_render.py",
            "--target_path", str(scene_path),
            "--llm_addr", VLM_HOST,
            "--llm_port", str(VLM_PORT),
            "--llm_name", VLM_MODEL,
        ]
        print("======== STAGE 2 ========", flush=True)
        print("+", " ".join(cmd2), flush=True)
        subprocess.check_call(cmd2, cwd=str(WORLDGEN))
        s2 = {
            "stage": 2, "ok": True,
            "seconds": round(time.time() - s2_t0, 2),
            "est_cost_usd": _price(gpu_label, time.time() - s2_t0),
            "cmd": cmd2,
        }
        _write_meta(scene, {**s2, "ts": datetime.now(timezone.utc).isoformat(), "gpu": gpu_label})
        results.append(s2)
    except subprocess.CalledProcessError as e:
        fail = {
            "ok": False,
            "error": f"exit {e.returncode}",
            "cmd": list(e.cmd) if e.cmd else None,
            "partial": results,
        }
        _write_meta(scene, {**fail, "ts": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        _stop_vlm(vlm_proc)
        outputs_vol.commit()
        weights_vol.commit()

    total = time.time() - t0
    summary = {
        "ok": True,
        "scene": scene,
        "gpu": gpu_label,
        "vlm_model": VLM_MODEL,
        "vlm_mode": vlm_mode,
        "vlm_mem_util": mem,
        "force_vlm": force_vlm,
        "apply_nav_traj": apply_nav_traj,
        "nframe": nframe,
        "split_view_num": split_view_num,
        "stages": results,
        "seconds": round(total, 2),
        "est_cost_usd": _price(gpu_label, total),
    }
    arts = []
    for p in sorted(scene_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".mp4", ".ply", ".json", ".png", ".glb"}:
            arts.append(str(p.relative_to(scene_path)))
        if len(arts) >= 32:
            break
    summary["artifacts_sample"] = arts
    _write_meta(scene, {"stage": "1+2", **summary, "ts": datetime.now(timezone.utc).isoformat()})
    outputs_vol.commit()
    return summary


def status_payload() -> dict[str, Any]:
    return {
        "app": APP_NAME,
        "version": "v8.2",
        "default_gpu": DEFAULT_GPU,
        "vlm": {
            "model": VLM_MODEL,
            "port": VLM_PORT,
            "default_mem_util": VLM_DEFAULT_MEM_UTIL,
            "max_model_len": VLM_MAX_MODEL_LEN,
            "modes": ["share", "split"],
        },
        "pipeline": [
            "prepare (008 pano)",
            "download vlm | worldstereo-dmd | worldmirror",
            "1 traj_generate + Qwen3-VL-8B",
            "2 traj_render + VLM captions",
            "3 worldstereo-memory-dmd",
            "4 gen_gs_data",
            "5 world_gs_trainer -> ply",
        ],
        "stage12": "one VLM lifecycle for stage1+2 (recommended)",
        "smoke_flags": {
            "split_view_num": 1,
            "nframe": 16,
            "wonder_topk": 1,
            "recon_topk": 0,
            "force_vlm": True,
            "apply_nav_traj": True,
            "apply_up_route": True,
            "apply_recon_iteration": False,
        },
        "detach": "use native: modal run --detach app.py <command> ...",
    }


@app.function(image=download_image, timeout=60)
def status() -> dict[str, Any]:
    return status_payload()


def _add_common_stage_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--scene", default="scene_from_008")
    parser.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    parser.add_argument("--nframe", type=int, default=16)
    parser.add_argument("--split-view-num", type=int, default=1)
    parser.add_argument("--wonder-topk", type=int, default=1)
    parser.add_argument("--recon-topk", type=int, default=0)
    parser.add_argument("--force-vlm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-nav-traj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-up-route", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-recon-iteration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vlm-mode", choices=["share", "split"], default="share")
    parser.add_argument("--vlm-mem-util", type=float, default=VLM_DEFAULT_MEM_UTIL)
    parser.add_argument("--vlm-max-model-len", type=int, default=VLM_MAX_MODEL_LEN)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="009 HY-World 2.0 world generation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印 pipeline / VLM 固定信息；纯本地")

    prepare = sub.add_parser("prepare", help="把 008 panorama 导入 scene")
    prepare.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    prepare.add_argument("--scene", default="scene_from_008")
    prepare.add_argument("--dry-run", action="store_true")

    download = sub.add_parser("download", help="下载 VLM / WorldStereo / WorldMirror 权重")
    download.add_argument("--which", default="vlm")
    download.add_argument("--dry-run", action="store_true")

    stage = sub.add_parser("stage", help="运行单个 stage 1..5")
    stage.add_argument("n", type=int, choices=[1,2,3,4,5])
    _add_common_stage_flags(stage)
    stage.add_argument("--max-steps", type=int, default=4000)
    stage.add_argument("--keep-vlm", action=argparse.BooleanOptionalAction, default=False)

    stage12 = sub.add_parser("stage12", help="Stage1+2 共用一次 VLM lifecycle")
    _add_common_stage_flags(stage12)

    smoke = sub.add_parser("smoke", help="prepare + download + stage12 + stage3..5")
    _add_common_stage_flags(smoke)
    smoke.add_argument("--max-steps", type=int, default=4000)
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def common_stage_plan(args: argparse.Namespace) -> dict[str, Any]:
    scene = args.scene.strip()
    from_008 = args.from_008.strip()
    if not scene or not from_008:
        raise ValueError("scene / from-008 不能为空")
    if args.nframe <= 0 or args.split_view_num <= 0:
        raise ValueError("nframe / split-view-num 必须 > 0")
    if args.wonder_topk < 0 or args.recon_topk < 0:
        raise ValueError("wonder-topk / recon-topk 必须 >= 0")
    if not 0 < args.vlm_mem_util < 1:
        raise ValueError("vlm-mem-util 必须在 0..1 之间")
    if args.vlm_max_model_len <= 0:
        raise ValueError("vlm-max-model-len 必须 > 0")
    return {
        "gpu": args.gpu,
        "scene": scene,
        "from_008": from_008,
        "split_view_num": args.split_view_num,
        "nframe": args.nframe,
        "wonder_topk": args.wonder_topk,
        "recon_topk": args.recon_topk,
        "force_vlm": args.force_vlm,
        "apply_nav_traj": args.apply_nav_traj,
        "apply_up_route": args.apply_up_route,
        "apply_recon_iteration": args.apply_recon_iteration,
        "vlm_mode": args.vlm_mode,
        "vlm_mem_util": args.vlm_mem_util,
        "vlm_max_model_len": args.vlm_max_model_len,
    }


def stage_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = {"action": args.command, **common_stage_plan(args)}
    if args.command == "stage":
        if args.max_steps <= 0:
            raise ValueError("--max-steps 必须 > 0")
        plan.update(stage=args.n, max_steps=args.max_steps, keep_vlm=args.keep_vlm)
    elif args.command == "smoke":
        if args.max_steps <= 0:
            raise ValueError("--max-steps 必须 > 0")
        plan["max_steps"] = args.max_steps
        plan["pipeline"] = ["prepare", "download:vlm", "download:worldstereo-dmd", "stage12", "stage3", "stage4", "stage5"]
    return plan


def _stage12_kwargs(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: plan[key] for key in (
        "scene", "from_008", "split_view_num", "nframe", "wonder_topk", "recon_topk",
        "force_vlm", "apply_nav_traj", "apply_up_route", "apply_recon_iteration",
        "vlm_mode", "vlm_mem_util", "vlm_max_model_len",
    )}


def _stage_kwargs(plan: dict[str, Any]) -> dict[str, Any]:
    kwargs = _stage12_kwargs(plan)
    kwargs.update(stage=plan["stage"], max_steps=plan["max_steps"], keep_vlm=plan["keep_vlm"])
    return kwargs


@app.local_entrypoint()
def main(*argv: str) -> None:
    args = parse_cli(argv)
    if args.command == "status":
        print(json.dumps(status_payload(), ensure_ascii=False, indent=2))
        return
    if args.command == "prepare":
        plan = {"action":"prepare", "from_008":args.from_008, "scene":args.scene}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2)); return
        print(json.dumps(prepare_scene.remote(from_008_run=args.from_008, scene_name=args.scene), ensure_ascii=False, indent=2)); return
    if args.command == "download":
        plan = {"action":"download", "which":args.which}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2)); return
        print(json.dumps(download_weights.remote(which=args.which), ensure_ascii=False, indent=2)); return

    try:
        plan = stage_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return

    if args.command == "stage12":
        result = run_stage12.with_options(gpu=plan["gpu"]).remote(gpu_label=plan["gpu"], **_stage12_kwargs(plan))
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    if args.command == "stage":
        result = run_stage.with_options(gpu=plan["gpu"]).remote(gpu_label=plan["gpu"], **_stage_kwargs(plan))
        print(json.dumps(result, ensure_ascii=False, indent=2)); return

    prepare_scene.remote(from_008_run=plan["from_008"], scene_name=plan["scene"])
    download_weights.remote(which="vlm")
    download_weights.remote(which="worldstereo-dmd")
    s12 = run_stage12.with_options(gpu=plan["gpu"]).remote(gpu_label=plan["gpu"], **_stage12_kwargs(plan))
    results = [s12]
    if s12.get("ok"):
        fn = run_stage.with_options(gpu=plan["gpu"])
        for stage_number in (3,4,5):
            meta = fn.remote(
                stage=stage_number, scene=plan["scene"], from_008=plan["from_008"],
                gpu_label=plan["gpu"], split_view_num=plan["split_view_num"],
                nframe=plan["nframe"], max_steps=plan["max_steps"],
            )
            results.append(meta)
            if not meta.get("ok"):
                break
    print(json.dumps({"pipeline": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
