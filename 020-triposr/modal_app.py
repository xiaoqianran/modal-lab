# -*- coding: utf-8 -*-
"""020-triposr — Stability/Tripo TripoSR image→mesh on Modal.

Speed baseline for open image-to-3D.
Default GPU: L40S · optional RTX-PRO-6000.
MIT · https://github.com/VAST-AI-Research/TripoSR
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-triposr"
DEFAULT_GPU = "L40S"
CODE_DIR = Path("/opt/src/TripoSR")
WEIGHTS = Path("/weights")
OUTPUTS = Path("/outputs")
MESHES = OUTPUTS / "meshes"
BENCH = OUTPUTS / "benchmarks"
INPUTS = OUTPUTS / "inputs"

VOLUME_WEIGHTS = "modal-lab-triposr-weights"
VOLUME_OUTPUTS = "modal-lab-triposr-outputs"

SAMPLE_URL = (
    "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png"
)
HF_MODEL = "stabilityai/TripoSR"

GPU_PRICE = {
    "L40S": 0.000542,
    "RTX-PRO-6000": 0.000842,
    "A100-40GB": 0.000583,
    "H100": 0.001097,
    "L4": 0.000222,
}

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
app = modal.App(APP_NAME)

# Shared pip deps. numpy pinned <2 — trimesh 4.0.x uses ndarray.ptp() removed in np2.
_COMMON_PIP = [
    "pip",
    "wheel",
    "setuptools>=49.6.0",
    "ninja",
    "cmake",
    "scikit-build-core",
    "pyproject-metadata",
    "pathspec",
    "pybind11",
    "numpy==1.26.4",
    "huggingface_hub[hf_transfer]>=0.26.0",
    "omegaconf==2.3.0",
    "Pillow==10.1.0",
    "einops==0.7.0",
    "transformers==4.40.2",
    "trimesh==4.0.5",
    "rembg",
    "onnxruntime",
    "imageio",
    "imageio-ffmpeg",
    "xatlas==0.0.9",
    "moderngl==5.10.0",
    "fastapi",
]

_APT = [
    "git",
    "build-essential",
    "g++",
    "gcc",
    "cmake",
    "ninja-build",
    "libgl1",
    "libglib2.0-0",
    "libegl1",
    "wget",
    "curl",
    "ca-certificates",
]


def _mcubes_cmd(arch: str) -> str:
    return (
        f"export CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ FORCE_CUDA=1 "
        f"TORCH_CUDA_ARCH_LIST={arch} "
        "CMAKE_PREFIX_PATH=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())') "
        "CMAKE_ARGS=\"-DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc "
        "-Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')\" && "
        "pip install --no-build-isolation --no-cache-dir "
        "'git+https://github.com/tatsy/torchmcubes.git'"
    )


# L40S / Ada sm_89: torch 2.5 + cu124
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(*_APT)
    .pip_install(*_COMMON_PIP)
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CC": "gcc",
            "CXX": "g++",
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "HF_HOME": str(WEIGHTS / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(CODE_DIR),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .run_commands(
        "mkdir -p /opt/src && cd /opt/src && "
        "git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git",
        _mcubes_cmd("8.9"),
        "python -c 'import torch,torchmcubes,numpy; print(torch.__version__, numpy.__version__, \"mcubes ok\")'",
    )
)

# PRO 6000 / Blackwell sm_120: torch 2.11 + cu128
image_pro6000 = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(*_APT)
    .pip_install(*_COMMON_PIP)
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CC": "gcc",
            "CXX": "g++",
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "HF_HOME": str(WEIGHTS / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(CODE_DIR),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .run_commands(
        "mkdir -p /opt/src && cd /opt/src && "
        "git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git",
        _mcubes_cmd("12.0"),
        "python -c 'import torch,torchmcubes,numpy; print(torch.__version__, numpy.__version__, \"mcubes ok\")'",
    )
)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def _safe(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return s or "mesh"


def _ensure_dirs() -> None:
    for p in (WEIGHTS / "hf" / "hub", MESHES, BENCH, INPUTS):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(WEIGHTS / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(WEIGHTS / "hf" / "hub")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def _smi() -> dict[str, Any] | None:
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
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "mem_used_mib": float(parts[1]),
            "mem_total_mib": float(parts[2]),
            "util_gpu_pct": float(parts[3]),
        }
    except Exception as e:
        return {"error": repr(e)}


def _publish(src: Path, name: str, meta: dict, input_path: Path | None = None) -> dict:
    try:
        outputs_vol.reload()
    except Exception:
        pass
    MESHES.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    INPUTS.mkdir(parents=True, exist_ok=True)
    if not src.is_file() or src.stat().st_size < 500:
        raise RuntimeError(f"invalid mesh: {src}")
    dest = MESHES / f"{name}{src.suffix}"
    latest = MESHES / f"latest{src.suffix}"
    shutil.copy2(src, dest)
    shutil.copy2(src, latest)
    size = dest.stat().st_size
    if input_path and input_path.is_file():
        ip = INPUTS / f"{name}{input_path.suffix.lower() or '.png'}"
        shutil.copy2(input_path, ip)
        meta["input_volume_path"] = f"inputs/{ip.name}"
    payload = {
        **meta,
        "volume_name": VOLUME_OUTPUTS,
        "volume_paths": {
            "mesh": f"meshes/{name}{src.suffix}",
            "latest": f"meshes/latest{src.suffix}",
            "meta": f"meshes/{name}_meta.json",
            "benchmark": f"benchmarks/{name}.json",
        },
        "bytes": size,
        "cli_get": f"modal volume get {VOLUME_OUTPUTS} meshes/{name}{src.suffix} ./{name}{src.suffix}",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (MESHES / f"{name}_meta.json").write_text(text, encoding="utf-8")
    (MESHES / "latest_meta.json").write_text(text, encoding="utf-8")
    (BENCH / f"{name}.json").write_text(text, encoding="utf-8")
    outputs_vol.commit()
    print(f"[VOLUME] {VOLUME_OUTPUTS}/meshes/{name}{src.suffix} ({size} bytes)", flush=True)
    return payload


def _run_i2v(
    *,
    gpu_label: str,
    image_url: str,
    output_name: str,
    mc_resolution: int,
    model_format: str,
    foreground_ratio: float,
) -> dict:
    import numpy as np
    import rembg
    import torch
    from PIL import Image

    sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)
    _ensure_dirs()

    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu_name = torch.cuda.get_device_name(0)
    price = GPU_PRICE.get(gpu_label, GPU_PRICE["L40S"])

    work = Path("/tmp/triposr_work")
    work.mkdir(parents=True, exist_ok=True)
    ext = Path(image_url.split("?")[0]).suffix or ".png"
    img_path = work / f"input{ext}"
    print(f"[i2v] fetch {image_url}", flush=True)
    urllib.request.urlretrieve(image_url, img_path)

    t0 = time.time()
    peak_mib = 0.0

    def _peak() -> None:
        nonlocal peak_mib
        q = _smi()
        if q and "mem_used_mib" in q:
            peak_mib = max(peak_mib, float(q["mem_used_mib"]))

    print("[model] load", HF_MODEL, flush=True)
    model = TSR.from_pretrained(
        HF_MODEL,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to("cuda")
    _peak()
    t_load = time.time() - t0

    t1 = time.time()
    rembg_session = rembg.new_session()
    image = remove_background(Image.open(img_path), rembg_session)
    image = resize_foreground(image, foreground_ratio)
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    image = Image.fromarray((arr * 255.0).astype(np.uint8))
    t_pre = time.time() - t1
    _peak()

    t2 = time.time()
    with torch.no_grad():
        scene_codes = model([image], device="cuda")
    torch.cuda.synchronize()
    t_infer = time.time() - t2
    _peak()

    t3 = time.time()
    meshes = model.extract_mesh(scene_codes, True, resolution=mc_resolution)
    out_path = work / f"{_safe(output_name)}.{model_format}"
    meshes[0].export(str(out_path))
    torch.cuda.synchronize()
    t_mesh = time.time() - t3
    _peak()
    total = time.time() - t0

    if not out_path.is_file():
        raise RuntimeError(f"mesh not written: {out_path}")

    meta = {
        "model": HF_MODEL,
        "upstream": "VAST-AI-Research/TripoSR",
        "license": "MIT",
        "gpu_request": gpu_label,
        "gpu_actual": gpu_name,
        "output_name": _safe(output_name),
        "bake_texture": False,
        "vertex_colors": True,
        "mc_resolution": mc_resolution,
        "format": out_path.suffix.lstrip("."),
        "seconds_load": round(t_load, 2),
        "seconds_preprocess": round(t_pre, 2),
        "seconds_infer": round(t_infer, 2),
        "seconds_mesh": round(t_mesh, 2),
        "seconds_total": round(total, 2),
        "peak_vram_gb": round(peak_mib / 1024.0, 2) if peak_mib else None,
        "est_cost_usd": round(total * price, 4),
        "price_per_sec_usd": price,
        "image_url": image_url,
        "smi_end": _smi(),
    }
    pub = _publish(out_path, _safe(output_name), meta, input_path=img_path)
    try:
        weights_vol.commit()
    except Exception:
        pass
    result = {
        "ok": True,
        "gpu_actual": gpu_name,
        "seconds_total": meta["seconds_total"],
        "seconds_infer": meta["seconds_infer"],
        "seconds_mesh": meta["seconds_mesh"],
        "peak_vram_gb": meta["peak_vram_gb"],
        "est_cost_usd": meta["est_cost_usd"],
        "bytes": pub["bytes"],
        "volume": VOLUME_OUTPUTS,
        "volume_file": pub["volume_paths"]["mesh"],
        "cli_get": pub["cli_get"],
    }
    print(json.dumps(result, indent=2), flush=True)
    return _jsonable(result)


@app.cls(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=45 * 60,
    memory=16384,
    cpu=4,
    volumes={str(WEIGHTS): weights_vol, str(OUTPUTS): outputs_vol},
)
class TripoSRWorker:
    @modal.enter()
    def enter(self) -> None:
        _ensure_dirs()
        sys.path.insert(0, str(CODE_DIR))

    @modal.method()
    def probe(self) -> dict:
        import torch

        info = {
            "app": APP_NAME,
            "model": HF_MODEL,
            "gpu_request": DEFAULT_GPU,
            "torch": str(torch.__version__),
            "cuda": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "smi": _smi(),
            "code_ok": (CODE_DIR / "tsr").is_dir(),
        }
        if torch.cuda.is_available():
            a = torch.randn(256, 256, device="cuda")
            b = a @ a
            torch.cuda.synchronize()
            info["matmul_ok"] = bool(int(b.numel()) > 0)
        print(json.dumps(info, indent=2), flush=True)
        return _jsonable(info)

    @modal.method()
    def image_to_3d(
        self,
        image_url: str = SAMPLE_URL,
        output_name: str = "smoke_l40s",
        mc_resolution: int = 256,
        model_format: str = "glb",
        foreground_ratio: float = 0.85,
    ) -> dict:
        return _run_i2v(
            gpu_label=DEFAULT_GPU,
            image_url=image_url,
            output_name=output_name,
            mc_resolution=mc_resolution,
            model_format=model_format,
            foreground_ratio=foreground_ratio,
        )


@app.cls(
    image=image_pro6000,
    gpu="RTX-PRO-6000",
    timeout=45 * 60,
    memory=16384,
    cpu=4,
    volumes={str(WEIGHTS): weights_vol, str(OUTPUTS): outputs_vol},
)
class TripoSRPro6000:
    @modal.enter()
    def enter(self) -> None:
        _ensure_dirs()
        sys.path.insert(0, str(CODE_DIR))

    @modal.method()
    def probe(self) -> dict:
        import torch

        info = {
            "app": APP_NAME,
            "model": HF_MODEL,
            "gpu_request": "RTX-PRO-6000",
            "torch": str(torch.__version__),
            "cuda": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "smi": _smi(),
            "code_ok": (CODE_DIR / "tsr").is_dir(),
        }
        if torch.cuda.is_available():
            a = torch.randn(256, 256, device="cuda")
            b = a @ a
            torch.cuda.synchronize()
            info["matmul_ok"] = bool(int(b.numel()) > 0)
        print(json.dumps(info, indent=2), flush=True)
        return _jsonable(info)

    @modal.method()
    def image_to_3d(
        self,
        image_url: str = SAMPLE_URL,
        output_name: str = "smoke_pro6000",
        mc_resolution: int = 256,
        model_format: str = "glb",
        foreground_ratio: float = 0.85,
    ) -> dict:
        return _run_i2v(
            gpu_label="RTX-PRO-6000",
            image_url=image_url,
            output_name=output_name,
            mc_resolution=mc_resolution,
            model_format=model_format,
            foreground_ratio=foreground_ratio,
        )


@app.local_entrypoint()
def main(
    action: str = "probe",
    gpu: str = "L40S",
    output_name: str = "",
    image_url: str = SAMPLE_URL,
):
    g = gpu.upper().replace("_", "-")
    use_pro = g in {"RTX-PRO-6000", "PRO-6000", "PRO6000"}
    worker = TripoSRPro6000() if use_pro else TripoSRWorker()
    default_name = "smoke_pro6000" if use_pro else "smoke_l40s"
    name = output_name or default_name
    if action in {"probe", "status"}:
        print(worker.probe.remote())
    elif action in {"smoke", "i2v"}:
        print(
            worker.image_to_3d.remote(
                image_url=image_url,
                output_name=name,
                model_format="glb",
            )
        )
    else:
        raise SystemExit(f"unknown action: {action}")
