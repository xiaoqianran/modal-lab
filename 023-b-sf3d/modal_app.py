# -*- coding: utf-8 -*-
"""023-b-sf3d — Stability SF3D (stable-fast-3d) image→GLB on Modal.

Fast feedforward mesh with UV-unwrapping + illumination disentanglement.
Default GPU: L40S · optional RTX-PRO-6000.
License: Stability AI Community.

Default weights: cocktailpeanut/sf3d (ungated mirror of official layout).
Official gated id: stabilityai/stable-fast-3d (pass --hf-model if access granted).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-sf3d"
DEFAULT_GPU = "L40S"
CODE_DIR = Path("/opt/src/stable-fast-3d")
WEIGHTS = Path("/data/weights")
OUTPUTS = Path("/data/outputs")
MESHES = OUTPUTS / "meshes"
BENCH = OUTPUTS / "benchmarks"
INPUTS = OUTPUTS / "inputs"

VOLUME_WEIGHTS = "modal-lab-sf3d-weights"
VOLUME_OUTPUTS = "modal-lab-sf3d-outputs"

SAMPLE_URL = (
    "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png"
)
# Official gated: stabilityai/stable-fast-3d
HF_MODEL = "cocktailpeanut/sf3d"
HF_MODEL_OFFICIAL = "stabilityai/stable-fast-3d"
UPSTREAM = "https://github.com/Stability-AI/stable-fast-3d"

GPU_PRICE = {
    "L40S": 0.000542,
    "RTX-PRO-6000": 0.000842,
    "A100-40GB": 0.000583,
    "H100": 0.001097,
    "L4": 0.000222,
}

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
app = modal.App(APP_NAME)

_APT = [
    "git",
    "build-essential",
    "g++",
    "gcc",
    "cmake",
    "ninja-build",
    "libgl1",
    "libglib2.0-0",
    "libgomp1",
    "libegl1",
    "wget",
    "curl",
    "ca-certificates",
]

_PIP_BASE = [
    "pip",
    "wheel",
    "setuptools==69.5.1",
    "ninja",
    "numpy==1.26.4",
    "einops==0.7.0",
    "jaxtyping==0.2.31",
    "omegaconf==2.3.0",
    "transformers==4.42.3",
    "open_clip_torch==2.24.0",
    "trimesh==4.4.1",
    "huggingface_hub[hf_transfer]>=0.26.0",
    "safetensors",
    "rembg[gpu]==2.0.57",
    "onnxruntime-gpu",
    "pynanoinstantmeshes==0.0.3",
    "gpytoolbox==0.2.0",
    "Pillow",
    "tqdm",
    "scipy",
    "fastapi",
]


def _env(arch: str) -> dict[str, str]:
    return {
        "CUDA_HOME": "/usr/local/cuda",
        "CC": "gcc",
        "CXX": "g++",
        "FORCE_CUDA": "1",
        "USE_CUDA": "1",
        "USE_NATIVE_ARCH": "0",
        "TORCH_CUDA_ARCH_LIST": arch,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(CODE_DIR),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }


def _build_ext_cmds() -> list[str]:
    return [
        "mkdir -p /opt/src && cd /opt/src && "
        f"git clone --depth 1 {UPSTREAM}.git stable-fast-3d",
        f"cd {CODE_DIR}/texture_baker && "
        "USE_CUDA=1 USE_NATIVE_ARCH=0 FORCE_CUDA=1 "
        "pip install --no-build-isolation --no-cache-dir .",
        f"cd {CODE_DIR}/uv_unwrapper && "
        "USE_NATIVE_ARCH=0 pip install --no-build-isolation --no-cache-dir .",
        "python -c \"import torch,texture_baker,uv_unwrapper,open_clip; "
        "print('torch', torch.__version__, 'ext ok')\"",
    ]


image_l40s = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(*_APT)
    .pip_install(*_PIP_BASE)
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .env(_env("8.9"))
    .run_commands(*_build_ext_cmds())
)

image_pro6000 = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(*_APT)
    .pip_install(*_PIP_BASE)
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(_env("12.0"))
    .run_commands(*_build_ext_cmds())
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
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


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
        "cli_get": (
            f"modal volume get {VOLUME_OUTPUTS} meshes/{name}{src.suffix} "
            f"./{name}{src.suffix}"
        ),
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
    texture_resolution: int,
    foreground_ratio: float,
    hf_model: str = HF_MODEL,
) -> dict:
    import rembg
    import torch
    from PIL import Image

    sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)
    _ensure_dirs()

    from sf3d.system import SF3D
    from sf3d.utils import remove_background, resize_foreground

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu_name = torch.cuda.get_device_name(0)
    price = GPU_PRICE.get(gpu_label, GPU_PRICE["L40S"])
    device = "cuda"

    work = Path("/tmp/sf3d_work")
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
        if torch.cuda.is_available():
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / (1024 * 1024))

    print("[model] load", hf_model, flush=True)
    model = SF3D.from_pretrained(
        hf_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()
    _peak()
    t_load = time.time() - t0

    t1 = time.time()
    rembg_session = rembg.new_session()
    image = remove_background(Image.open(img_path).convert("RGBA"), rembg_session)
    image = resize_foreground(image, foreground_ratio)
    image.save(work / "input_fg.png")
    t_pre = time.time() - t1
    _peak()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t2 = time.time()
    with torch.no_grad():
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        ):
            mesh, _glob = model.run_image(
                image,
                bake_resolution=texture_resolution,
                remesh="none",
                vertex_count=-1,
            )
    torch.cuda.synchronize()
    t_infer = time.time() - t2
    _peak()

    t3 = time.time()
    out_path = work / f"{_safe(output_name)}.glb"
    if isinstance(mesh, (list, tuple)):
        mesh[0].export(str(out_path), include_normals=True)
    else:
        mesh.export(str(out_path), include_normals=True)
    t_mesh = time.time() - t3
    total = time.time() - t0
    _peak()

    if not out_path.is_file():
        raise RuntimeError(f"mesh not written: {out_path}")

    meta = {
        "model": hf_model,
        "model_official": HF_MODEL_OFFICIAL,
        "upstream": "Stability-AI/stable-fast-3d",
        "license": "Stability AI Community License",
        "gpu_request": gpu_label,
        "gpu_actual": gpu_name,
        "output_name": _safe(output_name),
        "texture_resolution": texture_resolution,
        "foreground_ratio": foreground_ratio,
        "remesh": "none",
        "format": "glb",
        "uv_textured": True,
        "seconds_load": round(t_load, 2),
        "seconds_preprocess": round(t_pre, 2),
        "seconds_infer": round(t_infer, 2),
        "seconds_export": round(t_mesh, 2),
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
    image=image_l40s,
    gpu=DEFAULT_GPU,
    timeout=45 * 60,
    memory=32768,
    cpu=4,
    volumes={str(WEIGHTS): weights_vol, str(OUTPUTS): outputs_vol},
    secrets=[hf_secret],
)
class SF3DWorker:
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
            "model_official": HF_MODEL_OFFICIAL,
            "gpu_request": DEFAULT_GPU,
            "torch": str(torch.__version__),
            "cuda": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "smi": _smi(),
            "code_ok": (CODE_DIR / "sf3d").is_dir(),
            "hf_token": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            ),
        }
        try:
            import texture_baker  # noqa: F401
            import uv_unwrapper  # noqa: F401

            info["ext_ok"] = True
        except Exception as e:
            info["ext_ok"] = False
            info["ext_error"] = repr(e)
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
        texture_resolution: int = 1024,
        foreground_ratio: float = 0.85,
        hf_model: str = HF_MODEL,
    ) -> dict:
        return _run_i2v(
            gpu_label=DEFAULT_GPU,
            image_url=image_url,
            output_name=output_name,
            texture_resolution=texture_resolution,
            foreground_ratio=foreground_ratio,
            hf_model=hf_model,
        )


@app.cls(
    image=image_pro6000,
    gpu="RTX-PRO-6000",
    timeout=45 * 60,
    memory=32768,
    cpu=4,
    volumes={str(WEIGHTS): weights_vol, str(OUTPUTS): outputs_vol},
    secrets=[hf_secret],
)
class SF3DPro6000:
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
            "model_official": HF_MODEL_OFFICIAL,
            "gpu_request": "RTX-PRO-6000",
            "torch": str(torch.__version__),
            "cuda": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "smi": _smi(),
            "code_ok": (CODE_DIR / "sf3d").is_dir(),
            "hf_token": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            ),
        }
        try:
            import texture_baker  # noqa: F401
            import uv_unwrapper  # noqa: F401

            info["ext_ok"] = True
        except Exception as e:
            info["ext_ok"] = False
            info["ext_error"] = repr(e)
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
        texture_resolution: int = 1024,
        foreground_ratio: float = 0.85,
        hf_model: str = HF_MODEL,
    ) -> dict:
        return _run_i2v(
            gpu_label="RTX-PRO-6000",
            image_url=image_url,
            output_name=output_name,
            texture_resolution=texture_resolution,
            foreground_ratio=foreground_ratio,
            hf_model=hf_model,
        )


@app.local_entrypoint()
def main(
    action: str = "probe",
    gpu: str = "L40S",
    output_name: str = "",
    image_url: str = SAMPLE_URL,
    texture_resolution: int = 1024,
    hf_model: str = HF_MODEL,
):
    g = gpu.upper().replace("_", "-")
    use_pro = g in {"RTX-PRO-6000", "PRO-6000", "PRO6000"}
    worker = SF3DPro6000() if use_pro else SF3DWorker()
    default_name = "smoke_pro6000" if use_pro else "smoke_l40s"
    name = output_name or default_name
    if action in {"probe", "status"}:
        print(worker.probe.remote())
    elif action in {"smoke", "i2v"}:
        print(
            worker.image_to_3d.remote(
                image_url=image_url,
                output_name=name,
                texture_resolution=texture_resolution,
                hf_model=hf_model,
            )
        )
    else:
        raise SystemExit(f"unknown action: {action}")
