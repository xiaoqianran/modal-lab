# -*- coding: utf-8 -*-
"""021-trellis2 — Microsoft TRELLIS.2-4B image→GLB on Modal.

Quality mainline (MIT). Default GPU L40S (sm_89 source wheels);
optional RTX-PRO-6000 (sm_120 / cu128). Extension stack mirrors
005-v2 / 005-v3 Plan A without natten.

Sparse attention requires flash_attn or xformers (not sdpa) —
we use xformers prebuilt wheels.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-trellis2"
CODE_DIR = Path("/opt/src/TRELLIS.2")
WHEELS = Path("/wheels")
WEIGHTS = Path("/weights")
OUTPUTS = Path("/outputs")
MESHES = OUTPUTS / "meshes"
BENCH = OUTPUTS / "benchmarks"
INPUTS = OUTPUTS / "inputs"

VOLUME_WHEELS = "modal-lab-trellis2-wheels"
VOLUME_WEIGHTS = "modal-lab-trellis2-weights"
VOLUME_OUTPUTS = "modal-lab-trellis2-outputs"

HF_MODEL = "microsoft/TRELLIS.2-4B"
DINO_MIRROR = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
REMBG_MIRROR = "ZhengPeng7/BiRefNet"
AUX_HF_REPOS = (DINO_MIRROR, REMBG_MIRROR)

SAMPLE_URL = (
    "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png"
)

GPU_PRICE = {
    "L40S": 0.000542,
    "RTX-PRO-6000": 0.000842,
}

wheels_vol = modal.Volume.from_name(VOLUME_WHEELS, create_if_missing=True)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
app = modal.App(APP_NAME)

_APT_COMMON = [
    "git",
    "build-essential",
    "g++",
    "gcc",
    "clang",
    "ninja-build",
    "cmake",
    "wget",
    "curl",
    "libgl1",
    "libeigen3-dev",
    "libjpeg-dev",
    "ffmpeg",
    "ca-certificates",
]

_PIP_COMMON = [
    "pip",
    "wheel",
    "setuptools",
    "ninja",
    "packaging",
    "numpy",
    "pyyaml",
    "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
    "safetensors",
    "einops",
    "scipy",
    "pillow==12.0.0",
    "imageio==2.37.2",
    "imageio-ffmpeg==0.6.0",
    "tqdm==4.67.1",
    "easydict==1.13",
    "opencv-python-headless==4.12.0.88",
    "trimesh==4.10.1",
    "transformers==4.57.3",
    "zstandard==0.25.0",
    "kornia==0.8.2",
    "timm==1.0.22",
    "diffusers==0.37.1",
    "accelerate==1.13.0",
    "plyfile==1.1.3",
    "rembg",
    "onnxruntime",
    "fastapi",
    "lpips",
]

_CLONE = (
    "mkdir -p /opt/src && cd /opt/src && "
    "git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git && "
    "git clone --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git && "
    "git clone --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git && "
    "git clone --depth 1 -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git && "
    "git clone --depth 1 -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git"
)

_UTILS3D = (
    "pip install --no-deps "
    "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl "
    "|| pip install --no-cache-dir "
    "'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8'"
)

# Sparse attn needs flash_attn|xformers (sdpa not supported for sparse path)
_ATTN_ENV = {
    "ATTN_BACKEND": "xformers",
    "SPARSE_ATTN_BACKEND": "xformers",
    "SPARSE_CONV_BACKEND": "flex_gemm",
}

image_l40s = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(*_APT_COMMON, "libglib2.0-0")
    .pip_install(*_PIP_COMMON)
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "triton==3.2.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "xformers==0.0.29.post3",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "8.9",
            **_ATTN_ENV,
            "CUDA_HOME": "/usr/local/cuda",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "4",
            "CC": "gcc",
            "CXX": "g++",
            "HF_HOME": str(WEIGHTS / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
            "TORCH_HOME": str(WEIGHTS / "torch"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(CODE_DIR),
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CPLUS_INCLUDE_PATH": "/usr/include/eigen3",
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(CODE_DIR / "autotune_cache.json"),
            "FLEX_GEMM_AUTOTUNER_VERBOSE": "0",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .run_commands(
        _CLONE,
        _UTILS3D,
        "python -m pip install --no-cache-dir "
        "'huggingface_hub[hf_transfer]>=0.34.0,<1.0' 'transformers==4.57.3'",
        "python -c 'import xformers; print(\"xformers\", xformers.__version__)'",
    )
    .add_local_dir(
        str(Path(__file__).parent / "scripts"),
        remote_path="/opt/trellis2_scripts",
        copy=True,
    )
)

image_pro6000 = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.10",
    )
    .apt_install(*_APT_COMMON, "libglib2.0-0t64")
    .pip_install(*_PIP_COMMON)
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # xformers for torch 2.11+cu128 — pin loosely if exact wheel missing
    .pip_install(
        "xformers",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "12.0",
            **_ATTN_ENV,
            "CUDA_HOME": "/usr/local/cuda",
            "CUDACXX": "/usr/local/cuda/bin/nvcc",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "4",
            "CC": "gcc",
            "CXX": "g++",
            "HF_HOME": str(WEIGHTS / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
            "TORCH_HOME": str(WEIGHTS / "torch"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(CODE_DIR),
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CPLUS_INCLUDE_PATH": "/usr/include/eigen3",
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(CODE_DIR / "autotune_cache.json"),
            "FLEX_GEMM_AUTOTUNER_VERBOSE": "0",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .run_commands(
        _CLONE,
        _UTILS3D,
        "python -m pip install --no-cache-dir "
        "'huggingface_hub[hf_transfer]>=0.34.0,<1.0' 'transformers==4.57.3'",
        "python -c 'import xformers; print(\"xformers\", xformers.__version__)'",
    )
    .add_local_dir(
        str(Path(__file__).parent / "scripts"),
        remote_path="/opt/trellis2_scripts",
        copy=True,
    )
)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def _safe(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return s or "mesh"


def _wheel_dir(sm_tag: str, torch_tag: str) -> Path:
    return WHEELS / sm_tag / torch_tag


def _build_env(arch: str) -> dict:
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = arch
    env["CUDA_HOME"] = env.get("CUDA_HOME", "/usr/local/cuda")
    env["CPLUS_INCLUDE_PATH"] = "/usr/include/eigen3"
    env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
    env["FORCE_CUDA"] = "1"
    env["CC"] = "gcc"
    env["CXX"] = "g++"
    env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"
    shim = Path("/tmp/bin-shim")
    shim.mkdir(exist_ok=True)
    for name, target in (("clang++", "/usr/bin/g++"), ("clang", "/usr/bin/gcc")):
        link = shim / name
        if not link.exists():
            try:
                link.symlink_to(target)
            except FileExistsError:
                pass
    env["PATH"] = f"{shim}:{env['PATH']}"
    return env


def _existing_wheels(out: Path, name: str) -> list[Path]:
    if name == "nvdiffrec_render":
        return list(out.glob("*nvdiffrec*")) + list(out.glob("*renderutils*"))
    if name == "o_voxel":
        return list(out.glob("o_voxel*.whl")) + list(out.glob("o-voxel*.whl"))
    return list(out.glob(f"*{name}*.whl"))


def _pip_wheel(args: list[str], out: Path, env: dict) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        *args,
        "--no-build-isolation",
        "--no-deps",
        "-w",
        str(out),
    ]
    print("RUN:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env)


def _build_extensions(out: Path, arch: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = _build_env(arch)
    steps: list[tuple[str, list[str]]] = [
        ("nvdiffrast", ["/opt/src/nvdiffrast"]),
        ("nvdiffrec_render", ["/opt/src/nvdiffrec"]),
        ("flex_gemm", ["/opt/src/FlexGEMM"]),
        ("cumesh", ["/opt/src/CuMesh"]),
        ("o_voxel", [str(CODE_DIR / "o-voxel")]),
    ]
    log: list[dict] = []
    for name, args in steps:
        existing = _existing_wheels(out, name)
        if existing:
            log.append({"pkg": name, "skipped": True, "files": [p.name for p in existing]})
            continue
        before = {p.name for p in out.glob("*.whl")}
        rc = _pip_wheel(args, out, env)
        after = sorted(p.name for p in out.glob("*.whl") if p.name not in before)
        log.append({"pkg": name, "returncode": rc, "new_files": after})
        wheels_vol.commit()
        if rc != 0:
            return {
                "ok": False,
                "failed": name,
                "log": log,
                "files": sorted(p.name for p in out.glob("*.whl")),
            }
    files = sorted(p.name for p in out.glob("*.whl"))
    wheels_vol.commit()
    return {"ok": True, "wheel_dir": str(out), "files": files, "log": log}


def _install_wheels(out: Path) -> list[str]:
    whls = sorted(out.glob("*.whl"))
    if not whls:
        raise RuntimeError(f"no wheels in {out}; run build first")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        *[str(w) for w in whls],
    ]
    print("INSTALL:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    return [w.name for w in whls]


def _ensure_dirs() -> None:
    for sub in ("hf", "hf/hub", "torch", "TRELLIS.2-4B"):
        (WEIGHTS / sub).mkdir(parents=True, exist_ok=True)
    for p in (MESHES, BENCH, INPUTS):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(WEIGHTS / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(WEIGHTS / "hf" / "hub")
    os.environ["TORCH_HOME"] = str(WEIGHTS / "torch")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(CODE_DIR / "autotune_cache.json")
    )


def _model_path() -> Path:
    return WEIGHTS / "TRELLIS.2-4B"


def _model_ready() -> bool:
    root = _model_path()
    return (root / "pipeline.json").is_file() or any(root.rglob("*.safetensors"))


def _patch_pipeline(model_path: str | Path) -> dict:
    pj = Path(model_path) / "pipeline.json"
    if not pj.is_file():
        return {"ok": False, "reason": "no pipeline.json"}
    data = json.loads(pj.read_text(encoding="utf-8"))
    args = data.get("args") or data
    changed: list[str] = []
    icm = args.get("image_cond_model") if isinstance(args, dict) else None
    if isinstance(icm, dict):
        a = icm.setdefault("args", {})
        old = a.get("model_name")
        if old and "facebook/dinov3" in str(old):
            a["model_name"] = DINO_MIRROR
            changed.append(f"dino {old} → {DINO_MIRROR}")
    rembg = args.get("rembg_model") if isinstance(args, dict) else None
    if isinstance(rembg, dict):
        a = rembg.setdefault("args", {})
        old = a.get("model_name")
        if old and ("RMBG" in str(old) or "briaai" in str(old)):
            a["model_name"] = REMBG_MIRROR
            rembg["name"] = "BiRefNet"
            changed.append(f"rembg {old} → {REMBG_MIRROR}")
    if changed:
        pj.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        print("[patch]", "; ".join(changed), flush=True)
        try:
            weights_vol.commit()
        except Exception as e:
            print(f"[patch] commit warn: {e!r}", flush=True)
    return {"ok": True, "changed": changed, "path": str(pj)}


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


class VramSampler:
    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_mib = 0.0
        self.peak_util = 0.0
        self.samples: list[dict] = []

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        total = name = None
        if self.samples:
            total = self.samples[-1].get("mem_total_mib")
            name = self.samples[-1].get("name")
        return {
            "gpu_name_smi": name,
            "peak_mem_used_mib": round(self.peak_used_mib, 1),
            "peak_mem_used_gb": round(self.peak_used_mib / 1024.0, 2)
            if self.peak_used_mib
            else None,
            "mem_total_mib": total,
            "peak_util_gpu_pct": round(self.peak_util, 1),
            "n_samples": len(self.samples),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            q = _smi()
            if q and "mem_used_mib" in q:
                self.samples.append({"t": time.time(), **q})
                self.peak_used_mib = max(self.peak_used_mib, q["mem_used_mib"])
                self.peak_util = max(self.peak_util, q.get("util_gpu_pct") or 0.0)
            self._stop.wait(self.interval_s)


def _publish(src: Path, name: str, meta: dict, input_path: Path | None = None) -> dict:
    try:
        outputs_vol.reload()
    except Exception:
        pass
    MESHES.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    INPUTS.mkdir(parents=True, exist_ok=True)
    if not src.is_file() or src.stat().st_size < 1000:
        raise RuntimeError(f"invalid mesh: {src}")
    dest = MESHES / f"{name}.glb"
    latest = MESHES / "latest.glb"
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
            "mesh": f"meshes/{name}.glb",
            "latest": "meshes/latest.glb",
            "meta": f"meshes/{name}_meta.json",
            "benchmark": f"benchmarks/{name}.json",
        },
        "bytes": size,
        "cli_get": f"modal volume get {VOLUME_OUTPUTS} meshes/{name}.glb ./{name}.glb",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (MESHES / f"{name}_meta.json").write_text(text, encoding="utf-8")
    (MESHES / "latest_meta.json").write_text(text, encoding="utf-8")
    (BENCH / f"{name}.json").write_text(text, encoding="utf-8")
    outputs_vol.commit()
    print(f"[VOLUME] {VOLUME_OUTPUTS}/meshes/{name}.glb ({size} bytes)", flush=True)
    return payload


def _download_weights(force: bool = False) -> dict:
    from huggingface_hub import snapshot_download

    _ensure_dirs()
    root = _model_path()
    if force and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    if not _model_ready() or force:
        print(f"[download] {HF_MODEL} → {root}", flush=True)
        snapshot_download(repo_id=HF_MODEL, local_dir=str(root))
    else:
        print(f"[download] skip main {root}", flush=True)
    patch = _patch_pipeline(root)
    aux = []
    for repo in AUX_HF_REPOS:
        try:
            print(f"[download] aux {repo}", flush=True)
            p = snapshot_download(repo_id=repo)
            aux.append({"repo": repo, "ok": True, "path": p})
        except Exception as e:
            aux.append({"repo": repo, "ok": False, "error": repr(e)})
            print(f"[download] aux FAIL {repo}: {e!r}", flush=True)
    weights_vol.commit()
    return {
        "ok": True,
        "path": str(root),
        "ready": _model_ready(),
        "patch": patch,
        "aux": aux,
    }


def _run_i2v(
    *,
    gpu_label: str,
    arch: str,
    wheel_out: Path,
    image_url: str,
    output_name: str,
    pipeline_type: str,
    seed: int,
    texture_size: int,
    decimation_target: int,
) -> dict:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    price = GPU_PRICE.get(gpu_label, GPU_PRICE["L40S"])

    installed = _install_wheels(wheel_out)
    print("[wheels]", installed, flush=True)

    import o_voxel  # noqa: F401

    for pkg in ("flex_gemm", "o_voxel", "cumesh", "xformers"):
        __import__(pkg)
    print("[import] flex_gemm o_voxel cumesh xformers OK", flush=True)

    _ensure_dirs()
    if not _model_ready():
        _download_weights(force=False)
    else:
        _patch_pipeline(_model_path())
        from huggingface_hub import snapshot_download

        for repo in AUX_HF_REPOS:
            try:
                snapshot_download(repo_id=repo)
            except Exception as e:
                print(f"[aux] warn {repo}: {e!r}", flush=True)

    model_path = str(_model_path())
    work = Path("/tmp/trellis2_work")
    work.mkdir(parents=True, exist_ok=True)
    ext = Path(image_url.split("?")[0]).suffix or ".png"
    img_path = work / f"input{ext}"
    print(f"[i2v] fetch {image_url}", flush=True)
    urllib.request.urlretrieve(image_url, img_path)

    # Must set backends BEFORE importing trellis2 sparse modules
    os.environ["ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)

    # Force re-read sparse config if already imported
    import trellis2.modules.sparse.config as sparse_cfg

    sparse_cfg.ATTN = "xformers"
    sparse_cfg.CONV = "flex_gemm"

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    t0 = time.time()
    sampler = VramSampler(1.5)
    sampler.start()
    print("[model] load", model_path, flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    t_load = time.time() - t0

    t1 = time.time()
    image = Image.open(img_path)
    print(f"[i2v] run pipeline_type={pipeline_type} seed={seed}", flush=True)
    mesh = pipeline.run(
        image,
        seed=seed,
        pipeline_type=pipeline_type,
    )[0]
    torch.cuda.synchronize()
    t_infer = time.time() - t1

    t2 = time.time()
    try:
        mesh.simplify(16777216)
    except Exception as e:
        print(f"[mesh] simplify warn: {e!r}", flush=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    out_path = work / f"{_safe(output_name)}.glb"
    try:
        glb.export(str(out_path), extension_webp=True)
    except TypeError:
        glb.export(str(out_path))
    torch.cuda.synchronize()
    t_mesh = time.time() - t2
    vram = sampler.stop()
    total = time.time() - t0

    meta = {
        "model": HF_MODEL,
        "upstream": "microsoft/TRELLIS.2",
        "license": "MIT",
        "gpu_request": gpu_label,
        "gpu_actual": gpu_name,
        "capability": list(cap),
        "arch": arch,
        "output_name": _safe(output_name),
        "pipeline_type": pipeline_type,
        "seed": seed,
        "texture_size": texture_size,
        "decimation_target": decimation_target,
        "attn_backend": "xformers",
        "dino_mirror": DINO_MIRROR,
        "rembg_mirror": REMBG_MIRROR,
        "seconds_load": round(t_load, 2),
        "seconds_infer": round(t_infer, 2),
        "seconds_mesh": round(t_mesh, 2),
        "seconds_total": round(total, 2),
        "peak_vram_gb": vram.get("peak_mem_used_gb"),
        "est_cost_usd": round(total * price, 4),
        "price_per_sec_usd": price,
        "image_url": image_url,
        "wheels": installed,
        "vram": vram,
        "smi_end": _smi(),
        "stack": f"021 TRELLIS.2 source wheels ARCH={arch}",
    }
    pub = _publish(out_path, _safe(output_name), meta, input_path=img_path)
    try:
        weights_vol.commit()
    except Exception:
        pass
    result = {
        "ok": True,
        "gpu_actual": gpu_name,
        "capability": list(cap),
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


def _probe(gpu_label: str, wheel_out: Path) -> dict:
    import torch

    info: dict[str, Any] = {
        "app": APP_NAME,
        "model": HF_MODEL,
        "gpu_request": gpu_label,
        "torch": str(torch.__version__),
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "capability": None,
        "code_ok": (CODE_DIR / "trellis2").is_dir(),
        "o_voxel_src": (CODE_DIR / "o-voxel").is_dir(),
        "wheel_dir": str(wheel_out),
        "wheel_files": sorted(p.name for p in wheel_out.glob("*.whl"))
        if wheel_out.is_dir()
        else [],
        "model_ready": _model_ready(),
        "smi": _smi(),
    }
    try:
        import xformers

        info["xformers"] = str(xformers.__version__)
    except Exception as e:
        info["xformers"] = repr(e)
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        a = torch.randn(256, 256, device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        info["matmul_ok"] = bool(int(b.numel()) > 0)
    print(json.dumps(info, indent=2, default=str), flush=True)
    return _jsonable(info)


@app.cls(
    image=image_l40s,
    gpu="L40S",
    timeout=4 * 60 * 60,
    memory=32768,
    cpu=4,
    volumes={
        str(WHEELS): wheels_vol,
        str(WEIGHTS): weights_vol,
        str(OUTPUTS): outputs_vol,
    },
)
class Trellis2L40S:
    WHEEL_OUT = _wheel_dir("sm89", "torch260-cu124-cp310")

    @modal.enter()
    def enter(self) -> None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        os.environ["ATTN_BACKEND"] = "xformers"
        os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        _ensure_dirs()

    @modal.method()
    def probe(self) -> dict:
        return _probe("L40S", self.WHEEL_OUT)

    @modal.method()
    def build(self) -> dict:
        return _jsonable(_build_extensions(self.WHEEL_OUT, "8.9"))

    @modal.method()
    def verify(self) -> dict:
        installed = _install_wheels(self.WHEEL_OUT)
        verify = Path("/opt/trellis2_scripts/verify_sm89.py")
        v = subprocess.run(
            [
                sys.executable,
                str(verify),
                "--expect-gpu",
                "--packages",
                "flex_gemm,o_voxel,cumesh,nvdiffrast,renderutils",
                "--required",
                "flex_gemm,o_voxel,cumesh",
            ],
            capture_output=True,
            text=True,
        )
        return _jsonable(
            {
                "ok": v.returncode == 0,
                "returncode": v.returncode,
                "stdout": v.stdout,
                "stderr": v.stderr,
                "wheels_installed": installed,
            }
        )

    @modal.method()
    def download(self, force: bool = False) -> dict:
        return _jsonable(_download_weights(force=force))

    @modal.method()
    def image_to_3d(
        self,
        image_url: str = SAMPLE_URL,
        output_name: str = "smoke_l40s",
        pipeline_type: str = "512",
        seed: int = 42,
        texture_size: int = 2048,
        decimation_target: int = 500000,
    ) -> dict:
        return _run_i2v(
            gpu_label="L40S",
            arch="8.9",
            wheel_out=self.WHEEL_OUT,
            image_url=image_url,
            output_name=output_name,
            pipeline_type=pipeline_type,
            seed=seed,
            texture_size=texture_size,
            decimation_target=decimation_target,
        )


@app.cls(
    image=image_pro6000,
    gpu="RTX-PRO-6000",
    timeout=4 * 60 * 60,
    memory=32768,
    cpu=4,
    volumes={
        str(WHEELS): wheels_vol,
        str(WEIGHTS): weights_vol,
        str(OUTPUTS): outputs_vol,
    },
)
class Trellis2Pro6000:
    WHEEL_OUT = _wheel_dir("sm120", "torch211-cu128-cp310")

    @modal.enter()
    def enter(self) -> None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        os.environ["ATTN_BACKEND"] = "xformers"
        os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        _ensure_dirs()

    @modal.method()
    def probe(self) -> dict:
        return _probe("RTX-PRO-6000", self.WHEEL_OUT)

    @modal.method()
    def build(self) -> dict:
        return _jsonable(_build_extensions(self.WHEEL_OUT, "12.0"))

    @modal.method()
    def verify(self) -> dict:
        installed = _install_wheels(self.WHEEL_OUT)
        verify = Path("/opt/trellis2_scripts/verify_sm120.py")
        if not verify.is_file():
            oks = {}
            for pkg in ("flex_gemm", "o_voxel", "cumesh"):
                try:
                    __import__(pkg)
                    oks[pkg] = True
                except Exception as e:
                    oks[pkg] = repr(e)
            return _jsonable(
                {"ok": all(v is True for v in oks.values()), "imports": oks, "wheels": installed}
            )
        v = subprocess.run(
            [
                sys.executable,
                str(verify),
                "--expect-gpu",
                "--packages",
                "flex_gemm,o_voxel,cumesh,nvdiffrast,renderutils",
                "--required",
                "flex_gemm,o_voxel,cumesh",
            ],
            capture_output=True,
            text=True,
        )
        return _jsonable(
            {
                "ok": v.returncode == 0,
                "returncode": v.returncode,
                "stdout": v.stdout,
                "stderr": v.stderr,
                "wheels_installed": installed,
            }
        )

    @modal.method()
    def download(self, force: bool = False) -> dict:
        return _jsonable(_download_weights(force=force))

    @modal.method()
    def image_to_3d(
        self,
        image_url: str = SAMPLE_URL,
        output_name: str = "smoke_pro6000",
        pipeline_type: str = "512",
        seed: int = 42,
        texture_size: int = 2048,
        decimation_target: int = 500000,
    ) -> dict:
        return _run_i2v(
            gpu_label="RTX-PRO-6000",
            arch="12.0",
            wheel_out=self.WHEEL_OUT,
            image_url=image_url,
            output_name=output_name,
            pipeline_type=pipeline_type,
            seed=seed,
            texture_size=texture_size,
            decimation_target=decimation_target,
        )


@app.local_entrypoint()
def main(
    action: str = "probe",
    gpu: str = "L40S",
    output_name: str = "",
    image_url: str = SAMPLE_URL,
    pipeline_type: str = "512",
    seed: int = 42,
):
    g = gpu.upper().replace("_", "-")
    use_pro = g in {"RTX-PRO-6000", "PRO-6000", "PRO6000"}
    worker = Trellis2Pro6000() if use_pro else Trellis2L40S()
    default_name = "smoke_pro6000" if use_pro else "smoke_l40s"
    name = output_name or default_name
    if action in {"probe", "status"}:
        print(worker.probe.remote())
    elif action in {"build", "build-sm89", "build-sm120"}:
        print(worker.build.remote())
    elif action == "verify":
        print(worker.verify.remote())
    elif action == "download":
        print(worker.download.remote())
    elif action in {"smoke", "i2v"}:
        print(
            worker.image_to_3d.remote(
                image_url=image_url,
                output_name=name,
                pipeline_type=pipeline_type,
                seed=seed,
            )
        )
    else:
        raise SystemExit(f"unknown action: {action}")
