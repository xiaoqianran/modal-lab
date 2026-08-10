# -*- coding: utf-8 -*-
"""022-hunyuan3d-2.1 — Tencent Hunyuan3D-2.1 image→3D on Modal.

Default GPU: L40S · optional RTX-PRO-6000.
License: Tencent Hunyuan 3D 2.1 Community License (non-commercial / territory limits).
Upstream: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
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

APP_NAME = "modal-lab-hunyuan3d-2-1"
DEFAULT_GPU = "L40S"
CODE_DIR = Path("/opt/src/Hunyuan3D-2.1")
WEIGHTS = Path("/weights")
OUTPUTS = Path("/outputs")
MESHES = OUTPUTS / "meshes"
BENCH = OUTPUTS / "benchmarks"
INPUTS = OUTPUTS / "inputs"
WHEELS = WEIGHTS / "wheels"
HY3DGEN_MODELS = WEIGHTS / "hy3dgen"

VOLUME_WEIGHTS = "modal-lab-hunyuan3d21-weights"
VOLUME_OUTPUTS = "modal-lab-hunyuan3d21-outputs"

SAMPLE_URL = (
    "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png"
)
HF_MODEL = "tencent/Hunyuan3D-2.1"
UPSTREAM = "Tencent-Hunyuan/Hunyuan3D-2.1"

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
    "libsm6",
    "libxext6",
    "libxrender1",
    "wget",
    "curl",
    "ca-certificates",
    "python3-dev",
]

# Image deps stable (shape path). Paint extras via _ensure_paint_deps at runtime.
_COMMON_PIP = [
    "pip",
    "wheel",
    "setuptools>=49.6.0",
    "ninja==1.11.1.1",
    "pybind11==2.13.4",
    "numpy==1.26.4",
    "huggingface_hub>=0.26.0,<0.31",
    "transformers==4.46.0",
    "diffusers==0.30.0",
    "accelerate==1.1.1",
    "safetensors==0.4.4",
    "einops==0.8.0",
    "omegaconf==2.3.0",
    "pyyaml==6.0.2",
    "Pillow==10.4.0",
    "opencv-python-headless==4.10.0.84",
    "imageio==2.36.0",
    "scikit-image==0.24.0",
    "trimesh==4.4.7",
    "pymeshlab==2022.2.post3",
    "pygltflib==1.16.3",
    "xatlas==0.0.9",
    "rembg==2.0.65",
    "onnxruntime==1.16.3",
    "tqdm==4.66.5",
    "psutil==6.0.0",
    "timm",
    "torchdiffeq",
    "scipy==1.14.1",
    "basicsr==1.4.2",
    "realesrgan==0.3.0",
]


def _image_setup_cmds() -> list[str]:
    return [
        "mkdir -p /opt/src && cd /opt/src && "
        "git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git",
        f"cd {CODE_DIR}/hy3dpaint/DifferentiableRenderer && "
        "c++ -O3 -Wall -shared -std=c++11 -fPIC "
        "$(python -m pybind11 --includes) mesh_inpaint_processor.cpp "
        "-o mesh_inpaint_processor$(python3-config --extension-suffix) && "
        "ls -la mesh_inpaint_processor*.so",
        f"mkdir -p {CODE_DIR}/hy3dpaint/ckpt && "
        f"wget -q -O {CODE_DIR}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth "
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)\"",
    ]


def _make_image(*, cuda_tag: str, torch_pkgs: list[str], index_url: str, arch: str) -> modal.Image:
    return (
        modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.10")
        .apt_install(*_APT)
        .pip_install(*_COMMON_PIP)
        .pip_install(*torch_pkgs, index_url=index_url)
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "CC": "gcc",
                "CXX": "g++",
                "FORCE_CUDA": "1",
                "TORCH_CUDA_ARCH_LIST": arch,
                "HF_HOME": str(WEIGHTS / "hf"),
                "HUGGINGFACE_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
                "HF_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
                "TRANSFORMERS_CACHE": str(WEIGHTS / "hf" / "transformers"),
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
                "HF_HUB_DISABLE_XET": "1",
                "HY3DGEN_MODELS": str(HY3DGEN_MODELS),
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": f"{CODE_DIR}:{CODE_DIR}/hy3dshape:{CODE_DIR}/hy3dpaint",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            }
        )
        .run_commands(*_image_setup_cmds())
    )


image = _make_image(
    cuda_tag="12.4.1-devel-ubuntu22.04",
    torch_pkgs=["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"],
    index_url="https://download.pytorch.org/whl/cu124",
    arch="8.9",
)

image_pro6000 = _make_image(
    cuda_tag="12.8.0-devel-ubuntu22.04",
    torch_pkgs=["torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0"],
    index_url="https://download.pytorch.org/whl/cu128",
    arch="12.0",
)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def _safe(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return s or "mesh"


def _ensure_dirs() -> None:
    for p in (
        WEIGHTS / "hf" / "hub",
        WEIGHTS / "hf" / "transformers",
        HY3DGEN_MODELS,
        MESHES,
        BENCH,
        INPUTS,
        WHEELS,
    ):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(WEIGHTS / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(WEIGHTS / "hf" / "hub")
    os.environ["HF_HUB_CACHE"] = str(WEIGHTS / "hf" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(WEIGHTS / "hf" / "transformers")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HY3DGEN_MODELS"] = str(HY3DGEN_MODELS)
    for p in (str(CODE_DIR), str(CODE_DIR / "hy3dshape"), str(CODE_DIR / "hy3dpaint")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(CODE_DIR)


def _patch_mesh_utils_no_bpy() -> None:
    path = CODE_DIR / "hy3dpaint" / "DifferentiableRenderer" / "mesh_utils.py"
    if not path.is_file():
        print(f"[patch] missing {path}", flush=True)
        return
    text = path.read_text(encoding="utf-8")
    if "NO_BPY_PATCH" in text:
        print("[patch] mesh_utils already patched", flush=True)
        return
    text = text.replace(
        "import bpy\n",
        "try:\n    import bpy  # NO_BPY_PATCH\nexcept ImportError:\n    bpy = None  # NO_BPY_PATCH\n",
        1,
    )
    text += """

# === NO_BPY_PATCH — trimesh fallback (no Blender in container) ===
def convert_obj_to_glb(obj_path, glb_path, shade_type="SMOOTH", auto_smooth_angle=60, merge_vertices=False):
    try:
        import trimesh
        loaded = trimesh.load(obj_path, process=False, maintain_order=True)
        loaded.export(glb_path)
        print(f"[patch] trimesh OBJ->GLB {obj_path} -> {glb_path}", flush=True)
        return True
    except Exception as e:
        print(f"[patch] trimesh convert failed: {e}", flush=True)
        return False
"""
    path.write_text(text, encoding="utf-8")
    for mod in list(sys.modules):
        if "mesh_utils" in mod or "DifferentiableRenderer" in mod:
            del sys.modules[mod]
    print("[patch] mesh_utils no-bpy applied", flush=True)



def _patch_remesh_no_open3d() -> None:
    """Use pymeshlab decimation instead of trimesh/open3d."""
    path = CODE_DIR / "hy3dpaint" / "utils" / "simplify_mesh_utils.py"
    if not path.is_file():
        return
    cur = path.read_text(encoding="utf-8")
    if "NO_OPEN3D_PATCH" in cur:
        print("[patch] remesh already patched", flush=True)
        return
    body = (
        "# NO_OPEN3D_PATCH\n"
        "import pymeshlab\n"
        "import trimesh\n"
        "\n"
        "\n"
        "def remesh_mesh(mesh_path, remesh_path):\n"
        "    mesh_simplify_trimesh(mesh_path, remesh_path)\n"
        "\n"
        "\n"
        "def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):\n"
        "    ms = pymeshlab.MeshSet()\n"
        "    if str(inputpath).endswith('.glb'):\n"
        "        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)\n"
        "    else:\n"
        "        ms.load_new_mesh(inputpath)\n"
        "    try:\n"
        "        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=50)\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        n_faces = ms.current_mesh().face_number()\n"
        "        if n_faces > target_count:\n"
        "            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_count))\n"
        "    except Exception as e:\n"
        "        print(f'[patch] decimation skip: {e}', flush=True)\n"
        "    out = str(outputpath)\n"
        "    if out.endswith('.glb'):\n"
        "        tmp = out.replace('.glb', '.obj')\n"
        "        ms.save_current_mesh(tmp, save_textures=False)\n"
        "        m = trimesh.load(tmp, force='mesh')\n"
        "        m.export(out)\n"
        "    else:\n"
        "        ms.save_current_mesh(out, save_textures=False)\n"
        "    print(f'[patch] remesh {inputpath} -> {outputpath}', flush=True)\n"
    )
    path.write_text(body, encoding="utf-8")
    for mod in list(sys.modules):
        if "simplify_mesh" in mod:
            del sys.modules[mod]
    print("[patch] remesh no-open3d applied", flush=True)


def _ensure_paint_deps() -> None:
    """Install paint-only packages missing from the base image."""
    cmds: list[list[str]] = []
    try:
        import pkg_resources  # noqa: F401
    except Exception:
        # lightning_fabric needs pkg_resources (setuptools < ~81)
        cmds.append([sys.executable, "-m", "pip", "install", "--quiet", "setuptools==69.5.1"])
    try:
        import pytorch_lightning  # noqa: F401
    except Exception:
        cmds.append(
            [sys.executable, "-m", "pip", "install", "--quiet", "pytorch-lightning==1.9.5"]
        )
    try:
        import torchmetrics  # noqa: F401
    except Exception:
        cmds.append([sys.executable, "-m", "pip", "install", "--quiet", "torchmetrics==1.6.0"])
    if not cmds:
        print("[deps] paint extras ok", flush=True)
        return
    for c in cmds:
        print(f"[deps] {' '.join(c[-3:])}", flush=True)
        subprocess.check_call(c)
    # verify lightning imports after setuptools fix
    try:
        import pytorch_lightning  # noqa: F401
    except Exception as e:
        print(f"[deps] reinstall lightning after: {e}", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--force-reinstall",
                "setuptools==69.5.1",
                "pytorch-lightning==1.9.5",
            ]
        )
    print("[deps] paint extras installed", flush=True)



def _ensure_custom_rasterizer(arch: str) -> None:
    """Build/install custom_rasterizer CUDA extension (cached under /weights/wheels)."""
    import importlib
    import importlib.util

    def _ok() -> bool:
        try:
            # Prefer site-packages over hy3dpaint source tree
            import custom_rasterizer_kernel  # noqa: F401
            import custom_rasterizer as cr

            if not hasattr(cr, "rasterize"):
                return False
            print(
                f"[ext] custom_rasterizer ok rasterize={callable(cr.rasterize)} "
                f"kernel={custom_rasterizer_kernel.__file__}",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[ext] not ready: {e}", flush=True)
            return False

    def _purge() -> None:
        for mod in list(sys.modules):
            if mod == "custom_rasterizer" or mod.startswith("custom_rasterizer.") or mod.startswith(
                "custom_rasterizer_kernel"
            ):
                del sys.modules[mod]
        # drop hy3dpaint source path temporarily so site-packages wins after install
        drop = []
        for i, pth in enumerate(sys.path):
            if pth.rstrip("/").endswith("hy3dpaint") or pth.rstrip("/").endswith("custom_rasterizer"):
                drop.append(i)
        for i in reversed(drop):
            sys.path.pop(i)
        importlib.invalidate_caches()

    if _ok():
        return

    wheel_root = WHEELS / f"sm{arch.replace('.', '')}" / "custom_rasterizer"
    wheel_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["FORCE_CUDA"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = arch
    env["MAX_JOBS"] = "4"

    wheels = sorted(wheel_root.glob("custom_rasterizer-*.whl"))
    if wheels:
        wh = wheels[-1]
        print(f"[ext] install wheel {wh}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wh)]
        )
        _purge()
        if _ok():
            print("[ext] custom_rasterizer from wheel ok", flush=True)
            return
        print("[ext] wheel install insufficient, rebuilding", flush=True)

    src = CODE_DIR / "hy3dpaint" / "custom_rasterizer"
    print(f"[ext] building custom_rasterizer ARCH={arch} from {src}", flush=True)
    # clean previous incomplete build artifacts
    for junk in src.rglob("*.so"):
        try:
            junk.unlink()
        except Exception:
            pass
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "-w",
            str(wheel_root),
            str(src),
        ],
        cwd=str(src),
        env=env,
    )
    wheels = sorted(wheel_root.glob("custom_rasterizer-*.whl"))
    if wheels:
        wh = wheels[-1]
        print(f"[ext] install built wheel {wh}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wh)]
        )
    else:
        print("[ext] fallback editable install", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--force-reinstall",
                str(src),
            ],
            env=env,
        )
    _purge()
    # put hy3dpaint back for paint imports
    for pth in (str(CODE_DIR), str(CODE_DIR / "hy3dshape"), str(CODE_DIR / "hy3dpaint")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    if not _ok():
        # last resort: import from source package after kernel is in site-packages
        _purge()
        for pth in (str(CODE_DIR), str(CODE_DIR / "hy3dshape"), str(CODE_DIR / "hy3dpaint")):
            if pth not in sys.path:
                sys.path.insert(0, pth)
        if not _ok():
            raise RuntimeError("custom_rasterizer install failed (no rasterize)")
    print("[ext] custom_rasterizer build ok", flush=True)
    try:
        weights_vol.commit()
    except Exception:
        pass



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
            f"modal volume get {VOLUME_OUTPUTS} "
            f"meshes/{name}{src.suffix} ./{name}{src.suffix}"
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (MESHES / f"{name}_meta.json").write_text(text, encoding="utf-8")
    (MESHES / "latest_meta.json").write_text(text, encoding="utf-8")
    (BENCH / f"{name}.json").write_text(text, encoding="utf-8")
    outputs_vol.commit()
    print(f"[VOLUME] {VOLUME_OUTPUTS}/meshes/{name}{src.suffix} ({size} bytes)", flush=True)
    return payload


def _apply_torchvision_fix() -> None:
    try:
        sys.path.insert(0, str(CODE_DIR))
        from torchvision_fix import apply_fix

        apply_fix()
        print("[fix] torchvision_fix applied", flush=True)
    except Exception as e:
        print(f"[fix] skip: {e}", flush=True)


def _run_i2v(
    *,
    gpu_label: str,
    image_url: str,
    output_name: str,
    mode: str = "full",
    seed: int = 42,
    max_num_view: int = 6,
    paint_resolution: int = 512,
) -> dict:
    import torch
    from PIL import Image

    _ensure_dirs()
    _patch_mesh_utils_no_bpy()
    _apply_torchvision_fix()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu_name = torch.cuda.get_device_name(0)
    cap = list(torch.cuda.get_device_capability(0))
    price = GPU_PRICE.get(gpu_label, GPU_PRICE["L40S"])
    mode = (mode or "full").lower().strip()
    if mode not in {"shape", "full"}:
        raise ValueError("mode must be shape|full")

    work = Path("/tmp/hy3d21_work")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
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

    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    print(
        f"[shape] load {HF_MODEL} HY3DGEN_MODELS={os.environ.get('HY3DGEN_MODELS')}",
        flush=True,
    )
    t_load0 = time.time()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HY3DGEN_MODELS"] = str(HY3DGEN_MODELS)
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HF_MODEL)
    if hasattr(pipeline_shapegen, "to"):
        pipeline_shapegen.to("cuda")
    t_load = time.time() - t_load0
    _peak()
    try:
        weights_vol.commit()
        print("[weights] volume commit after shape load", flush=True)
    except Exception as e:
        print(f"[weights] commit skip: {e}", flush=True)

    image = Image.open(img_path).convert("RGBA")
    alpha = image.split()[-1]
    if min(alpha.getextrema()) >= 250:
        print("[shape] rembg (opaque input)", flush=True)
        rembg = BackgroundRemover()
        image = rembg(image.convert("RGB"))
    image.save(work / "input_fg.png")

    t1 = time.time()
    print(f"[shape] generate seed={seed}", flush=True)
    mesh = pipeline_shapegen(image=image, generator=torch.Generator("cuda").manual_seed(seed))[0]
    torch.cuda.synchronize()
    t_shape = time.time() - t1
    _peak()

    shape_glb = work / "shape.glb"
    mesh.export(str(shape_glb))
    print(f"[shape] wrote {shape_glb} ({shape_glb.stat().st_size} bytes)", flush=True)

    t_paint = 0.0
    out_path = shape_glb
    textured = False

    if mode == "full":
        try:
            del pipeline_shapegen
            torch.cuda.empty_cache()
        except Exception:
            pass

        major, minor = torch.cuda.get_device_capability(0)
        arch = f"{major}.{minor}"
        _ensure_paint_deps()
        _ensure_custom_rasterizer(arch)
        _patch_mesh_utils_no_bpy()
        _patch_remesh_no_open3d()

        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        conf = Hunyuan3DPaintConfig(max_num_view, paint_resolution)
        conf.realesrgan_ckpt_path = str(CODE_DIR / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
        conf.multiview_cfg_path = str(CODE_DIR / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = str(CODE_DIR / "hy3dpaint/hunyuanpaintpbr")
        conf.multiview_pretrained_path = HF_MODEL

        print(
            f"[paint] load max_view={max_num_view} res={paint_resolution}",
            flush=True,
        )
        t2 = time.time()
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        obj_out = work / "textured.obj"
        result = paint_pipeline(
            mesh_path=str(shape_glb),
            image_path=str(work / "input_fg.png"),
            output_mesh_path=str(obj_out),
            use_remesh=True,
            save_glb=True,
        )
        torch.cuda.synchronize()
        t_paint = time.time() - t2
        _peak()
        try:
            weights_vol.commit()
            print("[weights] volume commit after paint", flush=True)
        except Exception as e:
            print(f"[weights] commit skip: {e}", flush=True)
        if isinstance(result, str) and Path(result).is_file():
            cand = Path(result)
            if cand.suffix.lower() == ".obj" and cand.with_suffix(".glb").is_file():
                out_path = cand.with_suffix(".glb")
            else:
                out_path = cand
        elif (work / "textured.glb").is_file():
            out_path = work / "textured.glb"
        else:
            candidates = list(work.glob("**/*textured*.glb")) + list(work.glob("**/*.glb"))
            candidates = [c for c in candidates if c.name != "shape.glb"]
            if not candidates:
                raise RuntimeError(f"paint produced no glb under {work}: {list(work.rglob('*'))}")
            out_path = max(candidates, key=lambda p: p.stat().st_mtime)
        textured = True
        print(f"[paint] wrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)

    total = time.time() - t0
    if not out_path.is_file():
        raise RuntimeError(f"mesh not written: {out_path}")

    final = work / f"{_safe(output_name)}.glb"
    if out_path.resolve() != final.resolve():
        shutil.copy2(out_path, final)

    meta = {
        "model": HF_MODEL,
        "upstream": UPSTREAM,
        "license": "Tencent Hunyuan 3D 2.1 Community License",
        "gpu_request": gpu_label,
        "gpu_actual": gpu_name,
        "capability": cap,
        "output_name": _safe(output_name),
        "mode": mode,
        "textured": textured,
        "seed": seed,
        "max_num_view": max_num_view if mode == "full" else None,
        "paint_resolution": paint_resolution if mode == "full" else None,
        "format": "glb",
        "seconds_load": round(t_load, 2),
        "seconds_shape": round(t_shape, 2),
        "seconds_paint": round(t_paint, 2),
        "seconds_total": round(total, 2),
        "peak_vram_gb": round(peak_mib / 1024.0, 2) if peak_mib else None,
        "est_cost_usd": round(total * price, 4),
        "price_per_sec_usd": price,
        "image_url": image_url,
        "torch": str(torch.__version__),
        "smi_end": _smi(),
        "stack": f"022 Hunyuan3D-2.1 {gpu_label}",
        "hy3dgen_models": str(HY3DGEN_MODELS),
    }
    pub = _publish(final, _safe(output_name), meta, input_path=img_path)
    try:
        weights_vol.commit()
    except Exception:
        pass
    result = {
        "ok": True,
        "gpu_actual": gpu_name,
        "capability": cap,
        "mode": mode,
        "textured": textured,
        "seconds_total": meta["seconds_total"],
        "seconds_shape": meta["seconds_shape"],
        "seconds_paint": meta["seconds_paint"],
        "peak_vram_gb": meta["peak_vram_gb"],
        "est_cost_usd": meta["est_cost_usd"],
        "bytes": pub["bytes"],
        "volume": VOLUME_OUTPUTS,
        "volume_file": pub["volume_paths"]["mesh"],
        "cli_get": pub["cli_get"],
    }
    print(json.dumps(result, indent=2), flush=True)
    return _jsonable(result)


def _probe(gpu_label: str) -> dict:
    import torch

    _ensure_dirs()
    _patch_mesh_utils_no_bpy()
    info = {
        "app": APP_NAME,
        "model": HF_MODEL,
        "upstream": UPSTREAM,
        "gpu_request": gpu_label,
        "torch": str(torch.__version__),
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "smi": _smi(),
        "code_ok": (CODE_DIR / "hy3dshape").is_dir(),
        "paint_ok": (CODE_DIR / "hy3dpaint").is_dir(),
        "realesrgan": (CODE_DIR / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth").is_file(),
        "hy3dgen_models": str(HY3DGEN_MODELS),
        "hy3dgen_exists": HY3DGEN_MODELS.is_dir(),
        "mesh_utils_no_bpy": "NO_BPY_PATCH"
        in (
            (CODE_DIR / "hy3dpaint/DifferentiableRenderer/mesh_utils.py").read_text(
                encoding="utf-8"
            )
            if (CODE_DIR / "hy3dpaint/DifferentiableRenderer/mesh_utils.py").is_file()
            else ""
        ),
    }
    try:
        import custom_rasterizer  # noqa: F401

        info["custom_rasterizer"] = True
    except Exception as e:
        info["custom_rasterizer"] = False
        info["custom_rasterizer_err"] = repr(e)
    if torch.cuda.is_available():
        a = torch.randn(256, 256, device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        info["matmul_ok"] = bool(int(b.numel()) > 0)
    print(json.dumps(info, indent=2), flush=True)
    return _jsonable(info)


@app.cls(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=90 * 60,
    memory=32768,
    cpu=8,
    volumes={str(WEIGHTS): weights_vol, str(OUTPUTS): outputs_vol},
)
class Hunyuan3D21Worker:
    @modal.enter()
    def enter(self) -> None:
        _ensure_dirs()
        _patch_mesh_utils_no_bpy()

    @modal.method()
    def probe(self) -> dict:
        return _probe(DEFAULT_GPU)

    @modal.method()
    def image_to_3d(
        self,
        image_url: str = SAMPLE_URL,
        output_name: str = "smoke_l40s",
        mode: str = "full",
        seed: int = 42,
        max_num_view: int = 6,
        paint_resolution: int = 512,
    ) -> dict:
        return _run_i2v(
            gpu_label=DEFAULT_GPU,
            image_url=image_url,
            output_name=output_name,
            mode=mode,
            seed=seed,
            max_num_view=max_num_view,
            paint_resolution=paint_resolution,
        )


@app.cls(
    image=image_pro6000,
    gpu="RTX-PRO-6000",
    timeout=90 * 60,
    memory=32768,
    cpu=8,
    volumes={str(WEIGHTS): weights_vol, str(OUTPUTS): outputs_vol},
)
class Hunyuan3D21Pro6000:
    @modal.enter()
    def enter(self) -> None:
        _ensure_dirs()
        _patch_mesh_utils_no_bpy()

    @modal.method()
    def probe(self) -> dict:
        return _probe("RTX-PRO-6000")

    @modal.method()
    def image_to_3d(
        self,
        image_url: str = SAMPLE_URL,
        output_name: str = "smoke_pro6000",
        mode: str = "full",
        seed: int = 42,
        max_num_view: int = 6,
        paint_resolution: int = 512,
    ) -> dict:
        return _run_i2v(
            gpu_label="RTX-PRO-6000",
            image_url=image_url,
            output_name=output_name,
            mode=mode,
            seed=seed,
            max_num_view=max_num_view,
            paint_resolution=paint_resolution,
        )


@app.local_entrypoint()
def main(
    action: str = "probe",
    gpu: str = "L40S",
    output_name: str = "",
    image_url: str = SAMPLE_URL,
    mode: str = "full",
    seed: int = 42,
    max_num_view: int = 6,
    paint_resolution: int = 512,
):
    g = gpu.upper().replace("_", "-")
    use_pro = g in {"RTX-PRO-6000", "PRO-6000", "PRO6000"}
    worker = Hunyuan3D21Pro6000() if use_pro else Hunyuan3D21Worker()
    default_name = "smoke_pro6000" if use_pro else "smoke_l40s"
    if mode == "shape" and not output_name:
        default_name = default_name.replace("smoke_", "smoke_shape_")
    name = output_name or default_name
    if action in {"probe", "status"}:
        print(worker.probe.remote())
    elif action in {"smoke", "i2v"}:
        print(
            worker.image_to_3d.remote(
                image_url=image_url,
                output_name=name,
                mode=mode,
                seed=seed,
                max_num_view=max_num_view,
                paint_resolution=paint_resolution,
            )
        )
    else:
        raise SystemExit(f"unknown action: {action}")
