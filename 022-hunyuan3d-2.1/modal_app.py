"""Hunyuan3D-2.1 image -> 3D on one Modal L40S worker."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import modal

APP_NAME = "modal-lab-hunyuan3d-2-1"
GPU = "L40S"
MODEL = "tencent/Hunyuan3D-2.1"
UPSTREAM = "Tencent-Hunyuan/Hunyuan3D-2.1"
UPSTREAM_COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
SAMPLE_URL = "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png"
L40S_USD_PER_SECOND = 0.000542

CODE = Path("/opt/Hunyuan3D-2.1")
WEIGHTS = Path("/weights")
OUTPUTS = Path("/outputs")

weights = modal.Volume.from_name("modal-lab-hunyuan3d21-weights", create_if_missing=True)
outputs = modal.Volume.from_name("modal-lab-hunyuan3d21-outputs", create_if_missing=True)
app = modal.App(APP_NAME)

PIP = [
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
    "setuptools==69.5.1",
    "wheel==0.44.0",
    "ninja==1.11.1.1",
    "pybind11==2.13.4",
    "numpy==1.26.4",
    "transformers==4.46.0",
    "diffusers==0.30.0",
    "accelerate==1.1.1",
    "pytorch-lightning==1.9.5",
    "torchmetrics==1.6.0",
    "huggingface-hub==0.30.2",
    "hf_xet==1.1.9",
    "safetensors==0.4.4",
    "scipy==1.14.1",
    "einops==0.8.0",
    "opencv-python-headless==4.10.0.84",
    "imageio==2.36.0",
    "scikit-image==0.24.0",
    "rembg==2.0.65",
    "realesrgan==0.3.0",
    "basicsr==1.4.2",
    "pymeshlab==2022.2.post3",
    "trimesh==4.4.7",
    "fast-simplification==0.1.12",
    "pygltflib==1.16.3",
    "xatlas==0.0.9",
    "omegaconf==2.3.0",
    "pyyaml==6.0.2",
    "Pillow==10.4.0",
    "tqdm==4.66.5",
    "psutil==6.0.0",
    "onnxruntime==1.16.3",
    "cupy-cuda12x==13.4.1",
    "timm==1.0.11",
    "torchdiffeq==0.2.5",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "wget",
    )
    .pip_install(
        *PIP,
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "4",
            "CC": "gcc",
            "CXX": "g++",
            "HF_HOME": str(WEIGHTS / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
            "HF_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
            "HY3DGEN_MODELS": str(WEIGHTS / "hy3dgen"),
            "PYTHONPATH": f"{CODE}:{CODE}/hy3dshape:{CODE}/hy3dpaint",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        f"git clone https://github.com/{UPSTREAM}.git {CODE} && "
        f"cd {CODE} && git checkout {UPSTREAM_COMMIT}",
        f"python -c \"from pathlib import Path; p=Path('{CODE}/hy3dpaint/DifferentiableRenderer/mesh_utils.py'); s=p.read_text(); p.write_text(s.replace('import bpy\\n', 'try:\\n    import bpy\\nexcept ImportError:\\n    bpy = None\\n', 1))\"",
        f"cd {CODE}/hy3dpaint/custom_rasterizer && pip install --no-build-isolation .",
        f"cd {CODE}/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh",
        f"mkdir -p {CODE}/hy3dpaint/ckpt && "
        f"wget -q -O {CODE}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth "
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    )
)


def _safe_name(name: str) -> str:
    value = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return value or "mesh"


def _gpu() -> dict:
    import torch

    return {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }


def _fetch(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


def _publish(mesh: Path, name: str, meta: dict, input_image: Path) -> dict:
    mesh_dir = OUTPUTS / "meshes"
    input_dir = OUTPUTS / "inputs"
    bench_dir = OUTPUTS / "benchmarks"
    for path in (mesh_dir, input_dir, bench_dir):
        path.mkdir(parents=True, exist_ok=True)

    name = _safe_name(name)
    dest = mesh_dir / f"{name}.glb"
    shutil.copy2(mesh, dest)
    shutil.copy2(mesh, mesh_dir / "latest.glb")
    shutil.copy2(input_image, input_dir / f"{name}{input_image.suffix.lower()}")

    meta = {
        **meta,
        "bytes": dest.stat().st_size,
        "volume": "modal-lab-hunyuan3d21-outputs",
        "volume_file": f"meshes/{dest.name}",
    }
    text = json.dumps(meta, ensure_ascii=False, indent=2)
    (mesh_dir / f"{name}_meta.json").write_text(text, encoding="utf-8")
    (mesh_dir / "latest_meta.json").write_text(text, encoding="utf-8")
    (bench_dir / f"{name}.json").write_text(text, encoding="utf-8")
    outputs.commit()
    return meta


@app.cls(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    scaledown_window=300,
    max_containers=1,
    volumes={str(WEIGHTS): weights, str(OUTPUTS): outputs},
)
class Hunyuan3D21:
    @modal.enter()
    def load(self) -> None:
        import torch

        os.chdir(CODE)
        for path in (CODE, CODE / "hy3dshape", CODE / "hy3dpaint"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from torchvision_fix import apply_fix

        apply_fix()
        info = _gpu()
        if info["capability"] != [8, 9]:
            raise RuntimeError(f"expected L40S sm_89, got {info}")

        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        shape_dir = WEIGHTS / "hy3dgen" / MODEL / "hunyuan3d-dit-v2-1"
        shape_ckpt = shape_dir / "model.fp16.ckpt"
        if shape_dir.exists() and not shape_ckpt.is_file():
            print(f"[weights] remove incomplete shape cache: {shape_dir}", flush=True)
            shutil.rmtree(shape_dir)

        print(f"[load] shape {MODEL}", flush=True)
        self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL)
        self.shape.to("cuda")
        torch.cuda.synchronize()
        self.remover = None
        try:
            weights.commit()
        except Exception as exc:
            print(f"[weights] commit warning: {exc}", flush=True)
        print(f"[ready] {json.dumps(info)}", flush=True)

    def _load_paint(self, max_num_view: int, resolution: int):
        import fast_simplification
        import trimesh
        from DifferentiableRenderer import mesh_utils
        from utils import simplify_mesh_utils

        def convert_obj_to_glb(obj_path, glb_path, **_kwargs):
            trimesh.load(obj_path, process=False, maintain_order=True).export(glb_path)
            return True

        def remesh_mesh(mesh_path, remesh_path):
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            if len(mesh.faces) > 40_000:
                vertices, faces = fast_simplification.simplify(
                    mesh.vertices, mesh.faces, target_count=40_000
                )
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.export(remesh_path)

        mesh_utils.convert_obj_to_glb = convert_obj_to_glb
        simplify_mesh_utils.remesh_mesh = remesh_mesh
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        key = (max_num_view, resolution)
        if getattr(self, "paint_key", None) == key and getattr(self, "paint", None) is not None:
            return self.paint

        config = Hunyuan3DPaintConfig(max_num_view, resolution)
        config.realesrgan_ckpt_path = str(CODE / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
        config.multiview_cfg_path = str(CODE / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
        config.custom_pipeline = str(CODE / "hy3dpaint/hunyuanpaintpbr")
        config.multiview_pretrained_path = MODEL
        print(f"[load] paint views={max_num_view} resolution={resolution}", flush=True)
        self.paint = Hunyuan3DPaintPipeline(config)
        self.paint_key = key
        try:
            weights.commit()
        except Exception as exc:
            print(f"[weights] commit warning: {exc}", flush=True)
        return self.paint

    @modal.method()
    def probe(self) -> dict:
        import custom_rasterizer

        return {
            "app": APP_NAME,
            "model": MODEL,
            "upstream_commit": UPSTREAM_COMMIT,
            "gpu": _gpu(),
            "custom_rasterizer": callable(custom_rasterizer.rasterize),
        }

    @modal.method()
    def generate(
        self,
        image_url: str = SAMPLE_URL,
        output_name: str = "smoke_l40s",
        mode: str = "full",
        seed: int = 42,
        max_num_view: int = 6,
        paint_resolution: int = 512,
    ) -> dict:
        import torch
        from PIL import Image
        from hy3dshape.rembg import BackgroundRemover

        if mode not in {"shape", "full"}:
            raise ValueError("mode must be shape or full")

        request_started = time.perf_counter()
        work = Path("/tmp/hunyuan3d21")
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        image_path = work / ".input.png"

        preprocess_started = time.perf_counter()
        _fetch(image_url, image_path)
        image = Image.open(image_path).convert("RGBA")
        if min(image.getchannel("A").getextrema()) >= 250:
            if self.remover is None:
                self.remover = BackgroundRemover()
            image = self.remover(image.convert("RGB"))
        foreground = work / "input.png"
        image.save(foreground)
        preprocess_seconds = time.perf_counter() - preprocess_started

        torch.cuda.reset_peak_memory_stats()
        shape_started = time.perf_counter()
        mesh = self.shape(
            image=image,
            generator=torch.Generator("cuda").manual_seed(seed),
        )[0]
        torch.cuda.synchronize()
        shape_seconds = time.perf_counter() - shape_started

        shape_glb = work / "shape.glb"
        mesh.export(shape_glb)
        result_glb = shape_glb
        paint_seconds = 0.0

        paint_load_seconds = 0.0
        if mode == "full":
            paint_load_started = time.perf_counter()
            paint = self._load_paint(max_num_view, paint_resolution)
            paint_load_seconds = time.perf_counter() - paint_load_started
            paint_started = time.perf_counter()
            obj = work / "textured.obj"
            paint(
                mesh_path=str(shape_glb),
                image_path=str(foreground),
                output_mesh_path=str(obj),
                use_remesh=True,
                save_glb=True,
            )
            torch.cuda.synchronize()
            paint_seconds = time.perf_counter() - paint_started
            result_glb = obj.with_suffix(".glb")

        if not result_glb.is_file():
            raise RuntimeError(f"no GLB produced: {result_glb}")

        total_seconds = time.perf_counter() - request_started
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        meta = _publish(
            result_glb,
            output_name,
            {
                "model": MODEL,
                "upstream": UPSTREAM,
                "upstream_commit": UPSTREAM_COMMIT,
                "gpu": _gpu(),
                "mode": mode,
                "seed": seed,
                "max_num_view": max_num_view if mode == "full" else None,
                "paint_resolution": paint_resolution if mode == "full" else None,
                "seconds_preprocess": round(preprocess_seconds, 2),
                "seconds_shape": round(shape_seconds, 2),
                "seconds_paint_load": round(paint_load_seconds, 2),
                "seconds_paint": round(paint_seconds, 2),
                "seconds_total": round(total_seconds, 2),
                "peak_vram_gb": round(peak_vram_gb, 2),
                "estimated_cost_usd": round(total_seconds * L40S_USD_PER_SECOND, 4),
            },
            foreground,
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
        return meta


@app.local_entrypoint()
def main(
    action: str = "probe",
    image_url: str = SAMPLE_URL,
    output_name: str = "smoke_l40s",
    mode: str = "full",
    seed: int = 42,
    max_num_view: int = 6,
    paint_resolution: int = 512,
):
    worker = Hunyuan3D21()
    if action in {"probe", "status"}:
        print(worker.probe.remote())
        return
    if action in {"smoke", "generate", "i2v"}:
        print(
            worker.generate.remote(
                image_url=image_url,
                output_name=output_name,
                mode=mode,
                seed=seed,
                max_num_view=max_num_view,
                paint_resolution=paint_resolution,
            )
        )
        return
    raise SystemExit(f"unknown action: {action}")
