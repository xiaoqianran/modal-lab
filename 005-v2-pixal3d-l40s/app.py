"""005-v2 Pixal3D on L40S (Ada sm_89).

Plan A: source-built sm_89 wheels (no HF Spaces demo wheels).
Gates: build-sm89 → verify → smoke/i2v.
"""
from __future__ import annotations

import argparse
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

APP_NAME = "modal-lab-pixal3d-l40s"
DEFAULT_GPU = "L40S"
CUDA_ARCH = "8.9"
NATTEN_ARCH = "8.9"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

HF_MODEL_REPO = "TencentARC/Pixal3D"
CODE_DIR = Path("/opt/src/Pixal3D")
WHEELS = Path("/wheels")
WEIGHTS = Path("/weights")
OUTPUTS = Path("/outputs")
MESHES_DIR = OUTPUTS / "meshes"
BENCH_DIR = OUTPUTS / "benchmarks"
INPUTS_DIR = OUTPUTS / "inputs"

VOLUME_WHEELS = "modal-lab-pixal3d-l40s-wheels"
VOLUME_WEIGHTS = "modal-lab-pixal3d-l40s-weights"
VOLUME_OUTPUTS = "modal-lab-pixal3d-l40s-outputs"

AUX_HF_REPOS = (
    "Ruicheng/moge-2-vitl",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "ZhengPeng7/BiRefNet",
)
SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/TencentARC/Pixal3D/master/assets/images/5_img.webp"
)
L40S_PRICE_PER_SEC = 0.000542
INFER_TIMEOUT = 2 * 60 * 60
DOWNLOAD_TIMEOUT = 4 * 60 * 60

wheels_vol = modal.Volume.from_name(VOLUME_WHEELS, create_if_missing=True)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

app = modal.App(APP_NAME)

pixal_l40s_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "g++",
        "clang",
        "ninja-build",
        "cmake",
        "wget",
        "curl",
        "libgl1",
        "libglib2.0-0",
        "libeigen3-dev",
        "libjpeg-dev",
        "ffmpeg",
    )
    .uv_pip_install(
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
    )
    .uv_pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "triton==3.2.0",
        index_url=TORCH_INDEX,
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
            "NATTEN_CUDA_ARCH": NATTEN_ARCH,
            "ATTN_BACKEND": "sdpa",
            "CUDA_HOME": "/usr/local/cuda",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "4",
            "HF_HOME": str(WEIGHTS / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(WEIGHTS / "hf" / "hub"),
            "TORCH_HOME": str(WEIGHTS / "torch"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(CODE_DIR),
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PIXAL3D_L40S_PLAN": "A",
            "CPLUS_INCLUDE_PATH": "/usr/include/eigen3",
            "CC": "gcc",
            "CXX": "g++",
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(CODE_DIR / "autotune_cache.json"),
            "FLEX_GEMM_AUTOTUNER_VERBOSE": "0",
        }
    )
    .run_commands(
        "mkdir -p /opt/src && cd /opt/src && "
        "git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git && "
        "git clone --depth 1 https://github.com/TencentARC/Pixal3D.git && "
        "git clone --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git && "
        "git clone --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git && "
        "git clone --depth 1 -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git && "
        "git clone --depth 1 -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git",
        "pip install --no-deps "
        "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl "
        "|| pip install 'git+https://github.com/EasternJournalist/utils3d.git'",
        "pip install --no-cache-dir 'git+https://github.com/microsoft/MoGe.git'",
        "python -m pip install --no-cache-dir "
        "'huggingface_hub[hf_transfer]>=0.34.0,<1.0' 'transformers==4.57.3'",
    )
    .add_local_dir(
        str(Path(__file__).parent / "scripts"),
        remote_path="/opt/pixal3d_l40s_scripts",
        copy=True,
    )
)


def _wheel_cache_dir() -> Path:
    return WHEELS / "sm89" / "torch260-cu124-cp310"


def _build_env() -> dict:
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH
    env["NATTEN_CUDA_ARCH"] = NATTEN_ARCH
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


def _install_sm89_wheels() -> list[str]:
    d = _wheel_cache_dir()
    whls = sorted(d.glob("*.whl"))
    if not whls:
        raise RuntimeError(f"no sm_89 wheels in {d}; run build-sm89 first")
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


def _safe_name(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return s or "mesh"


def _ensure_cache_env() -> None:
    for sub in ("hf", "hf/hub", "torch", "Pixal3D"):
        (WEIGHTS / sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(WEIGHTS / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(WEIGHTS / "hf" / "hub")
    os.environ["TORCH_HOME"] = str(WEIGHTS / "torch")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["ATTN_BACKEND"] = "sdpa"


def _model_local_path() -> Path:
    return WEIGHTS / "Pixal3D"


def _model_ready() -> bool:
    root = _model_local_path()
    return (root / "pipeline.json").is_file() and (root / "ckpts").is_dir()


def _nvidia_smi_query() -> dict[str, Any] | None:
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
    except Exception as e:
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


class VramSampler:
    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self.peak_used_mib = 0.0
        self.peak_util = 0.0

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
            "peak_mem_used_gb": round(self.peak_used_mib / 1024.0, 2) if self.peak_used_mib else None,
            "mem_total_mib": total,
            "mem_total_gb": round(total / 1024.0, 2) if total else None,
            "peak_util_gpu_pct": round(self.peak_util, 1),
            "n_samples": len(self.samples),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            q = _nvidia_smi_query()
            if q and "mem_used_mib" in q:
                self.samples.append({"t": time.time(), **q})
                self.peak_used_mib = max(self.peak_used_mib, q["mem_used_mib"])
                self.peak_util = max(self.peak_util, q.get("util_gpu_pct") or 0.0)
            self._stop.wait(self.interval_s)


def _patch_pipeline_rembg(model_path: str | Path) -> None:
    pj = Path(model_path) / "pipeline.json"
    if not pj.is_file():
        return
    data = json.loads(pj.read_text(encoding="utf-8"))
    args = data.get("args") or data
    rembg = args.get("rembg_model") if isinstance(args, dict) else None
    if not isinstance(rembg, dict):
        return
    rargs = rembg.setdefault("args", {})
    old = rargs.get("model_name")
    if old and "RMBG" in str(old):
        rargs["model_name"] = "ZhengPeng7/BiRefNet"
        rembg["name"] = rembg.get("name") or "BiRefNet"
        pj.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"[rembg] patched {old} → ZhengPeng7/BiRefNet", flush=True)
        try:
            weights_vol.commit()
        except Exception as e:
            print(f"[rembg] commit warn: {e!r}", flush=True)


def _publish_mesh(src: Path, name: str, meta: dict[str, Any], input_image: Path | None = None) -> dict[str, Any]:
    try:
        outputs_vol.reload()
    except Exception as e:
        print(f"[volume] reload warn: {e!r}", flush=True)
    MESHES_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not src.is_file() or src.stat().st_size < 1000:
        raise RuntimeError(f"invalid GLB: {src}")
    named = MESHES_DIR / f"{name}.glb"
    latest = MESHES_DIR / "latest.glb"
    shutil.copy2(src, named)
    shutil.copy2(src, latest)
    size = named.stat().st_size
    if input_image and input_image.is_file():
        dest_in = INPUTS_DIR / f"{name}{input_image.suffix.lower() or '.png'}"
        shutil.copy2(input_image, dest_in)
        meta["input_volume_path"] = f"inputs/{dest_in.name}"
    payload = {
        **meta,
        "volume_name": VOLUME_OUTPUTS,
        "volume_paths": {
            "named": f"meshes/{name}.glb",
            "latest": "meshes/latest.glb",
            "meta": f"meshes/{name}_meta.json",
            "benchmark": f"benchmarks/{name}.json",
        },
        "bytes": size,
        "cli_get": f"modal volume get {VOLUME_OUTPUTS} meshes/{name}.glb ./{name}.glb",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (MESHES_DIR / f"{name}_meta.json").write_text(text, encoding="utf-8")
    (MESHES_DIR / "latest_meta.json").write_text(text, encoding="utf-8")
    (BENCH_DIR / f"{name}.json").write_text(text, encoding="utf-8")
    outputs_vol.commit()
    print(f"[VOLUME] {VOLUME_OUTPUTS}/meshes/{name}.glb ({size} bytes)", flush=True)
    return payload


def _assert_l40s() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    info = {"gpu": name, "capability": cap, "torch": torch.__version__}
    if cap != (8, 9):
        raise RuntimeError(f"expected L40S sm_89 (8,9), got {cap} on {name}")
    return info


@app.cls(
    image=pixal_l40s_image,
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    memory=32768,
    cpu=4,
    volumes={
        str(WHEELS): wheels_vol,
        str(WEIGHTS): weights_vol,
        str(OUTPUTS): outputs_vol,
    },
)
class Pixal3DL40S:
    @modal.enter()
    def enter(self) -> None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", CUDA_ARCH)
        os.environ.setdefault("NATTEN_CUDA_ARCH", NATTEN_ARCH)
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        os.environ.setdefault("CPLUS_INCLUDE_PATH", "/usr/include/eigen3")
        os.environ.setdefault("CC", "gcc")
        os.environ.setdefault("CXX", "g++")
        _ensure_cache_env()

    @modal.method()
    def status(self) -> dict:
        import torch

        info: dict[str, Any] = {
            "app": APP_NAME,
            "plan": "A",
            "cuda_available": torch.cuda.is_available(),
            "torch": torch.__version__,
            "gpu_name": None,
            "capability": None,
            "wheel_cache": str(_wheel_cache_dir()),
            "wheel_files": [],
            "model_ready": _model_ready(),
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
        d = _wheel_cache_dir()
        if d.is_dir():
            info["wheel_files"] = sorted(p.name for p in d.glob("*.whl"))
        return info

    @modal.method()
    def build_sm89(self) -> dict:
        out = _wheel_cache_dir()
        out.mkdir(parents=True, exist_ok=True)
        env = _build_env()
        steps: list[tuple[str, list[str]]] = [
            ("nvdiffrast", ["/opt/src/nvdiffrast"]),
            ("nvdiffrec_render", ["/opt/src/nvdiffrec"]),
            ("flex_gemm", ["/opt/src/FlexGEMM"]),
            ("cumesh", ["/opt/src/CuMesh"]),
            ("o_voxel", ["/opt/src/TRELLIS.2/o-voxel"]),
            ("natten", ["natten==0.21.0"]),
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

    @modal.method()
    def install_wheels_and_verify(self) -> dict:
        installed = _install_sm89_wheels()
        verify = Path("/opt/pixal3d_l40s_scripts/verify_sm89.py")
        v = subprocess.run(
            [sys.executable, str(verify), "--expect-gpu"],
            capture_output=True,
            text=True,
        )
        return {
            "ok": v.returncode == 0,
            "returncode": v.returncode,
            "stdout": v.stdout,
            "stderr": v.stderr,
            "wheels_installed": installed,
        }

    @modal.method()
    def download_weights(self, force: bool = False) -> dict:
        from huggingface_hub import snapshot_download

        _ensure_cache_env()
        root = _model_local_path()
        if force and root.exists():
            shutil.rmtree(root, ignore_errors=True)
        if not _model_ready() or force:
            print(f"[download] {HF_MODEL_REPO} → {root}", flush=True)
            snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(root))
        else:
            print(f"[download] skip main {root}", flush=True)
        _patch_pipeline_rembg(root)
        aux = []
        for repo in AUX_HF_REPOS:
            try:
                p = snapshot_download(repo_id=repo)
                aux.append({"repo": repo, "ok": True, "path": p})
            except Exception as e:
                aux.append({"repo": repo, "ok": False, "error": repr(e)})
        weights_vol.commit()
        return {"ok": True, "main_ready": _model_ready(), "aux": aux}

    @modal.method()
    def image_to_3d(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        output_name: str = "demo_l40s",
        seed: int = 42,
        low_vram: bool = True,
        resolution: int = 1024,
        fov: float = -1.0,
    ) -> dict:
        import torch

        gpu_info = _assert_l40s()
        installed = _install_sm89_wheels()
        print("[wheels]", installed, flush=True)

        # quick natten smoke
        try:
            from natten.functional import na2d
            import torch as T

            q = T.randn(1, 4, 16, 16, 32, device="cuda", dtype=T.float16)
            with T.no_grad():
                _ = na2d(q, q, q, kernel_size=3, dilation=1)
            T.cuda.synchronize()
            print("[natten] smoke OK", flush=True)
        except Exception as e:
            raise RuntimeError(f"natten smoke failed on L40S: {e!r}") from e

        _ensure_cache_env()
        if not _model_ready():
            print("[i2v] weights missing → download", flush=True)
            self.download_weights.local(force=False)

        model_path = str(_model_local_path())
        _patch_pipeline_rembg(model_path)
        safe = _safe_name(output_name)
        work = Path("/tmp/pixal3d_l40s_work")
        work.mkdir(parents=True, exist_ok=True)

        if image_bytes:
            ext = ".png"
            if image_bytes[:3] == b"\xff\xd8\xff":
                ext = ".jpg"
            elif image_bytes[:4] == b"RIFF":
                ext = ".webp"
            img_path = work / f"input{ext}"
            img_path.write_bytes(image_bytes)
        else:
            url = (image_url or SAMPLE_IMAGE_URL).strip()
            ext = Path(url.split("?")[0]).suffix or ".png"
            img_path = work / f"input{ext}"
            print(f"[i2v] fetch {url}", flush=True)
            urllib.request.urlretrieve(url, img_path)

        out_path = work / f"{safe}.glb"
        props = torch.cuda.get_device_properties(0)
        meta: dict[str, Any] = {
            "output_name": safe,
            "seed": seed,
            "low_vram": low_vram,
            "resolution": resolution if resolution > 0 else 1024,
            "fov": fov,
            "model_path": model_path,
            "gpu_request": DEFAULT_GPU,
            "gpu_actual": gpu_info["gpu"],
            "sm": "sm_89",
            "capability": list(gpu_info["capability"]),
            "vram_total_gb": round(int(props.total_memory) / 1024**3, 1),
            "price_per_sec_usd": L40S_PRICE_PER_SEC,
            "upstream": "TencentARC/Pixal3D",
            "stack": "005-v2 Plan A sm_89 source wheels",
            "wheels": installed,
        }
        print("[i2v] config", json.dumps(meta, ensure_ascii=False, indent=2), flush=True)

        sys.path.insert(0, str(CODE_DIR))
        os.chdir(CODE_DIR)
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(CODE_DIR / "autotune_cache.json"))

        from inference import run_inference

        sampler = VramSampler(2.0)
        t0 = time.time()
        meta["vram_idle"] = _nvidia_smi_query()
        sampler.start()
        try:
            run_inference(
                image_path=str(img_path),
                output_path=str(out_path),
                seed=seed,
                manual_fov=fov,
                model_path=model_path,
                low_vram=low_vram,
                resolution=resolution,
            )
        finally:
            vram = sampler.stop()
        total_s = round(time.time() - t0, 1)
        meta.update(
            {
                "seconds_total": total_s,
                "seconds_infer": total_s,
                "est_cost_usd_total": round(total_s * L40S_PRICE_PER_SEC, 4),
                "vram": vram,
            }
        )
        try:
            weights_vol.commit()
        except Exception as e:
            print(f"[weights] commit warn: {e!r}", flush=True)
        published = _publish_mesh(out_path, safe, meta, input_image=img_path)
        result = {
            "ok": True,
            "gpu_actual": gpu_info["gpu"],
            "sm": "sm_89",
            "seconds_total": total_s,
            "peak_vram_gb": vram.get("peak_mem_used_gb"),
            "est_cost_usd_total": meta["est_cost_usd_total"],
            "bytes": published["bytes"],
            "volume": VOLUME_OUTPUTS,
            "volume_file": f"meshes/{safe}.glb",
            "cli_get": published["cli_get"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="005-v2 Pixal3D L40S / sm_89")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印固定栈；纯本地")
    sub.add_parser("check", help="远程检查 L40S / wheel cache / model 状态")

    for name, help_text in (
        ("build-sm89", "源码编译并缓存 sm_89 wheels"),
        ("verify", "安装缓存 wheels 并运行 verify_sm89"),
        ("download", "下载 Pixal3D 权重到 Volume"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--i-know-this-costs-money", action="store_true")
        cmd.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("smoke", help="官方样例图 end-to-end -> GLB")
    smoke.add_argument("--i-know-this-costs-money", action="store_true")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--output-name", default="smoke_l40s")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--resolution", type=int, default=1024)
    smoke.add_argument("--fov", type=float, default=-1.0)
    low_vram = smoke.add_mutually_exclusive_group()
    low_vram.add_argument("--low-vram", action="store_true", dest="low_vram")
    low_vram.add_argument("--no-low-vram", action="store_false", dest="low_vram")
    smoke.set_defaults(low_vram=True)

    i2v = sub.add_parser("i2v", help="自定义图片 URL -> GLB")
    i2v.add_argument("--i-know-this-costs-money", action="store_true")
    i2v.add_argument("--dry-run", action="store_true")
    i2v.add_argument("--image-url", required=True)
    i2v.add_argument("--output-name", default="demo_l40s")
    i2v.add_argument("--seed", type=int, default=42)
    i2v.add_argument("--resolution", type=int, default=1024)
    i2v.add_argument("--fov", type=float, default=-1.0)
    low_vram_i2v = i2v.add_mutually_exclusive_group()
    low_vram_i2v.add_argument("--low-vram", action="store_true", dest="low_vram")
    low_vram_i2v.add_argument("--no-low-vram", action="store_false", dest="low_vram")
    i2v.set_defaults(low_vram=True)
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "005-v2-pixal3d-l40s",
        "app": APP_NAME,
        "plan": "A",
        "gpu": DEFAULT_GPU,
        "cuda_arch": CUDA_ARCH,
        "natten_arch": NATTEN_ARCH,
        "torch_index": TORCH_INDEX,
        "model": HF_MODEL_REPO,
        "wheels_volume": VOLUME_WHEELS,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "sample_image_url": SAMPLE_IMAGE_URL,
    }


def paid_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan: dict[str, Any] = {"action": args.command, "gpu": DEFAULT_GPU}
    if args.command in {"smoke", "i2v"}:
        plan.update(
            {
                "image_url": SAMPLE_IMAGE_URL if args.command == "smoke" else args.image_url,
                "output_name": args.output_name,
                "seed": args.seed,
                "low_vram": args.low_vram,
                "resolution": args.resolution,
                "fov": args.fov,
            }
        )
    return plan


def require_cost_ack(args: argparse.Namespace) -> None:
    if not args.i_know_this_costs_money:
        raise SystemExit(f"{args.command} requires --i-know-this-costs-money")


@app.local_entrypoint()
def main(*argv: str) -> None:
    args = parse_cli(argv)

    if args.command == "status":
        print(json.dumps(local_status(), ensure_ascii=False, indent=2))
        return

    worker = Pixal3DL40S()
    if args.command == "check":
        print(worker.status.remote())
        return

    plan = paid_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    require_cost_ack(args)

    if args.command == "build-sm89":
        print(worker.build_sm89.remote())
        return
    if args.command == "verify":
        print(worker.install_wheels_and_verify.remote())
        return
    if args.command == "download":
        print(worker.download_weights.remote())
        return

    print(
        worker.image_to_3d.remote(
            image_bytes=None,
            image_url=plan["image_url"],
            output_name=plan["output_name"],
            seed=plan["seed"],
            low_vram=plan["low_vram"],
            resolution=plan["resolution"],
            fov=plan["fov"],
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
