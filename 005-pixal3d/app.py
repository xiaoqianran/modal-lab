# -*- coding: utf-8 -*-
"""
005-pixal3d — TencentARC Pixal3D 单图 → GLB（Modal Volume 唯一权威输出）。

默认 GPU : H100（HF demo 轮子 = Hopper sm_90）
A100-40GB : 需 Volume 缓存 natten（build-natten 一次）
RTX-PRO-6000 / L40S : 当前 torch2.6 栈不可用（见 GPU_BENCHMARK.md）
"""

from __future__ import annotations

import argparse
import glob
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

APP_NAME = "modal-lab-pixal3d"
DEFAULT_GPU = "H100"
DEFAULT_MEMORY_MB = 24576
DEFAULT_CPU = 4.0
PIXAL3D_REPO = "https://github.com/TencentARC/Pixal3D.git"
PIXAL3D_COMMIT = "master"
HF_MODEL_REPO = "TencentARC/Pixal3D"
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
CODE_DIR = Path("/opt/Pixal3D")
MESHES_DIR = Path(OUTPUTS_MOUNT) / "meshes"
BENCH_DIR = Path(OUTPUTS_MOUNT) / "benchmarks"
INPUTS_DIR = Path(OUTPUTS_MOUNT) / "inputs"
VOLUME_OUTPUTS_NAME = "modal-lab-pixal3d-outputs"
VOLUME_WEIGHTS_NAME = "modal-lab-pixal3d-weights"

AUX_HF_REPOS = (
    "Ruicheng/moge-2-vitl",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "ZhengPeng7/BiRefNet",
)

GPU_PRICE_PER_SEC = {
    "RTX-PRO-6000": 0.000842,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "A100": 0.000583,
    "L40S": 0.000542,
    "H100": 0.001097,
    "H100!": 0.001097,
}

DOWNLOAD_TIMEOUT = 4 * 60 * 60
INFER_TIMEOUT = 2 * 60 * 60
SMOKE_TIMEOUT = 90 * 60
BUILD_NATTEN_TIMEOUT = 3 * 60 * 60

WHEEL_NATTEN = (
    "https://github.com/LDYang694/Storages/releases/download/20260430/"
    "natten-0.21.0+torch2.6cu124-cp310-cp310-linux_x86_64.whl"
)
WHEEL_UTILS3D = (
    "https://github.com/LDYang694/Storages/releases/download/20260430/"
    "utils3d-0.0.2-py3-none-any.whl"
)
WHEEL_FLASH_ATTN3 = (
    "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
    "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
)
WHEEL_CUMESH = (
    "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
    "cumesh-0.0.1-cp310-cp310-linux_x86_64.whl"
)
WHEEL_FLEX_GEMM = (
    "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
    "flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl"
)
WHEEL_O_VOXEL = (
    "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
    "o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl"
)
WHEEL_NVDIFFRAST = (
    "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
    "nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl"
)
WHEEL_NVDIFFREC = (
    "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
    "nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl"
)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = EXP_DIR / "inputs" / "sample.webp"

SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/TencentARC/Pixal3D/master/assets/images/5_img.webp"
)

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS_NAME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6",
        "libxrender1", "wget", "curl", "ca-certificates",
        "build-essential", "ninja-build", "cmake",
    )
    .uv_pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "triton==3.2.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .uv_pip_install(
        "pillow==12.0.0", "imageio==2.37.2", "imageio-ffmpeg==0.6.0",
        "tqdm==4.67.1", "easydict==1.13", "opencv-python-headless==4.12.0.88",
        "trimesh==4.10.1", "transformers==4.57.3", "zstandard==0.25.0",
        "kornia==0.8.2", "timm==1.0.22", "diffusers==0.37.1",
        "accelerate==1.13.0", "plyfile==1.1.3", "safetensors",
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0", "einops", "scipy",
        "numpy", "fastapi",
    )
    .run_commands(
        f"git clone --depth 1 --branch {PIXAL3D_COMMIT} {PIXAL3D_REPO} {CODE_DIR}",
        "python -m pip install --no-cache-dir 'git+https://github.com/microsoft/MoGe.git'",
        f"python -m pip install --no-cache-dir "
        f"{WHEEL_NATTEN} {WHEEL_UTILS3D} {WHEEL_FLASH_ATTN3} {WHEEL_CUMESH} "
        f"{WHEEL_FLEX_GEMM} {WHEEL_O_VOXEL} {WHEEL_NVDIFFRAST} {WHEEL_NVDIFFREC}",
        "python -m pip install --no-cache-dir "
        "'huggingface_hub[hf_transfer]>=0.34.0,<1.0' 'transformers==4.57.3'",
        "python -c \"import huggingface_hub as h, transformers as t; "
        "assert h.__version__.startswith('0.'), h.__version__; "
        "print('hub', h.__version__, 'transformers', t.__version__)\"",
        "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)\"",
    )
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf",
        "HUGGINGFACE_HUB_CACHE": f"{WEIGHTS_MOUNT}/hf/hub",
        "TORCH_HOME": f"{WEIGHTS_MOUNT}/torch",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(CODE_DIR),
        "OPENCV_IO_ENABLE_OPENEXR": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "ATTN_BACKEND": "sdpa",
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(CODE_DIR / "autotune_cache.json"),
        "FLEX_GEMM_AUTOTUNER_VERBOSE": "0",
    })
)

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "python-multipart")
    .env({"PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME)


def _list_dir_sizes(root: str | Path) -> dict[str, Any]:
    p = Path(root)
    if not p.exists():
        return {"exists": False, "path": str(root), "files": 0, "size_gb": 0.0}
    total = files = 0
    for f in p.rglob("*"):
        if f.is_file():
            files += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return {"exists": True, "path": str(root), "files": files, "size_gb": round(total / 1e9, 2)}


def _nvidia_smi_query() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
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
    def __init__(self, interval_s: float = 1.0) -> None:
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
        return self.summary()

    def _loop(self) -> None:
        while not self._stop.is_set():
            q = _nvidia_smi_query()
            if q and "mem_used_mib" in q:
                self.samples.append({"t": time.time(), **q})
                self.peak_used_mib = max(self.peak_used_mib, q["mem_used_mib"])
                self.peak_util = max(self.peak_util, q.get("util_gpu_pct") or 0.0)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
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


def _safe_name(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return s or "mesh"


def _normalize_gpu(label: str) -> str:
    key = (label or DEFAULT_GPU).strip()
    aliases = {
        "L40": "L40S", "l40": "L40S", "l40s": "L40S",
        "pro6000": "RTX-PRO-6000", "PRO-6000": "RTX-PRO-6000", "RTX_PRO_6000": "RTX-PRO-6000",
        "A100_40GB": "A100-40GB", "A100_80GB": "A100-80GB",
        "a100-40": "A100-40GB", "a100-80": "A100-80GB",
    }
    return aliases.get(key, key)


def _ensure_cache_env() -> None:
    for sub in ("hf", "hf/hub", "torch", "Pixal3D", "wheels/natten"):
        Path(WEIGHTS_MOUNT, sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = f"{WEIGHTS_MOUNT}/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{WEIGHTS_MOUNT}/hf/hub"
    os.environ["TORCH_HOME"] = f"{WEIGHTS_MOUNT}/torch"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def _model_local_path() -> Path:
    return Path(WEIGHTS_MOUNT) / "Pixal3D"


def _model_ready() -> bool:
    root = _model_local_path()
    return (root / "pipeline.json").is_file() and (root / "ckpts").is_dir()


def _publish_mesh_to_volume(
    src: Path, name: str, meta: dict[str, Any], input_image: Path | None = None,
) -> dict[str, Any]:
    try:
        outputs_vol.reload()
    except Exception as e:
        print(f"[volume] reload warn: {e!r}", flush=True)

    MESHES_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    named = MESHES_DIR / f"{name}.glb"
    latest = MESHES_DIR / "latest.glb"
    if not src.is_file() or src.stat().st_size < 1000:
        size_hint = src.stat().st_size if src.exists() else 0
        raise RuntimeError(f"源 GLB 无效: {src} size={size_hint}")

    shutil.copy2(src, named)
    shutil.copy2(src, latest)
    size = named.stat().st_size

    if input_image and input_image.is_file():
        dest_in = INPUTS_DIR / f"{name}{input_image.suffix.lower() or '.png'}"
        shutil.copy2(input_image, dest_in)
        meta["input_volume_path"] = f"inputs/{dest_in.name}"

    payload: dict[str, Any] = {
        **meta,
        "volume_name": VOLUME_OUTPUTS_NAME,
        "volume_paths": {
            "named": f"meshes/{name}.glb",
            "latest": "meshes/latest.glb",
            "meta": f"meshes/{name}_meta.json",
            "latest_meta": "meshes/latest_meta.json",
            "benchmark": f"benchmarks/{name}.json",
        },
        "bytes": size,
        "download_url_hint": f"https://seachenxyt--modal-lab-pixal3d-download.modal.run?name={name}",
        "cli_get": f"modal volume get {VOLUME_OUTPUTS_NAME} meshes/{name}.glb ./{name}.glb",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (MESHES_DIR / f"{name}_meta.json").write_text(text, encoding="utf-8")
    (MESHES_DIR / "latest_meta.json").write_text(text, encoding="utf-8")
    (BENCH_DIR / f"{name}.json").write_text(text, encoding="utf-8")
    outputs_vol.commit()
    print("=" * 60, flush=True)
    print("[VOLUME COMMITTED] GLB 已写入远程 Modal Volume", flush=True)
    print(f"  {VOLUME_OUTPUTS_NAME}/meshes/{name}.glb  ({size} bytes)", flush=True)
    print("=" * 60, flush=True)
    return payload


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


def _cuda_sm_tag() -> str:
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


def _assert_gpu_supported() -> dict[str, Any]:
    import torch
    name = str(torch.cuda.get_device_name(0))
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    info: dict[str, Any] = {
        "gpu": name, "sm": sm, "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    if hasattr(torch.cuda, "get_arch_list"):
        info["torch_archs"] = list(torch.cuda.get_arch_list())
    if major >= 12:
        raise RuntimeError(
            f"GPU {name} ({sm}) needs Blackwell PyTorch; "
            f"image is torch==2.6.0+cu124 (≤sm_90). Use H100 or A100-40GB. detail={info}"
        )
    return info


def _natten_smoke() -> bool:
    try:
        import torch
        from natten.functional import na2d  # type: ignore
        layouts = [
            ("BheadsHWC", torch.randn(1, 4, 16, 16, 32, device="cuda", dtype=torch.float16)),
            ("BHWC", torch.randn(1, 16, 16, 32, device="cuda", dtype=torch.float16)),
        ]
        last: Exception | None = None
        for name, q in layouts:
            try:
                with torch.no_grad():
                    _ = na2d(q, q, q, kernel_size=3, dilation=1)
                torch.cuda.synchronize()
                print(f"[natten] smoke OK layout={name}", flush=True)
                return True
            except Exception as e:
                last = e
                msg = repr(e).lower()
                if "no kernel image" in msg or "cuda error" in msg:
                    print(f"[natten] smoke CUDA fail {name}: {e!r}", flush=True)
                    return False
                print(f"[natten] smoke try {name}: {e!r}", flush=True)
        print(f"[natten] smoke fail: {last!r}", flush=True)
        return False
    except Exception as e:
        print(f"[natten] smoke import/fail: {e!r}", flush=True)
        return False


def _find_cached_natten_wheel(sm: str) -> Path | None:
    """Prefer PEP427-valid wheels; skip broken *linux_x86_64+sm*.whl names."""
    wheel_dir = Path(WEIGHTS_MOUNT) / "wheels" / "natten"
    if not wheel_dir.is_dir():
        return None
    sm_compact = sm.replace("_", "")
    good: list[Path] = []
    for p in sorted(wheel_dir.glob("*.whl")):
        name = p.name
        if "linux_x86_64+" in name or "any+" in name:
            print(f"[natten] skip invalid wheel name: {name}", flush=True)
            continue
        if f"+{sm_compact}" in name or f"+{sm}" in name or sm in name or sm_compact in name:
            good.append(p)
    if good:
        return good[-1]
    for p in sorted(wheel_dir.glob("*.whl")):
        if "linux_x86_64+" not in p.name:
            return p
    return None


def _pip_install_wheel(path: Path) -> None:
    import sys
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--force-reinstall", str(path),
    ])


def _tag_wheel_path(src: Path, sm: str, wheel_dir: Path) -> Path:
    stem = src.stem
    local = sm.replace("_", "")
    if "-cp" in stem:
        name_ver, rest = stem.split("-cp", 1)
        return wheel_dir / f"{name_ver}+{local}-cp{rest}{src.suffix}"
    return wheel_dir / f"{stem}+{local}{src.suffix}"


def _ensure_natten_for_device() -> dict[str, Any]:
    import sys
    import torch

    info = _assert_gpu_supported()
    sm = info["sm"]
    try:
        weights_vol.reload()
    except Exception as e:
        print(f"[natten] volume reload warn: {e!r}", flush=True)

    if sm == "sm_90":
        if _natten_smoke():
            info["action"] = "reuse_installed"
            print(f"[natten] OK installed for {sm}", flush=True)
            return info
    else:
        cached = _find_cached_natten_wheel(sm)
        if cached is not None:
            print(f"[natten] non-Hopper {sm}: install cached {cached.name}", flush=True)
            try:
                _pip_install_wheel(cached)
                if _natten_smoke():
                    info["action"] = "install_cached"
                    info["wheel"] = str(cached)
                    return info
                print("[natten] cached wheel unusable (need libnatten) → rebuild", flush=True)
            except Exception as e:
                print(f"[natten] cached install failed: {e!r} → rebuild", flush=True)
        elif _natten_smoke():
            info["action"] = "reuse_installed"
            return info

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"{major}.{minor}"
    wheel_dir = Path(WEIGHTS_MOUNT) / "wheels" / "natten"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("cmake") is None:
        subprocess.check_call(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq cmake"])

    print(f"[natten] building from source NATTEN_CUDA_ARCH={arch} …", flush=True)
    env = os.environ.copy()
    env["NATTEN_CUDA_ARCH"] = arch
    env["TORCH_CUDA_ARCH_LIST"] = arch
    env["NATTEN_N_WORKERS"] = "8"
    env["MAX_JOBS"] = "8"
    cmd = [
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "--force-reinstall", "--no-build-isolation",
        "git+https://github.com/SHI-Labs/NATTEN.git@v0.21.1",
    ]
    print(f"[natten] $ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env=env)

    if not _natten_smoke():
        raise RuntimeError(
            f"natten build finished but smoke still fails on {sm} ({info['gpu']}). Prefer H100."
        )

    candidates: list[str] = []
    for pat in (
        "/tmp/pip-ephem-wheel-cache-*/wheels/**/NATTEN-*.whl",
        "/tmp/pip-ephem-wheel-cache-*/wheels/**/natten-*.whl",
    ):
        candidates.extend(glob.glob(pat, recursive=True))
    if candidates:
        src = Path(sorted(candidates, key=lambda p: Path(p).stat().st_mtime)[-1])
        tagged = _tag_wheel_path(src, sm, wheel_dir)
        shutil.copy2(src, tagged)
        info["wheel"] = str(tagged)
        print(f"[natten] cached wheel → {tagged}", flush=True)
        try:
            weights_vol.commit()
        except Exception as e:
            print(f"[natten] volume commit warn: {e!r}", flush=True)

    info["action"] = "built_from_source"
    info["arch"] = arch
    print(f"[natten] build OK for {sm}", flush=True)
    return info


def _run_pixal3d_inference(
    image_path: Path, output_path: Path, seed: int, low_vram: bool,
    resolution: int, fov: float, model_path: str,
) -> dict[str, Any]:
    import sys
    os.environ["ATTN_BACKEND"] = "sdpa"
    _ensure_cache_env()
    try:
        import huggingface_hub as _hh
        if not str(_hh.__version__).startswith("0."):
            print(f"[deps] hub={_hh.__version__} → pin <1.0", flush=True)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "huggingface_hub[hf_transfer]>=0.34.0,<1.0", "transformers==4.57.3",
            ])
    except Exception as e:
        print(f"[deps] hub pin warn: {e!r}", flush=True)

    sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(CODE_DIR / "autotune_cache.json"))

    _patch_pipeline_rembg(model_path)
    natten_info = _ensure_natten_for_device()
    print("[natten] setup", natten_info, flush=True)

    from inference import run_inference
    t0 = time.time()
    run_inference(
        image_path=str(image_path), output_path=str(output_path), seed=seed,
        manual_fov=fov, model_path=model_path, low_vram=low_vram, resolution=resolution,
    )
    return {"seconds_infer": round(time.time() - t0, 1), "natten": natten_info}


@app.function(
    image=image, gpu=DEFAULT_GPU, volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=BUILD_NATTEN_TIMEOUT, memory=32768, cpu=8,
)
def build_natten(gpu_label: str = DEFAULT_GPU) -> dict[str, Any]:
    import torch
    _ensure_cache_env()
    print(json.dumps({
        "gpu_request": gpu_label,
        "gpu_actual": torch.cuda.get_device_name(0),
        "sm": _cuda_sm_tag(),
        "torch": torch.__version__,
    }, indent=2), flush=True)
    info = _ensure_natten_for_device()
    result = {"ok": True, **info}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=image, volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT, memory=16384, cpu=2,
)
def download_weights(force: bool = False, with_aux: bool = True) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    _ensure_cache_env()
    root = _model_local_path()
    result: dict[str, Any] = {"force": force, "with_aux": with_aux}
    if force and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    if not _model_ready() or force:
        print(f"[download] snapshot {HF_MODEL_REPO} → {root}", flush=True)
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(root))
    else:
        print(f"[download] skip main: {root}", flush=True)
    _patch_pipeline_rembg(root)
    aux_ok: list[Any] = []
    if with_aux:
        for repo in AUX_HF_REPOS:
            print(f"[download] aux {repo}", flush=True)
            try:
                p = snapshot_download(repo_id=repo)
                aux_ok.append({"repo": repo, "ok": True, "path": p})
            except Exception as e:
                aux_ok.append({"repo": repo, "ok": False, "error": repr(e)})
    try:
        weights_vol.commit()
    except Exception as e:
        print(f"[weights] commit warn: {e!r}", flush=True)
    result.update({
        "ok": True, "main_repo": HF_MODEL_REPO, "main_path": str(root),
        "main_ready": _model_ready(), "main_size": _list_dir_sizes(root),
        "hf_cache": _list_dir_sizes(Path(WEIGHTS_MOUNT) / "hf"),
        "aux_downloaded": aux_ok, "volume": VOLUME_WEIGHTS_NAME,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=image, gpu=DEFAULT_GPU,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=INFER_TIMEOUT, memory=DEFAULT_MEMORY_MB, cpu=DEFAULT_CPU,
)
def image_to_3d(
    image_bytes: bytes | None = None, image_url: str | None = None,
    output_name: str = "i2v", seed: int = 42, low_vram: bool = True,
    resolution: int = 1024, fov: float = -1.0, gpu_label: str = DEFAULT_GPU,
) -> dict[str, Any]:
    import torch
    gpu_label = _normalize_gpu(gpu_label)
    _ensure_cache_env()
    if not _model_ready():
        print("[i2v] weights missing → download", flush=True)
        download_weights.local(force=False, with_aux=True)

    model_path = str(_model_local_path())
    safe = _safe_name(output_name)
    work = Path("/tmp/pixal3d_work")
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
        print(f"[i2v] download image {url}", flush=True)
        urllib.request.urlretrieve(url, img_path)

    out_path = work / f"{safe}.glb"
    actual_gpu = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    vram_total_gb = round(int(props.total_memory) / 1024**3, 1) if props else None
    sm = None
    if props:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
    price = GPU_PRICE_PER_SEC.get(gpu_label)

    meta: dict[str, Any] = {
        "output_name": safe, "seed": seed, "low_vram": low_vram,
        "resolution": resolution if resolution > 0 else (1024 if low_vram else 1536),
        "fov": fov, "model_path": model_path, "gpu_request": gpu_label,
        "gpu_actual": actual_gpu, "vram_total_gb": vram_total_gb, "sm": sm,
        "price_per_sec_usd": price, "upstream": "TencentARC/Pixal3D",
        "pipeline": "Pixal3DImageTo3DPipeline",
        "container_memory_mb": DEFAULT_MEMORY_MB, "container_cpu": DEFAULT_CPU,
    }
    print("[i2v] config:", json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    if actual_gpu:
        print(f"[i2v] allocated GPU: {actual_gpu} ({vram_total_gb} GB)", flush=True)

    sampler = VramSampler(interval_s=2.0)
    t0 = time.time()
    try:
        meta["vram_idle"] = _nvidia_smi_query()
        sampler.start()
        infer_meta = _run_pixal3d_inference(
            img_path, out_path, seed=seed, low_vram=low_vram,
            resolution=resolution, fov=fov, model_path=model_path,
        )
        vram = sampler.stop()
        total_s = round(time.time() - t0, 1)
        infer_s = float(infer_meta.get("seconds_infer") or total_s)
        cost_total = round(total_s * price, 4) if price else None
        cost_infer = round(infer_s * price, 4) if price else None
        meta.update({
            **infer_meta, "seconds_total": total_s,
            "est_cost_usd_total": cost_total, "est_cost_usd_infer_only": cost_infer,
            "vram": vram,
        })
        try:
            weights_vol.commit()
        except Exception as e:
            print(f"[weights] commit warn: {e!r}", flush=True)
        published = _publish_mesh_to_volume(out_path, safe, meta, input_image=img_path)
        result = {
            "ok": True, "where": "REMOTE Modal Volume ONLY",
            "volume_name": VOLUME_OUTPUTS_NAME,
            "volume_file": f"meshes/{safe}.glb", "volume_latest": "meshes/latest.glb",
            "bytes": published["bytes"], "gpu_request": gpu_label, "gpu_actual": actual_gpu,
            "sm": sm, "seconds_total": total_s, "seconds_infer": infer_s,
            "peak_vram_gb": vram.get("peak_mem_used_gb"),
            "est_cost_usd_total": cost_total, "est_cost_usd_infer_only": cost_infer,
            "download_url": published["download_url_hint"], "cli_get": published["cli_get"],
            "paths": published["volume_paths"], "low_vram": low_vram,
            "resolution": meta["resolution"], "natten": infer_meta.get("natten"),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    except Exception:
        sampler.stop()
        raise


@app.function(
    image=image, gpu=DEFAULT_GPU,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=SMOKE_TIMEOUT, memory=DEFAULT_MEMORY_MB, cpu=DEFAULT_CPU,
)
def smoke(gpu_label: str = DEFAULT_GPU) -> dict[str, Any]:
    return image_to_3d.local(
        image_bytes=None, image_url=SAMPLE_IMAGE_URL, output_name="smoke_sample",
        seed=42, low_vram=True, resolution=1024, fov=-1.0,
        gpu_label=_normalize_gpu(gpu_label),
    )


@app.function(image=serve_image, volumes={OUTPUTS_MOUNT: outputs_vol}, timeout=120, cpu=1, memory=2048)
def list_outputs() -> dict[str, Any]:
    outputs_vol.reload()
    items = []
    if MESHES_DIR.is_dir():
        for f in sorted(MESHES_DIR.glob("*.glb")):
            items.append({
                "name": f.name, "volume_path": f"meshes/{f.name}", "bytes": f.stat().st_size,
                "download_url": f"https://seachenxyt--modal-lab-pixal3d-download.modal.run?name={f.stem}",
            })
    result = {
        "volume_name": VOLUME_OUTPUTS_NAME, "count": len(items), "meshes": items,
        "note": "远程 Modal Volume，不在仓库本地",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(image=serve_image, volumes={OUTPUTS_MOUNT: outputs_vol}, timeout=300, cpu=1, memory=4096)
@modal.fastapi_endpoint(method="GET")
def download(name: str = "latest"):
    from fastapi.responses import FileResponse, JSONResponse
    outputs_vol.reload()
    safe = "".join(c if c.isalnum() or c in "-_." else "" for c in name) or "latest"
    path = MESHES_DIR / (safe if safe.endswith(".glb") else f"{safe}.glb")
    if not path.is_file():
        alts = sorted(MESHES_DIR.glob("*.glb")) if MESHES_DIR.is_dir() else []
        return JSONResponse(
            {"error": f"not found: {path.name}", "volume": VOLUME_OUTPUTS_NAME,
             "available": [p.name for p in alts]},
            status_code=404,
        )
    return FileResponse(
        path=str(path), media_type="model/gltf-binary", filename=path.name,
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


@app.function(image=serve_image, volumes={OUTPUTS_MOUNT: outputs_vol}, timeout=120, cpu=1, memory=2048)
@modal.fastapi_endpoint(method="GET")
def index():
    from fastapi.responses import HTMLResponse
    outputs_vol.reload()
    rows = []
    if MESHES_DIR.is_dir():
        for f in sorted(MESHES_DIR.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
            mb = f.stat().st_size / 1e6
            rows.append(f'<li><a href="download?name={f.stem}"><b>{f.name}</b></a> ({mb:.2f} MB)</li>')
    body = "\n".join(rows) or "<li>(empty)</li>"
    return HTMLResponse(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>{VOLUME_OUTPUTS_NAME}</title>
<style>body{{font-family:system-ui;max-width:800px;margin:2rem auto;padding:0 1rem}}
code{{background:#f4f4f4;padding:2px 6px}}</style></head>
<body><h1>Pixal3D GLB · Volume</h1><p><code>{VOLUME_OUTPUTS_NAME}</code></p>
<ul>{body}</ul><p><a href="download?name=latest"><b>⬇ latest.glb</b></a></p></body></html>"""
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="005 Pixal3D official-stack image -> GLB")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")
    sub.add_parser("list-outputs", help="结构化列出远程 GLB")

    download_cmd = sub.add_parser("download", help="下载主权重 + 辅助模型")
    download_cmd.add_argument("--force", action="store_true")
    download_cmd.add_argument("--no-aux", action="store_true")
    download_cmd.add_argument("--dry-run", action="store_true")

    build_cmd = sub.add_parser("build-natten", help="在目标 GPU 编译/缓存 natten")
    build_cmd.add_argument("--gpu", default="A100-40GB")
    build_cmd.add_argument("--dry-run", action="store_true")

    smoke_cmd = sub.add_parser("smoke", help="官方样例图冒烟")
    smoke_cmd.add_argument("--gpu", default=DEFAULT_GPU)
    smoke_cmd.add_argument("--dry-run", action="store_true")

    i2v_cmd = sub.add_parser("i2v", help="Image-to-3D -> GLB")
    i2v_cmd.add_argument("--image", type=Path)
    i2v_cmd.add_argument("--image-url", default="")
    i2v_cmd.add_argument("--output-name", default="i2v")
    i2v_cmd.add_argument("--seed", type=int, default=42)
    i2v_cmd.add_argument("--resolution", type=int, choices=[1024, 1536], default=1024)
    i2v_cmd.add_argument("--fov", type=float, default=-1.0)
    i2v_cmd.add_argument("--full-vram", action="store_true")
    i2v_cmd.add_argument("--gpu", default=DEFAULT_GPU)
    i2v_cmd.add_argument("--dry-run", action="store_true")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "005-pixal3d",
        "app": APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "hf_model": HF_MODEL_REPO,
        "weights_volume": VOLUME_WEIGHTS_NAME,
        "outputs_volume": VOLUME_OUTPUTS_NAME,
        "defaults": {"low_vram": True, "resolution": 1024},
        "gpu_notes": {
            "H100": "推荐：HF demo 轮子原生 sm_90",
            "A100-40GB": "可用；首次 build-natten，之后 Volume 缓存",
            "RTX-PRO-6000": "当前 torch2.6 栈不可用；见 005-v3",
            "L40S": "当前 HF natten 栈不可用；见 005-v2",
        },
    }


def i2v_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_name = args.output_name.strip()
    if not output_name:
        raise ValueError("--output-name 不能为空")
    local_image = args.image
    image_url = args.image_url.strip()
    if local_image is not None and image_url:
        raise ValueError("--image 与 --image-url 二选一")
    if local_image is not None:
        local_image = local_image.expanduser().resolve()
        if not local_image.is_file():
            raise ValueError(f"本地图片不存在: {local_image}")
    elif not image_url:
        if DEFAULT_IMAGE.is_file():
            local_image = DEFAULT_IMAGE.resolve()
        else:
            image_url = SAMPLE_IMAGE_URL
    return {
        "action": "i2v",
        "local_image": str(local_image) if local_image else "",
        "image_url": image_url,
        "output_name": output_name,
        "seed": args.seed,
        "low_vram": not args.full_vram,
        "resolution": args.resolution,
        "fov": args.fov,
        "gpu": _normalize_gpu(args.gpu),
    }


@app.local_entrypoint()
def main(*argv: str) -> None:
    args = parse_cli(argv)
    if args.command == "status":
        print(json.dumps(local_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "list-outputs":
        print(json.dumps(list_outputs.remote(), ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        plan = {"action": "download", "force": args.force, "with_aux": not args.no_aux}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(download_weights.remote(force=args.force, with_aux=not args.no_aux), ensure_ascii=False, indent=2))
        return
    if args.command == "build-natten":
        gpu = _normalize_gpu(args.gpu)
        plan = {"action": "build-natten", "gpu": gpu}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(build_natten.with_options(gpu=gpu).remote(gpu_label=gpu), ensure_ascii=False, indent=2))
        return
    if args.command == "smoke":
        gpu = _normalize_gpu(args.gpu)
        plan = {"action": "smoke", "gpu": gpu}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(smoke.with_options(gpu=gpu).remote(gpu_label=gpu), ensure_ascii=False, indent=2))
        return

    try:
        plan = i2v_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    image_bytes = None
    if plan["local_image"]:
        path = Path(plan["local_image"])
        image_bytes = path.read_bytes()
        print(f"[local] upload {path} ({len(image_bytes)} bytes)", flush=True)
    result = image_to_3d.with_options(gpu=plan["gpu"]).remote(
        image_bytes=image_bytes,
        image_url=plan["image_url"] or None,
        output_name=plan["output_name"],
        seed=plan["seed"],
        low_vram=plan["low_vram"],
        resolution=plan["resolution"],
        fov=plan["fov"],
        gpu_label=plan["gpu"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
