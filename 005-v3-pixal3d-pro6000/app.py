"""005-v3 Pixal3D on RTX PRO 6000 (Blackwell sm_120).

Plan A*: torch 2.11.0+cu128 · TORCH_CUDA_ARCH_LIST=12.0 · source wheels.
Gates: probe → build-sm120 → verify → smoke.
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

APP_NAME = "modal-lab-pixal3d-pro6000"
DEFAULT_GPU = "RTX-PRO-6000"
CUDA_ARCH = "12.0"
NATTEN_ARCH = "12.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

HF_MODEL_REPO = "TencentARC/Pixal3D"
CODE_DIR = Path("/opt/src/Pixal3D")
WHEELS = Path("/wheels")
WEIGHTS = Path("/weights")
OUTPUTS = Path("/outputs")
MESHES_DIR = OUTPUTS / "meshes"
BENCH_DIR = OUTPUTS / "benchmarks"
INPUTS_DIR = OUTPUTS / "inputs"

VOLUME_WHEELS = "modal-lab-pixal3d-pro6000-wheels"
VOLUME_WEIGHTS = "modal-lab-pixal3d-pro6000-weights"
VOLUME_OUTPUTS = "modal-lab-pixal3d-pro6000-outputs"

AUX_HF_REPOS = (
    "Ruicheng/moge-2-vitl",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "ZhengPeng7/BiRefNet",
)
SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/TencentARC/Pixal3D/master/assets/images/5_img.webp"
)
PRO6000_PRICE_PER_SEC = 0.000842

wheels_vol = modal.Volume.from_name(VOLUME_WHEELS, create_if_missing=True)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)

app = modal.App(APP_NAME)

pixal_pro6000_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.10",
    )
    .apt_install(
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
        "libglib2.0-0t64",
        "libeigen3-dev",
        "libjpeg-dev",
        "ffmpeg",
        "ca-certificates",
    )
    .run_commands("gcc --version | head -1 && nvcc --version | tail -1 || true")
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
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url=TORCH_INDEX,
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
            "NATTEN_CUDA_ARCH": NATTEN_ARCH,
            "NATTEN_N_WORKERS": "8",
            "ATTN_BACKEND": "sdpa",
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
            "PIXAL3D_PRO6000_PLAN": "A_star",
            "CPLUS_INCLUDE_PATH": "/usr/include/eigen3",
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(CODE_DIR / "autotune_cache.json"),
            "FLEX_GEMM_AUTOTUNER_VERBOSE": "0",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
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
        "pip install --no-cache-dir 'git+https://github.com/microsoft/MoGe.git' || true",
        "python -m pip install --no-cache-dir "
        "'huggingface_hub[hf_transfer]>=0.34.0,<1.0' 'transformers==4.57.3'",
        "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); "
        "print('arch_list', torch.cuda.get_arch_list() if hasattr(torch.cuda,'get_arch_list') else None)\"",
    )
    .add_local_dir(
        str(Path(__file__).parent / "scripts"),
        remote_path="/opt/pixal3d_pro6000_scripts",
        copy=True,
    )
)


def _jsonable(obj: Any) -> Any:
    """Force plain JSON types so Modal client (no torch) can deserialize."""
    return json.loads(json.dumps(obj, default=str))


def _wheel_cache_dir() -> Path:
    return WHEELS / "sm120" / "torch211-cu128-cp310"


def _build_env() -> dict:
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH
    env["NATTEN_CUDA_ARCH"] = NATTEN_ARCH
    env["NATTEN_N_WORKERS"] = env.get("NATTEN_N_WORKERS", "8")
    env["CUDA_HOME"] = env.get("CUDA_HOME", "/usr/local/cuda")
    env["CUDACXX"] = f"{env['CUDA_HOME']}/bin/nvcc"
    env["CPLUS_INCLUDE_PATH"] = "/usr/include/eigen3"
    env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
    env["FORCE_CUDA"] = "1"
    env["CC"] = env.get("CC", "gcc")
    env["CXX"] = env.get("CXX", "g++")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
    shim = Path("/tmp/bin-shim")
    shim.mkdir(exist_ok=True)
    gxx = shutil.which("g++") or "/usr/bin/g++"
    gcc = shutil.which("gcc") or "/usr/bin/gcc"
    for name, target in (("clang++", gxx), ("clang", gcc)):
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
    if name == "drtk":
        return list(out.glob("drtk*.whl"))
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


def _install_sm120_wheels() -> list[str]:
    d = _wheel_cache_dir()
    whls = sorted(d.glob("*.whl"))
    if not whls:
        raise RuntimeError(f"no sm_120 wheels in {d}; run build-sm120 first")
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


def _publish_mesh(
    src: Path, name: str, meta: dict[str, Any], input_image: Path | None = None
) -> dict[str, Any]:
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


def _probe_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "app": APP_NAME,
        "plan": "A_star",
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "capability": None,
        "sm": None,
        "arch_list": [str(a) for a in (torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else [])],
        "smi": _nvidia_smi_query(),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "nvcc": None,
        "gcc": None,
        "wheel_cache": str(_wheel_cache_dir()),
        "wheel_files": [],
        "model_ready": bool(_model_ready()),
        "gates": {},
        "matmul_ok": False,
    }
    try:
        info["nvcc"] = subprocess.check_output(
            ["nvcc", "--version"], text=True, timeout=10
        ).strip().splitlines()[-1]
    except Exception as e:
        info["nvcc"] = repr(e)
    try:
        info["gcc"] = subprocess.check_output(["gcc", "--version"], text=True, timeout=5).splitlines()[0]
    except Exception as e:
        info["gcc"] = repr(e)

    if torch.cuda.is_available():
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = [int(cap[0]), int(cap[1])]
        info["sm"] = f"sm_{cap[0]}{cap[1]}"
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize()
        info["matmul_ok"] = bool(int(c.numel()) > 0)

    d = _wheel_cache_dir()
    if d.is_dir():
        info["wheel_files"] = sorted(p.name for p in d.glob("*.whl"))

    arch_list = info["arch_list"] or []
    cap_t = tuple(info["capability"] or ())
    info["gates"] = {
        "cuda_available": info["cuda_available"],
        "capability_is_12x": len(cap_t) == 2 and cap_t[0] == 12,
        "sm_120_in_arch_list": any("120" in str(a) for a in arch_list),
        "torch_is_cu128_plus": any(x in str(torch.__version__) for x in ("cu128", "cu129", "cu130")),
        "matmul_ok": bool(info.get("matmul_ok")),
    }
    info["gates"]["B0_pass"] = all(
        [
            info["gates"]["cuda_available"],
            info["gates"]["capability_is_12x"],
            info["gates"]["matmul_ok"],
            info["gates"]["torch_is_cu128_plus"] or info["gates"]["sm_120_in_arch_list"],
        ]
    )
    return _jsonable(info)


def _assert_pro6000() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    name = str(torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    info = {"gpu": name, "capability": [int(cap[0]), int(cap[1])], "torch": str(torch.__version__)}
    if cap[0] != 12:
        raise RuntimeError(f"expected Blackwell sm_12x, got {cap} on {name}")
    return info


@app.cls(
    image=pixal_pro6000_image,
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
class Pixal3DPro6000:
    @modal.enter()
    def enter(self) -> None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", CUDA_ARCH)
        os.environ.setdefault("NATTEN_CUDA_ARCH", NATTEN_ARCH)
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        _ensure_cache_env()

    @modal.method()
    def probe(self) -> dict:
        info = _probe_info()
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return info

    @modal.method()
    def build_sm120(self, only: str = "") -> dict:
        out = _wheel_cache_dir()
        out.mkdir(parents=True, exist_ok=True)
        env = _build_env()

        steps: list[tuple[str, list[str]]] = [
            ("nvdiffrast", ["/opt/src/nvdiffrast"]),
            ("nvdiffrec_render", ["/opt/src/nvdiffrec"]),
            ("flex_gemm", ["/opt/src/FlexGEMM"]),
            ("cumesh", ["/opt/src/CuMesh"]),
            ("o_voxel", ["/opt/src/TRELLIS.2/o-voxel"]),
            ("drtk", ["git+https://github.com/facebookresearch/drtk.git"]),
            ("natten", ["natten==0.21.0"]),
        ]
        if only:
            steps = [s for s in steps if s[0] == only]
            if not steps:
                return _jsonable({"ok": False, "error": f"unknown package: {only}"})

        log: list[dict] = []
        for name, args in steps:
            existing = _existing_wheels(out, name)
            if existing:
                log.append({"pkg": name, "skipped": True, "files": [p.name for p in existing]})
                continue
            before = {p.name for p in out.glob("*.whl")}
            rc = _pip_wheel(args, out, env)
            after = sorted(p.name for p in out.glob("*.whl") if p.name not in before)
            entry = {"pkg": name, "returncode": int(rc), "new_files": after}
            log.append(entry)
            wheels_vol.commit()
            if rc != 0:
                if name == "drtk":
                    print("[build] drtk failed (optional), continue", flush=True)
                    continue
                return _jsonable(
                    {
                        "ok": False,
                        "failed": name,
                        "log": log,
                        "files": sorted(p.name for p in out.glob("*.whl")),
                    }
                )
        files = sorted(p.name for p in out.glob("*.whl"))
        wheels_vol.commit()
        return _jsonable({"ok": True, "wheel_dir": str(out), "files": files, "log": log})

    @modal.method()
    def install_wheels_and_verify(self) -> dict:
        installed = _install_sm120_wheels()
        verify = Path("/opt/pixal3d_pro6000_scripts/verify_sm120.py")
        v = subprocess.run(
            [sys.executable, str(verify), "--expect-gpu"],
            capture_output=True,
            text=True,
        )
        return _jsonable(
            {
                "ok": v.returncode == 0,
                "returncode": int(v.returncode),
                "stdout": v.stdout,
                "stderr": v.stderr,
                "wheels_installed": installed,
            }
        )

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
                aux.append({"repo": repo, "ok": True, "path": str(p)})
            except Exception as e:
                aux.append({"repo": repo, "ok": False, "error": repr(e)})
        weights_vol.commit()
        return _jsonable({"ok": True, "main_ready": _model_ready(), "aux": aux})

    @modal.method()
    def image_to_3d(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        output_name: str = "smoke_pro6000",
        seed: int = 42,
        low_vram: bool = True,
        resolution: int = 1024,
        fov: float = -1.0,
    ) -> dict:
        import torch

        gpu_info = _assert_pro6000()
        installed = _install_sm120_wheels()
        print("[wheels]", installed, flush=True)

        try:
            from natten.functional import na2d

            q = torch.randn(1, 4, 16, 16, 32, device="cuda", dtype=torch.float16)
            with torch.no_grad():
                _ = na2d(q, q, q, kernel_size=3, dilation=1)
            torch.cuda.synchronize()
            print("[natten] smoke OK", flush=True)
        except Exception as e:
            raise RuntimeError(f"natten smoke failed on PRO 6000: {e!r}") from e

        _ensure_cache_env()
        if not _model_ready():
            print("[i2v] weights missing → download", flush=True)
            self.download_weights.local(force=False)

        model_path = str(_model_local_path())
        _patch_pipeline_rembg(model_path)
        safe = _safe_name(output_name)
        work = Path("/tmp/pixal3d_pro6000_work")
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
            "seed": int(seed),
            "low_vram": bool(low_vram),
            "resolution": int(resolution if resolution > 0 else 1024),
            "fov": float(fov),
            "model_path": model_path,
            "gpu_request": DEFAULT_GPU,
            "gpu_actual": gpu_info["gpu"],
            "sm": "sm_120",
            "capability": gpu_info["capability"],
            "vram_total_gb": round(int(props.total_memory) / 1024**3, 1),
            "price_per_sec_usd": PRO6000_PRICE_PER_SEC,
            "upstream": "TencentARC/Pixal3D",
            "stack": "005-v3 Plan A* sm_120 torch2.11+cu128",
            "wheels": installed,
        }
        print("[i2v] config", json.dumps(meta, ensure_ascii=False, indent=2), flush=True)

        sys.path.insert(0, str(CODE_DIR))
        os.chdir(CODE_DIR)
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ.setdefault(
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(CODE_DIR / "autotune_cache.json")
        )

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
                "est_cost_usd_total": round(total_s * PRO6000_PRICE_PER_SEC, 4),
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
            "sm": "sm_120",
            "seconds_total": total_s,
            "peak_vram_gb": vram.get("peak_mem_used_gb"),
            "est_cost_usd_total": meta["est_cost_usd_total"],
            "bytes": published["bytes"],
            "volume": VOLUME_OUTPUTS,
            "volume_file": f"meshes/{safe}.glb",
            "cli_get": published["cli_get"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return _jsonable(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="005-v3 Pixal3D RTX PRO 6000 / sm_120")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印固定栈；纯本地")

    probe = sub.add_parser("probe", help="远程检查 PRO 6000 / torch / sm_120")
    probe.add_argument("--dry-run", action="store_true")

    build = sub.add_parser("build-sm120", help="源码编译并缓存 sm_120 wheels")
    build.add_argument("--i-know-this-costs-money", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--only", default="", help="只构建一个 package；由远程 builder 校验")

    verify = sub.add_parser("verify", help="安装缓存 wheels 并验证 sm_120")
    verify.add_argument("--i-know-this-costs-money", action="store_true")
    verify.add_argument("--dry-run", action="store_true")

    download = sub.add_parser("download", help="下载 Pixal3D 权重")
    download.add_argument("--i-know-this-costs-money", action="store_true")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--force", action="store_true")

    smoke = sub.add_parser("smoke", help="官方样例图 end-to-end -> GLB")
    smoke.add_argument("--i-know-this-costs-money", action="store_true")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--output-name", default="smoke_pro6000")
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
    i2v.add_argument("--output-name", default="demo_pro6000")
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
        "experiment": "005-v3-pixal3d-pro6000",
        "app": APP_NAME,
        "plan": "A*",
        "gpu": DEFAULT_GPU,
        "cuda_arch": CUDA_ARCH,
        "natten_arch": NATTEN_ARCH,
        "torch": "2.11.0+cu128",
        "torch_index": TORCH_INDEX,
        "model": HF_MODEL_REPO,
        "wheels_volume": VOLUME_WHEELS,
        "weights_volume": VOLUME_WEIGHTS,
        "outputs_volume": VOLUME_OUTPUTS,
        "sample_image_url": SAMPLE_IMAGE_URL,
    }


def command_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan: dict[str, Any] = {"action": args.command, "gpu": DEFAULT_GPU}
    if args.command == "build-sm120":
        plan["only"] = args.only
    elif args.command == "download":
        plan["force"] = args.force
    elif args.command in {"smoke", "i2v"}:
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

    worker = Pixal3DPro6000()
    if args.command == "probe":
        if args.dry_run:
            print(json.dumps(command_plan(args), ensure_ascii=False, indent=2))
            return
        print(worker.probe.remote())
        return

    plan = command_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    require_cost_ack(args)

    if args.command == "build-sm120":
        print(worker.build_sm120.remote(only=args.only))
        return
    if args.command == "verify":
        print(worker.install_wheels_and_verify.remote())
        return
    if args.command == "download":
        print(worker.download_weights.remote(force=args.force))
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
