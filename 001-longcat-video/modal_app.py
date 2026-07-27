# -*- coding: utf-8 -*-
"""
001-longcat-video — Modal 上复现美团 LongCat-Video

心智模型:
  1) 上游代码: 实验目录内 LongCat-Video/（官方 repo 浅克隆）
  2) 权重 ~83GB: 挂到 Volume `/weights/LongCat-Video`，只下一次
  3) 推理: 容器内 torchrun 调官方 run_demo_*.py
  4) 输出: Volume `/outputs`，本机用 modal volume get 拉回

GPU（费用）:
  - 默认 RTX-PRO-6000（96GB）：显存余量够装 13.6B + 激活；比 A100-80GB 更宽裕
  - 多卡: run_demo_2gpu → "RTX-PRO-6000:2" + context_parallel=2
  - 备选: "A100-80GB" / "H100" / "H200"；L40S 24GB 大概率 OOM

RTX PRO 6000 = Blackwell sm_120，必须用 **PyTorch cu128**（2.7+），cu124 无法跑 kernel。
首次镜像构建会装 flash-attn，可能 15～40 分钟；之后有缓存。

命令（在 001-longcat-video 目录）:
  modal run modal_app.py::download_weights
  modal run modal_app.py::smoke
  modal run modal_app.py::run_demo --demo t2v
  # 或通过 run.py 统一入口
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal

# ---------- 可调配置 ----------
APP_NAME = "modal-lab-longcat-video"
# 默认 RTX PRO 6000 Blackwell（Modal 字符串 RTX-PRO-6000，96GB VRAM）
# 权重磁盘约 83GB；bf16 常驻远小于此，但 T2V 激活/中间帧吃显存，96GB 余量更稳
# 备选: "A100-80GB" / "H100" / "H200"；多卡在装饰器里拼 ":N"
DEFAULT_GPU = "RTX-PRO-6000"
HF_REPO = "meituan-longcat/LongCat-Video"
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
CODE_ROOT = "/root/LongCat-Video"
CHECKPOINT_DIR = f"{WEIGHTS_MOUNT}/LongCat-Video"

# 超时：下载 / 推理都很长
DOWNLOAD_TIMEOUT = 6 * 60 * 60  # 6h
INFER_TIMEOUT = 2 * 60 * 60  # 2h
SMOKE_TIMEOUT = 20 * 60

EXP_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = EXP_DIR / "LongCat-Video"

weights_vol = modal.Volume.from_name("modal-lab-longcat-weights", create_if_missing=True)
outputs_vol = modal.Volume.from_name("modal-lab-longcat-outputs", create_if_missing=True)

# 基础依赖（不含 flash-attn；flash-attn 单独装）
_BASE_PIP = [
    "numpy==1.26.4",
    "transformers==4.41.0",
    "loguru==0.7.2",
    "diffusers==0.35.1",
    "einops==0.8.0",
    "ftfy==6.2.0",
    "psutil==6.0.0",
    "av==12.0.0",
    "opencv-python-headless==4.9.0.80",
    "streamlit==1.50.0",
    "pyarrow==20.0.0",
    "imageio==2.37.0",
    "imageio-ffmpeg==0.6.0",
    "huggingface_hub[cli,hf_transfer]>=0.26.0",
    "safetensors",
    "sentencepiece",
    "accelerate",
    "Pillow",
]

# Blackwell (sm_120) 需要 cu128 轮子；官方 LongCat 钉的是 torch 2.6+cu124，PRO 6000 上不可用
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "wget",
        "ninja-build",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("ninja", "packaging", "wheel", "setuptools")
    # flash-attn 需 CUDA 编译；MAX_JOBS 限制并行以免 OOM
    .run_commands(
        "MAX_JOBS=4 pip install flash_attn==2.7.4.post1 --no-build-isolation"
    )
    .pip_install(*_BASE_PIP)
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            # 官方 demo 期望在仓库根目录 import longcat_video
            "PYTHONPATH": CODE_ROOT,
        }
    )
    .add_local_dir(
        str(UPSTREAM_DIR),
        remote_path=CODE_ROOT,
        # 不把 .git 打进镜像
        ignore=["**/.git/**", "**/*.mp4", "**/weights/**", "**/__pycache__/**"],
    )
)

app = modal.App(APP_NAME)


def _gpu_spec(gpu: str, count: int) -> str:
    count = max(1, int(count))
    if count == 1:
        return gpu
    # Modal 多卡: "A100-80GB:2"
    if ":" in gpu:
        return gpu
    return f"{gpu}:{count}"


def _list_dir_sizes(root: str) -> dict:
    p = Path(root)
    if not p.exists():
        return {"exists": False, "path": root}
    total = 0
    files = 0
    for f in p.rglob("*"):
        if f.is_file():
            files += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return {
        "exists": True,
        "path": root,
        "files": files,
        "size_gb": round(total / 1e9, 2),
    }


@app.function(
    image=image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=16384,
)
def download_weights(repo_id: str = HF_REPO, force: bool = False) -> dict:
    """把 HF 权重下到 Volume（约 83GB，可断点续传）。"""
    from huggingface_hub import snapshot_download

    dest = Path(CHECKPOINT_DIR)
    if dest.exists() and any(dest.iterdir()) and not force:
        info = _list_dir_sizes(str(dest))
        info["skipped"] = True
        info["hint"] = "已存在权重；加 force=True 可重下"
        print(info)
        return info

    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[download] {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    weights_vol.commit()
    info = _list_dir_sizes(str(dest))
    info["skipped"] = False
    print(info)
    return info


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=SMOKE_TIMEOUT,
)
def smoke() -> dict:
    """GPU + 依赖 + 权重目录冒烟，不跑完整生成。

    返回值只用纯 Python 类型，避免本机反序列化依赖 torch。
    """
    import torch

    out: dict = {
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "vram_gb": None,
        "sm": None,
        "arch_list": list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else [],
        "torch": str(torch.__version__),
        "code_root": CODE_ROOT,
        "code_exists": bool(Path(CODE_ROOT).is_dir()),
        "weights": _list_dir_sizes(CHECKPOINT_DIR),
        "cuda_matmul_ok": False,
    }
    if out["cuda"]:
        out["gpu_name"] = str(torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        out["vram_gb"] = float(round(int(props.total_memory) / 1024**3, 1))
        major, minor = torch.cuda.get_device_capability(0)
        out["sm"] = f"sm_{major}{minor}"
        # 真跑一个 kernel，验证 sm_120 可用
        try:
            a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            c = (a @ b).sum().item()
            out["cuda_matmul_ok"] = True
            out["cuda_matmul_sum"] = float(c)
        except Exception as e:  # noqa: BLE001
            out["cuda_matmul_ok"] = False
            out["cuda_matmul_error"] = repr(e)

    try:
        import longcat_video  # noqa: F401

        out["import_longcat_video"] = True
    except Exception as e:  # noqa: BLE001
        out["import_longcat_video"] = False
        out["import_error"] = repr(e)

    try:
        import flash_attn  # noqa: F401

        out["flash_attn"] = str(getattr(flash_attn, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        out["flash_attn"] = False
        out["flash_attn_error"] = repr(e)

    print(out)
    return out


# 官方 demo 脚本映射
DEMO_SCRIPTS = {
    "t2v": "run_demo_text_to_video.py",
    "i2v": "run_demo_image_to_video.py",
    "continuation": "run_demo_video_continuation.py",
    "long": "run_demo_long_video.py",
    "interactive": "run_demo_interactive_video.py",
}


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={
        WEIGHTS_MOUNT: weights_vol,
        OUTPUTS_MOUNT: outputs_vol,
    },
    timeout=INFER_TIMEOUT,
)
def run_demo(
    demo: str = "t2v",
    context_parallel_size: int = 1,
    enable_compile: bool = True,
    extra_args: list[str] | None = None,
) -> dict:
    """
    在容器内用 torchrun 跑官方 demo（单卡，gpu 由 DEFAULT_GPU 固定）。

    多卡请用 run_demo_2gpu；改卡型请改本文件 DEFAULT_GPU 后重新 modal run。
    """
    return _run_demo_impl(
        demo=demo,
        context_parallel_size=context_parallel_size,
        enable_compile=enable_compile,
        nproc=1,
        extra_args=extra_args or [],
    )


def _run_demo_impl(
    demo: str,
    context_parallel_size: int,
    enable_compile: bool,
    nproc: int,
    extra_args: list[str],
) -> dict:
    script = DEMO_SCRIPTS.get(demo)
    if not script:
        raise ValueError(f"unknown demo={demo!r}, choose from {list(DEMO_SCRIPTS)}")

    ckpt = Path(CHECKPOINT_DIR)
    if not ckpt.exists() or not any(ckpt.iterdir()):
        raise FileNotFoundError(
            f"权重未找到: {ckpt}。请先: python run.py download 或 "
            "modal run modal_app.py::download_weights"
        )

    work = Path(CODE_ROOT)
    out_dir = Path(OUTPUTS_MOUNT) / demo
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        script,
        f"--checkpoint_dir={CHECKPOINT_DIR}",
        f"--context_parallel_size={context_parallel_size}",
    ]
    if enable_compile:
        cmd.append("--enable_compile")
    cmd.extend(extra_args)

    print(f"[run_demo] cwd={work} cmd={' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_ROOT
    # 单机 torchrun
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    proc = subprocess.run(
        cmd,
        cwd=str(work),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"torchrun failed with code={proc.returncode}")

    # 官方脚本把 mp4 写在 cwd；拷到 outputs volume
    # 注意: pathlib.replace/rename 不能跨设备（Volume 与容器本地是不同挂载）
    moved = []
    for mp4 in work.glob("*.mp4"):
        dest = out_dir / mp4.name
        shutil.copy2(str(mp4), str(dest))
        try:
            mp4.unlink()
        except OSError:
            pass
        moved.append(str(dest))
        print(f"[run_demo] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    outputs_vol.commit()
    result = {
        "demo": demo,
        "nproc": int(nproc),
        "context_parallel_size": int(context_parallel_size),
        "outputs": moved,
        "output_dir": str(out_dir),
    }
    print(result)
    return result


# 多卡变体：2×RTX-PRO-6000（或改 DEFAULT_GPU 后同步生效）
@app.function(
    image=image,
    gpu=f"{DEFAULT_GPU}:2",
    volumes={
        WEIGHTS_MOUNT: weights_vol,
        OUTPUTS_MOUNT: outputs_vol,
    },
    timeout=INFER_TIMEOUT,
)
def run_demo_2gpu(
    demo: str = "t2v",
    context_parallel_size: int = 2,
    enable_compile: bool = True,
    extra_args: list[str] | None = None,
) -> dict:
    """2 卡 context parallel 推理（默认 2×RTX-PRO-6000）。"""
    return _run_demo_impl(
        demo=demo,
        context_parallel_size=context_parallel_size,
        enable_compile=enable_compile,
        nproc=2,
        extra_args=extra_args or [],
    )


@app.local_entrypoint()
def main(
    action: str = "smoke",
    demo: str = "t2v",
    force_download: bool = False,
    two_gpu: bool = False,
    enable_compile: bool = True,
):
    """
    modal run modal_app.py --action smoke
    modal run modal_app.py --action download
    modal run modal_app.py --action demo --demo t2v
    modal run modal_app.py --action demo --demo t2v --two-gpu
    """
    import json

    def _show(obj) -> None:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))

    action = action.lower().strip()
    if action in ("smoke", "status"):
        _show(smoke.remote())
    elif action in ("download", "dl"):
        _show(download_weights.remote(force=force_download))
    elif action in ("demo", "run", "t2v", "i2v"):
        # 允许 --action t2v 简写
        if action in DEMO_SCRIPTS:
            demo = action
        if two_gpu:
            _show(
                run_demo_2gpu.remote(
                    demo=demo,
                    enable_compile=enable_compile,
                )
            )
        else:
            _show(
                run_demo.remote(
                    demo=demo,
                    enable_compile=enable_compile,
                )
            )
    else:
        raise SystemExit(
            f"unknown action={action!r}; use smoke|download|demo "
            f"(demo in {list(DEMO_SCRIPTS)})"
        )
