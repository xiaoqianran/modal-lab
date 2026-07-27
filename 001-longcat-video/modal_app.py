# -*- coding: utf-8 -*-
"""
001-longcat-video — Modal 云端执行

- 基础设施: 资源档位 + Function.with_options 覆盖 gpu/cpu/memory/timeout
- 推理: torchrun + 官方 run_demo_*.py；script_argv 原样透传
- 权重 Volume: /weights ；输出 Volume: /outputs ；代码 add_local_dir

本地请用 run.py（配置合并与校验）；也可直接:
  modal run modal_app.py --action smoke
  modal run modal_app.py --action demo --demo t2v --script-argv '["--checkpoint_dir=/weights/LongCat-Video","--context_parallel_size=1","--enable_compile"]'
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import modal

# ---------- 与 lib/config 对齐的常量（容器内不依赖本地 lib 包名时重复一份最小集）----------
APP_NAME = "modal-lab-longcat-video"
DEFAULT_GPU = "RTX-PRO-6000"
HF_REPO = "meituan-longcat/LongCat-Video"
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
INPUTS_MOUNT = "/inputs"
CODE_ROOT = "/root/LongCat-Video"
CHECKPOINT_DIR = f"{WEIGHTS_MOUNT}/LongCat-Video"

DOWNLOAD_TIMEOUT = 6 * 60 * 60
INFER_TIMEOUT = 2 * 60 * 60
SMOKE_TIMEOUT = 20 * 60

EXP_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = EXP_DIR / "LongCat-Video"
INPUTS_DIR = EXP_DIR / "inputs"

# 资源档位（名称与 lib/config.RESOURCE_PROFILES 一致）
RESOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "pro6000-1": {
        "gpu": "RTX-PRO-6000",
        "cpu": 4.0,
        "memory_mb": 32768,
        "timeout_s": INFER_TIMEOUT,
        "nproc": 1,
    },
    "pro6000-2": {
        "gpu": "RTX-PRO-6000:2",
        "cpu": 8.0,
        "memory_mb": 65536,
        "timeout_s": INFER_TIMEOUT,
        "nproc": 2,
    },
    "a100-80-1": {
        "gpu": "A100-80GB",
        "cpu": 4.0,
        "memory_mb": 32768,
        "timeout_s": INFER_TIMEOUT,
        "nproc": 1,
    },
    "a100-80-2": {
        "gpu": "A100-80GB:2",
        "cpu": 8.0,
        "memory_mb": 65536,
        "timeout_s": INFER_TIMEOUT,
        "nproc": 2,
    },
    "h100-1": {
        "gpu": "H100",
        "cpu": 4.0,
        "memory_mb": 32768,
        "timeout_s": INFER_TIMEOUT,
        "nproc": 1,
    },
    "pro6000-long": {
        "gpu": "RTX-PRO-6000",
        "cpu": 8.0,
        "memory_mb": 65536,
        "timeout_s": 8 * 60 * 60,
        "nproc": 1,
    },
}

DEMO_SCRIPTS = {
    "t2v": "run_demo_text_to_video.py",
    "i2v": "run_demo_image_to_video.py",
    "continuation": "run_demo_video_continuation.py",
    "long": "run_demo_long_video.py",
    "interactive": "run_demo_interactive_video.py",
    "storyboard": "run_storyboard_longcat.py",
}

weights_vol = modal.Volume.from_name("modal-lab-longcat-weights", create_if_missing=True)
outputs_vol = modal.Volume.from_name("modal-lab-longcat-outputs", create_if_missing=True)

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
    "huggingface_hub[cli,hf_transfer]==0.36.2",
    "safetensors",
    "sentencepiece",
    "accelerate",
    "Pillow",
    "pyyaml>=6.0",
]

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
    .run_commands(
        "MAX_JOBS=4 pip install flash_attn==2.7.4.post1 --no-build-isolation"
    )
    .pip_install(*_BASE_PIP)
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": CODE_ROOT,
        }
    )
    .add_local_dir(
        str(UPSTREAM_DIR),
        remote_path=CODE_ROOT,
        ignore=["**/.git/**", "**/weights/**", "**/__pycache__/**"],
    )
)

# 可选：本地 inputs/ → 容器 /inputs（用户素材；官方路径仍在 CODE_ROOT/assets）
if INPUTS_DIR.is_dir():
    image = image.add_local_dir(str(INPUTS_DIR), remote_path=INPUTS_MOUNT)

app = modal.App(APP_NAME)

_INFER_VOLUMES = {
    WEIGHTS_MOUNT: weights_vol,
    OUTPUTS_MOUNT: outputs_vol,
}


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


def _print_summary(payload: dict[str, Any]) -> None:
    print("======== [modal] 生效配置摘要 ========", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    print("=====================================", flush=True)


@app.function(
    image=image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=16384,
)
def download_weights(repo_id: str = HF_REPO, force: bool = False) -> dict:
    from huggingface_hub import snapshot_download

    dest = Path(CHECKPOINT_DIR)
    if dest.exists() and any(dest.iterdir()) and not force:
        info = _list_dir_sizes(str(dest))
        info["skipped"] = True
        info["hint"] = "已存在权重；force=True 可重下"
        print(info)
        return info

    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[download] {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        token=token,
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
    import torch

    out: dict = {
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "vram_gb": None,
        "sm": None,
        "arch_list": list(torch.cuda.get_arch_list())
        if hasattr(torch.cuda, "get_arch_list")
        else [],
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


def _run_demo_impl(
    demo: str,
    script_argv: list[str],
    nproc: int,
    output_subdir: str | None,
) -> dict:
    script = DEMO_SCRIPTS.get(demo)
    if not script:
        raise ValueError(
            f"unknown demo={demo!r}, choose from {list(DEMO_SCRIPTS)}"
        )

    work = Path(CODE_ROOT)
    if not work.is_dir():
        raise FileNotFoundError(f"源码目录不存在: {work}")

    script_path = work / script
    if not script_path.is_file():
        raise FileNotFoundError(f"官方脚本不存在: {script_path}")

    ckpt_hint = CHECKPOINT_DIR
    for a in script_argv:
        if a.startswith("--checkpoint_dir="):
            ckpt_hint = a.split("=", 1)[1]
            break
    ckpt = Path(ckpt_hint)
    if not ckpt.exists() or not any(ckpt.iterdir()):
        raise FileNotFoundError(
            f"权重未找到: {ckpt}。请先 download。"
        )

    sub = output_subdir or demo
    out_dir = Path(OUTPUTS_MOUNT) / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        script,
        *script_argv,
    ]

    summary = {
        "demo": demo,
        "script": script,
        "nproc": nproc,
        "cwd": str(work),
        "cmd": cmd,
        "output_dir": str(out_dir),
    }
    _print_summary(summary)

    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_ROOT
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    proc = subprocess.run(
        cmd,
        cwd=str(work),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"官方脚本执行失败: torchrun exit={proc.returncode} demo={demo} "
            f"cmd={' '.join(cmd)}"
        )

    moved = []
    # 官方/分镜脚本输出多为 cwd 下 mp4；也收集子目录
    candidates = list(work.glob("*.mp4")) + list(work.glob("output_*/*.mp4"))
    # 去重保序
    seen: set[str] = set()
    unique_files = []
    for mp4 in candidates:
        key = str(mp4.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(mp4)

    for mp4 in unique_files:
        dest = out_dir / mp4.name
        shutil.copy2(str(mp4), str(dest))
        try:
            mp4.unlink()
        except OSError:
            pass
        moved.append(str(dest))
        print(f"[run_demo] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    if not moved:
        print("[run_demo] WARNING: 未发现 cwd 下 *.mp4 输出", flush=True)

    outputs_vol.commit()
    result = {
        "demo": demo,
        "nproc": int(nproc),
        "outputs": moved,
        "output_dir": str(out_dir),
        "script_argv": list(script_argv),
    }
    print(result)
    return result


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes=_INFER_VOLUMES,
    timeout=INFER_TIMEOUT,
    cpu=4.0,
    memory=32768,
)
def run_demo(
    demo: str = "t2v",
    script_argv: list[str] | None = None,
    nproc: int = 1,
    output_subdir: str | None = None,
) -> dict:
    """推理入口。资源请在调用侧 with_options 覆盖；此处为默认单卡池。"""
    return _run_demo_impl(
        demo=demo,
        script_argv=list(script_argv or []),
        nproc=max(1, int(nproc)),
        output_subdir=output_subdir,
    )


# 兼容旧调用名
@app.function(
    image=image,
    gpu=f"{DEFAULT_GPU}:2",
    volumes=_INFER_VOLUMES,
    timeout=INFER_TIMEOUT,
    cpu=8.0,
    memory=65536,
)
def run_demo_2gpu(
    demo: str = "t2v",
    script_argv: list[str] | None = None,
    nproc: int = 2,
    output_subdir: str | None = None,
    # 兼容旧签名
    context_parallel_size: int = 2,
    enable_compile: bool = True,
    extra_args: list[str] | None = None,
) -> dict:
    argv = list(script_argv or [])
    if not argv:
        # 旧路径：拼官方三参数
        argv = [
            f"--checkpoint_dir={CHECKPOINT_DIR}",
            f"--context_parallel_size={context_parallel_size}",
        ]
        if enable_compile:
            argv.append("--enable_compile")
        if extra_args:
            argv.extend(extra_args)
    return _run_demo_impl(
        demo=demo,
        script_argv=argv,
        nproc=max(1, int(nproc)),
        output_subdir=output_subdir,
    )


def _resolve_profile(
    profile: str,
    gpu: str | None,
    nproc: int | None,
    cpu: float | None,
    memory_mb: int | None,
    timeout_s: int | None,
) -> dict[str, Any]:
    if profile not in RESOURCE_PROFILES:
        raise ValueError(
            f"unknown profile={profile!r}; choose from {list(RESOURCE_PROFILES)}"
        )
    base = dict(RESOURCE_PROFILES[profile])
    g = gpu or base["gpu"]
    n = int(nproc if nproc is not None else base["nproc"])
    c = float(cpu if cpu is not None else base["cpu"])
    m = int(memory_mb if memory_mb is not None else base["memory_mb"])
    t = int(timeout_s if timeout_s is not None else base["timeout_s"])
    if ":" not in str(g) and n > 1:
        g = f"{g}:{n}"
    return {"gpu": g, "nproc": n, "cpu": c, "memory_mb": m, "timeout_s": t}


def _default_script_argv(
    enable_compile: bool,
    context_parallel_size: int,
    checkpoint_dir: str,
) -> list[str]:
    argv = [
        f"--checkpoint_dir={checkpoint_dir}",
        f"--context_parallel_size={context_parallel_size}",
    ]
    if enable_compile:
        argv.append("--enable_compile")
    return argv


@app.local_entrypoint()
def main(
    action: str = "smoke",
    demo: str = "t2v",
    force_download: bool = False,
    two_gpu: bool = False,
    enable_compile: bool = True,
    profile: str = "pro6000-1",
    gpu: str = "",
    nproc: int = 0,
    cpu: float = 0.0,
    memory_mb: int = 0,
    timeout_s: int = 0,
    context_parallel_size: int = 0,
    checkpoint_dir: str = "",
    output_subdir: str = "",
    # JSON 数组字符串，完整透传 argv；非空时优先于零散 enable_compile 等
    script_argv_json: str = "",
    hf_repo: str = "",
):
    """
    modal run modal_app.py --action smoke|download|demo
    modal run modal_app.py --action demo --demo t2v --two-gpu
    modal run modal_app.py --action demo --demo t2v \\
      --script-argv-json '["--checkpoint_dir=/weights/LongCat-Video","--context_parallel_size=1","--enable_compile"]'
    """
    import json as _json

    def _show(obj) -> None:
        print(_json.dumps(obj, ensure_ascii=False, indent=2, default=str))

    action = action.lower().strip()
    if action in ("smoke", "status"):
        _show(smoke.remote())
        return
    if action in ("download", "dl"):
        repo = hf_repo or HF_REPO
        _show(download_weights.remote(repo_id=repo, force=force_download))
        return

    if action in ("demo", "run") or action in DEMO_SCRIPTS:
        if action in DEMO_SCRIPTS:
            demo = action

        prof = "pro6000-2" if two_gpu else (profile or "pro6000-1")
        res = _resolve_profile(
            profile=prof,
            gpu=gpu or None,
            nproc=nproc or None,
            cpu=cpu or None,
            memory_mb=memory_mb or None,
            timeout_s=timeout_s or None,
        )

        if script_argv_json.strip():
            script_argv = _json.loads(script_argv_json)
            if not isinstance(script_argv, list):
                raise SystemExit("script_argv_json 必须是 JSON 数组")
            script_argv = [str(x) for x in script_argv]
        else:
            cp = context_parallel_size or res["nproc"]
            ckpt = checkpoint_dir or CHECKPOINT_DIR
            script_argv = _default_script_argv(
                enable_compile=enable_compile,
                context_parallel_size=int(cp),
                checkpoint_dir=ckpt,
            )

        out_sub = output_subdir or demo
        payload = {
            "action": "demo",
            "demo": demo,
            "profile": prof,
            "resources": res,
            "script_argv": script_argv,
            "output_subdir": out_sub,
        }
        _print_summary(payload)

        # with_options：动态资源；volumes 必须整表传入（替换语义）
        fn = run_demo.with_options(
            gpu=res["gpu"],
            cpu=res["cpu"],
            memory=res["memory_mb"],
            timeout=res["timeout_s"],
            volumes=_INFER_VOLUMES,
        )
        _show(
            fn.remote(
                demo=demo,
                script_argv=script_argv,
                nproc=res["nproc"],
                output_subdir=out_sub,
            )
        )
        return

    raise SystemExit(
        f"unknown action={action!r}; use smoke|download|demo "
        f"(demo in {list(DEMO_SCRIPTS)})"
    )
