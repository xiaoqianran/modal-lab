# -*- coding: utf-8 -*-
"""001-longcat-video — LongCat-Video on Modal。

设计约束：
- 本文件是实验唯一入口，也是唯一配置事实源。
- Modal 负责运行时；本实验不再包装 ``modal run``。
- 只建模 modal-lab 自己拥有的资源参数；模型脚本参数原样透传给上游。
- 纯规划函数保持无 I/O，Modal / subprocess / Volume 放在外围。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import modal


# ---------------------------------------------------------------------------
# Experiment facts — 单一事实源
# ---------------------------------------------------------------------------

APP_NAME = "modal-lab-longcat-video"
UPSTREAM_URL = "https://github.com/meituan-longcat/LongCat-Video.git"
HF_REPO = "meituan-longcat/LongCat-Video"

EXP_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = EXP_DIR / "LongCat-Video"
STORYBOARD_SCRIPT = EXP_DIR / "storyboard.py"
STORYBOARDS_DIR = EXP_DIR / "storyboards"
INPUTS_DIR = EXP_DIR / "inputs"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
INPUTS_MOUNT = "/inputs"
CODE_ROOT = "/root/LongCat-Video"
CHECKPOINT_DIR = f"{WEIGHTS_MOUNT}/LongCat-Video"

WEIGHTS_VOLUME = "modal-lab-longcat-weights"
OUTPUTS_VOLUME = "modal-lab-longcat-outputs"

DOWNLOAD_TIMEOUT = 6 * 60 * 60
INFER_TIMEOUT = 2 * 60 * 60
SMOKE_TIMEOUT = 20 * 60
DEFAULT_PROFILE = "pro6000-1"


@dataclass(frozen=True, slots=True)
class Profile:
    """一组必须一起变化的 Modal / torchrun 资源不变量。"""

    gpu: str
    nproc: int
    cpu: float
    memory_mb: int
    timeout_s: int = INFER_TIMEOUT


PROFILES: dict[str, Profile] = {
    "pro6000-1": Profile("RTX-PRO-6000", 1, 4.0, 32768),
    "pro6000-2": Profile("RTX-PRO-6000:2", 2, 8.0, 65536),
    "a100-80-1": Profile("A100-80GB", 1, 4.0, 32768),
    "a100-80-2": Profile("A100-80GB:2", 2, 8.0, 65536),
    "h100-1": Profile("H100", 1, 4.0, 32768),
    "pro6000-long": Profile("RTX-PRO-6000", 1, 8.0, 65536, 8 * 60 * 60),
}

DEMO_SCRIPTS = {
    "t2v": "run_demo_text_to_video.py",
    "i2v": "run_demo_image_to_video.py",
    "continuation": "run_demo_video_continuation.py",
    "long": "run_demo_long_video.py",
    "interactive": "run_demo_interactive_video.py",
    "storyboard": "storyboard.py",
}

DEMO_REQUIRED_ASSETS: dict[str, tuple[str, ...]] = {
    "i2v": ("assets/girl.png",),
    "continuation": ("assets/motorcycle.mp4",),
    "storyboard": ("storyboards/your_name_shinkai.json",),
}

DEMO_DEFAULT_PROFILES = {
    "storyboard": "pro6000-long",
}

# 这三项由实验基础设施拥有，必须与资源 profile 保持一致；禁止在透传区重复定义。
RESERVED_UPSTREAM_ARGS = frozenset(
    {"checkpoint_dir", "context_parallel_size", "enable_compile"}
)


# ---------------------------------------------------------------------------
# Pure planning — 无 Modal RPC / subprocess / 文件写入
# ---------------------------------------------------------------------------


def profile_for(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(PROFILES)
        raise ValueError(f"未知 profile={name!r}；可选: {known}") from None


def script_for(demo: str) -> str:
    try:
        return DEMO_SCRIPTS[demo]
    except KeyError:
        known = ", ".join(DEMO_SCRIPTS)
        raise ValueError(f"未知 demo={demo!r}；可选: {known}") from None


def default_profile_for(demo: str) -> str:
    script_for(demo)  # 同时校验 demo
    return DEMO_DEFAULT_PROFILES.get(demo, DEFAULT_PROFILE)


def _option_name(token: str) -> str | None:
    if not token.startswith("--") or token == "--":
        return None
    return token[2:].split("=", 1)[0].replace("-", "_")


def normalize_upstream_args(args: Sequence[str]) -> list[str]:
    """去掉 argparse.REMAINDER 可能保留的 ``--``，并保护本实验不变量。"""

    out = list(args)
    if out and out[0] == "--":
        out = out[1:]

    duplicated = sorted(
        {
            name
            for token in out
            if (name := _option_name(token)) in RESERVED_UPSTREAM_ARGS
        }
    )
    if duplicated:
        joined = ", ".join(f"--{name}" for name in duplicated)
        raise ValueError(
            f"这些参数由 modal-lab 管理，不能在 -- 后重复传入: {joined}"
        )
    return out


def build_script_argv(
    demo: str,
    *,
    profile: Profile,
    checkpoint_dir: str = CHECKPOINT_DIR,
    enable_compile: bool = True,
    upstream_args: Sequence[str] = (),
) -> list[str]:
    """构造官方/实验 workflow 的 argv；不复制上游 CLI schema。"""

    script_for(demo)
    argv = [
        f"--checkpoint_dir={checkpoint_dir}",
        f"--context_parallel_size={profile.nproc}",
    ]
    if enable_compile:
        argv.append("--enable_compile")
    return [*argv, *normalize_upstream_args(upstream_args)]


def run_summary(
    demo: str,
    *,
    profile_name: str,
    checkpoint_dir: str,
    enable_compile: bool,
    output_subdir: str,
    upstream_args: Sequence[str],
) -> dict:
    """产生 dry-run 与真正执行共用的、可读的最终计划。"""

    profile = profile_for(profile_name)
    script_argv = build_script_argv(
        demo,
        profile=profile,
        checkpoint_dir=checkpoint_dir,
        enable_compile=enable_compile,
        upstream_args=upstream_args,
    )
    return {
        "demo": demo,
        "script": script_for(demo),
        "profile": profile_name,
        "resources": {
            "gpu": profile.gpu,
            "nproc": profile.nproc,
            "cpu": profile.cpu,
            "memory_mb": profile.memory_mb,
            "timeout_s": profile.timeout_s,
        },
        "script_argv": script_argv,
        "output_subdir": output_subdir,
    }


# ---------------------------------------------------------------------------
# Local experiment operations
# ---------------------------------------------------------------------------


def validate_local_demo(demo: str) -> None:
    script = script_for(demo)
    if not (UPSTREAM_DIR / "longcat_video").is_dir():
        raise ValueError(f"上游源码缺失: {UPSTREAM_DIR}；先运行 setup")

    script_path = STORYBOARD_SCRIPT if demo == "storyboard" else UPSTREAM_DIR / script
    if not script_path.is_file():
        raise ValueError(f"执行脚本缺失: {script_path}")

    missing = []
    for rel in DEMO_REQUIRED_ASSETS.get(demo, ()):
        path = EXP_DIR / rel if demo == "storyboard" else UPSTREAM_DIR / rel
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise ValueError("缺少 demo 输入文件: " + ", ".join(missing))


def setup_upstream() -> dict:
    if (UPSTREAM_DIR / "longcat_video").is_dir():
        return {"upstream": str(UPSTREAM_DIR), "cloned": False}

    UPSTREAM_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            "main",
            UPSTREAM_URL,
            str(UPSTREAM_DIR),
        ],
        check=True,
    )
    return {"upstream": str(UPSTREAM_DIR), "cloned": True}


def local_status() -> dict:
    return {
        "experiment": "001-longcat-video",
        "upstream": {
            "path": str(UPSTREAM_DIR),
            "exists": (UPSTREAM_DIR / "longcat_video").is_dir(),
            "url": UPSTREAM_URL,
        },
        "storyboard": {
            "script": str(STORYBOARD_SCRIPT),
            "exists": STORYBOARD_SCRIPT.is_file(),
            "data_dir": str(STORYBOARDS_DIR),
        },
        "profiles": list(PROFILES),
        "demos": list(DEMO_SCRIPTS),
        "volumes": [WEIGHTS_VOLUME, OUTPUTS_VOLUME],
    }


def show(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str), flush=True)


# ---------------------------------------------------------------------------
# Modal infrastructure — 直接使用平台原语，不再二次包装
# ---------------------------------------------------------------------------


weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)

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
    .run_commands("MAX_JOBS=4 pip install flash_attn==2.7.4.post1 --no-build-isolation")
    .pip_install(*_BASE_PIP)
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": CODE_ROOT,
        }
    )
)

# setup/status/download 在源码缺失时仍应可用，因此本地文件按存在性挂载。
if UPSTREAM_DIR.is_dir():
    image = image.add_local_dir(
        str(UPSTREAM_DIR),
        remote_path=CODE_ROOT,
        ignore=["**/.git/**", "**/weights/**", "**/__pycache__/**"],
    )
if STORYBOARD_SCRIPT.is_file():
    image = image.add_local_file(
        str(STORYBOARD_SCRIPT), remote_path=f"{CODE_ROOT}/storyboard.py"
    )
if STORYBOARDS_DIR.is_dir():
    image = image.add_local_dir(
        str(STORYBOARDS_DIR), remote_path=f"{CODE_ROOT}/storyboards"
    )
if INPUTS_DIR.is_dir():
    image = image.add_local_dir(str(INPUTS_DIR), remote_path=INPUTS_MOUNT)

app = modal.App(APP_NAME)
INFER_VOLUMES = {
    WEIGHTS_MOUNT: weights_vol,
    OUTPUTS_MOUNT: outputs_vol,
}


def _dir_info(root: str) -> dict:
    path = Path(root)
    if not path.exists():
        return {"exists": False, "path": root}

    total = 0
    files = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        files += 1
        try:
            total += item.stat().st_size
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
    from huggingface_hub import snapshot_download

    dest = Path(CHECKPOINT_DIR)
    if dest.exists() and any(dest.iterdir()) and not force:
        return {**_dir_info(str(dest)), "skipped": True}

    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[download] {repo_id} -> {dest}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(dest), token=token)
    weights_vol.commit()
    return {**_dir_info(str(dest)), "skipped": False}


@app.function(
    image=image,
    gpu=PROFILES[DEFAULT_PROFILE].gpu,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=SMOKE_TIMEOUT,
)
def smoke() -> dict:
    import torch

    out = {
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "vram_gb": None,
        "sm": None,
        "torch": str(torch.__version__),
        "code_exists": Path(CODE_ROOT).is_dir(),
        "weights": _dir_info(CHECKPOINT_DIR),
        "cuda_matmul_ok": False,
    }
    if out["cuda"]:
        out["gpu_name"] = str(torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        out["vram_gb"] = round(int(props.total_memory) / 1024**3, 1)
        major, minor = torch.cuda.get_device_capability(0)
        out["sm"] = f"sm_{major}{minor}"
        try:
            a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            out["cuda_matmul_sum"] = float((a @ b).sum().item())
            out["cuda_matmul_ok"] = True
        except Exception as exc:  # noqa: BLE001
            out["cuda_matmul_error"] = repr(exc)

    try:
        import longcat_video  # noqa: F401

        out["import_longcat_video"] = True
    except Exception as exc:  # noqa: BLE001
        out["import_longcat_video"] = False
        out["import_error"] = repr(exc)

    try:
        import flash_attn

        out["flash_attn"] = str(getattr(flash_attn, "__version__", "ok"))
    except Exception as exc:  # noqa: BLE001
        out["flash_attn"] = False
        out["flash_attn_error"] = repr(exc)
    return out


@app.function(
    image=image,
    gpu=PROFILES[DEFAULT_PROFILE].gpu,
    volumes=INFER_VOLUMES,
    timeout=INFER_TIMEOUT,
    cpu=PROFILES[DEFAULT_PROFILE].cpu,
    memory=PROFILES[DEFAULT_PROFILE].memory_mb,
)
def run_demo(
    demo: str,
    script_argv: list[str],
    nproc: int,
    output_subdir: str,
) -> dict:
    """云端唯一推理边界：校验 -> torchrun -> 收集输出 -> commit。"""

    script = script_for(demo)
    work = Path(CODE_ROOT)
    script_path = work / script
    if not script_path.is_file():
        raise FileNotFoundError(f"执行脚本不存在: {script_path}")

    checkpoint = CHECKPOINT_DIR
    for arg in script_argv:
        if arg.startswith("--checkpoint_dir="):
            checkpoint = arg.split("=", 1)[1]
            break
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists() or not any(checkpoint_path.iterdir()):
        raise FileNotFoundError(f"权重未找到: {checkpoint}；先运行 download")

    output_dir = Path(OUTPUTS_MOUNT) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    command = ["torchrun", f"--nproc_per_node={nproc}", script, *script_argv]
    show(
        {
            "demo": demo,
            "cwd": str(work),
            "command": command,
            "output_dir": str(output_dir),
        }
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_ROOT
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    proc = subprocess.run(command, cwd=str(work), env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"torchrun exit={proc.returncode}: {' '.join(command)}")

    candidates = [*work.glob("*.mp4"), *work.glob("output_*/*.mp4")]
    seen: set[str] = set()
    saved: list[str] = []
    for video in candidates:
        key = str(video.resolve())
        if key in seen:
            continue
        seen.add(key)
        dest = output_dir / video.name
        shutil.copy2(video, dest)
        try:
            video.unlink()
        except OSError:
            pass
        saved.append(str(dest))
        print(f"[output] {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)

    if not saved:
        print("[output] WARNING: 没有发现生成的 mp4", flush=True)

    outputs_vol.commit()
    return {
        "demo": demo,
        "nproc": nproc,
        "outputs": saved,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# Single CLI boundary
# ---------------------------------------------------------------------------


def _add_demo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=PROFILES, default=None, help="资源档位")
    parser.add_argument("--no-compile", action="store_true", help="关闭 torch.compile")
    parser.add_argument(
        "--checkpoint-dir",
        default=CHECKPOINT_DIR,
        help="容器内权重目录",
    )
    parser.add_argument("--output-subdir", default=None, help="Volume 输出子目录")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印最终计划，不提交远程推理",
    )
    parser.add_argument(
        "upstream_args",
        nargs=argparse.REMAINDER,
        help="`--` 后的参数原样交给 LongCat / storyboard 脚本",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="001-longcat-video — 一个 app.py 的 LongCat-Video Modal 实验",
        epilog=(
            "示例:\n"
            "  modal run app.py status\n"
            "  modal run app.py download\n"
            "  modal run app.py t2v --profile pro6000-2\n"
            "  modal run app.py storyboard --dry-run -- --seed 7"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印本地实验状态")
    sub.add_parser("setup", help="上游不存在时浅克隆 LongCat-Video")

    download = sub.add_parser("download", help="下载权重到 Modal Volume")
    download.add_argument("--force", action="store_true")
    download.add_argument("--hf-repo", default=HF_REPO)

    smoke_parser = sub.add_parser("smoke", help="CUDA / flash-attn / 权重冒烟")
    smoke_parser.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)

    for demo in DEMO_SCRIPTS:
        demo_parser = sub.add_parser(demo, help=f"运行 {demo}")
        _add_demo_args(demo_parser)

    return parser


def parse_cli(argv: Sequence[str]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


@app.local_entrypoint()
def main(*argv: str) -> None:
    args = parse_cli(argv)

    if args.command == "status":
        show(local_status())
        return

    if args.command == "setup":
        show(setup_upstream())
        return

    if args.command == "download":
        show(download_weights.remote(repo_id=args.hf_repo, force=args.force))
        return

    if args.command == "smoke":
        profile = profile_for(args.profile)
        fn = smoke.with_options(
            gpu=profile.gpu,
            cpu=profile.cpu,
            memory=profile.memory_mb,
            timeout=SMOKE_TIMEOUT,
            volumes={WEIGHTS_MOUNT: weights_vol},
        )
        show(fn.remote())
        return

    demo = args.command
    try:
        validate_local_demo(demo)
        profile_name = args.profile or default_profile_for(demo)
        profile = profile_for(profile_name)
        output_subdir = args.output_subdir or (
            "your_name_shinkai" if demo == "storyboard" else demo
        )
        summary = run_summary(
            demo,
            profile_name=profile_name,
            checkpoint_dir=args.checkpoint_dir,
            enable_compile=not args.no_compile,
            output_subdir=output_subdir,
            upstream_args=args.upstream_args,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    show(summary)
    if args.dry_run:
        return

    fn = run_demo.with_options(
        gpu=profile.gpu,
        cpu=profile.cpu,
        memory=profile.memory_mb,
        timeout=profile.timeout_s,
        volumes=INFER_VOLUMES,
    )
    show(
        fn.remote(
            demo=demo,
            script_argv=summary["script_argv"],
            nproc=profile.nproc,
            output_subdir=output_subdir,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
