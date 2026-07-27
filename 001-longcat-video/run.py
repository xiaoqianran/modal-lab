#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""001-longcat-video 本地 CLI —— 调度 Modal 任务 / 拉输出。

Usage:
  python run.py status
  python run.py setup          # 若缺上游代码则浅克隆
  python run.py download       # 下载 HF 权重到 Modal Volume（~83GB）
  python run.py smoke          # GPU + 依赖 + 权重冒烟
  python run.py t2v            # Text-to-Video 官方 demo
  python run.py i2v
  python run.py continuation
  python run.py long
  python run.py interactive
  python run.py t2v --two-gpu  # 2×RTX-PRO-6000
  python run.py pull-outputs   # 把 Volume 输出拉到 ./outputs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
UPSTREAM = EXP_DIR / "LongCat-Video"
UPSTREAM_URL = "https://github.com/meituan-longcat/LongCat-Video.git"
MODAL_APP = EXP_DIR / "modal_app.py"
LOCAL_OUTPUTS = EXP_DIR / "outputs"

DEMOS = ("t2v", "i2v", "continuation", "long", "interactive")


def _which_modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit(
            "未找到 modal CLI。请: pip install modal && modal token new"
        )
    return m


def _run(cmd: list[str], **kwargs) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kwargs)


def cmd_setup(_: argparse.Namespace) -> int:
    if UPSTREAM.is_dir() and (UPSTREAM / "longcat_video").is_dir():
        print(f"[setup] 上游已存在: {UPSTREAM}")
        return 0
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            "main",
            UPSTREAM_URL,
            str(UPSTREAM),
        ]
    )


def cmd_status(_: argparse.Namespace) -> int:
    print("experiment: 001-longcat-video")
    print(f"  exp_dir:   {EXP_DIR}")
    print(f"  upstream:  {UPSTREAM}  exists={UPSTREAM.is_dir()}")
    print(f"  modal_app: {MODAL_APP}  exists={MODAL_APP.is_file()}")
    print(f"  outputs:   {LOCAL_OUTPUTS}")
    print()
    print("上游: https://github.com/meituan-longcat/LongCat-Video")
    print("权重: https://huggingface.co/meituan-longcat/LongCat-Video  (~83GB)")
    print("Volume: modal-lab-longcat-weights / modal-lab-longcat-outputs")
    print("默认 GPU: RTX-PRO-6000 (96GB VRAM)")
    print()
    print("常用:")
    print("  python run.py setup")
    print("  python run.py download")
    print("  python run.py smoke")
    print("  python run.py t2v")
    print("  python run.py pull-outputs")
    return 0


def cmd_modal_action(action: str, ns: argparse.Namespace) -> int:
    modal = _which_modal()
    if not UPSTREAM.is_dir():
        print("[warn] 上游代码缺失，先执行 setup …")
        rc = cmd_setup(ns)
        if rc != 0:
            return rc

    cmd = [
        modal,
        "run",
        str(MODAL_APP),
        "--action",
        action,
    ]
    if action == "download" and getattr(ns, "force", False):
        cmd.append("--force-download")
    if action == "demo":
        cmd.extend(["--demo", ns.demo])
        if ns.two_gpu:
            cmd.append("--two-gpu")
        if ns.no_compile:
            cmd.append("--no-enable-compile")
    return _run(cmd, cwd=str(EXP_DIR))


def cmd_demo(ns: argparse.Namespace) -> int:
    ns.demo = ns.command  # t2v / i2v / …
    return cmd_modal_action("demo", ns)


def cmd_pull_outputs(ns: argparse.Namespace) -> int:
    """从 Modal Volume 拉输出到本地 ./outputs。"""
    modal = _which_modal()
    LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    # modal volume get <name> <remote_path> <local_path>
    remote = ns.remote_path or "/"
    # CLI: modal volume get VOLUME_NAME REMOTE_PATH [LOCAL_PATH]
    return _run(
        [
            modal,
            "volume",
            "get",
            "modal-lab-longcat-outputs",
            remote,
            str(LOCAL_OUTPUTS),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="001-longcat-video — LongCat-Video on Modal",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印实验信息")
    sub.add_parser("setup", help="浅克隆上游仓库（若缺失）")

    dl = sub.add_parser("download", help="下载权重到 Modal Volume")
    dl.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载",
    )

    sub.add_parser("smoke", help="GPU / 依赖 / 权重冒烟")

    for name in DEMOS:
        d = sub.add_parser(name, help=f"跑官方 demo: {name}")
        d.add_argument(
            "--two-gpu",
            action="store_true",
            help="使用 2×RTX-PRO-6000 + context_parallel=2",
        )
        d.add_argument(
            "--no-compile",
            action="store_true",
            help="关闭 torch.compile",
        )

    pull = sub.add_parser("pull-outputs", help="拉取 Volume 输出到 ./outputs")
    pull.add_argument(
        "--remote-path",
        default="/",
        help="Volume 内路径，默认 /",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.command == "status":
        return cmd_status(ns)
    if ns.command == "setup":
        return cmd_setup(ns)
    if ns.command == "download":
        return cmd_modal_action("download", ns)
    if ns.command == "smoke":
        return cmd_modal_action("smoke", ns)
    if ns.command in DEMOS:
        return cmd_demo(ns)
    if ns.command == "pull-outputs":
        return cmd_pull_outputs(ns)

    parser.error(f"unknown command {ns.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
