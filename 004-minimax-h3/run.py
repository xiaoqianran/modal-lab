#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""004-minimax-h3 CLI — 成片写入远程 Modal Volume（不依赖本机目录）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-minimax-h3-outputs"

DEFAULT_PROMPT = (
    "新海诚风格车站偶遇的唯美画面："
    "傍晚蓝色时刻，乡村小站月台上微风轻拂，金色夕阳从云隙洒下，"
    "男女主角在列车进站的光影中不期而遇，樱花瓣缓缓飘落，"
    "镜头缓慢推进，电影感构图，细腻光影，治愈而浪漫。"
    "Audio: soft ambient wind, distant train horn, gentle piano score."
)


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def cmd_status(_: argparse.Namespace) -> int:
    print("成片位置（远程 Volume，不是本地文件夹）:")
    print(f"  volume : {VOL_OUT}")
    print("  path   : videos/<name>.mp4  +  videos/latest.mp4")
    print("  列表页 : https://seachenxyt--modal-lab-minimax-h3-index.modal.run")
    print("  下载   : https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=latest")
    print(f"  CLI    : modal volume ls {VOL_OUT} videos")
    return _run([_modal(), "run", str(MODAL_APP), "--action", "status"])


def cmd_download(ns: argparse.Namespace) -> int:
    cmd = [_modal(), "run", str(MODAL_APP), "--action", "download"]
    if ns.force:
        cmd.append("--force-download")
    return _run(cmd)


def cmd_smoke(ns: argparse.Namespace) -> int:
    return _run(
        [_modal(), "run", str(MODAL_APP), "--action", "smoke", "--gpu", ns.gpu]
    )


def cmd_list_outputs(_: argparse.Namespace) -> int:
    print(f"=== remote volume {VOL_OUT}/videos ===", flush=True)
    _run([_modal(), "volume", "ls", VOL_OUT, "videos"])
    return _run([_modal(), "run", str(MODAL_APP), "--action", "list-outputs"])


def cmd_t2v(ns: argparse.Namespace) -> int:
    prompt = ns.prompt or DEFAULT_PROMPT
    print(
        f"[t2v] GPU={ns.gpu} → 只写入远程 Volume "
        f"{VOL_OUT}/videos/{ns.output_name}.mp4",
        flush=True,
    )
    rc = _run(
        [
            _modal(),
            "run",
            str(MODAL_APP),
            "--action",
            "t2v",
            "--prompt",
            prompt,
            "--width",
            str(ns.width),
            "--height",
            str(ns.height),
            "--seconds",
            str(ns.seconds),
            "--steps",
            str(ns.steps),
            "--seed",
            str(ns.seed),
            "--output-name",
            ns.output_name,
            "--gpu",
            ns.gpu,
        ]
    )
    if rc == 0:
        print()
        print(">>> 请在远程 Volume 查看（不要看本地文件夹）:", flush=True)
        print(f"    modal volume ls {VOL_OUT} videos", flush=True)
        print(
            f"    https://seachenxyt--modal-lab-minimax-h3-download.modal.run"
            f"?name={ns.output_name}",
            flush=True,
        )
        print(
            "    https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=latest",
            flush=True,
        )
        # 立即列出 volume 证明在
        _run([_modal(), "volume", "ls", VOL_OUT, "videos"])
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    sm = sub.add_parser("smoke")
    sm.add_argument("--gpu", default="RTX-PRO-6000")
    sub.add_parser("list-outputs", help="列出远程 Volume 视频")
    t = sub.add_parser("t2v")
    t.add_argument("--prompt", default="")
    t.add_argument("--width", type=int, default=864)
    t.add_argument("--height", type=int, default=480)
    t.add_argument("--seconds", type=float, default=5.0)
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--output-name", default="t2v_shinkai")
    t.add_argument("--gpu", default="RTX-PRO-6000")
    ns = p.parse_args(argv)
    return int(
        {
            "status": cmd_status,
            "download": cmd_download,
            "smoke": cmd_smoke,
            "list-outputs": cmd_list_outputs,
            "t2v": cmd_t2v,
        }[ns.cmd](ns)
        or 0
    )


if __name__ == "__main__":
    sys.exit(main())
