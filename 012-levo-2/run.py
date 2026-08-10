#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""012-levo-2 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-levo-2-outputs"
VOL_W = "modal-lab-levo-2-weights"
DEFAULT_GPU = "L40S"
DEFAULT_MODEL = "v2-medium"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI，请先 pip install modal && modal token set …")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="LeVo 2 / SongGeneration v2 on Modal (default GPU: L40S)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Volume / 权重状态")

    d = sub.add_parser("download", help="CPU 下载 Runtime + 指定模型到 Volume")
    d.add_argument("--force", action="store_true")
    d.add_argument("--model", default=DEFAULT_MODEL, help="v2-medium | v2-large")

    sm = sub.add_parser(
        "smoke",
        help="冒烟：短英文歌词+描述 · 默认 L40S · v2-medium",
    )
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--model", default=DEFAULT_MODEL)
    sm.add_argument("--run-name", default="smoke_en")
    sm.add_argument("--low-mem", action="store_true")
    sm.add_argument("--no-flash", action="store_true")

    t = sub.add_parser("t2a", help="按歌词/描述生成")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--lyrics", default="", help="gt_lyric 字符串（含 [verse] 等结构）")
    t.add_argument("--descriptions", default="")
    t.add_argument("--idx", default="gen")
    t.add_argument("--run-name", default="")
    t.add_argument("--low-mem", action="store_true")
    t.add_argument("--no-flash", action="store_true")
    t.add_argument(
        "--generate-type",
        default="mixed",
        choices=["mixed", "bgm", "vocal", "separate"],
    )

    ls = sub.add_parser("ls", help="列出远程 outputs")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull", help="拉 run 到本地")
    pl.add_argument("--remote", default="runs/smoke_en")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 012-levo-2")
        print(f"default_gpu: {DEFAULT_GPU}")
        print(f"default_model: {DEFAULT_MODEL}")
        print(f"weights_volume: {VOL_W}")
        print(f"outputs_volume: {VOL_OUT}")
        return _run([m, "run", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [
            m,
            "run",
            str(MODAL_APP),
            "--action",
            "download",
            "--model",
            ns.model,
        ]
        if ns.force:
            cmd.append("--force-download")
        return _run(cmd)

    if ns.cmd == "smoke":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "smoke",
            "--gpu",
            ns.gpu,
            "--model",
            ns.model,
            "--run-name",
            ns.run_name,
        ]
        if ns.low_mem:
            cmd.append("--low-mem")
        if ns.no_flash:
            cmd.append("--no-flash")
        return _run(cmd)

    if ns.cmd == "t2a":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "t2a",
            "--gpu",
            ns.gpu,
            "--model",
            ns.model,
            "--idx",
            ns.idx,
            "--generate-type",
            ns.generate_type,
        ]
        if ns.lyrics:
            cmd += ["--lyrics", ns.lyrics]
        if ns.descriptions:
            cmd += ["--descriptions", ns.descriptions]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        if ns.low_mem:
            cmd.append("--low-mem")
        if ns.no_flash:
            cmd.append("--no-flash")
        return _run(cmd)

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "pull":
        dest = Path(ns.dest)
        dest.mkdir(parents=True, exist_ok=True)
        return _run([m, "volume", "get", VOL_OUT, ns.remote, str(dest)])

    raise SystemExit(f"unknown cmd {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
