#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""015-xiaomi-robotics-1-robocasa365 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-xr1-robocasa365-outputs"
VOL_W = "modal-lab-xr1-robocasa365-weights"
DEFAULT_GPU = "A100-40GB"


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
        description="Xiaomi-Robotics-1 RoboCasa365 VLA on Modal"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="查看 Volume / 权重 / 最近 runs")

    d = sub.add_parser("download", help="CPU 下载 HF 权重到 Volume (~10GB)")
    d.add_argument("--force", action="store_true")

    sm = sub.add_parser(
        "smoke",
        help="冒烟：合成三视角 + 指令 → 16 步动作（默认 A100-40GB）",
    )
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--instruction", default="close the blender lid")
    sm.add_argument("--run-name", default="smoke_close_blender_lid")
    sm.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    sm.add_argument("--num-steps", type=int, default=5)
    sm.add_argument("--obs-history", type=int, default=4)

    inf = sub.add_parser("infer", help="自定义指令动作生成")
    inf.add_argument("--instruction", required=True)
    inf.add_argument("--gpu", default=DEFAULT_GPU)
    inf.add_argument("--run-name", default="")
    inf.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    inf.add_argument("--num-steps", type=int, default=5)
    inf.add_argument("--obs-history", type=int, default=4)

    ls = sub.add_parser("ls", help="列出远程 outputs volume")
    ls.add_argument("--path", default="runs")

    sub.add_parser("list-outputs", help="JSON 列出 runs")

    pl = sub.add_parser("pull", help="从 Volume 拉某个 run 到本地")
    pl.add_argument("--remote", default="runs")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 015-xiaomi-robotics-1-robocasa365")
        print(f"modal_app: {MODAL_APP}")
        print("app: modal-lab-xr1-robocasa365")
        print(f"weights volume: {VOL_W}")
        print(f"outputs volume: {VOL_OUT}")
        print(f"default gpu: {DEFAULT_GPU}")
        print("hf: XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365")
        print("code: https://github.com/XiaomiRobotics/Xiaomi-Robotics-1")
        return _run([m, "run", "--timestamps", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [m, "run", "--timestamps", str(MODAL_APP), "--action", "download"]
        if ns.force:
            cmd.append("--force-download")
        return _run(cmd)

    if ns.cmd in {"smoke", "infer"}:
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            ns.cmd,
            "--gpu",
            ns.gpu,
            "--instruction",
            ns.instruction,
            "--attn",
            ns.attn,
            "--num-steps",
            str(ns.num_steps),
            "--obs-history",
            str(ns.obs_history),
        ]
        if ns.run_name:
            cmd.extend(["--run-name", ns.run_name])
        return _run(cmd)

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "list-outputs":
        return _run(
            [m, "run", "--timestamps", str(MODAL_APP), "--action", "list-outputs"]
        )

    if ns.cmd == "pull":
        dest = Path(ns.dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        return _run(
            [m, "volume", "get", VOL_OUT, ns.remote.lstrip("/"), str(dest)]
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
