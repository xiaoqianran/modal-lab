#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""011-stable-audio-3 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-stable-audio-3-outputs"
VOL_W = "modal-lab-stable-audio-3-weights"
DEFAULT_GPU = "L4"
DEFAULT_MODEL = "medium"


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
        description="Stable Audio 3 Medium on Modal (default GPU: L4 · cheapest fit)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="查看 Volume / 权重 / 环境")

    d = sub.add_parser("download", help="CPU 下载 HF medium 权重到 Volume（~10GB · 门禁）")
    d.add_argument("--force", action="store_true")

    sm = sub.add_parser("smoke", help="最低成本冒烟：20s house 器乐 · 8 steps · 默认 L4")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--duration", type=float, default=20.0)
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--run-name", default="smoke_house")

    t = sub.add_parser("t2a", help="Text-to-audio 生成")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--prompt", required=True)
    t.add_argument("--negative-prompt", default="")
    t.add_argument("--duration", type=float, default=30.0)
    t.add_argument("--steps", type=int, default=8)
    t.add_argument("--cfg-scale", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--model", default=DEFAULT_MODEL, help="medium | small-music | small-sfx …")
    t.add_argument("--format", dest="audio_format", default="flac")
    t.add_argument("--run-name", default="")

    ls = sub.add_parser("ls", help="列出远程 outputs volume")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull", help="从 Volume 拉某个 run 到本地")
    pl.add_argument("--remote", default="runs/smoke_house")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    sub.add_parser("list-outputs", help="结构化列出远程 runs")

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 011-stable-audio-3")
        print("app: modal-lab-stable-audio-3")
        print(f"default_gpu: {DEFAULT_GPU}")
        print(f"default_model: {DEFAULT_MODEL}")
        print(f"weights_volume: {VOL_W}")
        print(f"outputs_volume: {VOL_OUT}")
        return _run([m, "run", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [m, "run", str(MODAL_APP), "--action", "download"]
        if ns.force:
            cmd.append("--force-download")
        return _run(cmd)

    if ns.cmd == "smoke":
        return _run(
            [
                m,
                "run",
                "--timestamps",
                str(MODAL_APP),
                "--action",
                "smoke",
                "--gpu",
                ns.gpu,
                "--duration",
                str(ns.duration),
                "--seed",
                str(ns.seed),
                "--run-name",
                ns.run_name,
            ]
        )

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
            "--prompt",
            ns.prompt,
            "--duration",
            str(ns.duration),
            "--steps",
            str(ns.steps),
            "--cfg-scale",
            str(ns.cfg_scale),
            "--seed",
            str(ns.seed),
            "--model",
            ns.model,
            "--audio-format",
            ns.audio_format,
        ]
        if ns.negative_prompt:
            cmd += ["--negative-prompt", ns.negative_prompt]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "pull":
        dest = Path(ns.dest)
        dest.mkdir(parents=True, exist_ok=True)
        return _run([m, "volume", "get", VOL_OUT, ns.remote, str(dest)])

    if ns.cmd == "list-outputs":
        return _run([m, "run", str(MODAL_APP), "--action", "list-outputs"])

    raise SystemExit(f"unknown cmd {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
