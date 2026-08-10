#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""010-ace-step-1.5 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-ace-step-1.5-outputs"
VOL_W = "modal-lab-ace-step-1.5-weights"
DEFAULT_GPU = "L4"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI，请先 pip install modal && modal token set …")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ACE-Step 1.5 on Modal (music · default L4)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="查看 Volume / 权重 / 环境")

    d = sub.add_parser("download", help="CPU 下载 HF 主包到 Volume（~10GB）")
    d.add_argument("--force", action="store_true")

    sm = sub.add_parser("smoke", help="最低成本冒烟：20s 器乐 lo-fi · thinking 关 · 默认 L4")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--duration", type=float, default=20.0)
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--run-name", default="smoke_lofi")

    t = sub.add_parser("t2m", help="Text2Music 生成（可开 thinking / LM）")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--example", default="smoke_lofi", help="examples/*.json 名（无 .json）")
    t.add_argument("--caption", default="")
    t.add_argument("--lyrics", default="")
    t.add_argument("--duration", type=float, default=30.0)
    t.add_argument("--bpm", type=int, default=0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--thinking", action="store_true", help="启用 5Hz LM thinking")
    t.add_argument("--init-lm", action="store_true", help="即使不 thinking 也加载 LM")
    t.add_argument("--vocal", action="store_true", help="非纯器乐（关闭 instrumental）")
    t.add_argument("--steps", type=int, default=8)
    t.add_argument("--format", dest="audio_format", default="flac")
    t.add_argument("--run-name", default="")
    t.add_argument("--dit", default="acestep-v15-turbo")
    t.add_argument("--lm", default="acestep-5Hz-lm-1.7B")

    ls = sub.add_parser("ls", help="列出远程 outputs volume")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull", help="从 Volume 拉某个 run 到本地")
    pl.add_argument("--remote", default="runs/smoke_lofi")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    sub.add_parser("list-outputs", help="结构化列出远程 runs")

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 010-ace-step-1.5")
        print("app: modal-lab-ace-step-1.5")
        print(f"default_gpu: {DEFAULT_GPU}")
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

    if ns.cmd == "t2m":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "t2m",
            "--gpu",
            ns.gpu,
            "--example",
            ns.example,
            "--duration",
            str(ns.duration),
            "--seed",
            str(ns.seed),
            "--inference-steps",
            str(ns.steps),
            "--audio-format",
            ns.audio_format,
            "--dit-model",
            ns.dit,
            "--lm-model",
            ns.lm,
        ]
        if ns.caption:
            cmd.extend(["--caption", ns.caption])
        if ns.lyrics:
            cmd.extend(["--lyrics", ns.lyrics])
        if ns.bpm and ns.bpm > 0:
            cmd.extend(["--bpm", str(ns.bpm)])
        if ns.run_name:
            cmd.extend(["--run-name", ns.run_name])
        if ns.thinking:
            cmd.append("--thinking")
        if ns.init_lm:
            cmd.append("--init-lm")
        if ns.vocal:
            cmd.append("--no-instrumental")
        return _run(cmd)

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "list-outputs":
        return _run([m, "run", str(MODAL_APP), "--action", "list-outputs"])

    if ns.cmd == "pull":
        dest = Path(ns.dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        return _run(
            [m, "volume", "get", VOL_OUT, ns.remote.lstrip("/"), str(dest)]
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
