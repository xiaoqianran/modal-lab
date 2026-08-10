#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-musicgen 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-musicgen-outputs"
VOL_W = "modal-lab-musicgen-weights"
DEFAULT_GPU = "T4"
DEFAULT_MODEL = "small"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MusicGen on Modal (default T4 · small)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    d.add_argument("--model", default=DEFAULT_MODEL)

    sm = sub.add_parser("smoke", help="15s lo-fi · small · T4")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--model", default=DEFAULT_MODEL)
    sm.add_argument("--duration", type=float, default=15.0)
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--run-name", default="smoke_lofi")

    t = sub.add_parser("t2a")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--prompt", required=True)
    t.add_argument("--duration", type=float, default=15.0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--run-name", default="")
    t.add_argument("--guidance-scale", type=float, default=3.0)
    t.add_argument("--temperature", type=float, default=1.0)

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_lofi")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 016-musicgen")
        print(f"default_gpu: {DEFAULT_GPU}")
        print(f"default_model: {DEFAULT_MODEL}")
        return _run([m, "run", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [m, "run", str(MODAL_APP), "--action", "download", "--model", ns.model]
        if ns.force:
            cmd.append("--force-download")
        return _run(cmd)

    if ns.cmd == "smoke":
        return _run(
            [
                m, "run", "--timestamps", str(MODAL_APP),
                "--action", "smoke",
                "--gpu", ns.gpu,
                "--model", ns.model,
                "--duration", str(ns.duration),
                "--seed", str(ns.seed),
                "--run-name", ns.run_name,
            ]
        )

    if ns.cmd == "t2a":
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP),
            "--action", "t2a",
            "--gpu", ns.gpu,
            "--model", ns.model,
            "--prompt", ns.prompt,
            "--duration", str(ns.duration),
            "--seed", str(ns.seed),
            "--guidance-scale", str(ns.guidance_scale),
            "--temperature", str(ns.temperature),
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "pull":
        dest = Path(ns.dest)
        dest.mkdir(parents=True, exist_ok=True)
        return _run([m, "volume", "get", VOL_OUT, ns.remote, str(dest)])

    raise SystemExit(f"unknown {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
