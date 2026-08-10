#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""030-vibevoice 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-vibevoice-outputs"
DEFAULT_GPU = "L4"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        p = Path.home() / ".local" / "bin" / "modal"
        if p.is_file():
            return str(p)
        raise SystemExit("未找到 modal CLI")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="030 VibeVoice Realtime on Modal (L4 · MIT)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")

    sm = sub.add_parser("smoke")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--kind", default="en", choices=["en", "long", "emma"])
    sm.add_argument("--speaker", default="")
    sm.add_argument("--run-name", default="")
    sm.add_argument("--cfg-scale", type=float, default=1.5)

    t = sub.add_parser("t2s")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--text", required=True)
    t.add_argument("--speaker", default="Carter")
    t.add_argument("--cfg-scale", type=float, default=1.5)
    t.add_argument("--run-name", default="")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_en")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 030-vibevoice")
        print(f"default_gpu: {DEFAULT_GPU}")
        return _run([m, "run", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [m, "run", str(MODAL_APP), "--action", "download"]
        if ns.force:
            cmd.append("--force-download")
        return _run(cmd)

    if ns.cmd == "smoke":
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP),
            "--action", "smoke", "--gpu", ns.gpu,
            "--smoke-kind", ns.kind,
            "--cfg-scale", str(ns.cfg_scale),
        ]
        if ns.speaker:
            cmd += ["--speaker", ns.speaker]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "t2s":
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP),
            "--action", "t2s", "--gpu", ns.gpu, "--text", ns.text,
            "--speaker", ns.speaker, "--cfg-scale", str(ns.cfg_scale),
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
