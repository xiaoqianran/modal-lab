#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""029-voxcpm2 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-voxcpm2-outputs"
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
    p = argparse.ArgumentParser(description="029 VoxCPM2 on Modal (default L4 · Apache)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")

    sm = sub.add_parser("smoke")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--kind", default="en", choices=["en", "zh", "design", "clone"])
    sm.add_argument("--run-name", default="")
    sm.add_argument("--optimize", action="store_true")
    sm.add_argument("--timesteps", type=int, default=10)

    t = sub.add_parser("t2s")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--text", required=True)
    t.add_argument("--reference-wav", default="")
    t.add_argument("--prompt-wav", default="")
    t.add_argument("--prompt-text", default="")
    t.add_argument("--cfg-value", type=float, default=2.0)
    t.add_argument("--timesteps", type=int, default=10)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--run-name", default="")
    t.add_argument("--optimize", action="store_true")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_en")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 029-voxcpm2")
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
            "--inference-timesteps", str(ns.timesteps),
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        if ns.optimize:
            cmd.append("--optimize")
        return _run(cmd)

    if ns.cmd == "t2s":
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP),
            "--action", "t2s", "--gpu", ns.gpu, "--text", ns.text,
            "--cfg-value", str(ns.cfg_value),
            "--inference-timesteps", str(ns.timesteps),
            "--seed", str(ns.seed),
        ]
        if ns.reference_wav:
            cmd += ["--reference-wav", ns.reference_wav]
        if ns.prompt_wav:
            cmd += ["--prompt-wav", ns.prompt_wav]
        if ns.prompt_text:
            cmd += ["--prompt-text", ns.prompt_text]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        if ns.optimize:
            cmd.append("--optimize")
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
