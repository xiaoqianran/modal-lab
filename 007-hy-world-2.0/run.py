#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""007-hy-world-2.0 本地入口 — WorldMirror 2.0 on Modal (cheapest T4)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-hy-world-2-outputs"
VOL_W = "modal-lab-hy-world-2-weights"
DEFAULT_GPU = "T4"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HY-World 2.0 WorldMirror recon on Modal")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")

    sm = sub.add_parser("smoke", help="最低成本：Desk·2图·518·bf16·T4")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--max-images", type=int, default=2)

    inf = sub.add_parser("infer")
    inf.add_argument("--example", default="Desk")
    inf.add_argument("--max-images", type=int, default=2)
    inf.add_argument("--target-size", type=int, default=518)
    inf.add_argument("--run-name", default="")
    inf.add_argument("--gpu", default=DEFAULT_GPU)
    inf.add_argument("--no-bf16", action="store_true")
    inf.add_argument("--save-gs", action="store_true")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 007-hy-world-2.0")
        print("component: WorldMirror-2.0 (worldrecon only)")
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
                "--max-images",
                str(ns.max_images),
            ]
        )

    if ns.cmd == "infer":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "infer",
            "--example",
            ns.example,
            "--max-images",
            str(ns.max_images),
            "--target-size",
            str(ns.target_size),
            "--gpu",
            ns.gpu,
        ]
        if ns.run_name:
            cmd.extend(["--run-name", ns.run_name])
        if ns.no_bf16:
            cmd.append("--no-enable-bf16")
        if ns.save_gs:
            cmd.append("--save-gs")
        return _run(cmd)

    if ns.cmd == "pull":
        dest = Path(ns.dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        return _run([m, "volume", "get", VOL_OUT, ns.remote.lstrip("/"), str(dest)])

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
