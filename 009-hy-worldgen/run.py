#!/usr/bin/env python3
"""CLI for 009-hy-worldgen (HY-World 2.0 worldgen: pano → 3D world)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
APP = EXP / "modal_app.py"


def modal_run(args: list[str]) -> int:
    cmd = ["modal", "run", str(APP), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="009 HY-Worldgen on Modal")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="pipeline + volumes")

    prep = sub.add_parser("prepare", help="import 008 panorama into a scene dir")
    prep.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    prep.add_argument("--scene", default="scene_from_008")

    st = sub.add_parser("stage", help="run a numbered stage (gated)")
    st.add_argument("n", type=int, choices=[1, 2, 3, 4, 5])
    st.add_argument("--gpu", default="RTX-PRO-6000")
    st.add_argument("--scene", default="scene_from_008")

    args = p.parse_args(argv)

    if args.cmd == "status":
        return modal_run(["--action", "status"])
    if args.cmd == "prepare":
        return modal_run(
            ["--action", "prepare", "--from-008", args.from_008, "--scene", args.scene]
        )
    if args.cmd == "stage":
        return modal_run(
            [
                "--action",
                "stage",
                "--stage",
                str(args.n),
                "--gpu",
                args.gpu,
                "--scene",
                args.scene,
            ]
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
