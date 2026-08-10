#!/usr/bin/env python3
"""CLI for 020-triposr."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "modal_app.py"


def _modal(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "modal", "run", str(APP), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="020 TripoSR image→3D")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    pr = sub.add_parser("probe")
    pr.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])
    sm = sub.add_parser("smoke")
    sm.add_argument("--i-know-this-costs-money", action="store_true")
    sm.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])
    sm.add_argument("--output-name", default="")
    sm.add_argument(
        "--image-url",
        default="https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png",
    )
    sm.add_argument("--no-bake-texture", action="store_true")
    args = p.parse_args()
    if args.cmd == "status":
        print("020-triposr · stabilityai/TripoSR · MIT")
        print("default GPU: L40S · optional RTX-PRO-6000")
        print("flow: probe → smoke")
        return 0
    if args.cmd == "probe":
        return _modal(["--action", "probe", "--gpu", args.gpu])
    if args.cmd == "smoke":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        name = args.output_name or (
            "smoke_pro6000" if "PRO" in args.gpu.upper() else "smoke_l40s"
        )
        extra = [
            "--action",
            "smoke",
            "--gpu",
            args.gpu,
            "--output-name",
            name,
            "--image-url",
            args.image_url,
        ]
        if args.no_bake_texture:
            extra += ["--no-bake-texture"]
        return _modal(extra)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
