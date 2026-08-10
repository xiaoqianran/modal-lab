#!/usr/bin/env python3
"""CLI for 023-b-sf3d."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "modal_app.py"
SAMPLE = (
    "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png"
)


def _modal(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "modal", "run", str(APP), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="023-b SF3D image→3D")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    pr = sub.add_parser("probe")
    pr.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])
    sm = sub.add_parser("smoke")
    sm.add_argument("--i-know-this-costs-money", action="store_true")
    sm.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])
    sm.add_argument("--output-name", default="")
    sm.add_argument("--image-url", default=SAMPLE)
    sm.add_argument("--texture-resolution", type=int, default=1024)
    args = p.parse_args()
    if args.cmd == "status":
        print("023-b-sf3d · SF3D · Stability Community")
        print("weights default: cocktailpeanut/sf3d (official gated: stabilityai/stable-fast-3d)")
        print("default GPU: L40S · optional RTX-PRO-6000 · smoke OK both")
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
            "--texture-resolution",
            str(args.texture_resolution),
        ]
        return _modal(extra)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
