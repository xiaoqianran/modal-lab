#!/usr/bin/env python3
"""CLI for 022-hunyuan3d-2.1."""
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
    p = argparse.ArgumentParser(description="022 Hunyuan3D-2.1 image→3D")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    pr = sub.add_parser("probe")
    pr.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])

    sm = sub.add_parser("smoke")
    sm.add_argument("--i-know-this-costs-money", action="store_true")
    sm.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])
    sm.add_argument("--mode", default="full", choices=["shape", "full"])
    sm.add_argument("--output-name", default="")
    sm.add_argument("--image-url", default=SAMPLE)
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--max-num-view", type=int, default=6)
    sm.add_argument("--paint-resolution", type=int, default=512)

    args = p.parse_args()
    if args.cmd == "status":
        print("022-hunyuan3d-2.1 · tencent/Hunyuan3D-2.1")
        print("license: Tencent Hunyuan 3D 2.1 Community License")
        print("default GPU: L40S · optional RTX-PRO-6000")
        print("flow: probe → smoke --mode shape|full")
        print("volumes: modal-lab-hunyuan3d21-weights / -outputs")
        return 0
    if args.cmd == "probe":
        return _modal(["--action", "probe", "--gpu", args.gpu])
    if args.cmd == "smoke":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        tag = "pro6000" if "PRO" in args.gpu.upper() else "l40s"
        name = args.output_name or (
            f"smoke_shape_{tag}" if args.mode == "shape" else f"smoke_{tag}"
        )
        return _modal(
            [
                "--action",
                "smoke",
                "--gpu",
                args.gpu,
                "--output-name",
                name,
                "--image-url",
                args.image_url,
                "--mode",
                args.mode,
                "--seed",
                str(args.seed),
                "--max-num-view",
                str(args.max_num_view),
                "--paint-resolution",
                str(args.paint_resolution),
            ]
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
