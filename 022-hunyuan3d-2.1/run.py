#!/usr/bin/env python3
"""Small local CLI for the L40S Hunyuan3D-2.1 Modal app."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

APP = Path(__file__).with_name("modal_app.py")
SAMPLE = "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png"


def modal_run(*args: str) -> int:
    cmd = ["modal", "run", str(APP), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 on Modal L40S")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("probe")

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--i-know-this-costs-money", action="store_true")
    smoke.add_argument("--image-url", default=SAMPLE)
    smoke.add_argument("--output-name", default="smoke_l40s")
    smoke.add_argument("--mode", choices=["shape", "full"], default="full")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--max-num-view", type=int, default=6)
    smoke.add_argument("--paint-resolution", type=int, default=512)

    args = parser.parse_args()
    if args.command == "status":
        print("022-hunyuan3d-2.1 · L40S only · torch 2.5.1 / CUDA 12.4 / sm_89")
        print("commands: probe | smoke")
        return 0
    if args.command == "probe":
        return modal_run("--action", "probe")
    if not args.i_know_this_costs_money:
        parser.error("smoke requires --i-know-this-costs-money")

    return modal_run(
        "--action", "smoke",
        "--image-url", args.image_url,
        "--output-name", args.output_name,
        "--mode", args.mode,
        "--seed", str(args.seed),
        "--max-num-view", str(args.max_num_view),
        "--paint-resolution", str(args.paint_resolution),
    )


if __name__ == "__main__":
    raise SystemExit(main())
