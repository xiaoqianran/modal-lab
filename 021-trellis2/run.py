#!/usr/bin/env python3
"""CLI for 021-trellis2."""
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
    p = argparse.ArgumentParser(description="021 TRELLIS.2 image→3D")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name in ("probe", "build", "verify", "download"):
        pr = sub.add_parser(name)
        pr.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])
        if name == "download":
            pr.add_argument("--force", action="store_true")
    sm = sub.add_parser("smoke")
    sm.add_argument("--i-know-this-costs-money", action="store_true")
    sm.add_argument("--gpu", default="L40S", choices=["L40S", "RTX-PRO-6000"])
    sm.add_argument("--output-name", default="")
    sm.add_argument(
        "--image-url",
        default="https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png",
    )
    sm.add_argument(
        "--pipeline-type",
        default="512",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
    )
    sm.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.cmd == "status":
        print("021-trellis2 · microsoft/TRELLIS.2-4B · MIT")
        print("default GPU: L40S · optional RTX-PRO-6000")
        print("flow: probe → build → verify → download → smoke")
        return 0
    if args.cmd in {"probe", "build", "verify", "download"}:
        extra = ["--action", args.cmd, "--gpu", args.gpu]
        return _modal(extra)
    if args.cmd == "smoke":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        name = args.output_name or (
            "smoke_pro6000" if "PRO" in args.gpu.upper() else "smoke_l40s"
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
                "--pipeline-type",
                args.pipeline_type,
                "--seed",
                str(args.seed),
            ]
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
