#!/usr/bin/env python3
"""CLI for 005-v3 Pixal3D RTX PRO 6000 (sm_120)."""
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
    p = argparse.ArgumentParser(description="005-v3 Pixal3D PRO 6000 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="local help")
    sub.add_parser("probe", help="B0: GPU + torch arch on PRO 6000")

    b = sub.add_parser("build-sm120", help="compile/cache sm_120 wheels")
    b.add_argument("--i-know-this-costs-money", action="store_true")
    b.add_argument("--only", default="", help="single package e.g. nvdiffrast|natten")

    v = sub.add_parser("verify", help="install wheels + sm_120 verify")
    v.add_argument("--i-know-this-costs-money", action="store_true")

    d = sub.add_parser("download", help="download Pixal3D weights")
    d.add_argument("--i-know-this-costs-money", action="store_true")

    s = sub.add_parser("smoke", help="end-to-end sample → GLB")
    s.add_argument("--i-know-this-costs-money", action="store_true")
    s.add_argument("--output-name", default="smoke_pro6000")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--resolution", type=int, default=1024)

    args = p.parse_args()
    if args.cmd == "status":
        print("005-v3-pixal3d-pro6000 · Plan A* sm_120 · RTX-PRO-6000")
        print("torch target: 2.11.0+cu128 · ARCH=12.0")
        print("flow: probe → build-sm120 → verify → smoke")
        return 0
    if args.cmd == "probe":
        return _modal(["--action", "probe"])
    if args.cmd == "build-sm120":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        extra = ["--action", "build-sm120"]
        if args.only:
            extra += ["--only", args.only]
        return _modal(extra)
    if args.cmd == "verify":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        return _modal(["--action", "verify"])
    if args.cmd == "download":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        return _modal(["--action", "download"])
    if args.cmd == "smoke":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        return _modal(
            [
                "--action",
                "smoke",
                "--output-name",
                args.output_name,
                "--seed",
                str(args.seed),
                "--resolution",
                str(args.resolution),
            ]
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
