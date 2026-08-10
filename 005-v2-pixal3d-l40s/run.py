#!/usr/bin/env python3
"""CLI for 005-v2 Pixal3D L40S."""
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
    p = argparse.ArgumentParser(description="005-v2 Pixal3D L40S CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="local status")
    sub.add_parser("remote-status", help="Modal status on L40S")

    b = sub.add_parser("build-sm89", help="compile/cache sm_89 wheels")
    b.add_argument("--i-know-this-costs-money", action="store_true")

    v = sub.add_parser("verify", help="install wheels + verify_sm89")
    v.add_argument("--i-know-this-costs-money", action="store_true")

    d = sub.add_parser("download", help="download Pixal3D weights to Volume")
    d.add_argument("--i-know-this-costs-money", action="store_true")

    s = sub.add_parser("smoke", help="end-to-end sample → GLB on L40S")
    s.add_argument("--i-know-this-costs-money", action="store_true")
    s.add_argument("--output-name", default="smoke_l40s")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--resolution", type=int, default=1024)

    args = p.parse_args()
    if args.cmd == "status":
        print("005-v2-pixal3d-l40s · Plan A sm_89 · default GPU L40S")
        print("wheels volume: modal-lab-pixal3d-l40s-wheels")
        print("commands: build-sm89 | verify | download | smoke")
        return 0
    if args.cmd == "remote-status":
        return _modal(["--action", "status"])
    if args.cmd == "build-sm89":
        if not args.i_know_this_costs_money:
            print("need --i-know-this-costs-money", file=sys.stderr)
            return 2
        return _modal(["--action", "build-sm89"])
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
