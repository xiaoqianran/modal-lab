#!/usr/bin/env python3
"""CLI wrapper for 008-hy-pano (Modal HY-Pano 2.0 · lightweight Qwen default)."""

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
    p = argparse.ArgumentParser(description="008 HY-Pano 2.0 on Modal (Qwen default)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="print defaults / volumes")

    d = sub.add_parser("download", help="CPU download weights to Volume")
    d.add_argument("--backend", default="qwen", choices=["qwen", "full", "both"])

    s = sub.add_parser("smoke", help="one-image smoke (default backend=qwen)")
    s.add_argument("--backend", default="qwen", choices=["qwen", "full"])
    s.add_argument("--gpu", default=None, help="e.g. A100-80GB / L40S")
    s.add_argument("--image", default="desk.jpg")
    s.add_argument("--prompt", default=None)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument(
        "--load-mode",
        default="gpu",
        choices=["gpu", "cpu_offload", "sequential_offload"],
        help="qwen load strategy; auto-offload if VRAM < ~70GB when set to gpu",
    )
    s.add_argument("--height", type=int, default=960)
    s.add_argument("--width", type=int, default=1952)
    s.add_argument("--steps", type=int, default=40, dest="diff_infer_steps")

    i = sub.add_parser("infer", help="custom inference")
    i.add_argument("--backend", default="qwen", choices=["qwen", "full"])
    i.add_argument("--gpu", default=None)
    i.add_argument("--image", default="desk.jpg")
    i.add_argument("--prompt", default=None)
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--run-name", default=None)
    i.add_argument("--height", type=int, default=960)
    i.add_argument("--width", type=int, default=1952)
    i.add_argument("--steps", type=int, default=40, dest="diff_infer_steps")
    i.add_argument("--use-taylor-cache", action="store_true")
    i.add_argument(
        "--load-mode",
        default="gpu",
        choices=["gpu", "cpu_offload", "sequential_offload"],
    )

    sub.add_parser("ls", help="list output runs on Volume")

    pull = sub.add_parser("pull", help="modal volume get a run dir")
    pull.add_argument("--remote", required=True, help="e.g. runs/smoke_qwen")
    pull.add_argument("--dest", default=None)

    args = p.parse_args(argv)

    if args.cmd == "status":
        return modal_run(["--action", "status"])

    if args.cmd == "download":
        return modal_run(["--action", "download", "--backend", args.backend])

    if args.cmd == "ls":
        return modal_run(["--action", "ls"])

    if args.cmd in ("smoke", "infer"):
        margs = [
            "--action",
            args.cmd,
            "--backend",
            args.backend,
            "--image",
            args.image,
            "--seed",
            str(args.seed),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--diff-infer-steps",
            str(args.diff_infer_steps),
            "--load-mode",
            args.load_mode,
        ]
        if args.gpu:
            margs += ["--gpu", args.gpu]
        if args.prompt:
            margs += ["--prompt", args.prompt]
        if args.cmd == "infer":
            if args.run_name:
                margs += ["--run-name", args.run_name]
            if args.use_taylor_cache:
                margs += ["--use-taylor-cache"]
        return modal_run(margs)

    if args.cmd == "pull":
        dest = Path(args.dest) if args.dest else EXP / "outputs" / Path(args.remote).name
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            "modal",
            "volume",
            "get",
            "modal-lab-hy-pano-outputs",
            args.remote,
            str(dest),
        ]
        print("+", " ".join(cmd), flush=True)
        return subprocess.call(cmd)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
