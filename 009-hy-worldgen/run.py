#!/usr/bin/env python3
"""CLI for 009-hy-worldgen — single PRO-6000 worldgen smoke."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parent
APP = EXP / "modal_app.py"


def modal_run(args: list[str], *, detach: bool = False) -> int:
    cmd = ["modal", "run"]
    if detach:
        cmd.append("--detach")
    cmd += [str(APP), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    prep = sub.add_parser("prepare")
    prep.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    prep.add_argument("--scene", default="scene_from_008")

    dl = sub.add_parser("download")
    dl.add_argument("--which", default="worldstereo-dmd")

    st = sub.add_parser("stage")
    st.add_argument("n", type=int, choices=[1, 2, 3, 4, 5])
    st.add_argument("--gpu", default="RTX-PRO-6000")
    st.add_argument("--scene", default="scene_from_008")
    st.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    st.add_argument("--nframe", type=int, default=16)
    st.add_argument("--split-view-num", type=int, default=1)
    st.add_argument("--max-steps", type=int, default=4000)
    st.add_argument("--detach", action="store_true", help="modal run --detach")

    sm = sub.add_parser("smoke", help="prepare+download+stages 1-5 on one PRO-6000")
    sm.add_argument("--gpu", default="RTX-PRO-6000")
    sm.add_argument("--scene", default="scene_from_008")
    sm.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    sm.add_argument("--nframe", type=int, default=16)
    sm.add_argument("--max-steps", type=int, default=4000)
    sm.add_argument("--detach", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "status":
        return modal_run(["--action", "status"])
    if args.cmd == "prepare":
        return modal_run(
            ["--action", "prepare", "--from-008", args.from_008, "--scene", args.scene]
        )
    if args.cmd == "download":
        return modal_run(["--action", "download", "--which", args.which])
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
                "--from-008",
                args.from_008,
                "--nframe",
                str(args.nframe),
                "--split-view-num",
                str(args.split_view_num),
                "--max-steps",
                str(args.max_steps),
            ],
            detach=bool(args.detach),
        )
    if args.cmd == "smoke":
        return modal_run(
            [
                "--action",
                "smoke",
                "--gpu",
                args.gpu,
                "--scene",
                args.scene,
                "--from-008",
                args.from_008,
                "--nframe",
                str(args.nframe),
                "--max-steps",
                str(args.max_steps),
            ],
            detach=bool(args.detach),
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
