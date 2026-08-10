#!/usr/bin/env python3
"""CLI for 009-hy-worldgen — Stage1+2 with official Qwen3-VL-8B (vLLM)."""

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


def _flag(name: str, value: bool) -> str:
    """Modal bool CLI: --foo / --no-foo."""
    return f"--{name}" if value else f"--no-{name}"


def _common_stage_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", default="RTX-PRO-6000")
    p.add_argument("--scene", default="scene_from_008")
    p.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    p.add_argument("--nframe", type=int, default=16)
    p.add_argument("--split-view-num", type=int, default=1)
    p.add_argument("--wonder-topk", type=int, default=1)
    p.add_argument("--recon-topk", type=int, default=0)
    p.add_argument("--force-vlm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--apply-nav-traj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="WorldNav object/nav trajectories (needs SAM3/ZIM)",
    )
    p.add_argument(
        "--apply-up-route",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--apply-recon-iteration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--vlm-mode",
        choices=["share", "split"],
        default="share",
        help="share=same GPU as vision models; split=last GPU if multi-GPU",
    )
    p.add_argument("--vlm-mem-util", type=float, default=0.38)
    p.add_argument("--vlm-max-model-len", type=int, default=8192)
    p.add_argument("--detach", action="store_true", help="modal run --detach")


def _stage_args(ns: argparse.Namespace) -> list[str]:
    return [
        "--gpu",
        ns.gpu,
        "--scene",
        ns.scene,
        "--from-008",
        ns.from_008,
        "--nframe",
        str(ns.nframe),
        "--split-view-num",
        str(ns.split_view_num),
        "--wonder-topk",
        str(ns.wonder_topk),
        "--recon-topk",
        str(ns.recon_topk),
        _flag("force-vlm", ns.force_vlm),
        _flag("apply-nav-traj", ns.apply_nav_traj),
        _flag("apply-up-route", ns.apply_up_route),
        _flag("apply-recon-iteration", ns.apply_recon_iteration),
        "--vlm-mode",
        ns.vlm_mode,
        "--vlm-mem-util",
        str(ns.vlm_mem_util),
        "--vlm-max-model-len",
        str(ns.vlm_max_model_len),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="009 HY-Worldgen — panorama → trajectories (Qwen3-VL-8B) → 3D world"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="pipeline + VLM config")

    prep = sub.add_parser("prepare", help="import 008 panorama into scene dir")
    prep.add_argument("--from-008", default="smoke_qwen", dest="from_008")
    prep.add_argument("--scene", default="scene_from_008")

    dl = sub.add_parser("download", help="download weights to Modal volume")
    dl.add_argument(
        "--which",
        default="vlm",
        choices=[
            "vlm",
            "qwen3-vl",
            "qwen",
            "worldstereo-dmd",
            "worldstereo",
            "worldmirror",
            "wm",
            "all",
        ],
        help="vlm=Qwen3-VL-8B (Stage1/2); worldstereo-dmd (Stage3); worldmirror; all",
    )

    st = sub.add_parser("stage", help="run a single stage (1–5)")
    st.add_argument("n", type=int, choices=[1, 2, 3, 4, 5])
    _common_stage_flags(st)
    st.add_argument("--max-steps", type=int, default=4000)
    st.add_argument(
        "--keep-vlm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do not kill vLLM after stage 1/2 (chain stages manually)",
    )

    s12 = sub.add_parser(
        "stage12",
        help="Stage1+2 with ONE official Qwen3-VL-8B vLLM lifecycle (recommended)",
    )
    _common_stage_flags(s12)

    sm = sub.add_parser(
        "smoke",
        help="prepare + download vlm/dmd + stage12 + stages 3–5",
    )
    _common_stage_flags(sm)
    sm.add_argument("--max-steps", type=int, default=4000)

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
        flags = _stage_args(args) + [
            "--stage",
            str(args.n),
            "--max-steps",
            str(args.max_steps),
            _flag("keep-vlm", bool(args.keep_vlm)),
        ]
        return modal_run(["--action", "stage", *flags], detach=bool(args.detach))
    if args.cmd == "stage12":
        return modal_run(
            ["--action", "stage12", *_stage_args(args)],
            detach=bool(args.detach),
        )
    if args.cmd == "smoke":
        return modal_run(
            [
                "--action",
                "smoke",
                *_stage_args(args),
                "--max-steps",
                str(args.max_steps),
            ],
            detach=bool(args.detach),
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
