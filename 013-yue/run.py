#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""013-yue 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-yue-outputs"
DEFAULT_GPU = "L40S"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="YuE on Modal (default L40S · en-cot)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    d.add_argument("--stage1", default="en-cot")

    sm = sub.add_parser("smoke", help="2-segment English CoT smoke")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--stage1", default="en-cot")
    sm.add_argument("--run-n-segments", type=int, default=2)
    sm.add_argument("--max-new-tokens", type=int, default=3000)
    sm.add_argument("--stage2-batch-size", type=int, default=2)
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--run-name", default="smoke_en")

    g = sub.add_parser("generate")
    g.add_argument("--gpu", default=DEFAULT_GPU)
    g.add_argument("--stage1", default="en-cot")
    g.add_argument("--genre", required=True)
    g.add_argument("--lyrics-file", type=Path, required=True)
    g.add_argument("--run-n-segments", type=int, default=2)
    g.add_argument("--max-new-tokens", type=int, default=3000)
    g.add_argument("--stage2-batch-size", type=int, default=2)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--run-name", default="")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_en")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 013-yue")
        print(f"default_gpu: {DEFAULT_GPU}")
        return _run([m, "run", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [m, "run", str(MODAL_APP), "--action", "download", "--stage1", ns.stage1]
        if ns.force:
            cmd.append("--force-download")
        return _run(cmd)

    if ns.cmd == "smoke":
        return _run(
            [
                m, "run", "--timestamps", str(MODAL_APP),
                "--action", "smoke",
                "--gpu", ns.gpu,
                "--stage1", ns.stage1,
                "--run-n-segments", str(ns.run_n_segments),
                "--max-new-tokens", str(ns.max_new_tokens),
                "--stage2-batch-size", str(ns.stage2_batch_size),
                "--seed", str(ns.seed),
                "--run-name", ns.run_name,
            ]
        )

    if ns.cmd == "generate":
        lyrics = ns.lyrics_file.read_text(encoding="utf-8")
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP),
            "--action", "generate",
            "--gpu", ns.gpu,
            "--stage1", ns.stage1,
            "--genre", ns.genre,
            "--lyrics", lyrics,
            "--run-n-segments", str(ns.run_n_segments),
            "--max-new-tokens", str(ns.max_new_tokens),
            "--stage2-batch-size", str(ns.stage2_batch_size),
            "--seed", str(ns.seed),
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "pull":
        dest = Path(ns.dest)
        dest.mkdir(parents=True, exist_ok=True)
        return _run([m, "volume", "get", VOL_OUT, ns.remote, str(dest)])

    raise SystemExit(f"unknown {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
