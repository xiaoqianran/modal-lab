#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""028-fish-s2 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-fish-s2-outputs"
VOL_W = "modal-lab-fish-s2-weights"
VOL_P = "modal-lab-fish-s2-prompts"
DEFAULT_GPU = "L40S"
VOICES_DIR = EXP_DIR / "inputs" / "voices"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        p = Path.home() / ".local" / "bin" / "modal"
        if p.is_file():
            return str(p)
        raise SystemExit("未找到 modal CLI")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="028 Fish Audio S2 Pro on Modal (default L40S · Research License)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")

    sm = sub.add_parser("smoke")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument(
        "--kind",
        default="en",
        choices=["en", "zh", "tags", "clone"],
        help="smoke 场景",
    )
    sm.add_argument("--run-name", default="")
    sm.add_argument("--compile", action="store_true")

    t = sub.add_parser("t2s")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--text", required=True)
    t.add_argument("--ref-audio", default="", help="path under /prompts, local name, or URL")
    t.add_argument("--ref-text", default="", help="transcript of reference audio")
    t.add_argument("--voice", default="")
    t.add_argument("--temperature", type=float, default=0.8)
    t.add_argument("--top-p", type=float, default=0.8)
    t.add_argument("--repetition-penalty", type=float, default=1.1)
    t.add_argument("--max-new-tokens", type=int, default=1024)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--run-name", default="")
    t.add_argument("--compile", action="store_true")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_en")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 028-fish-s2")
        print(f"default_gpu: {DEFAULT_GPU}")
        return _run([m, "run", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [m, "run", str(MODAL_APP), "--action", "download"]
        if ns.force:
            cmd.append("--force-download")
        return _run(cmd)

    if ns.cmd == "smoke":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "smoke",
            "--gpu",
            ns.gpu,
            "--smoke-kind",
            ns.kind,
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        if ns.compile:
            cmd.append("--compile-model")
        return _run(cmd)

    if ns.cmd == "t2s":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "t2s",
            "--gpu",
            ns.gpu,
            "--text",
            ns.text,
            "--temperature",
            str(ns.temperature),
            "--top-p",
            str(ns.top_p),
            "--repetition-penalty",
            str(ns.repetition_penalty),
            "--max-new-tokens",
            str(ns.max_new_tokens),
            "--seed",
            str(ns.seed),
        ]
        if ns.ref_audio:
            cmd += ["--ref-audio-path", ns.ref_audio]
        if ns.ref_text:
            cmd += ["--ref-text", ns.ref_text]
        if ns.voice:
            cmd += ["--voice", ns.voice]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        if ns.compile:
            cmd.append("--compile-model")
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
