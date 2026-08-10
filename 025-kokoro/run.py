#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""025-kokoro 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-kokoro-outputs"
VOL_W = "modal-lab-kokoro-weights"
DEFAULT_GPU = "T4"
DEFAULT_MODEL = "v1"
DEFAULT_VOICE = "af_heart"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        # uv tool path
        p = Path.home() / ".local" / "bin" / "modal"
        if p.is_file():
            return str(p)
        raise SystemExit("未找到 modal CLI — pip/uv install modal 并 token set")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="025 Kokoro-82M on Modal (default T4 · af_heart · v1)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    d.add_argument("--model", default=DEFAULT_MODEL, help="v1 | v1.1-zh")

    sm = sub.add_parser("smoke", help="EN af_heart · T4（或 --lang zh）")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--model", default=DEFAULT_MODEL)
    sm.add_argument("--voice", default=DEFAULT_VOICE)
    sm.add_argument("--lang", default="en", help="en | zh")
    sm.add_argument("--speed", type=float, default=1.0)
    sm.add_argument("--run-name", default="")

    t = sub.add_parser("t2s", help="text → speech")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--text", required=True)
    t.add_argument("--voice", default=DEFAULT_VOICE)
    t.add_argument("--lang", default="", help="override lang_code a/b/z/...")
    t.add_argument("--speed", type=float, default=1.0)
    t.add_argument("--run-name", default="")

    v = sub.add_parser("voices")
    v.add_argument("--model", default=DEFAULT_MODEL)

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_en_heart")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 025-kokoro")
        print(f"default_gpu: {DEFAULT_GPU}")
        print(f"default_model: {DEFAULT_MODEL}")
        print(f"default_voice: {DEFAULT_VOICE}")
        return _run([m, "run", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download":
        cmd = [
            m,
            "run",
            str(MODAL_APP),
            "--action",
            "download",
            "--model",
            ns.model,
        ]
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
            "--model",
            ns.model,
            "--voice",
            ns.voice,
            "--speed",
            str(ns.speed),
            "--smoke-lang",
            ns.lang,
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
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
            "--model",
            ns.model,
            "--text",
            ns.text,
            "--voice",
            ns.voice,
            "--speed",
            str(ns.speed),
        ]
        if ns.lang:
            cmd += ["--lang", ns.lang]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "voices":
        return _run(
            [m, "run", str(MODAL_APP), "--action", "voices", "--model", ns.model]
        )

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "pull":
        dest = Path(ns.dest)
        dest.mkdir(parents=True, exist_ok=True)
        return _run([m, "volume", "get", VOL_OUT, ns.remote, str(dest)])

    raise SystemExit(f"unknown {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
