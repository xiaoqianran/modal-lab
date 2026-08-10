#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""026-chatterbox 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-chatterbox-outputs"
VOL_W = "modal-lab-chatterbox-weights"
VOL_P = "modal-lab-chatterbox-prompts"
DEFAULT_GPU = "L4"
DEFAULT_MODEL = "multilingual"
DEFAULT_VOICE = "Lucy"
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


def _upload_prompts() -> int:
    """Push local inputs/voices/*.wav into Modal volume via remote function."""
    import base64
    import json
    import os
    import tempfile

    # Use modal run with a small helper by calling upload via python -c embedding
    # Simpler: modal volume put
    m = _modal()
    if not VOICES_DIR.is_dir():
        raise SystemExit(f"missing {VOICES_DIR}")
    wavs = sorted(VOICES_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wav in {VOICES_DIR}")
    # volume put each file
    rc = 0
    for w in wavs:
        r = _run([m, "volume", "put", VOL_P, str(w), w.name, "--force"])
        if r != 0:
            rc = r
    print(f"uploaded {len(wavs)} prompts → {VOL_P}", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="026 Chatterbox on Modal (default L4 · multilingual V3)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    d.add_argument("--model", default=DEFAULT_MODEL, help="multilingual|turbo|original")

    sub.add_parser("upload-prompts", help="push inputs/voices/*.wav → Volume")

    sm = sub.add_parser("smoke")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument(
        "--kind",
        default="mtl_en",
        choices=["mtl_en", "mtl_zh", "turbo"],
        help="smoke 场景",
    )
    sm.add_argument("--voice", default=DEFAULT_VOICE)
    sm.add_argument("--exaggeration", type=float, default=0.5)
    sm.add_argument("--cfg-weight", type=float, default=0.5)
    sm.add_argument("--run-name", default="")
    sm.add_argument("--nano", action="store_true")

    t = sub.add_parser("t2s")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--text", required=True)
    t.add_argument("--lang", default="en")
    t.add_argument("--voice", default="")
    t.add_argument("--audio-prompt", default="", help="remote path under /prompts or empty")
    t.add_argument("--exaggeration", type=float, default=0.5)
    t.add_argument("--cfg-weight", type=float, default=0.5)
    t.add_argument("--run-name", default="")
    t.add_argument("--nano", action="store_true")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_mtl_en")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 026-chatterbox")
        print(f"default_gpu: {DEFAULT_GPU}")
        print(f"default_model: {DEFAULT_MODEL}")
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

    if ns.cmd == "upload-prompts":
        return _upload_prompts()

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
            "--voice",
            ns.voice,
            "--exaggeration",
            str(ns.exaggeration),
            "--cfg-weight",
            str(ns.cfg_weight),
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        if ns.nano:
            cmd.append("--nano")
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
            "--lang",
            ns.lang,
            "--voice",
            ns.voice or "",
            "--exaggeration",
            str(ns.exaggeration),
            "--cfg-weight",
            str(ns.cfg_weight),
        ]
        if ns.audio_prompt:
            cmd += ["--audio-prompt-path", ns.audio_prompt]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        if ns.nano:
            cmd.append("--nano")
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
