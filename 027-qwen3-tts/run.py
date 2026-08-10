#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""027-qwen3-tts 本地入口（调度 Modal）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-qwen3-tts-outputs"
VOL_W = "modal-lab-qwen3-tts-weights"
VOL_P = "modal-lab-qwen3-tts-prompts"
DEFAULT_GPU = "L4"
DEFAULT_MODEL = "custom_1.7"
DEFAULT_SPEAKER = "Vivian"


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
        description="027 Qwen3-TTS on Modal (default L4 · CustomVoice 1.7B)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    d.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="custom_1.7|custom_0.6|base_1.7|design_1.7|all",
    )

    sm = sub.add_parser("smoke")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument(
        "--kind",
        default="custom_zh",
        choices=["custom_zh", "custom_en", "design", "clone"],
        help="smoke 场景",
    )
    sm.add_argument("--run-name", default="")

    t = sub.add_parser("t2s")
    t.add_argument("--gpu", default=DEFAULT_GPU)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--text", required=True)
    t.add_argument("--lang", default="Chinese")
    t.add_argument("--speaker", default=DEFAULT_SPEAKER)
    t.add_argument("--instruct", default="")
    t.add_argument("--ref-audio", default="", help="clone: path/URL")
    t.add_argument("--ref-text", default="")
    t.add_argument("--run-name", default="")

    de = sub.add_parser("design", help="VoiceDesign shortcut")
    de.add_argument("--gpu", default=DEFAULT_GPU)
    de.add_argument("--text", required=True)
    de.add_argument("--lang", default="Chinese")
    de.add_argument("--instruct", required=True)
    de.add_argument("--run-name", default="")

    cl = sub.add_parser("clone", help="Base voice-clone shortcut")
    cl.add_argument("--gpu", default=DEFAULT_GPU)
    cl.add_argument("--text", required=True)
    cl.add_argument("--lang", default="English")
    cl.add_argument("--ref-audio", default="")
    cl.add_argument("--ref-text", default="")
    cl.add_argument("--run-name", default="")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")

    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs/smoke_custom_zh_vivian")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 027-qwen3-tts")
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
            "--speaker",
            ns.speaker,
            "--instruct",
            ns.instruct,
        ]
        if ns.ref_audio:
            cmd += ["--ref-audio", ns.ref_audio]
        if ns.ref_text:
            cmd += ["--ref-text", ns.ref_text]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "design":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "design",
            "--gpu",
            ns.gpu,
            "--text",
            ns.text,
            "--lang",
            ns.lang,
            "--instruct",
            ns.instruct,
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "clone":
        cmd = [
            m,
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "clone",
            "--gpu",
            ns.gpu,
            "--text",
            ns.text,
            "--lang",
            ns.lang,
        ]
        if ns.ref_audio:
            cmd += ["--ref-audio", ns.ref_audio]
        if ns.ref_text:
            cmd += ["--ref-text", ns.ref_text]
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
