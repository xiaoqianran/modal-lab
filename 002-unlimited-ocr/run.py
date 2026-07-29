#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""002-unlimited-ocr 本地入口。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
DEFAULT_PDF = EXP_DIR.parent / "books" / "EN-算法导论4.pdf"
DATA_VOLUME = "modal-lab-unlimited-ocr-data"


def modal_cli() -> str:
    executable = shutil.which("modal")
    if not executable:
        raise SystemExit("未找到 modal CLI，请先 pip install modal && modal token new")
    return executable


def call_modal(args: list[str], gpu: str | None = None) -> int:
    command = [modal_cli(), *args]
    print("+", " ".join(command), flush=True)
    env = os.environ.copy()
    if gpu:
        env["MODAL_LAB_GPU_TYPE"] = gpu
    return subprocess.call(command, cwd=EXP_DIR, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unlimited-OCR Modal 实验")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status")
    sub.add_parser("download")

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--pdf", default=str(DEFAULT_PDF))
    benchmark.add_argument("--seconds", type=int, default=300)
    benchmark.add_argument("--start-page", type=int, default=1)
    benchmark.add_argument("--dpi", type=int, default=200)
    benchmark.add_argument("--max-tokens", type=int, default=4096)
    benchmark.add_argument("--concurrencies", default="24")
    benchmark.add_argument("--gpu", default="H100!")

    parse = sub.add_parser("parse")
    parse.add_argument("--pdf", default=str(DEFAULT_PDF))
    parse.add_argument("--start-page", type=int, default=1)
    parse.add_argument("--end-page", type=int, default=0)
    parse.add_argument("--dpi", type=int, default=200)
    parse.add_argument("--max-tokens", type=int, default=4096)
    parse.add_argument("--concurrency", type=int, default=24)
    parse.add_argument("--retries", type=int, default=3)
    parse.add_argument("--no-resume", action="store_true")
    parse.add_argument("--gpu", default="H100!")

    pull = sub.add_parser("pull")
    pull.add_argument("--remote", default="/outputs")
    pull.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = parser.parse_args()
    command = ns.command or "status"
    if command == "status":
        print("experiment: 002-unlimited-ocr")
        print(f"modal_app: {MODAL_APP}")
        print(f"default_pdf: {DEFAULT_PDF} exists={DEFAULT_PDF.is_file()}")
        print("app: modal-lab-unlimited-ocr")
        print("volumes:")
        print("  modal-lab-unlimited-ocr-weights")
        print("  modal-lab-unlimited-ocr-data")
        print("gpu: H100! by default; override with --gpu")
        return 0
    if command == "download":
        return call_modal(
            ["run", str(MODAL_APP), "--action", "download"]
        )
    if command == "benchmark":
        return call_modal(
            [
                "run",
                str(MODAL_APP),
                "--action",
                "benchmark",
                "--local-pdf",
                ns.pdf,
                "--runtime-seconds",
                str(ns.seconds),
                "--start-page",
                str(ns.start_page),
                "--dpi",
                str(ns.dpi),
                "--max-tokens",
                str(ns.max_tokens),
                "--concurrencies",
                ns.concurrencies,
            ],
            gpu=ns.gpu,
        )
    if command == "parse":
        args = [
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            "parse",
            "--local-pdf",
            ns.pdf,
            "--start-page",
            str(ns.start_page),
            "--end-page",
            str(ns.end_page),
            "--dpi",
            str(ns.dpi),
            "--max-tokens",
            str(ns.max_tokens),
            "--concurrencies",
            str(ns.concurrency),
            "--retries",
            str(ns.retries),
        ]
        if ns.no_resume:
            args.append("--no-resume")
        return call_modal(args, gpu=ns.gpu)
    if command == "pull":
        destination = Path(ns.dest).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        return call_modal(
            [
                "volume",
                "get",
                DATA_VOLUME,
                ns.remote.lstrip("/"),
                str(destination),
            ]
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
