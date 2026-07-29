#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""003-mineru 本地入口。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
DEFAULT_PDF = EXP_DIR.parent / "books" / "EN-算法导论4.pdf"
DATA_VOLUME = "modal-lab-mineru-data"


def modal_cli() -> str:
    executable = shutil.which("modal")
    if not executable:
        raise SystemExit("未找到 modal CLI，请先安装并登录 Modal")
    return executable


def call_modal(args: list[str]) -> int:
    command = [modal_cli(), *args]
    print("+", " ".join(command), flush=True)
    return subprocess.call(command, cwd=EXP_DIR)


def add_parse_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument(
        "--backend",
        choices=("hybrid-engine", "pipeline", "vlm-engine"),
        default="hybrid-engine",
    )
    parser.add_argument("--effort", choices=("medium", "high"), default="medium")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(description="MinerU Modal 实验")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status")
    sub.add_parser("download")

    benchmark = sub.add_parser("benchmark")
    add_parse_options(benchmark)
    benchmark.add_argument("--pages", type=int, default=100)

    parse = sub.add_parser("parse")
    add_parse_options(parse)

    pull = sub.add_parser("pull")
    pull.add_argument("--remote", default="/outputs")
    pull.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = parser.parse_args()
    command = ns.command or "status"
    if command == "status":
        print("experiment: 003-mineru")
        print(f"modal_app: {MODAL_APP}")
        print(f"default_pdf: {DEFAULT_PDF} exists={DEFAULT_PDF.is_file()}")
        print("app: modal-lab-mineru")
        print("version: MinerU 3.4.4")
        print("gpu: H100!")
        print("default: hybrid-engine / effort=medium")
        print("volumes:")
        print("  modal-lab-mineru-models")
        print("  modal-lab-mineru-data")
        return 0
    if command == "download":
        return call_modal(
            ["run", str(MODAL_APP), "--action", "download"]
        )
    if command in {"benchmark", "parse"}:
        args = [
            "run",
            "--timestamps",
            str(MODAL_APP),
            "--action",
            command,
            "--local-pdf",
            ns.pdf,
            "--backend",
            ns.backend,
            "--effort",
            ns.effort,
            "--start-page",
            str(ns.start_page),
            "--end-page",
            str(ns.end_page),
        ]
        if command == "benchmark":
            args.extend(["--benchmark-pages", str(ns.pages)])
        if ns.no_resume:
            args.append("--no-resume")
        return call_modal(args)
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
