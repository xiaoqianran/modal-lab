#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""017-xr1-robocasa365-sim 本地入口。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-xr1-robocasa365-sim-outputs"
DEFAULT_GPU = "L40S"
DEFAULT_TASK = "CloseBlenderLid"


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI：pip install modal && modal token set …")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="XR-1 RoboCasa365 sim smoke on Modal")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    d = sub.add_parser("download-weights", help="拉 XR-1 RoboCasa365 权重（可与 015 共用 Volume）")
    d.add_argument("--force", action="store_true")
    a = sub.add_parser("download-assets", help="拉 RoboCasa 厨房资产 ~10GB")
    a.add_argument("--force", action="store_true")

    r = sub.add_parser("smoke-random", help="随机策略 1 局 → mp4（验证仿真+视频）")
    r.add_argument("--gpu", default=DEFAULT_GPU)
    r.add_argument("--task", default=DEFAULT_TASK)
    r.add_argument("--steps", type=int, default=80)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--split", default="pretrain")
    r.add_argument("--run-name", default="")

    pol = sub.add_parser("smoke-policy", help="XR-1 闭环 1 局短 horizon → mp4")
    pol.add_argument("--gpu", default=DEFAULT_GPU)
    pol.add_argument("--task", default=DEFAULT_TASK)
    pol.add_argument("--horizon", type=int, default=20)
    pol.add_argument("--seed", type=int, default=7)
    pol.add_argument("--split", default="pretrain")
    pol.add_argument("--run-name", default="")
    pol.add_argument("--attn", default="sdpa")

    # alias
    sm = sub.add_parser("smoke", help="= smoke-policy")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--task", default=DEFAULT_TASK)
    sm.add_argument("--horizon", type=int, default=20)
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--split", default="pretrain")
    sm.add_argument("--run-name", default="")
    sm.add_argument("--attn", default="sdpa")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")
    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 017-xr1-robocasa365-sim")
        print("note: 016 is MusicGen — sim is 017")
        print(f"app: modal-lab-xr1-robocasa365-sim")
        print(f"outputs: {VOL_OUT}")
        print(f"default: task={DEFAULT_TASK} gpu={DEFAULT_GPU}")
        return _run([m, "run", "--timestamps", str(MODAL_APP), "--action", "status"])

    if ns.cmd == "download-weights":
        cmd = [m, "run", "--timestamps", str(MODAL_APP), "--action", "download-weights"]
        if ns.force:
            cmd.append("--force")
        return _run(cmd)

    if ns.cmd == "download-assets":
        cmd = [m, "run", "--timestamps", str(MODAL_APP), "--action", "download-assets"]
        if ns.force:
            cmd.append("--force")
        return _run(cmd)

    if ns.cmd == "smoke-random":
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP),
            "--action", "smoke-random",
            "--gpu", ns.gpu, "--task", ns.task,
            "--steps", str(ns.steps), "--seed", str(ns.seed),
            "--split", ns.split,
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd in {"smoke-policy", "smoke"}:
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP),
            "--action", "smoke-policy",
            "--gpu", ns.gpu, "--task", ns.task,
            "--horizon", str(ns.horizon), "--seed", str(ns.seed),
            "--split", ns.split, "--attn", ns.attn,
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "ls":
        return _run([m, "volume", "ls", VOL_OUT, ns.path])

    if ns.cmd == "pull":
        dest = Path(ns.dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        return _run([m, "volume", "get", VOL_OUT, ns.remote.lstrip("/"), str(dest)])

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
