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
DEFAULT_POLICY_HORIZON = 100
DEFAULT_EVAL_HORIZON = 200
DEFAULT_EVAL_LONG_HORIZON = 500
MINI_TASKS = (
    "OpenStandMixerHead,"
    "TurnOnElectricKettle,"
    "CloseFridge,"
    "TurnOnSinkFaucet,"
    "CloseBlenderLid"
)


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI：pip install modal && modal token set …")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="XR-1 RoboCasa365 sim on Modal")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    d = sub.add_parser("download-weights")
    d.add_argument("--force", action="store_true")
    a = sub.add_parser("download-assets")
    a.add_argument("--force", action="store_true")

    r = sub.add_parser("smoke-random")
    r.add_argument("--gpu", default=DEFAULT_GPU)
    r.add_argument("--task", default=DEFAULT_TASK)
    r.add_argument("--steps", type=int, default=80)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--split", default="pretrain")
    r.add_argument("--run-name", default="")

    pol = sub.add_parser("smoke-policy")
    pol.add_argument("--gpu", default=DEFAULT_GPU)
    pol.add_argument("--task", default=DEFAULT_TASK)
    pol.add_argument("--horizon", type=int, default=DEFAULT_POLICY_HORIZON)
    pol.add_argument("--seed", type=int, default=7)
    pol.add_argument("--split", default="pretrain")
    pol.add_argument("--run-name", default="")
    pol.add_argument("--attn", default="sdpa")

    sm = sub.add_parser("smoke")
    sm.add_argument("--gpu", default=DEFAULT_GPU)
    sm.add_argument("--task", default=DEFAULT_TASK)
    sm.add_argument("--horizon", type=int, default=DEFAULT_POLICY_HORIZON)
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--split", default="pretrain")
    sm.add_argument("--run-name", default="")
    sm.add_argument("--attn", default="sdpa")

    ev = sub.add_parser("eval-mini", help="5×5 mini-eval @ h=200 + CBL long @ h=500")
    ev.add_argument("--gpu", default=DEFAULT_GPU)
    ev.add_argument("--tasks", default=MINI_TASKS)
    ev.add_argument("--num-seeds", type=int, default=5)
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--horizon", type=int, default=DEFAULT_EVAL_HORIZON)
    ev.add_argument("--long-horizon", type=int, default=DEFAULT_EVAL_LONG_HORIZON)
    ev.add_argument("--long-task", default="CloseBlenderLid")
    ev.add_argument("--split", default="pretrain")
    ev.add_argument("--run-name", default="")
    ev.add_argument("--attn", default="sdpa")

    ls = sub.add_parser("ls")
    ls.add_argument("--path", default="runs")
    pl = sub.add_parser("pull")
    pl.add_argument("--remote", default="runs")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs"))

    ns = p.parse_args(argv)
    m = _modal()

    if ns.cmd == "status":
        print("experiment: 017-xr1-robocasa365-sim")
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
            m, "run", "--timestamps", str(MODAL_APP), "--action", "smoke-random",
            "--gpu", ns.gpu, "--task", ns.task, "--steps", str(ns.steps),
            "--seed", str(ns.seed), "--split", ns.split,
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd in {"smoke-policy", "smoke"}:
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP), "--action", "smoke-policy",
            "--gpu", ns.gpu, "--task", ns.task, "--horizon", str(ns.horizon),
            "--seed", str(ns.seed), "--split", ns.split, "--attn", ns.attn,
        ]
        if ns.run_name:
            cmd += ["--run-name", ns.run_name]
        return _run(cmd)

    if ns.cmd == "eval-mini":
        cmd = [
            m, "run", "--timestamps", str(MODAL_APP), "--action", "eval-mini",
            "--gpu", ns.gpu,
            "--tasks-csv", ns.tasks,
            "--num-seeds", str(ns.num_seeds),
            "--seed", str(ns.seed),
            "--horizon", str(ns.horizon),
            "--long-horizon", str(ns.long_horizon),
            "--long-task", ns.long_task,
            "--split", ns.split,
            "--attn", ns.attn,
            "--run-long-track",
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
