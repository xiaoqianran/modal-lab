#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""001-longcat-video 本地 CLI —— 配置合并、校验、调度 Modal。

兼容:
  python run.py status|setup|download|smoke|t2v|i2v|continuation|long|interactive
  python run.py t2v --two-gpu --no-compile
  python run.py pull-outputs

配置优先级: CLI > 配置文件 > 默认值
透传官方参数（仅白名单）:
  python run.py t2v -- --context_parallel_size 1 --enable_compile
  python run.py t2v --script-arg context_parallel_size=2

资源:
  python run.py t2v --profile pro6000-2
  python run.py t2v --gpu RTX-PRO-6000 --nproc 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from lib.config import (  # noqa: E402
    ConfigError,
    DEFAULT_CONFIG_PATH,
    RESOURCE_PROFILES,
    config_summary,
    merge_run_config,
    parse_script_passthrough,
    print_summary,
    run_config_to_modal_payload,
    validate_local_prereqs,
)
from lib.demo_specs import DEMO_SPECS, list_demo_names  # noqa: E402

UPSTREAM = EXP_DIR / "LongCat-Video"
UPSTREAM_URL = "https://github.com/meituan-longcat/LongCat-Video.git"
MODAL_APP = EXP_DIR / "modal_app.py"
LOCAL_OUTPUTS = EXP_DIR / "outputs"
DEMOS = tuple(list_demo_names())


def _which_modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit(
            "未找到 modal CLI。请: pip install modal && modal token new"
        )
    return m


def _run(cmd: list[str], **kwargs) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kwargs)


def cmd_setup(_: argparse.Namespace) -> int:
    if UPSTREAM.is_dir() and (UPSTREAM / "longcat_video").is_dir():
        print(f"[setup] 上游已存在: {UPSTREAM}")
        return 0
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            "main",
            UPSTREAM_URL,
            str(UPSTREAM),
        ]
    )


def cmd_status(ns: argparse.Namespace) -> int:
    print("experiment: 001-longcat-video")
    print(f"  exp_dir:   {EXP_DIR}")
    print(f"  upstream:  {UPSTREAM}  exists={UPSTREAM.is_dir()}")
    print(f"  modal_app: {MODAL_APP}  exists={MODAL_APP.is_file()}")
    print(f"  outputs:   {LOCAL_OUTPUTS}")
    print(f"  inputs:    {EXP_DIR / 'inputs'}")
    print(f"  default_config: {DEFAULT_CONFIG_PATH} exists={DEFAULT_CONFIG_PATH.is_file()}")
    print()
    print("上游: https://github.com/meituan-longcat/LongCat-Video")
    print("权重: https://huggingface.co/meituan-longcat/LongCat-Video  (~83GB)")
    print("Volume: modal-lab-longcat-weights / modal-lab-longcat-outputs")
    print(f"资源档位: {', '.join(RESOURCE_PROFILES)}")
    print(f"Demos: {', '.join(DEMOS)}")
    print()
    print("官方 CLI（基础 demo 仅 3 项）: --checkpoint_dir --context_parallel_size --enable_compile")
    print("prompt/分辨率等写死在官方脚本内，不可伪造透传。")
    print()
    print("常用:")
    print("  python run.py setup && python run.py download && python run.py smoke")
    print("  python run.py t2v")
    print("  python run.py t2v --two-gpu --no-compile")
    print("  python run.py t2v --profile pro6000-1 -- --context_parallel_size 1 --enable_compile")
    print("  python run.py pull-outputs")
    # 可选：打印默认合并摘要
    if getattr(ns, "show_config", False):
        try:
            cfg = merge_run_config(command="status")
            print_summary(cfg)
        except ConfigError as e:
            print(f"[config] {e}")
    return 0


def _cli_infra_from_ns(ns: argparse.Namespace) -> dict:
    d = {}
    if getattr(ns, "profile", None):
        d["profile"] = ns.profile
    if getattr(ns, "gpu", None):
        d["gpu"] = ns.gpu
    if getattr(ns, "nproc", None) is not None:
        d["nproc"] = ns.nproc
    if getattr(ns, "cpu", None) is not None:
        d["cpu"] = ns.cpu
    if getattr(ns, "memory_mb", None) is not None:
        d["memory_mb"] = ns.memory_mb
    if getattr(ns, "timeout_s", None) is not None:
        d["timeout_s"] = ns.timeout_s
    if getattr(ns, "hf_repo", None):
        d["hf_repo"] = ns.hf_repo
    return d


def _collect_script_args(ns: argparse.Namespace, *, demo: str | None = None) -> dict:
    out: dict = {}
    # --script-arg key=value
    for item in getattr(ns, "script_arg", None) or []:
        if "=" not in item:
            raise ConfigError(f"--script-arg 需要 key=value，收到 {item!r}")
        k, v = item.split("=", 1)
        k = k.replace("-", "_")
        if k in (
            "enable_compile",
            "spatial_refine_only",
            "skip_refine",
        ):
            out[k] = v.lower() in ("1", "true", "yes", "on")
        elif k in (
            "context_parallel_size",
            "num_frames",
            "num_cond_frames",
            "num_inference_steps",
            "seed",
        ):
            out[k] = int(v)
        elif k == "guidance_scale":
            out[k] = float(v)
        else:
            out[k] = v
    # -- 之后
    passthrough = getattr(ns, "passthrough", None) or []
    if passthrough:
        out.update(parse_script_passthrough(list(passthrough), demo=demo))
    return out


def cmd_download(ns: argparse.Namespace) -> int:
    try:
        cfg = merge_run_config(
            command="download",
            config_path=Path(ns.config) if ns.config else None,
            cli_infra=_cli_infra_from_ns(ns),
            force_download=bool(ns.force),
        )
    except ConfigError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    print_summary(cfg)
    modal = _which_modal()
    cmd = [
        modal,
        "run",
        str(MODAL_APP),
        "--action",
        "download",
        "--hf-repo",
        cfg.infra.hf_repo,
    ]
    if cfg.infra.force_download:
        cmd.append("--force-download")
    return _run(cmd, cwd=str(EXP_DIR))


def cmd_smoke(ns: argparse.Namespace) -> int:
    try:
        cfg = merge_run_config(
            command="smoke",
            config_path=Path(ns.config) if ns.config else None,
            cli_infra=_cli_infra_from_ns(ns),
        )
        validate_local_prereqs(cfg, upstream=UPSTREAM)
    except ConfigError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    print_summary(cfg)
    modal = _which_modal()
    return _run(
        [modal, "run", str(MODAL_APP), "--action", "smoke"],
        cwd=str(EXP_DIR),
    )


def cmd_demo(ns: argparse.Namespace) -> int:
    demo = ns.command if ns.command in DEMOS else ns.demo
    try:
        script_cli = _collect_script_args(ns, demo=demo)
        # 长分镜默认用 pro6000-long（8h 超时），除非用户显式 --profile
        cli_infra = _cli_infra_from_ns(ns)
        if demo == "storyboard" and not cli_infra.get("profile"):
            cli_infra["profile"] = "pro6000-long"
        # storyboard 默认 spatial_refine_only + 24 步（更稳的 ~2min @15fps）
        if demo == "storyboard":
            script_cli.setdefault("spatial_refine_only", True)
            script_cli.setdefault("num_inference_steps", 24)
            script_cli.setdefault(
                "storyboard", "storyboards/your_name_shinkai.json"
            )
            script_cli.setdefault("output_prefix", "output_your_name_shinkai")
        cfg = merge_run_config(
            command="demo",
            demo=demo,
            config_path=Path(ns.config) if ns.config else None,
            cli_infra=cli_infra,
            cli_script_args=script_cli,
            cli_checkpoint_dir=getattr(ns, "checkpoint_dir", None),
            cli_output_subdir=getattr(ns, "output_subdir", None) or (
                "your_name_shinkai" if demo == "storyboard" else None
            ),
            two_gpu=bool(getattr(ns, "two_gpu", False)),
            no_compile=bool(getattr(ns, "no_compile", False)),
        )
        validate_local_prereqs(cfg, upstream=UPSTREAM)
    except ConfigError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    print_summary(cfg)
    payload = run_config_to_modal_payload(cfg)
    modal = _which_modal()

    cmd = [
        modal,
        "run",
        str(MODAL_APP),
        "--action",
        "demo",
        "--demo",
        str(cfg.demo),
        "--profile",
        cfg.infra.profile,
        "--gpu",
        cfg.resolved_gpu,
        "--nproc",
        str(cfg.resolved_nproc),
        "--cpu",
        str(cfg.resolved_cpu),
        "--memory-mb",
        str(cfg.resolved_memory_mb),
        "--timeout-s",
        str(cfg.resolved_timeout_s),
        "--script-argv-json",
        json.dumps(cfg.script_argv, ensure_ascii=False),
        "--output-subdir",
        str(cfg.output_subdir or cfg.demo),
    ]
    # enable_compile 已在 script_argv；仍传兼容字段
    if cfg.resolved_enable_compile:
        # Modal bool: 默认 True；用 flag 时注意
        pass
    else:
        cmd.append("--no-enable-compile")

    print("[payload]", json.dumps(payload, ensure_ascii=False), flush=True)
    return _run(cmd, cwd=str(EXP_DIR))


def cmd_pull_outputs(ns: argparse.Namespace) -> int:
    modal = _which_modal()
    LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    remote = ns.remote_path or "/"
    return _run(
        [
            modal,
            "volume",
            "get",
            "modal-lab-longcat-outputs",
            remote,
            str(LOCAL_OUTPUTS),
        ]
    )


def cmd_validate_config(ns: argparse.Namespace) -> int:
    """本地合并配置并打印，不提交 Modal。"""
    try:
        demo = ns.demo or "t2v"
        script_cli = _collect_script_args(ns, demo=demo)
        cli_infra = _cli_infra_from_ns(ns)
        if demo == "storyboard" and not cli_infra.get("profile"):
            cli_infra["profile"] = "pro6000-long"
        if demo == "storyboard":
            script_cli.setdefault("spatial_refine_only", True)
            script_cli.setdefault("num_inference_steps", 24)
            script_cli.setdefault(
                "storyboard", "storyboards/your_name_shinkai.json"
            )
        cfg = merge_run_config(
            command="demo",
            demo=demo,
            config_path=Path(ns.config) if ns.config else None,
            cli_infra=cli_infra,
            cli_script_args=script_cli,
            cli_checkpoint_dir=getattr(ns, "checkpoint_dir", None),
            two_gpu=bool(getattr(ns, "two_gpu", False)),
            no_compile=bool(getattr(ns, "no_compile", False)),
        )
        validate_local_prereqs(cfg, upstream=UPSTREAM)
    except ConfigError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    print_summary(cfg)
    print("[ok] 配置校验通过")
    return 0


def _add_shared_config_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default=None,
        help=f"配置文件路径（默认若存在则用 {DEFAULT_CONFIG_PATH}）",
    )
    p.add_argument("--profile", default=None, help="资源档位")
    p.add_argument("--gpu", default=None, help="覆盖 GPU 字符串，如 RTX-PRO-6000 或 A100-80GB:2")
    p.add_argument("--nproc", type=int, default=None, help="torchrun nproc_per_node")
    p.add_argument("--cpu", type=float, default=None, help="Modal CPU 核数")
    p.add_argument("--memory-mb", type=int, default=None, dest="memory_mb")
    p.add_argument("--timeout-s", type=int, default=None, dest="timeout_s")
    p.add_argument("--hf-repo", default=None, dest="hf_repo")


def _add_demo_flags(p: argparse.ArgumentParser) -> None:
    _add_shared_config_flags(p)
    p.add_argument(
        "--two-gpu",
        action="store_true",
        help="便捷: profile=pro6000-2 + nproc=2 + context_parallel_size=2",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="关闭 --enable_compile",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        dest="checkpoint_dir",
        help="官方 --checkpoint_dir（默认 /weights/LongCat-Video）",
    )
    p.add_argument(
        "--output-subdir",
        default=None,
        dest="output_subdir",
        help="Volume 内输出子目录（默认 demo 名）",
    )
    p.add_argument(
        "--script-arg",
        action="append",
        default=[],
        help="官方参数 key=value，可重复；如 context_parallel_size=1",
    )
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="在 -- 之后透传官方 CLI，如 -- --enable_compile",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="001-longcat-video — LongCat-Video on Modal（可配置）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py t2v
  python run.py t2v --two-gpu --no-compile
  python run.py t2v --profile pro6000-1 -- --context_parallel_size 1 --enable_compile
  python run.py validate-config --demo i2v
  python run.py download --force
        """,
    )
    sub = p.add_subparsers(dest="command", required=True)

    st = sub.add_parser("status", help="打印实验信息")
    st.add_argument("--show-config", action="store_true")

    sub.add_parser("setup", help="浅克隆上游（若缺失）")

    dl = sub.add_parser("download", help="下载权重到 Modal Volume")
    _add_shared_config_flags(dl)
    dl.add_argument("--force", action="store_true")

    sm = sub.add_parser("smoke", help="GPU / 依赖 / 权重冒烟")
    _add_shared_config_flags(sm)

    for name in DEMOS:
        d = sub.add_parser(name, help=f"官方 demo: {name}")
        _add_demo_flags(d)

    vc = sub.add_parser("validate-config", help="只合并/校验配置，不跑 Modal")
    vc.add_argument("--demo", default="t2v")
    _add_demo_flags(vc)

    pull = sub.add_parser("pull-outputs", help="拉取 Volume 输出到 ./outputs")
    pull.add_argument("--remote-path", default="/")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    # 规范化: 支持 `t2v -- --enable_compile` 时 REMAINDER 带前导 --
    parser = build_parser()
    try:
        ns = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    # REMAINDER 若以 -- 开头则去掉
    if getattr(ns, "passthrough", None):
        pt = list(ns.passthrough)
        if pt and pt[0] == "--":
            pt = pt[1:]
        ns.passthrough = pt

    try:
        if ns.command == "status":
            return cmd_status(ns)
        if ns.command == "setup":
            return cmd_setup(ns)
        if ns.command == "download":
            return cmd_download(ns)
        if ns.command == "smoke":
            return cmd_smoke(ns)
        if ns.command in DEMOS:
            return cmd_demo(ns)
        if ns.command == "validate-config":
            return cmd_validate_config(ns)
        if ns.command == "pull-outputs":
            return cmd_pull_outputs(ns)
    except ConfigError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    parser.error(f"unknown command {ns.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
