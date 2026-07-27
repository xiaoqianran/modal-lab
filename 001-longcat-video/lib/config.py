# -*- coding: utf-8 -*-
"""配置：默认值 / 文件 / CLI 合并、官方 argv 构建、校验与摘要。

优先级: CLI 显式 > 配置文件 > 项目默认值
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .demo_specs import (
    DEMO_SPECS,
    OFFICIAL_CLI_SPECS,
    STORYBOARD_CLI_SPECS,
    DemoSpec,
    cli_specs_for_demo,
    get_demo,
    list_demo_names,
)

EXP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = EXP_DIR / "configs" / "default.yaml"

# ---------- 资源档位（与 modal_app.RESOURCE_PROFILES 名称对齐）----------
RESOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "pro6000-1": {
        "gpu": "RTX-PRO-6000",
        "cpu": 4.0,
        "memory_mb": 32768,
        "timeout_s": 2 * 60 * 60,
        "nproc": 1,
    },
    "pro6000-2": {
        "gpu": "RTX-PRO-6000:2",
        "cpu": 8.0,
        "memory_mb": 65536,
        "timeout_s": 2 * 60 * 60,
        "nproc": 2,
    },
    "a100-80-1": {
        "gpu": "A100-80GB",
        "cpu": 4.0,
        "memory_mb": 32768,
        "timeout_s": 2 * 60 * 60,
        "nproc": 1,
    },
    "a100-80-2": {
        "gpu": "A100-80GB:2",
        "cpu": 8.0,
        "memory_mb": 65536,
        "timeout_s": 2 * 60 * 60,
        "nproc": 2,
    },
    "h100-1": {
        "gpu": "H100",
        "cpu": 4.0,
        "memory_mb": 32768,
        "timeout_s": 2 * 60 * 60,
        "nproc": 1,
    },
    # 长分镜 ~2min：多段 vc + refine，墙钟可达数小时
    "pro6000-long": {
        "gpu": "RTX-PRO-6000",
        "cpu": 8.0,
        "memory_mb": 65536,
        "timeout_s": 8 * 60 * 60,
        "nproc": 1,
    },
}

DEFAULT_PROFILE = "pro6000-1"

# Volume / 路径常量（与 modal_app 一致）
WEIGHTS_VOLUME = "modal-lab-longcat-weights"
OUTPUTS_VOLUME = "modal-lab-longcat-outputs"
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
CODE_ROOT = "/root/LongCat-Video"
CHECKPOINT_DIR_DEFAULT = f"{WEIGHTS_MOUNT}/LongCat-Video"
HF_REPO_DEFAULT = "meituan-longcat/LongCat-Video"

SENSITIVE_KEYS = frozenset(
    {
        "token",
        "secret",
        "password",
        "api_key",
        "apikey",
        "authorization",
        "hf_token",
        "hugging_face_hub_token",
    }
)


@dataclass
class InfraConfig:
    profile: str = DEFAULT_PROFILE
    gpu: str | None = None  # 覆盖 profile
    nproc: int | None = None
    cpu: float | None = None
    memory_mb: int | None = None
    timeout_s: int | None = None
    enable_compile: bool | None = None  # None=用 script 默认逻辑
    # 下载
    hf_repo: str = HF_REPO_DEFAULT
    force_download: bool = False


@dataclass
class RunConfig:
    """一次实验运行的完整生效配置。"""

    experiment: str = "001-longcat-video"
    command: str = "status"  # status|setup|download|smoke|demo|pull-outputs
    demo: str | None = None
    infra: InfraConfig = field(default_factory=InfraConfig)
    # 官方脚本参数（已合并）：仅允许的 key
    script_args: dict[str, Any] = field(default_factory=dict)
    # 透传后的 argv 列表（不含脚本名），供 torchrun 使用
    script_argv: list[str] = field(default_factory=list)
    # 路径
    checkpoint_dir: str = CHECKPOINT_DIR_DEFAULT
    output_subdir: str | None = None  # Volume 内子目录，默认 demo 名
    pull_remote_path: str = "/"
    config_file: str | None = None
    # 解析后的资源（填满后供 modal with_options）
    resolved_gpu: str = ""
    resolved_nproc: int = 1
    resolved_cpu: float = 4.0
    resolved_memory_mb: int = 32768
    resolved_timeout_s: int = 7200
    resolved_context_parallel_size: int = 1
    resolved_enable_compile: bool = True


class ConfigError(Exception):
    """配置 / 校验错误（本地应 SystemExit 友好信息）。"""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ConfigError(
                f"加载 YAML 需要 PyYAML: pip install pyyaml（文件: {path}）"
            ) from e
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件根节点必须是 mapping: {path}")
        return data
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件根节点必须是 object: {path}")
        return data
    raise ConfigError(f"不支持的配置格式 {suffix}（请用 .yaml/.yml/.json）: {path}")


def default_config_dict() -> dict[str, Any]:
    return {
        "infra": {
            "profile": DEFAULT_PROFILE,
            "gpu": None,
            "nproc": None,
            "cpu": None,
            "memory_mb": None,
            "timeout_s": None,
            "enable_compile": True,
            "hf_repo": HF_REPO_DEFAULT,
            "force_download": False,
        },
        "script_args": {
            # 官方默认 context_parallel_size=1；enable_compile 由 infra 便捷开关驱动
            "context_parallel_size": 1,
        },
        "checkpoint_dir": CHECKPOINT_DIR_DEFAULT,
        "output_subdir": None,
    }


def parse_script_passthrough(
    tokens: list[str], *, demo: str | None = None
) -> dict[str, Any]:
    """解析 `--` 之后的官方风格参数。

    支持:
      --context_parallel_size 2
      --context_parallel_size=2
      --enable_compile
      --checkpoint_dir=/weights/LongCat-Video
    """
    catalog = cli_specs_for_demo(demo) if demo else {**OFFICIAL_CLI_SPECS, **STORYBOARD_CLI_SPECS}
    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ConfigError(
                f"透传参数必须以 -- 开头: {tok!r}（完整: {tokens!r}）"
            )
        body = tok[2:]
        if "=" in body:
            key, raw = body.split("=", 1)
            key = key.replace("-", "_")
            out[key] = _coerce_official_value(key, raw, catalog=catalog)
            i += 1
            continue
        key = body.replace("-", "_")
        spec = catalog.get(key)
        if spec is None:
            raise ConfigError(
                f"未知参数 --{body}；当前允许: "
                f"{', '.join(sorted(catalog))}"
            )
        if spec["kind"] == "store_true":
            out[key] = True
            i += 1
            continue
        if i + 1 >= len(tokens):
            raise ConfigError(f"参数 --{body} 需要值")
        out[key] = _coerce_official_value(key, tokens[i + 1], catalog=catalog)
        i += 2
    return out


def _coerce_official_value(
    key: str, raw: str, *, catalog: dict[str, dict[str, Any]] | None = None
) -> Any:
    catalog = catalog or OFFICIAL_CLI_SPECS
    spec = catalog.get(key)
    if spec is None:
        raise ConfigError(f"未知参数: {key}")
    t = spec.get("type", "str")
    if t == "int":
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigError(f"--{key} 需要 int，收到 {raw!r}") from e
    if t == "float":
        try:
            return float(raw)
        except ValueError as e:
            raise ConfigError(f"--{key} 需要 float，收到 {raw!r}") from e
    return raw


def build_script_argv(
    script_args: dict[str, Any],
    *,
    demo: str,
    default_checkpoint: str,
) -> list[str]:
    """把合并后的 script_args 变成官方 CLI argv（保留语义）。"""
    dem = get_demo(demo)
    catalog = cli_specs_for_demo(demo)
    unknown = set(script_args) - set(dem.allowed_script_args)
    if unknown:
        raise ConfigError(
            f"demo={demo} 不支持参数: {sorted(unknown)}；"
            f"允许: {sorted(dem.allowed_script_args)}"
        )

    # checkpoint_dir：默认注入
    args = dict(script_args)
    if not args.get("checkpoint_dir"):
        args["checkpoint_dir"] = default_checkpoint

    # storyboard 默认
    if demo == "storyboard" and not args.get("storyboard"):
        args["storyboard"] = "storyboards/your_name_shinkai.json"

    argv: list[str] = []
    # 稳定顺序：通用在前，其余按名字
    order = [
        "checkpoint_dir",
        "context_parallel_size",
        "enable_compile",
        "storyboard",
        "mode",
        "num_frames",
        "num_cond_frames",
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "output_prefix",
        "spatial_refine_only",
        "skip_refine",
    ]
    seen: set[str] = set()
    for key in order:
        if key not in args:
            continue
        seen.add(key)
        val = args[key]
        meta = catalog.get(key) or OFFICIAL_CLI_SPECS.get(key)
        if not meta:
            continue
        flag = f"--{key}"
        if meta["kind"] == "store_true":
            if val:
                argv.append(flag)
            continue
        if val is None:
            continue
        argv.append(f"{flag}={val}")
    for key, val in args.items():
        if key in seen:
            continue
        meta = catalog.get(key)
        if not meta:
            continue
        flag = f"--{key}"
        if meta["kind"] == "store_true":
            if val:
                argv.append(flag)
        elif val is not None:
            argv.append(f"{flag}={val}")
    return argv


def resolve_resources(infra: InfraConfig) -> dict[str, Any]:
    profile = infra.profile or DEFAULT_PROFILE
    if profile not in RESOURCE_PROFILES:
        raise ConfigError(
            f"未知资源档位 profile={profile!r}；可选: "
            f"{', '.join(RESOURCE_PROFILES)}"
        )
    base = dict(RESOURCE_PROFILES[profile])
    gpu = infra.gpu if infra.gpu else base["gpu"]
    nproc = infra.nproc if infra.nproc is not None else int(base["nproc"])
    cpu = float(infra.cpu if infra.cpu is not None else base["cpu"])
    memory_mb = int(
        infra.memory_mb if infra.memory_mb is not None else base["memory_mb"]
    )
    timeout_s = int(
        infra.timeout_s if infra.timeout_s is not None else base["timeout_s"]
    )

    # gpu 字符串里的 :N 与 nproc 对齐检查
    if ":" in str(gpu):
        try:
            gpu_count = int(str(gpu).rsplit(":", 1)[-1])
        except ValueError:
            gpu_count = nproc
        if gpu_count != nproc:
            raise ConfigError(
                f"GPU 规格 {gpu!r} 的卡数 {gpu_count} 与 nproc={nproc} 不一致"
            )
    elif nproc > 1:
        # 自动拼 :N
        gpu = f"{gpu}:{nproc}"

    return {
        "profile": profile,
        "gpu": gpu,
        "nproc": nproc,
        "cpu": cpu,
        "memory_mb": memory_mb,
        "timeout_s": timeout_s,
    }


def merge_run_config(
    *,
    command: str,
    demo: str | None = None,
    config_path: Path | None = None,
    cli_infra: dict[str, Any] | None = None,
    cli_script_args: dict[str, Any] | None = None,
    cli_checkpoint_dir: str | None = None,
    cli_output_subdir: str | None = None,
    pull_remote_path: str = "/",
    two_gpu: bool = False,
    no_compile: bool = False,
    force_download: bool = False,
) -> RunConfig:
    """合并默认 → 文件 → CLI，得到 RunConfig。"""
    merged = default_config_dict()

    # 默认配置文件若存在则加载（可被 --config 覆盖路径）
    path = config_path
    if path is None and DEFAULT_CONFIG_PATH.is_file():
        path = DEFAULT_CONFIG_PATH
    file_data = load_config_file(path) if path else {}
    if file_data:
        merged = _deep_merge(merged, file_data)

    cli_infra = dict(cli_infra or {})
    if two_gpu:
        # 便捷：双卡档位（CLI 优先于文件里的 profile）
        cli_infra["profile"] = "pro6000-2"
        cli_infra.setdefault("nproc", 2)
    if no_compile:
        cli_infra["enable_compile"] = False
    if force_download:
        cli_infra["force_download"] = True

    # 去掉 None，避免覆盖
    cli_infra_clean = {k: v for k, v in cli_infra.items() if v is not None}
    if cli_infra_clean:
        merged["infra"] = _deep_merge(merged.get("infra") or {}, cli_infra_clean)

    script_merged = dict(merged.get("script_args") or {})
    if cli_script_args:
        script_merged.update(cli_script_args)

    infra_d = merged.get("infra") or {}
    infra = InfraConfig(
        profile=str(infra_d.get("profile") or DEFAULT_PROFILE),
        gpu=infra_d.get("gpu"),
        nproc=infra_d.get("nproc"),
        cpu=infra_d.get("cpu"),
        memory_mb=infra_d.get("memory_mb"),
        timeout_s=infra_d.get("timeout_s"),
        enable_compile=infra_d.get("enable_compile"),
        hf_repo=str(infra_d.get("hf_repo") or HF_REPO_DEFAULT),
        force_download=bool(infra_d.get("force_download", False)),
    )

    resources = resolve_resources(infra)

    # context_parallel_size：CLI 透传优先；否则与 nproc 对齐（避免 profile 多卡与 yaml 里 cp=1 冲突）
    explicit_cp = bool(cli_script_args and "context_parallel_size" in cli_script_args)
    if explicit_cp:
        cp = int(script_merged["context_parallel_size"])
        if cp != resources["nproc"]:
            raise ConfigError(
                f"context_parallel_size={cp} 与 nproc={resources['nproc']} "
                f"(profile/gpu={resources['gpu']!r}) 不一致。"
            )
    else:
        script_merged["context_parallel_size"] = resources["nproc"]

    # enable_compile：infra 便捷开关写入 script_args（除非 CLI 透传已设）
    explicit_compile = bool(cli_script_args and "enable_compile" in cli_script_args)
    if not explicit_compile and infra.enable_compile is not None:
        script_merged["enable_compile"] = bool(infra.enable_compile)

    checkpoint = (
        cli_checkpoint_dir
        or merged.get("checkpoint_dir")
        or CHECKPOINT_DIR_DEFAULT
    )
    if cli_checkpoint_dir:
        script_merged["checkpoint_dir"] = cli_checkpoint_dir
    elif not script_merged.get("checkpoint_dir"):
        script_merged["checkpoint_dir"] = checkpoint

    # 校验 demo
    if command in DEMO_SPECS or command == "demo":
        dname = demo or (command if command in DEMO_SPECS else None)
        if not dname:
            raise ConfigError("demo 命令需要指定 demo 名称")
        try:
            get_demo(dname)
        except ValueError as e:
            raise ConfigError(str(e)) from e
    else:
        dname = demo

    script_argv: list[str] = []
    if dname:
        script_argv = build_script_argv(
            script_merged,
            demo=dname,
            default_checkpoint=str(script_merged.get("checkpoint_dir") or checkpoint),
        )

    out_sub = cli_output_subdir or merged.get("output_subdir") or dname
    final_ckpt = str(script_merged.get("checkpoint_dir") or checkpoint)

    cfg = RunConfig(
        experiment="001-longcat-video",
        command=command if command not in DEMO_SPECS else "demo",
        demo=dname,
        infra=infra,
        script_args=script_merged,
        script_argv=script_argv,
        checkpoint_dir=final_ckpt,
        output_subdir=out_sub,
        pull_remote_path=pull_remote_path,
        config_file=str(path) if path else None,
        resolved_gpu=str(resources["gpu"]),
        resolved_nproc=int(resources["nproc"]),
        resolved_cpu=float(resources["cpu"]),
        resolved_memory_mb=int(resources["memory_mb"]),
        resolved_timeout_s=int(resources["timeout_s"]),
        resolved_context_parallel_size=int(
            script_merged.get("context_parallel_size") or resources["nproc"]
        ),
        resolved_enable_compile=bool(script_merged.get("enable_compile", False)),
    )
    return cfg


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in SENSITIVE_KEYS or any(
                s in str(k).lower() for s in ("token", "secret", "password")
            ):
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


def config_summary(cfg: RunConfig) -> dict[str, Any]:
    """运行前打印的生效配置摘要（无敏感信息）。"""
    demo_meta = None
    if cfg.demo:
        d = get_demo(cfg.demo)
        demo_meta = {
            "name": d.name,
            "script": d.script,
            "description": d.description,
            "required_assets": list(d.required_assets),
            "output_globs": list(d.output_globs),
        }
    summary = {
        "experiment": cfg.experiment,
        "command": cfg.command,
        "demo": cfg.demo,
        "demo_meta": demo_meta,
        "config_file": cfg.config_file,
        "infra": {
            "profile": cfg.infra.profile,
            "gpu": cfg.resolved_gpu,
            "nproc": cfg.resolved_nproc,
            "cpu": cfg.resolved_cpu,
            "memory_mb": cfg.resolved_memory_mb,
            "timeout_s": cfg.resolved_timeout_s,
            "enable_compile": cfg.resolved_enable_compile,
            "hf_repo": cfg.infra.hf_repo,
            "force_download": cfg.infra.force_download,
        },
        "paths": {
            "checkpoint_dir": cfg.checkpoint_dir,
            "code_root": CODE_ROOT,
            "weights_volume": WEIGHTS_VOLUME,
            "outputs_volume": OUTPUTS_VOLUME,
            "weights_mount": WEIGHTS_MOUNT,
            "outputs_mount": OUTPUTS_MOUNT,
            "output_subdir": cfg.output_subdir,
            "local_outputs": str(EXP_DIR / "outputs"),
            "local_inputs": str(EXP_DIR / "inputs"),
        },
        "script_args": cfg.script_args,
        "script_argv": cfg.script_argv,
        "torchrun_nproc": cfg.resolved_nproc,
    }
    return redact(summary)


def print_summary(cfg: RunConfig) -> None:
    print("======== 生效配置摘要 ========", flush=True)
    print(
        json.dumps(config_summary(cfg), ensure_ascii=False, indent=2),
        flush=True,
    )
    print("==============================", flush=True)


def validate_local_prereqs(cfg: RunConfig, *, upstream: Path) -> None:
    """本地侧前置校验。"""
    if not upstream.is_dir() or not (upstream / "longcat_video").is_dir():
        raise ConfigError(
            f"上游源码不存在或不完整: {upstream}\n请先: python run.py setup"
        )
    if cfg.demo:
        d = get_demo(cfg.demo)
        script = upstream / d.script
        if not script.is_file():
            raise ConfigError(f"官方脚本不存在: {script}")
        for rel in d.required_assets:
            asset = upstream / rel
            if not asset.is_file():
                raise ConfigError(
                    f"demo={cfg.demo} 需要资产文件（官方硬编码路径）: {asset}\n"
                    f"相对 CODE_ROOT: {rel}"
                )


def run_config_to_modal_payload(cfg: RunConfig) -> dict[str, Any]:
    """序列化给 modal local_entrypoint / 环境传递。"""
    return {
        "demo": cfg.demo,
        "script_argv": cfg.script_argv,
        "nproc": cfg.resolved_nproc,
        "gpu": cfg.resolved_gpu,
        "cpu": cfg.resolved_cpu,
        "memory_mb": cfg.resolved_memory_mb,
        "timeout_s": cfg.resolved_timeout_s,
        "profile": cfg.infra.profile,
        "checkpoint_dir": cfg.checkpoint_dir,
        "output_subdir": cfg.output_subdir,
        "enable_compile": cfg.resolved_enable_compile,
        "context_parallel_size": cfg.resolved_context_parallel_size,
        "hf_repo": cfg.infra.hf_repo,
        "force_download": cfg.infra.force_download,
    }
