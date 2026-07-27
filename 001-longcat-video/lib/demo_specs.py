# -*- coding: utf-8 -*-
"""各 Demo 的官方脚本映射与真实 CLI 白名单（仅基于 run_demo_*.py 实读）。

基础 5 个 Demo 的 argparse 仅有:
  --context_parallel_size  int  default=1
  --checkpoint_dir         str  default=None
  --enable_compile         store_true

prompt / 分辨率 / 输入路径等均在官方脚本内硬编码，不是 CLI 参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 官方基础 Demo 真实支持的 CLI 参数（名 → 元数据）
OFFICIAL_CLI_SPECS: dict[str, dict[str, Any]] = {
    "context_parallel_size": {
        "kind": "value",  # --key=value 或 --key value
        "type": "int",
        "default": 1,
    },
    "checkpoint_dir": {
        "kind": "value",
        "type": "str",
        "default": None,
    },
    "enable_compile": {
        "kind": "store_true",  # 仅 --enable_compile，无否定长选项于官方
        "default": False,
    },
}

# 自定义 run_storyboard_longcat.py 额外 CLI（仅 storyboard demo）
STORYBOARD_CLI_SPECS: dict[str, dict[str, Any]] = {
    **OFFICIAL_CLI_SPECS,
    "storyboard": {"kind": "value", "type": "str", "default": "storyboards/your_name_shinkai.json"},
    "mode": {"kind": "value", "type": "str", "default": "long"},
    "num_frames": {"kind": "value", "type": "int", "default": 93},
    "num_cond_frames": {"kind": "value", "type": "int", "default": 13},
    "num_inference_steps": {"kind": "value", "type": "int", "default": 24},
    "guidance_scale": {"kind": "value", "type": "float", "default": 4.0},
    "seed": {"kind": "value", "type": "int", "default": 42},
    "output_prefix": {"kind": "value", "type": "str", "default": "output_storyboard"},
    "spatial_refine_only": {"kind": "store_true", "default": False},
    "skip_refine": {"kind": "store_true", "default": False},
}


def cli_specs_for_demo(demo: str) -> dict[str, dict[str, Any]]:
    if demo == "storyboard":
        return STORYBOARD_CLI_SPECS
    return OFFICIAL_CLI_SPECS


@dataclass(frozen=True)
class DemoSpec:
    """单个 Demo 的实验内规格。"""

    name: str
    script: str  # 相对 LongCat-Video 根
    # 该 demo 允许透传的官方参数名（子集）；未知参数应拒绝
    allowed_script_args: frozenset[str]
    # 容器内、相对 CODE_ROOT 的输入依赖（官方硬编码路径）；用于校验/说明
    required_assets: tuple[str, ...] = ()
    # 官方写到 cwd 的输出 glob
    output_globs: tuple[str, ...] = ("*.mp4",)
    description: str = ""


DEMO_SPECS: dict[str, DemoSpec] = {
    "t2v": DemoSpec(
        name="t2v",
        script="run_demo_text_to_video.py",
        allowed_script_args=frozenset(OFFICIAL_CLI_SPECS),
        required_assets=(),
        output_globs=(
            "output_t2v.mp4",
            "output_t2v_distill.mp4",
            "output_t2v_refine.mp4",
        ),
        description="Text-to-Video（官方内置 prompt；CLI 仅 3 参数）",
    ),
    "i2v": DemoSpec(
        name="i2v",
        script="run_demo_image_to_video.py",
        allowed_script_args=frozenset(OFFICIAL_CLI_SPECS),
        required_assets=("assets/girl.png",),
        output_globs=(
            "output_i2v.mp4",
            "output_i2v_distill.mp4",
            "output_i2v_refine.mp4",
        ),
        description="Image-to-Video（硬编码 assets/girl.png）",
    ),
    "continuation": DemoSpec(
        name="continuation",
        script="run_demo_video_continuation.py",
        allowed_script_args=frozenset(OFFICIAL_CLI_SPECS),
        required_assets=("assets/motorcycle.mp4",),
        output_globs=(
            "output_vc.mp4",
            "output_vc_distill.mp4",
            "output_vc_refine.mp4",
        ),
        description="Video-Continuation（硬编码 assets/motorcycle.mp4）",
    ),
    "long": DemoSpec(
        name="long",
        script="run_demo_long_video.py",
        allowed_script_args=frozenset(OFFICIAL_CLI_SPECS),
        required_assets=(),
        output_globs=("output_long_video_*.mp4", "output_longvideo_refine_*.mp4"),
        description="Long-Video（num_segments 等写死在脚本内）",
    ),
    "interactive": DemoSpec(
        name="interactive",
        script="run_demo_interactive_video.py",
        allowed_script_args=frozenset(OFFICIAL_CLI_SPECS),
        required_assets=(),
        output_globs=(
            "output_interactive_*.mp4",
            "output_interactive_refine_*.mp4",
        ),
        description="Interactive Video（prompt_list 写死在脚本内）",
    ),
    "storyboard": DemoSpec(
        name="storyboard",
        script="run_storyboard_longcat.py",
        allowed_script_args=frozenset(STORYBOARD_CLI_SPECS),
        required_assets=("storyboards/your_name_shinkai.json",),
        output_globs=("output_storyboard_*.mp4", "output_storyboard_refine_*.mp4"),
        description="自定义分镜长视频（JSON prompts + t2v/vc/refine）",
    ),
}


def list_demo_names() -> list[str]:
    return list(DEMO_SPECS.keys())


def get_demo(name: str) -> DemoSpec:
    if name not in DEMO_SPECS:
        known = ", ".join(list_demo_names())
        raise ValueError(f"未知 demo={name!r}；可选: {known}")
    return DEMO_SPECS[name]
