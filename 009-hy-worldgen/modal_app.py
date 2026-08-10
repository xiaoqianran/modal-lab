# -*- coding: utf-8 -*-
"""
009-hy-worldgen — HY-World 2.0 World Generation (panorama → 3D world)

Stages (after 008 pano):
  1 traj_generate  2 traj_render  3 video_gen(WorldStereo)
  4 gen_gs_data    5 world_gs_trainer

Default: status / prepare only. Heavy stages require explicit --stage.
See PLAN.md for cost gates.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-hy-worldgen"
UPSTREAM = "https://github.com/Tencent-Hunyuan/HY-World-2.0"
WORLDSTEREO_HF = "hanshanxue/WorldStereo"

VOLUME_WEIGHTS = "modal-lab-hy-worldgen-weights"
VOLUME_OUTPUTS = "modal-lab-hy-worldgen-outputs"
VOLUME_PANO_OUT = "modal-lab-hy-pano-outputs"

WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
PANO_MOUNT = "/pano_out"

# Cheapest reasonable default for light stages; Stage3 may override.
DEFAULT_GPU = "RTX-PRO-6000"

EXP_DIR = Path(__file__).resolve().parent

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS, create_if_missing=True)
pano_vol = modal.Volume.from_name(VOLUME_PANO_OUT, create_if_missing=True)

app = modal.App(APP_NAME)

light_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("pillow", "huggingface_hub[hf_transfer]>=0.26.0")
    .env({"PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=light_image,
    volumes={OUTPUTS_MOUNT: outputs_vol, PANO_MOUNT: pano_vol},
    timeout=600,
    cpu=2,
    memory=4096,
)
def prepare_scene(
    from_008_run: str = "smoke_qwen",
    scene_name: str = "scene_from_008",
) -> dict[str, Any]:
    """Copy 008 panorama into a worldgen scene directory layout."""
    src_candidates = [
        Path(PANO_MOUNT) / "runs" / from_008_run / "panorama.png",
        Path(PANO_MOUNT) / "runs" / from_008_run / from_008_run / "panorama.png",
    ]
    src = next((p for p in src_candidates if p.is_file()), None)
    if src is None:
        # list available
        runs = Path(PANO_MOUNT) / "runs"
        avail = []
        if runs.is_dir():
            avail = sorted(p.name for p in runs.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"008 panorama not found under runs/{from_008_run}. available={avail}"
        )

    scene = Path(OUTPUTS_MOUNT) / "scenes" / scene_name
    scene.mkdir(parents=True, exist_ok=True)
    dst = scene / "panorama.png"
    shutil.copy2(src, dst)
    meta = {
        "ok": True,
        "scene": scene_name,
        "panorama": str(dst),
        "source_008_run": from_008_run,
        "source_path": str(src),
        "next": [
            "stage1 traj_generate (WorldNav + VLM)",
            "stage2 traj_render",
            "stage3 video_gen WorldStereo-2",
            "stage4 gen_gs_data",
            "stage5 world_gs_trainer → ply/spz/mesh",
        ],
        "cost_note": "prepare is CPU-only. Stage3 is the expensive step — confirm budget.",
    }
    (scene / "meta_prepare.json").write_text(json.dumps(meta, indent=2))
    outputs_vol.commit()
    return meta


@app.function(image=light_image, timeout=60)
def status() -> dict[str, Any]:
    return {
        "app": APP_NAME,
        "upstream": UPSTREAM,
        "component": "worldgen (stages 1–5 after HY-Pano)",
        "default_gpu": DEFAULT_GPU,
        "volumes": {
            "weights": VOLUME_WEIGHTS,
            "outputs": VOLUME_OUTPUTS,
            "pano_inputs": VOLUME_PANO_OUT,
        },
        "worldstereo_hf": WORLDSTEREO_HF,
        "pipeline": [
            "008 panogen → panorama",
            "1 traj_generate",
            "2 traj_render",
            "3 video_gen (WorldStereo ~17B)",
            "4 gen_gs_data",
            "5 world_gs_trainer → 3D world",
        ],
        "plan": "see PLAN.md — do not run full multi-GPU by default",
    }


@app.local_entrypoint()
def main(
    action: str = "status",
    from_008: str = "smoke_qwen",
    scene: str = "scene_from_008",
    stage: str | None = None,
    gpu: str = DEFAULT_GPU,
):
    action = action.lower().strip()
    if action == "status":
        print(json.dumps(status.remote(), indent=2))
        return
    if action == "prepare":
        print(json.dumps(prepare_scene.remote(from_008_run=from_008, scene_name=scene), indent=2))
        return
    if action == "stage":
        raise SystemExit(
            f"stage={stage} not implemented yet on Modal. "
            "Scaffold only — see PLAN.md. Explicit stage runners come after budget confirm."
        )
    raise SystemExit(f"unknown action: {action}. Use status|prepare|stage")
