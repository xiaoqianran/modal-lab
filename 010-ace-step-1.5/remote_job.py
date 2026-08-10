#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container-side ACE-Step 1.5 worker (invoked via `uv run` inside Modal)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ACE-Step repo is the working project for uv run
REPO_DIR = Path(os.environ.get("ACESTEP_REPO_DIR", "/opt/ACE-Step-1.5"))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_example(path: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    # payload overrides example fields
    for k in (
        "caption",
        "lyrics",
        "bpm",
        "duration",
        "keyscale",
        "language",
        "timesignature",
        "instrumental",
        "think",
        "thinking",
        "seed",
        "inference_steps",
        "guidance_scale",
    ):
        if k in payload and payload[k] is not None and payload[k] != "":
            data[k] = payload[k]
    return data


def cmd_generate(payload: dict[str, Any]) -> dict[str, Any]:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    checkpoints = Path(
        os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/weights/checkpoints")
    )
    save_dir = Path(payload.get("save_dir") or "/outputs/runs/default")
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = payload.get("dit_model") or "acestep-v15-turbo"
    lm_model = payload.get("lm_model") or "acestep-5Hz-lm-1.7B"
    lm_backend = payload.get("lm_backend") or "pt"
    device = payload.get("device") or "cuda"
    init_lm = bool(payload.get("init_lm", False))
    offload = bool(payload.get("offload_to_cpu", False))

    ex = _load_example(payload.get("example_path"), payload)
    thinking = bool(
        payload.get("thinking", ex.get("thinking", ex.get("think", False)))
    )
    # thinking requires LM
    if thinking:
        init_lm = True

    t_init0 = time.time()
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=str(REPO_DIR),
        config_path=config_path,
        device=device,
        offload_to_cpu=offload,
    )
    if not success:
        return {
            "success": False,
            "error": f"DiT init failed: {status_msg}",
            "status_message": status_msg,
        }
    dit_init_s = time.time() - t_init0

    llm_handler = LLMHandler()
    lm_init_s = 0.0
    lm_status = "skipped"
    if init_lm:
        t_lm0 = time.time()
        lm_status, lm_ok = llm_handler.initialize(
            checkpoint_dir=str(checkpoints),
            lm_model_path=lm_model,
            backend=lm_backend,
            device=device,
            offload_to_cpu=offload,
            dtype=None,
        )
        lm_init_s = time.time() - t_lm0
        if not lm_ok:
            return {
                "success": False,
                "error": f"LM init failed: {lm_status}",
                "status_message": lm_status,
                "dit_init_s": round(dit_init_s, 2),
            }

    duration = ex.get("duration", payload.get("duration", 20))
    if duration is None:
        duration = 20
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 20.0

    seed = payload.get("seed", ex.get("seed", 42))
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 42

    steps = payload.get("inference_steps", ex.get("inference_steps", 8))
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = 8

    guidance = payload.get("guidance_scale", ex.get("guidance_scale", 1.0))
    try:
        guidance = float(guidance)
    except (TypeError, ValueError):
        guidance = 1.0

    instrumental = bool(ex.get("instrumental", payload.get("instrumental", False)))
    lyrics = ex.get("lyrics", payload.get("lyrics", ""))
    if instrumental and not lyrics:
        lyrics = "[Instrumental]"

    params = GenerationParams(
        task_type="text2music",
        thinking=thinking,
        caption=str(ex.get("caption", payload.get("caption", "")) or ""),
        lyrics=str(lyrics or ""),
        instrumental=instrumental,
        bpm=ex.get("bpm", payload.get("bpm")),
        keyscale=str(ex.get("keyscale", payload.get("keyscale", "")) or ""),
        timesignature=str(
            ex.get("timesignature", payload.get("timesignature", "")) or ""
        ),
        vocal_language=str(
            ex.get("language", payload.get("language", "en")) or "en"
        ),
        duration=duration,
        inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
        use_cot_metas=bool(payload.get("use_cot_metas", thinking)),
        use_cot_caption=bool(payload.get("use_cot_caption", False)),
        use_cot_lyrics=bool(payload.get("use_cot_lyrics", False)),
        use_cot_language=bool(payload.get("use_cot_language", thinking)),
    )

    audio_format = payload.get("audio_format") or "flac"
    config = GenerationConfig(
        batch_size=int(payload.get("batch_size") or 1),
        audio_format=audio_format,
        use_random_seed=(seed < 0),
        seeds=[seed] if seed >= 0 else None,
    )

    t_gen0 = time.time()
    result = generate_music(
        dit_handler,
        llm_handler,
        params=params,
        config=config,
        save_dir=str(save_dir),
    )
    gen_s = time.time() - t_gen0

    audios_out: list[dict[str, Any]] = []
    if result.success and result.audios:
        for a in result.audios:
            p = a.get("path")
            size = None
            if p and os.path.isfile(p):
                size = os.path.getsize(p)
            audios_out.append(
                {
                    "path": p,
                    "key": a.get("key"),
                    "sample_rate": a.get("sample_rate"),
                    "size_bytes": size,
                }
            )

    extra = result.extra_outputs or {}
    time_costs = extra.get("time_costs") if isinstance(extra, dict) else None

    return {
        "success": bool(result.success),
        "status_message": result.status_message,
        "error": result.error,
        "audios": audios_out,
        "save_dir": str(save_dir),
        "dit_model": config_path,
        "lm_model": lm_model if init_lm else None,
        "lm_backend": lm_backend if init_lm else None,
        "thinking": thinking,
        "duration_req": duration,
        "seed": seed,
        "inference_steps": steps,
        "dit_init_s": round(dit_init_s, 2),
        "lm_init_s": round(lm_init_s, 2),
        "lm_status": lm_status,
        "generate_s": round(gen_s, 2),
        "time_costs": time_costs,
        "caption": params.caption[:200],
    }


def cmd_check_env(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    ckpt = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/weights/checkpoints"))
    components = [
        "acestep-v15-turbo",
        "vae",
        "Qwen3-Embedding-0.6B",
        "acestep-5Hz-lm-1.7B",
    ]
    comp_info = {}
    for name in components:
        p = ckpt / name
        files = list(p.rglob("*")) if p.is_dir() else []
        files = [f for f in files if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        comp_info[name] = {
            "exists": p.is_dir(),
            "files": len(files),
            "size_gb": round(size / 1e9, 3),
        }

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "mem_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            ),
            "cuda": torch.version.cuda,
        }

    return {
        "success": True,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "checkpoints_dir": str(ckpt),
        "components": comp_info,
        "repo_dir": str(REPO_DIR),
        "repo_exists": REPO_DIR.is_dir(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=["generate", "check-env"])
    ap.add_argument("--payload", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    result_path = Path(args.result)

    try:
        if args.action == "generate":
            out = cmd_generate(payload)
        else:
            out = cmd_check_env(payload)
    except Exception as e:  # noqa: BLE001
        out = {
            "success": False,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }

    _write_result(result_path, out)
    # also print for Modal logs
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return 0 if out.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
