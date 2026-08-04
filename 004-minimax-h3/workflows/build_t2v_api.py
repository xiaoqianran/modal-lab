# -*- coding: utf-8 -*-
"""Build ComfyUI API-format workflow for MiniMax H3 T2V (no UI / no subgraph)."""

from __future__ import annotations

from typing import Any


def frame_length_from_seconds(seconds: float) -> int:
    """Match Comfy H3 grid: align to 17k+5 at 24fps (same as nodes_minimax_h3.align_frame_count)."""
    n = max(5, int(round(float(seconds) * 24)))
    while n % 17 != 5:
        n += 1
    return n


def build_t2v_prompt(
    *,
    prompt: str,
    width: int = 864,
    height: int = 480,
    seconds: float = 5.0,
    steps: int = 20,
    seed: int = 42,
    unet: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    clip: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    filename_prefix: str = "minimax_h3/t2v",
    fps: float = 24.0,
) -> dict[str, Any]:
    length = frame_length_from_seconds(seconds)
    # Node IDs match expanded official subgraph wiring (see UPSTREAM.md).
    return {
        "6": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "13": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip,
                "type": "minimax",
                "device": "default",
            },
        },
        "11": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae},
        },
        "24": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae},
        },
        "104": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["13", 0],
                "vae": ["11", 0],
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
                "length": int(length),
            },
        },
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["6", 0],
                "scheduler": scheduler,
                "steps": int(steps),
                "denoise": 1.0,
            },
        },
        "17": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": sampler},
        },
        "15": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(seed)},
        },
        "16": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["6", 0],
                "conditioning": ["104", 0],
            },
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["15", 0],
                "guider": ["16", 0],
                "sampler": ["17", 0],
                "sigmas": ["9", 0],
                "latent_image": ["104", 1],
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["14", 0],
                "vae": ["11", 0],
            },
        },
        "23": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["14", 0],
                "vae": ["24", 0],
            },
        },
        "91": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["10", 0],
                "audio": ["23", 0],
                "fps": float(fps),
                "bit_depth": 8,
            },
        },
        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["91", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }
