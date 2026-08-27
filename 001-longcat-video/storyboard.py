# -*- coding: utf-8 -*-
"""可配置分镜：

- mode=long（默认）: t2v + 多段 generate_vc + refine（长视频续写）
- mode=shots: 每条 prompt 独立 T2V（约 93 帧 @15fps ≈ 6s），适合多组短分镜
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from torchvision.io import write_video
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_storyboard(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts") or []
    mode = (data.get("mode") or "long").lower()
    if mode == "shots":
        if len(prompts) < 1:
            raise ValueError("mode=shots 需要至少 1 条 prompt")
    elif len(prompts) < 2:
        raise ValueError("mode=long 需要至少 2 条 prompt（t2v + 续写）")
    neg = data.get("negative_prompt") or (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )
    return {
        "title": data.get("title", "storyboard"),
        "prompts": prompts,
        "negative_prompt": neg,
        "mode": mode,
    }


def _init_dist_and_pipe(args):
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile

    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24)
    )
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir,
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        torch_dtype=torch.bfloat16,
    )

    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    global_seed = int(args.seed)
    seed = global_seed + global_rank
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    return local_rank, pipe, dit, generator, checkpoint_dir, enable_compile


def generate_shots(args, sb, pipe, dit, local_rank, generator, checkpoint_dir, enable_compile):
    """每条 prompt 独立 T2V ≈ 6s（num_frames/15）。"""
    prompt_list = sb["prompts"]
    negative_prompt = sb["negative_prompt"]
    num_frames = int(args.num_frames)
    steps = int(args.num_inference_steps)
    guidance = float(args.guidance_scale)
    skip_refine = bool(args.skip_refine)
    spatial_refine_only = bool(args.spatial_refine_only)
    out_prefix = args.output_prefix

    n = len(prompt_list)
    est = num_frames / 15.0
    if local_rank == 0:
        print(
            f"[shots] title={sb['title']!r} n={n} "
            f"each≈{est:.1f}s @15fps frames={num_frames} steps={steps}"
        )

    stage1_clips = []  # list of frame lists (PIL) for optional refine

    for i, prompt in enumerate(prompt_list):
        if local_rank == 0:
            print(f"[shot {i + 1}/{n}] T2V: {prompt[:140]}...")

        # 每镜固定 seed 偏移，便于复现且有变化
        generator.manual_seed(int(args.seed) + i * 17 + local_rank)

        output = pipe.generate_t2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )[0]

        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(output))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            path = f"{out_prefix}_shot{i + 1:02d}.mp4"
            write_video(
                path,
                output_tensor,
                fps=15,
                video_codec="libx264",
                options={"crf": "18"},
            )
            print(f"[shot {i + 1}/{n}] saved {path} ≈{num_frames / 15:.1f}s")

        frames = [
            PIL.Image.fromarray((output[j] * 255).astype(np.uint8))
            for j in range(output.shape[0])
        ]
        stage1_clips.append(frames)
        del output
        torch_gc()

    if skip_refine:
        if local_rank == 0:
            print("[shots] skip_refine=True，完成")
        return

    # 逐镜 refine（短，风险低）
    refinement_lora_path = os.path.join(
        checkpoint_dir, "lora/refinement_lora.safetensors"
    )
    pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
    pipe.dit.enable_loras(["refinement_lora"])
    pipe.dit.enable_bsa()
    if enable_compile:
        dit = torch.compile(dit)
    torch_gc()

    for i, frames in enumerate(stage1_clips):
        if local_rank == 0:
            print(f"[shot {i + 1}/{n}] refine...")
        generator.manual_seed(int(args.seed) + i * 17 + local_rank)
        output_refine = pipe.generate_refine(
            video=None,
            prompt="",
            stage1_video=frames,
            num_cond_frames=0,
            num_inference_steps=steps,
            generator=generator,
            spatial_refine_only=spatial_refine_only,
        )[0]
        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(output_refine))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            fps = 15 if spatial_refine_only else 30
            path = f"{out_prefix}_shot{i + 1:02d}_refine.mp4"
            write_video(
                path,
                output_tensor,
                fps=fps,
                video_codec="libx264",
                options={"crf": "10"},
            )
            print(f"[shot {i + 1}/{n}] saved {path} fps={fps}")
        del output_refine
        torch_gc()

    if local_rank == 0:
        print(f"[shots] all done n={n}")


def generate_long(args, sb, pipe, dit, local_rank, generator, checkpoint_dir, enable_compile):
    """原长视频：t2v + vc 链 + refine。"""
    prompt_list = sb["prompts"]
    negative_prompt = sb["negative_prompt"]
    title = sb["title"]

    num_segments = len(prompt_list) - 1
    num_frames = int(args.num_frames)
    num_cond_frames = int(args.num_cond_frames)
    spatial_refine_only = bool(args.spatial_refine_only)
    steps = int(args.num_inference_steps)
    guidance = float(args.guidance_scale)
    skip_refine = bool(args.skip_refine)
    out_prefix = args.output_prefix

    est_frames = num_frames + num_segments * (num_frames - num_cond_frames)
    est_sec = est_frames / 15.0
    print(
        f"[storyboard-long] title={title!r} prompts={len(prompt_list)} "
        f"vc_segments={num_segments} est_stage1≈{est_sec:.1f}s @15fps "
        f"frames≈{est_frames} steps={steps} skip_refine={skip_refine}"
    )

    if local_rank == 0:
        print(f"[seg 0/{num_segments}] T2V: {prompt_list[0][:120]}...")

    output = pipe.generate_t2v(
        prompt=prompt_list[0],
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )[0]

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(
            f"{out_prefix}_0.mp4",
            output_tensor,
            fps=15,
            video_codec="libx264",
            options={"crf": "18"},
        )

    video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]
    del output
    torch_gc()

    target_size = video[0].size
    current_video = video
    all_generated_frames = list(video)

    for segment_idx in range(num_segments):
        prompt = prompt_list[segment_idx + 1]
        if local_rank == 0:
            print(
                f"[seg {segment_idx + 1}/{num_segments}] VC: {prompt[:120]}..."
            )

        output = pipe.generate_vc(
            video=current_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution="480p",
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
        )[0]

        new_video = [
            (output[i] * 255).astype(np.uint8) for i in range(output.shape[0])
        ]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        new_video = [
            frame.resize(target_size, PIL.Image.BICUBIC) for frame in new_video
        ]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_generated_frames))
            write_video(
                f"{out_prefix}_{segment_idx + 1}.mp4",
                output_tensor,
                fps=15,
                video_codec="libx264",
                options={"crf": "18"},
            )
            del output_tensor
            print(
                f"[seg {segment_idx + 1}] cumulative_frames={len(all_generated_frames)} "
                f"≈{len(all_generated_frames)/15:.1f}s"
            )

    if skip_refine:
        if local_rank == 0:
            print("[storyboard-long] skip_refine=True")
        return

    refinement_lora_path = os.path.join(
        checkpoint_dir, "lora/refinement_lora.safetensors"
    )
    pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
    pipe.dit.enable_loras(["refinement_lora"])
    pipe.dit.enable_bsa()
    if enable_compile:
        dit = torch.compile(dit)
    torch_gc()

    cur_condition_video = None
    cur_num_cond_frames = 0
    start_id = 0
    all_refine_frames = []
    n_refine = num_segments + 1

    for segment_idx in range(n_refine):
        if local_rank == 0:
            print(f"[refine {segment_idx + 1}/{n_refine}]...")

        end_id = min(start_id + num_frames, len(all_generated_frames))
        stage1_chunk = all_generated_frames[start_id:end_id]
        if len(stage1_chunk) < 8:
            break

        output_refine = pipe.generate_refine(
            video=cur_condition_video,
            prompt="",
            stage1_video=stage1_chunk,
            num_cond_frames=cur_num_cond_frames,
            num_inference_steps=steps,
            generator=generator,
            spatial_refine_only=spatial_refine_only,
        )[0]

        new_video = [
            (output_refine[i] * 255).astype(np.uint8)
            for i in range(output_refine.shape[0])
        ]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output_refine

        all_refine_frames.extend(new_video[cur_num_cond_frames:])
        cur_condition_video = new_video
        cur_num_cond_frames = (
            num_cond_frames if spatial_refine_only else num_cond_frames * 2
        )
        start_id = start_id + num_frames - num_cond_frames

        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_refine_frames))
            fps = 15 if spatial_refine_only else 30
            write_video(
                f"{out_prefix}_refine_{segment_idx}.mp4",
                output_tensor,
                fps=fps,
                video_codec="libx264",
                options={"crf": "10"},
            )
            del output_tensor

    if local_rank == 0:
        fps = 15 if spatial_refine_only else 30
        print(
            f"[storyboard-long] done refine_frames={len(all_refine_frames)} "
            f"≈{len(all_refine_frames)/fps:.1f}s @{fps}fps"
        )


def generate(args):
    sb = load_storyboard(args.storyboard)
    # CLI 可覆盖 JSON mode
    mode = (args.mode or sb.get("mode") or "long").lower()
    sb["mode"] = mode

    local_rank, pipe, dit, generator, checkpoint_dir, enable_compile = (
        _init_dist_and_pipe(args)
    )

    if mode == "shots":
        generate_shots(
            args,
            sb,
            pipe,
            dit,
            local_rank,
            generator,
            checkpoint_dir,
            enable_compile,
        )
    else:
        generate_long(
            args,
            sb,
            pipe,
            dit,
            local_rank,
            generator,
            checkpoint_dir,
            enable_compile,
        )


def _parse_args():
    p = argparse.ArgumentParser(description="LongCat storyboard")
    p.add_argument("--context_parallel_size", type=int, default=1)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--enable_compile", action="store_true")
    p.add_argument(
        "--storyboard",
        type=str,
        default="storyboards/your_name_shinkai.json",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="",
        choices=["", "long", "shots"],
        help="空则用 JSON 内 mode；shots=独立短镜",
    )
    p.add_argument("--num_frames", type=int, default=93)
    p.add_argument("--num_cond_frames", type=int, default=13)
    p.add_argument("--num_inference_steps", type=int, default=24)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_prefix", type=str, default="output_storyboard")
    p.set_defaults(spatial_refine_only=True)
    refine = p.add_mutually_exclusive_group()
    refine.add_argument("--spatial_refine_only", action="store_true", dest="spatial_refine_only")
    refine.add_argument("--full_refine", action="store_false", dest="spatial_refine_only")
    p.add_argument("--skip_refine", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sp = Path(args.storyboard)
    if not sp.is_file():
        for cand in (
            Path("/root/LongCat-Video") / args.storyboard,
            Path("/root/LongCat-Video/storyboards") / sp.name,
            Path("/inputs") / sp.name,
            Path("/inputs/storyboards") / sp.name,
        ):
            if cand.is_file():
                args.storyboard = str(cand)
                break
    generate(args)
