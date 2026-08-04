# Upstream pins — 004-minimax-h3

| 组件 | 来源 | 版本 / 备注 |
|------|------|-------------|
| 模型权重 | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | pruned INT8 FL2VA + nvfp4 TE + VAEs（~42.5GB） |
| 原始权重（未用） | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 全量 BF16 FL2VA ~144GB，本实验默认不下 |
| 推理引擎 | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | **v0.30.0**（Day-0 H3 原生节点） |
| 官方 T2V 模板 | [workflow_templates `video_minimax_h3_t2v.json`](https://github.com/Comfy-Org/workflow_templates) | 子图展开为 API workflow |
| 文档 | [Comfy MiniMax H3 tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) | — |
| 博客 | [Comfy H3 Day-0](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui) | 42.5GB 变体 + Dynamic VRAM / 3060 可跑 |

## 默认模型文件

```
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

备用 TE（若 nvfp4 在目标 GPU 上出问题）：`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
