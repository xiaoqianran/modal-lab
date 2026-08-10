# Upstream

| 项 | 值 |
|---|---|
| 项目 | HY-World 2.0 · World Generation |
| 仓库 | https://github.com/Tencent-Hunyuan/HY-World-2.0 |
| 模块 | `hyworld2/worldgen` |
| VLM | **Qwen/Qwen3-VL-8B-Instruct** via vLLM（官方默认） |
| 权重 | WorldStereo-2: https://huggingface.co/hanshanxue/WorldStereo |
| 官方环境 | CUDA 12.8 · Python 3.11+ · ≥4 GPU 推荐（单卡可跑，砍规模） |

## 流水线位置

```
008 panogen → 009 worldgen (1–5) → 3D world
007 worldrecon 既可独立，也可在 Stage4 被调用
```

## Stage1/2 VLM（本仓库）

- 容器内 `vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000`
- 传给脚本：`--llm_addr 127.0.0.1 --llm_port 8000 --llm_name Qwen/Qwen3-VL-8B-Instruct`
- 同卡默认 `--gpu-memory-utilization 0.35–0.38` + `--max-model-len 8192` + `--enforce-eager`
- `stage12`：起服 → traj_generate → traj_render → 杀服

## 与官方差异

- 默认少轨迹 / DMD 4-step / 1 卡优先
- `apply_recon_iteration` 默认关
- Stage3 默认 SKIP_SAM3=1（室内）
- 3DGS `max_steps` 按 GPU 数比例缩放
- 不默认启动 8 卡 torchrun
