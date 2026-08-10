# Upstream

| 项 | 值 |
|---|---|
| 项目 | HY-World 2.0 · World Generation |
| 仓库 | https://github.com/Tencent-Hunyuan/HY-World-2.0 |
| 模块 | `hyworld2/worldgen` |
| 权重 | WorldStereo-2: https://huggingface.co/hanshanxue/WorldStereo |
| VLM | Qwen3-VL-8B（官方例；可换更小以省钱） |
| 官方环境 | CUDA 12.8 · Python 3.11+ · ≥4 GPU 推荐 |

## 流水线位置

```
008 panogen → 009 worldgen (1–5) → 3D world
007 worldrecon 既可独立，也可在 Stage4 被调用
```

## 与官方差异（规划）

- 默认少轨迹 / DMD 4-step / 1–2 卡优先
- VLM 尽量单卡小模型
- 3DGS `max_steps` 按 GPU 数比例缩放
- 不默认启动 8 卡 torchrun
