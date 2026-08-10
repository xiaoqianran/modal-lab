# Upstream

| 项 | 值 |
|---|---|
| 项目 | HY-World 2.0 |
| 仓库 | https://github.com/Tencent-Hunyuan/HY-World-2.0 |
| 本实验组件 | **HY-Pano 2.0**（`hyworld2/panogen`） |
| 文档 | `hyworld2/panogen/README.md` · 根 `DOCUMENTATION.md` § Panorama |
| 官方环境 | CUDA 12.8 推荐 · Python 3.10/3.11 · torch 2.7.1 |

## 权重

| 后端 | HF | 体积（约） |
|---|---|---:|
| Full · HunyuanImage-3 | `tencent/HY-World-2.0` / `HY-Pano-2.0`（32 shards） | **~169GB** |
| Qwen LoRA | 同上目录 `pytorch_lora_weights.safetensors` | **~0.85GB** |
| Qwen base | `Qwen/Qwen-Image-Edit-2509` | **~58GB** |

## 与官方差异（本 lab）

- 默认只启用 **Qwen 后端**；全量 80B 需显式 `--backend full`
- 镜像只装 panogen 依赖，不装 worldgen / worldrecon 全树
- 权重与输出进 Modal Volume；大权重不入库
- 设备：见 [PLAN.md](PLAN.md)
