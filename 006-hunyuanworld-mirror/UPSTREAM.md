# Upstream

| 项 | 值 |
|---|---|
| 项目 | HunyuanWorld-Mirror |
| 仓库 | https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror |
| 论文 | ICML 2026 |
| 权重 | https://huggingface.co/tencent/HunyuanWorld-Mirror |
| Demo | https://huggingface.co/spaces/tencent/HunyuanWorld-Mirror |

## 官方环境（摘录）

- CUDA 12.4
- Python 3.10
- `torch==2.4.0` / `torchvision==0.19.0` (cu124)
- `gsplat` via `https://docs.gsplat.studio/whl/pt24cu124`
- `pip install -r requirements.txt`（本实验只装推理子集，跳过 lightning 训练栈）

## 与本实验差异

- 不启 Gradio；不跑 training / evaluation
- smoke 默认 2 视图、不写 COLMAP、不渲染 3DGS 插值视频（省 GPU 秒数）
- 权重与输出全部进 Modal Volume，不入库
