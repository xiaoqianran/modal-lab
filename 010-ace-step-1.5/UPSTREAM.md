# Upstream

| 项 | 值 |
|---|---|
| 项目 | ACE-Step 1.5 |
| 仓库 | https://github.com/ace-step/ACE-Step-1.5 |
| 固定 commit | `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`（镜像 build 时 shallow clone + checkout） |
| 论文 | https://arxiv.org/abs/2602.00744 |
| 权重（主包） | https://huggingface.co/ACE-Step/Ace-Step1.5 |
| 含组件 | `acestep-v15-turbo` · `vae` · `Qwen3-Embedding-0.6B` · `acestep-5Hz-lm-1.7B` |
| Demo | https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5 |
| 许可 | MIT |

## 官方环境（摘录）

- Python **3.11–3.12**
- CUDA **12.8**（`torch==2.10.0+cu128` on Linux x86_64）
- 安装：`uv sync`（本实验镜像同样用 `uv sync --frozen`）
- 默认 DiT：`acestep-v15-turbo`（8 steps，无需 CFG）
- 默认 LM：`acestep-5Hz-lm-1.7B`（主包内，`backend=pt` 更稳）

## 与本实验差异

- 不启 Gradio / REST API 常驻服务；一次性 `modal run` 生成
- 权重与音频全部进 Modal Volume，不入库
- smoke：20s 纯器乐、`thinking=False`（不加载 LM 推理路径），默认 **L4**
- t2m：可选 `thinking` + 主包 1.7B LM；仍默认 L4
