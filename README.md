# modal-lab

Modal 实验台：按 `NNN-topic` 编号做可复现实验。

## 结构

```text
main.py                 # 入口，调度到 001 / 002 / …
001-longcat-video/      # 美团 LongCat-Video 视频生成复现
002-unlimited-ocr/      # 百度 Unlimited-OCR 文档解析
003-mineru/             # OpenDataLab MinerU 文档解析
004-minimax-h3/         # MiniMax H3 文生/图生视频（Comfy headless · PRO 6000）
005-pixal3d/            # TencentARC Pixal3D 单图 → GLB（默认 H100）
006-hunyuanworld-mirror/# Tencent HunyuanWorld-Mirror 3D 重建（默认 L4 最低成本）
```

命名约定：`NNN-topic`（序号 + 主题）。`python main.py 001 …` 等短号在唯一时可解析到对应目录。

## 环境

```bash
pip install modal
modal token new   # 或确保已配置 ~/.modal.toml
export HF_TOKEN=...   # 可选：门禁模型
```

## 用法

```bash
python main.py 001 status
python main.py 004 t2v --prompt "..."
python main.py 005 i2v --image 005-pixal3d/inputs/sample.webp --gpu H100
python main.py 005 build-natten --gpu A100-40GB   # A100 首次
python main.py 006 smoke          # HunyuanWorld-Mirror 最低成本冒烟

# 也可直接进目录
cd 001-longcat-video && python run.py status
cd 005-pixal3d && python run.py smoke
```

## 实验一览

| 目录 | 作用 |
|------|------|
| `001-longcat-video` | 复现 [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)（T2V / I2V / 续写等，跑在 Modal GPU） |
| `002-unlimited-ocr` | 用 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) 在 Modal 上逐页解析 PDF |
| `003-mineru` | 用 [MinerU](https://github.com/opendatalab/MinerU) 在 Modal 上解析 PDF，并与 002 对照 |
| `004-minimax-h3` | [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 量化包 + ComfyUI headless，单卡 PRO 6000 文生视频 |
| `005-pixal3d` | [Pixal3D](https://github.com/TencentARC/Pixal3D) 图生 3D → GLB；**默认 H100**；A100-40GB 可选 |
| `006-hunyuanworld-mirror` | [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) 多视图 3D；**默认 L4 最低成本** |

H100 / PRO 6000 / A100 对照（OCR 等）见 [GPU_COMPARISON.md](GPU_COMPARISON.md)。  
Pixal3D 专项实测见 [005-pixal3d/GPU_BENCHMARK.md](005-pixal3d/GPU_BENCHMARK.md)。

## 约定

- 每个实验：`run.py` + `README.md`；权重 / 输出走 Modal Volume，不入库
- GPU：`001` / `004` 默认 **RTX-PRO-6000**；`005` 默认 **H100**（PRO 6000 / L40S 当前不可用）；`006` 默认 **L4**
- 上游代码可 vendoring，或镜像 build 时 clone（005 采用后者）

远程：https://github.com/xiaoqianran/modal-lab
