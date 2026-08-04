# modal-lab

Modal 实验台：按 `NNN-topic` 编号做可复现实验（风格对齐 [LightningAI-Lab](../LightningAI-Lab) / [kaggle-lab](../kaggle-lab)）。

## 结构

```text
main.py                 # 入口，调度到 001 / 002 / …
001-longcat-video/      # 美团 LongCat-Video 视频生成复现
002-unlimited-ocr/      # 百度 Unlimited-OCR 文档解析
003-mineru/             # OpenDataLab MinerU 文档解析
004-minimax-h3/         # MiniMax H3 文生/图生视频（Comfy headless · PRO 6000）
```

命名约定：`NNN-topic`（序号 + 主题）。`python main.py 001 …` 等短号在唯一时可解析到对应目录。

## 环境

```bash
# 本机需已安装并登录 Modal
pip install modal
modal token new   # 或确保已配置 ~/.modal.toml

# 可选：HF 下载加速 / 门禁模型
export HF_TOKEN=...   # 若模型需鉴权
```

## 用法

```bash
# 列出 / 进入实验（短号或全名）
python main.py 001 status
python main.py 001-longcat-video status
python main.py 004 t2v --prompt "..."

# 也可直接进目录
cd 001-longcat-video && python run.py status
```

## 实验一览

| 目录 | 作用 |
|------|------|
| `001-longcat-video` | 复现 [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)（T2V / I2V / 续写等，跑在 Modal GPU） |
| `002-unlimited-ocr` | 用 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) 在 Modal 上逐页解析 PDF |
| `003-mineru` | 用 [MinerU](https://github.com/opendatalab/MinerU) 在 Modal 上解析 PDF，并与 002 对照 |
| `004-minimax-h3` | [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 量化包 + ComfyUI headless，单卡 PRO 6000 文生视频 |

H100、RTX PRO 6000 与 A100 的同书实测、费用和选卡结论见
[GPU_COMPARISON.md](GPU_COMPARISON.md)。

## 约定

- 每个实验：`run.py` + `README.md`；权重 / 输出走 Modal Volume，不入库
- GPU：`001-longcat-video` 与 `004-minimax-h3` 默认 **RTX-PRO-6000**（96GB）；改配置前先看各实验 README
- 上游代码可 vendoring 在实验目录内，便于对照官方 demo

远程：https://github.com/xiaoqianran/modal-lab
