# 028 · Fish Audio S2 Pro（TTS Tier S4）

[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) · **Research License**  
代码：[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

| 项 | 值 |
|----|-----|
| 槽位 | **028**（见 [TTS_ROADMAP.md](../TTS_ROADMAP.md)） |
| 默认模型 | **S2-Pro 4B** Dual-AR |
| 默认 GPU | **L40S**（峰值 ~22GB VRAM） |
| 许可 | **Fish Audio Research**（非商用；商用需授权） |
| 排名 | AA Elo 开源 **#1 (1121)** · GH ~32k · HF ~428k |
| smoke 估费 | **~$0.05–0.07** / 次（L40S 冷启动） |

## 快速开始

```bash
cd 028-fish-s2

python run.py download
python run.py status

python run.py smoke --kind en      # 英文 · 随机音色
python run.py smoke --kind zh      # 中文
python run.py smoke --kind tags    # [excited] / [whisper]
python run.py smoke --kind clone   # 参考音克隆

python run.py t2s --text "Hello [laughing], how are you?"
python run.py pull --remote runs/smoke_en
```

## 能力

| 模式 | 说明 |
|------|------|
| 随机音色 | `references=[]` |
| 行内标签 | `[excited]` `[whisper]` `[chuckle]` 等 free-form |
| 克隆 | ref wav + 对齐 transcript |
| 多语 | 80+（中英日为核心） |

## Volume

| 名 | 用途 |
|----|------|
| `modal-lab-fish-s2-weights` | `checkpoints/s2-pro`（~11GB） |
| `modal-lab-fish-s2-prompts` | 克隆参考 wav |
| `modal-lab-fish-s2-outputs` | `runs/<name>/audio.wav` |

## Gallery

[`gallery/index.html`](gallery/index.html) · 实测见 [`COST_BENCHMARK.md`](COST_BENCHMARK.md)

## 许可脚注

本实验仅用于 **研究 / 评估**。部署到产品或收费服务前请取得 Fish Audio 商业许可。

## 下一条

**TTS Tier S 线收官。** 第二波：`029-voxcpm2` · `030-vibevoice` · `031-cosyvoice3` · …
