# 025 · Kokoro-82M（TTS 用量榜 #1）

[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · 推理库 [kokoro](https://github.com/hexgrad/kokoro)

| 项 | 值 |
|----|-----|
| 槽位 | **025** · TTS 线起点（见 [TTS_ROADMAP.md](../TTS_ROADMAP.md)） |
| 默认模型 | **v1** `hexgrad/Kokoro-82M` |
| 可选 | **v1.1-zh** 更好中文（100 speakers） |
| 默认音色 | **af_heart**（美式女声 · grade A） |
| 默认 GPU | **T4**（$0.000164/s · 82M 用不完） |
| 许可 | **Apache-2.0** |
| 采样率 | 24 kHz mono WAV |
| 排名 | HF TTS downloads **#1 ~11.5M** · AA open-weights Elo **~1056** |

## 实测 smoke（2026-08-11）

| run | GPU | 墙钟 | 生成 | VRAM | 估费 | 音频 |
|-----|-----|------|------|------|------|------|
| EN · af_heart | T4 | **7.44 s** | 1.97 s | **0.79 GB** | **$0.0012** | 11.1 s |
| ZH · zf_001 · v1.1-zh | T4 | **9.94 s** | 2.72 s | 0.79 GB | **$0.0016** | 11.0 s |

试听：[`gallery/index.html`](gallery/index.html)

## 快速开始

```bash
cd 025-kokoro
# 或: python ../main.py 025 status

python run.py status
python run.py download                 # CPU · v1 权重 → Volume
python run.py smoke                    # T4 · 英文 af_heart
python run.py smoke --lang zh          # 自动切 v1.1-zh · zf_001
python run.py t2s --text "Hello from modal-lab." --voice af_bella
python run.py voices
python run.py pull --remote runs/smoke_en_heart
```

换卡：`--gpu L4`（通常更贵没必要）。

## 远程产物

| Volume | 路径 |
|--------|------|
| `modal-lab-kokoro-weights` | `/weights/models/{v1,v1.1-zh}` + HF hub cache |
| `modal-lab-kokoro-outputs` | `runs/<name>/audio.wav` + `meta.json` |

## 模型

| key | HF repo | 用途 |
|-----|---------|------|
| `v1` | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 默认 · 54 音色 · 8 语 |
| `v1.1-zh` | [Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) | 中文加强 · 会丢掉部分旧音色 |

语言代码（音色前缀）：`a` 美英 · `b` 英英 · `z` 中文 · `j` 日语 · …

## 成本

见 [COST_BENCHMARK.md](COST_BENCHMARK.md)。

## 许可

- 权重与代码：**Apache-2.0**
- 生成内容请自行合规。

## 下一条

按 [TTS_ROADMAP.md](../TTS_ROADMAP.md)：`026-chatterbox` → `027-qwen3-tts` → `028-fish-s2`。
