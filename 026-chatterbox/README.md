# 026 · Chatterbox（TTS Tier S2）

[Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) · MIT  
Modal 官方例：[chatterbox_tts](https://modal.com/docs/examples/chatterbox_tts)

| 项 | 值 |
|----|-----|
| 槽位 | **026**（见 [TTS_ROADMAP.md](../TTS_ROADMAP.md)） |
| 默认模型 | **multilingual V3**（23 语 · 含中文） |
| 可选 | **turbo**（350M 英文 · 克隆 + `[chuckle]`）· **original** |
| 默认 GPU | **L4** |
| 许可 | **MIT** |
| 排名 | HF ~2.1M · GH ~26k · AA Elo ~1014 |
| Image | `chatterbox-tts==0.1.7` + peft |

## 快速开始

```bash
cd 026-chatterbox

python run.py upload-prompts   # Lucy.wav 等 → Volume（turbo 克隆用）
python run.py download         # HF 权重 → Volume
python run.py status

python run.py smoke --kind mtl_en      # 多语英文
python run.py smoke --kind mtl_zh      # 中文
python run.py smoke --kind turbo       # Turbo + Lucy 克隆 + [chuckle]

python run.py t2s --model multilingual --lang zh --text "你好，世界。"
python run.py t2s --model turbo --voice Lucy --text "Hello [laugh], how are you?"
python run.py pull --remote runs/smoke_mtl_en
```

## Smoke 实测（2026-08-11 · L4 冷启动）

| run | model | gen_s | wall_s | est_usd | audio |
|-----|-------|------:|-------:|--------:|------:|
| smoke_mtl_en | multilingual | 21.87 | 64.49 | $0.0143 | 5.6s |
| smoke_mtl_zh | multilingual | 28.65 | 68.98 | $0.0153 | 12.1s |
| smoke_turbo_lucy | turbo · Lucy | 27.54 | 61.70 | $0.0137 | 7.2s |

详见 [COST_BENCHMARK.md](COST_BENCHMARK.md) · 试听 [gallery/](gallery/index.html)。

## 变体

| key | 类 | 需要 ref？ | 语言 |
|-----|-----|-----------|------|
| `multilingual` | `ChatterboxMultilingualTTS` · t3=`v3` | 可选 | 23 语 |
| `turbo` | `ChatterboxTurboTTS` | **需要** `inputs/voices/*.wav` | 英文 |
| `original` | `ChatterboxTTS` | 可选 | 英文 · CFG/exaggeration |

## Volume

| 名 | 用途 |
|----|------|
| `modal-lab-chatterbox-weights` | HF hub cache（~35.8 GB） |
| `modal-lab-chatterbox-prompts` | 克隆参考 wav（20 条） |
| `modal-lab-chatterbox-outputs` | `runs/<name>/audio.wav` + `meta.json` |

## 下一条

`027-qwen3-tts` · `028-fish-s2`
