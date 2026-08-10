# 027 · Qwen3-TTS（TTS Tier S3）

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · Apache-2.0  
家族 HF 合计 ~**6.6M+** downloads · GH ~13k

| 项 | 值 |
|----|-----|
| 槽位 | **027**（见 [TTS_ROADMAP.md](../TTS_ROADMAP.md)） |
| 默认模型 | **1.7B-CustomVoice**（9 预设 + instruct） |
| 可选 | **0.6B-Custom** · **1.7B-Base 克隆** · **1.7B-VoiceDesign** |
| 默认 GPU | **L4** |
| 许可 | **Apache-2.0** |
| Image | `qwen-tts==0.1.1` · attn SDPA |

## 快速开始

```bash
cd 027-qwen3-tts

python run.py download                 # tokenizer + CustomVoice 1.7B
python run.py status

python run.py smoke --kind custom_zh   # Vivian 中文 + instruct
python run.py smoke --kind custom_en   # Ryan 英文
python run.py smoke --kind design      # 文本造声
python run.py smoke --kind clone       # Base 克隆（官方 demo ref）

python run.py t2s --text "你好世界" --speaker Vivian --lang Chinese
python run.py design --text "要抱抱！" --instruct "撒娇萝莉女声…"
python run.py clone --text "Hello from clone." --lang English

python run.py pull --remote runs/smoke_custom_zh_vivian
```

## Smoke 实测（2026-08-11 · L4 冷启动）

| run | model | gen_s | wall_s | est_usd | audio |
|-----|-------|------:|-------:|--------:|------:|
| smoke_custom_zh_vivian | custom_1.7 | 16.83 | 24.04 | **$0.0053** | 9.3s |
| smoke_custom_en_ryan | custom_1.7 | 13.22 | 18.79 | $0.0042 | 7.7s |
| smoke_design_zh | design_1.7 | 11.46 | 18.45 | $0.0041 | 5.4s |
| smoke_clone_en | base_1.7 | 19.73 | 26.37 | $0.0059 | 6.6s |

详见 [COST_BENCHMARK.md](COST_BENCHMARK.md) · 试听 [gallery/](gallery/index.html)。

## 变体

| key | HF | API |
|-----|-----|-----|
| `custom_1.7` | CustomVoice 1.7B | `generate_custom_voice` |
| `custom_0.6` | CustomVoice 0.6B | 同上 |
| `base_1.7` | Base 1.7B | `generate_voice_clone` |
| `design_1.7` | VoiceDesign 1.7B | `generate_voice_design` |

## Volume

| 名 | 用途 |
|----|------|
| `modal-lab-qwen3-tts-weights` | HF hub cache（~28.5 GB） |
| `modal-lab-qwen3-tts-prompts` | 克隆参考 wav（可选） |
| `modal-lab-qwen3-tts-outputs` | `runs/<name>/audio.wav` |

## 下一条

`028-fish-s2`
