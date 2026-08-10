# 033 · F5-TTS（TTS Tier A5）

F5TTS_v1_Base · 零样本克隆 · 默认 **L4** · Code MIT / Model **CC-BY-NC**

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_en | clone EN | 12.0s | **$0.0027** | 8.4s | 2.1G |
| smoke_zh | clone ZH | 11.2s | **$0.0025** | 11.0s | 2.1G |

```bash
cd 033-f5tts
python run.py download
python run.py smoke --kind en
python run.py smoke --kind zh
```

本 lab **最便宜的零样本克隆**档之一。见 [`COST_BENCHMARK.md`](COST_BENCHMARK.md)
