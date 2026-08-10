# 032 · IndexTTS-2（TTS Tier A4）

IndexTeam/IndexTTS-2 · 时长+情感 · 默认 **L4** · fp16

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_zh | zero-shot 中文 | 33.0s | **$0.0073** | 8.3s | 7.4G |
| smoke_en | zero-shot EN | 36.9s | **$0.0082** | 7.6s | 7.3G |
| smoke_emo | emo_text 悲伤 | 41.4s | **$0.0092** | 8.3s | 7.3G |

```bash
cd 032-indextts2
python run.py seed-prompt
python run.py download
python run.py smoke --kind zh
python run.py smoke --kind en
python run.py smoke --kind emo
```

许可：Bilibili IndexTTS（商用注册）。见 [`COST_BENCHMARK.md`](COST_BENCHMARK.md) · [`gallery/`](gallery/)
