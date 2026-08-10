# 031 · CosyVoice3（TTS Tier A3）

Fun-CosyVoice3-0.5B · Apache-2.0 · 中文方言 SOTA · 默认 **L4**

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_zh | zero-shot 中文 | 42.3s | **$0.0094** | 9.4s | 3.6G |
| smoke_dialect | 四川话 instruct | 21.9s | **$0.0049** | 8.8s | 3.6G |
| smoke_en | cross-lingual EN | 31.2s | **$0.0069** | 7.2s | 3.6G |

```bash
cd 031-cosyvoice3
python run.py download
python run.py smoke --kind zh
python run.py smoke --kind dialect
python run.py smoke --kind en
python run.py pull --remote runs/smoke_zh --dest outputs
```

试听：[`gallery/`](gallery/) · 成本：[`COST_BENCHMARK.md`](COST_BENCHMARK.md) · 上游：[`UPSTREAM.md`](UPSTREAM.md)
