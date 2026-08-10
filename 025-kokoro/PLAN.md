# 025 · Kokoro — 执行计划

对齐 [TTS_ROADMAP.md](../TTS_ROADMAP.md) **Tier S1**。

| Phase | 状态 | 内容 |
|-------|------|------|
| P0 脚手架 | ✅ | modal_app / run / README / UPSTREAM / gallery |
| P1 download | ✅ | v1 + v1.1-zh → Volume（~0.36–0.38 GB 本地快照） |
| P2 smoke EN | ✅ | T4 · af_heart · **7.44s · $0.0012 · 0.79GB** |
| P3 smoke ZH | ✅ | v1.1-zh · zf_001 · **9.94s · $0.0016** |
| P4 gallery + COST | ✅ | assets 已 pull · COST_BENCHMARK 已填 |
| P5 t2s 多样音色 | 可选 | `python run.py t2s --voice af_bella --text "..."` |

默认：**T4 · v1 · af_heart** · 无 keep_warm · scaledown 30s。
