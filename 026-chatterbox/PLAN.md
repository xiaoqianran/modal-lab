# 026 · Chatterbox — 执行计划

对齐 [TTS_ROADMAP.md](../TTS_ROADMAP.md) **Tier S2**。

| Phase | 状态 | 内容 |
|-------|------|------|
| P0 脚手架 | ✅ | modal_app / run / docs |
| P1 prompts | ✅ | 20× Modal CDN voices → Volume |
| P2 download | ✅ | HF base + turbo → Volume (~35.8 GB) |
| P3 smoke mtl_en / mtl_zh | ✅ | L4 · wall ~65–69s · $0.014–0.015 |
| P4 smoke turbo | ✅ | Lucy + `[chuckle]` · wall 61.7s · $0.0137 |
| P5 gallery + COST | ✅ | assets + COST_BENCHMARK |

默认：**L4 · multilingual V3**。

## 下一步（可选 / 下一号）

- `original` 变体 smoke（CFG/exaggeration 扫）
- turbo `nano` 路径
- **027-qwen3-tts** · **028-fish-s2**
