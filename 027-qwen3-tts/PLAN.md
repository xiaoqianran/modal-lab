# 027 · Qwen3-TTS — 执行计划

对齐 [TTS_ROADMAP.md](../TTS_ROADMAP.md) **Tier S3**。

| Phase | 状态 | 内容 |
|-------|------|------|
| P0 脚手架 | ✅ | modal_app / run / docs |
| P1 download | ✅ | tokenizer + Custom / Design / Base 1.7B（~28.5 GB） |
| P2 smoke custom_zh / custom_en | ✅ | Vivian / Ryan · L4 · $0.004–0.005 |
| P3 smoke design | ✅ | VoiceDesign · wall 18.5s · $0.0041 |
| P4 smoke clone | ✅ | Base + 官方 ref · wall 26.4s · $0.0059 |
| P5 gallery + COST | ✅ | assets + COST_BENCHMARK |

默认：**L4 · CustomVoice 1.7B · speaker Vivian**。

## 下一步（可选 / 下一号）

- custom_0.6 对照 smoke（可试 T4）
- **028-fish-s2**（Elo 开源 #1 · L40S）
