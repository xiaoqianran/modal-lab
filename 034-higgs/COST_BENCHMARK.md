# 034 · Higgs Audio v2 成本实测（2026-08-11 · 收官）

| run | model | 场景 | GPU | wall_s | load_s | gen_s | duration_s | vram_gb | est_usd | 状态 |
|-----|-------|------|-----|--------|--------|-------|------------|---------|---------|------|
| **smoke_en** | higgs_v2_3b | quiet room EN | **L40S** | **49.65** | 43.42 | **6.23** | 8.32 | **15.66** | **$0.0269** | ✅ |
| **smoke_expressive** | higgs_v2_3b | excited scene | **L40S** | **21.06** | 14.01 | **7.05** | 9.88 | **15.68** | **$0.0114** | ✅ |

单价：L40S `$0.000542/s`。权重 pin：model `10840182ca4a` · tokenizer `9d4988fbd4ad`（兼容 github loader）。

## 对比（本 lab TTS 全线）

| 实验 | 默认 GPU | smoke 估费 | VRAM |
|------|----------|------------|------|
| 025 Kokoro | T4 | **~$0.001** | 0.8 GB |
| 033 F5-TTS | L4 | **~$0.0025** | 2.1 GB |
| 030 VibeVoice RT | L4 | ~$0.005–0.006 | 2.8 GB |
| 027 Qwen3-TTS | L4 | ~$0.005 | 4.5 GB |
| 031 CosyVoice3 | L4 | ~$0.005–0.009 | 3.6 GB |
| 032 IndexTTS-2 | L4 | ~$0.007–0.009 | 7.3 GB |
| 029 VoxCPM2 | L4 | ~$0.007 | 5.8 GB |
| 026 Chatterbox | L4 | ~$0.014 | 3.5 GB |
| **034 Higgs v2** | **L40S** | **~$0.011–0.027** | **15.7 GB** |
| 028 Fish S2 | L40S | ~$0.06 | 22 GB |

→ Higgs 冷启动贵（load ~43s）；热路径 gen 仅 ~6–7s。VRAM 15.7GB 必须 L40S 档。

## 备注

- Main 分支 HF 权重已切 transformers-native flat config；本槽 **pin 旧 rev** 对齐 `boson-ai/higgs-audio` GitHub API。
- 场景描述 system prompt + ChatML。
- **TTS 线到 034 收官，不做 035。**
