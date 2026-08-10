# Upstream — 034 Higgs Audio v2

| 项 | 值 |
|----|-----|
| Code | https://github.com/boson-ai/higgs-audio |
| Model | bosonai/higgs-audio-v2-generation-3B-base |
| Model rev (pinned) | `10840182ca4a`（nested `text_config`，兼容 github loader） |
| Tokenizer | bosonai/higgs-audio-v2-tokenizer |
| Tokenizer rev (pinned) | `9d4988fbd4ad`（`model.pth` + flat n_filters config） |
| API | `HiggsAudioServeEngine` |
| GPU | L40S（3B · ~15.7GB VRAM） |
| Note | HF main 已切 transformers-native flat config；本槽固定旧 rev 对齐 github |
| 收官 | **TTS 线终点 · 不做 035** |
