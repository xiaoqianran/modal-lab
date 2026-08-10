# 027 · Qwen3-TTS 成本实测（2026-08-11）

| run | model | speaker/mode | GPU | wall_s | load_s | gen_s | duration_s | vram_gb | est_usd | 状态 |
|-----|-------|--------------|-----|--------|--------|-------|------------|---------|---------|------|
| **smoke_custom_zh_vivian** | custom_1.7 | Vivian · zh · instruct | **L4** | **24.04** | 7.20 | **16.83** | 9.28 | **4.47** | **$0.0053** | ✅ |
| **smoke_custom_en_ryan** | custom_1.7 | Ryan · en · instruct | **L4** | **18.79** | 5.57 | **13.22** | 7.68 | **4.43** | **$0.0042** | ✅ |
| **smoke_design_zh** | design_1.7 | 文本造声 | **L4** | **18.45** | 6.99 | **11.46** | 5.36 | **4.35** | **$0.0041** | ✅ |
| **smoke_clone_en** | base_1.7 | 官方 ref 克隆 | **L4** | **26.37** | 6.64 | **19.73** | 6.56 | **4.70** | **$0.0059** | ✅ |

单价：L4 `$0.000222/s` · T4 `$0.000164/s`。attn = **SDPA**（未装 flash-attn）。

## 对比（本 lab TTS 线）

| 实验 | 任务 | 默认 GPU | smoke 估费 | VRAM |
|------|------|----------|------------|------|
| **025 Kokoro** | 预设音色 ~11s | T4 | **~$0.0012** | 0.79 GB |
| **026 Chatterbox** | 多语/克隆 5–12s | L4 | **~$0.014** | ~3.4 GB |
| **027 Qwen3-TTS** | 预设+instruct / 造声 / 克隆 | L4 | **~$0.004–0.006** | ~4.5 GB |

→ Qwen3 冷启动约 **$0.005**，比 Chatterbox 冷启动便宜约 **3×**，比 Kokoro 贵约 **4×**；VRAM ~4.5GB 仍落在 L4 舒适区。

## 备注

- 权重 Volume 合计 ~**28.5 GB**（tokenizer 1.4 + Custom 9.0 + Design 9.0 + Base 9.1）。
- 克隆 smoke 用官方 demo URL，无需本地 ref。
- `qwen-tts==0.1.1` 钉 `transformers==4.57.3`。
