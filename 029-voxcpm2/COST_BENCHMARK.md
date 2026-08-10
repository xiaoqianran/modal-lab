# 029 · VoxCPM2 成本实测（2026-08-11）

| run | model | 场景 | GPU | wall_s | load_s | gen_s | duration_s | vram_gb | est_usd | 状态 |
|-----|-------|------|-----|--------|--------|-------|------------|---------|---------|------|
| **smoke_en** | voxcpm2 | en TTS | **L4** | **32.97** | 19.83 | **13.14** | 8.48 | **5.77** | **$0.0073** | ✅ |
| **smoke_zh** | voxcpm2 | zh TTS | **L4** | **32.03** | 18.15 | **13.87** | 9.28 | **5.79** | **$0.0071** | ✅ |
| **smoke_design** | voxcpm2 | 文本造声 `(desc)text` | **L4** | **34.78** | 25.22 | **9.55** | 4.00 | **5.63** | **$0.0077** | ✅ |
| **smoke_clone** | voxcpm2 | reference 克隆 | **L4** | **66.99** | 27.33 | **39.66** | 4.16 | **5.84** | **$0.0149** | ✅ |

单价：L4 `$0.000222/s`。`optimize=False` · `inference_timesteps=10` · `cfg_value=2.0` · 48 kHz。

## 对比（本 lab TTS 线）

| 实验 | 任务 | 默认 GPU | smoke 估费 | VRAM |
|------|------|----------|------------|------|
| **025 Kokoro** | 预设 ~11s | T4 | **~$0.001** | 0.8 GB |
| **027 Qwen3-TTS** | Custom/Design/Clone ~6–9s | L4 | **~$0.005** | 4.5 GB |
| **029 VoxCPM2** | en/zh/design ~4–9s | L4 | **~$0.007** | **~5.8 GB** |
| **026 Chatterbox** | MTL/Turbo ~6–12s | L4 | **~$0.014** | 3.5 GB |
| **028 Fish S2** | S2-Pro ~6–10s | L40S | **~$0.06–0.07** | **22 GB** |

→ 纯 TTS / design 冷启动约 **$0.007**（介于 Qwen3 与 Chatterbox 之间）；**克隆 gen 明显更慢**（本轮 40s vs TTS 10–14s），墙钟翻倍到 ~$0.015。VRAM **~5.8 GB** 落在 L4 舒适区，无需 L40S。

## 备注

- 权重 Volume ~**5.0 GB**（`openbmb/VoxCPM2` · model.safetensors + AudioVAE）。
- PyPI `voxcpm==2.0.3` 的 `generate` **无 seed 参数** → `inspect` 过滤 + `torch.manual_seed` 兜底。
- `load_denoiser=False`（跳过 ZipEnhancer）；smoke 未开 `optimize`/`torch.compile`。
- Voice design 语法：`(A young woman, gentle…)Hello…` 嵌在 text 里。
- Clone 用官方 `reference_speaker.wav`（prompts volume）；`reference_wav_path` 可控克隆。
- 输出 **48 kHz** mono；tokenizer 警告 `VoxCPM2Tokenizer` vs `LlamaTokenizerFast` 可忽略。
