# 031 · CosyVoice3 成本实测（2026-08-11）

| run | model | 场景 | GPU | wall_s | load_s | gen_s | duration_s | vram_gb | est_usd | 状态 |
|-----|-------|------|-----|--------|--------|-------|------------|---------|---------|------|
| **smoke_zh** | cosyvoice3_0.5b | zero-shot 中文 | **L4** | **42.25** | 29.08 | **13.17** | 9.36 | **3.60** | **$0.0094** | ✅ |
| **smoke_dialect** | cosyvoice3_0.5b | instruct 四川话 | **L4** | **21.85** | 13.09 | **8.76** | 8.80 | **3.59** | **$0.0049** | ✅ |
| **smoke_en** | cosyvoice3_0.5b | cross-lingual EN | **L4** | **31.21** | 21.23 | **9.98** | 7.20 | **3.57** | **$0.0069** | ✅ |

单价：L4 `$0.000222/s`。权重 `Fun-CosyVoice3-0.5B-2512` · sample_rate 24 kHz · Apache-2.0。

## 对比（本 lab TTS 线）

| 实验 | 任务 | 默认 GPU | smoke 估费 | VRAM |
|------|------|----------|------------|------|
| **025 Kokoro** | 预设 | T4 | **~$0.001** | 0.8 GB |
| **030 VibeVoice RT** | EN/长句 | L4 | **~$0.005–0.006** | 2.8 GB |
| **031 CosyVoice3** | 中文/方言/EN | L4 | **~$0.005–0.009** | **3.6 GB** |
| **027 Qwen3-TTS** | 三模式 | L4 | **~$0.005** | 4.5 GB |
| **029 VoxCPM2** | en/design | L4 | **~$0.007** | 5.8 GB |
| **026 Chatterbox** | MTL/Turbo | L4 | **~$0.014** | 3.5 GB |
| **028 Fish S2** | 质量旗舰 | L40S | **~$0.06** | 22 GB |

→ CosyVoice3 **方言 instruct 很划算**（~$0.005）；首跑 zero-shot 含冷加载 ~$0.009。VRAM 3.6GB 可下探 T4。

## 备注

- 模式：`zero_shot` / `instruct2`（方言）/ `cross_lingual`（EN）。
- 参考音：官方 `zero_shot_prompt.wav`。
- 镜像：跳过 tensorrt/deepspeed；需 `openai-whisper` + setuptools。
- onnxruntime CUDA provider 缺 libcublasLt.so.11 会告警，CPU EP 仍可跑（不影响 smoke）。
