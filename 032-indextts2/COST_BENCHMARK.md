# 032 · IndexTTS-2 成本实测（2026-08-11）

| run | model | 场景 | GPU | wall_s | load_s | gen_s | duration_s | vram_gb | est_usd | 状态 |
|-----|-------|------|-----|--------|--------|-------|------------|---------|---------|------|
| **smoke_zh** | indextts2 | zero-shot 中文 | **L4** | **32.99** | 19.86 | **13.13** | 8.30 | **7.36** | **$0.0073** | ✅ |
| **smoke_en** | indextts2 | zero-shot EN | **L4** | **36.88** | 22.29 | **14.59** | 7.58 | **7.27** | **$0.0082** | ✅ |
| **smoke_emo** | indextts2 | emo_text 悲伤 | **L4** | **41.39** | 25.42 | **15.97** | 8.34 | **7.32** | **$0.0092** | ✅ |

单价：L4 `$0.000222/s`。fp16 · `use_cuda_kernel=False` · wetext 文本规范化。

## 对比

| 实验 | 默认 GPU | smoke 估费 | VRAM |
|------|----------|------------|------|
| 033 F5-TTS | L4 | **~$0.0025–0.003** | 2.1 GB |
| 030 VibeVoice RT | L4 | ~$0.005–0.006 | 2.8 GB |
| 031 CosyVoice3 | L4 | ~$0.005–0.009 | 3.6 GB |
| **032 IndexTTS-2** | L4 | **~$0.007–0.009** | **7.3 GB** |
| 028 Fish S2 | L40S | ~$0.06 | 22 GB |

→ IndexTTS-2 配音向；VRAM 7.3GB 仍吃得下 L4；情感/时长控制是差异点。

## 备注

- 许可：Bilibili IndexTTS（商用需注册）。
- Linux 默认 WeTextProcessing 换成 **wetext**（镜像 sed + runtime patch）。
- 参考音：`prompts/ref.wav`（Cosy 官方 zero_shot prompt）。
