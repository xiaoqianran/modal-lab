# 029 · VoxCPM2（TTS Tier A1）

[OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) · **Apache-2.0**  
代码：[OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)

| 项 | 值 |
|----|-----|
| 槽位 | **029**（见 [TTS_ROADMAP.md](../TTS_ROADMAP.md)） |
| 默认模型 | **VoxCPM2 2B** tokenizer-free |
| 默认 GPU | **L4** |
| 许可 | **Apache-2.0** |
| 排名 | HF ~643k · GH ~35k · 速度/延迟向 |
| smoke | en / zh / design / clone **全绿** · 见 [COST](COST_BENCHMARK.md) |
| Gallery | [gallery/](gallery/index.html) · Pages: `/029-voxcpm2/` |

## 快速开始

```bash
cd 029-voxcpm2
python run.py download
python run.py smoke --kind en
python run.py smoke --kind zh
python run.py smoke --kind design
python run.py smoke --kind clone
python run.py t2s --text "Hello from VoxCPM2."
python run.py pull --remote runs/smoke_en --dest outputs/smoke_en
```

## 模式

| kind | 说明 | 本轮估费 |
|------|------|----------|
| en / zh | 纯 TTS（无 language_id） | ~$0.007 |
| design | `(voice description)text…` | ~$0.008 |
| clone | `reference_wav` 可控克隆 | ~$0.015 |

## Volume

| 名 | 用途 |
|----|------|
| `modal-lab-voxcpm2-weights` | 模型权重 ~5GB |
| `modal-lab-voxcpm2-prompts` | 克隆参考 |
| `modal-lab-voxcpm2-outputs` | runs |

## 成本速览

L4 冷启动：TTS **~$0.007** · clone **~$0.015** · VRAM **~5.8 GB** · 48 kHz。  
细节与横向对比见 [COST_BENCHMARK.md](COST_BENCHMARK.md)。
