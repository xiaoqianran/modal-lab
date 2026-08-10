# 034 · Higgs Audio v2（TTS Tier A6 · **收官**）

bosonai/higgs-audio-v2-generation-3B-base · 默认 **L40S** · 场景描述/表现力

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_en | quiet room | 49.7s | **$0.0269** | 8.3s | 15.7G |
| smoke_expressive | excited | 21.1s | **$0.0114** | 9.9s | 15.7G |

```bash
cd 034-higgs
python run.py download
python run.py smoke --kind en
python run.py smoke --kind expressive
```

权重 pin（兼容 github loader）：model `10840182ca4a` · tokenizer `9d4988fbd4ad`。

**本号为 TTS 线终点。不做 035。**
