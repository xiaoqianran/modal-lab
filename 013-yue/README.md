# 013 · YuE

M-A-P **YuE** 歌词→全曲（人声 + 伴奏）。Apache 2.0。

| 项 | 值 |
|----|-----|
| 槽位 | **013** |
| Stage1 | `m-a-p/YuE-s1-7B-anneal-en-cot` |
| Stage2 | `m-a-p/YuE-s2-1B-general` |
| Codec | `m-a-p/xcodec_mini_infer` |
| 默认 GPU | **L40S**（48GB） |
| smoke | 2 segments · max_new_tokens 3000 |

## 快速开始

```bash
python run.py download
python run.py smoke                 # L40S · 英文 CoT · 2 段
python run.py generate \
  --genre "inspiring female uplifting pop airy vocal" \
  --lyrics-file examples/smoke_lyrics.txt \
  --run-n-segments 2
```

## GPU

优先 **L40S**；OOM / 多段可 `--gpu RTX-PRO-6000` 或 `A100-80GB`。  
全曲 4+ segments 官方建议 80GB 级。

## 许可

Apache 2.0 · 请标注 “YuE by HKUST/M-A-P” 与 AI 生成。

## Gallery

打开 [`gallery/index.html`](gallery/index.html)。

## Smoke（L40S · 2 segments）

| 墙钟 | 估费 |
|------|------|
| **787 s** (~13.1 min) | **~$0.43** |
