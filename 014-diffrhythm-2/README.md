# 014 · DiffRhythm 2

ASLP-lab **DiffRhythm 2**（谛韵）— 半自回归扩散全曲生成。Apache 2.0。

| 项 | 值 |
|----|-----|
| 槽位 | **014**（原 MusicGen 计划已迁至 016） |
| 模型 | `ASLP-lab/DiffRhythm2` + MuQ-MuLan |
| 默认 GPU | **L4**（快且便宜） |
| smoke | 60s · 16 steps · text style |

## 快速开始

```bash
python run.py download
python run.py smoke                 # L4 · 60s
python run.py generate \
  --lyrics-file examples/smoke_en.lrc \
  --style "Pop, Piano, Bass, Drums, Happy" \
  --max-secs 120
```

## 与其它音乐号

```text
010 ACE · 011 SA3 · 012 LeVo · 013 YuE · 014 DiffRhythm2 · 016 MusicGen
```

## Gallery

[`gallery/index.html`](gallery/index.html)

## Smoke（L4 · 60s）

| 墙钟 | 生成 | 估费 | VRAM |
|------|------|------|------|
| **70.8 s** | 22.0 s | **~$0.016** | 7.7 GB |
