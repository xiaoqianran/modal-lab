# 016 · MusicGen

Meta **MusicGen** text→instrumental 基线（HuggingFace `transformers`）。

| 项 | 值 |
|----|-----|
| 槽位 | **016**（占号 · 014 不再用） |
| 默认模型 | `facebook/musicgen-small` |
| 默认 GPU | **T4**（最便宜） |
| 可选 | medium @ L4 |
| 许可 | **CC-BY-NC 4.0**（非商用） |

## 快速开始

```bash
python run.py download
python run.py smoke                 # 15s lo-fi · T4
python run.py t2a --prompt "jazz piano trio, swinging" --duration 20
```

## Gallery

打开 [](gallery/index.html)。

## 与其它音乐号

```text
010 ACE-Step · 011 SA3 · 012 LeVo2 · 013 YuE(plan) · 016 MusicGen
```

017 起为仿真等其它线，不冲突。
