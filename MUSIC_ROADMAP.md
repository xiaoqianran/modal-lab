# 音乐实验路线图

> 更新：2026-08-10 — **014 = DiffRhythm 2**；MusicGen 仍为 **016**。  
> 017 为仿真线，不占用音乐号。

## 编号

| 号 | 目录 | 模型 | 默认 GPU | 状态 |
|----|------|------|----------|------|
| **010** | `010-ace-step-1.5` | ACE-Step 1.5 | L4 | ✅ |
| **011** | `011-stable-audio-3` | Stable Audio 3 Medium | L4 | ✅ |
| **012** | `012-levo-2` | LeVo 2 v2-medium | L40S | ✅ |
| **013** | `013-yue` | YuE en-cot | L40S | ✅ smoke ~13min / $0.43 |
| **016** | `016-musicgen` | **MusicGen** small | **T4** | ✅ smoke OK · T4 ~$0.005 |
| **014** | `014-diffrhythm-2` | DiffRhythm 2 | **L4** | ✅ smoke 60s · ~$0.016 |
| 015 | robotics | Xiaomi | — | 非音乐 |
| 017 | sim | XR1 RoboCasa | — | 非音乐 |

```text
010 ACE → 011 SA3 → 012 LeVo2 → 013 YuE → 014 DiffRhythm2 → 016 MusicGen
```

## GPU 原则

| GPU | $/s | 用法 |
|-----|-----|------|
| **T4** | **0.000164** | MusicGen small 默认 |
| L4 | 0.000222 | SA3 / ACE / MusicGen medium |
| **L40S** | **0.000542** | 全曲大模型（LeVo / YuE） |
| PRO 6000 | 0.000842 | 大显存备选 |
