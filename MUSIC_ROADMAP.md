# 音乐实验路线图

> 更新：2026-08-10 — 用户确认 **MusicGen = 014**；**两个 012 起连续音乐号到 014**。  
> DiffRhythm 2 **延后**（不占号）。  
> GPU：优先 **L40S / PRO 6000** 性价比，不默认 A100。

## 编号

| 号 | 目录 | 模型 | 默认 GPU | 状态 |
|----|------|------|----------|------|
| **010** | `010-ace-step-1.5` | ACE-Step 1.5 | L4 | ✅ |
| **011** | `011-stable-audio-3` | Stable Audio 3 Medium | L4 | ✅ |
| **012** | `012-levo-2` | **LeVo 2** v2-medium | **L40S** | ✅ smoke OK · $0.086 |
| **013** | `013-yue` | YuE | L40S / PRO 6000 | 📋 planned |
| **014** | `014-musicgen` | MusicGen | T4 / L4 | 📋 planned |
| 015 | `015-xiaomi-robotics-1-…` | robotics | — | 非音乐 · 已占用 |
| — | DiffRhythm 2 | 延后 | — | 不占 012–014 |

```text
010 ACE → 011 SA3 → 012 LeVo2 → 013 YuE → 014 MusicGen
```

## GPU 原则

| GPU | $/s | 用法 |
|-----|-----|------|
| L4 | 0.000222 | 轻量（SA3 / ACE turbo / MusicGen） |
| **L40S** | **0.000542** | **全曲大模型默认**（比 A100-40 便宜，48GB） |
| A100-40GB | 0.000583 | 一般不优先 |
| PRO 6000 | 0.000842 | 大显存/更快备选 |
| A100-80GB | 0.000694 | 仅 OOM 升级 |

## 约定

每号：`modal_app.py` + `run.py` + volumes + smoke + `gallery/index.html`
