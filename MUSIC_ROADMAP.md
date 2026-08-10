# 音乐实验路线图

> 审计更新：2026-08-10 — 六号全部 smoke 通过；**016 已清理误入的 017 仿真资产**。  
> 017 为仿真线，**禁止**再写入 `016-musicgen/`。

## 编号与实测

| 号 | 目录 | 模型 | 默认 GPU | 墙钟（smoke） | 估费 $ | 许可 | 状态 |
|----|------|------|----------|---------------|--------|------|------|
| **010** | `010-ace-step-1.5` | ACE-Step 1.5 turbo | L4 | ~33 s / 20s 器乐 | ~0.007 | 开源 | ✅ gallery+GPU bench |
| **011** | `011-stable-audio-3` | SA3 Medium | L4 | ~44 s / 20s | ~0.010 | Stability Community | ✅ gallery |
| **012** | `012-levo-2` | LeVo 2 v2-medium | L40S | ~159 s 全曲 | ~0.086 | 研究/非商用 | ✅ gallery |
| **013** | `013-yue` | YuE en-cot 7B+1B | L40S | ~787 s / 2seg | ~0.43 | Apache 2.0 | ✅ gallery stems |
| **014** | `014-diffrhythm-2` | DiffRhythm 2 | L4 | ~71 s / 60s 曲 | ~0.016 | Apache 2.0 | ✅ gallery |
| **016** | `016-musicgen` | MusicGen small | **T4** | ~32 s / 15s | ~0.005 | CC-BY-NC | ✅ cost bench |
| 015 | robotics | Xiaomi | — | — | — | — | 非音乐 |
| 017 | sim | XR1 RoboCasa | — | — | — | — | 非音乐 · 勿占 016 |

```text
010 ACE → 011 SA3 → 012 LeVo2 → 013 YuE → 014 DiffRhythm2 → 016 MusicGen
```

## 性价比排序（同类用途粗比）

| 场景 | 推荐 | 理由 |
|------|------|------|
| 最便宜短器乐 | **016 MusicGen · T4** | ~$0.004–0.005 / 次 |
| 快速全曲（词+曲） | **014 DiffRhythm2 · L4** | 60s 曲 ~$0.016 · 生成仅 ~22s |
| 高质量全曲（人声对齐） | **013 YuE · L40S** | 质量高但贵慢（~$0.43 / 2seg） |
| 全曲（研究许可） | **012 LeVo2 · L40S** | ~$0.09 · 研究许可 |
| 长器乐/氛围 | **011 SA3 · L4** | 长时频域控制 |
| 多 GPU 对照基线 | **010 ACE** | 已有 T4→H100 表 |

## GPU 单价（Modal 参考）

| GPU | $/s |
|-----|-----|
| T4 | 0.000164 |
| L4 | 0.000222 |
| L40S | 0.000542 |
| PRO 6000 | 0.000842 |
| A100-80 / H100 | 更高 · 音乐线非默认 |

## 审计检查清单（2026-08-10）

- [x] 六号均有 `modal_app.py` + `run.py` + `README` + `gallery`
- [x] Modal volumes 均存在且有权重
- [x] 本地 gallery 音频可播（非空）
- [x] 016 误混入的 CloseBlenderLid 仿真 mp4 **已删除**
- [x] 根 README 补齐 013 / 014
- [ ] 未做：生产环境二次 smoke 回归（权重仍在 volume，可 `python main.py N smoke`）
- [ ] 未做：012 v2-large / 013 4+ segments 全曲压力测试
