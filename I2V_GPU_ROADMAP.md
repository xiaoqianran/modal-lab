# Image→3D · GPU 路线（L40S × PRO 6000）

> 更新：2026-08-11 — **022 Hunyuan3D-2.1** 双卡 shape+full 完成。

## 默认卡池

| GPU | Modal 请求 | 架构 | 单价 $/s | 角色 |
|-----|------------|------|----------|------|
| **L40S** | `L40S` | Ada · sm_89 | **0.000542** | 默认出片 / 更省 |
| **RTX PRO 6000** | `RTX-PRO-6000` | Blackwell SE · sm_120 | **0.000842** | 更快墙钟 / 96 GB |

原则：

1. 新 image→3D 实验默认双卡 smoke（先 L40S，再 PRO 6000）。
2. 扩展栈：L40S 用 cu124 + ARCH=8.9；PRO 用 torch2.11+cu128 + ARCH=12.0。
3. 不默认 H100，除非用户明确要 Hopper 对照。

## 已完成对照

| 实验 | 模型 | L40S | PRO 6000 | 角色 |
|------|------|------|----------|------|
| **020** | TripoSR | **13.7 s · $0.007** | **9.7 s · $0.008** | 速度 |
| **021** | TRELLIS.2-4B @512 | **215 s · $0.12** | **122 s · $0.10** | 质量 MIT |
| **005-v2/v3** | Pixal3D @1024 | **311 s · $0.17** | **230 s · $0.19** | 慢·对齐 |
| **022** | Hunyuan3D-2.1 full | **shape 30 + paint 65 · $0.13** | **shape 18 + paint 67 · $0.16** | PBR · Community |

本地统一预览：[`gpu-gallery/index.html`](gpu-gallery/)

## 结论（当前样本）

- **要快**：TripoSR · 两卡都够用；PRO 约快 29%。
- **要质量 / MIT**：TRELLIS.2 · PRO 约快 43% 且略更省（$0.10 vs $0.12）。
- **要对齐/重栈**：Pixal3D · PRO 更快但更贵；L40S 性价比更好。
- **要官方 PBR 线**：Hunyuan3D-2.1 · shape PRO 约 **1.6×** 快；paint 两卡接近；**注意 Community License**。

## 下一步

- [x] 022 probe + shape/full smoke 双卡
- [ ] TRELLIS.2 `pipeline_type=1024` 双卡对照
- [ ] 023 候选：InstantMesh / SF3D / SPAR3D
