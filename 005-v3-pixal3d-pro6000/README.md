# 005-v3-pixal3d-pro6000 — Pixal3D × **RTX PRO 6000 (sm_120)**

> **第二轮方案已收紧 · 仍未实跑 GPU。** 详见 [`SOLUTION.md`](SOLUTION.md)。

## 一句话

**有方案，且 Linux 上已有同架构先例**（5090 跑通 TRELLIS.2 GLB；PRO 6000 有 Pixal3D 专用安装文档）。  
Modal 落地 = **torch 2.11+cu128 + `TORCH_CUDA_ARCH_LIST=12.0` 源码编扩展 + sdpa**，**不是** 005/v2 改一行 GPU。

## 黄金配方（Plan A*）

| 项 | 值 |
|----|-----|
| 证据 | animede/image-3d `requirements-pixal3d.txt`（**PRO 6000 实机**） |
| torch | **2.11.0+cu128** |
| arch | **12.0** |
| 扩展 | o-voxel(FlexGEMM/CuMesh) · **drtk** · natten 0.21 · (可选 nvdiffrast) |
| 注意力 | **sdpa only**（不装 flash_attn） |
| 编译 | CUDA **12.8** devel，**CUDA_HOME 勿漂到 13.x**，gcc **≤13** |

次要证据：TRELLIS.2#143 · RTX 5090 WSL · 同 sm_120 · 端到端 GLB。

## 和 L40S 比

| | v2 L40S | v3 PRO 6000 |
|--|---------|-------------|
| 状态 | ✅ 已出片 ~$0.17 | 🔒 方案可执行、未跑 |
| 难度 | 中（只重编扩展） | **高（换 torch+CUDA+重编）** |
| 建议 | **默认出片** | 卡池必须用再做 B0 探针 |

## 目录

- [`SOLUTION.md`](SOLUTION.md) — 配方 α/β/γ、门禁、风险  
- [`PLAN.md`](PLAN.md) — 阶段  
- **无 modal 推理入口**（防误烧钱）

## 若开跑

只允许：B0 认卡 → B1 o-voxel → B2 光栅 → B3 natten → B4 smoke。  
生产出片请用 [`../005-v2-pixal3d-l40s`](../005-v2-pixal3d-l40s) 或 [`../005-pixal3d`](../005-pixal3d)。
