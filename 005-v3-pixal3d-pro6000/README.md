# 005-v3-pixal3d-pro6000 — Pixal3D on **RTX PRO 6000 (Blackwell sm_120)**

> **状态：方案讨论 / 预研 · 未实跑 GPU。** 完整分析见 [`SOLUTION.md`](SOLUTION.md)。

| | 005 | 005-v2 | **005-v3** |
|--|-----|--------|------------|
| GPU | H100 | L40S | **RTX-PRO-6000** |
| SM | 90 | 89 | **120** |
| 是否已出 GLB | ✅ | ✅ | ❌ 未跑 |
| 关键难点 | — | 重编扩展 | **换 torch/CUDA + 重编扩展** |

## 有没有方案？

**有。** 社区（含 Windows Comfy Trellis2 轮子）证明 sm_120 上 **flex_gemm / o_voxel / nvdiffrast… 可编译**。  
Modal Linux 上应对齐：

1. **torch ≥ 2.7 + cu128**（二进制含 sm_120）  
2. **CUDA toolkit ≥ 12.8**  
3. **`TORCH_CUDA_ARCH_LIST=12.0` 源码编全部扩展**  
4. **`ATTN_BACKEND=sdpa`**，先避开 flash_attn  

**不能**复用 005 HF demo 轮子，也 **不能** 直接装 Windows 的 PRO 6000 `.whl`。

## 和 L40S 比

- v2：torch2.6 不变，只编 sm_89 → **已便宜出片 (~$0.17)**  
- v3：整栈升级 + 编 sm_120 → **更贵、更脆**；仅当卡池绑死 PRO 6000 时值得

## 本目录

```text
SOLUTION.md   # 根因、社区路径、Plan A/B、验收、决策
README.md
PLAN.md
configs/default.yaml
```

**当前无 `modal_app` 推理入口**（避免误跑烧钱）。若开跑，先按 PLAN 做 B0–B2 探针。

## 建议

| 目标 | 选择 |
|------|------|
| 稳定出模型 | 005 H100 / **005-v2 L40S** |
| 探索 PRO 6000 | 读 SOLUTION → 分阶段 Plan A |
| 本机 Windows 创作 | 可看 PixWizardry Comfy 轮子（非本 lab） |
