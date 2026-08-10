# 005-v2 L40S benchmark

## Smoke · 2026-08-10

| 项 | 值 |
|----|-----|
| GPU | **NVIDIA L40S** |
| SM | **sm_89** |
| 栈 | Plan A · torch 2.6.0+cu124 · 源码编 sm_89 轮子 |
| 模式 | low_vram · resolution 1024 · seed 42 |
| 样本图 | 官方 `5_img.webp` |
| 总耗时 | **310.7 s** |
| 峰值显存 | **15.42 GB** / 48 GB |
| 估费（仅 GPU） | **~$0.17**（$0.000542/s） |
| 输出 | `smoke_l40s.glb` · **38.3 MB** |
| Volume | `modal-lab-pixal3d-l40s-outputs` |

## 与 005 官版对照

| GPU | 栈 | 纯推理/总时 | 峰值 VRAM | 估费 | 状态 |
|-----|-----|-------------|-----------|------|------|
| H100 | 005 HF demo sm_90 | ~279 s | ~15.9 GB | ~$0.31 | ✅ 官版 |
| A100-40GB | 005 + natten rebuild | ~491 s | ~15.4 GB | ~$0.29 | ✅ 官版 |
| **L40S** | **v2 sm_89 全扩展** | **~311 s** | **~15.4 GB** | **~$0.17** | ✅ **已通** |

结论：L40S 在正确 sm_89 轮子下 **可出片**；本 smoke 总时长与 H100 接近，**单价更低**（首次 flex_gemm autotune 已计入，后续会略快）。

## 门禁结果

- `build-sm89`：nvdiffrast · nvdiffrec · flex_gemm · cumesh · o_voxel · natten 均成功  
- `verify`：capability=(8,9)，`HAS_LIBNATTEN=True`，RESULT PASS  
- `smoke`：采样 + 抽 GLB 全流程成功，无 `no kernel image`
