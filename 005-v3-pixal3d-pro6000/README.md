# 005-v3-pixal3d-pro6000 — Pixal3D on **RTX PRO 6000 (sm_120)**

> **状态：已在 Modal PRO 6000 端到端跑通**（smoke → GLB）。方案见 [`SOLUTION.md`](SOLUTION.md)，实测见 [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)。

| | 005 | 005-v2 | **005-v3** |
|--|-----|--------|------------|
| GPU | H100 | L40S | **RTX-PRO-6000** |
| SM | 90 | 89 | **120** |
| torch | 2.6 cu124 | 2.6 cu124 | **2.11.0+cu128** |
| 出 GLB | ✅ | ✅ | ✅ **~230s · ~$0.19** |

## 实测 smoke

| GPU | 时间 | VRAM | 估费 |
|-----|------|------|------|
| **PRO 6000** | **230 s** | **15.6 GB** | **~$0.19** |

本地：[`viewer/index.html`](viewer/index.html) + [`viewer/smoke_pro6000.glb`](viewer/smoke_pro6000.glb)

## 栈（Plan A*）

```text
镜像: nvidia/cuda:12.8.1-devel-ubuntu24.04
torch: 2.11.0+cu128
TORCH_CUDA_ARCH_LIST=12.0
NATTEN_CUDA_ARCH=12.0
ATTN_BACKEND=sdpa
扩展: nvdiffrast · nvdiffrec · flex_gemm · cumesh · o_voxel · drtk · natten
```

## 用法

```bash
python run.py probe
python run.py build-sm120 --i-know-this-costs-money
python run.py verify --i-know-this-costs-money
python run.py smoke --i-know-this-costs-money

modal volume get modal-lab-pixal3d-pro6000-outputs meshes/smoke_pro6000.glb ./viewer/
```

## 相关

- 官版 H100：[`005-pixal3d`](../005-pixal3d/)  
- L40S：[`005-v2-pixal3d-l40s`](../005-v2-pixal3d-l40s/)  
- 上游 MIT：见 [UPSTREAM.md](UPSTREAM.md)
