# 005-v2-pixal3d-l40s — Pixal3D on **L40S (Ada sm_89)**

> **状态：已在 Modal L40S 端到端跑通**（smoke → GLB）。方案见 [`SOLUTION.md`](SOLUTION.md)，实测见 [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)。

相对 [`005-pixal3d`](../005-pixal3d/)（官版 HF demo · 默认 H100）：

| | 005 官版 | **005-v2（本目录）** |
|--|----------|----------------------|
| 默认 GPU | H100 | **L40S** |
| CUDA 扩展 | Spaces 预编译（≈ sm_90） | **按 sm_89 源码构建并缓存** |
| L40S | ❌ 旧文档不可用 | ✅ **已验证出 GLB** |
| 权重 | TencentARC/Pixal3D | 同左 |

## 实测（smoke）

| GPU | 总时 | 峰值 VRAM | 估费 | 输出 |
|-----|------|-----------|------|------|
| **L40S** | **~311 s** | **15.4 GB** | **~$0.17** | 38 MB GLB |

对照官版：H100 ~279s / ~$0.31 · A100 ~491s / ~$0.29。

本地查看：[`viewer/index.html`](viewer/index.html) + [`viewer/smoke_l40s.glb`](viewer/smoke_l40s.glb)

## 解法摘要

禁止 HF demo 轮子。Plan A：

```text
TORCH_CUDA_ARCH_LIST=8.9
NATTEN_CUDA_ARCH=8.9
ATTN_BACKEND=sdpa
```

源码编译：`flex_gemm` · `o_voxel` · `cumesh` · `nvdiffrast` · `nvdiffrec` · `natten`  
轮子缓存 Volume：`modal-lab-pixal3d-l40s-wheels`

## 用法

```bash
python run.py status
python run.py build-sm89 --i-know-this-costs-money   # 首次 / 缺轮子
python run.py verify --i-know-this-costs-money
python run.py smoke --i-know-this-costs-money --output-name demo_l40s

modal volume get modal-lab-pixal3d-l40s-outputs meshes/smoke_l40s.glb ./viewer/
```

## 许可

上游 MIT（见 [UPSTREAM.md](UPSTREAM.md)）。
