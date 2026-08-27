# 005-v3 · RTX PRO 6000（Blackwell sm_120）方案

> **状态：已在 Modal 实跑通过（2026-08-10 smoke）**  
> 输出：`modal-lab-pixal3d-pro6000-outputs/meshes/smoke_pro6000.glb`  
> **~230 s · ~15.6 GB · ~$0.19 · 41 MB**

---

## 0. 结论

| 问题 | 答案 |
|------|------|
| 能不能跑？ | **能，已出 GLB。** |
| 旧 005/v2 栈？ | **不能**（torch2.6+cu124 无 sm_120）。 |
| 真正栈 | **torch 2.11.0+cu128 + ARCH=12.0 源码轮子 + sdpa** |
| 默认出片推荐 | L40S 仍更便宜；PRO 6000 适合卡池统一 / 要更快采样 |

## 1. 配方（Plan A* · 已落地）

| 项 | 值 |
|----|-----|
| GPU | `RTX-PRO-6000` |
| 镜像 | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Python | 3.10 |
| torch | **2.11.0+cu128** / torchvision 0.26.0+cu128 |
| ARCH | **TORCH_CUDA_ARCH_LIST=12.0** · NATTEN_CUDA_ARCH=12.0 |
| ATTN | **sdpa**（不装 flash_attn） |
| 扩展 | nvdiffrast · nvdiffrec · flex_gemm · cumesh · o_voxel · drtk · **natten 0.21.0** |

## 2. 证据链

- B0：capability (12,0)，arch_list 含 sm_120，matmul OK  
- 7 wheels 编译成功  
- verify：natten-forward PASS  
- smoke：端到端 GLB  

社区：animede Pixal3D PRO 6000 文档 · TRELLIS.2 #143 5090 e2e · PixWizardry Windows 轮子（旁证）

## 3. 对照

| GPU | 时间 | 估费 |
|-----|------|------|
| H100 | ~279s | ~$0.31 |
| L40S | ~311s | ~$0.17 |
| **PRO 6000** | **~230s** | **~$0.19** |

## 4. 参考

见 `GPU_BENCHMARK.md` · `app.py` · `viewer/`
