# 005-v2 · L40S（Ada sm_89）真实解法

> **状态：已在 Modal L40S 验证通过（2026-08-10 smoke）**  
> 输出：`modal-lab-pixal3d-l40s-outputs/meshes/smoke_l40s.glb`（~38MB，~311s，~15.4GB VRAM，~$0.17）

---

## 0. 一句话结论

| 问题 | 答案 |
|------|------|
| L40S 能不能跑 Pixal3D？ | **能，已跑通。** |
| 005 官版为何曾标不可用？ | 官版锁了 **HF Spaces 预编译轮子（≈ Hopper sm_90）**。 |
| 真正要修什么？ | **整组 CUDA 扩展按 sm_89 重编**，不是只改 `gpu=`。 |
| 推荐路径 | **Plan A：官方源码 + `TORCH_CUDA_ARCH_LIST=8.9`**（已验证）。 |

---

## 1. 硬件

L40S = **Ada Lovelace sm_89**（与 RTX 4070/4090 同族），48GB。Modal 代号 `L40S`。

## 2. 根因

HF `requirements-hfdemo` 轮子面向 H 系列；`sm_90` cubin / `compute_90` PTX **不能**在 sm_89 上跑 → `no kernel image`。

## 3. Plan A（已采用并验证）

| 组件 | 版本 |
|------|------|
| GPU | L40S |
| Python / torch | 3.10 · **2.6.0+cu124** |
| TORCH_CUDA_ARCH_LIST | **8.9** |
| NATTEN_CUDA_ARCH | **8.9** |
| 扩展 | nvdiffrast · nvdiffrec · flex_gemm · cumesh · o_voxel · **natten 0.21.0** 源码编 |
| ATTN | **sdpa** |
| 权重 | TencentARC/Pixal3D |
| rembg | BiRefNet（公开） |

构建产物缓存在 `modal-lab-pixal3d-l40s-wheels/sm89/torch260-cu124-cp310/`。

### 构建时注意

- 镜像需 **clang 或 g++**（nvdiffrast 链接曾缺 `clang++`）  
- 每个扩展编完 **commit Volume**，避免中断丢轮子  
- 推理前 `pip install` Volume 内全部 `.whl` + natten 小 smoke  

## 4. 验收（已完成）

| 门禁 | 结果 |
|------|------|
| G0 capability (8,9) | ✅ |
| G1–G4 扩展安装 / natten smoke | ✅ |
| G5 端到端 GLB | ✅ 38.3 MB |
| G6 显存 < 40GB | ✅ 15.4 GB |
| 对照 H100 可出片 | ✅（时间/费用见 GPU_BENCHMARK） |

## 5. Plan B（未使用）

carroyoaesa 社区 sm_89 轮子（py3.12 + torch 2.9.1+cu129）仍可作为备选；Plan A 已足够。

## 6. 参考

- 官方 Pixal3D / TRELLIS.2 setup  
- 社区 Ada 文档：carroyoaesa sm89-wheels  
- 本 lab 官版：`005-pixal3d`  
- 实测：[`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)
