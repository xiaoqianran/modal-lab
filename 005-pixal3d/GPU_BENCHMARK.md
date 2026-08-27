# Pixal3D GPU 实测（005）

条件：`low_vram` · `resolution=1024` · 官方样例 `5_img.webp` · seed 42  
栈：`torch==2.6.0+cu124` · HF demo CUDA 扩展 · `ATTN_BACKEND=sdpa`

## 结果总表

| GPU 请求 | 实卡 | SM | 纯推理 | 墙钟合计 | 峰值显存 | 估费(infer) | 状态 |
|----------|------|----|-------:|---------:|---------:|------------:|------|
| **H100** | H100 80GB HBM3 | sm_90 | **278.9 s** | 284.7 s | **15.87 GB** | **$0.306** | ✅ 推荐默认 |
| **A100-40GB** | A100-SXM4-40GB | sm_80 | **460.2 s** | 2660 s* | **15.41 GB** | **$0.268** | ✅ |
| RTX-PRO-6000 | Blackwell SE 95GB | sm_120 | — | — | — | — | ❌ |
| L40S | L40S 44.4GB | sm_89 | — | — | — | — | ❌ |

\* A100 墙钟 2660s **含首次 `build-natten` 源码编译**（~37 min）。  
纯推理约 460s / ~$0.27。请用 `python main.py 005-pixal3d build-natten --gpu A100-40GB` 单独编译并正确缓存 wheel（需含 libnatten）。

## 单价

| GPU | $/秒 |
|-----|-----:|
| H100 | 0.001097 |
| RTX-PRO-6000 | 0.000842 |
| A100-40GB | 0.000583 |
| L40S | 0.000542 |

## 失败根因

### RTX-PRO-6000

- sm_120 Blackwell；torch 2.6+cu124 **无 sm_12x**
- 与 Modal 文档一致：Hopper 预编译生态更成熟

### L40S

- HF natten 无 sm_89 kernel → `no kernel image`

### A100 natten

- HF demo 轮子仅 Hopper；需 `NATTEN_CUDA_ARCH=8.0` 源码编译
- Volume 缓存 wheel 必须带 **libnatten**（纯 Python sdist 不可用）

## 产物

| 名称 | Volume | 大小 |
|------|--------|-----:|
| demo_sample.glb | meshes/ | ~39.9 MB |
| demo_a100_40gb.glb | meshes/ | ~40.7 MB |

## 推荐

| 目标 | 选择 |
|------|------|
| 默认 / 省心 | **H100** |
| 最低纯推理 GPU 费 | **A100-40GB** + `build-natten` |
| PRO 6000 | **不可行**（当前栈） |
