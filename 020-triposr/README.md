# 020-triposr — **TripoSR** 速度基线（image → mesh）

| | |
|--|--|
| 模型 | [stabilityai/TripoSR](https://huggingface.co/stabilityai/TripoSR) |
| 代码 | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) MIT |
| 默认 GPU | **L40S** |
| 可选 | **RTX-PRO-6000**（独立镜像 torch+cu128 / sm_120） |
| 定位 | 开源 image→3D **速度基线** |
| 状态 | ✅ L40S + PRO 6000 smoke 已出 GLB |

## 实测（smoke · chair.png）

| GPU | 总时 | 推理 | mesh | 峰值 VRAM | 估费 | 输出 |
|-----|------|------|------|-----------|------|------|
| **L40S** | **13.7 s** | 1.5 s | 1.7 s | **~5.0 GB** | **~$0.007** | 1.6 MB GLB |
| **PRO 6000** | **9.7 s** | 1.6 s | 0.7 s | **~5.1 GB** | **~$0.008** | 1.6 MB GLB |

本地查看：[`viewer/index.html`](viewer/index.html)

## 用法

```bash
python run.py status
python run.py probe --gpu L40S
python run.py smoke --i-know-this-costs-money --gpu L40S
python run.py smoke --i-know-this-costs-money --gpu RTX-PRO-6000
```

拉结果：

```bash
modal volume get modal-lab-triposr-outputs meshes/smoke_l40s.glb ./viewer/
modal volume get modal-lab-triposr-outputs meshes/smoke_pro6000.glb ./viewer/
```

## 技术栈

| GPU | torch | CUDA | 扩展 |
|-----|-------|------|------|
| L40S | 2.5.1+cu124 | 12.4 | torchmcubes ARCH=8.9 |
| PRO 6000 | 2.11+cu128 | 12.8 | torchmcubes ARCH=12.0 |

`numpy==1.26.4`（trimesh 4.0.x 与 numpy 2 不兼容）。

## 对照计划

| 实验 | 模型 | 角色 |
|------|------|------|
| **020** | TripoSR | 快 |
| **021** | TRELLIS.2 | 质量 / MIT |
| 005 | Pixal3D | 慢·高对齐 |
