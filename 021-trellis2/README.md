# 021-trellis2 — **TRELLIS.2-4B**（质量 / MIT）

| | |
|--|--|
| 模型 | [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) |
| 代码 | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) **MIT** |
| 默认 GPU | **L40S**（sm_89 源码扩展） |
| 可选 | **RTX-PRO-6000**（sm_120） |
| 定位 | 开源 image→3D **质量主线** |
| 状态 | ✅ L40S + PRO 6000 smoke 已出 GLB |

## 实测（smoke · chair.png · pipeline=512）

| GPU | 总时 | 推理 | 峰值 VRAM | 估费 | 输出 |
|-----|------|------|-----------|------|------|
| **L40S** | **~215 s** | ~89 s | **~3.2 GB** | **~$0.12** | 16.9 MB PBR |
| **PRO 6000** | **~122 s** | ~51 s | **~3.3 GB** | **~$0.10** | 17.1 MB PBR |

本地查看：[`viewer/index.html`](viewer/index.html)

## 用法

```bash
python run.py status
python run.py probe --gpu L40S
python run.py build --gpu L40S          # 首次编译 sm_89 扩展
python run.py verify --gpu L40S
python run.py download --gpu L40S       # 权重 + DINOv3/BiRefNet 镜像
python run.py smoke --i-know-this-costs-money --gpu L40S --pipeline-type 512
python run.py smoke --i-know-this-costs-money --gpu RTX-PRO-6000
```

## 技术栈

| GPU | torch | CUDA | 扩展 ARCH | attn |
|-----|-------|------|-----------|------|
| L40S | 2.6.0+cu124 | 12.4 | 8.9 | **xformers** |
| PRO 6000 | 2.11.0+cu128 | 12.8 | 12.0 | **xformers** |

源码：`flex_gemm` · `o_voxel` · `cumesh` · `nvdiffrast` · `nvdiffrec`  
门禁绕过：`camenduru/dinov3-vitl16-pretrain-lvd1689m` · `ZhengPeng7/BiRefNet`

## 对照

| 实验 | 模型 | L40S | PRO6000 | 角色 |
|------|------|------|---------|------|
| 020 | TripoSR | **~14 s** | **~10 s** | 速度 |
| **021** | TRELLIS.2 | **~215 s** | **~122 s** | 质量 MIT |
| 005 | Pixal3D | ~311 s | ~230 s | 慢·对齐 |
