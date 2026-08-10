# 023-b-sf3d — **SF3D** 快速纹理档（image → textured GLB）

| | |
|--|--|
| 模型 | [stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d)（默认镜像 `cocktailpeanut/sf3d`） |
| 代码 | [Stability-AI/stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d) |
| 默认 GPU | **L40S** |
| 可选 | **RTX-PRO-6000**（torch 2.11+cu128 / sm_120） |
| 定位 | 开源 image→3D **快速 UV 纹理**（TripoSR 后继 · delight 材质） |
| 许可 | Stability AI Community |
| 状态 | ✅ L40S + PRO 6000 smoke 已出 GLB |

## 实测（smoke · chair.png）

| GPU | 总时 | 推理 | 峰值 VRAM | 估费 | 输出 |
|-----|------|------|-----------|------|------|
| **L40S** | **54.0 s** | **1.45 s** | **6.8 GB** | **~$0.029** | 0.79 MB GLB |
| **PRO 6000** | **17.8 s** | **0.96 s** | **7.0 GB** | **~$0.015** | 0.79 MB GLB |

本地查看：[`viewer/index.html`](viewer/index.html)

## 用法

```bash
python run.py status
python run.py probe --gpu L40S
python run.py smoke --i-know-this-costs-money --gpu L40S
python run.py smoke --i-know-this-costs-money --gpu RTX-PRO-6000
# 官版门禁（需 HF 已同意）
# modal run modal_app.py --action smoke --gpu L40S --hf-model stabilityai/stable-fast-3d
```

拉结果：

```bash
modal volume get modal-lab-sf3d-outputs meshes/smoke_l40s.glb ./viewer/
modal volume get modal-lab-sf3d-outputs meshes/smoke_pro6000.glb ./viewer/
```

## 技术栈

| GPU | torch | CUDA | 扩展 |
|-----|-------|------|------|
| L40S | 2.5.1+cu124 | 12.4 | texture_baker ARCH=8.9 · uv_unwrapper |
| PRO 6000 | 2.11.0+cu128 | 12.8 | texture_baker ARCH=12.0 · uv_unwrapper |

## 对照计划

| 实验 | 模型 | 角色 |
|------|------|------|
| **020** | TripoSR | 最快 · vertex color |
| **023-b** | SF3D | 快 · UV 纹理 |
| **023-a** | SPAR3D | 中速 · 点云条件背面 |
| **021** | TRELLIS.2 | 质量 / MIT |
| 005 | Pixal3D | 慢·高对齐 |
