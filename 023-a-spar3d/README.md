# 023-a-spar3d — **SPAR3D** 中速档（image → textured GLB）

| | |
|--|--|
| 模型 | [stabilityai/stable-point-aware-3d](https://huggingface.co/stabilityai/stable-point-aware-3d)（默认镜像 `zimengxiong/Modelr-SPAR3D`） |
| 代码 | [Stability-AI/stable-point-aware-3d](https://github.com/Stability-AI/stable-point-aware-3d) |
| 默认 GPU | **L40S** |
| 可选 | **RTX-PRO-6000**（torch 2.11+cu128 / sm_120） |
| 定位 | 开源 image→3D **点云条件 / 中速·纹理**（基于 SF3D 改进背面） |
| 许可 | Stability AI Community |
| 状态 | ✅ L40S + PRO 6000 smoke 已出 GLB |

023-a 已迁移到 v2：单入口 `app.py`，不再使用 `run.py -> modal_app.py` 翻译层。

## 实测（smoke · chair.png）

| GPU | 总时 | 推理 | 峰值 VRAM | 估费 | 输出 |
|-----|------|------|-----------|------|------|
| **L40S** | **96.2 s** | **2.07 s** | **11.1 GB** | **~$0.052** | 0.87 MB GLB |
| **PRO 6000** | **22.1 s** | **1.43 s** | **11.2 GB** | **~$0.019** | 0.89 MB GLB |

本地查看：[`viewer/index.html`](viewer/index.html)

## 用法

```bash
# 本地状态，不启动 GPU
python main.py 023-a-spar3d status

# 远程 probe
python main.py 023-a-spar3d probe --gpu L40S

# 无成本 dry-run
python main.py 023-a-spar3d smoke --dry-run --gpu RTX-PRO-6000 --low-vram-mode

# 真正 smoke
python main.py 023-a-spar3d smoke --i-know-this-costs-money --gpu L40S
python main.py 023-a-spar3d smoke --i-know-this-costs-money --gpu RTX-PRO-6000

# 官方 gated 权重（需 HF 已同意）
python main.py 023-a-spar3d smoke --i-know-this-costs-money \
  --hf-model stabilityai/stable-point-aware-3d
```

拉结果：

```bash
modal volume get modal-lab-spar3d-outputs meshes/smoke_l40s.glb ./viewer/
modal volume get modal-lab-spar3d-outputs meshes/smoke_pro6000.glb ./viewer/
```

## 技术栈

| GPU | torch | CUDA | 扩展 |
|-----|-------|------|------|
| L40S | 2.5.1+cu124 | 12.4 | texture_baker ARCH=8.9 · uv_unwrapper |
| PRO 6000 | 2.11.0+cu128 | 12.8 | texture_baker ARCH=12.0 · uv_unwrapper |

额外：AlphaCLIP · transparent-background 1.2.12 · 默认 texture 1024 · remesh=none。

## 对照计划

| 实验 | 模型 | 角色 |
|------|------|------|
| **020** | TripoSR | 最快 · vertex color |
| **023-b** | SF3D | 快 · UV 纹理 |
| **023-a** | SPAR3D | 中速 · 点云条件背面 |
| **021** | TRELLIS.2 | 质量 / MIT |
| 005 | Pixal3D | 慢·高对齐 |

## 测试

```bash
python -m unittest discover -s 023-a-spar3d/tests -v
python -m py_compile 023-a-spar3d/app.py
python main.py 023-a-spar3d status
python main.py 023-a-spar3d smoke --dry-run
```

以上测试不启动付费 GPU。
