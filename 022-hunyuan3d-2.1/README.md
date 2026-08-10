# 022-hunyuan3d-2.1 — **Hunyuan3D-2.1**（PBR / Community License）

| | |
|--|--|
| 模型 | [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) |
| 代码 | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) |
| 许可 | **Tencent Hunyuan 3D 2.1 Community License**（非 MIT） |
| 默认 GPU | **L40S** |
| 可选 | **RTX-PRO-6000** |
| 定位 | 开源 image→3D **PBR 纹理**（对照 020 速度 / 021 MIT 质量） |
| 状态 | ✅ L40S + PRO · shape & full 均通过 |

## 用法

```bash
python run.py status
python run.py probe --gpu L40S
python run.py smoke --i-know-this-costs-money --gpu L40S --mode shape
python run.py smoke --i-know-this-costs-money --gpu L40S --mode full
python run.py smoke --i-know-this-costs-money --gpu RTX-PRO-6000 --mode full
```

```bash
python -m modal volume get modal-lab-hunyuan3d21-outputs meshes/smoke_l40s.glb ./viewer/
```

## 实测（chair.png · seed=42）

| GPU | mode | shape | paint | VRAM | GLB |
|-----|------|-------|-------|------|-----|
| L40S | shape | **29 s** | — | 8.4 GB | 12.4 MB |
| L40S | full | 30 s | **65 s** | 16.5 GB | 1.25 MB |
| PRO 6000 | shape | **18 s** | — | 8.5 GB | 12.2 MB |
| PRO 6000 | full | 18 s | **67 s** | 16.3 GB | 1.23 MB |

**结论**：PRO 在 shape 上约 **1.6×** 快于 L40S；paint 两卡接近（~65–67 s）。

## 对照（同 lab · chair.png）

| 实验 | 模型 | 角色 | 许可 |
|------|------|------|------|
| 020 | TripoSR | 速度 | MIT |
| 021 | TRELLIS.2-4B | 质量 | MIT |
| **022** | **Hunyuan3D-2.1** | **PBR 纹理** | Community |
| 005 | Pixal3D | 慢·对齐 | MIT |

## 技术栈

| GPU | torch | CUDA | ARCH |
|-----|-------|------|------|
| L40S | 2.5.1+cu124 | 12.4 | 8.9 |
| PRO 6000 | 2.11+cu128 | 12.8 | 12.0 |

补丁：无 bpy · 无 open3d remesh · `HY3DGEN_MODELS`→volume · runtime `custom_rasterizer`。

见 [PLAN.md](PLAN.md) · [UPSTREAM.md](UPSTREAM.md) · [GPU_BENCHMARK.md](GPU_BENCHMARK.md)
