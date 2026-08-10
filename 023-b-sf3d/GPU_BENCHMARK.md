# 023-b SF3D — GPU 实测

样本：与 020 相同 `chair.png` · texture_resolution=1024 · remesh=none · UV GLB  
权重镜像：`cocktailpeanut/sf3d`（官版 `stabilityai/stable-fast-3d` 门禁）

| GPU | 总时 | load | pre | 推理 | 峰值 VRAM | 估费 | GLB |
|-----|------|------|-----|------|-----------|------|-----|
| **L40S** | **54.0 s** | 48.0 s | 4.5 s | **1.45 s** | **6.79 GB** | **~$0.029** | 0.79 MB |
| **RTX-PRO-6000** | **17.8 s** | 13.3 s | 3.5 s | **0.96 s** | **6.96 GB** | **~$0.015** | 0.79 MB |

> 总时含冷启动权重下载/缓存；**纯推理** PRO 比 L40S 快约 **34%**（0.96 vs 1.45 s）。

## 对照（同 lab · 同 chair.png）

| 实验 | 模型 | L40S 推理 | 纹理 | 角色 |
|------|------|-----------|------|------|
| 020 | TripoSR | ~1.5 s | vertex | 速度 |
| **023-b** | SF3D | **~1.5 s** | UV | 快·纹理 |
| **023-a** | SPAR3D | ~2.1 s | UV+points | 中速·背面 |
| 021 | TRELLIS.2-4B@512 | ~215 s 总 | PBR | 质量 MIT |

## 备注

- 栈：L40S = torch 2.5.1+cu124；PRO 6000 = torch 2.11.0+cu128（sm_120）。
- 默认约 7 GB VRAM；remesh=none（未装 remesh 依赖也可用）。
- 官版门禁：HF 同意 Community License 后可 `--hf-model stabilityai/stable-fast-3d`。
