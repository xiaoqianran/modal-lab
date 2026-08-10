# 023-a SPAR3D — GPU 实测

样本：与 020 相同 `chair.png` · texture_resolution=1024 · remesh=none · UV GLB  
权重镜像：`zimengxiong/Modelr-SPAR3D`（官版 `stabilityai/stable-point-aware-3d` 门禁）

| GPU | 总时 | load | pre | 推理 | 峰值 VRAM | 估费 | GLB |
|-----|------|------|-----|------|-----------|------|-----|
| **L40S** | **96.2 s** | 80.9 s | 13.2 s | **2.07 s** | **11.06 GB** | **~$0.052** | 0.87 MB |
| **RTX-PRO-6000** | **22.1 s** | 16.0 s | 4.6 s | **1.43 s** | **11.23 GB** | **~$0.019** | 0.89 MB |

> 总时含冷启动权重下载/缓存；**纯推理** PRO 比 L40S 快约 **31%**（1.43 vs 2.07 s）。

## 对照（同 lab · 同 chair.png）

| 实验 | 模型 | L40S 推理 | 纹理 | 角色 |
|------|------|-----------|------|------|
| 020 | TripoSR | ~1.5 s | vertex | 速度 |
| **023-b** | SF3D | ~1.5 s | UV | 快·纹理 |
| **023-a** | SPAR3D | **~2.1 s** | UV+points | 中速·背面 |
| 021 | TRELLIS.2-4B@512 | ~215 s 总 | PBR | 质量 MIT |

## 备注

- 栈：L40S = torch 2.5.1+cu124；PRO 6000 = torch 2.11.0+cu128（sm_120）。
- 默认 ~11 GB VRAM（官方 ~10.5）；`--low-vram-mode` 约 7 GB 更慢。
- 依赖 AlphaCLIP + transparent-background==1.2.12（避免 1.3.x flet GUI 冲突）。
- 官版门禁：HF 同意后可 `--hf-model stabilityai/stable-point-aware-3d`。
