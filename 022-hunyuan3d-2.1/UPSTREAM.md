# 022 · Upstream

| | |
|--|--|
| 代码 | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) |
| 权重 | [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) |
| 许可 | **Tencent Hunyuan 3D 2.1 Community License**（非商用 / 地域限制，见上游 LICENSE） |
| Shape | `hunyuan3d-dit-v2-1` · 3.3B · ~10 GB VRAM |
| Paint | `hunyuan3d-paintpbr-v2-1` · 2B · ~21 GB VRAM |
| 合计 | shape+texture ~29 GB（官方） |

## 本实验栈

| GPU | torch | CUDA | ARCH | 扩展 |
|-----|-------|------|------|------|
| L40S | 2.5.1+cu124 | 12.4 | 8.9 | custom_rasterizer · mesh_inpaint_processor |
| PRO 6000 | 2.11+cu128 | 12.8 | 12.0 | 同上 |

对照图默认与 020/021 相同：`chair.png`（TripoSR 示例椅）。
