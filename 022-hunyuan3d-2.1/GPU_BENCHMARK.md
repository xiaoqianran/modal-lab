# 022 · Hunyuan3D-2.1 GPU 实测

样本：与 020/021 相同 `chair.png` · seed=42 · max_num_view=6 · paint_res=512

| GPU | mode | 总时 | shape | paint | 峰值 VRAM | 估费 | GLB | 状态 |
|-----|------|------|-------|-------|-----------|------|-----|------|
| L40S | shape | 358 s* | **29.2 s** | — | **8.4 GB** | $0.19 | 12.4 MB | ✅ |
| L40S | full | **242 s** | **30.3 s** | **65.1 s** | **16.5 GB** | $0.13 | 1.25 MB | ✅ |
| RTX-PRO-6000 | shape | 70 s* | **18.0 s** | — | **8.5 GB** | $0.06 | 12.2 MB | ✅ |
| RTX-PRO-6000 | full | **188 s** | **18.1 s** | **66.7 s** | **16.3 GB** | $0.16 | 1.23 MB | ✅ |

\* shape 总时含首次权重下载 / rembg；对照 **seconds_shape** 更公平。

## 对照同图

| 实验 | 模型 | L40S | PRO6000 | 纹理 |
|------|------|------|---------|------|
| 020 | TripoSR | ~14 s | ~10 s | vertex |
| 021 | TRELLIS.2 @512 | ~215 s | ~122 s | PBR |
| **022** | **Hunyuan3D-2.1 full** | **shape 30 + paint 65 ≈ 95 s 有效** | **shape 18 + paint 67** | **PBR** |

## 备注

- 官方：shape ~10 GB · texture ~21 GB · 合计 ~29 GB；本 smoke peak ~**16.5 GB**（paint 512 / 6 view）
- 权重 `/weights/hy3dgen`（`HY3DGEN_MODELS`）· 扩展轮子 `/weights/wheels/sm{89,120}/`
- 无 bpy（trimesh GLB）· 无 open3d remesh（pymeshlab）
- 许可 Community License，商用/地域见上游 LICENSE
