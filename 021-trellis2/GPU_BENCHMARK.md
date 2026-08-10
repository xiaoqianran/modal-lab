# 021 TRELLIS.2 — GPU 实测

样本：与 020 相同 `chair.png` · `pipeline_type=512` · PBR GLB

| GPU | 总时 | 推理 | mesh | 峰值 VRAM | 估费 | GLB |
|-----|------|------|------|-----------|------|-----|
| **L40S** | **214.5 s** | 89.1 s | 38.1 s | **3.2 GB** | **~$0.12** | 16.9 MB |
| **RTX-PRO-6000** | **121.9 s** | 50.9 s | 18.0 s | **3.3 GB** | **~$0.10** | 17.1 MB |

PRO 6000 比 L40S 快约 **43%**（总墙钟），推理 ~1.75×。

## 对照同图

| 实验 | 模型 | L40S 总时 | PRO6000 | 纹理 | 估费(L40S) |
|------|------|-----------|---------|------|------------|
| **020** | TripoSR | **~14 s** | **~10 s** | vertex | ~$0.007 |
| **021** | TRELLIS.2-4B @512 | **~215 s** | **~122 s** | PBR | ~$0.12 |
| 005-v2/v3 | Pixal3D @1024 | ~311 s | ~230 s | PBR | ~$0.17 |

## 备注

- 栈 L40S：torch2.6+cu124 · sm_89 源码轮 · **xformers** · flex_gemm
- 栈 PRO：torch2.11+cu128 · sm_120 · **xformers** · flex_gemm
- DINOv3 / RMBG 门禁 → `camenduru/dinov3…` + `ZhengPeng7/BiRefNet`
- 质量可升 `pipeline_type=1024` / `1024_cascade`（更慢）
