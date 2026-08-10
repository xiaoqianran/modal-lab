# 009-hy-worldgen

[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · **World Generation**  
全景 → 轨迹 → WorldStereo 扩帧 → WorldMirror 对齐 → 3DGS → **3D 世界**

> 前置：  
> - **007** WorldMirror 2.0（重建权重 / Stage3 对齐）  
> - **008** HY-Pano（产出 `panorama.png`）  
> 详细阶段与省钱策略 → **[PLAN.md](PLAN.md)**  
> 结果预览 → **[gallery/](gallery/)**

## Smoke 结果（单卡 RTX-PRO-6000）

| Stage | 内容 | ok 阶段估费 |
|------|------|-------------|
| 1 | traj_generate · nframe=21 | ~$0.05 |
| 2 | traj_render · 3 traj | ~$0.33 |
| 3 | WorldStereo-memory-dmd + WorldMirror (SDPA) | ~$0.12 |
| 4 | gen_gs_data · 127 cams | ~$0.05 |
| 5 | 3DGS 4000 steps · ~1.36M GS | ~$0.18 |
| **Σ ok stages** | | **~$0.72** |

3DGS val：PSNR **19.96** / SSIM **0.73** / LPIPS **0.19**

产物：`gallery/assets/ply/point_cloud_3999.ply` + traj mp4 + val render。

## 关键修复

- **Stage3 WorldMirror**：无 `flash_attn` → SDPA 回退；本地权重 `/weights/HY-WorldMirror-2.0`
- **Stage5**：`--disable_viewer`，训练完直接退出（不再卡 viser）
- SAM3 gated → `SKIP_SAM3=1`（室内 smoke）

## 命令

```bash
python main.py 009 status
python main.py 009 prepare --from-008 smoke_qwen
python main.py 009 download --which worldstereo-dmd   # or worldmirror / all
python main.py 009 stage 1 --gpu RTX-PRO-6000 --nframe 21
python main.py 009 stage 2 --gpu RTX-PRO-6000 --nframe 21
python main.py 009 stage 3 --gpu RTX-PRO-6000 --nframe 21
python main.py 009 stage 4 --gpu RTX-PRO-6000
python main.py 009 stage 5 --gpu RTX-PRO-6000 --max-steps 4000
# 或
python main.py 009 smoke --gpu RTX-PRO-6000 --nframe 21 --max-steps 4000
```

Gallery：

```bash
python -m http.server 8765 --directory 009-hy-worldgen/gallery
```

## Volume

| Volume | 用途 |
|---|---|
| `modal-lab-hy-worldgen-weights` | WorldStereo + HY-WorldMirror-2.0 |
| `modal-lab-hy-worldgen-outputs` | scene 中间产物 + 3DGS |
| （复用）`modal-lab-hy-pano-outputs` | 读 008 全景 |

见 [UPSTREAM.md](UPSTREAM.md) · [PLAN.md](PLAN.md) · [gallery/](gallery/)。
