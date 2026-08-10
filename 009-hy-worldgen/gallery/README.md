# 009 Gallery — HY-World Observatory

Final exhibition page for the `scene_from_008` single-GPU smoke run.

## Live

After Pages deploy:

**https://xiaoqianran.github.io/modal-lab/009-hy-worldgen/**

## Local preview

```bash
python3 -m http.server 8765 --directory 009-hy-worldgen/gallery
# → http://127.0.0.1:8765/
```

## What's inside

| Section | Content |
| --- | --- |
| **World** | Full-bleed three.js point-cloud stage (`world_preview.ply`) |
| **Quality** | PSNR / SSIM / LPIPS / Gaussian count |
| **Pipeline** | Stages 1–5 timing & cost |
| **Cinema** | traj0–2 · geometric render vs WorldStereo |
| **Stills** | Panorama · start frame · 3DGS val |
| **Download** | Preview / global / aligned / full 3DGS PLY |

## Controls

- Drag — orbit · scroll — zoom · right-drag — pan  
- Model chips — 3DGS / Global / Dense  
- Size slider · Reset · Auto-orbit · Fullscreen  

## Latest smoke metrics

- GPU: RTX-PRO-6000  
- nframe: 21 · 3 traj · max_steps 4000  
- Val: **PSNR 31.3** / SSIM **0.86** / LPIPS **0.16** · **1.51M** GS  
- Est. cost: **~$0.69** end-to-end  
