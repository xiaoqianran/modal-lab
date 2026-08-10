# 009 Gallery — HY-World 2.0 Generated World

Immersive single-page exhibition of the `scene_from_008` smoke run.

## Open

Serve this directory over HTTP (module imports + PLY fetch need it):

```bash
python3 -m http.server 8765 --directory 009-hy-worldgen/gallery
# → http://127.0.0.1:8765/
```

## What’s inside

| Section | Content |
| --- | --- |
| **World** | Full-bleed three.js point-cloud viewer (`world_preview.ply` default) |
| **Metrics** | PSNR / SSIM / Gaussians / GPU cost |
| **Pipeline** | Stages 1–5 timing & cost |
| **Motion** | traj0–2 render vs WorldStereo videos |
| **Stills** | Panorama, start frame, 3DGS val render |
| **Downloads** | Preview / global / aligned / full 3DGS PLY |

## Controls

- Drag — orbit · scroll — zoom · right-drag — pan  
- Model chips — 3DGS preview / WorldMirror global / dense aligned  
- Point size slider · Reset view · Auto-orbit · Fullscreen  

## Assets

All paths are relative under `assets/` (see `assets/meta.json` for the machine-readable summary).
