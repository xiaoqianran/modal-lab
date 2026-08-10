# 009 Gallery

Local HTML gallery for the single-GPU (RTX-PRO-6000) HY-World 2.0 smoke run.

## Open

```bash
# from repo root
python -m http.server 8765 --directory 009-hy-worldgen/gallery
# then open http://127.0.0.1:8765/
```

Or open `index.html` directly (videos/ply still work as relative paths).

## Assets

| Path | Source |
|------|--------|
| `assets/panorama.png` | 008 → 009 seed |
| `assets/videos/traj*_render.mp4` | Stage2 |
| `assets/videos/traj*_worldstereo.mp4` | Stage3 WorldStereo-memory-dmd |
| `assets/renders/val_step3999.png` | Stage5 val |
| `assets/ply/aligned_pcd.ply` | Stage3 WorldMirror alignment |
| `assets/ply/global_pcd.ply` | generation bank |
| `assets/ply/point_cloud_3999.ply` | Stage5 3DGS (~76MB) |
| `assets/meta.json` | stage cost / metrics summary |

Pulled from Modal volume `modal-lab-hy-worldgen-outputs` / `scenes/scene_from_008`.
