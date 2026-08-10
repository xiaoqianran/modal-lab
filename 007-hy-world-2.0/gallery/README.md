# Gallery · 007 HY-World 2.0

交互页三块：

1. **GPU 对比** — T4 / L4 / L40S / RTX PRO 6000 同条件实测（成本条形图 + 表）
2. **007 vs 006** — 两代 WorldMirror 差异卡片
3. **3D 重建** — smoke_desk 点云 + 深度/法线

数据：

- `assets/manifest.json` + `points.bin` — 点云预览
- `assets/bench.json` — 四卡 bench 汇总（源 meta 在 `bench/*.json`）

```bash
cd 007-hy-world-2.0/gallery
python3 -m http.server 8080 --bind 0.0.0.0
```

| 字段 | smoke_desk |
|---|---|
| GPU | Tesla T4 |
| Peak VRAM | 4.89 GB |
| Total | 27.7 s |
| Est. cost | ~$0.0045 |
| Points | 330,776 full / 80k preview |
