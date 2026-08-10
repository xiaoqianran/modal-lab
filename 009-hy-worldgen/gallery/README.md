# 009 Gallery — 可交互 3D 世界

页面顶部是 **3DGS 渲染的房间画面**，下方是 **可拖拽旋转的点云世界**（three.js）。

## 打开方式（必须用 HTTP）

```bash
# 仓库根目录
python -m http.server 8765 --directory 009-hy-worldgen/gallery
# 浏览器打开 http://127.0.0.1:8765/
```

不要直接双击 `index.html`（`file://` 会拦 ES module / PLY 加载）。

## 操作

- 左键拖拽：旋转
- 滚轮：缩放
- 右键拖拽：平移
- 下拉框：切换 3DGS 预览 / WorldMirror 点云

## 资产

| 文件 | 说明 |
|------|------|
| `assets/renders/val_step3999.png` | 3DGS 渲染世界（最直观） |
| `assets/ply/world_preview.ply` | 网页点云预览 ~35 万点 · 5MB |
| `assets/ply/point_cloud_3999.ply` | 完整 3DGS 高斯 ~73MB |
| `assets/videos/traj*_worldstereo.mp4` | WorldStereo 扩帧视频 |
