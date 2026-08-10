# Upstream

| 项 | 值 |
|---|---|
| 项目 | HY-World 2.0 |
| 仓库 | https://github.com/Tencent-Hunyuan/HY-World-2.0 |
| 本实验组件 | WorldMirror 2.0 (`hyworld2.worldrecon`) |
| 权重 | `tencent/HY-World-2.0` / `HY-WorldMirror-2.0` |
| 官方推荐环境 | CUDA 12.8 · Python 3.11 · torch 2.7.1 |

## 与官方差异

- 只装 worldrecon 依赖；镜像内删除 `panogen` / `worldgen` 树以减小体积
- 不装 FlashAttention（代码自动回退 SDPA）
- smoke 默认 2 视图、518、bf16、关 sky/COLMAP/视频渲染
- 默认 GPU：**T4**（实测 peak ~5GB）；多视图可切 L4+
- 权重与输出进 Modal Volume
