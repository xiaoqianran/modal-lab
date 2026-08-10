# 006-hunyuanworld-mirror

在 Modal 上以**最低成本**跑通 [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)
（ICML 2026 · feed-forward 多视图 3D 几何：点云 / 深度 / 法线 / 相机）。

跳过 005，本实验编号 **006**。

## 成本策略（默认）

| 项 | 选择 | 原因 |
|---|---|---|
| GPU | **L4**（24GB） | 约 `$0.000222/s`，比 A100/H100/PRO6000 便宜 3–5×；5GB 权重 + 2–4 视图够用 |
| 下载 | **CPU only** | 不计 GPU 费；权重落 Volume 复用 |
| smoke | **2 张图** `Bright_Room` | 最短前向；关 3DGS 视频 / COLMAP / sky mask |
| 容器 | 无 `keep_warm`，`scaledown_window=30s` | 跑完尽快放掉 GPU |

可选更省：`--gpu T4`（16GB · 更便宜，多视图可能 OOM）。  
需要更稳/更大分辨率：`--gpu A10` / `L40S` / `A100-40GB`。

## 快速开始

```bash
# 需已 modal token set
python main.py 006 status
python main.py 006 download          # CPU 拉 HF 权重 → Volume
python main.py 006 smoke             # L4 · Bright_Room · 2 图
python main.py 006 infer --example Ireland_Landscape --max-images 4
python main.py 006 ls
python main.py 006 pull --remote runs/smoke_bright_room
```

或：

```bash
cd 006-hunyuanworld-mirror
python run.py smoke --gpu L4
```

## 结果展示（本地 Gallery）

smoke 产物已拉取并做成静态页：

```text
gallery/
  index.html          # 交互式 3D 点云 + 深度/法线对照 + meta
  assets/             # images / depth / normal / points.bin / manifest.json
```

```bash
cd 006-hunyuanworld-mirror/gallery
python3 -m http.server 8080 --bind 0.0.0.0
```

实测 smoke：NVIDIA L4 · peak VRAM **7.66 GB** · forward **~3 s** · 估 **~$0.01** · 全量点云 274k（网页预览 80k）。

## Volume

| Volume | 路径 |
|---|---|
| `modal-lab-hunyuanworld-mirror-weights` | `/weights/ckpts` · HF cache |
| `modal-lab-hunyuanworld-mirror-outputs` | `/outputs/runs/<name>/` |

单次 run 产物：

```text
runs/<name>/
  pts_from_pointmap.ply
  depth/depth_XXXX.{png,npy}
  normal/normal_XXXX.png
  cameras.json
  images/…
  meta.json          # 含秒数、估算 USD、峰值显存
```

## 上游

- Code: https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror  
- Weights: https://huggingface.co/tencent/HunyuanWorld-Mirror（~5.05 GB `model.safetensors`）  
- 镜像构建对齐官方：CUDA 12.4 · torch 2.4 · gsplat pt24cu124 · python 3.10  

详见 [UPSTREAM.md](UPSTREAM.md)。
