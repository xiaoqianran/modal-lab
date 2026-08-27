# 005-pixal3d — Pixal3D 单图生成 3D（Modal · 单卡）

[TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D)（SIGGRAPH 2026）在 Modal 上的可复现实验：
**一张参考图 → 带 PBR 贴图的 GLB**，输出只写远程 Volume。

005 官版路径已迁移到 v2：一个 `app.py` 同时拥有 natten build、本地图片输入、CLI、结构化输出和 Web 下载端点。

| 项 | 选择 |
|----|------|
| **默认 GPU** | **`H100`**（HF demo 预编译轮子原生 sm_90） |
| 降本可选 | **`A100-40GB`**（首次需 `build-natten`；纯推理约 460s / ~$0.27） |
| **不可用** | `RTX-PRO-6000`（Blackwell sm_120 · torch2.6 无 kernel）· `L40S`（natten 无 sm_89） |
| 主权重 | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) ~24GB |
| 辅助 | MoGe-2 · DINOv3 · **BiRefNet（公开）** · NAF |
| 默认模式 | **low_vram + 1024 cascade**（峰值约 **15–16GB**） |
| 容器 | CPU 4 · RAM 24GB |
| 输出 Volume | `modal-lab-pixal3d-outputs` |

实测与选卡见 [GPU_BENCHMARK.md](GPU_BENCHMARK.md)。

## 看模型

- 列表：https://seachenxyt--modal-lab-pixal3d-index.modal.run
- 最新：https://seachenxyt--modal-lab-pixal3d-download.modal.run?name=latest
- CLI：`modal volume get modal-lab-pixal3d-outputs meshes/latest.glb ./outputs/latest.glb`
- 本地 HTML 查看器：[`viewer/index.html`](viewer/index.html)（需先 `pull` GLB 到 `viewer/`）

Volume 布局：

```text
meshes/<name>.glb
meshes/latest.glb
meshes/<name>_meta.json
benchmarks/<name>.json
inputs/<name>.*
```

## 用法

```bash
python main.py 005-pixal3d status

# 预拉权重（CPU only）
python main.py 005-pixal3d download

# 推荐：H100
python main.py 005-pixal3d smoke
python main.py 005-pixal3d i2v --image 005-pixal3d/inputs/sample.webp --output-name demo_cat --gpu H100

# 降本：A100-40GB（首次编译 natten）
python main.py 005-pixal3d build-natten --gpu A100-40GB
python main.py 005-pixal3d i2v --image 005-pixal3d/inputs/sample.webp --output-name demo_a100 --gpu A100-40GB

# 结构化查看 run；文件拉取直接使用 Modal
python main.py 005-pixal3d list-outputs
modal volume get modal-lab-pixal3d-outputs meshes/demo_cat.glb 005-pixal3d/viewer/demo_sample.glb
```

## 选卡结论（实测）

| GPU | 纯推理 | 峰值显存 | 估费（仅 GPU） | 状态 |
|-----|--------|----------|---------------|------|
| **H100** | **~279 s** | **15.9 GB** | **~$0.31** | ✅ 推荐默认 |
| **A100-40GB** | **~460 s** | **15.4 GB** | **~$0.27**（不含首次编译） | ✅ 需 `build-natten` |
| RTX-PRO-6000 | — | — | — | ❌ torch2.6 无 sm_120 |
| L40S | — | — | — | ❌ HF natten 无 sm_89 |

> 单价 A100 更低，但 H100 更快；**端到端费用接近，H100 更省心**（零编译）。

## 设计取舍

1. **镜像**：CUDA 12.4 + Python 3.10 + 官方 HF demo 预编译 CUDA 扩展。
2. **不在镜像 multi-arch 编 natten**：镜像构建慢/易抢占；A100 用 **Volume 缓存轮子**。
3. **ATTN_BACKEND=sdpa**：`flash_attn` 与 `flash_attn_3` 包名不一致。
4. **rembg**：门禁 `briaai/RMBG-2.0` → 公开 `ZhengPeng7/BiRefNet`。
5. **huggingface_hub < 1.0**：兼容 transformers 4.57.3。
6. **输出只认 Volume**。

## 许可

上游 **MIT**（见 [UPSTREAM.md](UPSTREAM.md)）。


## 测试

```bash
python -m unittest discover -s 005-pixal3d/tests -v
python -m py_compile 005-pixal3d/app.py
python main.py 005-pixal3d status
python main.py 005-pixal3d i2v --dry-run
```

以上测试不启动付费 GPU。
