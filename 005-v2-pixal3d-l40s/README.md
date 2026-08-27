# 005-v2-pixal3d-l40s — Pixal3D on **L40S (Ada sm_89)**

> **状态：已在 Modal L40S 端到端跑通**（smoke → GLB）。方案见 [`SOLUTION.md`](SOLUTION.md)，实测见 [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)。

005-v2 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI、wheel build 生命周期和 Modal worker，不再使用 `run.py -> modal_app.py` 翻译层。

相对 [`005-pixal3d`](../005-pixal3d/)（官版 HF demo · 默认 H100）：

| | 005 官版 | **005-v2（本目录）** |
|--|----------|----------------------|
| 默认 GPU | H100 | **L40S** |
| CUDA 扩展 | Spaces 预编译（≈ sm_90） | **按 sm_89 源码构建并缓存** |
| L40S | ❌ 旧文档不可用 | ✅ **已验证出 GLB** |
| 权重 | TencentARC/Pixal3D | 同左 |

## 实测（smoke）

| GPU | 总时 | 峰值 VRAM | 估费 | 输出 |
|-----|------|-----------|------|------|
| **L40S** | **~311 s** | **15.4 GB** | **~$0.17** | 38 MB GLB |

对照官版：H100 ~279s / ~$0.31 · A100 ~491s / ~$0.29。

本地查看：[`viewer/index.html`](viewer/index.html) + [`viewer/smoke_l40s.glb`](viewer/smoke_l40s.glb)

## 解法摘要

禁止 HF demo 轮子。Plan A：

```text
TORCH_CUDA_ARCH_LIST=8.9
NATTEN_CUDA_ARCH=8.9
ATTN_BACKEND=sdpa
```

源码编译：`flex_gemm` · `o_voxel` · `cumesh` · `nvdiffrast` · `nvdiffrec` · `natten`  
轮子缓存 Volume：`modal-lab-pixal3d-l40s-wheels`

## 用法

```bash
# 纯本地固定栈
python main.py 005-v2-pixal3d-l40s status

# 远程检查 GPU / wheel cache / model
python main.py 005-v2-pixal3d-l40s check

# 无成本查看付费动作计划
python main.py 005-v2-pixal3d-l40s build-sm89 --dry-run
python main.py 005-v2-pixal3d-l40s verify --dry-run
python main.py 005-v2-pixal3d-l40s download --dry-run
python main.py 005-v2-pixal3d-l40s smoke --dry-run --output-name demo_l40s

# 真正执行付费动作
python main.py 005-v2-pixal3d-l40s build-sm89 --i-know-this-costs-money
python main.py 005-v2-pixal3d-l40s verify --i-know-this-costs-money
python main.py 005-v2-pixal3d-l40s download --i-know-this-costs-money
python main.py 005-v2-pixal3d-l40s smoke --i-know-this-costs-money --output-name demo_l40s

# 自定义图片 URL，不再和 smoke 混在同一语义里
python main.py 005-v2-pixal3d-l40s i2v --dry-run \
  --image-url https://example.com/chair.png \
  --output-name chair
```

`smoke --output-name` 现在会真正进入 `image_to_3d()`；旧双入口架构中该参数会在第二层被硬编码的 `smoke_l40s` 覆盖。

## Volume

不包装 Modal Volume CLI：

```bash
modal volume get modal-lab-pixal3d-l40s-outputs meshes/smoke_l40s.glb ./viewer/
```

## 测试

```bash
python -m unittest discover -s 005-v2-pixal3d-l40s/tests -v
python -m py_compile 005-v2-pixal3d-l40s/app.py
python main.py 005-v2-pixal3d-l40s status
python main.py 005-v2-pixal3d-l40s smoke --dry-run
```

以上测试不启动付费 GPU。

## 许可

上游 MIT（见 [UPSTREAM.md](UPSTREAM.md)）。
