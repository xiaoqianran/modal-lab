# 005-v3-pixal3d-pro6000 — Pixal3D on **RTX PRO 6000 (sm_120)**

> **状态：已在 Modal PRO 6000 端到端跑通**（smoke → GLB）。方案见 [`SOLUTION.md`](SOLUTION.md)，实测见 [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)。

005-v3 已迁移到 v2：一个 `app.py` 同时拥有 CLI、sm_120 wheel build 生命周期和 Modal worker，不再使用 `run.py -> modal_app.py` 翻译层。

| | 005 | 005-v2 | **005-v3** |
|--|-----|--------|------------|
| GPU | H100 | L40S | **RTX-PRO-6000** |
| SM | 90 | 89 | **120** |
| torch | 2.6 cu124 | 2.6 cu124 | **2.11.0+cu128** |
| 出 GLB | ✅ | ✅ | ✅ **~230s · ~$0.19** |

## 实测 smoke

| GPU | 时间 | VRAM | 估费 |
|-----|------|------|------|
| **PRO 6000** | **230 s** | **15.6 GB** | **~$0.19** |

本地：[`viewer/index.html`](viewer/index.html) + [`viewer/smoke_pro6000.glb`](viewer/smoke_pro6000.glb)

## 栈（Plan A*）

```text
镜像: nvidia/cuda:12.8.1-devel-ubuntu24.04
torch: 2.11.0+cu128
TORCH_CUDA_ARCH_LIST=12.0
NATTEN_CUDA_ARCH=12.0
ATTN_BACKEND=sdpa
扩展: nvdiffrast · nvdiffrec · flex_gemm · cumesh · o_voxel · drtk · natten
```

## 用法

```bash
# 纯本地固定栈
python main.py 005-v3-pixal3d-pro6000 status

# probe 可先 dry-run
python main.py 005-v3-pixal3d-pro6000 probe --dry-run
python main.py 005-v3-pixal3d-pro6000 probe

# wheel lifecycle
python main.py 005-v3-pixal3d-pro6000 build-sm120 --dry-run --only natten
python main.py 005-v3-pixal3d-pro6000 build-sm120 --i-know-this-costs-money
python main.py 005-v3-pixal3d-pro6000 verify --i-know-this-costs-money
python main.py 005-v3-pixal3d-pro6000 download --i-know-this-costs-money

# smoke
python main.py 005-v3-pixal3d-pro6000 smoke --dry-run --output-name demo_pro6000
python main.py 005-v3-pixal3d-pro6000 smoke \
  --i-know-this-costs-money --output-name demo_pro6000

# 自定义图片
python main.py 005-v3-pixal3d-pro6000 i2v --dry-run \
  --image-url https://example.com/chair.png \
  --output-name chair
```

`build-sm120 --only` 不复制远程 builder 的 package schema；字符串直接交给 `build_sm120()` 校验，因此 builder 新增 package 时无需同步第二份 choices。

`smoke --output-name` 现在会真正进入 `image_to_3d()`；旧双入口中它会被硬编码的 `smoke_pro6000` 覆盖。

## Volume

直接使用 Modal：

```bash
modal volume get modal-lab-pixal3d-pro6000-outputs meshes/smoke_pro6000.glb ./viewer/
```

## 测试

```bash
python -m unittest discover -s 005-v3-pixal3d-pro6000/tests -v
python -m py_compile 005-v3-pixal3d-pro6000/app.py
python main.py 005-v3-pixal3d-pro6000 status
python main.py 005-v3-pixal3d-pro6000 smoke --dry-run
```

以上测试不启动付费 GPU。

## 相关

- 官版 H100：[`005-pixal3d`](../005-pixal3d/)  
- L40S：[`005-v2-pixal3d-l40s`](../005-v2-pixal3d-l40s/)  
- 上游 MIT：见 [UPSTREAM.md](UPSTREAM.md)
