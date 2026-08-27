# 009-hy-worldgen

[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · **World Generation**  
全景 → 轨迹（**Qwen3-VL-8B**）→ WorldStereo 扩帧 → WorldMirror 对齐 → 3DGS → **3D 世界**

009 已迁移到 v2：一个 `app.py` 同时拥有 pipeline planning、VLM 生命周期、CLI 与 Modal remote functions；`run.py -> modal_app.py` 已删除。

> 前置：  
> - **007** WorldMirror 2.0（重建权重 / Stage3 对齐）  
> - **008** HY-Pano（产出 `panorama.png`）  
> 详细阶段与省钱策略 → **[PLAN.md](PLAN.md)**  
> 结果预览 → **[gallery/](gallery/)**

## v8 变更（Stage 1+2 官方 VLM）

- **官方 VLM**：`Qwen/Qwen3-VL-8B-Instruct` via **vLLM**（容器内起服）
- **`stage12`**：Stage1 + Stage2 **共用一次** VLM 生命周期（推荐）
- 默认 flags：`--force-vlm --apply-nav-traj --apply-up-route`（`--no-apply-recon-iteration`）
- 同卡共享：`vlm_mode=share`，`gpu_memory_utilization≈0.35–0.38`（给 MoGe/SAM 留显存）
- 多卡可 `split`：VLM 落最后一张卡

## Smoke 结果（历史 · 单卡 RTX-PRO-6000 · 无 VLM 极简）

| Stage | 内容 | ok 阶段估费 |
|------|------|-------------|
| 1 | traj_generate · nframe=21 | ~$0.05 |
| 2 | traj_render · 3 traj | ~$0.33 |
| 3 | WorldStereo-memory-dmd + WorldMirror (SDPA) | ~$0.12 |
| 4 | gen_gs_data · 127 cams | ~$0.05 |
| 5 | 3DGS 4000 steps · ~1.36M GS | ~$0.18 |
| **Σ ok stages** | | **~$0.72** |

3DGS val：PSNR **19.96** / SSIM **0.73** / LPIPS **0.19**

> 加上官方 Qwen3-VL-8B 后 Stage1/2 会更贵（VLM 冷启动 + 推理），预计 Stage1+2 **+$1–4** 量级。

## 命令

```bash
# 状态
python main.py 009 status

# 0. 接 008 全景
python main.py 009 prepare --from-008 smoke_qwen

# 1. 下官方 VLM 权重（CPU volume，可先跑）
python main.py 009 download --which vlm
# 可选：python main.py 009 download --which worldstereo-dmd

# 2. ★ 推荐：Stage1+2 一次跑完（官方 Qwen3-VL-8B）
python main.py 009 stage12 --gpu RTX-PRO-6000 --nframe 16

# 关闭 nav（更省 SAM/ZIM，仍 force_vlm 写 meta/objects 路径）
python main.py 009 stage12 --gpu RTX-PRO-6000 --no-apply-nav-traj

# 分 stage（keep-vlm 可链式）
python main.py 009 stage 1 --gpu RTX-PRO-6000 --keep-vlm
python main.py 009 stage 2 --gpu RTX-PRO-6000

# 多卡：VLM 占最后一张
python main.py 009 stage12 --gpu H100:2 --vlm-mode split

# 全 pipeline smoke（含 3–5；Stage3 仍最贵）
python main.py 009 smoke --gpu RTX-PRO-6000 --nframe 16 --max-steps 4000
```

## v2 边界

实验入口只拥有世界生成领域动作：

```text
status
prepare
download
stage 1..5
stage12
smoke
```

旧 `--detach` 不再作为实验参数，因为它只是 Modal 运行方式。需要 detached execution 时直接：

```bash
cd 009-hy-worldgen
modal run --detach app.py stage12 --gpu RTX-PRO-6000 --nframe 16
```

`stage12` 保留为真实 workflow 边界：Stage1 和 Stage2 共用一次 Qwen3-VL 生命周期；它不是 CLI 包装。

## 无成本验证

```bash
python -m unittest discover -s 009-hy-worldgen/tests -v
python -m py_compile 009-hy-worldgen/app.py
python main.py 009 status
python main.py 009 smoke --dry-run
```

以上验证不启动付费 GPU。

## 官方 flags 对照

| 官方脚本 | 本仓库默认 |
|---|---|
| `--llm_name Qwen/Qwen3-VL-8B-Instruct` | ✅ 固定 |
| `--force_vlm` | ✅ 默认开（`--no-force-vlm` 可关） |
| `--apply_nav_traj` | ✅ 默认开 |
| `--apply_up_route` | ✅ 默认开 |
| `--apply_recon_iteration` | ❌ 默认关（smoke 省轨迹） |

## Volume

| Volume | 用途 |
|---|---|
| `modal-lab-hy-worldgen-weights` | **Qwen3-VL-8B** + WorldStereo + HY-WorldMirror-2.0 |
| `modal-lab-hy-worldgen-outputs` | scene 中间产物 + 3DGS |
| （复用）`modal-lab-hy-pano-outputs` | 读 008 全景 |

见 [UPSTREAM.md](UPSTREAM.md) · [PLAN.md](PLAN.md) · [gallery/](gallery/)。
