# 017-xr1-robocasa365-sim

在 Modal 上跑 **RoboCasa365 仿真**，并尽量产出 **episode `.mp4` 回放**。

017 已迁移到 v2：一个 `app.py` 同时拥有仿真 planning、资产/权重生命周期、CLI 与 Modal remote functions；`run.py -> modal_app.py` 已删除。

> **016 是 MusicGen**，所以仿真开成 **017**。

相对 [015](../015-xiaomi-robotics-1-robocasa365)（只吐动作数字）：

| | 015 | **017（本实验）** |
|--|-----|-------------------|
| 输入图 | 合成色块 | **仿真器渲染的厨房** |
| 动作 | 有数字 | 有数字，**还会被执行** |
| 成功？ | 不知道 | `stats.json` |
| 视频 | 无 | **`.mp4` 回放** |

## 默认闭环

**horizon=100**（官方 smoke 是 20；100 才像真在干活）。

## 默认 GPU

**L40S**（不是 A100-40GB）。

| GPU | 对本任务 | 备注 |
|-----|----------|------|
| **L40S（默认）** | 推荐 | XR-1 峰值 ~10GB，L40S 够用且更便宜 |
| A100-40GB | 可 | 和 015 对齐时用 `--gpu A100-40GB` |
| RTX PRO 6000 | 不推荐默认 | 更贵、Blackwell 兼容坑，收益小 |

## 仿真结果是什么形式？

不是「文生视频」，而是：

1. 每一局一个 **`.mp4`**（虚拟相机录的机械臂干活过程）  
2. **`stats.json` / `meta.json`**（成功/失败、步数、耗时、费用）

官方完整榜：2500 局 → 大量 mp4 + 一个总成功率。  
本实验只做 **1 局 smoke**。

## 时间 & 费用（估）

Modal 单价量级（2026 公开价，仅 GPU 秒费，不含镜像构建）：

| 步骤 | 资源 | 时间（量级） | 费用（量级） |
|------|------|--------------|--------------|
| 首次镜像构建 | CPU build | 15–40 min | 镜像构建另计/常有免费额度 |
| `download-assets` | CPU · ~10GB | 10–30 min | **~$0.02–0.10**（一次） |
| `download-weights` | CPU · ~10GB | ~1 min（若 015 已下过则 skip） | **~$0.01–0.05** |
| **`smoke-random`** 1 局 80 步 | L40S | **3–15 min** | **~$0.05–0.30** |
| **`smoke-policy`** horizon=100 | **L40S** | **~1–3 min**（资产就绪时 ~90s） | **~$0.05–0.20** |
| 官方完整 2500 局 | 多卡长时间 | **数小时～数天** | **常 $50–500+**（不做） |

> 第一次最贵的是 **镜像 + 资产下载**；之后 Volume 复用会便宜很多。  
> 官方 smoke 是 `horizon=20`（很短）。本实验默认 **100** 才像真在干活；单局仍可能任务失败。

## 关键修复（policy shape）

`crop_ratio=0.95` 把 256 裁成 **243**，不是 32 的倍数 → Qwen-VL reshape 崩溃。  
现已：**中心裁剪后 resize 到 320×256**（与 015 一致，`do_resize=False`）。

## 用法

```bash
python main.py 017 status           # 纯本地固定信息
python main.py 017 check            # 远程 readiness / 最近 runs
python main.py 017 download-weights   # 可与 015 共用 weights Volume
python main.py 017 download-assets    # 默认只确保容器 assets + marker，不做 20GB 全量 Volume 镜像
# 只有明确需要跨冷启动复用完整资产树时：
python main.py 017 download-assets --full-cache
python main.py 017 smoke-random       # 随机乱动 → mp4
python main.py 017 smoke-policy       # XR-1 闭环 → mp4（默认 L40S）
python main.py 017 eval-mini          # 5×5 @ h=200 + CBL long @ h=500
modal volume get modal-lab-xr1-robocasa365-sim-outputs runs/<name> ./017-xr1-robocasa365-sim/outputs
```

产物 Volume：`modal-lab-xr1-robocasa365-sim-outputs`  
```text
runs/<name>/
  meta.json
  CloseBlenderLid/
    episode_000_seed_7_failure.mp4   # 或 success
    stats.json
```

## v2 CLI 边界

三个 workflow 保持独立：

```text
smoke-random  -> 仿真器随机动作闭环
smoke-policy  -> XR-1 policy 单局闭环
eval-mini     -> task × seeds grid + optional long track
```

远程函数原本已有、但旧 wrapper 没暴露的参数现在直接进入唯一 CLI：

```text
crop_ratio
num_denoise_steps
save_every_video   # eval-mini
```

例如：

```bash
python main.py 017 smoke-policy --dry-run \
  --horizon 120 --crop-ratio 0.9 --num-denoise-steps 7

python main.py 017 eval-mini --dry-run \
  --tasks OpenDrawer,CloseFridge \
  --num-seeds 3 --no-long --no-save-every-video
```

纯文件拉取不再包装，直接使用 `modal volume get`。

## 测试

```bash
python -m unittest discover -s 017-xr1-robocasa365-sim/tests -v
python -m py_compile 017-xr1-robocasa365-sim/app.py
python main.py 017 status
python main.py 017 eval-mini --dry-run
```

以上测试不启动付费 GPU。

## 设计取舍

1. **单进程闭环**（模型+仿真同容器），不搞官方 8 server socket。  
2. 资产进 **Volume**，避免每次重建镜像下 10GB。  
3. 权重复用 015 的 `modal-lab-xr1-robocasa365-weights`。  
4. 默认任务 `CloseBlenderLid`（官方 smoke 同款）。  
5. 完整 2500 局 **明确不做**——时间和钱都不划算当入门 demo。
