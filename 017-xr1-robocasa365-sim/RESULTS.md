# 017 · XR-1 + RoboCasa365 仿真结果记录

记录时间：2026-08-10 UTC  
默认 GPU：**L40S**（非 A100-40GB）

## 总览

| 跑次 | 模式 | GPU | 墙钟 | 估费 | 视频帧 | 任务成功 | 状态 |
|------|------|-----|------|------|--------|----------|------|
| `smoke_random_v3` | 随机 | A100-40GB* | 1029 s | $0.60 | 60 | 否 | ✅ 出片 |
| **`smoke_policy_l40s_v1`** | **XR-1 闭环** | **L40S** | **375 s** | **$0.20** | **21** | 否（horizon 短） | ✅ 链路通 |

\* 早期默认 A100；现已改为 L40S。

## 关键修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `shape '[3,2,2,3,7,2,16,7,2,16]' is invalid for input of size 2125764` | `crop_ratio=0.95` 把 256 裁成 **243**（非 32 倍数） | 裁剪后 **resize 320×256**（与 015 一致） |

## smoke_policy_l40s_v1（主结果）

```text
task:        CloseBlenderLid
instruction: Close the lid blender by securely placing the lid on top.
GPU:         NVIDIA L40S (46 GB)
VRAM peak:   ~10.5 GB
load_s:      14.14
infer_s:     2.15 → 0.325（两 chunk）
steps_run:   20 / horizon 20
episode_ok:  false
pipeline:    success (error=null)
video:       runs/smoke_policy_l40s_v1/CloseBlenderLid/episode_000_seed_7_failure.mp4
video_bytes: 92120
wall_s:      374.72
cost_est:    $0.2031
attn:        eager（sdpa 回落）
cam_raw:     3× 256×256×3
policy_size: 320×256 (W×H)
```

Modal run：`ap-mwHsJGPbrz3jLpHK3hHH5c`（workspace `shuhuaqaq`）

### 为何任务失败不算 bug

官方 smoke `horizon=20` 往往不够完成「盖搅拌机盖」。本实验目标是 **闭环 + mp4**，不是刷成功率。

## 随机对照 smoke_random_v3

```text
mode: random · 60 steps · seed 7
wall_s: 1029（含首次 ~24GB 资产）
cost:   ~$0.60
video:  CloseBlenderLid_random.mp4 (214507 B)
```

## GPU 结论（本任务）

| GPU | 结论 |
|-----|------|
| **L40S** | **默认**。显存够（~10GB）、更便宜 |
| A100-40GB | 可对齐 015，非必须 |
| RTX PRO 6000 | 更贵、对 ~10GB 任务无收益 |

单价参考（$/s）：L40S **0.000542** · A100-40GB 0.000583 · PRO 6000 0.000842

## Gallery

- 本地 / 预览：`016-musicgen/gallery/`（顶栏 policy + random）
- 本目录：`gallery/data/policy_episode.mp4` · `gallery/data/CloseBlenderLid_random.mp4`
- Volume：`modal-lab-xr1-robocasa365-sim-outputs`

## 明确不做

完整 **2500 局** 官方评测（数小时～数天，常 $50–500+）。
