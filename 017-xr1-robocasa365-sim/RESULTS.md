# 017 · XR-1 + RoboCasa365 仿真结果记录

更新：2026-08-10 UTC  
默认 GPU：**L40S** · 默认闭环 **horizon=100**

## 总览

| 跑次 | 模式 | GPU | horizon/步 | 墙钟 | 估费 | 帧 | 任务成功 | 状态 |
|------|------|-----|------------|------|------|----|----------|------|
| `smoke_random_v3` | 随机 | A100* | 60 | 1029 s | $0.60 | 60 | 否 | ✅ |
| `smoke_policy_l40s_v1` | XR-1 | L40S | 20 | 375 s | $0.20 | 21 | 否 | ✅ 短 smoke |
| **`smoke_policy_h100_v1`** | **XR-1** | **L40S** | **100** | **88 s** | **$0.048** | **101** | 否 | ✅ **像样闭环** |

\* 早期默认；现 L40S。

## 像样闭环：smoke_policy_h100_v1

```text
task:           CloseBlenderLid
instruction:    Close the lid blender by securely placing the lid on top.
GPU:            NVIDIA L40S (~49 GB visible)
VRAM:           ~10.5 GB
horizon:        100
steps_run:      100
episode_success:false
load_s:         14.63
infer_count:    7  (replan every 16)
infer_times_s:  1.80, 0.21×6
infer_total_s:  3.04
assets_source:  volume_symlink (reused, skip download)
assets_skipped: true
video:          episode_000_seed_7_failure.mp4
video_frames:   101
video_bytes:    285076
wall_s:         87.8
cost_est_usd:   0.0476
error:          null
attn:           eager
policy_size:    320×256
```

Modal：`ap-NV2oCu4r1R6zwlrO11n4lF`

### 相对 horizon=20 的变化

| | h=20 | **h=100** |
|--|------|-----------|
| 视频长度 | 21 帧 | **101 帧** |
| 规划次数 | 2 | **7** |
| 墙钟（资产已就绪） | ~含下载 6 min | **~88 s** |
| 费用 | ~$0.20 | **~$0.05** |
| 任务成功 | 否 | 否（单 seed） |

## 工程改进

1. **默认 horizon=100**（`DEFAULT_POLICY_HORIZON`）  
2. **输入 320×256** 修 Qwen-VL shape  
3. **资产**：优先 volume symlink；全量约 20GB 拷贝默认关闭；只有显式 `download-assets --full-cache` 才写入完整 Volume 镜像
4. 进度日志：每 10 步 / 每次 replan

## 随机对照

`smoke_random_v3`：60 步 · 214KB mp4 · 首次含资产 ~$0.60

## Gallery

- `016-musicgen/gallery/` 顶栏 h=100 policy  
- `017-.../gallery/data/policy_episode_h100.mp4`  
- Volume：`modal-lab-xr1-robocasa365-sim-outputs`

## 下一步（可选）

- 多 seed / 多任务小评测出成功率表  
- 更长 horizon（150–200）看能否盖上盖  
- 018 VLABench  

## 不做

完整 2500 局官方榜。


## eval_mini 5×5 + long（2026-08-10）

详见 [EVAL_MINI.md](./EVAL_MINI.md)。

| 集合 | SR |
|------|-----|
| grid 5×5 @ h=200 | **32%** (8/25) |
| 可运行任务 only (Mixer+Kettle) | **80%** (8/10) |
| OpenStandMixerHead | **100%** (5/5) |
| TurnOnElectricKettle | **60%** (3/5) |
| CBL h=200 / h=500 | 0% / 0% |
| 费用 | **$0.94** · 29 min · L40S |

**首次任务成功视频**：`gallery/data/success_OpenStandMixerHead_seed7.mp4`（step 95）。

## Fridge/Sink 补跑（ObjCat 修复后 · 2026-08-10）

`eval_mini_fridge_sink_h200_v1` · L40S · h=200 · 10 局 · **$0.53** · 16 min

| task | SR | 成功 |
|------|-----|------|
| CloseFridge | **1/5 = 20%** | seed8 @ step 130 |
| TurnOnSinkFaucet | **1/5 = 20%** | seed11 @ step 186 |

全部 env 创建成功（修复前 10/10 NaN）。合成 5 任务 SR **40%** (10/25)。  
视频：`gallery/data/success_CloseFridge_seed8.mp4` · `success_TurnOnSinkFaucet_seed11.mp4`
