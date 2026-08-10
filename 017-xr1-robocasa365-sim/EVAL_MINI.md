# eval_mini · 5×5 + 长 horizon 轨道

**run**: `eval_mini_5x5_h200_long500_v1`  
**GPU**: L40S · **墙钟**: 1739 s ≈ 29 min · **估费**: **$0.94**  
**assets**: volume_symlink（未重下）

## 设计

| 轨道 | 配置 | 目的 |
|------|------|------|
| **grid** | 5 任务 × 5 seed (7–11) · **h=200** | 多 seed 成功率表 |
| **long** | CloseBlenderLid × 5 · **h=500** | 加长步数冲任务成功 |

官方对照：CBL 成功局通常 **≥236 步**，任务 horizon **900**；简单任务官方 SR 很高（OpenStandMixerHead ~98%）。

## 成功率表

| track/task | h | succ | n | SR | 备注 |
|---|---:|---:|---:|---:|---|
| **OpenStandMixerHead** | 200 | **5** | 5 | **100%** | 任务成功 ✓ |
| **TurnOnElectricKettle** | 200 | **3** | 5 | **60%** | 任务成功 ✓ |
| CloseFridge | 200 | 0 | 5 | 0% | env 创建失败 `NaN`（资产/采样） |
| TurnOnSinkFaucet | 200 | 0 | 5 | 0% | 同上 |
| CloseBlenderLid | 200 | 0 | 5 | 0% | 跑满 200 仍失败（步数偏短） |
| CloseBlenderLid **long** | 500 | 0 | 5 | 0% | 仍失败（官方 h=900，5 seed 小样本） |

### 汇总

| 集合 | succ/n | SR |
|------|--------|-----|
| **grid 全体 5×5** | 8/25 | **32.0%** |
| grid 仅可创建 env 的任务 | 8/10 | **80.0%** |
| long CBL | 0/5 | 0% |
| overall | 8/30 | 26.7% |

## 成功步数（实测）

| 任务 | seeds 成功 | success_step |
|------|------------|--------------|
| OpenStandMixerHead | 7,8,9,10,11 | 95, 93, 99, 103, **72** |
| TurnOnElectricKettle | 8,10,11 | 92, 105, 139 |

## 结论

1. **多 seed 评测链路通**，单容器加载一次模型/资产，30 局 ~$0.94。  
2. **任务可以成功**：简单任务在 h=200 内就有 **100% / 60%**。  
3. **CloseBlenderLid 仍难**：h=200 不够；h=500×5 仍 0（小样本 + 仍短于官方 900）。  
4. **CloseFridge / TurnOnSinkFaucet** 在当前资产包上报 `Probabilities contain NaN`，是 **环境创建**问题，不是策略推理失败。

## 产物

```text
runs/eval_mini_5x5_h200_long500_v1/
  summary.json
  SUMMARY.md
  grid_h200/<task>/episode_*.mp4
  long_h500/CloseBlenderLid/episode_*.mp4
```

Gallery：`gallery/data/success_OpenStandMixerHead_seed7.mp4`（任务成功回放）
