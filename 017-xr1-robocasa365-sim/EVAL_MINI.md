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

## 关于 CloseFridge / TurnOnSinkFaucet 的 `Probabilities contain NaN`

**不是 XR-1 推理挂了。** 证据：

| 信号 | Mixer / Kettle / CBL | Fridge / Sink |
|------|----------------------|---------------|
| `steps` | 72–500（真的在跑） | **0** |
| `infer_count` | 有 | **null**（没进策略） |
| `video` | 有 mp4 | 无 |
| 报错时机 | 无 / 任务失败 | **`Creating …` 创建环境时** |
| 错误 | — | `ValueError('Probabilities contain NaN')` |

### 机制（RoboCasa 物体采样）

任务创建厨房时会 `sample_kitchen_object`：按类别从 `objaverse` / `lightwheel` 注册表里抽模型路径。权重大致是：

```text
p[reg] = len(choices[reg]) / sum(len(choices[*]))
```

若**所有 registry 的候选路径都是空列表**，分母为 0 → 概率全是 **NaN** → `numpy.random.choice` 抛：

```text
ValueError('Probabilities contain NaN')
```

官方 / LeRobot 文档也写过：缺 object pack 时会 NaN；fixture 向任务（Mixer、Kettle、Blender）几乎不采 free object，所以仍能跑。

### 我们这边的根因

Volume 里 **其实有** fridges / sinks / objects（~24GB 完整包）。但：

1. `import robocasa` 会在**导入瞬间**扫描 `models/assets/objects/**` 填进 `ObjCat.mjcf_paths`
2. 旧代码先 `import robocasa`，**之后**才把 Volume symlink 到 package 路径
3. 扫描时目录还是空的 → `mjcf_paths=[]` 被冻住
4. Fixture 按 YAML **运行时**读盘（所以 Mixer 正常）；需要放置可抓物体的任务（关冰箱、开水龙头周围）走空的 object zoo → NaN

### 修复（已写入 `modal_app.py`）

1. **先**把 Volume 资产 symlink 到 `/opt/robocasa/.../assets`（不 import）
2. **再** `import robocasa`（各 entrypoint 也改成 assets → import）
3. 再跑一遍 `_rebuild_obj_cat_mjcf_paths()`，把仍为空的类别补扫

### 验证（2026-08-10）

```text
run: diagnose_closefridge_nan_v1
task: CloseFridge · random · 3 steps · L40S
obj_cat_rebuild: 295 cats → 3001 mjcf paths (was all empty at import)
Creating CloseFridge … OK
error: null · video frames=3 · ~$0.05
```

NaN 已消失。Fridge/Sink 现在可以正常进策略评测。

## 结论

1. **多 seed 评测链路通**，单容器加载一次模型/资产，30 局 ~$0.94。  
2. **任务可以成功**：简单任务在 h=200 内就有 **100% / 60%**。  
3. **CloseBlenderLid 仍难**：h=200 不够；h=500×5 仍 0（小样本 + 仍短于官方 900）。  
4. **CloseFridge / TurnOnSinkFaucet** 的 NaN 是 **object zoo 在 import 时为空**（不是策略）；修复后 env 可创建。

## 产物

```text
runs/eval_mini_5x5_h200_long500_v1/
  summary.json
  SUMMARY.md
  grid_h200/<task>/episode_*.mp4
  long_h500/CloseBlenderLid/episode_*.mp4
```

Gallery：`gallery/data/success_OpenStandMixerHead_seed7.mp4`（任务成功回放）
  long_h500/CloseBlenderLid/episode_*.mp4
```

Gallery：`gallery/data/success_OpenStandMixerHead_seed7.mp4`（任务成功回放）
