# 015-xiaomi-robotics-1-robocasa365

在 Modal 上加载小米 **Xiaomi-Robotics-1** 的
[RoboCasa365](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365)
checkpoint，做 **VLA 动作生成冒烟**。

> 这东西是啥？一句话：  
> **看厨房多视角画面 + 听人话指令 → 吐出机械臂未来 16 步动作** 的机器人基础策略模型。  
> 官方在 RoboCasa365 仿真基准上跑到约 **57%** 成功率（当时 SOTA）。

完整 2500 episode 仿真评测需要 MuJoCo / RoboCasa365 环境，**不在本实验范围**  
（见 [UPSTREAM.md](UPSTREAM.md)）。这里只验证：下载权重 → 加载 → 合成观测前向 → 动作张量。

## 成本策略（默认）

| 项 | 选择 | 原因 |
|---|---|---|
| GPU | **A100-40GB** | 5.4B bf16 ≈10GB 权重 + 多视角激活，40GB 够用 |
| 下载 | **CPU only** | ~10GB 落 Volume，复用不计 GPU 费 |
| smoke | 合成三视角 × 4 帧历史 · 指令 `close the blender lid` | 零仿真依赖 |
| attn | **sdpa**（可换 `flash_attention_2`） | 镜像冷启动更稳 |
| 容器 | 无 keep_warm | 跑完放掉 GPU |

可选：`--gpu A100-80GB` / `H100` / `L40S`。

## 快速开始

```bash
# 需已 modal token set
python main.py 015 status
python main.py 015 download          # CPU 拉 HF 权重 → Volume
python main.py 015 smoke             # A100-40GB 合成观测 → 动作
python main.py 015 infer --instruction "open the drawer"
python main.py 015 ls
python main.py 015 pull --remote runs/smoke_close_blender_lid
```

或：

```bash
cd 015-xiaomi-robotics-1-robocasa365
python run.py smoke --gpu A100-40GB
python run.py infer --instruction "turn on the stove" --run-name stove_on
```

## 远程产物

| Volume | 路径 |
|---|---|
| `modal-lab-xr1-robocasa365-weights` | `/Xiaomi-Robotics-1-RoboCasa365/`（完整 HF snapshot） |
| `modal-lab-xr1-robocasa365-outputs` | `runs/<name>/{actions_*.npy,meta.json,input_views.jpg}` |

```bash
modal volume ls modal-lab-xr1-robocasa365-outputs runs
modal volume get modal-lab-xr1-robocasa365-outputs runs/smoke_close_blender_lid ./outputs/
```

## 模型要点

| 组件 | 说明 |
|---|---|
| 架构 | MiBoT = **Qwen3-VL-4B** + **DiT** action head（MoT 式） |
| 参数量 | ~5.4B（HF metadata） |
| 输入 | 左/右 agentview + wrist 视频历史 · 语言指令 · 本体感觉 state |
| 输出 | `(16, 60)` 归一化动作；RoboCasa365 取前 **12** 维 |
| robot_type | `robocasa365`（processor `action_config` 键） |
| 官方 SR | **57.28%** @ target50 / 2500 eps |

数据流（本实验）：

```text
合成 RGB 帧 ×3 相机 ×4 历史
        + 指令文本
        + state (1,4,60)
        ↓
  MiBotProcessor.apply_chat_template
        ↓
  MiBoTForActionGeneration (5-step denoise)
        ↓
  actions (1,16,60) → decode_action → EE12
```

## 它有什么用？

1. **通用家务操作策略**：关盖、开关抽屉/门、操作灶台/微波炉、抓放物体…  
2. **仿真基准打榜**：RoboCasa365（365 日常任务、2500 厨房场景）是目前家庭移动操作的主战场之一。  
3. **真机迁移底座**：论文强调 100K+ 小时真实轨迹预训练 → 少样本后训练适配新任务/新本体。  
4. **研究对照**：和其他 VLA（OpenVLA、π0、GR00T…）比 scaling / 数据配方 / DiT head。

本 015 实验的用途更窄：**在 Modal 上把官方 HF 权重跑通一遍**，确认栈可用，再决定是否接完整 eval 或真机。

## 许可

上游 **Apache-2.0**。生成/部署请自行合规。
