# Upstream

| 项 | 值 |
|---|---|
| 模型 | **Xiaomi-Robotics-1**（内部名 MiBoT · VLA） |
| 论文 | [arXiv:2607.15330](https://arxiv.org/abs/2607.15330) · *Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories* |
| 项目页 | https://robotics.xiaomi.com/xiaomi-robotics-1.html |
| 代码 | https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 |
| 本实验权重 | https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 |
| 同系列 | [RoboCasa](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa) · [VLABench](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-VLABench) · [5B base](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) |
| 基准 | [RoboCasa365](https://robocasa.ai/) · [论文](https://arxiv.org/abs/2603.04356) |
| 许可 | Apache-2.0 |

## 模型是什么

**Vision-Language-Action (VLA)** 基础模型：

1. **VLM backbone**：Qwen3-VL-4B（看图/视频 + 读语言指令）
2. **DiT action head**：在 VLM KV-cache 条件下做 flow/diffusion 式动作去噪
3. **输出**：每 query 生成 **16 步 × 60 维** 归一化动作；RoboCasa365 实际用前 **12 维**（EE 位姿 + gripper 等打包）

预训练：>100K 小时真实操作轨迹（UMI 采集 + 自动语言标注场景状态转移）。  
本 checkpoint：在 RoboCasa365 上后训练/对齐后的评测权重。

## 官方参考结果（RoboCasa365）

| 项 | 值 |
|---|---|
| split | `pretrain` |
| task set | `target50`（50 任务） |
| episodes | 2500（50 × 50） |
| success | 1432 |
| **episode success rate** | **57.28%**（论文约 57.6%，SOTA） |

评测架构：`deploy/server.py`（模型）↔ socket ↔ `eval_robocasa365/` 客户端（MuJoCo 仿真）。

## 官方环境（摘录）

```text
Python 3.11–3.12
PyTorch 2.8.0 + CUDA 12.8
transformers==4.57.1   # 版本钉死
FlashAttention 2（可选；本实验默认 sdpa）
trust_remote_code=True
```

## 与本实验差异

| 官方完整评测 | 本 Modal 实验 015 |
|---|---|
| RoboCasa365 MuJoCo 环境 + 50 任务 × 50 episode | **不跑仿真** |
| 8×GPU server + 动态 rollout 队列 | 单卡一次前向 |
| 真实仿真相机观测 | **合成**三视角帧 |
| 产出 success rate / 视频 | 产出 `actions_*.npy` + `meta.json` |

目标：用最低成本验证 **权重可加载、processor 协议正确、动作 chunk 可生成**，作为后续接仿真 / 真机的地基。
