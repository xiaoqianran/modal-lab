# 009-hy-worldgen

[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · **World Generation**  
全景 → 轨迹 → WorldStereo 扩帧 → 3DGS → **真正的 3D 世界**

> 前置：  
> - **007** WorldMirror 2.0（重建 / Stage4 会用到）  
> - **008** HY-Pano（产出 `panorama.png`）  
> 详细阶段与省钱策略 → **[PLAN.md](PLAN.md)**

## 还有没有步骤？

**有。** 008 只做完「全景」。要出最终世界，官方还有：

| # | 阶段 | 脚本 | 本实验 |
|---|---|---|---|
| 0 | 全景 | panogen | **008**（已完成） |
| 1 | 轨迹规划 WorldNav | `traj_generate.py` | **009** |
| 2 | 轨迹渲染 | `traj_render.py` | **009** |
| 3 | 世界扩展 WorldStereo-2 ~17B | `video_gen.py` | **009**（最贵） |
| 4 | GS 数据 | `gen_gs_data.py` | **009** |
| 5 | 3DGS 训练导出 | `world_gs_trainer.py` | **009** |

## 成本警告

官方推荐 **≥4 GPU**（例 8×H20）+ 独立 **vLLM VLM**。  
比 008 的 **~$0.11/张全景** 贵得多。默认 **分 stage、可中断**，禁止默认 8 卡全开。

## 命令（骨架）

```bash
python main.py 009 status
python main.py 009 prepare --from-008 smoke_qwen   # 用 008 全景建 scene
# 后续 stage 见 PLAN — 需确认预算后再跑
```

## Volume（规划）

| Volume | 用途 |
|---|---|
| `modal-lab-hy-worldgen-weights` | WorldStereo / VLM 等 |
| `modal-lab-hy-worldgen-outputs` | scene 中间产物 + 最终 3DGS |
| （复用）`modal-lab-hy-pano-outputs` | 读 008 全景 |
| （复用）`modal-lab-hy-world-2-weights` | WorldMirror（Stage4） |

见 [UPSTREAM.md](UPSTREAM.md) · [PLAN.md](PLAN.md)。
