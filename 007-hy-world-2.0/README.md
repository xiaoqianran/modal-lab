# 007-hy-world-2.0

在 Modal 上以**最低成本**跑通 [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) 的
**World Reconstruction（WorldMirror 2.0）** 路径。

## 为什么只跑 WorldMirror 2.0？

| 组件 | 参数量 | 本实验 |
|---|---|---|
| **WorldMirror 2.0** | **~1.2B** · ~5GB | **默认（便宜）** |
| HY-Pano 2.0 | ~80B | 不跑 |
| WorldStereo 2.0 | ~17B | 不跑 |
| 完整 World Generation 四阶段 | 上述全部 | 不跑 |

完整文/图 → 可导航 3D 世界链路成本高几个数量级。本实验对齐官方「先 worldrecon」建议。

## 与 006 的区别

| | **006 HunyuanWorld-Mirror** | **007 HY-World 2.0** |
|---|---|---|
| 上游 | [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)（独立仓库 · WorldMirror **1.x**） | [HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) 框架里的 **WorldMirror 2.0** |
| 权重 | `tencent/HunyuanWorld-Mirror` | `tencent/HY-World-2.0` / `HY-WorldMirror-2.0` |
| 能力范围 | 仅多视图 3D recon | 仓库含 panogen / worldgen / worldrecon；**本实验只启用 worldrecon** |
| 环境 | CUDA 12.4 · torch 2.4 · py3.10 | CUDA 12.8 · torch 2.7.1 · py3.11 |
| 默认 GPU | L4 | T4（同负载 peak ~5GB 够用） |
| smoke 样例 | Bright_Room · ~274k 点 | Desk · ~330k 点 |
| 006 L4 实测 | peak **7.66 GB** · total ~45s · ~$0.01 | 见下表 |

二者都是 feed-forward 多视图 → 深度/法线/相机/点云；007 是同一产品线的 **2.0 升级 + 更大 monorepo**，不是「换个壳包同一权重」。

## GPU 实测（同条件）

条件：`Desk` · 2 图 · `target_size=518` · bf16 · 关 sky/COLMAP/渲染视频 · **每次冷启动含权重 load**。  
单价为 lab 内记录的 Modal 公开价（$/s），`est_cost ≈ total_s × price/s`（仅 GPU 段，不含 CPU 下载）。

| GPU | $/s | load | forward+save | **total** | **est $** | peak VRAM | 实际设备 |
|---|---:|---:|---:|---:|---:|---:|---|
| **T4** | 0.000164 | 18.17s | 4.92s | **27.18s** | **$0.0045** | 4.89 GB | Tesla T4 |
| L4 | 0.000222 | 17.69s | **2.63s** | 25.96s | $0.0058 | 5.00 GB | NVIDIA L4 |
| L40S | 0.000542 | 13.02s | **1.82s** | **17.91s** | $0.0097 | 5.27 GB | NVIDIA L40S |
| RTX-PRO-6000 | 0.000842 | **11.87s** | 2.54s | 18.90s | $0.0159 | 5.45 GB | Blackwell Server Ed. |

### 怎么读这张表

- **冷启动总账（当前默认）→ T4 最省**（$0.0045）。L4 前向更快，但总时长只略短、单价更高 → 总账仍贵约 29%。
- **前向算力**（去掉 load）：L4 2.63s 最优性价比；L40S 最快前向但单价高，前向段费用反而高过 L4。
- **显存不是瓶颈**：四卡 peak 都在 **~5 GB**，T4 16GB 无 OOM。大卡并没因「显存更大」变便宜。
- **L40S / PRO6000 适合**：要更低延迟、或后续开 GS/多视图/高 res 时可能吃满算力；**不是**这个 smoke 的省钱卡。
- 若以后做 **keep_warm / 多请求摊销 load**，应用「前向时间 × 单价」重算，默认结论会偏向 L4。

## 成本策略（默认）

| 项 | 选择 |
|---|---|
| GPU | **T4**（冷启动总账最低） |
| 下载 | CPU-only → Volume |
| smoke | **Desk · 2 图 · 518 · bf16** |
| 关闭 | sky / COLMAP / 视频 / keep_warm / flash-attn |

## 用法

```bash
python main.py 007 status
python main.py 007 download
python main.py 007 smoke              # 默认 T4
python main.py 007 smoke --gpu L4
python main.py 007 infer --example Desk --max-images 4 --gpu L4
python main.py 007 ls
python main.py 007 pull --remote runs/bench_T4_desk2
```

## Volume

| Volume | 内容 |
|---|---|
| `modal-lab-hy-world-2-weights` | `HY-WorldMirror-2.0/` |
| `modal-lab-hy-world-2-outputs` | `runs/<name>/` |

## 上游

- https://github.com/Tencent-Hunyuan/HY-World-2.0  
- Weights: https://huggingface.co/tencent/HY-World-2.0/tree/main/HY-WorldMirror-2.0  
- Env: CUDA 12.8 · Python 3.11 · torch 2.7.1  

见 [UPSTREAM.md](UPSTREAM.md)。
