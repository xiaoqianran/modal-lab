# H100 与 RTX PRO 6000 实测

2026-07-30 在 Modal 上使用同一份 1677 页
`books/EN-算法导论4.pdf`，对 002 Unlimited-OCR 和 003 MinerU 分别进行
H100 与 RTX PRO 6000 对照。这里的 RTX 卡是 Modal 实际分配的
**NVIDIA RTX PRO 6000 Blackwell Server Edition**，不是同名工作站卡。

## 测试条件

| 项目 | H100 | RTX PRO 6000 |
|---|---|---|
| Modal 请求 | `H100!`，固定为 H100，不自动升级 H200 | `RTX-PRO-6000` |
| 实际设备 | NVIDIA H100 SXM，Hopper GH100 | NVIDIA RTX PRO 6000 Blackwell Server Edition，Blackwell GB202 |
| 显存 | 80 GB HBM3 | 96 GB GDDR7 |
| 显存带宽（NVIDIA 标称） | 3.35 TB/s | 1,597 GB/s |
| FP16/BF16 Tensor 峰值（NVIDIA 标称） | 1.979 PFLOPS（稀疏） | 1 PFLOP |
| 最大功耗 | 700 W，可配置 | 600 W，可配置 |
| Compute Capability | 9.0 | 12.0 |
| Modal GPU 单价（2026-07-30） | $0.001097/秒 | $0.000842/秒 |
| 002 attention backend | FlashAttention 3 | FlashInfer |
| 003 vLLM attention backend | FlashAttention 3 | FlashAttention 2 |

纸面上 RTX 多 16 GB 显存且便宜，但 H100 的显存带宽约为 2.10 倍，
NVIDIA 公布的 FP16/BF16 Tensor 峰值也接近 2 倍。跨架构峰值不能代替
应用实测，因此下文只用同一程序、同一输入的页吞吐作最终判断。

Modal 说明所有平台 H100 都是 SXM 版本，并建议基准测试使用 `H100!` 避免
自动升级为 H200。Modal 也明确指出 Hopper 当前拥有更成熟的预编译 kernel
支持；Blackwell 的理论算力并不保证现有推理栈立即兑现为更高速度。

资料：

- [Modal GPU 类型和 H100! 说明](https://modal.com/docs/guide/gpu)
- [Modal 当前资源单价](https://modal.com/pricing)
- [NVIDIA H100 官方规格](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- [NVIDIA 支持 GPU 与 Compute Capability](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)

## 002 Unlimited-OCR

条件：200 DPI、`gundam`、`max_tokens=4096`、每档 300 秒。RTX 的
c16/c24/c32 使用三个独立 GPU 容器并行测试；H100 是此前相同条件的测试。

| GPU | 并发 | 完成页 | 页/分钟 | 输出 token/s | GPU util 均值 / P95 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| H100 | 16 | 767 | 151.701 | 1,494.846 | 43.02% / 70% | 约 65.7 GiB |
| **H100** | **24** | **848** | **167.348** | **1,667.128** | **38.17% / 68%** | **约 65.7 GiB** |
| H100 | 32 | 733 | 144.578 | 1,421.787 | 33.00% / 66% | 约 65.7 GiB |
| RTX PRO 6000 | 16 | 564 | 111.256 | 1,103.284 | 46.01% / 74% | 80,702 MiB |
| **RTX PRO 6000** | **24** | **738** | **145.081** | **1,433.662** | **51.35% / 76%** | **80,711 MiB** |
| RTX PRO 6000 | 32 | 665 | 130.699 | 1,294.203 | 43.00% / 73% | 80,708 MiB |

c24 遥测：

| GPU | GPU util 均值 / P95 / 最大 | 显存峰值 | 功耗均值 / 峰值 |
|---|---:|---:|---:|
| H100 | 38.17% / 68% / 未记录 | 67,270 MiB | 270.49 / 310.48 W |
| RTX PRO 6000 | 51.35% / 76% / 80% | 80,711 MiB | 247.52 / 279.20 W |

两张卡的最优点都是 c24。H100 页吞吐比 RTX PRO 6000 高 **15.35%**；
RTX 每秒价格低 **23.25%**，所以只计算 GPU 的持续推理成本时，RTX 反而
低约 **11.47%**：

| GPU | 最优速度 | GPU 成本/页 | 1677 页估算（含实测冷启动） | 估算 GPU 成本 |
|---|---:|---:|---:|---:|
| H100 | 167.348 页/分钟 | $0.000393 | 约 11.33 分钟 | $0.746 |
| RTX PRO 6000 | 145.081 页/分钟 | $0.000348 | 约 12.94 分钟 | $0.654 |

费用只包含 GPU，未含 CPU、内存和 Volume。H100 的正式整书实测是
1677 页、解析 616.438 秒、含冷启动约 11 分 35 秒；表中的整书数据则是
为了同口径比较而由 5 分钟样本外推。

RTX 实测任务：
[Modal run ap-vcerfAZeNLWa9wkRS36iY2](https://modal.com/apps/seachenxyt/main/ap-vcerfAZeNLWa9wkRS36iY2)

## 003 MinerU

条件：Hybrid Engine、`effort=medium`、vLLM、64 页窗口、从第一页开始解析
100 页。模型已预下载，但每张卡都包含新容器中的 predictor 初始化。

| GPU | predictor 冷启动 | 100 页总耗时 | 端到端页/分钟 | 文档阶段页/分钟 | processing window 页/分钟 |
|---|---:|---:|---:|---:|---:|
| H100 | 180.18 秒 | 241.12 秒 | 24.884 | 98.462 | 106.3 |
| RTX PRO 6000 | 233.20 秒 | 321.47 秒 | 18.664 | 67.973 | 70.5 |

GPU 遥测：

| GPU | GPU util 均值 / P95 / 最大 | 显存峰值 | 功耗均值 / 峰值 |
|---|---:|---:|---:|
| H100 | 3.32% / 24% / 100% | 56,395 MiB | 116.37 / 341.85 W |
| RTX PRO 6000 | 1.14% / 2% / 74% | 63,770 MiB | 85.95 / 356.47 W |

当前 MinerU/vLLM 组合下：

- H100 的 100 页端到端速度高 **33.33%**。
- 扣除 predictor 冷启动后，H100 文档阶段吞吐高 **44.86%**。
- RTX 冷启动长 53.02 秒，且处理阶段也更慢，不只是启动差异。
- 100 页 GPU 成本约为 H100 **$0.265**、RTX **$0.271**；RTX 低单价不足以
  抵消耗时。
- 按各自冷启动和文档阶段外推 1677 页，H100 约 **20.0 分钟 / $1.32**，
  RTX 约 **28.6 分钟 / $1.44**。这是样本估算，不是整书完成值。

RTX 实测任务：
[Modal run ap-uDjuDQUJFcUbQhoUMwne3C](https://modal.com/apps/seachenxyt/main/ap-uDjuDQUJFcUbQhoUMwne3C)

## 结论

- **Unlimited-OCR：优先速度选 H100，优先 GPU 单价/页选 RTX PRO 6000。**
  H100 快 15.35%，但 RTX 的单页 GPU 成本低 11.47%。
- **MinerU：当前直接选 H100。** 它同时有更低延迟、更高持续吞吐和略低的
  GPU 成本。
- **模型或上下文无法装入 H100 80 GB 时再优先 RTX。** RTX 的 96 GB 是
  硬优势；本次两项负载都能在 H100 上完成。不同 attention backend 会使用
  不同的预留策略，不能把 `nvidia-smi` 峰值直接当作最低显存需求。
- 两套程序的 GPU 平均利用率都不能单独用来判断是否“拉满”。002 是视觉
  prefill 与自回归 decode 混合负载，c32 已比 c24 退化；003 是多阶段异构
  流水线，CPU、多个 GPU 模型、VLM 和写盘交替运行，局部 GPU 已到 100%。

## 复现

```bash
# 002：各卡内部同时比较 c16/c24/c32
python main.py 002 benchmark --seconds 300 \
  --concurrencies 16,24,32 --gpu 'H100!'
python main.py 002 benchmark --seconds 300 \
  --concurrencies 16,24,32 --gpu RTX-PRO-6000

# 003：相同前 100 页
python main.py 003 benchmark --pages 100 --gpu 'H100!'
python main.py 003 benchmark --pages 100 --gpu RTX-PRO-6000
```
