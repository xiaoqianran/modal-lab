# 010 ACE-Step 1.5 · GPU 对照

日期：2026-08-10 · Modal 账号实测  
任务：**同一** smoke — `acestep-v15-turbo` · 20s 器乐 lo-fi · seed **42** · 8 steps · `thinking=false`（不加载 LM）

原始 meta：[`bench/`](bench/) · 汇总 [`bench/summary.json`](bench/summary.json)  
交互 Gallery：[`gallery/gpu-bench.html`](gallery/gpu-bench.html)（总表 + 各卡 FLAC）· 主曲试听见 [`gallery/index.html`](gallery/index.html)  
远程音频：`modal volume ls modal-lab-ace-step-1.5-outputs runs`

## 条件

| 项 | 值 |
|---|---|
| DiT | `acestep-v15-turbo`（主包 [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5)） |
| LM | 未加载（pure DiT） |
| 时长 / seed / steps | 20 s / 42 / 8 |
| 输出 | FLAC · 48 kHz |
| 费用估算 | Modal 公开单价（lab notes 2026-07）× **墙钟 wall_s**（含冷启动 / Volume I/O，偏保守） |

单价（$/s）：T4 0.000164 · L4 0.000222 · A10 0.000306 · L40S 0.000542 · A100-40GB 0.000583 · A100-80GB 0.000694 · H100 0.001097 · RTX-PRO-6000 0.000842

## 结果总表（按墙钟升序）

| GPU | 墙钟 s | DiT 加载 s | 生成 s | 扩散 s | VAE s | 估费 USD | 成功 | 远程 run |
|---|---:|---:|---:|---:|---:|---:|:---:|---|
| **A100-80GB** | **22.43** | 6.44 | **3.78** | **1.16** | 0.31 | 0.0156 | ✅ | `runs/bench_A100_80GB` |
| **A10** | **23.01** | 6.71 | 4.02 | 1.36 | 0.59 | **0.0070** | ✅ | `runs/bench_A10` |
| L40S | 25.17 | 6.88 | 4.90 | 1.77 | 0.32 | 0.0136 | ✅ | `runs/bench_L40S` |
| RTX-PRO-6000 | 26.37 | 7.17 | 3.98 | 1.48 | 0.29 | 0.0222 | ✅ | `runs/bench_RTX_PRO_6000` |
| **L4**（默认） | 31.04 | 9.35 | 5.93 | 2.02 | 0.80 | **0.0069** | ✅ | `runs/bench_L4` |
| **T4** | 34.43 | 11.66 | 5.59 | 1.79 | 1.20 | **0.0056** | ✅ | `runs/bench_T4` |
| A100-40GB | 35.61 | 9.74 | 7.04 | 2.46 | 0.49 | 0.0208 | ✅ | `runs/bench_A100_40GB` |
| H100 | 37.82 | 9.60 | 5.87 | 1.79 | 0.32 | 0.0415 | ✅ | `runs/bench_H100` |

> **墙钟**受调度 / 冷启动抖动影响大；比纯算力时看 **扩散 s** 与 **生成 s** 更稳。  
> 本轮 H100 / A100-40GB 墙钟偏慢，更像排队与冷启动，不是模型算不动（扩散仍 ~1.8–2.5 s）。

## 性价比结论

| 目标 | 推荐 | 理由 |
|---|---|---|
| **默认日常 / 最低成本优先** | **L4** 或 **T4** | 估费 ≈ $0.006–0.007；T4 最便宜，L4 稍快且 24GB 余量更大（开 LM 更稳） |
| **速度与费用折中** | **A10** | 墙钟 ~23 s，估费 ~$0.007，接近 L4 价格、接近高端墙钟 |
| **纯推理最快（本轮）** | **A100-80GB** | 扩散 ~1.16 s，生成 ~3.8 s；费用约 L4 的 2× |
| **大显存 / 以后开 LM·XL** | A100-80GB / H100 / RTX-PRO-6000 | 本 smoke 吃不满；留给 thinking / XL 4B |
| **不推荐本任务当默认** | H100 | 单价最高；短 20s turbo 任务费用 ≈ L4 的 **6×**，收益不明显 |

## 音质 / 效果（主观 + 客观代理）

- **同一模型权重 + 同一 seed + 同一 prompt**：各卡均成功出片（~3.0–3.2 MB FLAC）。跨架构 **bf16 非 bit 级可复现**，波形会有细微差异，但听感同属「短 lo-fi 器乐」一档。
- **T4** 文件略小（~2.97 MB），VAE 解码最慢（~1.2 s）；可用，但不建议作为开 LM 的默认卡。
- **效果差异主要来自模型档位（turbo vs sft vs XL、是否 thinking）**，不是这 8 张卡之间的数量级差距。本 bench **只比 GPU 时延与费用**，不宣称某卡「更好听」。

若要比音质，应固定 GPU，横比：

```bash
python main.py 010 t2m --example example_01 --duration 30 --thinking   # + LM
# 或换 --dit acestep-v15-sft / xl-turbo（需另下子模型）
```

## 复现

```bash
cd 010-ace-step-1.5
python main.py 010 download
for g in T4 L4 A10 L40S A100-40GB A100-80GB H100 RTX-PRO-6000; do
  safe=$(echo "$g" | tr '-' '_')
  python main.py 010 smoke --gpu "$g" --run-name "bench_${safe}"
done
modal volume ls modal-lab-ace-step-1.5-outputs runs
```

## 与默认策略

实验默认仍为 **L4**：smoke 估费最低档之一、显存够开 1.7B LM，且本轮全部卡均已冒烟通过。需要更快墙钟时用 **A10 / A100-80GB**；抠费用用 **T4**。
