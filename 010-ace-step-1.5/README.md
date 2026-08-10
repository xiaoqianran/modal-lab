# 010-ace-step-1.5

在 Modal 上跑 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 开源音乐生成
（DiT turbo + 可选 5Hz LM）。**音乐阶段从 010 起**（005–009 留给其他实验）。

## 成本策略（默认）

| 项 | 选择 | 原因 |
|---|---|---|
| GPU | **L4**（24GB） | 约 `$0.000222/s`；turbo DiT <4GB，主包 LM 1.7B 也够 |
| 下载 | **CPU only** | 主包 ~10GB 落 Volume，复用不计 GPU 费 |
| smoke | **20s 器乐 · thinking 关** | 最短可听成片；不走 LM 路径 |
| 容器 | 无 `keep_warm`，`scaledown_window=30s` | 跑完尽快放掉 GPU |

可选：`--gpu A10` / `L40S` / `A100-40GB` / `H100`。  
开 thinking（LM）时建议 ≥16GB 显存；L4 24GB 足够 `pt` backend。

## 快速开始

```bash
# 需已 modal token set
python main.py 010 status
python main.py 010 download          # CPU 拉 ACE-Step/Ace-Step1.5 → Volume
python main.py 010 smoke             # L4 · 20s lo-fi 器乐
python main.py 010 t2m --example example_01 --duration 30 --thinking
python main.py 010 ls
python main.py 010 pull --remote runs/smoke_lofi
```

或：

```bash
cd 010-ace-step-1.5
python run.py smoke --gpu L4
python run.py t2m --caption "dreamy synthwave sunset drive" --duration 25 --vocal
```

## 远程产物

| Volume | 路径 |
|---|---|
| `modal-lab-ace-step-1.5-weights` | `/checkpoints/{turbo,vae,embedding,lm-1.7B}` |
| `modal-lab-ace-step-1.5-outputs` | `runs/<name>/*.flac` + `meta.json` |

```bash
modal volume ls modal-lab-ace-step-1.5-outputs runs
modal volume get modal-lab-ace-step-1.5-outputs runs/smoke_lofi ./outputs/
```

## 模型

| 组件 | 来源 | 约大小 |
|---|---|---|
| DiT `acestep-v15-turbo` | [Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) | ~4.8 GB |
| VAE | 同上 | ~0.3 GB |
| Qwen3-Embedding-0.6B | 同上 | ~1.2 GB |
| LM `acestep-5Hz-lm-1.7B` | 同上 | ~3.7 GB |

镜像 build 固定上游 commit，见 [UPSTREAM.md](UPSTREAM.md)。

## GPU 对照

同条件 8 卡实测见 [GPU_BENCHMARK.md](GPU_BENCHMARK.md)（`bench/summary.json`）。

生成试听：[`gallery/index.html`](gallery/index.html)  
GPU 对照：[`gallery/gpu-bench.html`](gallery/gpu-bench.html) · [GPU_BENCHMARK.md](GPU_BENCHMARK.md)。

## 许可

上游 **MIT**。生成内容请遵循 ACE-Step 项目声明（商用友好数据集，但仍须自行合规）。
