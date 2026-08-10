# 008 · HY-Pano 2.0 规划（省钱优先）

> 上游：[Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)  
> 组件：`hyworld2/panogen` — 全景生成  
> **默认：只跑 Qwen 轻量后端；全量 80B 默认关闭。**

---

## 0. 先把「跑一次这么多？」说清楚

之前写的 **$10–25** 是 **全量 ~80B + 4×H100** 的**最贵档**，不是默认。

| 你实际会跑的 | 大概单次 GPU |
|---|---:|
| **默认：Qwen+LoRA @ L40S / A100-80** | **大约 $0.3–2** |
| 全量 80B 多卡 | **$10+**（**别默认开**） |
| 下载权重 | **CPU**，**不计 GPU 费**（但 full 要下 ~169GB，费时间/存储） |

**没那么多钱 → 永远不要开 `--backend full`。**  
官方本来就给了第二条路：**HY-Pano-2-Qwen**（基座 Qwen-Image-Edit + 全景 LoRA）。

---

## 1. 有没有量化？

### 官方 HY-World-2.0 / panogen

| 官方给的 | 量化？ | 说明 |
|---|---|---|
| HY-Pano-2 full（HunyuanImage-3 系） | **无官方现成 INT4/INT8 包** | 32 shard bf16 级，~169GB |
| **HY-Pano-2-Qwen** | 本身就是「小路径」 | 基座 ~20B + **LoRA ~850MB**，不是 80B |

代码里探测了 `bitsandbytes` / `gguf` 等，但 **没有一键 `--quantize int4` 的官方 panogen 配方**。

### 社区（全量 80B 同源 HunyuanImage-3）

| 资源 | 磁盘 | 大致 VRAM | 备注 |
|---|---:|---|---|
| [EricRollei INT8](https://huggingface.co/EricRollei/HunyuanImage-3.0-Instruct-INT8-v2) | ~83GB | 理想 80GB+；可 block-swap 到更低 | **Comfy 向**，不是官方 `pipeline.py` 一行加载 |
| [EricRollei NF4](https://huggingface.co/EricRollei/HunyuanImage-3.0-Instruct-NF4-v2) | ~48GB | 更低，靠 block-swap | 集成成本高，和 HY-Pano panogen 流水线要对齐 |
| 理论 INT4 | — | ~45GB 量级（估算） | 无官方 panogen 保证 |

→ **对「穷跑 008」不优先**：接进官方 panogen + 全景语义要额外工程，失败重试更烧钱。

### 社区（Qwen-Image-Edit 系 — **真·省钱**）

| 手段 | VRAM 量级 | 说明 |
|---|---|---|
| bf16 原版 | ~40–48GB+ | Modal：L40S 紧 / A100-80 稳 |
| **NF4 / 4bit / bitsandbytes** | **~16–24GB** | 社区大量 4090/3090 报告 |
| GGUF Q4 + Lightning 少步 | 可到 **个位数–16GB** | 多在 Comfy；要接到 panogen LoRA 需验证 |

→ **008 省钱主路径 = Qwen 后端 +（可选）4bit + 少步 + 更小分辨率**。

---

## 2. 省钱阶梯（推荐你走左边）

```text
最省（目标）     Qwen · 4bit/NF4 · L4 或 L40S · 少步 · 半分辨率
默认稳妥         Qwen · bf16 · A100-80GB · 官方默认分辨率
可尝试           Qwen · bf16 · L40S
别碰（除非有钱） full 80B · H100:4 / A100-80:4
```

### 单次成本直觉（只 GPU，粗估）

| 方案 | 卡 | $/h | 假设占卡 | 粗估 |
|---|---|---:|---:|---:|
| **Qwen 4bit smoke** | **L4** | ~0.80 | 5–15 min | **~$0.07–0.20** |
| Qwen bf16 | L40S | ~1.95 | 5–15 min | ~$0.16–0.50 |
| Qwen bf16 | A100-80 | ~2.50 | 5–15 min | ~$0.20–0.65 |
| full 80B | H100×4 | ~15.8 | 15–40 min | **$4–10+ 起，失败更贵** |

冷启动 load 仍占时间；**第二张图会便宜一点**（若 keep_warm，默认我们不开）。

---

## 3. 设备选型（穷版）

| GPU | VRAM | 用途 |
|---|---:|---|
| **L4** | 24GB | **4bit Qwen 目标卡**（最省能跑通） |
| **L40S** | 48GB | bf16 Qwen 尝试 / 4bit 很宽裕 |
| A100-80GB | 80GB | bf16 Qwen 稳妥（稍贵） |
| T4 | 16GB | 仅极致量化+少步时试；不保证 |
| H100:4 | 4×80 | **禁用**（除非你明确说开 full） |

---

## 4. 执行计划（按省钱重写）

### Phase 0 — Scaffold ✅

代码骨架、双后端入口、PLAN。

### Phase 1 — 只下 Qwen（~59GB CPU）

```bash
python main.py 008 download --backend qwen
# 禁止：download --backend full   ← 169GB，没必要
```

### Phase 2 — 最省 smoke（优先）

目标：**L4 + 量化/省显存**（实现顺序）：

1. 先 **bf16 @ L40S** 或 **A100-80** 验证 panogen 通路（一次，接受 ~$0.5 级）  
2. 再加 **`bitsandbytes` 4bit / `enable_model_cpu_offload`**，压到 **L4**  
3. 分辨率先 **480×976 或 512×1024**（官方默认 960×1952 可后开）  
4. 步数先 **20–28**（官方 50 可后开）

```bash
# 稳妥一枪（建议先这枪）
python main.py 008 smoke --backend qwen --gpu L40S

# 或
python main.py 008 smoke --backend qwen --gpu A100-80GB
```

### Phase 3 — 量化落地（工程）

- [ ] `modal_app` 增加 `--load-mode bf16|nf4|cpu-offload`
- [ ] smoke 默认 `load-mode=nf4` + 半分辨率  
- [ ] 失败自动停，不重试烧钱  
- [ ] 记录 peak VRAM / $ 进 meta

### Phase 4 — Gallery + push  

单张输入 | 全景 | 成本卡。

### 明确不做

- 默认不下 169GB full  
- 默认不租 H100:4  
- 不把 Comfy INT8 当第一优先（集成贵、排错贵）

---

## 5. 和 007 对比（为啥 007 便宜）

| | 007 WorldMirror | 008 Pano full | 008 Pano Qwen |
|---|---|---|---|
| 参数 | ~1.2B | ~80B | ~20B + LoRA |
| Peak | ~5GB | 多卡 80GB 级 | 24–48GB 可做 |
| 默认卡 | **T4 ~$0.0045** | 别跑 | **L4/L40S 角** |

---

## 6. 你现在可以怎么回我

1. **「只跑最省 Qwen」** → 下 Qwen 权重 + 先 L40S/L4 一枪（推荐）  
2. **「先别跑，只改代码支持 4bit」** → 只加量化加载，等你点头再 smoke  
3. **「我有预算上 full」** → 再说 H100:4（默认不碰）

**建议选 1 或 2。** 全量 80B 不是「量化一下就变 T4」——社区 INT8 仍常要大显存或很慢的 CPU swap。
