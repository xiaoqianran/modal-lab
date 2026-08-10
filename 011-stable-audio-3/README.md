# 011 · Stable Audio 3 Medium（Modal）

[Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) · 权重 [`stabilityai/stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium)

快速、可变时长的 latent diffusion 音乐/音效生成（44.1 kHz 立体声，最长约 **6 分 20 秒**）。本实验默认 **medium（1.4B）** + **L4**，在保证能跑的前提下压 GPU 费用。

## 模型与 GPU 选型

| 项 | 选择 | 理由 |
|---|---|---|
| **模型** | **medium** | 用户指定；开源权重里质量/时长最好的一档（large 仅 API） |
| **备选更小** | `small-music` / `small-sfx` | 更轻、可 CPU；质量与时长（≤120s）不如 medium |
| **默认 GPU** | **L4**（$0.000222/s） | medium 强制 FlashAttention 2 → 需 **Ampere+**；T4(Turing) 不可用 |
| 不可用 | T4 | FlashAttn 仅 Ampere+；T4 会 RuntimeError |
| 不推荐默认 | H100 | 推理本身亚秒～数秒，短任务墙钟被冷启动主导，性价比差 |

费用粗算（墙钟含加载，偏保守）：

| 场景 | 估时 | 估费 (L4) |
|---|---|---|
| smoke 20s · 8 steps（实测 L4） | **墙钟 43.7s**（加载 40s + 生成 3.4s） | **~$0.0097** |
| 纯生成（热模型） | ~3–4s | ≪ $0.01 |
| 峰值 VRAM | ~9.3 GB | — |

## 前置

1. **Hugging Face 门禁**：打开模型页同意 Stability Community License，账号能拉权重  
2. Modal Secret：`modal secret create huggingface HF_TOKEN=hf_xxx`（本环境已建 `huggingface`）  
3. `pip install modal` 且已 `modal token`

## 用法

```bash
cd 011-stable-audio-3
# 或: python ../main.py 011 status

python run.py status
python run.py download          # CPU · ~10GB（model + T5Gemma）
python run.py smoke             # L4 · 20s house 器乐
python run.py t2a --prompt "dreamy synthpop instrumental 120 BPM" --duration 30
python run.py pull --remote runs/smoke_house
python run.py list-outputs
```

换卡：

```bash
python run.py smoke --gpu A10
python run.py t2a --gpu A10 --prompt "..." --duration 60
```

## 架构要点

- 镜像：CUDA 12.6 + Python 3.11 + `uv` 安装 [stable-audio-3](https://github.com/Stability-AI/stable-audio-3) + **FlashAttention 2**（medium 必需）
- 权重：Volume `modal-lab-stable-audio-3-weights`（`HF_HOME` + 显式 snapshot）
- 输出：Volume `modal-lab-stable-audio-3-outputs/runs/<name>/`
- 推理：`StableAudioModel.from_pretrained("medium").generate(...)` · fp16 · chunked decode

## 实测 smoke（2026-08-10）

| 项 | 值 |
|---|---|
| GPU | L4 |
| 时长 / steps / seed | 20s / 8 / 42 |
| 墙钟 / 加载 / 生成 | 43.7s / 40.1s / 3.4s |
| 估费 | ~$0.0097 |
| 峰值 VRAM | 9.3 GB |
| 输出 | 44.1 kHz stereo FLAC · 20.0s |

> **T4 不可用**：FlashAttention 2 仅支持 Ampere+，T4(Turing) 会 `RuntimeError: FlashAttention only supports Ampere GPUs or newer`。

## Gallery

生成试听：打开 [`gallery/index.html`](gallery/index.html)。

## 许可

- [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3/blob/main/LICENSE.md)
- 含 [Gemma Terms](https://ai.google.dev/gemma/terms) 组件（T5Gemma 文本条件）
- 商用请查阅 [stability.ai/license](https://stability.ai/license)
