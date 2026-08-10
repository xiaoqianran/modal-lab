# 008-hy-pano

[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · **全景生成（HY-Pano 2.0）**

> **省钱默认：只跑 Qwen+LoRA，不开 80B 全量。**  
> 设备/量化/成本细节 → **[PLAN.md](PLAN.md)**

## 先读这三句

1. **$10–25 是 4×H100 跑全量 80B 的天价档**，不是默认。  
2. 官方自带轻量后端 **HY-Pano-2-Qwen**（Qwen-Image-Edit + 全景 LoRA）——**穷跑就走这个**。  
3. **全量 80B 官方没有一键 INT4 包**；社区有 INT8/NF4（HunyuanImage-3），但接 panogen 费工程，**不是第一选择**。

## 后端

| 后端 | 何时 | 默认卡 | 单次粗估 |
|---|---|---|---:|
| **`qwen`（默认）** | 有预算限制时唯一推荐 | L40S / A100-80；目标压到 **L4+4bit** | **~$0.1–1** |
| `full` | 明确要官方 80B 质量 | H100:4 | **$10+**（别默认） |

## 量化现状

| 路径 | 有没有现成量化 | 实用建议 |
|---|---|---|
| full 80B | 官方无；社区 INT8/NF4（Comfy 向） | 穷 → **跳过** |
| Qwen 基座 | 社区 **NF4/4bit/GGUF** 成熟 | **省钱主路径** |
| 其它省显存 | 半分辨率、少步、cpu offload | smoke 优先用 |

## 命令

```bash
python main.py 008 status
python main.py 008 download --backend qwen     # ~59GB CPU，勿下 full
python main.py 008 smoke --backend qwen --gpu L40S
# 禁止默认：
# python main.py 008 download --backend full
# python main.py 008 smoke --backend full --gpu H100:4
```

## Volume

| Volume | 内容 |
|---|---|
| `modal-lab-hy-pano-weights` | Qwen + LoRA（可选 full，不推荐） |
| `modal-lab-hy-pano-outputs` | `runs/<name>/panorama.png` + meta |

见 [UPSTREAM.md](UPSTREAM.md) · [PLAN.md](PLAN.md)。
