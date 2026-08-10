# 008-hy-pano

[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · **全景生成（HY-Pano 2.0）**

> **默认：Qwen+LoRA · GPU = RTX-PRO-6000（实测最省 ~$0.11）**  
> 全量 80B 默认关闭。后续世界生成 → [009-hy-worldgen](../009-hy-worldgen/)

## 实测 GPU（desk · 960×1952 · 40 步 · bf16）

| GPU | total | est $ |
|---|---:|---:|
| **RTX-PRO-6000** | **132s** | **$0.111** ← 默认 |
| H100 | 111s | $0.122 |
| A100-80GB | 208s | $0.144 |

## 命令

```bash
python main.py 008 download --backend qwen
python main.py 008 smoke --backend qwen          # PRO 6000
python main.py 008 smoke --backend qwen --gpu H100
```

## Volume

- `modal-lab-hy-pano-weights` · `modal-lab-hy-pano-outputs`
