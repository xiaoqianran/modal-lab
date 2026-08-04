# 004-minimax-h3 — MiniMax H3 文生视频（Modal · 单卡）

ComfyUI headless + Comfy-Org 量化包，脚本出片。

| 项 | 选择 |
|----|------|
| 默认 GPU | **`RTX-PRO-6000`**（实测更快更省） |
| 备选 | `A100-80GB`（可用但更慢更贵）、`L40S`（建议下一测） |
| 权重 | ~42.5GB pruned INT8 |
| 输出 Volume | `modal-lab-minimax-h3-outputs` |

**GPU / 显存 / 费用对照 → [GPU_BENCHMARK.md](GPU_BENCHMARK.md)**

## 看视频

- 列表：https://seachenxyt--modal-lab-minimax-h3-index.modal.run  
- 最新：https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=latest  
- CLI：`modal volume get modal-lab-minimax-h3-outputs videos/latest.mp4 ./latest.mp4`

Volume 布局：

```text
videos/<name>.mp4
videos/latest.mp4
benchmarks/<name>.json   # 含 peak VRAM / 墙钟 / 估价
```

## 用法

```bash
python run.py download
python run.py smoke --gpu A100-80GB
python run.py t2v --gpu A100-80GB --output-name t2v_shinkai_a100
python run.py t2v --gpu RTX-PRO-6000 --output-name t2v_shinkai_pro6000
python run.py t2v --gpu L40S --output-name t2v_shinkai_l40s   # 下一测
python run.py list-outputs
```

默认出片：864×480 · 5s · 20 steps · seed 42。

## 许可

MiniMax H3 Community License。上游钉死见 [UPSTREAM.md](UPSTREAM.md)。
