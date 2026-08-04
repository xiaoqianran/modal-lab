# MiniMax H3 单卡基准：PRO 6000 vs A100-80GB vs L40S

实验：`004-minimax-h3`  
权重：Comfy-Org pruned INT8 ~42.5GB · ComfyUI 0.30 · `--gpu-only`  
任务：同一 prompt · **864×480 · 5s · 20 steps · seed 42**

**成片只在远程 Volume：`modal-lab-minimax-h3-outputs/videos/`**  
（不自动出现在仓库本地文件夹）

## 对比指标

| 指标 | 含义 |
|------|------|
| Peak VRAM | nvidia-smi 1Hz 峰值 |
| 纯生成 / 端到端 | 墙钟 |
| 估 GPU 费 | 端到端 × Modal 单价 |

单价：$0.000842 PRO · $0.000694 A100-80 · **$0.000542 L40S**

---

## 实测总表

| 指标 | **PRO 6000** | **A100-80GB** | **L40S** |
|------|--------------|---------------|----------|
| 实卡 | Blackwell 96GB | A100-SXM4-80GB | **L40S 48GB（上报 44.4）** |
| **Peak VRAM** | **42.97 GB** | **43.38 GB** | **42.76 GB** |
| 纯生成 | **159.6 s** | 288.8 s | 355.6 s |
| 端到端 | **180.7 s** | 303.9 s | 373.7 s |
| 估 GPU 费 | **~$0.152** | ~$0.211 | ~$0.203 |
| 相对 PRO 生成 | 1.0× | 1.81× 慢 | **2.23× 慢** |
| Volume 文件 | `videos/t2v_shinkai_pro6000.mp4` | `videos/t2v_shinkai_a100.mp4` | **`videos/t2v_shinkai_l40s.mp4`** |
| 字节 | — | 834015 | **787237** |

### 浏览器直接看（远程 Volume 经 HTTP）

- L40S：https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=t2v_shinkai_l40s  
- PRO：https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=t2v_shinkai_pro6000  
- A100：https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=t2v_shinkai_a100  
- 最新：https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=latest  
- 列表：https://seachenxyt--modal-lab-minimax-h3-index.modal.run  

CLI：

```bash
modal volume ls modal-lab-minimax-h3-outputs videos
```

---

## 结论

1. **显存三卡都是 ~43GB 峰值** → 当前设定装得下 L40S（48GB，余量紧但已跑通）。  
2. **速度：PRO ≫ A100 > L40S**。  
3. **费用：PRO 最省（~$0.15）**；L40S 虽单价最低，但太慢，总账 ≈ A100。  
4. **默认仍建议 PRO 6000**；L40S 可作为「能跑、更便宜单价」的备胎，不是最优。  
5. **输出权威位置只有远程 Volume**，脚本已强调 `outputs_vol.commit()`。

原始 JSON：`modal-lab-minimax-h3-outputs/benchmarks/t2v_shinkai_{pro6000,a100,l40s}.json`
