# 005-v3 · RTX PRO 6000 (sm_120) benchmark

## Smoke · 2026-08-10

| 项 | 值 |
|----|-----|
| GPU | **NVIDIA RTX PRO 6000 Blackwell Server Edition** |
| SM | **sm_120 / (12, 0)** |
| 栈 | Plan A* · torch **2.11.0+cu128** · ARCH=12.0 源码轮子 |
| 模式 | low_vram · 1024 · seed 42 |
| 总耗时 | **230.3 s** |
| 峰值显存 | **15.58 GB** / ~96 GB |
| 估费 | **~$0.19**（$0.000842/s） |
| 输出 | `smoke_pro6000.glb` · **41.0 MB** |
| Volume | `modal-lab-pixal3d-pro6000-outputs` |

## 三卡对照（同图同 seed · low_vram 1024）

| GPU | 实验 | 时间 | 峰值 VRAM | 估费 | 状态 |
|-----|------|------|-----------|------|------|
| H100 | 005 | ~279 s | ~15.9 GB | ~$0.31 | ✅ |
| L40S | 005-v2 | ~311 s | ~15.4 GB | ~$0.17 | ✅ |
| **PRO 6000** | **005-v3** | **~230 s** | **~15.6 GB** | **~$0.19** | ✅ |

结论：PRO 6000 **已真实出片**；本次总时长快于 H100/L40S smoke（含首次 autotune），单价介于 L40S 与 H100 之间。

## 门禁

- B0 probe：capability (12,0)，arch_list 含 sm_120，matmul OK  
- build-sm120：7 wheels（含 natten + drtk）  
- verify：natten-forward PASS  
- smoke：GLB 写出
