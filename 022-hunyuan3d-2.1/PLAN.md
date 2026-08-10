# 022 Hunyuan3D-2.1 计划

## Phase 0 — Scaffold ✅
## Phase 1 — Probe ✅（L40S 8.9 · PRO 12.0）
## Phase 2 — Smoke ✅

- [x] L40S shape · full
- [x] PRO 6000 shape · full
- [x] viewer GLB + GPU_BENCHMARK

## 关键结果（chair.png）

| GPU | shape | paint | VRAM peak |
|-----|-------|-------|-----------|
| L40S | 29–30 s | 65 s | 16.5 GB full |
| PRO 6000 | 18 s | 67 s | 16.3 GB full |

## 命令

```bash
python run.py status
python run.py smoke --i-know-this-costs-money --gpu L40S --mode full
python run.py smoke --i-know-this-costs-money --gpu RTX-PRO-6000 --mode full
```
