# 012 · LeVo 2（SongGeneration v2）

Tencent **LeVo 2** / SongGeneration v2 全曲生成（歌词 + 描述 → 人声/伴奏）。

| 项 | 值 |
|----|-----|
| 默认模型 | **v2-medium**（`lglg666/SongGeneration-v2-medium` · ~12G/18G） |
| 可选 | **v2-large**（~22G/28G） |
| 默认 GPU | **L40S**（$0.000542/s · 48GB） |
| 为何不是 A100 | L40S 更便宜，显存够 medium/large；PRO 6000 更贵（~$0.000842/s）适合要极致吞吐时 |
| 许可 | **仅学术/研究/教育，禁止商用**（Tencent SongGeneration 条款） |

## 快速开始

```bash
# 下载 Runtime(~15GB) + v2-medium(~7GB)
python run.py download

# 冒烟（短英文结构歌词）
python run.py smoke

# large（仍建议 L40S；若 OOM 加 --low-mem 或换 PRO 6000）
python run.py smoke --model v2-large --gpu L40S
python run.py smoke --model v2-large --gpu RTX-PRO-6000
```

自定义：

```bash
python run.py t2a \
  --descriptions "male, rock, energetic, electric guitar" \
  --lyrics "[intro-short] ; [verse] Hello world in the neon rain. ; [chorus] We rise again. ; [outro-short]" \
  --run-name demo_rock
```

## GPU 性价比（本实验默认）

| GPU | $/s | 48GB? | 建议 |
|-----|-----|-------|------|
| **L40S** | 0.000542 | 48GB | **默认** · medium/large |
| A100-40GB | 0.000583 | 40GB | 不划算；large 可能紧 |
| A100-80GB | 0.000694 | 80GB | 不如 L40S 性价比 |
| RTX-PRO-6000 | 0.000842 | 96GB | large 无压力，更贵 |
| L4 | 0.000222 | 24GB | 仅 medium + 可能 `--low-mem` |

## 结构

- `modal_app.py` — Modal image / volume / download / smoke / t2a
- `run.py` — CLI
- `UPSTREAM.md` — pin
- `examples/` — 样例 jsonl
- `gallery/` — 成功后 HTML 试听

## 上游

- 代码：[6Morpheus6/songgeneration-tencent](https://github.com/6Morpheus6/songgeneration-tencent)（commit 见 UPSTREAM）
- Runtime：[SongGeneration-Runtime](https://huggingface.co/lglg666/SongGeneration-Runtime)
- 模型集合：[lglg666/levo](https://huggingface.co/collections/lglg666/levo)
