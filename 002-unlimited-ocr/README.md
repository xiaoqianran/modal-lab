# 002-unlimited-ocr

基于百度开源的 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)，
在 Modal H100 上识别 `books/EN-算法导论4.pdf`（1677 页）。

## 当前方案

- 后端：百度仓库自带的 SGLang wheel，使用 continuous batching。
- GPU：Modal `H100!`，默认并发 24（实测吞吐最高）。
- 流水线：PDF 顺序渲染与 24 路 OCR 并行，避免先生成全部页面。
- 可靠性：每页最多重试 3 次，每 100 页提交一次 Modal Volume。
- 断点续跑：固定输出目录；已有 raw/clean/JSON 三个文件的页面会跳过。
- 输出：逐页原始 Markdown、去坐标 Markdown、指标 JSON、合并后的
  `book.md`、`summary.json` 和 SGLang 服务日志。

固定版本：

- Unlimited-OCR GitHub：`4ba2ea3eb384757710bc7f7678922b0b61045448`
- Hugging Face 模型：`3f2e9c956588f5560efcfb7c62240f5d67b63e60`
- 输入 SHA256：`a57ade157363dca885d18a95d16e634de6fe1ba74e2b7b25b7fcfece17d63b59`

## 已完成测试

旧版 Transformers 单请求 H100 仅使用约 7.56 GB 已分配显存，5 分钟完成
38 页，即 7.528 页/分钟。瓶颈是逐页自回归生成，而不是显存容量。

改用 SGLang 后，在相同 PDF、200 DPI、`gundam` 模式、
`max_tokens=4096` 下得到：

| 并发 | 测试窗口 | 完成页 | 页/分钟 | 输出 token/s | GPU util 均值 / P95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 30 秒 | 17 | 33.644 | 262.848 | 39.21% / — |
| 4 | 300 秒 | 432 | 86.142 | 852.648 | 50.48% / 68% |
| 8 | 300 秒 | 585 | 116.099 | 1,150.963 | 49.55% / 73% |
| 16 | 300 秒 | 767 | 151.701 | 1,494.846 | 43.02% / 70% |
| **24** | **300 秒** | **848** | **167.348** | **1,667.128** | **38.17% / 68%** |
| 32 | 300 秒 | 733 | 144.578 | 1,421.787 | 33.00% / 66% |

c24 比旧 H100 的页吞吐提升 22.23 倍，纯推理外推整本约 10.0 分钟；
c32 因视觉 prefill 与 decode 争用反而下降。SGLang 会预留约 56.5 GB
KV cache，因此 `nvidia-smi` 显示约 65.7 GiB，但日志中的实际 cache
使用率约 3%，显存不是当前瓶颈。

一致性检查：

- c4/c8/c16 共同完成的前 432 页，去坐标后 82.18% 文本完全一致。
- 其余页面三档之间平均最低字符相似度为 98.96%。
- 抽查差异页未发现 c24 系统性质量下降。

## 正式整书结果

2026-07-29 使用上述生产流水线从空输出目录解析全书：

| 指标 | 实测 |
|---|---:|
| 完成 | 1677 / 1677 页 |
| 解析流水线耗时 | 616.438 秒（10 分 16 秒） |
| 冷启动 | 78.749 秒 |
| 冷启动 + 解析 | 约 11 分 35 秒 |
| 端到端解析速度 | 163.228 页/分钟 |
| 输出 | 1,102,141 tokens |
| 输出吞吐 | 1,787.918 tokens/s |
| 请求 / 渲染错误 | 0 / 0 |
| GPU util 均值 / P95 / 最大 | 43.76% / 71% / 75% |
| 显存均值 / 峰值 | 67,263 / 67,272 MiB |
| 功耗均值 / 峰值 | 289.36 / 330.13 W |

Volume 中已核对 `raw.md`、清理后的 `.md` 和 `.json` 各 1677 个；
合并后的 `book.md` 为 3,015,910 bytes，含 1677 个页码标记。

完整测试过程、L4 对照、费用和质量抽查见
[BENCHMARK.md](BENCHMARK.md)。历史测试输出已按本轮要求删除，表中数据
保留用于复现与决策。

## 运行

从仓库根目录运行完整 1677 页：

```bash
python main.py 002 parse
```

或：

```bash
cd 002-unlimited-ocr
python run.py parse
```

复现 5 分钟吞吐测试：

```bash
python run.py benchmark --seconds 300 --concurrencies 24
```

并行比较多个并发档位：

```bash
python run.py benchmark --seconds 300 --concurrencies 4,8,16
```

拉取完整结果：

```bash
python run.py pull \
  --remote /outputs/EN-算法导论4/full-c24-gundam-dpi200 \
  --dest ./outputs
```

Modal 资源：

```text
App:     modal-lab-unlimited-ocr
Weights: modal-lab-unlimited-ocr-weights
Data:    modal-lab-unlimited-ocr-data
Output:  /outputs/EN-算法导论4/full-c24-gundam-dpi200
```
