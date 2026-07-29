# 003-mineru

使用 [MinerU](https://github.com/opendatalab/MinerU) 3.4.4 在 Modal
H100 / RTX PRO 6000 上解析 `books/EN-算法导论4.pdf`，并与
002 Unlimited-OCR 做同书对照。

## 选择

默认使用：

```text
backend=hybrid-engine
effort=medium
method=auto
formula=true
table=true
image_analysis=false（medium 的官方行为）
```

原因：该书是带文本层的数字 PDF。Hybrid 会结合原生文本/pipeline 与 VLM，
比每页全部重做视觉 OCR 更适合；`medium` 是官方默认，在 OmniDocBench v1.6
上总分 95.26，仅比 `high` 的 95.39 低 0.13，但官方报告 Linux 文本 PDF
约快 80%。需要图表语义解析时再切换 `high`。

代码也保留：

- `pipeline`：资源最省、兼容性最好，官方总分 86.47。
- `vlm-engine`：全部走 MinerU2.5-Pro-2605-1.2B，官方总分 95.30。
- `hybrid-engine --effort high`：精度优先，并启用图表分析。

以上分数来自 MinerU 官方 OmniDocBench v1.6 表，只能用于理解后端取舍；
不能直接替代在《算法导论》上的同书实测。

## 与 002 Unlimited-OCR 的差异

| 维度 | 002 Unlimited-OCR | 003 MinerU Hybrid |
|---|---|---|
| 定位 | 单一视觉生成模型 | 完整文档解析系统 |
| 数字 PDF | 每页重新视觉识别 | 可利用原生文本并处理复杂区域 |
| 公式/表格/版面 | 统一生成 Markdown | 专用 pipeline + VLM + 结构化 JSON |
| 输出 | raw/clean Markdown + 页指标 | Markdown、content list、middle JSON、图片 |
| 实测速度 | 163.228 页/分钟（整书） | 约 98.5 页/分钟（扣除冷启动） |
| 部署复杂度 | 较低 | 较高，模型与依赖更多 |
| 许可证 | MIT | Apache 2.0 基础上的 MinerU 附加条款 |

实测判断：如果目标是高吞吐的纯页面转 Markdown，002 更简单且更快；如果
目标是 RAG 数据生产、公式/表格/阅读顺序、图片资产和结构化中间结果，
MinerU 更像完整的生产系统。

## 2026-07-29 至 2026-07-30 同书实测

测试文件为 1677 页的 `EN-算法导论4.pdf`。模型 Volume 已提前下载完成，
下面的耗时不包含镜像构建和模型下载。

| 路径 | 页数 | 总耗时 | 端到端速度 | 文档阶段速度 | GPU / 显存 |
|---|---:|---:|---:|---:|---|
| Hybrid medium + Transformers（修复前回退） | 10 | 62.46 秒 | 9.606 页/分钟 | 未单独记录 | GPU 平均 5.46%，显存最大 12.1 GiB |
| Hybrid medium + vLLM（冷启动） | 10 | 186.84 秒 | 3.211 页/分钟 | 约 28.3 页/分钟 | GPU 平均 1.36%，显存最大 45.3 GiB |
| Hybrid medium + vLLM，H100（冷启动） | 100 | 241.12 秒 | 24.884 页/分钟 | 约 98.5 页/分钟 | GPU 平均 3.32%，P95 24%，最大 100%；显存最大 56.4 GiB |
| Hybrid medium + vLLM，RTX PRO 6000（冷启动） | 100 | 321.47 秒 | 18.664 页/分钟 | 约 68.0 页/分钟 | GPU 平均 1.14%，P95 2%，最大 74%；显存最大 63,770 MiB |
| 002 Unlimited-OCR c24-gundam（整书） | 1677 | 616.44 秒 | 163.228 页/分钟 | 同左 | 见 002 README |

100 页任务中，vLLM predictor 冷启动 180.18 秒，之后文档阶段约 60.94 秒；
MinerU 自身记录两个 processing window 共 56.44 秒，即 106.3 页/分钟。
两种口径的差异来自结果整理、写盘等外围工作。

若整本只在一个容器内运行一次，冷启动只支付一次。按 100 页样本外推，
H100 解析 1677 页约需 **20.0 分钟**，RTX PRO 6000 约需
**28.6 分钟**；这是估算值，不是整书完成值。002 已完成的整书实测为
**10.27 分钟**，因此在本书上 Unlimited-OCR 的持续吞吐约为 H100 MinerU
Hybrid 的 1.7 倍。

H100 的 100 页端到端速度高 33.33%，扣除 predictor 冷启动后的文档阶段
吞吐高 44.86%；按 Modal 单价估算，H100 的整书 GPU 成本也更低。详细
H100 / RTX 测试口径、费用和任务链接见
[仓库 GPU 对比](../GPU_COMPARISON.md)。

GPU 平均利用率低并不表示简单增加 batch 就能等比例提速：Hybrid 会依次
执行 PDF 渲染、layout、表格方向、VLM、公式/OCR 和结果组装，CPU 与多个
GPU 模型交替工作；瞬时 GPU 已达到 100%。100 页输出包含 1 个 Markdown、
3 个 JSON 和 45 个图片资产，结构化程度明显高于 002。

### 测试中发现并修复的问题

- Modal 给官方 vLLM 镜像注入 Python 后，最初未能导入 vLLM，MinerU 静默
  回退到 Transformers；现在会显式安装并在构建时校验 vLLM。
- 若先安装 `mineru[core]`，会得到 PyTorch 2.13，与 vLLM 0.21 的 PyTorch
  2.11 二进制不兼容。当前镜像先安装 vLLM，再安装 MinerU，最终固定为
  PyTorch 2.11.0。
- `summary.json` 现在记录 `inference_engine`，避免将 fallback 成绩误认为
  vLLM 成绩。

## 运行

```bash
# 查看配置
python main.py 003 status

# 下载 pipeline + VLM 模型
python main.py 003 download

# 先解析前 100 页作基准
python main.py 003 benchmark --pages 100

# 在 RTX PRO 6000 上跑相同基准
python main.py 003 benchmark --pages 100 --gpu RTX-PRO-6000

# 解析整本
python main.py 003 parse
```

三路对照：

```bash
python main.py 003 benchmark --pages 100 \
  --backend hybrid-engine --effort medium
python main.py 003 benchmark --pages 100 \
  --backend hybrid-engine --effort high
python main.py 003 benchmark --pages 100 --backend pipeline
```

结果使用独立 Modal Volumes：

```text
Models: modal-lab-mineru-models
Data:   modal-lab-mineru-data
Output: /outputs/EN-算法导论4/hybrid-medium
```

完整解析支持结果级续跑：检测到参数一致且状态为 completed 的
`summary.json` 时直接跳过。MinerU 3.4 本身使用滑动窗口和流式写盘处理
超长文档。

## 资源和版本

- GPU：默认 `H100!`，可用 `--gpu RTX-PRO-6000` 切换
- CPU / RAM：16 CPU / 64 GiB
- vLLM：`0.21.0-cu129`
- PyTorch：`2.11.0`（由 vLLM 0.21 锁定）
- MinerU：`3.4.4`
- 默认处理窗口：64 页
- PDF 渲染线程：8

许可证注意：MinerU 3.4.4 使用基于 Apache 2.0 的自定义许可证。对外提供
在线服务需要注明使用 MinerU；达到许可证规定的超大规模商业门槛时需另行
取得商业许可，细节见 [UPSTREAM.md](UPSTREAM.md)。
