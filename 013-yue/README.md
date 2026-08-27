# 013 · YuE

M-A-P **YuE** 歌词→全曲（人声 + 伴奏）· Apache-2.0 · 默认 L40S。

013 已迁移到 v2：一个 `app.py` 同时拥有本地歌词输入、Stage1 归一化、CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 模型

```text
Stage1:
  en-cot -> m-a-p/YuE-s1-7B-anneal-en-cot
  en-icl -> m-a-p/YuE-s1-7B-anneal-en-icl
  zh-cot -> m-a-p/YuE-s1-7B-anneal-zh-cot

Stage2:
  m-a-p/YuE-s2-1B-general

Codec:
  m-a-p/xcodec_mini_infer
```

## 用法

```bash
python main.py 013 status
python main.py 013 check

python main.py 013 download --dry-run --stage1 zh-cot --force
python main.py 013 download --stage1 en-cot

python main.py 013 smoke --dry-run \
  --run-n-segments 2 \
  --max-new-tokens 3000 \
  --stage2-batch-size 2 \
  --repetition-penalty 1.1

python main.py 013 generate --dry-run \
  --genre 'inspiring female uplifting pop airy vocal' \
  --lyrics-file 013-yue/examples/smoke_lyrics.txt \
  --run-n-segments 2
```

歌词是本地输入边界，所以由 `app.py` 直接读取。也支持短歌词 inline：

```bash
python main.py 013 generate --dry-run \
  --genre 'rock' \
  --lyrics '[verse]\nHello world'
```

真正的生成参数直接由唯一 CLI 管理：

```text
stage1
genre
lyrics
run_n_segments
max_new_tokens
stage2_batch_size
seed
repetition_penalty
```

旧 wrapper 没暴露 `repetition_penalty`；v2 已纳入唯一入口。

## GPU

默认 L40S；多段 / 全曲可显式：

```bash
python main.py 013 smoke --dry-run --gpu RTX-PRO-6000
```

全曲 4+ segments 官方建议 80GB 级显存。

## Volume

v2 不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-yue-outputs runs
modal volume get modal-lab-yue-outputs runs/smoke_en ./013-yue/outputs
```

## 测试

```bash
python -m unittest discover -s 013-yue/tests -v
python -m py_compile 013-yue/app.py
python main.py 013 status
python main.py 013 smoke --dry-run
```

以上测试不启动付费 GPU。

## Smoke 基线

L40S · 2 segments：约 **787s / $0.43**。Gallery：[`gallery/index.html`](gallery/index.html)。
