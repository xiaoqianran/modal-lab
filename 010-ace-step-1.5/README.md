# 010 · ACE-Step 1.5

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 开源音乐生成 · 默认 L4。

010 已迁移到 v2：一个 `app.py` 同时拥有 benchmark 不变量、CLI、结构化 run 元数据和 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 成本策略

```text
默认 GPU    L4
下载        CPU-only
smoke       20s lo-fi instrumental
thinking    false
init_lm     false
steps       8
```

## 用法

```bash
python main.py 010 status
python main.py 010 check
python main.py 010 download --dry-run --force
python main.py 010 download

python main.py 010 smoke --dry-run --seed 7
python main.py 010 smoke --gpu L4

python main.py 010 t2m --dry-run \
  --caption 'dreamy synthwave sunset drive' \
  --lyrics '[verse] hello' \
  --duration 25 \
  --bpm 120 \
  --thinking \
  --init-lm \
  --vocal \
  --steps 12

# 解析每个 run 的 meta.json
python main.py 010 list-outputs
```

`smoke` 是固定 benchmark，不允许 thinking/LM/vocal 改变语义。自由参数只属于 `t2m`：

```text
example
caption
lyrics
duration
bpm
seed
thinking
init_lm
instrumental/vocal
steps
audio_format
dit_model
lm_model
```

## Volume

v2 删除纯 `ls/pull` 包装：

```bash
modal volume ls modal-lab-ace-step-1.5-outputs runs
modal volume get modal-lab-ace-step-1.5-outputs runs/smoke_lofi ./010-ace-step-1.5/outputs
```

`list-outputs` 仍保留，因为它会读取 `meta.json` 并返回 audio / size / wall / success，而不是简单文件列表。

## 模型

```text
DiT  acestep-v15-turbo
VAE
Qwen3-Embedding-0.6B
LM   acestep-5Hz-lm-1.7B
```

上游 MIT。GPU 对照见 [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)，试听见 [`gallery/`](gallery/)。

## 测试

```bash
python -m unittest discover -s 010-ace-step-1.5/tests -v
python -m py_compile 010-ace-step-1.5/app.py
python main.py 010 status
python main.py 010 smoke --dry-run
```

以上测试不启动付费 GPU。
