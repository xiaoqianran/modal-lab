# 011 · Stable Audio 3 Medium

[Stability AI Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) · gated `stabilityai/stable-audio-3-medium` · 默认 L4。

011 已迁移到 v2：一个 `app.py` 同时拥有 CLI、结构化 run 元数据和 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 用法

```bash
python main.py 011 status
python main.py 011 check
python main.py 011 download --dry-run --force
python main.py 011 download

python main.py 011 smoke --dry-run --duration 12 --seed 7
python main.py 011 smoke

python main.py 011 t2a --dry-run \
  --prompt 'dreamy synthpop instrumental 120 BPM' \
  --negative-prompt 'distortion' \
  --duration 30 \
  --steps 12 \
  --cfg-scale 1.5 \
  --format wav

# 读取每个 run/meta.json 的 GPU / wall / cost / success 汇总
python main.py 011 list-outputs
```

`list-outputs` 保留，因为它不是 `modal volume ls` 的同义包装，而是解析实验自己的 `meta.json`。纯文件操作则直接使用 Modal：

```bash
modal volume ls modal-lab-stable-audio-3-outputs runs
modal volume get modal-lab-stable-audio-3-outputs runs/smoke_house ./011-stable-audio-3/outputs
```

## Smoke 不变量

```text
prompt = uplifting house instrumental
steps = 8
cfg_scale = 1.0
audio_format = flac
model = medium
```

`duration / seed / gpu` 可用于成本对照；通用生成参数只属于 `t2a`。

## GPU

Stable Audio 3 Medium 要求 FlashAttention 2 / Ampere+：

```text
默认 L4
T4 不支持
```

历史 smoke：20s / 8 steps，L4 墙钟约 43.7s，估费约 $0.0097，峰值约 9.3GB。

## 测试

```bash
python -m unittest discover -s 011-stable-audio-3/tests -v
python -m py_compile 011-stable-audio-3/app.py
python main.py 011 status
python main.py 011 smoke --dry-run
```

以上测试不启动付费 GPU。
