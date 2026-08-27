# 033 · F5-TTS（TTS Tier A5）

F5TTS_v1_Base · 零样本克隆 · 默认 **L4** · Code MIT / Model **CC-BY-NC**。

033 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_en | clone EN | 12.0s | **$0.0027** | 8.4s | 2.1G |
| smoke_zh | clone ZH | 11.2s | **$0.0025** | 11.0s | 2.1G |

## 用法

```bash
# 本地固定信息，不触云
python main.py 033 status

# 远程检查权重 / outputs / prompts Volume
python main.py 033 check

# 下载权重和参考音频
python main.py 033 download
python main.py 033 download --force

# 无成本 smoke 规划
python main.py 033 smoke --dry-run --kind en
python main.py 033 smoke --dry-run --kind zh --nfe-step 24

# 真正 smoke
python main.py 033 smoke --kind en
python main.py 033 smoke --kind zh

# 自定义 zero-shot TTS
python main.py 033 t2s --dry-run --text 'Hello from F5-TTS.' --lang en
python main.py 033 t2s --text 'Hello from F5-TTS.' --lang en

# 自定义参考音频 / 文本
python main.py 033 t2s --text 'Custom voice clone.' \
  --ref-audio /prompts/custom.wav \
  --ref-text 'Reference transcript.' \
  --nfe-step 24 --seed 7
```

也可以直接使用 Modal：

```bash
cd 033-f5tts
modal run app.py check
modal run app.py download
modal run app.py smoke --kind zh
```

## Volume 操作

v2 不再包装 Modal 自带的 `volume ls/get`：

```bash
modal volume ls modal-lab-f5tts-outputs runs
modal volume get --force modal-lab-f5tts-outputs runs/smoke_en ./033-f5tts/outputs
```

## CLI 边界

`smoke` 只维护 EN / ZH 两种固定基线。真正属于 F5-TTS 推理的参数由 `t2s` 直接暴露：

```text
lang
ref_audio
ref_text
nfe_step
seed
```

不再经过第二层 wrapper 转译。

## 测试

```bash
python -m unittest discover -s 033-f5tts/tests -v
python -m py_compile 033-f5tts/app.py
python main.py 033 status
python main.py 033 smoke --dry-run --kind zh
```

以上测试不启动付费 GPU。

本 lab **最便宜的零样本克隆**档之一。见 [`COST_BENCHMARK.md`](COST_BENCHMARK.md)
