# 032 · IndexTTS-2（TTS Tier A4）

IndexTeam/IndexTTS-2 · 时长 + 情感 · 默认 **L4** · fp16。

032 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_zh | zero-shot 中文 | 33.0s | **$0.0073** | 8.3s | 7.4G |
| smoke_en | zero-shot EN | 36.9s | **$0.0082** | 7.6s | 7.3G |
| smoke_emo | emo_text 悲伤 | 41.4s | **$0.0092** | 8.3s | 7.3G |

## Prompt Volume

默认 zero-shot speaker prompt 是：

```text
modal-lab-indextts2-prompts:/ref.wav
```

v2 删除了旧 `seed-prompt` 和 smoke/t2s 里的隐式自动上传。准备 prompt 直接使用 Modal 原生 CLI：

```bash
modal volume put --force \
  modal-lab-indextts2-prompts \
  ./032-indextts2/inputs/voices/ref.wav \
  ref.wav
```

这样推理命令不会在背后修改 Volume。

## 用法

```bash
# 本地固定信息
python main.py 032 status

# 远程检查权重 / prompt / outputs
python main.py 032 check

# 权重
python main.py 032 download --dry-run
python main.py 032 download

# smoke
python main.py 032 smoke --dry-run --kind zh
python main.py 032 smoke --kind zh
python main.py 032 smoke --kind en
python main.py 032 smoke --kind emo

# emotion TTS
python main.py 032 t2s --dry-run \
  --text '这些年的时光……' \
  --emo-text '极度悲伤'

# 指定 prompts Volume 中的远程 wav
python main.py 032 t2s --dry-run \
  --text '你好' --spk-audio /prompts/ref.wav
```

## Volume 输出

不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-indextts2-outputs runs
modal volume get --force modal-lab-indextts2-outputs runs/smoke_zh ./032-indextts2/outputs
```

## 测试

```bash
python -m unittest discover -s 032-indextts2/tests -v
python -m py_compile 032-indextts2/app.py
python main.py 032 status
python main.py 032 smoke --dry-run --kind emo
```

以上测试不启动付费 GPU。

许可：Bilibili IndexTTS（商用注册）。见 [`COST_BENCHMARK.md`](COST_BENCHMARK.md) · [`gallery/`](gallery/)
