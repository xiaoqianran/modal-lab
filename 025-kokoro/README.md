# 025 · Kokoro-82M（TTS 用量榜 #1）

[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · Apache-2.0 · 默认 T4。

025 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 用法

```bash
python main.py 025 status
python main.py 025 check
python main.py 025 download --dry-run --model v1.1-zh
python main.py 025 download --model v1

python main.py 025 voices --dry-run --model v1
python main.py 025 voices --model v1

python main.py 025 smoke --dry-run
python main.py 025 smoke --lang zh --speed 1.1

python main.py 025 t2s --dry-run \
  --text 'Hello from Kokoro.' --voice af_bella --speed 0.95
```

中文 smoke 保留一个明确不变量：

```text
lang=zh
  -> model=v1.1-zh
  -> 默认 voice=zf_001
  -> Kokoro lang_code=z
```

显式传入中文 voice 会覆盖 `zf_001`。

`voices` 是模型自身的真实领域能力，因此继续保留；`ls/pull` 则回归 Modal Volume CLI：

```bash
modal volume ls modal-lab-kokoro-outputs runs
modal volume get modal-lab-kokoro-outputs runs/smoke_en_heart ./025-kokoro/outputs
```

## 模型

| key | HF repo |
|---|---|
| `v1` | `hexgrad/Kokoro-82M` |
| `v1.1-zh` | `hexgrad/Kokoro-82M-v1.1-zh` |

## 测试

```bash
python -m unittest discover -s 025-kokoro/tests -v
python -m py_compile 025-kokoro/app.py
python main.py 025 status
python main.py 025 smoke --dry-run --lang zh
```

以上测试不启动付费 GPU。
