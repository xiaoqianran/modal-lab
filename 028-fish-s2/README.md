# 028 · Fish Audio S2 Pro（TTS Tier S4）

[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) · Research License · 默认 **L40S**。

028 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 用法

```bash
python main.py 028 status                 # 纯本地
python main.py 028 check                  # 远程权重 / prompts / outputs
python main.py 028 download --dry-run
python main.py 028 download

python main.py 028 smoke --dry-run --kind en
python main.py 028 smoke --kind zh
python main.py 028 smoke --kind tags
python main.py 028 smoke --kind clone

python main.py 028 t2s --dry-run \
  --text 'Hello [laughing], how are you?' \
  --temperature 0.7 --top-p 0.9 --chunk-length 160 --seed 7
```

`clone` 默认使用公开参考音频和对应 transcript；显式 `--ref-audio/--ref-text` 会覆盖。

真正的生成参数直接由 `app.py` 管理：

```text
ref_audio
ref_text
voice
temperature
top_p
repetition_penalty
max_new_tokens
chunk_length
seed
compile
```

旧 wrapper 没有暴露 `chunk_length`；v2 已直接纳入唯一 CLI。

## Smoke 场景

```text
en     -> 英文随机音色
zh     -> 中文随机音色
tags   -> 固定 [excited] / [chuckle] / [whisper] benchmark
clone  -> 参考音克隆
```

## Volume

v2 不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-fish-s2-outputs runs
modal volume get modal-lab-fish-s2-outputs runs/smoke_en ./028-fish-s2/outputs
```

## 测试

```bash
python -m unittest discover -s 028-fish-s2/tests -v
python -m py_compile 028-fish-s2/app.py
python main.py 028 status
python main.py 028 smoke --dry-run --kind clone
```

以上测试不启动付费 GPU。

许可：Fish Audio Research License，研究/非商用；商用需单独授权。
