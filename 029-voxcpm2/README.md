# 029 · VoxCPM2（TTS Tier A1）

[OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) · Apache-2.0 · 默认 L4。

029 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 用法

```bash
python main.py 029 status                 # 纯本地
python main.py 029 check                  # 远程 Volume / 权重状态
python main.py 029 download --dry-run
python main.py 029 download

python main.py 029 smoke --dry-run --kind en
python main.py 029 smoke --kind zh
python main.py 029 smoke --kind design
python main.py 029 smoke --kind clone

python main.py 029 t2s --dry-run \
  --text 'Hello from VoxCPM2.' \
  --cfg-value 1.8 --timesteps 8 --seed 7 --optimize
```

`clone` 默认使用 `download_weights()` 放入 prompts Volume 的 `reference_speaker.wav`；显式 `--reference-wav` 会覆盖默认值。

真正的生成参数直接由 `app.py` 管理：

```text
reference_wav
prompt_wav
prompt_text
cfg_value
inference_timesteps
seed
optimize
```

## Volume

v2 不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-voxcpm2-outputs runs
modal volume get modal-lab-voxcpm2-outputs runs/smoke_en ./029-voxcpm2/outputs
```

## Smoke 基线

| kind | 说明 | 估费 |
|---|---|---:|
| en / zh | tokenizer-free TTS | ~$0.007 |
| design | 文本级 voice description | ~$0.008 |
| clone | reference_wav 可控克隆 | ~$0.015 |

## 测试

```bash
python -m unittest discover -s 029-voxcpm2/tests -v
python -m py_compile 029-voxcpm2/app.py
python main.py 029 status
python main.py 029 smoke --dry-run --kind clone
```

以上测试不启动付费 GPU。
