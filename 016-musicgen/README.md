# 016 · MusicGen

Meta MusicGen text→instrumental 基线 · 默认 T4 · CC-BY-NC 4.0。

016 已迁移到 v2：一个 `app.py` 同时拥有模型归一化、CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 用法

```bash
python main.py 016 status
python main.py 016 check
python main.py 016 download --dry-run --model medium
python main.py 016 download --model small

python main.py 016 smoke --dry-run \
  --duration 10 --guidance-scale 2.5 --temperature 0.9

python main.py 016 t2a --dry-run \
  --model medium \
  --prompt 'jazz piano trio, swinging' \
  --duration 20 --seed 7
```

模型 canonical key：

```text
small
medium
large
melody
```

真正的生成参数直接由唯一 CLI 管理：

```text
prompt
duration
seed
guidance_scale
temperature
```

## Volume

v2 不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-musicgen-outputs runs
modal volume get modal-lab-musicgen-outputs runs/smoke_lofi ./016-musicgen/outputs
```

## 测试

```bash
python -m unittest discover -s 016-musicgen/tests -v
python -m py_compile 016-musicgen/app.py
python main.py 016 status
python main.py 016 smoke --dry-run
```

以上测试不启动付费 GPU。
