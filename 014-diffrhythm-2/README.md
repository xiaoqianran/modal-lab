# 014 · DiffRhythm 2

ASLP-lab **DiffRhythm 2**（谛韵）— 半自回归扩散全曲生成 · Apache-2.0 · 默认 L4。

014 已迁移到 v2：一个 `app.py` 同时拥有本地歌词输入、CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 用法

```bash
python main.py 014 status
python main.py 014 check
python main.py 014 download --dry-run --force
python main.py 014 download

python main.py 014 smoke --dry-run --steps 12 --cfg-strength 1.8
python main.py 014 smoke

python main.py 014 generate --dry-run \
  --lyrics-file 014-diffrhythm-2/examples/smoke_en.lrc \
  --style 'Pop, Piano, Bass, Drums, Happy' \
  --max-secs 120
```

`generate` 的歌词输入是本地职责，因此由同一个 `app.py` 直接读取：

```text
--lyrics-file PATH
       或
--lyrics '...'
```

这不是额外 runner；Modal `local_entrypoint` 本来就在客户端执行，可以直接处理本地文件。

## 生成控制

```text
style
max_secs
steps
cfg_strength
seed
```

## Volume

v2 不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-diffrhythm-2-outputs runs
modal volume get modal-lab-diffrhythm-2-outputs runs/smoke_en60 ./014-diffrhythm-2/outputs
```

## 测试

```bash
python -m unittest discover -s 014-diffrhythm-2/tests -v
python -m py_compile 014-diffrhythm-2/app.py
python main.py 014 status
python main.py 014 smoke --dry-run
```

以上测试不启动付费 GPU。
