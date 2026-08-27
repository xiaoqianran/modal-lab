# 026 · Chatterbox（TTS Tier S2）

[Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) · MIT · 默认 L4。

026 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 用法

```bash
python main.py 026 status
python main.py 026 check
python main.py 026 download --dry-run --model turbo
python main.py 026 download --model multilingual

python main.py 026 smoke --dry-run --kind mtl_en
python main.py 026 smoke --kind mtl_zh
python main.py 026 smoke --kind turbo --voice Lucy

python main.py 026 t2s --dry-run \
  --model original --text 'Hello.' --exaggeration 0.7 --cfg-weight 0.4
```

模型 canonical key 只有三种：

```text
multilingual
turbo
original
```

alias 统一由 `_norm_model()` 归一化，不在 wrapper 再复制 choices。

`nano` 只对 Turbo 生效；multilingual/original 会忽略它，避免产生假的跨模型配置。

## Prompt Volume

旧 `upload-prompts` 只是 `modal volume put` 的循环包装，v2 已删除。上传本地 voice 直接使用 Modal：

```bash
for f in 026-chatterbox/inputs/voices/*.wav; do
  modal volume put --force modal-lab-chatterbox-prompts "$f" "$(basename "$f")"
done
```

推理命令不再隐式修改 prompts Volume。

## Smoke 场景

```text
mtl_en -> multilingual 英文，无默认 voice prompt
mtl_zh -> multilingual 中文，无默认 voice prompt
turbo  -> Turbo 英文，默认 Lucy prompt
```

## Volume 输出

```bash
modal volume ls modal-lab-chatterbox-outputs runs
modal volume get modal-lab-chatterbox-outputs runs/smoke_mtl_en ./026-chatterbox/outputs
```

## 测试

```bash
python -m unittest discover -s 026-chatterbox/tests -v
python -m py_compile 026-chatterbox/app.py
python main.py 026 status
python main.py 026 smoke --dry-run --kind turbo --nano
```

以上测试不启动付费 GPU。
