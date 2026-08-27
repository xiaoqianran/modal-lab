# 030 · VibeVoice（TTS Tier A2）

[Microsoft VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) · **MIT**  
代码：[microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)

030 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

| 项 | 值 |
|----|-----|
| 默认模型 | **Realtime 0.5B** 流式 TTS |
| 默认 GPU | **L4** |
| 默认 speaker | **Carter** |
| 许可 | **MIT** |
| 排名 | GH ~52k stars · Realtime HF ~594k |

> 官方长文多说话人 TTS 推理码已撤；本槽用 Realtime 变体。

## 用法

```bash
# 纯本地固定信息
python main.py 030 status

# 远程检查权重 / voices / outputs
python main.py 030 check

# 下载模型与官方 voice presets
python main.py 030 download --dry-run
python main.py 030 download

# 固定 smoke
python main.py 030 smoke --dry-run --kind en
python main.py 030 smoke --kind en
python main.py 030 smoke --kind long
python main.py 030 smoke --kind emma

# 自定义 TTS
python main.py 030 t2s --dry-run \
  --text 'Hello from VibeVoice.' --speaker Grace --cfg-scale 1.7 --ddpm-steps 7
```

`ddpm_steps` 是底层 `generate_fn()` 的真实参数，旧 wrapper 没有暴露；v2 直接由同一个 `app.py` CLI 管理。

Smoke 场景的 speaker 语义固定：

```text
en    -> 默认 Carter，可显式覆盖
long  -> 默认 Emma，可显式覆盖
emma  -> 强制 Emma
```

## Volume

v2 不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-vibevoice-outputs runs
modal volume get modal-lab-vibevoice-outputs runs/smoke_en ./030-vibevoice/outputs
```

## 测试

```bash
python -m unittest discover -s 030-vibevoice/tests -v
python -m py_compile 030-vibevoice/app.py
python main.py 030 status
python main.py 030 smoke --dry-run --kind long
```

以上测试不启动付费 GPU。
