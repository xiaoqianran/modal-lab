# 031 · CosyVoice3（TTS Tier A3）

Fun-CosyVoice3-0.5B · Apache-2.0 · 中文方言 SOTA · 默认 **L4**。

031 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_zh | zero-shot 中文 | 42.3s | **$0.0094** | 9.4s | 3.6G |
| smoke_dialect | 四川话 instruct | 21.9s | **$0.0049** | 8.8s | 3.6G |
| smoke_en | cross-lingual EN | 31.2s | **$0.0069** | 7.2s | 3.6G |

## 用法

```bash
# 本地固定信息，不触云
python main.py 031 status

# 真正读取远程权重 / outputs / prompts Volume 状态
python main.py 031 check

# 下载权重和默认 prompt wav
python main.py 031 download
python main.py 031 download --force

# 无成本 smoke 规划
python main.py 031 smoke --dry-run --kind zh
python main.py 031 smoke --dry-run --kind dialect
python main.py 031 smoke --dry-run --kind en

# 真正 smoke
python main.py 031 smoke --kind zh
python main.py 031 smoke --kind dialect
python main.py 031 smoke --kind en

# 自定义 TTS
python main.py 031 t2s --dry-run --text '你好，这是 CosyVoice3。'
python main.py 031 t2s --text '你好，这是 CosyVoice3。'

# instruct 模式
python main.py 031 t2s --text '今天真开心。' \
  --mode instruct --instruct '请用四川话开心地说。'
```

也可以直接使用 Modal：

```bash
cd 031-cosyvoice3
modal run app.py check
modal run app.py download
modal run app.py smoke --kind dialect
```

## Volume 操作

v2 不再包装 Modal 自带的 Volume CLI：

```bash
modal volume ls modal-lab-cosyvoice3-outputs runs
modal volume get modal-lab-cosyvoice3-outputs runs/smoke_zh ./031-cosyvoice3/outputs
```

## CLI 边界

`smoke` 只拥有四种稳定场景：

```text
zh       -> zero_shot
 tongue  -> zero_shot 绕口令
dialect  -> instruct 四川话
en       -> cross-lingual English
```

更自由的 `mode / instruct / prompt-text / prompt-wav` 属于 `t2s`，不再通过额外 wrapper 转译。

## 测试

```bash
python -m unittest discover -s 031-cosyvoice3/tests -v
python -m py_compile 031-cosyvoice3/app.py
python main.py 031 status
python main.py 031 smoke --dry-run --kind dialect
```

以上测试不启动付费 GPU。

试听：[`gallery/`](gallery/) · 成本：[`COST_BENCHMARK.md`](COST_BENCHMARK.md) · 上游：[`UPSTREAM.md`](UPSTREAM.md)
