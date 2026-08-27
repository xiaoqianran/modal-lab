# 034 · Higgs Audio v2（TTS Tier A6 · 收官）

`bosonai/higgs-audio-v2-generation-3B-base` · 默认 **L40S** · 场景描述 / 表现力。

034 已迁移到 v2：一个 `app.py` 同时拥有实验 CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

| run | 场景 | wall | 估费 | 时长 | VRAM |
|-----|------|------|------|------|------|
| smoke_en | quiet room | 49.7s | **$0.0269** | 8.3s | 15.7G |
| smoke_expressive | excited | 21.1s | **$0.0114** | 9.9s | 15.7G |

## 用法

```bash
# 本地固定信息，不触云
python main.py 034 status

# 真正读取远程权重 / outputs Volume 状态
python main.py 034 check

# 下载 pinned 权重
python main.py 034 download
python main.py 034 download --force

# 无成本 smoke 规划
python main.py 034 smoke --dry-run --kind expressive

# 真正 smoke
python main.py 034 smoke --kind en
python main.py 034 smoke --kind expressive

# TTS dry-run / 真正生成
python main.py 034 t2s --dry-run --text 'Hello from Higgs.'
python main.py 034 t2s --text 'Hello from Higgs.'
```

也可以直接使用 Modal：

```bash
cd 034-higgs
modal run app.py check
modal run app.py download
modal run app.py smoke --kind expressive
modal run app.py t2s --text 'Hello from Higgs.'
```

## Volume 操作

v2 不再包装 Modal 自带的 `volume ls/get`。

```bash
# 列出 runs
modal volume ls modal-lab-higgs-outputs runs

# 拉取某次 run
modal volume get --force modal-lab-higgs-outputs runs/smoke_en ./034-higgs/outputs
```

这样 Volume 文件操作只有 Modal CLI 一个事实源，不在实验代码里重复维护。

## 权重 pin

兼容 github loader：

```text
model      10840182ca4a
tokenizer  9d4988fbd4ad
```

## 测试

```bash
python -m unittest discover -s 034-higgs/tests -v
python -m py_compile 034-higgs/app.py
python main.py 034 status
python main.py 034 smoke --dry-run --kind expressive
python main.py 034 t2s --dry-run --text 'Hello'
```

以上测试不启动付费 GPU。

**本号为 TTS 线终点。不做 035。**
