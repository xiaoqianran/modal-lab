# 021-trellis2 — TRELLIS.2-4B（质量 / MIT）

[microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) · 默认 L40S · 可选 RTX-PRO-6000。

021 已迁移到 v2：保留一个高内聚的 `app.py`，不再使用 79 行 `run.py -> modal_app.py` 翻译层。主体虽然较大，但 wheel build / verify / inference 共享同一扩展栈与资源知识，目前不做无意义拆分。

## 用法

```bash
python main.py 021 status

python main.py 021 probe --dry-run --gpu L40S
python main.py 021 probe --gpu L40S
python main.py 021 build --dry-run --gpu RTX-PRO-6000
python main.py 021 build --gpu RTX-PRO-6000
python main.py 021 verify --gpu L40S

# --force 现在真正下沉到 worker.download(force=...)
python main.py 021 download --dry-run --force
python main.py 021 download --force

# 无成本查看最终 3D 计划
python main.py 021 smoke --dry-run \
  --gpu RTX-PRO-6000 \
  --pipeline-type 1024_cascade \
  --texture-size 1024 \
  --decimation-target 200000

# 真正生成
python main.py 021 smoke --i-know-this-costs-money --gpu L40S

# 自定义图片语义独立于 smoke
python main.py 021 i2v --dry-run \
  --image-url https://example.com/chair.png \
  --pipeline-type 1024
```

## 本次迁移修复的 CLI 断层

旧 wrapper 的 `download --force` 会解析参数，但不会传给 `modal_app.py`；v2 已修复。

远程 `image_to_3d()` 原本支持但 wrapper 未暴露的参数也已进入唯一 CLI：

```text
texture_size
decimation_target
```

## 技术栈

| GPU | torch | CUDA | 扩展 ARCH | attn |
|---|---|---|---|---|
| L40S | 2.6.0+cu124 | 12.4 | 8.9 | xformers |
| PRO 6000 | 2.11.0+cu128 | 12.8 | 12.0 | xformers |

生命周期仍然清晰地保留在同一个实验文件：

```text
probe -> build -> verify -> download -> smoke/i2v
```

## 测试

```bash
python -m unittest discover -s 021-trellis2/tests -v
python -m py_compile 021-trellis2/app.py
python main.py 021 status
python main.py 021 smoke --dry-run
```

以上测试不启动付费 GPU。
