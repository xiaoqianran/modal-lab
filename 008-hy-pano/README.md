# 008 · HY-Pano 2.0

[Tencent HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · 全景生成。

008 已迁移到 v2：一个 `app.py` 同时拥有 backend/GPU 选择、CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

默认走轻量 **Qwen+LoRA**；full 约 80B、多 GPU，必须显式 `--backend full`。

## 默认资源

```text
qwen -> RTX-PRO-6000
full -> H100:4
```

这个选择现在由 `default_gpu_for()` 单点维护；显式 `--gpu` 始终覆盖默认。

## 用法

```bash
python main.py 008 status

python main.py 008 download --dry-run --backend both
python main.py 008 download --backend qwen

python main.py 008 smoke --dry-run \
  --backend qwen --steps 30

python main.py 008 smoke --backend qwen
python main.py 008 smoke --backend qwen --gpu H100

python main.py 008 infer --dry-run \
  --backend full \
  --image desk.jpg \
  --use-taylor-cache
```

真正的推理参数直接归 `app.py`：

```text
backend
gpu
image
prompt
seed
height
width
steps
load_mode
use_taylor_cache
run_name
```

Qwen `load_mode`：

```text
gpu
cpu_offload
sequential_offload
```

Taylor cache 只在 full backend 里有意义。

## 实测 GPU（desk · 960×1952 · 40 步 · bf16）

| GPU | total | est $ |
|---|---:|---:|
| **RTX-PRO-6000** | **132s** | **$0.111** |
| H100 | 111s | $0.122 |
| A100-80GB | 208s | $0.144 |

## Volume

v2 删除 `ls/pull/list_runs` 包装，直接使用 Modal：

```bash
modal volume ls modal-lab-hy-pano-outputs runs
modal volume get modal-lab-hy-pano-outputs runs/smoke_qwen ./008-hy-pano/outputs
```

## 测试

```bash
python -m unittest discover -s 008-hy-pano/tests -v
python -m py_compile 008-hy-pano/app.py
python main.py 008 status
python main.py 008 smoke --dry-run
```

以上测试不启动付费 GPU。

后续世界生成见 [`009-hy-worldgen`](../009-hy-worldgen/)。
