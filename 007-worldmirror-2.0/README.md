# 007 · WorldMirror 2.0

在 Modal 上以最低成本运行 Tencent HY-World 2.0 的 **World Reconstruction / WorldMirror 2.0** 路径。

007 已迁移到 v2：一个 `app.py` 同时拥有 smoke 不变量、CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 为什么只跑 WorldMirror 2.0

```text
WorldMirror 2.0   ~1.2B   默认
HY-Pano 2.0       ~80B    不跑
WorldStereo 2.0   ~17B    不跑
Full WorldGen              不跑
```

这是明确的成本边界，不在实验入口里偷偷升级到完整 world generation。

## 用法

```bash
python main.py 007 status
python main.py 007 check
python main.py 007 download --dry-run --force
python main.py 007 download

# 固定最低成本 smoke
python main.py 007 smoke --dry-run
python main.py 007 smoke --gpu L4

# 自定义 recon
python main.py 007 infer --dry-run \
  --example Desk \
  --max-images 4 \
  --target-size 640 \
  --gpu L40S \
  --no-bf16 \
  --save-gs
```

`smoke` 不是一套可任意覆盖的配置，而是 benchmark 不变量：

```text
example=Desk
max_images<=2
target_size=518
bf16=true
save_gs=false
run_name=smoke_desk
```

自由参数只属于 `infer`。

## GPU 实测（Desk · 2 图 · 518 · bf16）

| GPU | total | est $ | peak VRAM |
|---|---:|---:|---:|
| **T4** | **27.18s** | **$0.0045** | 4.89 GB |
| L4 | 25.96s | $0.0058 | 5.00 GB |
| L40S | 17.91s | $0.0097 | 5.27 GB |
| RTX-PRO-6000 | 18.90s | $0.0159 | 5.45 GB |

默认 T4 仍是冷启动总账最低。

## Volume

v2 删除 `ls/pull` 包装：

```bash
modal volume ls modal-lab-hy-world-2-outputs runs
modal volume get modal-lab-hy-world-2-outputs runs/bench_T4_desk2 ./007-worldmirror-2.0/outputs
```

## 测试

```bash
python -m unittest discover -s 007-worldmirror-2.0/tests -v
python -m py_compile 007-worldmirror-2.0/app.py
python main.py 007 status
python main.py 007 smoke --dry-run
```

以上测试不启动付费 GPU。

与 006 的版本/产品线区别见 [`006-hunyuanworld-mirror`](../006-hunyuanworld-mirror/)。
