# 006 · HunyuanWorld-Mirror

Tencent HunyuanWorld-Mirror · 多视图 3D 几何重建 · 默认 L4。

006 已迁移到 v2：一个 `app.py` 同时拥有 smoke 不变量、CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 成本策略

```text
默认 GPU   L4
下载       CPU-only
smoke      Bright_Room · 2 图 · 518
关闭       GS video / COLMAP / sky mask
```

## 用法

```bash
python main.py 006 status
python main.py 006 check
python main.py 006 download --dry-run --force
python main.py 006 download

python main.py 006 smoke --dry-run
python main.py 006 smoke --gpu T4

python main.py 006 infer --dry-run \
  --example Ireland_Landscape \
  --max-images 4 \
  --target-size 640 \
  --gpu L40S \
  --save-gs
```

`smoke` 是固定 benchmark：

```text
example=Bright_Room
max_images<=2
target_size=518
save_gs=false
run_name=smoke_bright_room
```

自由参数只属于 `infer`。

## Volume

v2 删除 `ls/pull`：

```bash
modal volume ls modal-lab-hunyuanworld-mirror-outputs runs
modal volume get modal-lab-hunyuanworld-mirror-outputs runs/smoke_bright_room ./006-hunyuanworld-mirror/outputs
```

## Smoke 基线

L4 实测：peak VRAM **7.66 GB** · forward **~3s** · 估 **~$0.01** · 点云约 274k。

## 测试

```bash
python -m unittest discover -s 006-hunyuanworld-mirror/tests -v
python -m py_compile 006-hunyuanworld-mirror/app.py
python main.py 006 status
python main.py 006 smoke --dry-run
```

以上测试不启动付费 GPU。

与 007 WorldMirror 2.0 的产品线差异见 [`007-worldmirror-2.0`](../007-worldmirror-2.0/)。
