# 022 · Hunyuan3D-2.1 on L40S

腾讯 `tencent/Hunyuan3D-2.1` 的精简 Modal 实验。目录只维护 **L40S / Ada sm_89** 路径，不维护第二套 GPU image。

022 已迁移到 v2：**单入口 `app.py`**，不再使用 `run.py -> modal_app.py` 二次包装。

## 固定栈

- GPU: `L40S`
- CUDA: `12.4.1`
- PyTorch: `2.5.1`
- CUDA arch: `8.9`
- Upstream: `Tencent-Hunyuan/Hunyuan3D-2.1`
- Upstream commit: `82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- Weights: `tencent/Hunyuan3D-2.1`
- License: Tencent Hunyuan 3D 2.1 Community License

`custom_rasterizer` 和 `mesh_inpaint_processor` 都在 Modal image build 阶段编译。Worker 运行时不再 `pip install`、不再现场编 CUDA 扩展。

## 结构

```text
022-hunyuan3d-2.1/
├── app.py              # 唯一 CLI + Modal app + remote worker
├── tests/test_app.py   # 本地 CLI / planning 测试
├── examples/
├── gallery/
├── viewer/
└── README.md
```

远程推理主体仍保持高内聚：

```text
app.py
 ├── Modal Image
 ├── Hunyuan3D21.load()
 ├── Hunyuan3D21.probe()
 ├── Hunyuan3D21.generate()
 └── local entrypoint
```

## 用法

推荐从仓库根目录进入：

```bash
# 纯本地状态，不启动 GPU
python main.py 022 status

# 远程 probe
python main.py 022 probe

# 无成本检查最终 smoke 参数
python main.py 022 smoke --dry-run --mode shape

# 真正付费 smoke
python main.py 022 smoke --i-know-this-costs-money --mode shape
python main.py 022 smoke --i-know-this-costs-money --mode full
```

也可以直接：

```bash
cd 022-hunyuan3d-2.1
modal run app.py probe
modal run app.py smoke --i-know-this-costs-money --mode full
```

默认 smoke 使用 `chair.png`、`seed=42`、6 views、paint resolution 512。

输出进入：

```text
modal-lab-hunyuan3d21-outputs
├── meshes/<name>.glb
├── meshes/<name>_meta.json
├── benchmarks/<name>.json
└── inputs/<name>.png
```

下载结果直接使用 Modal Volume CLI，不再额外包装：

```bash
modal volume get modal-lab-hunyuan3d21-outputs meshes/smoke_l40s.glb ./viewer/
```

## 已有 L40S 基线

当前 L40S smoke（`chair.png`, seed 42）：shape `29.95s`（request `30.10s`）、峰值 `7.63 GiB`；full 为 shape `31.77s` + paint load `19.71s` + paint `37.66s`，request 总计 `89.43s`、峰值 `20.22 GiB`，GLB `1.28 MB`。纹理前处理直接用 `trimesh` + `fast-simplification` 收敛到 40k faces，不依赖 Open3D/PyVista。

`seconds_total` 和成本估算是 request scope：包含输入预处理和 lazy paint load，不包含 container 启动与 `@modal.enter()` 的 shape 模型加载。

## 测试

```bash
python -m unittest discover -s 022-hunyuan3d-2.1/tests -v
python -m py_compile 022-hunyuan3d-2.1/app.py
python main.py 022 status
python main.py 022 smoke --dry-run
```

这些验证不会启动付费 GPU。
