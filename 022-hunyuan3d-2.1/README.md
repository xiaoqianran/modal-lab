# 022 · Hunyuan3D-2.1 on L40S

腾讯 `tencent/Hunyuan3D-2.1` 的精简 Modal 实验。这个目录只维护 **L40S / Ada sm_89** 路径，不再维护第二套 GPU image。

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

## 用法

```bash
python run.py status
python run.py probe
python run.py smoke --i-know-this-costs-money --mode shape
python run.py smoke --i-know-this-costs-money --mode full
```

默认 smoke 使用 `chair.png`、`seed=42`、6 views、paint resolution 512。输出进入：

```text
modal-lab-hunyuan3d21-outputs
├── meshes/<name>.glb
├── meshes/<name>_meta.json
├── benchmarks/<name>.json
└── inputs/<name>.png
```

下载最近的 L40S smoke：

```bash
modal volume get modal-lab-hunyuan3d21-outputs meshes/smoke_l40s.glb ./viewer/
```

## 已有 L40S 基线

当前 L40S smoke（`chair.png`, seed 42）：shape `29.72s`、峰值 `7.63 GiB`；full 为 shape `30.23s` + paint `46.73s`，总计 `110.84s`、峰值 `20.22 GiB`，GLB `1.32 MB`。纹理前处理用 `fast-simplification` 将 mesh 收敛到 40k faces，不依赖 Open3D。旧的 PRO 6000 对照不再属于本目录的运行路径。
