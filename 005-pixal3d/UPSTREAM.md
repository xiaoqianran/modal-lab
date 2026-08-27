# Upstream pins — 005-pixal3d

| 组件 | 来源 | 版本 / 备注 |
|------|------|-------------|
| 推理代码 | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | **`master`**（Trellis.2 backbone 改进版；仓库默认分支） |
| 主权重 | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) | ~24GB · `pipeline.json` + `ckpts/*` |
| 论文版分支 | `paper`（Direct3D-S2） | **本实验不用**；要对齐论文数字再切 |
| Backbone 参考 | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | 安装依赖对齐其 CUDA 扩展生态 |
| 相机估计 | [microsoft/MoGe](https://github.com/microsoft/MoGe) · `Ruicheng/moge-2-vitl` | 无 `--fov` 时自动估计 |
| 图像条件 | `camenduru/dinov3-vitl16-pretrain-lvd1689m` + [valeoai/NAF](https://github.com/valeoai/NAF) | DinoV3ProjFeatureExtractor |
| 去背景 | `briaai/RMBG-2.0`（pipeline.json `rembg_model`） | BiRefNet 封装 |
| 预编译轮子 | LDYang694 / JeffreyXiang Storages releases | 见 `app.py` 顶部 URL |
| 论文 | [arXiv:2605.10922](https://arxiv.org/abs/2605.10922) | SIGGRAPH 2026 |
| 项目页 | https://ldyang694.github.io/projects/pixal3d/ | |
| 在线 Demo | https://huggingface.co/spaces/TencentARC/Pixal3D | |

## 镜像内关键路径

```text
/opt/Pixal3D/                 # git clone 的上游
/weights/Pixal3D/             # snapshot_download 主权重
/weights/hf/                  # HF_HOME
/weights/torch/               # TORCH_HOME（NAF 等）
/outputs/meshes/*.glb         # 成片
```

## 默认推理参数（对齐官方 `inference.py`）

| 参数 | 默认 |
|------|------|
| low_vram | True |
| resolution | 1024（low_vram）/ 1536（full） |
| seed | 42 |
| FOV | MoGe 自动（可用 `--fov 0.2` 手动） |
| cascade | `{resolution}_cascade` |
| GLB | remesh + texture 4096 + webp 扩展 |

## 主权重文件清单

```text
pipeline.json
ckpts/ss_dec_conv3d_16l8_fp16.safetensors
ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
ckpts/shape_dec_next_dc_f16c32_fp16.safetensors
ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors
ckpts/tex_dec_next_dc_f16c32_fp16.safetensors
ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors
```
