# Upstream & third-party

## Primary model / code

| 项目 | 链接 | 许可（据公开声明） |
|------|------|-------------------|
| TencentARC/Pixal3D | https://github.com/TencentARC/Pixal3D | MIT |
| microsoft/TRELLIS.2 | https://github.com/microsoft/TRELLIS.2 | MIT |
| HF 权重 TencentARC/Pixal3D | https://huggingface.co/TencentARC/Pixal3D | 随模型卡 |

## CUDA extensions (Plan A build sources)

| 包 | 源 |
|----|-----|
| FlexGEMM | https://github.com/JeffreyXiang/FlexGEMM |
| CuMesh | https://github.com/JeffreyXiang/CuMesh |
| o-voxel | microsoft/TRELLIS.2 `o-voxel/` |
| nvdiffrast | https://github.com/NVlabs/nvdiffrast (v0.4.0) |
| nvdiffrec renderutils | https://github.com/JeffreyXiang/nvdiffrec (branch `renderutils`) |
| NATTEN | https://github.com/SHI-Labs/NATTEN |

## Community Ada (Plan B only)

| 项目 | 链接 | 用途 |
|------|------|------|
| carroyoaesa … sm89-wheels | https://github.com/carroyoaesa/comfyui-trellis2-pixal3d-rtx40-ada-sm89-wheels-debian13 | 预编译 sm_89 Linux wheels + 文档 |
| visualbruno forks | FlexGEMM / CuMesh / TRELLIS.2 forks | 社区 wheel 的源码树（API 可能与官方分叉） |

## Related lab experiment

- [`005-pixal3d`](../005-pixal3d/) — HF demo stack, H100 default (unchanged by v2)
