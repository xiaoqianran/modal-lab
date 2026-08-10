# 021 TRELLIS.2 计划

## Phase 0

- [x] 选型：MIT · 4B · image→GLB
- [x] modal 镜像 + o-voxel/flex_gemm/cumesh/nvdiffrast
- [x] L40S build-sm89 + verify
- [x] L40S smoke @ 512 → **214.5s · $0.12 · 16.9MB GLB**
- [x] PRO 6000 build-sm120 + smoke → **121.9s · $0.10 · 17.1MB GLB**
- [x] 与 020 对照表 + viewer

## 已解坑

| 问题 | 解法 |
|------|------|
| facebook DINOv3 门禁 | camenduru 镜像 + patch pipeline.json |
| briaai RMBG 门禁 | ZhengPeng7/BiRefNet |
| sparse attn 硬依赖 flash_attn | **xformers**（`SPARSE_ATTN_BACKEND=xformers`） |
| sm_120 | torch2.11+cu128 · ARCH=12.0 源码轮 |

## GPU

| GPU | 栈 | 状态 |
|-----|-----|------|
| L40S | sm_89 · torch2.6+cu124 · xformers | ✅ |
| PRO 6000 | sm_120 · torch2.11+cu128 · xformers | ✅ |
