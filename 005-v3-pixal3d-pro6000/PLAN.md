# 005-v3 计划（第二轮更新）

## ✅ 已完成

- [x] 根因：sm_120 vs torch2.6/cu124  
- [x] Windows 社区轮子旁证  
- [x] **Linux 配方 α**：animede Pixal3D@PRO6000（torch2.11+cu128, arch=12.0, drtk, natten）  
- [x] **Linux 配方 β**：TRELLIS.2#143 · 5090 · e2e GLB  
- [x] 合并为 Plan A* + B0–B4 门禁  

## 🔒 未做（等你下令）

- [ ] B0：Modal `RTX-PRO-6000` + cu128 认卡（几分钟）  
- [ ] B1：编 o-voxel  
- [ ] B2：drtk / nvdiffrast  
- [ ] B3：natten smoke  
- [ ] B4：smoke GLB  

## 实现时注意

1. 镜像 `cuda:12.8*-devel`，`g++-13`  
2. `CUDA_HOME` 与 torch cu128 一致  
3. 每步 Volume commit 轮子（抄 v2）  
4. 默认 `ATTN_BACKEND=sdpa`  
5. 与 L40S 比价后再决定是否默认  

## 决策

- 出片：**v2 L40S**  
- 探索 PRO 6000：先 B0，再往下  
