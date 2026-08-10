# 005-v3 计划

## Phase 0 — 研究（✅）

- [x] 确认 PRO 6000 = Blackwell **sm_120**  
- [x] 确认 torch2.6/cu124 不可用  
- [x] 收集 PyTorch cu128、NATTEN、Windows 社区轮子信息  
- [x] 写入 SOLUTION.md  

## Phase 1 — 仓库骨架（✅ 本提交）

- [x] `005-v3-pixal3d-pro6000/` 文档  
- [ ] **不**默认 `modal run`  

## Phase 2 — 探针（等你下令再烧钱）

1. 最小镜像：torch 2.7+/cu128 on `RTX-PRO-6000`  
2. 打印 capability + arch_list（B0–B2）  
3. 单扩展 `nvdiffrast` 或 `flex_gemm` wheel  

## Phase 3 — 全扩展 + verify

同 v2 的 build/verify，但 arch=`12.0`

## Phase 4 — smoke GLB

仅当 Phase 3 全绿

## 决策点

- 是否值得相对 L40S 继续：看 B0–B2 与单价  
- 若 L40S 更便宜且够用：**暂停 v3 实现**
