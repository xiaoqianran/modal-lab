# 005-v3 · RTX PRO 6000（Blackwell sm_120）方案讨论

> **状态：方案研究完成 · 尚未在 Modal 上实跑**  
> 目标：评估 PRO 6000 是否能跑官方 Pixal3D 权重出 GLB，以及如何做。

---

## 0. 结论（先看这个）

| 问题 | 答案 |
|------|------|
| 有没有方案？ | **有，但是「整栈换代」**，不是改 `gpu=`。 |
| 005 / 005-v2 栈能直接用吗？ | **不能。** torch 2.6 + cu124 **最高到 sm_90**；我们实测 PRO 6000 会报无 kernel / 不兼容。 |
| 和 L40S (v2) 比难度？ | **明显更难。** L40S 仍可用 torch2.6 只重编扩展；PRO 6000 必须先换 **torch≥2.7 + CUDA≥12.8**，再按 **sm_120** 重编全部扩展。 |
| 有没有现成 Linux Modal 轮子？ | **几乎没有可直接抄的 Linux 预编译全家桶。** 社区有 **Windows** 向 PRO 6000 Trellis2 轮子。 |
| 现在推不推荐立刻烧钱？ | **谨慎。** 先做 `torch+sm_120` 冒烟 + 单扩展编译探针；**不**一上来全量 natten+推理。 |
| 出片首选？ | 仍推 **L40S (v2 已通 ~$0.17)** 或 **H100 (005)**。PRO 6000 适合「同账号卡型统一 / 以后想降本」的探索线。 |

---

## 1. 硬件事实

| | RTX PRO 6000 Blackwell | L40S (v2) | H100 (005) |
|--|------------------------|-----------|------------|
| 架构 | **Blackwell** | Ada | Hopper |
| Compute | **sm_120 / 12.0**（常见 Workstation） | sm_89 | sm_90 |
| 显存 | ~96 GB 级 | 48 GB | 80 GB |
| Modal 代号 | `RTX-PRO-6000` | `L40S` | `H100` |

注意：数据中心 **B100/B200 常是 sm_100**，与消费/工作站 **sm_120** 不互通；编轮子时 **不要** 假设 sm_100 能在 PRO 6000 上跑。

---

## 2. 为什么 005 官版 / v2 栈必挂

| 层 | 005 / v2 现状 | PRO 6000 需要 |
|----|---------------|---------------|
| PyTorch | 2.6.0+**cu124** | **≥2.7 且 cu128+**（官方二进制含 sm_120） |
| CUDA toolkit（编译） | 12.4 | **≥12.8**（Blackwell 内核从 12.8 起） |
| 扩展 cubin | sm_89 / sm_90 | **sm_120** |
| HF demo 轮子 | H 系列 | 完全不可用 |

症状回顾（005 实测）：torch 警告 *not compatible* + CUDA `no kernel image`。

**根因公式与 L40S 相同（架构不匹配），但修复面更大：先换框架，再换扩展。**

---

## 3. 社区已有路径

### 3.1 Windows Comfy 预编译（存在，但不等于 Modal）

[PixWizardry/AI_Trellis2-WHLs-RTX-PRO-6000](https://huggingface.co/PixWizardry/AI_Trellis2-WHLs-RTX-PRO-6000)

| 项 | 内容 |
|----|------|
| 目标卡 | RTX PRO 6000 · **sm120** |
| OS | **Windows 11 only** |
| Python | 3.12 |
| torch | **2.10** |
| CUDA | **13.0** |
| 含轮子 | flash_attn · nvdiffrast · flex_gemm · cumesh · o_voxel · nvdiffrec |
| 用途 | ComfyUI-Trellis2 |
| 对 Modal | **不能直接 pip**（win_amd64）；证明 **扩展可在 sm_120 上编出** |

### 3.2 PyTorch 官方

- **2.7.0+cu128** 起稳定二进制支持 Blackwell / sm_120  
- 例：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`  
- Docker 例：`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`

### 3.3 NATTEN（Pixal3D 关键）

- 新版本倾向 **PyTorch ≥ 2.8**  
- **Blackwell 内核需要 CUDA Toolkit ≥ 12.8**  
- 预编译 wheel 若仅 cu126/cu127 会警告 *Blackwell FNA/FMHA kernels not available*  
- 源码编：`NATTEN_CUDA_ARCH` / `TORCH_CUDA_ARCH_LIST` 指向 **12.0**（PRO 6000）；文档有时写 10.0/10.3 针对 **sm_100 系**，**PRO 6000 应优先 12.0**，并以 `cuobjdump` 验收

### 3.4 其它

- FlashAttention 等对 sm_120 支持仍在追赶；Pixal3D 可用 **`ATTN_BACKEND=sdpa`** 绕开  
- 多篇 vLLM / 5090 经验：必须 **源码编** 依赖 + 新 torch，预编译镜像经常缺 sm_120

---

## 4. Modal 上可行的 Plan（推荐排序）

### Plan A — 新栈源码编 sm_120（**主推研究路径，对齐 v2 方法论**）

| 组件 | 建议钉死 |
|------|----------|
| GPU | `RTX-PRO-6000` |
| 镜像 | `nvidia/cuda:12.8.x-devel` 或 12.9 devel |
| Python | 3.11 或 3.12（与新 torch 生态更顺） |
| torch | **2.7+cu128** 或 **2.8/2.9+cu128**（以 `torch.cuda.get_arch_list()` 含 `sm_120` 为准） |
| TORCH_CUDA_ARCH_LIST | **`12.0`** |
| ATTN | **sdpa**（先别上 flash_attn） |
| 扩展 | 同 v2 列表，**全部**对 12.0 源码 `pip wheel` |
| natten | 较新 tag（≥0.21.5 系）+ CUDA 12.8+ 编 |
| 权重 | 仍用 `TencentARC/Pixal3D`（权重与架构无关） |
| 门禁 | `capability==(12,0)` + 每扩展 `.so` 含 sm_120 + natten smoke |

**分阶段烧钱（强烈建议）：**

1. **P0** 仅启动容器：打印 `torch.__version__`、`get_arch_list()`、`get_device_capability()`  
2. **P1** 只编 `nvdiffrast` 或 `flex_gemm` 一个扩展 + import  
3. **P2** 编完 6 扩展 + verify  
4. **P3** download 权重 + smoke GLB  

任一步失败就停，避免一次挂 2 小时 natten。

**风险：**

- Pixal3D / TRELLIS.2 代码钉在 torch2.6 API 上可能有小 diff（通常可修）  
- natten 编译时间与 ABI 坑  
- Modal 镜像体积与 cold start 变大  
- 单价若高于 L40S，**降本意义变弱**（PRO 6000 单价通常不低）

### Plan B — 借鉴 Windows 社区轮子的「版本目标」

把目标栈对齐 PixWizardry：**torch 2.10 + CUDA 13 + py3.12**，在 **Linux** 上 **自己编** 同版本扩展（不能装 win wheel）。

- 优点：与已验证 Comfy 生态版本接近  
- 缺点：比 Plan A 更新、依赖更难找 Linux wheel，集成进官方 `inference.py` 工作量更大  

### Plan C — ComfyUI 整条链（**不推荐作 lab 主路径**）

同 005-v2 讨论：适合本机创作，不适合本仓库 Modal 脚本实验。Windows 轮子可参考，但 lab 默认仍走 Python `inference.py`。

### Plan D — 放弃 PRO 6000 出片

若目标是 **稳定出 GLB / 控成本**：继续 **v2 L40S** 或 **005 H100**。PRO 6000 仅作探索。

---

## 5. 与 005 / 005-v2 对照

| | 005 | 005-v2 | **005-v3（本目录）** |
|--|-----|--------|---------------------|
| 默认 GPU | H100 | L40S | **RTX-PRO-6000** |
| SM | 90 | 89 | **120** |
| torch | 2.6 cu124 | 2.6 cu124 | **须 ≥2.7 cu128+** |
| 扩展策略 | HF demo 轮子 | 源码 sm_89 | **源码 sm_120（新栈）** |
| 实跑 | ✅ | ✅ | 🔒 未跑 |
| 推荐生产 | 稳 | **性价比已证** | 探索 |

---

## 6. 验收标准（若开跑）

| # | 检查 | 通过条件 |
|---|------|----------|
| B0 | torch 认卡 | `is_available` + name 含 PRO 6000 |
| B1 | 架构 | `get_device_capability() == (12, 0)`（或 Modal 报告的实际 minor） |
| B2 | torch 二进制 | `sm_120` ∈ `get_arch_list()` |
| B3 | 扩展 | 每个关键 `.so` 含 sm_120（cuobjdump/strings） |
| B4 | natten | 前向 smoke 无 `no kernel image` |
| B5 | e2e | 同图 seed42 → GLB > 1MB |
| B6 | 成本 | 记录纯推理 vs L40S/H100，**若更贵且不更快则不设默认** |

---

## 7. 明确不做什么（当前阶段）

- ❌ 不在 005 官版镜像上强行 `gpu=RTX-PRO-6000`  
- ❌ 不装 Windows `.whl` 到 Modal Linux  
- ❌ 不默认 GGUF 当「官方质量」对照  
- ❌ 未做 B0–B2 前不启动全量 natten 编译  

---

## 8. 推荐决策

1. **日常出片：L40S (v2) 或 H100 (005)** — 已验证。  
2. **若必须统一 PRO 6000 卡池：** 走 Plan A，严格分阶段探针。  
3. **预期工期：** 工程 1–3 次 Modal 长任务量级（视 natten 编译而定），失败概率高于 v2。  
4. **我方立场：** 方案 **存在且可执行**，但 **ROI 弱于 L40S**；v3 以研究/预研为主，不替代 v2。

---

## 9. 参考

- PyTorch cu128 / sm_120：官方 previous-versions & forums  
- NATTEN install：https://natten.org/install/  
- Windows PRO 6000 Trellis 轮子：https://huggingface.co/PixWizardry/AI_Trellis2-WHLs-RTX-PRO-6000  
- 本 lab：[`005-pixal3d`](../005-pixal3d/) · [`005-v2-pixal3d-l40s`](../005-v2-pixal3d-l40s/)
