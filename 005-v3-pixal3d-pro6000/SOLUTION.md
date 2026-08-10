# 005-v3 · RTX PRO 6000（Blackwell sm_120）真实方案（第二轮深挖）

> **状态：方案已收紧到可执行配方 · 仍未在 Modal 上实跑**  
> 本轮新增：**Linux 上已有人跑通 TRELLIS.2 / Pixal3D 同族栈** 的可复现步骤（非仅 Windows 轮子）。

---

## 0. 结论（更新）

| 问题 | 答案 |
|------|------|
| 有没有方案？ | **有，而且比第一轮更具体。** |
| 最硬的证据 | ① **microsoft/TRELLIS.2 #143** 评论：RTX **5090 (sm_120)** WSL2 上 **example.py → 带贴图 GLB**；② **animede/image-3d** `requirements-pixal3d.txt`：**RTX PRO 6000 实机** 的 **Pixal3D 专用** 安装说明（torch 2.11+cu128 · arch 12.0）。 |
| 和「只有 Windows 轮子」比 | Windows 仍是旁证；**Linux 源码编 + cu128 是 Modal 正路**。 |
| 005 / v2 栈？ | **仍不可用**（torch2.6+cu124）。 |
| 推荐默认出片？ | 仍是 **v2 L40S / 005 H100**。PRO 6000 是「卡池绑定」时的 Plan A。 |
| 下一步烧钱顺序 | **B0 认卡 → B1 编 o-voxel 一条线 → B2 natten → B3 smoke**，禁止一步梭哈。 |

---

## 1. 硬件 & 为什么旧栈必死

| | PRO 6000 | 5090（同族证据） | L40S (v2) |
|--|----------|------------------|-----------|
| SM | **sm_120 / 12.0** | sm_120 | sm_89 |
| 旧栈 torch2.6 cu124 | ❌ | ❌ | ✅ 已通 |
| 需要 | **cu128+ torch + arch=12.0 扩展** | 同左 | arch=8.9 即可 |

**根因：** 缺 **sm_120 cubin**，不是显存。

---

## 2. 本轮新发现的「可抄配方」

### 配方 α — **Pixal3D 专用 · PRO 6000 实机文档**（最贴我们）

来源：[`animede/image-3d` → `requirements-pixal3d.txt`](https://github.com/animede/image-3d/blob/master/requirements-pixal3d.txt)

作者写明：**RTX PRO 6000 Blackwell (sm_120) 上验证**，且目标就是 **TencentARC/Pixal3D** 隔离环境。

| 项 | 钉死值 |
|----|--------|
| Python | **3.10**（Pixal3D 钉 pin 友好） |
| torch | **`2.11.0+cu128`** · torchvision **`0.26.0+cu128`** |
| CUDA 编译 | **`CUDA_HOME=/usr/local/cuda-12.8`**（**不要**让 nvcc 漂到系统 CUDA 13.x） |
| ARCH | **`TORCH_CUDA_ARCH_LIST=12.0`** |
| o-voxel 链 | TRELLIS.2 的 `o-voxel`（会拉 CuMesh / FlexGEMM）源码编 |
| 光栅 | 优先 **`facebookresearch/drtk`（MIT）** 替代 nvdiffrast；nvdiffrast 仅 fallback |
| NATTEN | **`NATTEN_CUDA_ARCH=12.0` · `natten==0.21.0`** 源码编（文档写 ~9 分钟） |
| Attention | **`ATTN_BACKEND=sdpa`** · **不装 flash_attn**（ABI/预编译坑） |
| MoGe | **可不装**，固定 FOV |
| rembg | 避开门禁 RMBG-2.0，用公开 BiRefNet / 应用侧 RGBA |

关键环境变量（原文精要）：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"

# o-voxel (+ CuMesh/FlexGEMM)
pip install --no-build-isolation third_party/TRELLIS.2/o-voxel

# 光栅（推荐 drtk）
pip install --no-build-isolation "git+https://github.com/facebookresearch/drtk.git"

# 可选 nvdiffrast
# pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"

NATTEN_CUDA_ARCH="12.0" NATTEN_N_WORKERS=8 \
  pip install natten==0.21.0 --no-build-isolation
```

**对 Modal 的含义：** 这就是 **Plan A 的黄金模板**——比「抽象的整栈升级」多了：**具体 torch 版本、CUDA_HOME 对齐、drtk 替代、明确不装 flash_attn**。

---

### 配方 β — **TRELLIS.2 官方 issue · 5090 端到端 GLB**

来源：[microsoft/TRELLIS.2#143](https://github.com/microsoft/TRELLIS.2/issues/143) 高赞实操评论（WSL2 Ubuntu · RTX **5090 sm_120** · **example.py → textured GLB**）

| 项 | 值 |
|----|-----|
| Python | 3.11 |
| torch | **stable `2.11.0+cu128`**（明确写：比 nightly 更稳） |
| 编译器 | **gcc/g++ 13**（gcc 14 会被 CUDA 12.8 拒） |
| 工具链坑 | Ubuntu 新 glibc 需 **旧 sysroot** 或修 CUDA `math_functions` 冲突 |
| 构建 | `TORCH_CUDA_ARCH_LIST=12.0` + `./setup.sh --basic --nvdiffrast --nvdiffrec --o-voxel` |
| flash-attn | 他们装了 2.8.3；**Pixal3D 可用 sdpa 跳过**（配方 α） |
| 结果 | 形状+贴图采样 + GLB，5090 上「远不到 1 分钟级采样」（下载后） |

**对 Modal 的含义：** **同架构 sm_120 已有人在 Linux 容器式环境出 GLB**；我们 v3 不是赌一张不存在的牌。Modal 镜像里用 **devel CUDA 12.8 + gcc-13** 对齐即可。

---

### 配方 γ — **Windows 预编译全家桶**（旁证 / 不进 Modal）

| 源 | 内容 |
|----|------|
| [PixWizardry HF · PRO 6000](https://huggingface.co/PixWizardry/AI_Trellis2-WHLs-RTX-PRO-6000) | flex_gemm · o_voxel · cumesh · nvdiffrast · nvdiffrec · flash_attn；**Win11 · py3.12 · torch2.10 · CUDA13 · sm120** |
| [Saganaki22/Pixal3D-ComfyUI windows_wheels](https://github.com/Saganaki22/Pixal3D-ComfyUI/blob/main/docs/windows_wheels.md) | **NATTEN sm120** 社区 wheel 表（drbaph torch2.10 cu130；naxneri cu128） |
| [drbaph NATTEN HF](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64) | natten 0.21.6 **Blackwell sm120**（Windows） |

**结论：** 扩展集合 **可以** 在 sm_120 上编出；**win_amd64 不能装进 Modal**。用途：对照「该编哪些包、版本附近落点」。

---

### 配方 δ — **仅换 torch 的「半吊子」**（❌ 不足）

只装 `torch+cu128` 仍用 HF demo **sm_90 扩展** → 依旧 `no kernel image`。  
**必须** 重编 flex_gemm / o_voxel / cumesh / nvdiffrast|drtk / natten。

---

## 3. Modal 上推荐落地：Plan A*（合并 α+β）

### 3.1 镜像与运行时

```text
GPU:        RTX-PRO-6000
Base:       nvidia/cuda:12.8.1-devel-ubuntu22.04  (或 12.8.0)
Python:     3.10 或 3.11
torch:      2.11.0+cu128  (首选；2.7–2.10+cu128 也可，以 arch_list 含 sm_120 为准)
g++:        13.x  (避免 14+)
ENV:
  TORCH_CUDA_ARCH_LIST=12.0
  NATTEN_CUDA_ARCH=12.0
  CUDA_HOME=/usr/local/cuda   # 镜像内须是 12.8 系
  ATTN_BACKEND=sdpa
  SPARSE_ATTN_BACKEND=sdpa    # 若代码支持
  FORCE_CUDA=1
  MAX_JOBS=4
```

**Modal 注意：**

- 镜像 **devel** 才能编扩展；runtime-only 不够。  
- **nvcc 版本必须与 torch 的 CUDA major 对齐**（α 文中系统 13.0 nvcc + torch cu128 = 灾难）。  
- Volume 缓存：`/wheels/sm120/torch211-cu128-cp310/` 类似 v2。  
- 权重可与 005/v2 **共用逻辑**（HF `TencentARC/Pixal3D`），Volume 可独立以免踩版本。

### 3.2 扩展构建顺序（建议）

| 顺序 | 包 | 备注 |
|------|-----|------|
| 1 | utils3d | wheel 或 git |
| 2 | **o-voxel**（含 flex_gemm / cumesh） | TRELLIS.2 子树；最关键 |
| 3 | **drtk** | 优先于 nvdiffrast |
| 4 | nvdiffrast / nvdiffrec | 若坚持官方路径 |
| 5 | **natten==0.21.0** | `NATTEN_CUDA_ARCH=12.0`；最长 |

每步：`pip wheel` → Volume commit → import smoke。

### 3.3 推理侧减负（提高一次成功率）

对齐 α：

- `ATTN_BACKEND=sdpa`，**跳过 flash_attn**  
- rembg → **BiRefNet**（005 已有补丁）  
- 可选固定 FOV，减少 MoGe 依赖  
- low_vram + 1024 与 005/v2 对齐便于对照  

### 3.4 门禁（烧钱顺序）

| 阶段 | 动作 | 通过标准 | 估时（量级） |
|------|------|----------|--------------|
| **B0** | 只起 PRO 6000 + torch cu128 | name / capability≈(12,0) / `sm_120∈arch_list` | 几分钟 |
| **B1** | 编 o-voxel 链 + import | 无 compile error；`import o_voxel` | 数十分钟 |
| **B2** | drtk 或 nvdiffrast | import OK | 数分钟–十几分钟 |
| **B3** | natten 源码 + na2d smoke | 无 no kernel image | ~10–40 min |
| **B4** | 权重 + smoke GLB | GLB > 1MB | ~5–15 min 推理 + 下载 |

**任一步失败就停**，写进 `GPU_BENCHMARK.md` 再改。

---

## 4. Plan 对照表

| Plan | 内容 | 可行性 | 适合 |
|------|------|--------|------|
| **A\*** | 配方 α+β：torch2.11+cu128 · arch12.0 · o-voxel · drtk · natten · sdpa | **高（有 Linux+PRO6000/5090 先例）** | **Modal 主推** |
| B | 对齐 Windows torch2.10+cu130 再 Linux 自编 | 中 | 想贴 Comfy 版本 |
| C | 直接装 PixWizardry / drbaph **win** wheel | ❌ Modal | 本机 Windows |
| D | 只换 gpu=PRO6000 不换栈 | ❌ 已证失败 | — |
| E | 继续 L40S/H100 出片 | ✅ 已通 | **生产默认** |

---

## 5. 与 005 / 005-v2 的关系

| | 005 | 005-v2 | **005-v3** |
|--|-----|--------|------------|
| GPU | H100 | L40S | PRO 6000 |
| torch | 2.6 cu124 | 2.6 cu124 | **2.11 cu128（目标）** |
| ARCH | sm_90 轮子 | **8.9 自编** | **12.0 自编** |
| 光栅 | nvdiffrast HF | nvdiffrast 自编 | **优先 drtk** |
| 实跑 | ✅ | ✅ | 🔒 未跑 |
| 估费参考 | ~$0.31 | **~$0.17** | 待测（卡单价通常不低） |

**工程复用：** v2 的 Volume 轮子策略、verify 脚本结构、rembg 补丁、smoke CLI 可直接迁；**仅替换 base image / arch / 包列表**。

---

## 6. 风险清单（实装前必读）

1. **CUDA_HOME 漂到 13.x** → 编译挂（α 已踩）  
2. **gcc 14+** → torch/CUDA 拒编（β 已踩）  
3. **glibc 过新** → nvcc math 冲突（Modal Ubuntu 22.04 通常比 25.04 省心）  
4. **natten 编很久 / 吃内存** → `MAX_JOBS=2~4`  
5. **transformers 大版本** → DINOv3 属性路径可能变（β 有补丁经验）  
6. **PRO 6000 单价** → 即使跑通也可能 **不如 L40S 划算**  

---

## 7. 决策建议（更新）

1. **出片生产：** 继续 **005-v2 L40S**（已证）或 005 H100。  
2. **必须 PRO 6000：** 按 **Plan A\***，从 **B0 探针** 开始；模板优先抄 **animede requirements-pixal3d** + **TRELLIS.2#143 工具链约束**。  
3. **不要** 再试 005 官版 HF 轮子上 PRO 6000。  
4. **不要** 装 Windows wheel。  
5. 我方判断：**方案真实存在，Linux 先例已有；Modal 落地是工程问题不是科学问题**——但 **ROI 仍可能弱于 L40S**。

---

## 8. 参考链接

| 类型 | 链接 |
|------|------|
| Pixal3D@PRO6000 安装说明 | https://github.com/animede/image-3d/blob/master/requirements-pixal3d.txt |
| TRELLIS.2 5090 端到端 | https://github.com/microsoft/TRELLIS.2/issues/143 |
| TRELLIS.2 CUDA 错配讨论 | https://github.com/microsoft/TRELLIS.2/issues/19 |
| Windows PRO6000 扩展轮子 | https://huggingface.co/PixWizardry/AI_Trellis2-WHLs-RTX-PRO-6000 |
| Pixal3D-Comfy NATTEN 表 | https://github.com/Saganaki22/Pixal3D-ComfyUI/blob/main/docs/windows_wheels.md |
| NATTEN 官方 Blackwell | https://natten.org/install/ |
| Modal GPU 列表 | https://modal.com/docs/guide/gpu |
| 本 lab | `005-pixal3d` · `005-v2-pixal3d-l40s` |
