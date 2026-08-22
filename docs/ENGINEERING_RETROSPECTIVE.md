# Engineering Retrospective

这份文档是我在维护 `modal-lab` / `modal-build` 过程中持续更新的工程反思记录。

它不是 changelog，也不是“做了什么”的总结。它只记录真正影响工程质量的事情：**哪里判断错了、为什么错、什么信号本来可以更早发现、以后应该形成什么固定规则。**

目标只有一个：同一类错误不要重复发生。

---

## 1. 总原则：精炼不是“代码更少”，而是“状态更少”

我一开始很容易把“精炼”理解成删文件、删依赖、删分支、把实现缩短。这个理解不够完整。

真正有价值的精炼，是减少系统需要同时维护的状态：

- 只支持一个明确 GPU 时，就不要保留第二套 GPU image / patch / benchmark 分支。
- build 阶段能完成的事情，不要推迟到 runtime。
- runtime image 已经提供的 CUDA 库，不要重复塞进 release bundle。
- upstream 能稳定完成的逻辑，不要再造一层 manager / adapter / factory。
- 一个 benchmark 只改变一个核心变量，不要同时改 quantization、resolution、decimation。

反过来，**为了少一个依赖而制造 6 倍性能退化，不叫精炼。**

Hunyuan3D 2.1 就出现过这个反例：为了不引入 Open3D，我把 `use_remesh` 直接关掉。代码确实更少了，但 paint 从约 46–65 秒恶化到 415 秒。这个结果说明 remesh 不是装饰步骤，而是在显著缩小纹理阶段的几何工作量。

最终更合理的解法不是恢复一个很重的 Open3D 依赖，而是保留“必须 remesh”这个事实，用已有的 `fast-simplification` 底层数组 API 把 mesh 收敛到 40k faces。

**永久规则：**

> 精炼时优先删除冗余状态，不要删除必要工作。任何“删掉以后更简单”的改动，都必须重新验证 latency、VRAM、输出规模和功能完整性。

---

## 2. Build 与 Runtime 的边界必须是硬边界

### 现象

旧实现里曾经出现：

- Worker 启动后再 `pip install`。
- Worker 启动后编译 CUDA/C++ extension。
- `trellis.cpp` release bundle 里重复携带 CUDA runtime libraries。

这些方案短期都能“跑起来”，但会让生产容器承担编译器、包管理器和推理服务三种角色。

### 为什么这是错的

生产推理 Worker 的职责应该尽可能单一：

```text
load artifact / weights
→ start engine
→ inference
→ save result
```

如果 runtime 还在做：

```text
apt/pip install
compile
link
patch dependency
```

那冷启动时间、失败模式、网络依赖、编译环境、GPU 架构和推理生命周期全部耦合在一起。

### 正确做法

对于 Python CUDA extension：

```text
modal-build
→ compile wheels
→ validate
→ manifest + sha256
→ GitHub Release
```

对于 native C++ runtime：

```text
modal-build
→ compile binaries/shared libs
→ ldd validation
→ package
→ GitHub Release
```

对于 Modal inference：

```text
runtime image
→ download/install prebuilt artifact
→ load weights
→ inference
```

Hunyuan3D 2.1 同样按这个原则收敛：

- `custom_rasterizer` 在 Modal image build 阶段编译。
- `mesh_inpaint_processor` 在 Modal image build 阶段编译。
- Worker runtime 不再现场编 CUDA extension。

### 永久规则

> 生产 Worker 禁止现场编译 CUDA/C++。如果某个依赖必须编译，它属于 image build 或独立 build repository，而不是 request lifecycle。

---

## 3. 不要把 NVIDIA Driver Library 打进应用制品

`trellis.cpp` 第一版 bundle 一度包含：

- `libcublasLt`
- `libcublas`
- `libcudart`

结果 bundle 从约 162 MiB 膨胀到约 716 MiB。

这是典型的责任边界错误。

生产 image 已经是：

```text
nvidia/cuda:12.9.1-runtime-ubuntu22.04
```

CUDA runtime libraries 应由这个 image 提供。

而 `libcuda.so.1` 更特殊：它不是 CUDA runtime image 应打包的文件，它属于 NVIDIA driver，由 Modal GPU host 在容器启动时注入。

因此动态依赖检查必须区分：

```text
真正 missing 的 application/runtime dependency
→ build fail

libcuda.so.1
→ build 阶段允许 missing
```

### 永久规则

> 应用制品只携带“应用自己拥有的二进制和动态库”。CUDA runtime 属于基础镜像，`libcuda.so.1` 属于 GPU host/driver。

---

## 4. “编译成功”不是 artifact 验收标准

早期的危险心态是：

> binary 出来了，应该能跑。

这不够。

native artifact 必须至少经过：

```text
compile
→ package
→ ldd / dynamic dependency validation
→ manifest
→ sha256
→ release
```

`ldd trellis-server` / `ldd trellis-cli` 的 validator 就实际发现了环境边界问题。

### 永久规则

> Build success 只证明 compiler 成功，不证明 artifact 可部署。Release 前必须验证动态依赖闭合性。

---

## 5. 依赖解析必须从一开始就被约束，不能“后面覆盖回来”

Hunyuan3D 2.1 image 第一次精简时，我把一批 Python 包先安装，再单独安装：

```text
torch==2.5.1
cu124
```

结果 `realesrgan / basicsr / timm` 等依赖在前一层解析时，先把 Torch 2.12.1 + CUDA 13 相关包拉了进来。

即使后面再覆盖回 Torch 2.5.1，也已经造成：

- image layer 巨大；
- 依赖语义混乱；
- CUDA 版本可能共存；
- 后续排障困难。

正确做法是让 Torch pin 和其他依赖进入同一次受约束解析，并给 PyTorch cu124 index 作为 extra index，让 resolver 从一开始就没有机会漂到 CUDA 13。

### 永久规则

> 关键基础依赖（Torch / CUDA ABI / Python major-minor）必须在第一次 dependency resolution 时就 pin 住。不要依赖“最后再 pip install 一遍覆盖”。

---

## 6. 缺依赖时，先判断它属于“必要功能”还是“偶然实现”

### `bpy` 的案例

Hunyuan 上游 `mesh_utils` 顶层 import `bpy`，但 Modal PyPI 镜像没有合适的 `bpy==4.0.0`。

最差的反应是：为了满足一个 import，把 Blender 整套运行时拉进来。

实际检查代码后发现，我们的主路径只需要 OBJ/GLB 转换，不需要 Blender 操作。因此更合理的做法是：

- 允许 `bpy` 缺失；
- GLB 转换走 `trimesh`；
- 不引入 Blender runtime。

### `open3d` 的案例

另一个方向也不能走极端。

缺 Open3D 时，我一开始直接关闭 `use_remesh`。虽然解决 import，但性能从约 46–65 秒退化到 415 秒。

这说明判断问题不能停在：

> 这个包能不能删？

而应该继续问：

> 这个包提供的“能力”是不是必要？如果必要，能不能用更轻的实现替代？

最终答案是：remesh 必要，Open3D 不必要。

### 永久规则

> 遇到缺依赖，不先“补包”，也不先“关功能”。先定位这个依赖提供的能力，再判断：能力是否必要、是否可以用更薄的实现替代。

---

## 7. Convenience API 可能偷偷带入大依赖

为了替代 Open3D，我选择了仓库里已经安装的 `fast-simplification`。

第一次用了：

```python
fast_simplification.simplify_mesh(...)
```

这个 convenience wrapper 又依赖 `pyvista`。

如果看到错误后直接 `pip install pyvista`，依赖树会继续变胖。

继续往下看 API 才发现底层：

```python
fast_simplification.simplify(vertices, faces, target_count=...)
```

可以直接处理 NumPy 数组，不需要 PyVista。

### 永久规则

> 遇到 wrapper 缺额外依赖时，先检查底层 API。很多 convenience layer 只是为了用户体验，不是核心算法的真正依赖。

---

## 8. CUDA 扩展编译失败时，不要默认怀疑 CUDA

Hunyuan `custom_rasterizer` 编译过程中，日志已经明确产生：

```text
compute_89
sm_89
```

说明 NVCC / GPU arch 路径是正确的。

真正失败点是在 host link 阶段调用了不存在的 `clang++`。

如果只看到“CUDA extension build failed”就开始改 CUDA version、Torch version 或 arch，会把问题越搞越大。

最终只需要明确：

```text
CC=gcc
CXX=g++
```

### 永久规则

> CUDA build 错误必须先区分：Python packaging、NVCC compile、host compile、link、runtime load。不要把所有 native build failure 都归类成“CUDA 不兼容”。

---

## 9. Python packaging metadata 失败，不等于源码编译失败

`custom_rasterizer` 第一次 build 失败，是 metadata 阶段缺：

```text
wheel
```

它甚至还没有进入真正 CUDA 编译。

如果不看错误发生阶段，就容易开始改 compiler/image/CUDA。

### 永久规则

> native Python package build 先判断失败阶段：metadata → build isolation → C/CUDA compile → link → import。每一层对应完全不同的修法。

---

## 10. Hugging Face 大模型下载必须考虑中断一致性

Hunyuan 首次下载模型时，任务中途被停止，Volume 中留下了半成品目录。

上游逻辑只检查：

```text
目录是否存在
```

于是下一次启动把半成品误认为完整模型，最后缺：

```text
model.fp16.ckpt
```

这不是简单的“下载失败”，而是缓存一致性问题。

最终代码改为验证关键文件：

```text
model directory exists
AND model.fp16.ckpt missing
→ remove incomplete cache
→ clean download
```

同时安装 `hf_xet`，避免 Xet repo 退回低效普通 HTTP 下载。

### 永久规则

> 大模型缓存不能以“目录存在”作为完整性判断。至少验证关键权重文件；如果下载过程不是原子的，就必须能自动恢复半成品缓存。

---

## 11. 请求取消必须传播到 native GPU process

`trellis.cpp` 使用常驻 C++ server。

早期问题是：Modal request 被取消后，Python 调用结束了，但 C++ server 可能继续做 GPU inference。

结果：

```text
request canceled
≠
GPU work canceled
```

这会产生孤儿 GPU 任务和直接成本浪费。

最终生命周期变成：

```text
@modal.enter
→ start server

generate
→ call server

cancel / exception
→ terminate server immediately

next request
→ restart if dead

@modal.exit
→ terminate server
```

### 永久规则

> 外层 request 生命周期必须能终止底层 native/GPU 工作。任何 subprocess/server 模型都要明确 cancel/exception/exit 三条清理路径。

---

## 12. 看到“远程崩溃”不能直接推断 OOM / segfault

一次 `trellis.cpp` 运行被认为“远程崩溃”。

实际 Modal 日志已经运行到：

```text
DINOv3
→ sparse structure
→ shape LR
→ shape HR
→ FlexiDualGrid decode
→ mesh generated
```

GPU 也正常识别：

```text
NVIDIA L40S
compute capability 8.9
~45 GiB VRAM
```

真正结束原因是：

```text
Stopping app - user stopped from dashboard.
Runner terminated.
```

### 永久规则

> “任务结束”不是“程序 crash”。判断顺序应该是：process exit signal → CUDA OOM → Python exception → timeout → eviction → app/user stop。先从调度器和容器日志确认终止源。

---

## 13. 不要优化已经证明不是瓶颈的地方

`trellis.cpp` 常驻 server 之后，warm inference 仍然约 113 秒。

profile 已经明确：

```text
sparse     ~4.4s
shape LR   ~4.5s
shape HR   ~54s
mesh decode ~20+s
```

这时继续优化：

```text
server startup ~0.3s
```

没有意义。

### 永久规则

> 优化前先做阶段级 timing。任何只占总时长 1% 以下的步骤，没有明确原因不要优先优化。

---

## 14. Benchmark 必须先定义“比较的是什么”

Hermite 和 `trellis.cpp` 第一轮结果大约是：

```text
~12s vs ~121s
```

但同时：

```text
trellis.cpp GLB ~264 MiB
Hermite GLB     ~48 MiB
```

`trellis.cpp` 中间 mesh 甚至达到约：

```text
7.7M vertices
15.4M faces
```

因此这组数字可以回答：

> 两个项目按当前 pipeline/default workload 跑，谁更快？

但不能直接回答：

> 在同等输出质量/几何预算下，哪个模型效率更高？

### 正确 benchmark 顺序

第一层先一次只变一个变量：

```text
F16 / Q8 / Q4
1024 / 512
```

记录：

- inference latency
- wall time
- peak VRAM
- model/runtime storage
- vertices/faces
- GLB size

第二层再统一输出预算：

```text
500k / 1M / 2M faces
```

然后比较 latency / size / visual quality。

### 永久规则

> Benchmark 结果必须附带 workload 和 output complexity。没有这些上下文的单个 latency 数字非常容易误导。

---

## 15. Cold、Warm、首次权重同步不能混成一个数字

Hunyuan full 第一次运行还需要下载 texture 侧模型文件。

这个时间不应该被混进 warm inference benchmark。

同理：

- image build
- artifact download
- model weight sync
- model load
- first inference
- warm inference

必须分开记录。

### 永久规则

> benchmark metadata 至少区分：cold container、cold weights、warm container、warm request。不要用一个 wall time 覆盖所有阶段。

---

## 16. 文档必须跟代码一起更新，否则旧文档会反向制造 bug

`trellis.cpp v2` 已经不再打包 CUDA runtime libraries，但 README 和 env 描述仍然写：

```text
bundle contains CUDA runtime libraries
```

这类文档债务不是“文字小问题”。下一位维护者很可能根据它重新把 `libcublasLt` 塞回 bundle。

### 永久规则

> 修改 runtime contract / artifact contract 时，README、manifest schema、env 描述必须一起检查。文档里的错误架构描述视同代码 bug。

---

## 17. 不要为了架构漂亮制造抽象层

这类工程很容易出现：

```text
Factory
Adapter
EngineManager
ServiceManager
BackendRegistry
```

但当实际只有两个引擎，而且它们生命周期、依赖栈、数据路径完全不同，这些抽象只会隐藏差异。

目前更合理的是：

```text
modal-build/
├── hermit_trellis2_plus_plus.py
└── trellis_cpp.py

modal-3D/
├── hermit_trellis2_plus_plus.py
└── trellis_cpp.py
```

每个文件直接表达自己的真实 runtime。

### 永久规则

> 在重复模式真正稳定出现之前，不抽象。允许两个清晰的 200 行文件，好过一个 500 行“通用框架”。

---

## 18. 并行协作时，Git 操作必须限定责任范围

这个仓库经常有人同时在其他目录工作，例如 Pixal3D。

因此不能把：

```bash
git add .
```

当成默认动作。

安全流程应该是：

```text
git status --short
→ 明确目标目录
→ git add <target>
→ 检查 commit file list
→ fetch
→ rebase origin/main
→ 再次检查
→ push
```

`modal-build` 文档提交时也遇到过远端 `master` 在本地提交期间前进。第一次 push 正常被拒绝，随后 rebase，再确认自己的 commit 只包含 README 后推送。

### 永久规则

> 多人并行仓库里，提交前和 rebase 后都检查“我的 commit 实际包含哪些文件”。不要因为其他人的工作区变化而覆盖、清理或顺手提交。

---

## 19. 长任务不要绑死在一次远程 Shell 调用上

VPS 网关对长命令多次出现 Cloudflare 524。

一开始我用了：

```text
long modal run
或
sleep 150 && tail log
```

这让“远程 shell 请求超时”和“Modal 任务失败”很容易被混淆。

更可靠的方式是：

```text
nohup modal run ... > log 2>&1
→ shell 立即返回
→ 短轮询 log / exit file
```

这样任务生命周期和控制通道生命周期解耦。

### 永久规则

> 预计超过远程控制通道超时时间的任务，用后台进程 + 日志 + exit code 文件。不要让一次 shell RPC 承担长任务生命周期。

---

## 20. 工具链存在时不要重复安装

这轮里我犯过一个很具体的低级错误：机器已经通过 `uv tool` 安装了 Modal CLI，我却先尝试：

```text
pip install modal
uvx ...
```

这是多余的，而且可能污染本机环境。

正确做法应该先：

```bash
command -v modal
uv tool list
modal --version
```

确认已有工具，再直接使用。

### 永久规则

> 安装工具前先检查系统已有工具。尤其是用户明确维护的 `uv tool` / system package / project venv，不要擅自创建第二套。

---

## 21. L40S-only 的真正含义是删除第二套状态，而不是加一个注释

用户明确说尽量只用 L40S。

真正的收敛应该是：

- `gpu="L40S"`
- `TORCH_CUDA_ARCH_LIST=8.9`
- runtime probe 验证 capability `(8, 9)`
- 删除 PRO6000 专属路径
- 删除对应 viewer 样本/文档分支
- benchmark 只维护 L40S 基线

而不是保留两套代码，然后写一句：

```text
# currently prefer L40S
```

### 永久规则

> “只支持 X”应该反映在代码路径、build arch、测试、文档和资产上，而不是只反映在默认参数里。

---

## 22. 真实验证必须逐层推进

这轮 Hunyuan 重构最终采用了以下验证顺序：

```text
1. Python syntax
2. local CLI status
3. Modal image build
4. native extension build
5. L40S probe
6. shape smoke
7. full smoke
8. performance sanity check
9. git diff scope
10. commit / rebase / push
```

这个顺序很重要。

如果直接跑 full smoke，遇到错误时不知道是 image、extension、weight、shape、paint 还是 export。

如果只跑 probe，又可能出现“模型能 load，但真实生成失败”。

### 永久规则

> 验证从便宜到昂贵、从局部到完整。每一层只证明一件事，并为下一层缩小故障空间。

---

## 23. 当前 Hunyuan3D 2.1 这一轮最值得保留的结果

这不是 benchmark 文档，只记录用于验证工程判断的结果。

L40S / Torch 2.5.1+cu124 / CUDA 12.4 / sm_89：

```text
shape only
~29.72s
peak VRAM ~7.63 GiB

full
shape ~30.23s
paint ~46.73s
total ~110.84s
peak VRAM ~20.22 GiB
GLB ~1.32 MB
```

最关键的不是这些数字本身，而是中间的反例：

```text
关闭 remesh
→ paint ~415s
```

这证明“更少代码/依赖”不能脱离 workload profile 判断。

---

# 每次改动前的固定自检

以后处理类似 Modal / CUDA / 3D pipeline 任务，我应该先回答这些问题：

1. **职责边界**：这件事属于 build、image build、runtime 还是 request？
2. **GPU 目标**：实际只支持哪个 GPU / SM？是否还残留第二套状态？
3. **依赖归属**：这个库应该由 artifact、runtime image 还是 GPU host 提供？
4. **依赖解析**：Torch/CUDA/Python 是否从第一次 resolve 就被 pin 住？
5. **缓存一致性**：模型下载中断后能否自动恢复？
6. **生命周期**：request cancel 能否停止底层 GPU/native process？
7. **性能**：我删除的“复杂步骤”是否其实承担重要的 workload reduction？
8. **benchmark**：比较双方的输入、resolution、precision、mesh complexity 是否真的可比？
9. **验证层级**：syntax → build → probe → smoke → full 是否逐层通过？
10. **协作范围**：最终 commit 是否只包含我负责的目录/文件？
11. **文档一致性**：runtime/artifact contract 改了，README 是否同步？
12. **工具环境**：已有 CLI/toolchain 是否已经存在，是否在重复安装？

---

# 维护约定

以后只要出现下面任意一种情况，我都应该更新这份文档：

- 我做出了错误判断并被真实运行结果推翻；
- 一个错误暴露了新的工程边界；
- 为了修问题引入的方案后来证明太重；
- benchmark 结果揭示之前的比较方法不公平；
- Git/Modal/VPS 工具链出现可复用的操作教训；
- 用户指出我的工作方式存在可以固化为规则的问题。

如果只是新增功能、正常修 bug、更新版本，而没有产生新的工程认识，则不需要机械追加。

**这份文档的价值不在长度，而在它能否减少下一次重复犯错。**
