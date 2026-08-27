# modal-lab Architecture v2

`modal-lab` 是**实验集合**，不是 Runner Framework。

目标：高内聚、少边界、少状态、少重复事实源；只有真实复杂度出现后才拆文件。

## 总体结构

```text
modal-lab/
│
├── main.py
│   │  仓库级 Dispatcher
│   │  只负责：experiment id -> experiment entry
│   │
│   └──────────────────────────────────────────────┐
│                                                  │
├── 001-longcat-video/                             │
│   ├── app.py  <──────────────────────────────────┘
│   │   │
│   │   ├── Experiment facts
│   │   ├── Small domain values
│   │   ├── Pure planning
│   │   ├── Modal infrastructure
│   │   ├── Remote operations
│   │   └── Local entrypoint
│   │
│   └── storyboard.py      # 真实独立领域 workflow 才拆
│
├── 002-.../
│   └── run.py             # legacy，逐实验迁移
│
└── ...
```

最终目标：

```text
experiment/
└── app.py
```

只有真实业务边界出现后才变成：

```text
experiment/
├── app.py
└── <domain-workflow>.py
```

## 调用链

```text
用户
 │
 ▼
main.py
 │
 │ experiment dispatch
 ▼
experiment/app.py
 │
 ├── Pure planning
 │      │
 │      └── list[str] / Profile / Path
 │
 └── Modal shell
        │
        └── Function.remote(...)
               │
               ▼
          upstream code
```

禁止恢复下面这种链：

```text
run.py
  -> config.py
    -> RunConfig
      -> payload dict
        -> JSON
          -> modal CLI
            -> JSON decode
              -> modal_app.py
```

## 核心规则

1. **一个实验一个变化边界。**
2. **默认一个 `app.py`。**
3. **一个事实只允许一个 source of truth。**
4. **相邻两层必须提供不同抽象，否则合并。**
5. **Modal 已有能力，不二次包装。**
6. **上游 CLI schema 不复制；模型参数尽量 opaque passthrough。**
7. **Python 层之间直接传 Python 数据，不用 JSON/CLI 做内部 IPC。**
8. **资源不变量用小类型建模，不建立通用配置框架。**
9. **允许少量局部重复，拒绝错误的跨实验抽象。**
10. **按压力拆分，不按技术名词拆分。**

## 什么时候拆文件

```text
                某块代码开始增长
                       │
                       ▼
            是否有独立知识/生命周期？
                   /          \
                 否            是
                 │             │
                 ▼             ▼
            留在 app.py    是否独立变化/测试/依赖？
                               /          \
                             否            是
                             │             │
                             ▼             ▼
                        留在 app.py      拆文件
```

行数只能作为气味，不能作为拆分依据。

## 允许的 Dispatcher 例外

仓库 `main.py` 可以存在，因为它提供真实的新抽象：

```text
001 -> 001-longcat-video
022 -> 022-hunyuan3d-2.1
```

迁移期：

```text
app.py 优先
run.py legacy fallback
```

纯本地操作（`status/setup/help/dry-run`）直接运行 `app.py`，避免无意义地初始化 Modal App；真正云端操作才进入 `modal run`。

## 参数所有权

```text
modal-lab owns                upstream owns
---------------------------   ---------------------------
GPU profile                   prompt / seed
nproc                         inference steps
checkpoint mount              model-specific options
compile strategy              input image/video flags
Modal Volume                  future upstream flags
```

边界上的少量参数（例如 `context_parallel_size`）若必须与 GPU profile 保持不变量，则归实验基础设施所有，并禁止 upstream passthrough 重复定义。

## 参考思想

- Vertical Slice Architecture — 按变化轴组织代码
- Parnas Information Hiding — 模块隐藏设计决策，而不是隐藏执行步骤
- Ousterhout Deep Modules — 接口成本必须换来真实隐藏复杂度
- Functional Core / Imperative Shell — 纯规划与副作用边界清晰
- Monolith First — 先形成自然边界，再物理拆分

当前已迁移到 v2：

```text
001-longcat-video
005-v2-pixal3d-l40s
005-v3-pixal3d-pro6000
020-triposr
022-hunyuan3d-2.1
023-a-spar3d
023-b-sf3d
026-chatterbox
027-qwen3-tts
028-fish-s2
029-voxcpm2
030-vibevoice
031-cosyvoice3
032-indextts2
033-f5tts
034-higgs
```

这些实验都已删除 `run.py -> modal_app.py` pass-through 层。其他实验继续逐个迁移，不做机械批量改名。
