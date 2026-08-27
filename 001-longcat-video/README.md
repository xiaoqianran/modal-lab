# 001-longcat-video — LongCat-Video on Modal

本实验复现美团开源 [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)，覆盖 T2V / I2V / Video Continuation / Long Video / Interactive / Storyboard。

## 架构

001 已迁移到 v2：**一个实验一个 `app.py`**。

```text
001-longcat-video/
├── app.py                 # 唯一入口：资源、Modal、CLI、远程执行
├── storyboard.py          # 独立 storyboard workflow
├── storyboards/           # storyboard 数据
├── LongCat-Video/         # 上游源码
├── inputs/                # 用户输入
├── tests/test_app.py      # 纯规划测试，不连接 Modal
├── UPSTREAM.md
└── README.md
```

不再存在：

```text
run.py
modal_app.py
lib/config.py
lib/demo_specs.py
configs/default.yaml
```

设计原则：

```text
仓库 main.py          只做实验 ID dispatch
      │
      ▼
experiment/app.py     实验唯一 composition root
      │
      ├── Pure planning
      │     ├── Profile
      │     ├── build_script_argv()
      │     └── run_summary()
      │
      └── Imperative shell
            ├── Modal Image / Volume
            ├── download_weights()
            ├── smoke()
            └── run_demo()
```

## 快速使用

推荐从仓库根目录运行：

```bash
# 本地状态，不初始化 Modal App、不构建镜像
python main.py 001 status

# 下载权重
python main.py 001 download

# GPU / CUDA / flash-attn / 权重冒烟
python main.py 001 smoke

# T2V
python main.py 001 t2v

# 双卡 profile
python main.py 001 t2v --profile pro6000-2

# 只看最终执行计划，不提交远程任务
python main.py 001 t2v --profile pro6000-2 --dry-run

# storyboard 默认使用 pro6000-long
python main.py 001 storyboard

# upstream 参数统一放在 -- 后，原样透传
python main.py 001 storyboard --dry-run -- --seed 7 --guidance_scale 5.0
```

也可直接使用 Modal：

```bash
cd 001-longcat-video
modal run app.py download
modal run app.py smoke
modal run app.py t2v
modal run app.py t2v --profile pro6000-2
```

纯本地的 `status / setup / --dry-run / --help` 推荐通过仓库 `main.py`，避免 `modal run` 为本地操作初始化整个 App。

## 资源 Profile

资源是原子配置，不再把 GPU / nproc / CPU / RAM / timeout 拆成五套独立覆盖参数。

| Profile | GPU | nproc | CPU | RAM | timeout |
|---|---|---:|---:|---:|---:|
| `pro6000-1` | RTX PRO 6000 | 1 | 4 | 32 GiB | 2h |
| `pro6000-2` | RTX PRO 6000 ×2 | 2 | 8 | 64 GiB | 2h |
| `a100-80-1` | A100 80GB | 1 | 4 | 32 GiB | 2h |
| `a100-80-2` | A100 80GB ×2 | 2 | 8 | 64 GiB | 2h |
| `h100-1` | H100 | 1 | 4 | 32 GiB | 2h |
| `pro6000-long` | RTX PRO 6000 | 1 | 8 | 64 GiB | 8h |

`context_parallel_size` 始终由 `Profile.nproc` 注入，避免多卡配置不一致。

## 参数所有权

`modal-lab` 只拥有基础设施参数：

```text
--profile
--no-compile
--checkpoint-dir
--output-subdir
--dry-run
```

上游模型参数不在本仓库复制 schema，统一原样透传：

```bash
python main.py 001 storyboard -- --seed 42 --num_frames 93
```

下面三项是实验不变量，不能在 `--` 后重复定义：

```text
--checkpoint_dir
--context_parallel_size
--enable_compile
```

这样 upstream 新增模型参数时，`modal-lab` 通常不需要修改代码。

## Storyboard

`storyboard.py` 是实验自己的领域 workflow，不再伪装成 `LongCat-Video/` 上游文件。

默认数据：

```text
storyboards/your_name_shinkai.json
```

默认行为：

```text
profile = pro6000-long
num_inference_steps = 24
spatial_refine_only = true
```

需要完整 refine：

```bash
python main.py 001 storyboard -- --full_refine
```

## Volume 与输出

```text
weights: modal-lab-longcat-weights  -> /weights
outputs: modal-lab-longcat-outputs  -> /outputs
checkpoint: /weights/LongCat-Video
```

生成结果会收集到：

```text
/outputs/<output-subdir>/
```

不再包装 Modal 的 Volume CLI。拉取结果直接使用：

```bash
modal volume get modal-lab-longcat-outputs / ./001-longcat-video/outputs
```

## Setup

如果本地没有上游源码：

```bash
python main.py 001 setup
```

上游版本见 [UPSTREAM.md](UPSTREAM.md)。

## 测试

纯逻辑测试不连接 Modal、不启动 GPU：

```bash
python -m unittest discover -s 001-longcat-video/tests -v
```

真实 Modal SDK 导入检查：

```bash
modal --version
```

## 已知输入依赖

官方 I2V 依赖：

```text
LongCat-Video/assets/girl.png
```

官方 Video Continuation 脚本依赖：

```text
LongCat-Video/assets/motorcycle.mp4
```

`app.py` 会在提交远程任务之前检查这些文件，避免付费 GPU 启动后才发现本地输入缺失。

## 权重与镜像

- 权重：`meituan-longcat/LongCat-Video`，全量约 83GB
- CUDA：12.8
- PyTorch：2.7.1 + cu128
- flash-attn：2.7.4.post1
- 默认 GPU：RTX PRO 6000 96GB

Avatar / 音频驱动系列暂不属于 001 的运行入口。
