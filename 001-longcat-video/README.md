# 001-longcat-video — 复现 LongCat-Video（Modal）

在 [Modal](https://modal.com) 上复现美团开源视频生成模型
[**LongCat-Video**](https://github.com/meituan-longcat/LongCat-Video)
（13.6B，T2V / I2V / Video-Continuation / 长视频等）。

| 项 | 链接 |
|----|------|
| 代码 | https://github.com/meituan-longcat/LongCat-Video |
| 权重 | https://huggingface.co/meituan-longcat/LongCat-Video |
| 论文 | https://arxiv.org/abs/2510.22200 |
| 项目页 | https://meituan-longcat.github.io/LongCat-Video/ |

## 目录

```text
001-longcat-video/
  run.py              # 本机入口（setup / download / smoke / t2v …）
  modal_app.py        # Modal 镜像、Volume、下载与推理
  README.md
  LongCat-Video/      # 上游浅克隆（官方 demo + longcat_video 包）
  outputs/            # pull-outputs 后的本地结果（不入库）
```

## 资源与费用预期

| 资源 | 说明 |
|------|------|
| 权重体积 | HF 全量约 **83GB 磁盘**（DiT 分片 ~54GB + UMT5 ~23GB + VAE/LoRA）；**不等于** VRAM 占用 |
| Volume | `modal-lab-longcat-weights`、`modal-lab-longcat-outputs` |
| 默认 GPU | **`RTX-PRO-6000`**（NVIDIA RTX PRO 6000 Blackwell，**96GB** VRAM） |
| 多卡 | `--two-gpu` → `RTX-PRO-6000:2` + context parallel 2 |
| 镜像构建 | CUDA **12.8** + PyTorch **2.7.1+cu128**（sm_120）+ flash-attn 编译，首次约 15～40 分钟 |
| 下载 | 全量权重视带宽约数十分钟～数小时（支持续传） |
| 推理 | 官方 480p T2V 50 步 + distill + 720p refine，单次可能 **数十分钟** |

### 为什么选 RTX PRO 6000

- Modal 字符串：`gpu="RTX-PRO-6000"`（见 [Modal GPU 文档](https://modal.com/docs/guide/gpu)）
- **96GB** 显存：比 A100-80GB 更宽裕；权重在磁盘上 ~83GB，但 bf16 推理常驻大约是「DiT 13.6B ≈ 27GB + 文本编码器 + VAE + 激活/中间帧」
- **Blackwell sm_120**：必须用 PyTorch **cu128**（本实验 `torch==2.7.1+cu128`）。官方 README 的 2.6+cu124 在这张卡上会报 *not compatible*，kernel 跑不起来
- 单卡跑官方 480p→720p 流程更不容易顶满；若仍 OOM，再开 `--two-gpu` 或降分辨率/帧数
- 定价参考（Modal，以官网为准）：RTX PRO 6000 约 **$0.000842/s**，介于 H100 与 A100-80GB 之间

> L40S / A10G（24GB）大概率 OOM。配额无 PRO 6000 时，改 `modal_app.py` 的 `DEFAULT_GPU` 为 `"A100-80GB"` / `"H100"` / `"H200"`。

## 前置

```bash
pip install modal
modal token new          # 若尚未登录
# 可选：HF 令牌（公开模型一般不需要）
export HF_TOKEN=hf_...
```

## 用法

在仓库根或本目录均可：

```bash
# 状态
python main.py 001 status
# 或
cd 001-longcat-video && python run.py status

# 1) 确保上游代码存在（已随实验目录克隆则可跳过）
python run.py setup

# 2) 下载权重到 Modal Volume（只做一次）
python run.py download

# 3) 冒烟：CUDA / flash-attn / 权重目录
python run.py smoke

# 4) 官方 Text-to-Video demo
python run.py t2v
# 双卡（profile=pro6000-2，nproc=cp=2）
python run.py t2v --two-gpu
# 关 compile + 显式透传官方参数
python run.py t2v --no-compile -- --context_parallel_size 1
# 资源档位
python run.py t2v --profile a100-80-1

# 只校验配置（不连云）
python run.py validate-config --demo i2v --two-gpu

# 其他官方 demo
python run.py i2v
python run.py continuation
python run.py long
python run.py interactive

# 5) 把生成的 mp4 拉回本地 ./outputs
python run.py pull-outputs
```

### 配置优先级

```text
CLI 显式参数  >  配置文件（--config 或 configs/default.yaml）  >  代码默认值
```

基础设施（GPU 档位、CPU、内存、超时、compile 便捷开关）与官方脚本参数分离。  
官方基础 Demo **真实 CLI 仅 3 项**：`--checkpoint_dir` / `--context_parallel_size` / `--enable_compile`。  
prompt、分辨率、图/视频路径写死在官方脚本内，**不能**当作 CLI 伪造透传。

透传示例：

```bash
python run.py t2v --script-arg context_parallel_size=1
python run.py t2v -- --checkpoint_dir=/weights/LongCat-Video --enable_compile
```

资源档位（`lib/config.py` / `modal_app.RESOURCE_PROFILES`）：  
`pro6000-1`（默认）、`pro6000-2`、`a100-80-1`、`a100-80-2`、`h100-1`。  
云端用 Modal `Function.with_options` 覆盖 gpu/cpu/memory/timeout。

等价的纯 Modal 调用：

```bash
cd 001-longcat-video
modal run modal_app.py --action download
modal run modal_app.py --action smoke
modal run modal_app.py --action demo --demo t2v
modal run modal_app.py --action demo --demo t2v --two-gpu
```

本地单测（不连云）：

```bash
cd 001-longcat-video && python -m unittest tests.test_config -v
```

## 与官方步骤的对应关系

| 官方 | 本实验 |
|------|--------|
| `git clone … LongCat-Video` | `LongCat-Video/` 或 `python run.py setup` |
| `conda` + `pip install -r requirements.txt` + flash-attn | `modal_app.py` 镜像（CUDA 12.4 + torch 2.6 + flash-attn） |
| `huggingface-cli download … ./weights/LongCat-Video` | `python run.py download` → Volume `/weights/LongCat-Video` |
| `torchrun run_demo_text_to_video.py --checkpoint_dir=…` | `python run.py t2v` |

官方单卡示例：

```bash
torchrun run_demo_text_to_video.py \
  --checkpoint_dir=./weights/LongCat-Video \
  --enable_compile
```

Avatar / 音频驱动（`LongCat-Video-Avatar`、`Avatar-1.5`）权重与依赖更大，**本实验 001 先覆盖基础 LongCat-Video**；需要时再开 `002-longcat-avatar`。

## 输出

- 容器内官方脚本在仓库根目录写 `output_*.mp4`
- 任务结束会挪到 Volume：`/outputs/<demo>/`
- 本机：`python run.py pull-outputs` → `./outputs/`

## 排错

| 现象 | 处理 |
|------|------|
| 找不到 `modal` | `pip install modal` 并 `modal token new` |
| 镜像构建卡在 flash-attn | 正常，首次编译很慢；失败可看 Modal 构建日志，检查 CUDA devel 镜像 |
| `权重未找到` | 先 `python run.py download`，再用 `smoke` 看 `weights.size_gb` |
| CUDA OOM | 确认是 `RTX-PRO-6000`；仍不够则 `--two-gpu`，或改官方脚本分辨率/帧数 / 换 `H200` |
| GPU 配额 / 支付 | 参考 modal101 `14_gpu_matrix.py` 探测可用卡型 |
| 多卡 NCCL | 确保 `context_parallel_size` 与 `nproc_per_node` 一致（`run_demo_2gpu` 已对齐为 2） |

## 许可

- 上游代码与模型权重：**MIT**（见 `LongCat-Video/LICENSE`）
- 本实验脚手架代码随工作区约定；使用时请遵守 Meituan LongCat 的使用说明与当地法规
