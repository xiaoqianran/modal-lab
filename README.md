# modal-lab

Modal 实验台：按 `NNN-topic` 编号做可复现实验。

## 结构

```text
main.py                 # 入口，调度到 001 / 002 / …
001-longcat-video/      # 美团 LongCat-Video 视频生成复现
002-unlimited-ocr/      # 百度 Unlimited-OCR 文档解析
003-mineru/             # OpenDataLab MinerU 文档解析
004-minimax-h3/         # MiniMax H3 文生/图生视频（Comfy headless · PRO 6000）
005-pixal3d/            # TencentARC Pixal3D 单图 → GLB（官版 HF demo · 默认 H100）
005-v2-pixal3d-l40s/    # Pixal3D L40S（sm_89 源码轮子 · 已出 GLB）
005-v3-pixal3d-pro6000/ # Pixal3D PRO 6000（sm_120 方案预研 · 未跑）
006-hunyuanworld-mirror/# Tencent HunyuanWorld-Mirror 3D 重建（默认 L4 最低成本）
007-worldmirror-2.0/    # WorldMirror 2.0 recon（HY-World 2.0 · 默认 T4）
008-hy-pano/            # HY-Pano 2.0 全景 · 默认 Qwen@PRO-6000（最省）
009-hy-worldgen/        # HY-World 2.0 worldgen：全景→3D 世界（分 stage）
010-ace-step-1.5/       # ACE-Step 1.5 音乐生成（默认 L4 · turbo DiT）
011-stable-audio-3/     # Stable Audio 3 Medium（默认 L4 · FlashAttn）
015-xiaomi-robotics-1-robocasa365/  # Xiaomi-Robotics-1 VLA · RoboCasa365 冒烟（默认 A100-40GB）
012-levo-2/             # [PLAN] LeVo 2 全曲（听感 S · 先核 license）
013-yue/                # [PLAN] YuE 全曲（结构 · A100）
014-diffrhythm-2/       # [PLAN] DiffRhythm 2 全曲 diffusion（默认 L4）
016-musicgen/           # [PLAN] MusicGen 器乐基线（T4/L4）· 跳过 015
017-xr1-robocasa365-sim/ # Xiaomi XR-1 · RoboCasa365 仿真 smoke + mp4（默认 A100）
```

命名约定：`NNN-topic`（序号 + 主题）。`python main.py 001 …` 等短号在唯一时可解析到对应目录。

## 环境

```bash
pip install modal
modal token new   # 或确保已配置 ~/.modal.toml
export HF_TOKEN=...   # 可选：门禁模型
```

## 用法

```bash
python main.py 001 status
python main.py 004 t2v --prompt "..."
python main.py 005 i2v --image 005-pixal3d/inputs/sample.webp --gpu H100
python main.py 005 build-natten --gpu A100-40GB   # A100 首次
cd 005-v2-pixal3d-l40s && python run.py smoke --i-know-this-costs-money  # L40S
python main.py 006 smoke          # HunyuanWorld-Mirror 最低成本冒烟
python main.py 007 smoke          # WorldMirror 2.0 recon（T4）
python main.py 008 smoke          # HY-Pano 轻量 Qwen（PRO 6000）
python main.py 009 status         # worldgen 全景→3D 世界（分 stage）
python main.py 010 smoke          # ACE-Step 1.5 音乐（L4 · 20s 器乐）
python main.py 011 smoke          # Stable Audio 3 Medium（L4 · 20s）
python main.py 015 smoke          # Xiaomi-Robotics-1 RoboCasa365 动作冒烟（A100-40GB）
python main.py 017 smoke-random   # RoboCasa365 仿真随机 1 局 → mp4
python main.py 017 smoke-policy   # XR-1 闭环短 horizon → mp4

# 也可直接进目录
cd 001-longcat-video && python run.py status
cd 005-pixal3d && python run.py smoke
cd 005-v2-pixal3d-l40s && python run.py status
cd 008-hy-pano && python run.py status
```

## 实验一览

| 目录 | 作用 |
|------|------|
| `001-longcat-video` | 复现 [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)（T2V / I2V / 续写等，跑在 Modal GPU） |
| `002-unlimited-ocr` | 用 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) 在 Modal 上逐页解析 PDF |
| `003-mineru` | 用 [MinerU](https://github.com/opendatalab/MinerU) 在 Modal 上解析 PDF，并与 002 对照 |
| `004-minimax-h3` | [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 量化包 + ComfyUI headless，单卡 PRO 6000 文生视频 |
| `005-pixal3d` | [Pixal3D](https://github.com/TencentARC/Pixal3D) 图生 3D → GLB；**官版 · 默认 H100** |
| `005-v2-pixal3d-l40s` | 同上模型，**L40S sm_89 源码栈 · 已验证出 GLB**（~$0.17 smoke） |
| `005-v3-pixal3d-pro6000` | 同上模型，**PRO 6000 sm_120 方案预研**（未跑；见 SOLUTION） |
| `006-hunyuanworld-mirror` | [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) 多视图 3D；**默认 L4 最低成本** |
| `007-worldmirror-2.0` | [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) **WorldMirror 2.0** recon；**默认 T4** |
| `008-hy-pano` | **HY-Pano 2.0 全景**；默认 **Qwen+LoRA @ RTX-PRO-6000**（~$0.11） |
| `009-hy-worldgen` | **World Generation** 全景→轨迹→WorldStereo→3DGS；见 [PLAN](009-hy-worldgen/PLAN.md) |
| `010-ace-step-1.5` | [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 开源音乐生成；**默认 L4**；主包 turbo + 可选 1.7B LM |
| `011-stable-audio-3` | [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium)；**默认 L4**（T4 无 FlashAttn） |
| `015-xiaomi-robotics-1-robocasa365` | [Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) RoboCasa365 VLA 冒烟；**默认 A100-40GB** |
| `017-xr1-robocasa365-sim` | 同上 + **RoboCasa365 仿真 1 局** → **mp4**；默认 A100-40GB |
| `012-levo-2` | **[PLAN](012-levo-2/PLAN.md)** LeVo 2 全曲音乐（待实现） |
| `013-yue` | **[PLAN](013-yue/PLAN.md)** YuE 全曲（待实现） |
| `014-diffrhythm-2` | **[PLAN](014-diffrhythm-2/PLAN.md)** DiffRhythm 2（待实现） |
| `016-musicgen` | **[PLAN](016-musicgen/PLAN.md)** MusicGen 基线（待实现 · **跳过 015**） |
| — | 音乐队列总览见 [MUSIC_ROADMAP.md](MUSIC_ROADMAP.md) |

H100 / PRO 6000 / A100 对照（OCR 等）见 [GPU_COMPARISON.md](GPU_COMPARISON.md)。  
Pixal3D 官版： [005-pixal3d/GPU_BENCHMARK.md](005-pixal3d/GPU_BENCHMARK.md)  
Pixal3D L40S： [005-v2-pixal3d-l40s/GPU_BENCHMARK.md](005-v2-pixal3d-l40s/GPU_BENCHMARK.md)  
Pixal3D PRO 6000 方案： [005-v3-pixal3d-pro6000/SOLUTION.md](005-v3-pixal3d-pro6000/SOLUTION.md)  
HY-Pano 设备与成本规划见 [008-hy-pano/PLAN.md](008-hy-pano/PLAN.md)。

## 约定

- 每个实验：`run.py` + `README.md`；权重 / 输出走 Modal Volume，不入库
- GPU：`001` / `004` 默认 **RTX-PRO-6000**；`005` 默认 **H100**；`005-v2` 默认 **L40S**；`006` 默认 **L4**；`007` 默认 **T4**；`008` 默认 **RTX-PRO-6000**（Qwen）；`009` 分 stage；`010` 默认 **L4**；`011` 默认 **L4**；`015` 默认 **A100-40GB**
- 上游代码可 vendoring，或镜像 build 时 clone（005 / 005-v2 / 008 采用后者）

远程：https://github.com/xiaoqianran/modal-lab
