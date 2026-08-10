# modal-lab

Modal 实验台：按 `NNN-topic` 编号做可复现实验。

## Gallery (GitHub Pages) — 请把 HTML 结果挂上来

线上橱窗：**https://xiaoqianran.github.io/modal-lab/**

| 入口 | URL |
|------|-----|
| Lab 首页（筛选 / 收藏 / 随便逛逛） | https://xiaoqianran.github.io/modal-lab/ |
| 009 HY-World 世界馆 | https://xiaoqianran.github.io/modal-lab/009-hy-worldgen/ |
| Image→3D Lab（L40S×PRO） | https://xiaoqianran.github.io/modal-lab/gpu-gallery/ |
| 022 Hunyuan3D-2.1 | https://xiaoqianran.github.io/modal-lab/gpu-gallery/#hunyuan3d21 |
| **023-a SPAR3D Mesh Studio** | https://xiaoqianran.github.io/modal-lab/023-a-spar3d/ |
| **023-b SF3D Mesh Studio** | https://xiaoqianran.github.io/modal-lab/023-b-sf3d/ |
| 各实验 gallery | `https://xiaoqianran.github.io/modal-lab/<NNN-topic>/` |

### 每位作者请记住：出了 HTML 结果 → 部署到 Pages

不要只把结果丢在 Modal Volume 或本机。**把可展示的 HTML 放进仓库的 `gallery/`，推 `main`，Actions 会自动发到 Pages。**

1. **写 gallery**  
   在 `NNN-topic/gallery/index.html`（+ `assets/` 小体积媒体）。  
   本地预览：`python3 -m http.server 8765 --directory NNN-topic/gallery`

2. **媒体体积**  
   只提交预览级文件（截图、短 mp4、小 ply/glb）。大权重 / 完整 outputs 仍走 Modal Volume，**不要**塞进 git。  
   `.gitignore` 已允许 `gallery/**/*.mp4` 等小媒体。

3. **登记到首页（推荐）**  
   新实验在 `pages/data/experiments.json` 加一条（标题、简介、标签、accent）。  
   不登记也能通过 `/NNN-topic/` 直达，但首页卡片不会出现。

4. **推送即部署**  
   ```bash
   git add NNN-topic/gallery pages/data/experiments.json
   git commit -m "gallery(NNN): …"
   git push origin main
   ```  
   改动 `**/gallery/**`、`pages/**` 或 `gpu-gallery/**` 会触发  
   `.github/workflows/pages.yml` → build `_site` → **GitHub Pages**。  
   也可在 Actions 里手动 **Run workflow**。

5. **自检**  
   - Actions：`Deploy GitHub Pages` 绿勾  
   - 打开 `https://xiaoqianran.github.io/modal-lab/NNN-topic/`  
   - 资源用**相对路径**（`assets/...`），不要写死 `localhost`

> 约定：**有可看结果 = 有 gallery HTML + 已上 Pages。** 写 README 时顺手贴上 Pages 链接。

## 结构

```text
main.py                 # 入口，调度到 001 / 002 / …
gpu-gallery/            # L40S × PRO 6000 Image→3D 对照预览（020/021/005）
I2V_GPU_ROADMAP.md      # image→3D 卡池约定与结论
001-longcat-video/      # 美团 LongCat-Video 视频生成复现
002-unlimited-ocr/      # 百度 Unlimited-OCR 文档解析
003-mineru/             # OpenDataLab MinerU 文档解析
004-minimax-h3/         # MiniMax H3 文生/图生视频（Comfy headless · PRO 6000）
005-pixal3d/            # TencentARC Pixal3D 单图 → GLB（官版 HF demo · 默认 H100）
005-v2-pixal3d-l40s/    # Pixal3D L40S（sm_89 源码轮子 · 已出 GLB）
005-v3-pixal3d-pro6000/ # Pixal3D PRO 6000（sm_120 · torch2.11+cu128 · 已出 GLB）
006-hunyuanworld-mirror/# Tencent HunyuanWorld-Mirror 3D 重建（默认 L4 最低成本）
007-worldmirror-2.0/    # WorldMirror 2.0 recon（HY-World 2.0 · 默认 T4）
008-hy-pano/            # HY-Pano 2.0 全景 · 默认 Qwen@PRO-6000（最省）
009-hy-worldgen/        # HY-World 2.0 worldgen：全景→3D 世界（分 stage）
010-ace-step-1.5/       # ACE-Step 1.5 音乐生成（默认 L4 · turbo DiT）
011-stable-audio-3/     # Stable Audio 3 Medium（默认 L4 · FlashAttn）
012-levo-2/             # LeVo 2 / SongGeneration v2（默认 L40S · v2-medium）
013-yue/                # YuE 歌词→全曲（默认 L40S · en-cot · smoke OK）
014-diffrhythm-2/       # DiffRhythm 2 全曲扩散（默认 L4）
015-xiaomi-robotics-1-robocasa365/  # Xiaomi-Robotics-1 VLA · RoboCasa365 冒烟（默认 A100-40GB）
016-musicgen/           # MusicGen small 器乐基线（默认 T4 · 占 016）
017-xr1-robocasa365-sim/ # XR1 RoboCasa 仿真
020-triposr/            # TripoSR image→mesh 速度基线（L40S+PRO6000 · 已出 GLB）
021-trellis2/           # TRELLIS.2-4B 质量主线 MIT（L40S+PRO6000 · 已出 GLB）
022-hunyuan3d-2.1/      # Hunyuan3D-2.1 PBR（L40S+PRO6000 · smoke full 完成 · Pages）
023-a-spar3d/           # SPAR3D point-aware（L40S+PRO6000 · Gallery 已上 Pages）
023-b-sf3d/             # SF3D fast path（L40S+PRO6000 · Gallery 已上 Pages）
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
cd 005-v3-pixal3d-pro6000 && python run.py smoke --i-know-this-costs-money  # PRO 6000
cd 020-triposr && python run.py smoke --i-know-this-costs-money --gpu L40S
cd 021-trellis2 && python run.py build --gpu L40S && python run.py smoke --i-know-this-costs-money --gpu L40S
python main.py 006 smoke          # HunyuanWorld-Mirror 最低成本冒烟
python main.py 007 smoke          # WorldMirror 2.0 recon（T4）
python main.py 008 smoke          # HY-Pano 轻量 Qwen（PRO 6000）
python main.py 009 status         # worldgen 全景→3D 世界（分 stage）
python main.py 010 smoke          # ACE-Step 1.5 音乐（L4 · 20s 器乐）
python main.py 011 smoke          # Stable Audio 3 Medium（L4 · 20s）
python main.py 014 smoke          # DiffRhythm 2（L4 · 60s）
python main.py 013 smoke          # YuE（L40S · 2 segments）
python main.py 012 smoke          # LeVo 2（L40S · v2-medium）
python main.py 016 smoke          # MusicGen small（T4 · 15s）
python main.py 015 smoke          # Xiaomi-Robotics-1 RoboCasa365 动作冒烟（A100-40GB）

# 也可直接进目录
cd 001-longcat-video && python run.py status
cd 005-pixal3d && python run.py smoke
cd 005-v2-pixal3d-l40s && python run.py status
cd 005-v3-pixal3d-pro6000 && python run.py status
cd 008-hy-pano && python run.py status
cd 020-triposr && python run.py status
cd 021-trellis2 && python run.py status
cd 022-hunyuan3d-2.1 && python run.py probe --gpu L40S
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
| `005-v3-pixal3d-pro6000` | 同上模型，**PRO 6000 sm_120 · torch2.11+cu128 · 已验证**（~$0.19 · ~230s） |
| `006-hunyuanworld-mirror` | [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) 多视图 3D；**默认 L4 最低成本** |
| `007-worldmirror-2.0` | [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) **WorldMirror 2.0** recon；**默认 T4** |
| `008-hy-pano` | **HY-Pano 2.0 全景**；默认 **Qwen+LoRA @ RTX-PRO-6000**（~$0.11） |
| `009-hy-worldgen` | **World Generation** 全景→轨迹→WorldStereo→3DGS；见 [PLAN](009-hy-worldgen/PLAN.md) |
| `010-ace-step-1.5` | [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 开源音乐生成；**默认 L4**；主包 turbo + 可选 1.7B LM |
| `011-stable-audio-3` | [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium)；**默认 L4**（T4 无 FlashAttn） |
| `012-levo-2` | [LeVo 2](https://github.com/levo-demo/LeVo) SongGeneration v2；**默认 L40S · v2-medium**（研究许可） |
| `013-yue` | [YuE](https://github.com/multimodal-art-projection/YuE) 歌词→全曲；**L40S · en-cot · ~$0.43/2seg** |
| `014-diffrhythm-2` | [DiffRhythm 2](https://github.com/ASLP-lab/DiffRhythm2) 全曲扩散；**L4 · 60s ~$0.016** |
| `015-xiaomi-robotics-1-robocasa365` | [Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) RoboCasa365 VLA 冒烟；**默认 A100-40GB** |
| `016-musicgen` | [MusicGen](https://huggingface.co/facebook/musicgen-small) 器乐基线；**默认 T4 · small**（CC-BY-NC） |
| `017-xr1-robocasa365-sim` | XR1 RoboCasa 仿真；**L40S 闭环 horizon=100** |
| `020-triposr` | [TripoSR](https://github.com/VAST-AI-Research/TripoSR) **速度基线**；**L40S ~14s / PRO6000 ~10s · ~$0.01** |
| `021-trellis2` | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) **质量主线 MIT**；**L40S ~215s / PRO6000 ~122s · 已出 GLB** |
| `022-hunyuan3d-2.1` | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) **PBR 纹理**；L40S+PRO6000 · Community License |
| `023-a-spar3d` | [SPAR3D](https://github.com/Stability-AI/stable-point-aware-3d) point-aware 图生网格；**L40S/PRO · [Gallery](https://xiaoqianran.github.io/modal-lab/023-a-spar3d/)** |
| `023-b-sf3d` | [SF3D](https://github.com/Stability-AI/stable-fast-3d) 快速图生网格；**infer PRO ~0.96s · [Gallery](https://xiaoqianran.github.io/modal-lab/023-b-sf3d/)** |
| `gpu-gallery` | **L40S × PRO 6000 对照预览**（020 / 021 / 005） |
| — | 音乐队列总览见 [MUSIC_ROADMAP.md](MUSIC_ROADMAP.md) · image→3D 卡池见 [I2V_GPU_ROADMAP.md](I2V_GPU_ROADMAP.md) |

统一预览（L40S × PRO 6000）：[`gpu-gallery/`](gpu-gallery/) · 路线图 [`I2V_GPU_ROADMAP.md`](I2V_GPU_ROADMAP.md)

Pixal3D / 开源 I2V：  
- [005 GPU_BENCHMARK](005-pixal3d/GPU_BENCHMARK.md)  
- [005-v2 L40S](005-v2-pixal3d-l40s/GPU_BENCHMARK.md)  
- [005-v3 PRO 6000](005-v3-pixal3d-pro6000/GPU_BENCHMARK.md)  
- [020 TripoSR](020-triposr/GPU_BENCHMARK.md)  
- [021 TRELLIS.2](021-trellis2/GPU_BENCHMARK.md)
- [022 Hunyuan3D-2.1](022-hunyuan3d-2.1/README.md)

远程：https://github.com/xiaoqianran/modal-lab
