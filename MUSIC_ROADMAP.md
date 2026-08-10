# 音乐实验路线图（010–016）

> 状态：2026-08-10 pull 后规划。  
> **015 已被** `015-xiaomi-robotics-1-robocasa365` **占用 → MusicGen 跳到 016。**

## 已完成

| 号 | 目录 | 模型 | 默认 GPU | 备注 |
|----|------|------|----------|------|
| **010** | `010-ace-step-1.5` | ACE-Step 1.5 turbo | L4 | 全曲 / MIT · 已有 gallery |
| **011** | `011-stable-audio-3` | Stable Audio 3 Medium | L4 | 器乐可控 · 门禁 HF · 已有 gallery |

## 待实现（按用户指定顺序）

| 顺序 | 号 | 目录（规划） | 模型 | 赛道 | 建议默认 GPU | 粗估成本/难度 |
|------|----|--------------|------|------|--------------|----------------|
| **1** | **012** | `012-levo-2` | **LeVo 2** (SongGeneration) | 全曲 + 歌词 · 听感 S 档 | A100-40GB 或 L40S | 高 · **先核 license** |
| **2** | **013** | `013-yue` | **YuE** (M-A-P) | 全曲 + 歌词 · 结构强 | A100-40GB / A100-80GB | 高 · 慢 · 权重大 |
| **3** | **014** | `014-diffrhythm-2` | **DiffRhythm 2** | 全曲 diffusion · 极速 | **L4**（或 T4 试） | 中 · 性价比高 |
| **4** | **016** | `016-musicgen` | **MusicGen** (Meta AudioCraft) | 短/中器乐 · 经典基线 | **T4 / L4** | 低 · 最稳 |

> **不要占用 015**。机器人实验保留。

## 实现约定（对齐 010 / 011）

每个目录统一：

```text
NNN-topic/
  modal_app.py      # Modal image + volume + download / smoke / t2a
  run.py            # 本地 CLI → modal run
  README.md
  UPSTREAM.md       # 上游 repo + commit pin
  examples/         # 默认 prompt / lyrics
  gallery/          # index.html + assets/*.flac（成功后）
  outputs/          # 本地拉回的结果（gitignore）
```

- Volume 命名：`modal-lab-<topic>-weights` / `modal-lab-<topic>-outputs`
- smoke：固定 seed、~15–30s、估费写入 `meta.json`
- 成功后：`gallery/index.html` 试听 + 根 `README.md` 挂链
- 调度：`python main.py 012 smoke` 等（目录有 `run.py` 即自动发现）

## 建议落地顺序与理由

```text
012 LeVo 2        ← 用户点名第一 · 全曲听感对照 010
013 YuE           ← 重型全曲上限 / 歌词结构
014 DiffRhythm 2  ← 快速全曲 · 与 010/012 对照速度
016 MusicGen      ← 经典基线 · 器乐对照 011 · 实现最简单收尾
```

### 012 LeVo 2 · 注意

- 上游：Tencent SongGeneration / LeVo 2（实现前再 pin 确切 GitHub + HF）
- **许可可能非商业** → README 必须写清；冒烟可做，商用结论另说
- 多阶段 / 较大权重 → 默认不要用 T4；优先 **A100-40GB**，成本敏感可试 L40S
- smoke 目标：短歌词 + 风格 caption，输出 FLAC

### 013 YuE · 注意

- 上游：`multimodal-art-projection/YuE` 系（双轨 stage1/stage2 常见）
- 双模型串联 → 冷启动与 VRAM 高；**A100-40GB 起**，完整长曲考虑 80GB
- smoke：极短片段或官方 mini 配置，避免一次烧满时长

### 014 DiffRhythm 2 · 注意

- 上游：`ASLP-lab/DiffRhythm`（v2 权重）
- 设计目标：**L4 默认**，对齐 010/011 的优惠线
- 全曲 diffusion，适合和 ACE/LeVo 做速度–质量表

### 016 MusicGen · 注意

- 上游：`facebook/audiocraft` + HF `facebook/musicgen-small|medium`
- **small @ T4** 即可做最便宜基线；medium @ L4
- 不做「带歌词全曲」，定位 **text→instrumental 基线**

## 与 015 的编号关系

```text
010 music · ACE
011 music · SA3
012 music · LeVo 2      ← next
013 music · YuE
014 music · DiffRhythm 2
015 robotics · Xiaomi   ← 已存在，跳过
016 music · MusicGen
```

## 状态板

| 号 | 状态 |
|----|------|
| 010 | ✅ done |
| 011 | ✅ done |
| 012 | 📋 planned（本文件 + `012-levo-2/PLAN.md`） |
| 013 | 📋 planned |
| 014 | 📋 planned |
| 016 | 📋 planned |

实现时从 **012** 开干，完成一个再进下一个；不要并行占多卡除非用户要求。
