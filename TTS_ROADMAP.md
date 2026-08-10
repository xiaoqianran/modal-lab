# 025+ · TTS 实验线 — **按真实使用排行**分档（2026-08-11）

> 数据口径（同一天拉取，可复现）：  
> - **HF downloads / likes**：`huggingface.co/api/models`（`pipeline_tag=text-to-speech` + 定点 model card）  
> - **GitHub stars**：`gh api repos/...`  
> - **质量 Elo**：[Artificial Analysis · Open Weights](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights)  
> - 不是「我觉得好」排序；**先看用量，再谈该不该进 lab**。

---

## 0. 三榜总览（真正在用的人）

### A. Hugging Face 下载量（部署/调用次数代理）

| 榜 | 模型 | 月下载≈ | likes | 备注 |
|----|------|---------|-------|------|
| **1** | **hexgrad/Kokoro-82M** | **11.5M** | 6666 | 绝对用量王；小模型被全网集成 |
| **2** | **coqui/XTTS-v2** | **8.2M** | 3725 | 经典克隆；非商用许可 CPML |
| **3** | **Qwen3-TTS-12Hz-1.7B-Base** | **2.6M** | 479 | 克隆入口；家族合计更高 |
| **4** | **ResembleAI/chatterbox** | **2.1M** | 1733 | 英文克隆 / Turbo 生态 |
| **5** | **Qwen3-TTS-12Hz-1.7B-CustomVoice** | **2.1M** | 1869 | 预设音色 + instruct |
| **6** | Qwen3-TTS-12Hz-0.6B-CustomVoice | 1.4M | 174 | 轻量档 |
| **7** | k2-fsa/OmniVoice | 769k | 1251 | 多语新锐 |
| **8** | SWivid/F5-TTS | 740k | 1191 | 零样本扩散克隆 |
| **9** | openbmb/VoxCPM2 | 643k | 1526 | 速度/延迟向 |
| **10** | microsoft/VibeVoice-Realtime-0.5B | 594k | 1267 | 实时长对话 |
| **11** | Qwen3-TTS VoiceDesign 1.7B | 462k | 386 | 文本造声 |
| **12** | **fishaudio/s2-pro** | **428k** | 1222 | 质量榜常年 Top1 开源权重 |
| **13** | bosonai/higgs-tts-2-3b-base | 406k | 694 | 表情/多说话人 |
| **14** | bosonai/higgs-tts-3-4b | 301k | 708 | Higgs 下一代 |
| **15** | sesame/csm-1b | 180k | 2424 | 对话式（likes 很高） |
| … | microsoft/VibeVoice-1.5B | 85k | **2454** | 下载一般 likes 极高 |
| … | nari-labs/Dia-1.6B | 23k | **2904** | 对白/非语言音；likes 爆 |
| … | stepfun-ai/Step-Audio-EditX | 21k | 137 | **Elo 开源 #2**，下载尚低 |
| … | Fun-CosyVoice3-0.5B | 19k HF | 612 | **GitHub 2.2万星**；权重多走 ModelScope |
| … | IndexTeam/IndexTTS-2 | 14k HF | 775 | **GitHub 2.3万星**；中文圈硬通货 |

> **Qwen3-TTS 家族合并下载（CustomVoice 1.7/0.6 + Base 1.7 + Design）≈ 6.6M+**，仅次于 Kokoro / XTTS 的单卡模型量级。

### B. GitHub stars（开发者关注）

| 榜 | 仓库 | stars | 对应模型 |
|----|------|------:|----------|
| 1 | microsoft/VibeVoice | **52.3k** | VibeVoice 系列 |
| 2 | coqui-ai/TTS | **45.9k** | XTTS 等全家桶（历史底座） |
| 3 | OpenBMB/VoxCPM | **35.2k** | VoxCPM2 |
| 4 | **fishaudio/fish-speech** | **32.1k** | S2 Pro / S1 |
| 5 | **resemble-ai/chatterbox** | **25.9k** | Chatterbox |
| 6 | **QwenAudio/CosyVoice** | **22.7k** | CosyVoice 2/3 |
| 7 | **index-tts/index-tts** | **22.5k** | IndexTTS-2 |
| 8 | nari-labs/dia | 19.4k | Dia |
| 9 | SWivid/F5-TTS | 15.1k | F5-TTS |
| 10 | SesameAILabs/csm | 14.7k | CSM-1B |
| 11 | **QwenLM/Qwen3-TTS** | **12.9k** | Qwen3-TTS |
| 12 | k2-fsa/OmniVoice | 8.9k | OmniVoice |
| 13 | hexgrad/kokoro | 8.4k | Kokoro（+ FastAPI 5.3k） |
| 14 | boson-ai/higgs-audio | 8.3k | Higgs |
| 15 | zai-org/GLM-TTS | 1.0k | GLM-TTS |

### C. Artificial Analysis · Open Weights Elo（盲听偏好 · 质量）

来源：AA open-weights 榜（2026-08 抓取）

| Elo 榜 | 模型 | Elo | 与下载榜关系 |
|--------|------|-----|--------------|
| **1** | **Fish Audio S2 Pro** | **1121** | 下载中上 · stars 极高 |
| **2** | **Step Audio EditX** | **1109** | 下载很低 · **质量黑马** |
| **3** | Voxtral TTS (Mistral) | 1067 | 权重/API 边界需核 |
| **4** | Magpie-Multilingual 357M (NVIDIA) | 1065 | 下载尚低 |
| **5** | **Kokoro 82M** | **1056** | **下载 #1** · 小模型打进质量前五 |
| **6** | Maya1 | 1042 | — |
| **7** | Higgs Audio V3 TTS | 1036 | 下载中上 |
| **8** | **Chatterbox** | **1014** | 下载/stars 双高 |
| **9–10** | Magpie / Zonos | ~1000 | — |
| 11 | VibeVoice 7B | 957 | stars  explodes |
| 13 | XTTS v2 | 914 | 下载 #2 但 Elo 已落后 |

**注意：**  
- 闭源总榜常年是 **Qwen-Audio-3.0-TTS-Plus / Simba / Gemini TTS** 等（Elo ~1200+），**不是**开源 Qwen3-TTS 权重。  
- **Qwen3-TTS / CosyVoice3 / IndexTTS-2** 在 AA open-weights 表上经常**缺席或样本不足** → 不能说它们质量差，只能说 **AA 用量榜覆盖不全**（中文场景更明显）。

---

## 1. 综合「真正在用」分档（给 lab 编号用）

把 **下载 + stars + Elo** 合成四档。权重直觉：

```text
Tier S  用量霸主 或 质量开源榜 Top2（人人都在碰）
Tier A  下载≥500k 或 stars≥20k 或 Elo Top8 开源
Tier B  明确场景冠军（方言/时长/对白/实时）· 中文圈刚需
Tier C  遗产 / 小众 / 未进主流调用
```

### Tier S — 必须先覆盖（真实流量中心）

| 优先级 | 模型 | 证据 | Modal 建议号 | 默认 GPU |
|--------|------|------|--------------|----------|
| **S1** | **Kokoro-82M** | HF **11.5M dl** · Elo **#5** 开源 · Apache | **`025-kokoro`** | **T4** 或 CPU |
| **S2** | **Chatterbox**（Turbo → Multilingual） | HF **2.1M** · GH **26k** · Elo **#8** · Modal **官方例** · MIT | **`026-chatterbox`** | L4 / A10 |
| **S3** | **Qwen3-TTS 家族** | 单卡合计 **~6.6M+** · GH 13k · Apache · 中英+克隆+造声 | **`027-qwen3-tts`** | L4 |
| **S4** | **Fish Audio S2 Pro** | Elo **开源 #1 (1121)** · GH **32k** · Research 许可 | **`028-fish-s2`** | L40S（12–24GB） |

### Tier A — 高热度第二梯队

| 优先级 | 模型 | 证据 | 建议号 | 备注 |
|--------|------|------|--------|------|
| A1 | **VoxCPM2** | GH **35k** · HF 643k | `029-voxcpm2` | 延迟/速度向 |
| A2 | **VibeVoice** | GH **52k** 全场最高 · Realtime 0.5B 594k dl | `030-vibevoice` | 长对话/播客 |
| A3 | **CosyVoice 3** | GH **22.7k** · 中文方言 SOTA · HF 下载偏低（ModelScope） | `031-cosyvoice3` | **中文刚需** |
| A4 | **IndexTTS-2** | GH **22.5k** · 时长+情感控制 | `032-indextts2` | **配音刚需** |
| A5 | **F5-TTS** | GH 15k · HF 740k · 零样本 | `033-f5tts` | 轻量克隆对照 |
| A6 | **Higgs Audio** | Elo #7 · HF ~400k · GH 8k | `034-higgs` | 表情/多说话人 |
| A7 | **XTTS-v2** | HF **8.2M** · Elo 已掉 · **CPML 非商用** | `035-xtts-v2` | 遗产基线，可选 |

### Tier B — 场景冠军 / 质量黑马

| 模型 | 为何在榜 | 建议 |
|------|----------|------|
| **Step Audio EditX** | Elo 开源 **#2 (1109)**，下载仅 21k | 质量对照，不必最先做 |
| **OmniVoice** | HF 769k · GH 9k | 多语 |
| **Dia-1.6B** | likes 2904 · 对白/笑声咳嗽 | 剧本对白线 |
| **Sesame CSM-1B** | likes 2424 · 对话 | 对话 agent |
| **GLM-TTS** | seed-tts CER 强 · stars 1k | 中英准确率对照 |
| **Magpie-Multilingual** | Elo #4 | NVIDIA 多语 |
| **Orpheus** | 多尺寸 | 分级参数量对照 |

### Tier C — 暂缓

Mozilla/Piper（边缘 CPU）、Bark（旧）、StyleTTS2（Elo 低）、MetaVoice、各种小众微调包。  
除非用户指定，**不进 025 起编号**。

---

## 2. 和「闭源总榜」的关系（别混）

| 层级 | 代表 | 能否进 modal-lab |
|------|------|------------------|
| 闭源总榜 Top | Qwen-Audio-3.0-TTS-**Plus**、Simba、Gemini TTS、ElevenLabs v3、Inworld Realtime | ❌ 仅 API；可作听感 ceiling |
| 开源权重榜 Top | **Fish S2 · Step EditX · Voxtral · Magpie · Kokoro · Chatterbox…** | ✅ |
| 中文开源实务 | **Qwen3-TTS · CosyVoice3 · IndexTTS-2** | ✅（AA 覆盖不全） |

lab 只做 **可自托管权重** 行。

---

## 3. 推荐落地顺序（按「真实使用」而不是个人口味）

与音乐线 `010→016` 同理：**先用量最大的便宜基线，再质量/克隆旗舰，再中文/场景**。

```text
025-kokoro          Tier S1  用量#1 · 最便宜 · smoke 基线（对标 016 MusicGen）
026-chatterbox      Tier S2  用量+stars+Elo+Modal官方例 · 英文克隆主力
027-qwen3-tts       Tier S3  家族下载合计#3 · 中英+三模式 · 中文主线
028-fish-s2         Tier S4  Elo开源#1 · 质量上限（脚注 Research 许可）
── 第二波 ──
029-voxcpm2         速度
030-vibevoice       长音频/对话（stars#1）
031-cosyvoice3      中文方言
032-indextts2       精确时长配音
```

### 025 第一个该是谁？

| 若你的目标… | 第一个号 |
|-------------|----------|
| **对齐「真正使用排行」从头部开始** | **`025-kokoro`**（HF 下载断层第一） |
| 英文克隆 + 最快 Modal 范例 | `026-chatterbox` |
| 中文内容生产 | `027-qwen3-tts` 或 `031-cosyvoice3` |
| 只追盲听质量 | `028-fish-s2` |

**按「真正使用排行」严格执行 → 025 = Kokoro，然后 Chatterbox，然后 Qwen3-TTS。**

上一版把 Qwen3-TTS 直接定为 025，是 **lab 适配偏好**（许可+中文+VRAM），**不是下载榜第一**。现已纠正为分档表。

---

## 4. 单模型速查（实现时用）

### 025-kokoro（建议先做）

| 项 | 值 |
|----|-----|
| 权重 | `hexgrad/Kokoro-82M`（+ 语音包） |
| 规模 | **82M** · ~2–3GB · **可 CPU** |
| 许可 | Apache-2.0 |
| 克隆 | ❌ 预设 ~50+ 音色 |
| 默认 GPU | **T4**（或 CPU-only 更便宜） |
| smoke | 中英短句 · 2–3 个预设 speaker |
| 估费 | 墙钟若 15–30s → T4 **≪ $0.01** |
| 代码 | `kokoro` / `misaki` 或 Kokoro-FastAPI 推理路径 |

### 026-chatterbox

| 项 | 值 |
|----|-----|
| 权重 | ResembleAI/chatterbox · Turbo 350M / MTL 0.5B |
| 许可 | **MIT** |
| 克隆 | ✅ 7–10s ref |
| 语言 | Turbo 英；MTL **23 语含 zh** |
| GPU | L4 或 A10（官方例 A10G） |
| 参考 | https://modal.com/docs/examples/chatterbox_tts |

### 027-qwen3-tts

| 项 | 值 |
|----|-----|
| 权重 | `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{CustomVoice,Base,VoiceDesign}` |
| 许可 | Apache-2.0 |
| 能力 | 预设+instruct / 文本造声 / **3s 克隆** / 流式 ~97ms |
| 语言 | 10 语（中英核心） |
| GPU | **L4**；0.6B 可试 T4 |
| 安装 | `pip install -U qwen-tts` |

### 028-fish-s2

| 项 | 值 |
|----|-----|
| 权重 | `fishaudio/s2-pro` |
| 许可 | **Fish Research（商用要授权）** |
| Elo | 开源 **#1** |
| VRAM | **12–24GB** → 默认 **L40S** |
| 特点 | 80+ 语 · 多说话人 · 行内情绪标记 |

### 031-cosyvoice3 / 032-indextts2（中文场景必补）

| | CosyVoice3 | IndexTTS-2 |
|--|------------|------------|
| 王牌 | **18+ 方言** · instruct | **时长毫秒控制** · 情感解耦 |
| 规模 | 0.5B | 工业 AR ~9GB fp16 |
| 坑 | 依赖重；torch≥2.7 有报告 | HF 下载少但中文 GH 热 |

---

## 5. 统一实验脚手架（每号共用）

```text
NNN-<model>/
  PLAN.md | README.md | UPSTREAM.md
  modal_app.py      # App · Image · Volume · download/smoke/t2s
  run.py
  examples/  inputs/voices/  gallery/
  COST_BENCHMARK.md
```

| Volume | 模式 |
|--------|------|
| `modal-lab-<name>-weights` | CPU download |
| `modal-lab-<name>-outputs` | `runs/<name>/{audio.wav,meta.json}` |

CLI 最小集：`status | download | smoke | t2s | pull | ls`  
（有克隆的号加 `clone`；Qwen 加 `design`。）

GPU 单价（仓库既有）：T4 0.000164 · L4 0.000222 · L40S 0.000542 $/s。

---

## 6. 决策（已拍板）

1. **按真实用量编号：025=Kokoro → 026=Chatterbox → 027=Qwen3-TTS → 028=Fish S2** ✅  
2. 一次 **1 个** smoke 通再开下一个（对齐音乐线）。

**进度：** 025 ✅ · 026 ✅ · 027 ✅ · 028 ✅ · **TTS Tier S 收官。** 下一波 029+。


---

## 7. 数据快照附录（2026-08-11）

### HF 定点 downloads / likes

```
Kokoro-82M                         11,500,170 / 6666
XTTS-v2                             8,222,217 / 3725
Qwen3-TTS-1.7B-Base                 2,626,232 /  479
chatterbox                          2,127,987 / 1733
Qwen3-TTS-1.7B-CustomVoice          2,072,253 / 1869
Qwen3-TTS-0.6B-CustomVoice          1,435,679 /  174
OmniVoice                             769,095 / 1251
F5-TTS                                740,462 / 1191
VoxCPM2                               642,761 / 1526
VibeVoice-Realtime-0.5B               593,675 / 1267
Qwen3-TTS-1.7B-VoiceDesign            462,066 /  386
fishaudio/s2-pro                      428,160 / 1222
higgs-tts-2-3b-base                   405,796 /  694
higgs-tts-3-4b                        301,420 /  708
sesame/csm-1b                         180,286 / 2424
VibeVoice-1.5B                         85,223 / 2454
Dia-1.6B                               22,921 / 2904
Step-Audio-EditX                       20,985 /  137
Fun-CosyVoice3-0.5B-2512               19,442 /  612
IndexTTS-2                             14,052 /  775
```

### GitHub stars

```
VibeVoice 52.3k · coqui-TTS 45.9k · VoxCPM 35.2k · fish-speech 32.1k
chatterbox 25.9k · CosyVoice 22.7k · index-tts 22.5k · Dia 19.4k
F5-TTS 15.1k · CSM 14.7k · Qwen3-TTS 12.9k · OmniVoice 8.9k
kokoro 8.4k · higgs-audio 8.3k · GLM-TTS 1.0k
```

### AA Open Weights Elo

```
Fish S2 Pro 1121 · Step EditX 1109 · Voxtral 1067 · Magpie 1065
Kokoro 1056 · Maya1 1042 · Higgs V3 1036 · Chatterbox 1014
```

---

## 8. 目录状态

| 号 | 目录 | 状态 |
|----|------|------|
| 025 | `025-kokoro/` | ✅ smoke EN/ZH · gallery · COST |
| 026 | `026-chatterbox/` | ✅ mtl_en/zh + turbo · gallery · COST |
| 027 | `027-qwen3-tts/` | ✅ custom/design/clone · gallery · COST |
| 028 | `028-fish-s2/` | ✅ en/zh/tags/clone · gallery · COST · **Tier S 收官** |

