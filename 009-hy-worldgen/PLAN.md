# 009 · World Generation 规划（**单卡 PRO 6000 优先**）

> **实现状态（v8）**  
> - ✅ Stage1 + Stage2：**官方 Qwen3-VL-8B（vLLM）** + `stage12` 一键  
> - ✅ Stage3–5：WorldStereo-dmd / GS（既有）  
> - 默认串行同卡 `vlm_mode=share`；多卡可 `split`  
> - Stage3 前仍需预算确认（见下）

---

## 0. 必须多卡吗？

| 说法 | 真相 |
|---|---|
| 官方 README「≥4 GPU recommended / 例 8×H20」 | **推荐配置 / 加速**，不是硬编码「少于 4 卡拒绝运行」 |
| Stage1 `traj_generate.py` | 文档写明 **single GPU**；外挂 vLLM 可另机/另卡 |
| Stage2 `traj_render.py` | `WORLD_SIZE` 默认 **1**；多卡只是分轨迹加速 |
| Stage3 `video_gen.py` | `WORLD_SIZE` 默认 **1**；`--fsdp` **可选**（多卡切分） |
| Stage4 `gen_gs_data.py` | 默认 **1** 进程即可 |
| Stage5 `world_gs_trainer` | 官方写了 **x1 GPU: max_steps 8000** |

**→ 一张 RTX PRO 6000（~96GB）理论上可以跑通整条链路。**  
多卡主要省时间；单卡用 **更少轨迹 + DMD 4-step + 串行占卡** 换可负担成本。

---

## 1. 单卡难点（不是“不能”，是“要挤”）

### 1.1 VLM 和视觉模型抢同一张卡

Stage1/2 设计是：

- **本卡**：MoGe 深度 + SAM3 + ZIM/GD 等  
- **另起 vLLM**：Qwen3-VL-8B 做规划/caption  

只有 1×PRO 6000 时：

| 策略 | 做法 | 评价 |
|---|---|---|
| **A. 串行（推荐）** | 先起小 VLM → 写完轨迹/caption → **杀掉 vLLM 腾显存** → 再跑渲染/扩散 | 最稳 |
| B. 同卡硬塞 | 8B VLM + SAM3 + MoGe 一起 | 96GB 可能顶，风险 OOM |
| C. VLM 改 CPU / 更小 VL | 3B/慢 | 省显存、费时间 |
| D. 跳过部分 VLM | 少 `force_vlm`、少 wonder/recon 轨迹 | 最便宜，质量降 |

### 1.2 Stage3 WorldStereo 权重本身就大

HF `hanshanxue/WorldStereo` 约 **68GB 盘**：

| 变体 | 权重文件约 |
|---|---:|
| **worldstereo-memory-dmd**（默认，4-step） | **~35GB** |
| worldstereo-memory（全步） | ~22GB（另有 camera 等） |
| worldstereo-camera | ~11GB |

单卡 96GB：权重 ~35GB bf16 + MoGe + SAM3-Video + 激活，**紧但有希望**；不行再开 **CPU offload / 降分辨率 / 少 reference**。  
官方多卡 FSDP 是为了更快更稳，不是唯一路径。

### 1.3 Stage5 单卡会更久

官方：8 卡 `max_steps=1500` → **1 卡 `max_steps=8000`**（steps 按卡数反比拉长）。  
时间变长，**单价不变，总价上升**。

---

## 2. 单卡 PRO 6000 最小可跑配置（smoke）

目标：**证明能出 ply/spz**，不是官方 demo 画质。

| 旋钮 | 官方味 | **单卡省钱 smoke** |
|---|---|---|
| 轨迹数量 | wonder_topk=3, recon_topk=5 + 多 view | **view≤2，wonder=1，recon=0～1** |
| nframe | 21 | **12～16**（若接口允许） |
| WorldStereo | memory-dmd | **memory-dmd**（必选） |
| FSDP | 8 卡 | **关**（单进程） |
| max_reference | 8 | **2～4** |
| Stage5 steps | 1500×8 卡等价 | **4000～8000**（先 4000 试） |
| 场景 | 多 case | **仅 scene_from_008 一张全景** |
| VLM | Qwen3-VL-8B ×8 卡 | **单卡串行 / 或更小 VL** |

---

## 3. 花费粗估（**仅 1× RTX-PRO-6000**）

单价按 Modal 约 **$0.000842 / s ≈ $3.03 / h**（与 008 实测同表）。

| 阶段 | 在干什么 | 单卡耗时粗估（smoke） | **GPU $ 粗估** |
|---|---|---:|---:|
| 0 prepare | 拷 008 全景 | CPU ≈0 | **$0** |
| 下权重 | WorldStereo ~35–68GB + SAM/MoGe 等 | CPU | **$0**（时间/存储） |
| **1 轨迹** | 深度/分割 + VLM 规划 | 15–40 min | **$0.8–2.0** |
| **2 渲染** | 少轨迹点云视频 + caption | 10–30 min | **$0.5–1.5** |
| **3 扩帧** | WorldStereo-dmd × 少轨迹 | **40–120 min** | **$2–6** ← 大头 |
| **4 GS 数据** | 抽帧/深度/法线 | 15–40 min | **$0.8–2.0** |
| **5 3DGS** | 单卡 4k–8k steps | 40–100 min | **$2–5** |
| **合计（顺利一枪）** | | **约 2–5.5 h** | **约 $6–18** |
| 含失败重试 / 调参 | | ×1.5–3 | **$10–40** 更现实缓冲 |

### 对照

| 配置 | 粗估 |
|---|---|
| **1× PRO 6000 · 最小 smoke** | **~$6–18 成功；预算预留 $20–30 更安心** |
| 1× H100 · 同规模 | 更快，总价常接近或略高（单价贵） |
| 官方味 4–8 卡 · 多轨迹 | **$50–150+** 很容易 |

**008 全景 ~$0.11**；009 完整世界是 **两位数美元级**，不是「再花一毛钱」。

---

## 4. 单卡执行顺序（规划，未默认开火）

```text
[已完成] 008 panorama → prepare scene_from_008
[已完成] Phase A/B/C 代码：download --which vlm + stage12（官方 Qwen3-VL-8B）

Phase A  下载权重（CPU）· `download --which vlm`
Phase B  Stage1  单卡 · vLLM 同卡 · 极少轨迹      预算点 ~$1–3（含 VLM）
Phase C  Stage2  单进程渲染 + VLM caption         预算点 ~$0.5–2
Phase D  Stage3  单进程 DMD · 无 FSDP             预算点 ~$6  ⚠️ 最贵确认点
Phase E  Stage4  单进程                           预算点 ~$2
Phase F  Stage5  max_steps=4000 先试              预算点 ~$3
Phase G  拉 ply/spz + HTML 预览
```

**门禁：每个 Phase 结束看 meta / 产物；Stage3 前必须你口头确认。**

---

## 5. 我认为「只有一张 PRO 6000」可不可行？

| | |
|---|---|
| **可行吗** | **可以尝试**，代码路径支持单进程；96GB 对 DMD ~35GB 权重是目前 Modal 单卡里较合适的一档 |
| **风险** | Stage3 OOM / VLM 同卡冲突 / Stage5 时间长导致账单超预期 |
| **不建议** | 一上来按官方 8 卡命令抄；多轨迹 + 全步 WorldStereo |
| **建议默认** | 单卡 PRO 6000 + 极少轨迹 + DMD + 分 stage 确认 |

---

## 6. 当前可执行命令

```bash
python main.py 009 download --which vlm
python main.py 009 stage12 --gpu RTX-PRO-6000 --nframe 16
# Stage3 前确认预算：
# python main.py 009 stage 3 --gpu RTX-PRO-6000
```

**不会在没预算确认前默认开 Stage3。** Stage1+2（`stage12`）已实现官方 Qwen3-VL-8B。
