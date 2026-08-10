# 009 · HY-World 2.0 **World Generation**（全景 → 可导航 3D 世界）

> 上游：[Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · `hyworld2/worldgen`  
> 输入：008 产出的 **panorama.png**（或官方 case）  
> 输出：**3DGS / mesh / spz** 可导入引擎的世界

---

## 1. 全仓库两条线（你已经跑了什么）

| 线 | 模块 | 实验 | 状态 |
|---|---|---|---|
| **World Reconstruction** | WorldMirror 2.0 | **007** | ✅ 多视图 → 点云/深度/相机 |
| **Panorama Generation** | HY-Pano 2.0 | **008** | ✅ 单图 → 360° 全景 |
| **World Generation** | WorldNav + WorldStereo + 3DGS | **009（本实验）** | ⏳ 全景 → 最终 3D 世界 |

官方「生成一个世界」不是一步模型，而是：

```text
图/文
  │
  ▼
[008] HY-Pano          全景 panorama.png
  │
  ▼
[009-1] Trajectory Planning   WorldNav + VLM（Qwen3-VL）
  │
  ▼
[009-2] Trajectory Rendering  沿轨迹渲染点云视频 + caption
  │
  ▼
[009-3] World Expansion       WorldStereo 2.0（~17B）扩关键帧
  │
  ▼
[009-4] GS Data Prep          帧/深度/法线/相机（含 WorldMirror）
  │
  ▼
[009-5] 3DGS Training         gsplat 优化 → ply / spz / mesh
  │
  ▼
        真正「可玩」的 3D 世界
```

官方 README 把中间叫 4 大能力（Pano / WorldNav / WorldStereo / Composition），`worldgen` 代码拆成 **5 个脚本阶段**。

---

## 2. 各阶段脚本与依赖

| Stage | 脚本 | 核心依赖 | 官方 GPU 暗示 | 大概体量 |
|---|---|---|---|---|
| 1 | `traj_generate.py` | **vLLM + Qwen3-VL-8B**、SAM3、navmesh | 单卡 + 另起 VLM 服 | VLM 8B |
| 2 | `traj_render.py` | 多卡 torchrun、VLM caption | 官方例 8 卡 | 渲染为主 |
| 3 | `video_gen.py` | **WorldStereo-2 ~17B**、FSDP | 官方例 8 卡 | 最重扩散段 |
| 4 | `gen_gs_data.py` | WorldMirror 系几何 | 多卡 | 中 |
| 5 | `world_gs_trainer.py` | gsplat_maskgaussian | 1–8 卡（steps 反比） | 训练段 |

官方前提：

- ≥4 GPU recommended（测过 8×H20）
- 单独起 vLLM 给 Stage 1–2
- WorldStereo 权重：`hanshanxue/WorldStereo`（~17B）
- 自定义编译：`third_party/gsplat_maskgaussian`、`third_party/navmesh`

---

## 3. 成本现实（比 008 贵一个数量级）

008 单张全景 ~**$0.11**（PRO 6000）。  
009 官方配置是 **多卡 + 17B 视频扩散 + 3DGS 训练 + VLM 服务**：

| 策略 | 卡 | 粗估（极粗） | 备注 |
|---|---|---|---|
| **最小 smoke**（少轨迹、DMD 4-step、1–2 卡、少 steps） | PRO 6000 / H100×1–2 | **数美元级** | 质量会降，先验证通路 |
| 官方推荐味 | H100×4–8 | **$20–100+** | 别默认 |
| 完整精品场景 | 8× 高端 | **很容易三位数** | 需你明确点头 |

**默认原则：分 stage 跑、每 stage 可停、禁止一键全开 8 卡。**

---

## 4. 省钱执行规划（009）

### Phase 0 — Scaffold ✅（本目录）

- PLAN / README / run CLI 骨架  
- 不默认扣费

### Phase 1 — 数据衔接

- 从 **008** `runs/smoke_qwen/panorama.png` 拷进 scene dir  
- 或用官方 `examples/worldgen/case000`

### Phase 2 — Stage 1 轨迹（相对便宜）

- 起 **小 VLM**：Qwen2.5-VL-3B/7B 或官方 8B 单卡  
- 单卡 **PRO 6000** / L40S  
- 少目标、短轨迹 → 验证 `navmesh/` 产出

### Phase 3 — Stage 2 渲染

- 先 **1–2 GPU**，不要 8 卡  
- 输出 `render_results/**/render.mp4`

### Phase 4 — Stage 3 WorldStereo（贵）

- 默认 **worldstereo-memory-dmd**（4-step）  
- 先试 **H100×1–2 + FSDP 或单卡 offload**；不行再升  
- 轨迹数砍到最少

### Phase 5 — Stage 4–5 成世界

- Stage 4：复用 007 WorldMirror 权重（已有 volume）  
- Stage 5：单卡 PRO 6000 / H100，`max_steps` 按官方比例拉长（x1 → 8000）  
- 导出 ply / spz / mesh + 简单 HTML 预览

### 明确不做（默认）

- 8×H100 全 pipeline 一键  
- 无预算确认的 Stage 3 多轨迹  
- 再下 80B HY-Pano full（继续用 008 Qwen 全景）

---

## 5. 与 007 / 008 的关系

| | 007 WorldMirror | 008 HY-Pano | 009 WorldGen |
|---|---|---|---|
| 输入 | 多图/视频 | 单图 | **全景**（来自 008） |
| 输出 | 点云/深度/相机/GS attrs | panorama.png | **可导航 3D 世界** |
| 默认卡 | T4 | **PRO 6000** | 分 stage；Stage3 才上高端 |
| 角色 | 重建支线；也被 Stage4 调用 | 生成管线第 0 步 | 生成管线后半程 |

---

## 6. 你确认后才开火的命令（草案）

```bash
python main.py 009 status
python main.py 009 prepare --from-008 runs/smoke_qwen   # 接 008 全景
python main.py 009 stage1 --gpu RTX-PRO-6000            # 轨迹
# … stage2 / stage3 需你确认预算后再跑
```

**下一步建议：** 先 `prepare` + `stage1`（最便宜验证），Stage3 前再问你一次。
