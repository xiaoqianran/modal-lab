# 012 · LeVo 2 — PLAN

## Slot

- **目录**：`012-levo-2`
- **顺序**：音乐队列第 1 个待实现（在 010/011 之后）
- **模型**：LeVo 2（Tencent SongGeneration 线）
- **赛道**：全曲 + 歌词 · 2026 横评听感常排第一

## 依赖与风险

1. **License**：实现前确认是否允许复现/分享生成物；README 顶栏写清
2. **权重**：HF 路径与是否门禁（可能需要已有 `hf-token` secret）
3. **VRAM**：默认规划 **A100-40GB**；失败再升 80GB / 降采样时长

## 交付物（对齐 010/011）

- [ ] `modal_app.py` — download / smoke / t2a
- [ ] `run.py` + `README.md` + `UPSTREAM.md`
- [ ] Volume：`modal-lab-levo-2-weights` / `modal-lab-levo-2-outputs`
- [ ] smoke FLAC + `gallery/index.html`
- [ ] 根 README 挂链

## Smoke 规格（草案）

| 项 | 值 |
|----|-----|
| duration | 20–30 s（先短） |
| seed | 42 |
| 输入 | 英文/中文短歌词 + style caption |
| 输出 | FLAC 44.1k stereo |
| meta | wall_s / est_gpu_usd / vram |

## 不在本号

- YuE → `013-yue`
- DiffRhythm 2 → `014-diffrhythm-2`
- MusicGen → `016-musicgen`（跳过 015 robotics）
