# 016 · MusicGen — PLAN

## Slot

- **目录**：`016-musicgen`（**跳过 015**，015 = Xiaomi Robotics）
- **顺序**：音乐队列最后（DiffRhythm 之后）
- **模型**：Meta MusicGen（AudioCraft）
- **赛道**：短/中时长 text→instrumental 基线

## 依赖与风险

1. `audiocraft` + HF `facebook/musicgen-small`（默认）/ `medium`（可选）
2. 不做歌词人声全曲；与 011 SA3 对照器乐
3. 默认 GPU：**T4**（small）或 **L4**（medium）— 全音乐线最便宜收尾

## 交付物

- [ ] modal_app / run / README / UPSTREAM
- [ ] volumes + smoke + gallery
- [ ] 根 README 挂链

## Smoke 规格（草案）

| 项 | 值 |
|----|-----|
| model | musicgen-small |
| duration | 15–20 s |
| seed | 42 |
| GPU | T4 |
| 输出 | WAV/FLAC + meta.json |
