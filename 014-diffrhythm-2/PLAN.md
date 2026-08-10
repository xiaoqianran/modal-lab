# 014 · DiffRhythm 2 — PLAN

## Slot

- **目录**：`014-diffrhythm-2`
- **顺序**：YuE（013）之后
- **模型**：DiffRhythm 2（ASLP-lab）
- **赛道**：全曲 diffusion · 极速

## 依赖与风险

1. 上游 `ASLP-lab/DiffRhythm` + v2 权重 pin
2. 歌词条件格式可能与 ACE/LeVo 不同（phoneme / 时间戳历史包袱）→ 跟官方 inference 脚本
3. 目标默认 GPU：**L4**（优惠线，对齐 010/011）

## 交付物

- [ ] modal_app / run / README / UPSTREAM
- [ ] volumes + smoke + gallery
- [ ] 可选：与 010 同 prompt 的速度对照表

## Smoke 规格（草案）

| 项 | 值 |
|----|-----|
| duration | 20–30 s |
| seed | 42 |
| GPU | L4 |
| 输出 | FLAC + meta.json |
