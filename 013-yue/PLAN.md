# 013 · YuE — PLAN

## Slot

- **目录**：`013-yue`
- **顺序**：LeVo 2（012）之后
- **模型**：YuE（M-A-P）
- **赛道**：全曲 + 歌词 · 长程结构

## 依赖与风险

1. 常见 **stage1 + stage2** 双权重，镜像体积与下载时间大
2. 推理慢 → smoke 必须限制时长 / token
3. 默认 GPU：**A100-40GB**（不够则 80GB）

## 交付物

- [ ] modal_app / run / README / UPSTREAM
- [ ] volumes + smoke + gallery
- [ ] 根 README 挂链

## Smoke 规格（草案）

| 项 | 值 |
|----|-----|
| duration | 尽量短的官方可跑配置 |
| seed | 42 |
| GPU | A100-40GB |
| 输出 | FLAC + meta.json |
