# Gallery — HunyuanWorld-Mirror smoke 结果

从 Modal Volume `modal-lab-hunyuanworld-mirror-outputs/runs/smoke_bright_room`
拉取后生成的本地展示页。

## 内容

- 交互式 3D 点云（`assets/points.bin`，全量 PLY 下采样至 80k）
- 每帧 Input / Depth / Normal 对照
- Run meta（GPU、时长、费用、显存）

## 本地预览

```bash
cd gallery
python3 -m http.server 8080 --bind 0.0.0.0
# 打开 http://127.0.0.1:8080/
```

## 重新拉取并构建

```bash
# 从 Volume 拉最新 run
modal volume get modal-lab-hunyuanworld-mirror-outputs runs/smoke_bright_room gallery-assets/

# 再跑一次资产准备（见仓库脚本或手工 python）
```
