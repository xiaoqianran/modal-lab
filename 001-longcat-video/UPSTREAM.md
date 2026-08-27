# 上游锁定

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/meituan-longcat/LongCat-Video |
| 分支 | `main` |
| 本地路径 | `LongCat-Video/` |
| 浅克隆 | `--depth 1` |

记录克隆时的 commit：

```text
6b3f4b8582a8bc3f20f795735f5383716c4ba794  # Update README.md（2026-07-27 浅克隆）
```

重新拉取上游：

```bash
rm -rf 001-longcat-video/LongCat-Video
python main.py 001 setup
```

或手工更新：

```bash
cd 001-longcat-video/LongCat-Video
git fetch --depth 1 origin main
git checkout FETCH_HEAD
```

实验自己的代码不要放进 `LongCat-Video/`。例如 storyboard workflow 已迁移到：

```text
001-longcat-video/storyboard.py
```

权重仓库：

```text
meituan-longcat/LongCat-Video
```

权重不入 Git，统一走 Modal Volume。
