# 上游锁定

| 项 | 值 |
|----|-----|
| 仓库 | https://github.com/meituan-longcat/LongCat-Video |
| 分支 | `main` |
| 本地路径 | `LongCat-Video/` |
| 浅克隆 | `--depth 1` |

记录克隆时的 commit（便于对照官方变更）：

```text
6b3f4b8582a8bc3f20f795735f5383716c4ba794  # Update README.md（2026-07-27 浅克隆）
```

更新上游：

```bash
cd LongCat-Video
git fetch --depth 1 origin main
git checkout FETCH_HEAD
# 或删掉后: python run.py setup
```

权重仓库（不入库，走 Modal Volume）：

- `meituan-longcat/LongCat-Video`
- Avatar 系列留给后续 `002-*` 实验
