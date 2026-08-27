# Plan

唯一变化轴：

```text
041: source → Caller preprocess → canonical → modal-3D
042: source → modal-3D → Provider InputConditioner → canonical → Worker
```

默认输入沿用 040 的 `sana-sprint-1.6b / seed 42` opaque PNG；也支持 `--source` 指定任意 PNG/JPEG/WebP。
