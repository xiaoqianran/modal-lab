# 040 · modal-2D Provider Verification

目的：验证 `modal-2D` 当前公开能力是否真实可用，而不是只验证 API 存在。

```text
capabilities
    ↓
2 models × 2 seeds
    ↓
real Modal L40S generation
    ↓
Artifact Descriptor
    ↓
modal-2d-artifacts Volume
    ↓
PNG header / dimensions / bytes / SHA-256
```

## 验收

- capability 必须是 `image.generate`。
- 两个声明模型都必须真实生成成功。
- 每个模型使用 seed `42` / `73`，共 4 个候选。
- 每个 Artifact 必须是 `1024×1024 image/png`。
- descriptor `bytes / digest / sha256 / producer` 与真实 bytes 一致。
- 同模型两个 seed 的 digest 必须不同。

结果写入 `results/latest.json`；PNG 仅作为实验 Artifact 保留在 ignored `results/`。
