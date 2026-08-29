# 045 · HiDream-O1-Image

目的：验证 `hidream-o1-image` 在当前 `modal-2D` 生产 contract 下真实可用，并记录 GPU 选择结论。

```text
modal-2D capabilities
    ↓ generation_entrypoint
Modal Cls.from_name(...)
    ↓
HiDream-O1-Image Worker @ RTX-PRO-6000
    ↓
1024×1024 PNG Artifact
    ↓
read_artifact + PNG / bytes / SHA-256 校验
```

## 已验证结论

- **L40S：通过**，load **41.75s**，infer **50.57s**，峰值约 **17.38 GiB**。
- **RTX PRO 6000：通过**，load **22.82s**，infer **33.48s**，峰值约 **17.41 GiB**。
- RTX PRO 6000 推理约快 **34%**，模型加载也显著更快。
- HiDream 官方 runtime 会把 1024×1024 snap 到 2048×2048；生产 Worker 已在模型边界做 LANCZOS 2048→1024，保持统一 2D Artifact contract。
- 生产建议：`RTX-PRO-6000`。


## 运行

```bash
python main.py 045 --check-env
python main.py 045
```

结果写入 ignored `results/latest.json`，生成 PNG 写入同目录 `results/`。

生产实现真值：`xiaoqianran/modal-provider` → `modal-2D`。
