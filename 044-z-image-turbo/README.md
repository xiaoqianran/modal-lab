# 044 · Z-Image-Turbo

目的：验证 `z-image-turbo` 在当前 `modal-2D` 生产 contract 下真实可用，并记录 GPU 选择结论。

```text
modal-2D capabilities
    ↓ generation_entrypoint
Modal Cls.from_name(...)
    ↓
Z-Image-Turbo Worker @ L40S
    ↓
1024×1024 PNG Artifact
    ↓
read_artifact + PNG / bytes / SHA-256 校验
```

## 已验证结论

- **L40S：通过**。
- 1024×1024 PNG 真实生成成功，Artifact descriptor / SHA-256 正常。
- 9 steps，guidance 固定为 `0.0`。
- 生产建议：`L40S`，没有必要上更高规格 GPU。


## 运行

```bash
python main.py 044 --check-env
python main.py 044
```

结果写入 ignored `results/latest.json`，生成 PNG 写入同目录 `results/`。

生产实现真值：`xiaoqianran/modal-provider` → `modal-2D`。
