# 043 · Qwen-Image-2512

目的：验证 `qwen-image-2512` 在当前 `modal-2D` 生产 contract 下真实可用，并记录 GPU 选择结论。

```text
modal-2D capabilities
    ↓ generation_entrypoint
Modal Cls.from_name(...)
    ↓
Qwen-Image-2512 Worker @ RTX-PRO-6000
    ↓
1024×1024 PNG Artifact
    ↓
read_artifact + PNG / bytes / SHA-256 校验
```

## 已验证结论

- **L40S：失败（OOM）**。44.39 GiB 基本吃满，最后额外申请约 72 MiB 时失败。
- **RTX PRO 6000：通过**。
- 实测加载约 **30.9s**，单张 1024×1024 / 50 steps 推理约 **17.28s**。
- 峰值 CUDA allocated 约 **57.96 GiB**。
- 生产建议：`RTX-PRO-6000`。


## 运行

```bash
python main.py 043 --check-env
python main.py 043
```

结果写入 ignored `results/latest.json`，生成 PNG 写入同目录 `results/`。

生产实现真值：`xiaoqianran/modal-provider` → `modal-2D`。
