# 041 · modal-3D Provider Verification

目的：用同一个真实 2D 候选，验证当前 3D 输入准备与四个 enabled Image→3D 模型是否真实可用。

```text
040 real PNG
    ↓
BiRefNet background removal
    ↓
component analysis
    ↓
1024 RGBA canonical
    ↓
modal-3D gateway
    ├─ FastSAM3D++
    ├─ Hermite-TRELLIS2++
    ├─ Hunyuan2.1++
    └─ Pixal3D
         ↓
GLB Artifact
         ↓
Volume stream / glTF v2 / bytes / SHA-256
```

## 验收

- rembg 必须产生非空 foreground 与有效 alpha。
- canonical 必须为 `1024×1024 RGBA PNG`。
- 每个 enabled model 必须接受同一 canonical input。
- 对同一 `model + input_path + options` 重复 gateway submit，`call_id` 必须一致。
- 每个成功结果必须产生真实 GLB，且 `glTF` magic、version=2、declared bytes、SHA-256 全匹配。
- 某一个模型失败不阻塞其他模型结果落盘。
