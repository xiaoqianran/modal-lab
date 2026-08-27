# 042 · modal-3D Source Image Contract Verification

验证新的 `modal-3D` 公开输入路径：Caller 直接提交 PNG/JPEG/WebP 原图，由 `modal-3D` Provider 负责 Input Conditioning，再进入四个 3D Worker。

```text
040 opaque source image
        │
        ▼
source-inputs/<sha>.*
        │
        ▼
modal-3D Gateway
        │
        ▼
Provider InputConditioner
        │
        ├─ valid alpha → preserve
        └─ opaque → pinned BiRefNet
                 │
                 ▼
       client-inputs/<canonical sha>.png
                 │
        ┌────────┼────────┬────────┐
        ▼        ▼        ▼        ▼
   FastSAM   TRELLIS   Hunyuan   Pixal3D
        │        │        │        │
        └────────┴────────┴────────┘
                 ▼
            GLB Artifact
```

Gate：public source contract、duplicate submit 幂等、4/4 模型、conditioning evidence、同源 canonical digest 一致、GLB v2/bytes/SHA-256 全通过。若 041 baseline 在本机存在，则只记录性能/bytes 差异，不要求 GLB digest 完全相同。
