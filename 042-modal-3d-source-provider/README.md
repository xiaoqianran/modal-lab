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


## 本地环境检查

脚本通过 PEP 723 声明 `modal + Pillow`；根 launcher 优先复用已有实验/root venv，没有 venv 时通过 `uv run --script` 隔离执行。

```bash
python main.py 042 --check-env
```

该命令只验证本地依赖，并报告 040/041 baseline 文件是否存在；不连接远端 provider、不启动 GPU。直接 `python main.py 042` 才执行完整 source-image parity 验收。

## Verified Result — 2026-08-28

```text
status                        PASS
source contract               source_image / provider conditioning
source sha256                 eebd43afbda1c79d8b1e70b7b7cb8264f05dde1594f6bda94f4a11fee705e0e4
canonical sha256              32284d4987cb805f340cba211a53caece8302399bf52f34b37522f87d611f5c7
foreground ratio              0.2843132019042969
```

| Model | Result | Elapsed | GLB bytes |
|---|---|---:|---:|
| FastSAM3D++ | PASS | 141.795s | 7,524,532 |
| Hermite-TRELLIS2++ | PASS | 458.769s | 35,302,568 |
| Hunyuan2.1++ | PASS | 256.701s | 3,610,148 |
| Pixal3D | PASS | 569.518s | 35,762,932 |

四个模型均满足：duplicate submit 复用同一 `callId`、`strategy=birefnet`、`engine=birefnet-general-lite`、source digest 一致、canonical digest 一致、GLB v2 / bytes / SHA-256 全匹配。

与 041 相比，GLB bytes 与耗时有明显变化，因此本实验确认：**迁移 Gate 应验证 contract / conditioning / Artifact integrity，而不能要求 exact output bytes parity。**
