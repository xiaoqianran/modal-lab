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

## 本地环境检查

脚本使用 PEP 723 声明最小依赖；根 launcher 优先复用已有实验/root venv；没有 venv 时通过 `uv run --script` 创建隔离运行环境，不修改全局 Python。

```bash
python main.py 041 --check-env
```

该命令验证本地依赖、AgentScape 模块和 ONNX Runtime provider，不连接远端 provider、不启动 GPU。直接 `python main.py 041` 才会执行完整 provider 验收。

## 验收

- rembg 必须产生非空 foreground 与有效 alpha。
- canonical 必须为 `1024×1024 RGBA PNG`。
- 每个 enabled model 必须接受同一 canonical input。
- 对同一 `model + input_path + options` 重复 gateway submit，`call_id` 必须一致。
- 每个成功结果必须产生真实 GLB，且 `glTF` magic、version=2、declared bytes、SHA-256 全匹配。
- 某一个模型失败不阻塞其他模型结果落盘。
