# 027 · Qwen3-TTS（TTS Tier S3）

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · Apache-2.0  
家族 HF 合计 ~**6.6M+** downloads · GH ~13k。

027 已迁移到 v2：一个 `app.py` 同时拥有模型 family、CLI 和 Modal remote functions，不再使用 `run.py -> modal_app.py` 二次转译。

| 项 | 值 |
|----|-----|
| 槽位 | **027** |
| 默认模型 | **1.7B-CustomVoice**（9 预设 + instruct） |
| 可选 | **0.6B-Custom** · **1.7B-Base 克隆** · **1.7B-VoiceDesign** |
| 默认 GPU | **L4** |
| 许可 | **Apache-2.0** |
| Image | `qwen-tts==0.1.1` · attn SDPA |

## 用法

```bash
# 纯本地固定信息
python main.py 027 status

# 远程检查模型 / prompts / outputs Volume
python main.py 027 check

# 下载 canonical model；alias 由 app.py 自己归一化
python main.py 027 download --dry-run --model design
python main.py 027 download --model custom_1.7
python main.py 027 download --model all

# 四种固定 smoke 基线
python main.py 027 smoke --dry-run --kind custom_zh
python main.py 027 smoke --kind custom_zh
python main.py 027 smoke --kind custom_en
python main.py 027 smoke --kind design
python main.py 027 smoke --kind clone

# 通用 CustomVoice / model-select TTS
python main.py 027 t2s --dry-run \
  --model custom_0.6 --text '你好世界' --speaker Vivian --lang zh

# VoiceDesign：模型固定 design_1.7
python main.py 027 design --dry-run \
  --text '要抱抱！' --instruct '撒娇萝莉女声'

# Voice Clone：模型固定 base_1.7；不传 ref 时使用官方 demo ref
python main.py 027 clone --dry-run --text 'Hello from clone.'
```

也可以直接使用 Modal：

```bash
cd 027-qwen3-tts
modal run app.py check
modal run app.py download --model custom_1.7
modal run app.py smoke --kind design
```

## CLI / 模型边界

模型 canonical key 只维护在：

```text
HF_REPOS
MODEL_ALIASES
_norm_model()
```

CLI 不再复制一套模型 choices，因此新增 alias / family member 时不会出现 wrapper 和远端入口不同步。

三个生成入口保持明确领域语义：

```text
t2s    -> 通用模型选择；默认 CustomVoice
design -> 强制 design_1.7
clone  -> 强制 base_1.7
```

四种 smoke 则是稳定 benchmark 场景，而不是通用配置系统。

## Smoke 实测（2026-08-11 · L4 冷启动）

| run | model | gen_s | wall_s | est_usd | audio |
|-----|-------|------:|-------:|--------:|------:|
| smoke_custom_zh_vivian | custom_1.7 | 16.83 | 24.04 | **$0.0053** | 9.3s |
| smoke_custom_en_ryan | custom_1.7 | 13.22 | 18.79 | $0.0042 | 7.7s |
| smoke_design_zh | design_1.7 | 11.46 | 18.45 | $0.0041 | 5.4s |
| smoke_clone_en | base_1.7 | 19.73 | 26.37 | $0.0059 | 6.6s |

详见 [COST_BENCHMARK.md](COST_BENCHMARK.md) · 试听 [gallery/](gallery/index.html)。

## Volume

v2 不再包装 `ls/pull`：

```bash
modal volume ls modal-lab-qwen3-tts-outputs runs
modal volume get modal-lab-qwen3-tts-outputs runs/smoke_custom_zh_vivian ./027-qwen3-tts/outputs
```

| 名 | 用途 |
|----|------|
| `modal-lab-qwen3-tts-weights` | HF hub cache（~28.5 GB） |
| `modal-lab-qwen3-tts-prompts` | 克隆参考 wav（可选） |
| `modal-lab-qwen3-tts-outputs` | `runs/<name>/audio.wav` |

## 测试

```bash
python -m unittest discover -s 027-qwen3-tts/tests -v
python -m py_compile 027-qwen3-tts/app.py
python main.py 027 status
python main.py 027 smoke --dry-run --kind clone
```

以上测试不启动付费 GPU。

## 下一条

`028-fish-s2`
