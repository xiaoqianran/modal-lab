# Upstream — 027 Qwen3-TTS

| 项 | 值 |
|----|-----|
| Code | https://github.com/QwenLM/Qwen3-TTS |
| PyPI | `qwen-tts==0.1.1` |
| Collection | https://huggingface.co/collections/Qwen/qwen3-tts |
| License | **Apache-2.0** |
| Paper | https://arxiv.org/abs/2601.15621 |

## 权重

| key | HF repo | 用途 |
|-----|---------|------|
| `custom_1.7` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 预设 + instruct（默认） |
| `custom_0.6` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 轻量预设 |
| `base_1.7` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 3s 语音克隆 |
| `design_1.7` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 文本造声 |
| tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 随 download 拉取 |

## API

```python
from qwen_tts import Qwen3TTSModel
import torch, soundfile as sf

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
wavs, sr = m.generate_custom_voice(
    text="你好",
    language="Chinese",
    speaker="Vivian",
    instruct="用温和的语气说",
)
sf.write("out.wav", wavs[0], sr)
```

## 预设 speakers

Vivian · Serena · Uncle_Fu · Dylan · Eric · Ryan · Aiden · Ono_Anna · Sohee

## 备注

- 不强制 flash-attn；缺省 **SDPA**，有 flash_attn 再试 `flash_attention_2`。
- transformers 由包钉死 `==4.57.3`。
