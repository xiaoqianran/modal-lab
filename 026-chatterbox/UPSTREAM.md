# Upstream — 026 Chatterbox

| 项 | 值 |
|----|-----|
| Code | https://github.com/resemble-ai/chatterbox |
| PyPI | `chatterbox-tts==0.1.7`（+ `peft`） |
| Turbo weights | https://huggingface.co/ResembleAI/chatterbox-turbo |
| Base / MTL | https://huggingface.co/ResembleAI/chatterbox |
| License | MIT |
| Modal example | https://modal.com/docs/examples/chatterbox_tts |
| Voice prompts | https://modal-cdn.com/blog/audio/chatterbox-tts-voices.zip |
| Watermark | Perth (built-in) |

## API

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.tts import ChatterboxTTS

m = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
wav = m.generate("你好", language_id="zh")

t = ChatterboxTurboTTS.from_pretrained(device="cuda")
wav = t.generate("Hi [chuckle]", audio_prompt_path="Lucy.wav")
```

## 0.1.7 备注

- `t3_model` / `nano` 等 kwargs 经 `inspect.signature` 过滤后传入 `from_pretrained`（0.1.6 无 `t3_model` 会炸）。
- Image 固定 `chatterbox-tts==0.1.7`。
