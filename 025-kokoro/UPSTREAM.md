# Upstream — 025 Kokoro

| 项 | 值 |
|----|-----|
| Code | https://github.com/hexgrad/kokoro |
| PyPI | `kokoro>=0.9.4` · G2P `misaki[en,zh]` |
| Weights v1 | https://huggingface.co/hexgrad/Kokoro-82M |
| Weights v1.1-zh | https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh |
| Architecture | StyleTTS 2 decoder · ISTFTNet · **82M** |
| License | Apache-2.0 |
| Voices doc | https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md |
| Sample rate | 24000 Hz |

## 推理入口

```python
from kokoro import KModel, KPipeline
model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda").eval()
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
for gs, ps, audio in pipe("Hello", voice="af_heart"):
    ...
```

中文 v1.1-zh：`lang_code='z'` · `repo_id='hexgrad/Kokoro-82M-v1.1-zh'` · voice e.g. `zf_001`。
