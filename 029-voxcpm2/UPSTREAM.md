# Upstream — 029 VoxCPM2

| 项 | 值 |
|----|-----|
| Code | https://github.com/OpenBMB/VoxCPM |
| Weights | https://huggingface.co/openbmb/VoxCPM2 |
| PyPI | `voxcpm==2.0.3` |
| Size | **2B** · model.safetensors ~4.6GB + AudioVAE |
| License | **Apache-2.0** |
| Docs | https://voxcpm.readthedocs.io/ |
| Paper | https://arxiv.org/abs/2606.06928 |

## API

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
wav = model.generate(text="Hello", cfg_value=2.0, inference_timesteps=10, seed=42)
sf.write("out.wav", wav, model.tts_model.sample_rate)  # 48kHz

# Voice design
wav = model.generate(text="(gentle young female)Hello!")

# Controllable clone
wav = model.generate(text="Hi", reference_wav_path="ref.wav")
```
