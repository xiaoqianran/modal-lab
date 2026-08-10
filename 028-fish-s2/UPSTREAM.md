# Upstream — 028 Fish Audio S2 Pro

| 项 | 值 |
|----|-----|
| Code | https://github.com/fishaudio/fish-speech |
| Weights | https://huggingface.co/fishaudio/s2-pro |
| Size | **4B** Dual-AR · codec + sharded safetensors ~**11GB** |
| License | **Fish Audio Research License**（研究/非商用；商用联系 business@fish.audio） |
| Docs | https://speech.fish.audio/ |
| Paper | https://arxiv.org/abs/2603.08823 |
| Blog | https://fish.audio/blog/fish-audio-open-sources-s2/ |
| Elo | AA open-weights **#1 (1121)** |

## Install (upstream)

```bash
pip install -e .[cu126]   # torch 2.8
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## Programmatic path (本实验使用)

```python
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest

llama_queue = launch_thread_safe_queue(
    checkpoint_path="checkpoints/s2-pro",
    device="cuda",
    precision=torch.bfloat16,
    compile=False,
)
decoder = load_decoder_model(
    config_name="modded_dac_vq",
    checkpoint_path="checkpoints/s2-pro/codec.pth",
    device="cuda",
)
engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder, precision=..., compile=False)
for r in engine.inference(ServeTTSRequest(text="Hello [excited]!", references=[])):
    ...
```

## 许可注意

- 研究 / 个人评估 / 非商用：本协议免费
- **商用产品 / 托管 API / 内部付费服务**：需单独商业授权
- 分发需保留 Notice + “Built with Fish Audio”
