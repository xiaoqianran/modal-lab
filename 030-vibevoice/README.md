# 030 · VibeVoice（TTS Tier A2）

[Microsoft VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) · **MIT**  
代码：[microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)

| 项 | 值 |
|----|-----|
| 槽位 | **030** |
| 默认模型 | **Realtime 0.5B** 流式 TTS |
| 默认 GPU | **L4** |
| 许可 | **MIT** |
| 排名 | GH ~52k stars **#1** · Realtime HF ~594k |

> 官方长文多说话人 TTS 推理码已撤；本槽用 Realtime 变体。

## 快速开始

```bash
cd 030-vibevoice
python run.py download
python run.py smoke --kind en
python run.py smoke --kind long
python run.py smoke --kind emma
```
