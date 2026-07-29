# 上游固定版本

- 仓库：https://github.com/baidu/Unlimited-OCR
- Git commit：`4ba2ea3eb384757710bc7f7678922b0b61045448`
- 模型：https://huggingface.co/baidu/Unlimited-OCR
- 模型 revision：`3f2e9c956588f5560efcfb7c62240f5d67b63e60`

推理参数与官方单页 `gundam` 配置一致：

```text
prompt=<image>document parsing.
base_size=1024
image_size=640
crop_mode=True
max_length=32768
no_repeat_ngram_size=35
ngram_window=128
temperature=0
```
