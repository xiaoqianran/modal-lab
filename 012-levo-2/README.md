# 012 · LeVo 2 / SongGeneration v2

Tencent LeVo 2 全曲生成（歌词 + 描述 → 人声/伴奏）· 默认 L40S。

012 已迁移到 v2：一个 `app.py` 同时拥有模型归一化、显存策略、CLI 与 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

## 模型

```text
v2-medium -> lglg666/SongGeneration-v2-medium
v2-large  -> lglg666/SongGeneration-v2-large
runtime   -> lglg666/SongGeneration-Runtime
```

## 用法

```bash
python main.py 012 status
python main.py 012 check
python main.py 012 download --dry-run --model v2-large --force
python main.py 012 download --model v2-medium

python main.py 012 smoke --dry-run --model v2-large --gpu L4
python main.py 012 smoke --model v2-medium

python main.py 012 t2a --dry-run \
  --lyrics '[verse] Hello world in the neon rain.' \
  --descriptions 'male, rock, energetic, electric guitar' \
  --generate-type separate \
  --no-flash
```

## 真实不变量

large 模型在小显存 GPU 上 smoke 会自动低内存：

```text
model=v2-large
AND gpu in {L4,T4,A10}
  -> low_mem=true
```

显式 `--low-mem` 始终生效；RTX-PRO-6000 等大卡不会被自动强制。

生成参数直接归唯一 CLI：

```text
model
lyrics
descriptions
idx
generate_type
low_mem
flash_attn
run_name
```

## Volume

v2 删除 `ls/pull`：

```bash
modal volume ls modal-lab-levo-2-outputs runs
modal volume get modal-lab-levo-2-outputs runs/smoke_en ./012-levo-2/outputs
```

## 许可

Tencent SongGeneration 条款：仅研究 / 学术 / 教育用途，禁止商用。

## 测试

```bash
python -m unittest discover -s 012-levo-2/tests -v
python -m py_compile 012-levo-2/app.py
python main.py 012 status
python main.py 012 smoke --dry-run --model v2-large --gpu L4
```

以上测试不启动付费 GPU。
