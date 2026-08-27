# 015 · Xiaomi-Robotics-1 RoboCasa365

在 Modal 上加载 `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`，做 **VLA 动作生成冒烟**。

015 已迁移到 v2：一个 `app.py` 同时拥有 smoke/infer planning、CLI、结构化 run 元数据和 Modal remote functions，不再使用 `run.py -> modal_app.py` 包装层。

本实验范围很窄：

```text
合成三视角历史帧
+ 语言指令
+ proprio state
        ↓
Xiaomi-Robotics-1 / MiBoT
        ↓
16-step action chunk
```

完整 RoboCasa365 MuJoCo 成功率评测不在本实验范围。

## 用法

```bash
python main.py 015 status
python main.py 015 check
python main.py 015 download --dry-run --force
python main.py 015 download

# 固定 smoke
python main.py 015 smoke --dry-run
python main.py 015 smoke --gpu A100-40GB

# 自定义动作生成
python main.py 015 infer --dry-run \
  --instruction 'open the drawer' \
  --gpu L40S \
  --attn eager \
  --num-steps 7 \
  --obs-history 3

# 解析远程 run 目录
python main.py 015 list-outputs
```

## Smoke 不变量

默认：

```text
instruction = close the blender lid
run_name = smoke_close_blender_lid
attn = sdpa
num_steps = 5
obs_history = 4
gpu = A100-40GB
```

`infer` 才用于自由指令和不同 attention / denoise / history 参数。

## 输出

`list-outputs` 保留，因为它按 run 返回文件集合；纯文件浏览/拉取直接使用 Modal：

```bash
modal volume ls modal-lab-xr1-robocasa365-outputs runs
modal volume get \
  modal-lab-xr1-robocasa365-outputs \
  runs/smoke_close_blender_lid \
  ./015-xiaomi-robotics-1-robocasa365/outputs
```

Volume：

```text
modal-lab-xr1-robocasa365-weights
modal-lab-xr1-robocasa365-outputs
```

## 模型要点

```text
架构        Qwen3-VL-4B + DiT action head
robot_type  robocasa365
state_dim   60
action_dim  前 12 维
输出        (16, 60) normalized actions
```

默认用 `sdpa` 是为了降低 cold-start 依赖复杂度；`flash_attention_2` / `eager` 可显式选择。

## 测试

```bash
python -m unittest discover -s 015-xiaomi-robotics-1-robocasa365/tests -v
python -m py_compile 015-xiaomi-robotics-1-robocasa365/app.py
python main.py 015 status
python main.py 015 smoke --dry-run
```

以上测试不启动付费 GPU。

上游 Apache-2.0。完整评测边界见 [`UPSTREAM.md`](UPSTREAM.md)。
