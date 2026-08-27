# 016 MusicGen · 成本对照

Modal 价目（$/s）：T4 **0.000164** · L4 0.000222 · L40S 0.000542

估费 = 墙钟 × GPU 单价（含冷启动加载；无 keep_warm）。

## 结论

1. **最便宜默认：T4 + musicgen-small**（已是价目最低档）
2. **L4 并不更省**：同 15s lo-fi，L4 墙钟 35.3s / **$0.0078**，T4 31.8s / **$0.0052**（L4 生成还更慢，可能排队/波动）
3. **最短冒烟**：T4 · **8s** → **~$0.0037**（加载仍占 ~12s，再短收益递减）
4. **其它任务**（jazz / EDM · 12s · T4）约 **$0.004–0.005**，风格不影响成本结构
5. 若未来做 **keep_warm / 常驻**：生成段 T4 约 **$0.00017/s 音频时间**；15s 生成 ~$0.003

## 实测表

| run | GPU | 请求时长 | 实际音频 | 墙钟 s | 加载 s | 生成 s | 估费 $ | VRAM GB |
|-----|-----|----------|----------|--------|--------|--------|--------|---------|
| `bench_l4_15s` | L4 | — | 14.94 | 35.31 | 12.41 | 22.89 | 0.0078 | 2.972 |
| `bench_t4_8s` | T4 | — | 7.94 | 22.85 | 12.38 | 10.47 | 0.0037 | 2.702 |
| `bench_t4_edm` | T4 | — | 11.94 | 31.35 | 15.96 | 15.37 | 0.0051 | 2.854 |
| `bench_t4_jazz` | T4 | — | 11.94 | 27.09 | 12.2 | 14.88 | 0.0044 | 2.853 |
| `smoke_lofi` | T4 | — | 14.94 | 31.76 | 13.97 | 17.79 | 0.0052 | 2.974 |

## 任务 prompt

| run | prompt |
|-----|--------|
| `bench_l4_15s` | lo-fi hip hop instrumental, soft piano, dusty drums, chill study beat |
| `bench_t4_8s` | lo-fi hip hop instrumental, soft piano, dusty drums, chill study beat |
| `bench_t4_edm` | energetic electronic dance music, four on the floor kick, bright synth stabs, festival ene |
| `bench_t4_jazz` | upbeat jazz piano trio, walking bass, brushes on drums, swing feel |
| `smoke_lofi` | lo-fi hip hop instrumental, soft piano, dusty drums, warm vinyl crackle, chill study beat |

## 更省钱的操作建议

| 策略 | 做法 | 约省 |
|------|------|------|
| 缩短冒烟 | `python main.py 016 smoke --duration 8` | 15s→8s 约少 $0.0015 |
| 坚持 T4 | 不要默认切 L4/A100 | 相对 L4 省 ~30%+ |
| 批量同容器 | 多次 t2a 同一次 GPU 会话（需改 app 支持 batch） | 摊薄 ~12s 加载 |
| 勿上 medium 除非质量需要 | medium 更大更慢 | 成本明显上升 |

试听：[`gallery/index.html`](gallery/index.html)
