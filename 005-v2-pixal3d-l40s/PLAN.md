# 005-v2 实施计划

## 已完成

- [x] Phase 0 研究（L40S=sm_89，HF demo 根因，社区对照）  
- [x] Phase 1 目录骨架  
- [x] Phase 2 镜像 + `build-sm89`（6 个 wheel 入 Volume）  
- [x] Phase 3 `verify`（L40S (8,9) + HAS_LIBNATTEN）  
- [x] Phase 4 权重下载 + smoke 端到端 GLB  
- [x] 文档 / 本地 viewer  

## 可选后续

- [ ] 推送 `005-v2` 到远程（**默认未推**，等你点头）  
- [ ] 第二次 smoke 测「热轮子 / 无 autotune」更准的纯推理时间  
- [ ] 与 005 H100 同 seed 肉眼对照写进 gallery  

## 关键命令

```bash
python run.py build-sm89 --i-know-this-costs-money
python run.py verify --i-know-this-costs-money
python run.py smoke --i-know-this-costs-money
```
