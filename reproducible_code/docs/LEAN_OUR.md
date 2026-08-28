# Lean OUR

正式候选方法使用标签 `lean_our`，算法实现为
`causal_telemetryPD3QN`。

## 方法结构

正式方法是“一个强化学习主干 + 两个创新模块”：

1. 主干：Pairwise Dueling Double DQN，学习任务到服务器的卸载动作。
2. 创新一：Causal history telemetry，只用已完成任务的执行时延历史估计服务器质量。
3. 创新二：Causal dependency-aware joint cache，依据历史请求、依赖局部性和历史服务器质量更新服务缓存。

训练目标采用 makespan increment reward。它只是把最终完成时间分解成逐步可观测奖励，未折扣回报严格等于负的应用完成时间，不作为额外算法模块。

## 从正式方法删除

- Hindsight Critical-Path Replay (HCPR)
- Bottleneck-Contribution Replay (BCR)
- Workload-normalized telemetry
- Quantile/risk head
- Entropy regularization
- Historical feedback guidance
- Adaptive guidance gate

旧模块代码和旧结果目录继续保留，仅用于复现实验历史，不再进入默认严格实验。

## 实验入口

低成本检查：

```bash
.venv/bin/python run_strict_environment_suite.py \
  --stage smoke \
  --environments e0_original \
  --suite-dir results/lean_our_smoke
```

收敛筛选：

```bash
.venv/bin/python run_strict_environment_suite.py \
  --stage converged \
  --environments e0_original,e_cache \
  --suite-dir results/lean_our_converged_3seed \
  --workers 1
```

默认严格实验会自动比较 `guided_full` 和 `lean_our`。若需要复核旧方法，可显式传入
`--our-label hcpr_telemetry_pd3qn`。
