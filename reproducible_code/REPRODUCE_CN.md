# OUR 可执行代码说明

## 1. 已验证环境

- Python 3.12.8
- PyTorch 2.7.0
- NumPy 1.26.4
- SciPy 1.15.1
- SimPy 4.1.2

可按 `requirements.txt` 安装依赖；当前正式 OUR/Pegasus 入口不依赖历史 TensorFlow 代码。

## 2. 数据集

主数据文件：

```text
datasets/pegasus_pscale/dag_pegasus5_full31.json
SHA-256: 0671d8ea1ecdd8165062e19733e4859edb8e5ce87ecdd054bcef290abc49d5a5
```

它包含 Montage、CyberShake、Epigenomics、Inspiral 和 Sipht 五类工作流，每个 DAG 保留
24--30 个真实任务。`datasets/alibaba_cp100/` 仅用于受控跨数据集压力测试。

## 3. 核心测试

```bash
python -m unittest -q \
  test_information_protocol.py \
  test_capacity_protocol.py \
  test_dag_completion_semantics.py \
  test_critical_path_cache.py \
  test_critical_path_reward.py \
  test_critical_path_rl.py \
  test_pegasus_pscale_protocol.py \
  test_pegasus_b8_heterogeneity.py
```

本包实测：70 项测试全部通过。

## 4. 训练链 smoke

```bash
python run_reproduction_suite.py \
  --profile pegasus_pscale_p2_smoke \
  --suite-dir results/repro_smoke \
  --seeds 41 \
  --labels daoc_paper,lean_our \
  --dag-dataset-path datasets/pegasus_pscale/dag_pegasus5_full31.json \
  --dag-dataset-sha256 0671d8ea1ecdd8165062e19733e4859edb8e5ce87ecdd054bcef290abc49d5a5 \
  --eval-dag-families Montage,CyberShake,Epigenomics,Inspiral,Sipht \
  --server-capacity-multiset 0,0,0,0,1,1,1,1,2,2 \
  --capacity-assignment-namespace pegasus_pscale_p2 \
  --workers 2
```

本包实测 DAOC 和 OUR 均完成 200 轮训练及 20 场景评估。smoke 只验证执行完整性，不能
作为论文性能结论。

## 5. 正式统一预算入口

```bash
python run_pegasus_common_horizon_suite.py \
  --profile pegasus_common_horizon_26k \
  --suite-dir results/repro_common_horizon_26k \
  --seeds 51,52,53,54,55,56,57,58,59,60 \
  --labels lean_our,coord_cache_discrete_sac,discrete_sac_std_cache \
  --workers 4 \
  --seed-partition confirmation \
  --keep-going
```

正式训练成本较高；Pro 终审可先执行测试与 smoke，再审查包内已有 seed 级结果和协议锁。

## 6. 当前核心源码哈希

对以下文件逐文件 SHA-256 后再次哈希所得：

```text
807cedc78f858ee5e3e66e6781b7753b86e79a27f12b8a9545fb62258a79471a
```

范围：`agent.py`、`broker.py`、`capacity_protocol.py`、`critical_path_cache.py`、
`critical_path_reward.py`、`critical_path_rl.py`、`discrete_sac.py`、`dqn.py`、
`information_protocol.py`、`run_independent_experiment.py`、`run_reproduction_suite.py`、
`server.py`、`simulator.py`、`task.py` 和 `user.py`。

本包未放入旧 checkpoint。原因是 2026-08-21 后到达时刻、奖励和 DAG 终止语义已修订；
旧 checkpoint 可用于历史审计，但不应冒充与当前代码完全锁定的正式模型。
