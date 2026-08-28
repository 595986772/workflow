# 当前实现阅读索引

## OUR 与环境

1. `run_reproduction_suite.py`：正式方法标签、模块配置和多 seed 入口。
2. `run_independent_experiment.py`：单 seed 训练、checkpoint 和冻结评估。
3. `agent.py`：入口服务器级代理、Pairwise 状态构造与动作选择。
4. `dqn.py`：Pairwise Dueling Double DQN、目标网络与回放更新。
5. `critical_path_rl.py`：拓扑特征、逐候选服务器特征和 n-step 辅助逻辑。
6. `critical_path_reward.py`：相对应用到达时刻的 makespan-increment 奖励。
7. `critical_path_cache.py`：DCC、历史 EMA、覆盖和容量约束。
8. `capacity_protocol.py`：逐服务器异构容量、K=0 语义和确定性打乱。
9. `information_protocol.py`：因果可用信息协议。
10. `simulator.py`、`server.py`、`user.py`、`task.py`、`broker.py`：环境执行、
    多出口完成和时延统计。

## 正式实验协议

- `pegasus_pscale_protocol.py`：Pegasus 五类工作流和主容量环境。
- `pegasus_p6_protocol.py`：主基线和受控比较。
- `pegasus_common_horizon_protocol.py`：统一 26,000 轮训练协议。
- `pegasus_server_scaling_heuristics_protocol.py`：服务器规模实验。
- `pegasus_b8_heterogeneity_protocol.py`：固定 B=8 容量异构度实验。
- `pegasus_service_sensitivity_protocol.py`：服务压力与镜像大小实验。

## 本轮最重要的回归测试

- `test_dag_completion_semantics.py`：非零到达、单出口兼容和多出口终止。
- `test_critical_path_reward.py`：奖励平移不变性和响应时间望远镜性质。
- `test_information_protocol.py`：未来信息与特权状态审计。
- `test_capacity_protocol.py`：K=0/1/2 和容量随机分配。

历史 HCPR、BCR、quantile risk head、熵正则和 guidance gate 代码因审计需要仍可能存在，
但不属于 `lean_our` 的正式贡献。Pro 应以 `run_reproduction_suite.py` 中 `lean_our` 的实际
配置判断启用模块，而不能仅根据文件存在与否推断。
