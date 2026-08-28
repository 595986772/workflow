# 论文关键实验结果

## 1. 主协议

| 项目 | 设置 |
|---|---|
| 数据 | Pegasus 五类工作流，24--30 个真实任务/DAG |
| 基础设施 | 20 用户，10 服务器，10 服务，15 kHz |
| 容量 | `[0,0,0,0,1,1,1,1,2,2]`，总缓存预算 8 |
| 容量分配 | 按 seed 确定性打乱，与算力和位置独立 |
| 正式 seeds | 51--60，配对最终确认，不称独立 holdout |
| 评估 | 每 seed 100 个完全配对场景，每类工作流 20 个 |
| 主目标 | 平均 DAG makespan；P95 作尾时延辅助指标 |

## 2. 正式方法命名

| 论文名称 | 内部标签 | 身份 |
|---|---|---|
| Random | `random` | 随机卸载 + 独立标准缓存 |
| Nearest | `nearest` | 最近服务器 + 独立标准缓存 |
| SA-Nearest | `greedy` | 服务感知最近启发式 |
| DQN-NoDSA | `dqn_wdsa_std_cache` | 无 DAOC 引导的基础 DQN |
| SAC + DAOC-Cache | `discrete_sac_std_cache` | 离散 SAC + DAOC 式独立缓存 |
| DAOC | `daoc_paper` | 原 DAOC 对比方法 |
| DAOC + DCC | `daoc_our_coord_cache` | DAOC 卸载器 + 本文 DCC，受控混合方法 |
| SAC + DCC | `coord_cache_discrete_sac` | 离散 SAC + 本文 DCC，同缓存强基线 |
| OUR | `lean_our` | Pairwise PD3QN + DCC |

## 3. 主对比

| 方法 | 平均完成时间 (s) | P95 (s) | OUR 平均时延改善 | OUR P95 改善 |
|---|---:|---:|---:|---:|
| Random | 1.8290 | 2.1189 | 85.41% | 79.54% |
| Nearest | 1.5859 | 2.1285 | 83.18% | 79.63% |
| SA-Nearest | 1.6462 | 2.1863 | 83.79% | 80.17% |
| DQN-NoDSA | 1.1308 | 1.4698 | 76.41% | 70.50% |
| SAC + DAOC-Cache | 1.1357 | 1.4511 | 76.51% | 70.12% |
| DAOC | 0.7736 | 1.0610 | 65.51% | 59.14% |
| DAOC + DCC | 0.3773 | 0.5750 | 29.29% | 24.60% |
| SAC + DCC | 0.3001 | 0.4463 | 11.09% | 2.86% |
| **OUR** | **0.2668** | **0.4336** | -- | -- |

关键显著性：

- OUR vs DAOC：10/10 seed，95% CI 下界 0.3906 s，`p=0.00098`。
- OUR vs DAOC + DCC：10/10 seed，95% CI `[0.0785, 0.1425]` s，`p=0.00098`。
- OUR vs SAC + DCC 的平均时延：10/10 seed，95% CI `[0.0175, 0.0491]` s，`p=0.00098`。
- OUR vs SAC + DCC 的 P95：仅 6/10 seed，95% CI `[-0.0222, 0.0478]` s，`p=0.2158`，不显著。

## 4. 2x2 因子证据

| 卸载器 | DAOC-Cache/独立缓存 (s) | DCC (s) |
|---|---:|---:|
| Flat DDQN | 1.0387 | 0.3579 |
| Pairwise PD3QN | 0.7373 | 0.2668 |

- DCC 对 Flat DDQN 改善 65.54%，10/10 seed，`p=0.00098`。
- DCC 对 Pairwise PD3QN 改善 63.82%，10/10 seed，`p=0.00098`。
- Pairwise PD3QN 在独立缓存下改善 29.01%，9/10 seed，`p=0.00195`。
- Pairwise PD3QN 在 DCC 下改善 25.46%，10/10 seed，`p=0.00098`。

## 5. 机制消融

| 方法 | 平均完成时间 (s) | 完整 OUR 改善 | 配对结论 |
|---|---:|---:|---|
| OUR-noTaskDependency | 0.2981 | 10.49% | 10/10，`p=0.00098`，支持 |
| OUR-noDependencyCache | 0.2725 | 2.08% | 5/10，`p=0.42285`，不支持 |
| OUR-TerminalReward | 0.3101 | 13.97% | 8/10，`p=0.00488`，支持 |
| OUR | 0.2668 | -- | 完整方法 |

论文贡献列表中应保留任务依赖状态和因果 makespan-increment 奖励；缓存评分中的依赖加权不应被声称为已显著验证的单独创新。

## 6. 缓存机制

| 方法 | 命中率 | 服务覆盖率 | 远程加载率 |
|---|---:|---:|---:|
| SA-Nearest | 0.4206 | 0.39 | 0.5794 |
| DAOC | 0.1815 | 0.56 | 0.8185 |
| SAC + DAOC-Cache | 0.2385 | 0.46 | 0.7615 |
| DAOC + DCC | 0.3062 | 0.80 | 0.6938 |
| SAC + DCC | 0.4212 | 0.80 | 0.5788 |
| OUR | **0.5889** | **0.80** | **0.4111** |

正确叙事是：DCC 先把服务覆盖提高到总预算允许的水平，Pairwise PD3QN 再使卸载动作与副本布局更匹配，因而 OUR 在相同 0.80 覆盖率下比 DAOC + DCC 和 SAC + DCC 获得更高命中率和更低远程加载。

## 7. 统一 26,000 轮训练预算

| 方法 | 训练尾部平均 (s) | 冻结评估 (s) |
|---|---:|---:|
| DQN-NoDSA | 1.0639 | 1.1308 |
| SAC + DAOC-Cache | 1.5292 | 1.1615 |
| DAOC | 0.7079 | 0.7736 |
| DAOC + DCC | 0.2726 | 0.3773 |
| SAC + DCC | 0.4883 | 0.3057 |
| OUR | **0.2223** | **0.2634** |

这一组用于收敛与统一训练预算说明，与第 3 节 P6/P7/P8 主结果口径分开报告。

## 8. 跨工作流与跨数据集

OUR 在 CyberShake、Epigenomics、Inspiral、Montage 和 Sipht 五类工作流上的平均完成时间分别为 0.1164、0.3076、0.1718、0.4937 和 0.2444 s，均低于图中其他正式方法。

Alibaba-CP100 受控零样本压力测试中，OUR 为 1.1989 s，SAC + DCC 为 1.2601 s，DAOC + DCC 为 1.4526 s，DAOC 为 2.1019 s。该数据集经过选择，只能称受控跨数据集压力测试，不能称无偏 holdout。

## 9. 系统规模扩展

下表为 3 seed 平均；服务器数 5/10/15/20 时，缓存总预算为 4/8/12/16，因此表达按比例扩容。

| 服务器 | Random | Nearest | SA-Nearest | DAOC | DDQN + DCC | SAC + DCC | OUR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2.3147 | 2.2990 | 2.2764 | 1.1170 | 1.0737 | 1.0514 | 1.0549 |
| 10 | 1.9422 | 1.8982 | 1.9581 | 0.8078 | 0.4489 | 0.3770 | 0.3547 |
| 15 | 1.8726 | 1.7288 | 1.9501 | 0.5564 | 0.2882 | 0.2392 | 0.1991 |
| 20 | 1.7651 | 1.7216 | 1.5991 | 0.6016 | 0.2932 | 0.2297 | 0.1809 |

## 10. 服务压力与镜像大小

该组仅为 3 seed 趋势证据。

| 活跃服务数 | OUR (s) | 相对 DAOC | 相对 DDQN + DCC | 相对 SAC + DCC |
|---:|---:|---:|---:|---:|
| 4 | 0.1004 | 60.65% | 37.30% | 19.99% |
| 6 | 0.1382 | 73.93% | 28.87% | 12.43% |
| 8 | 0.1337 | 84.36% | 46.09% | 33.12% |
| 10 | 0.3547 | 56.09% | 20.98% | 5.92% |

| 镜像大小 | OUR (s) | 相对 DAOC | 相对 DDQN + DCC | 相对 SAC + DCC |
|---:|---:|---:|---:|---:|
| 0.5x | 0.2195 | 50.62% | 18.36% | 4.26% |
| 1x | 0.3567 | 55.74% | 21.42% | 5.52% |
| 2x | 0.6308 | 60.37% | 22.87% | 6.83% |
| 4x | 1.1776 | 63.71% | 26.18% | 7.34% |

## 11. 开销与 Oracle

- OUR 冻结推理约 0.193 ms/任务决策，DAOC 约 0.0217 ms/决策。
- OUR 的平均绝对 Oracle gap 约 0.2249 s，说明仍存在优化空间。
- 该 Oracle 为容量感知诊断参考，不是已认证的理论下界。
- 冻结评估不执行在线缓存更新，所以其中缓存决策时间为 0，不能解释为实际协调开销为 0。

## 12. 数据来源索引

- 主对比、2x2 和消融：`results/pegasus_pscale/p6_baselines_ablation/analysis/`
- 标准缓存 SAC：`results/pegasus_pscale/p7_std_cache_discrete_sac/analysis/`
- DAOC + DCC：`results/pegasus_pscale/p8_daoc_our_coord_cache/analysis/`
- 统一 26,000 轮：`results/pegasus_pscale/p9_common_horizon_26k/analysis/`
- Alibaba-CP100：`results/pegasus_pscale/p10_alibaba_cp100_cross_dataset/`
- 服务器规模：`results/pegasus_pscale/p13_server_scaling_heuristics/analysis/`
- 服务压力：`results/pegasus_pscale/p14_service_sensitivity/analysis/`
- 直接作图 CSV：`figure_data/`
