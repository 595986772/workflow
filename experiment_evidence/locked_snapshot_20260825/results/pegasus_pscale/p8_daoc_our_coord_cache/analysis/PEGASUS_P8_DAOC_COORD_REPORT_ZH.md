# Pegasus-B8 DAOC + OUR 中央协调缓存受控实验

DAOC+OUR-CoordCache 完整保留 DAOC 的状态、DQN 调度器、衰减动作引导、终止奖励和收敛超参数，只将 DAOC 独立缓存替换为 OUR 的完整中央协调缓存。

| 方法 | 平均完成时间 (s) | P95 (s) | 命中率 | 覆盖率 | 远程加载率 |
|---|---:|---:|---:|---:|---:|
| DAOC-paper | 0.7736 | 1.0610 | 0.1815 | 0.5600 | 0.8185 |
| DAOC+OUR-CoordCache | 0.3773 | 0.5750 | 0.3062 | 0.8000 | 0.6938 |
| DiscreteSAC-StdCache | 1.1357 | 1.4511 | 0.2385 | 0.4600 | 0.7615 |
| CoordCache-DiscreteSAC | 0.3001 | 0.4463 | 0.4212 | 0.8000 | 0.5788 |
| OUR | 0.2668 | 0.4336 | 0.5889 | 0.8000 | 0.4111 |

## 配对统计

- DAOC+CoordCache versus DAOC: 51.23% (10/10 seeds, 95% CI [0.2703, 0.5223] s, p=0.00098); passes.
- OUR versus DAOC+CoordCache: 29.29% (10/10 seeds, 95% CI [0.0785, 0.1425] s, p=0.00098); passes.
- CoordCache-SAC versus DAOC+CoordCache: 20.47% (10/10 seeds, 95% CI [0.0537, 0.1007] s, p=0.00098); passes.
- P95: DAOC+CoordCache versus DAOC: 45.81% (10/10 seeds, 95% CI [0.3331, 0.6390] s, p=0.00098); passes.
- P95: OUR versus DAOC+CoordCache: 24.60% (9/10 seeds, 95% CI [0.0785, 0.2043] s, p=0.00195); passes.

所有结果使用 seeds 51–60，每个 seed 100 个完全配对冻结场景。这些 seed 属于配对最终确认，不称为独立 holdout。
