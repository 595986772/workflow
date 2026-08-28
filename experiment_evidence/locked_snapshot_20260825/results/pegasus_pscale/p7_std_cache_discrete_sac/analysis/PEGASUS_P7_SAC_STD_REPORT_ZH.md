# Pegasus-B8 标准缓存 Discrete SAC 补充实验

DiscreteSAC-StdCache 与 CoordCache-DiscreteSAC 的状态、因果稠密奖励、离散动作空间和训练超参数完全相同，只将协调缓存换为逐服务器独立 popularity EMA 缓存。全部结果使用 seeds 51–60，每个 seed 100 个完全配对场景。

| 方法 | 平均完成时间 (s) | P95 (s) | 命中率 | 覆盖率 | 远程加载率 |
|---|---:|---:|---:|---:|---:|
| DAOC-paper | 0.7736 | 1.0610 | 0.1815 | 0.5600 | 0.8185 |
| DiscreteSAC-StdCache | 1.1357 | 1.4511 | 0.2385 | 0.4600 | 0.7615 |
| CoordCache-DiscreteSAC | 0.3001 | 0.4463 | 0.4212 | 0.8000 | 0.5788 |
| OUR | 0.2668 | 0.4336 | 0.5889 | 0.8000 | 0.4111 |

## 配对统计

- CoordCache-SAC 相对 StdCache-SAC: 73.58% (10/10 seeds, 95% CI [0.6030, 1.0681] s, p=0.00098); passes.
- OUR 相对 StdCache-SAC: 76.51% (10/10 seeds, 95% CI [0.6374, 1.1004] s, p=0.00098); passes.
- OUR 相对 CoordCache-SAC: 11.09% (10/10 seeds, 95% CI [0.0175, 0.0491] s, p=0.00098); passes.
- P95：OUR 相对 StdCache-SAC: 70.12% (10/10 seeds, 95% CI [0.7451, 1.2901] s, p=0.00098); passes.
- P95：OUR 相对 CoordCache-SAC: 2.86% (6/10 seeds, 95% CI [-0.0222, 0.0478] s, p=0.21582); does not pass.

Seeds 51–60 属于配对最终确认，不称为独立 holdout。DiscreteSAC-StdCache 是当前项目中的受控基线，不声称为某篇外部 SAC 论文的逐字复现。
