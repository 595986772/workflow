# 最新证据索引

## 正式主结果

| 协议 | 作用 | 权威文件 |
|---|---|---|
| P6 | 主基线、2x2 因子与机制消融 | `results/pegasus_pscale/p6_baselines_ablation/analysis/pegasus_p6_summary.json` |
| P7 | SAC + DAOC-Cache | `results/pegasus_pscale/p7_std_cache_discrete_sac/analysis/sac_std_cache_extension_summary.json` |
| P8 | DAOC + DCC | `results/pegasus_pscale/p8_daoc_our_coord_cache/analysis/daoc_coord_cache_extension_summary.json` |
| P9 | 统一 26,000 轮收敛轨迹 | `results/pegasus_pscale/p9_common_horizon_26k/analysis/common_horizon_summary.json` |

P6/P7/P8 冻结评估是性能主口径；P9 只用于收敛过程。配对统计的单位是 seed，不是
单个 DAG 场景或训练 episode。

## 扩展与机制证据

| 协议 | 作用 | 边界 |
|---|---|---|
| P10 | Alibaba-CP100 受控跨数据集测试 | 不是无偏 holdout |
| P11--P13 | 服务器 5/10/15/20 与预算 4/8/12/16 | 是基础设施同比扩展，不是纯服务器数敏感性 |
| P14 | 活跃服务数与镜像大小 | 三 seed 趋势，不作显著性声称 |
| P15-H | H0--H3 固定 B=8 容量异构度 | 三 seed 趋势，不声称单调改善 |
| P15-L | 关键路径时延构成修复 | 避免并行分支重复累加 |

## 正式图数据

`figure_data/` 仅保留当前正式图的 CSV。图文用途、协议来源和允许结论见
`../05_FORMAL_FIGURES/FIGURE_MANIFEST.md` 与 `FIGURE_USAGE_ZH.md`。历史合并图、Central-Greedy 和人工延长训练曲线
不进入终稿。

