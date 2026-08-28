# Paper Assets

本目录只保存当前稿件实际使用或继续编辑所必需的图形资产，不包含旧版本、候选图标库和废弃预览。

## Architecture

- `architecture/hero_dag_system_model_polished_v8_scheduler_gap_plus30.svg`：第一章当前使用的独立SVG。
- `architecture/hero_dag_system_model_polished_v8_scheduler_gap_plus30_editable.drawio`：对应的可编辑Draw.io源文件。

Draw.io文件已内嵌11个图像对象，未引用本机文件路径，因此单独打开即可继续编辑。独立候选图标库没有上传，因为它不是复现当前图所必需的依赖。

## Figures Used by the Current Manuscript

| 论文编号 | 仓库文件 | 用途 |
|---|---|---|
| 图2 | `formal_figures/fig01_main_baselines.svg` | 主环境平均与P95完成时间 |
| 图3 | `formal_figures/fig03a_pegasus_workflows.svg` | 五类Pegasus工作流比较 |
| 图4 | `formal_figures/fig03b_latency_composition.svg` | 关键完成路径时延构成 |
| 图5 | `formal_figures/fig04_ablation.svg` | DCC与Pairwise PD3QN受控比较 |
| 图6 | `formal_figures/fig05b_cache_effectiveness.svg` | 命中率、覆盖率与远程加载率 |
| 图7 | `formal_figures/fig08_convergence.svg` | 26000轮在线训练轨迹 |
| 图8 | `formal_figures/fig10_server_count_sensitivity_7methods.svg` | 基础设施同比扩展 |
| 图9 | `formal_figures/fig_cache_capacity_heterogeneity.svg` | 固定预算容量异构性 |

作图脚本位于 `reproducible_code/`，锁定统计与源CSV位于 `experiment_evidence/locked_snapshot_20260825/`。`metadata/FIGURE_PROVENANCE.json`保留实验图的原始来源索引，架构图以本文件列出的v8版本为准。
