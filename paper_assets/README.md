# Paper Assets

本目录保存当前稿件实际使用或继续编辑所必需的图形资产。正式图片与明确标注的排版预览分目录管理，不纳入候选图标库和废弃版本。

## Architecture

- `architecture/hero_dag_system_model_polished_v8_scheduler_gap_plus30.svg`：第一章当前使用的独立SVG。
- `architecture/hero_dag_system_model_polished_v8_scheduler_gap_plus30_editable.drawio`：对应的可编辑Draw.io源文件。

Draw.io文件已内嵌11个图像对象，未引用本机文件路径，因此单独打开即可继续编辑。独立候选图标库没有上传，因为它不是复现当前图所必需的依赖。

`architecture/icons/`进一步保存从Draw.io中提取的6个唯一原始资产，包括5个SVG和1个PNG；`manifest.json`记录11个图像对象与这些去重资产的对应关系。运行`architecture/extract_embedded_icons.py`可以从当前Draw.io源文件重新生成它们。

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

## Validation-Curve Layout Preview

[2026-09-08 v3 验证曲线预览与说明](previews/validation_20260908_v3/README.md)包含 PNG、SVG、PDF、图注、正文说明及源数据。本版图例使用 `DQN+DAOC-Cache`，OUR 和 SAC + DCC 的参考延长段沿用前段标记样式。

该预览包含非实测的水平参考延长段，不替换上表中的正式图7。其学习曲线取自每seed50个固定验证场景，不应与主结果柱状图的100个最终评估场景混同；完整使用边界见预览说明。
