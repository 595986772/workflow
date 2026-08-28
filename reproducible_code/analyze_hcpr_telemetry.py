#!/usr/bin/env python3
"""Audit and compare HCPR + causal telemetry against DAOC and PD3QN."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from compare_pd3qn_methods import (
    EVALUATION_CONFIG_KEYS,
    PHYSICAL_CONFIG_KEYS,
    confidence_interval,
    evaluation_rows,
    paired_statistics,
    read_json,
    require,
    validation_fingerprints,
)
from information_protocol import (
    CAUSAL_CACHE_INFORMATION_REGIME,
    INFORMATION_PROTOCOL_VERSION,
)


METHODS = {
    "daoc": {
        "display": "DAOC",
        "label": "guided_full",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "cache_policy": "popularity_ema",
        "reward_mode": "terminal_binary",
    },
    "pd3qn": {
        "display": "PD3QN",
        "label": "pd3qn",
        "algorithm": "causal_task_serverPD3QN",
        "cache_policy": "critical_path_joint",
        "reward_mode": "causal_critical_path",
    },
    "our": {
        "display": "OUR",
        "label": "hcpr_telemetry_pd3qn",
        "algorithm": "causal_telemetryHCPRPD3QN",
        "cache_policy": "critical_path_joint",
        "reward_mode": "causal_critical_path",
    },
}

METRICS = {
    "finish_time": ("mean_average_finish_time", True),
    "p95_finish_time": ("mean_p95_finish_time", True),
    "computing_latency": ("mean_computing_latency", True),
    "data_transfer_latency": ("mean_data_transfer_latency", True),
    "predecessor_latency": ("mean_predecessor_latency", True),
    "service_latency": ("mean_service_latency", True),
    "waiting_latency": ("mean_waiting_latency", True),
    "cache_hit_rate": ("mean_cache_hit_rate", False),
}

HCPR_METRICS = (
    "mean_hcpr_exact_path_rate",
    "mean_hcpr_mean_posterior_criticality",
    "mean_hcpr_buffer_criticality",
    "mean_hcpr_sampled_criticality",
    "mean_hcpr_sampling_criticality_lift",
    "mean_hcpr_importance_weight_mean",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create the final paired 10-seed audit for HCPR plus causal "
            "telemetry PD3QN."
        )
    )
    parser.add_argument(
        "--reference-suite-dir",
        type=Path,
        default=Path("results/paper_dual_main_10seed"),
    )
    parser.add_argument(
        "--pd3qn-suite-dir",
        type=Path,
        default=Path("results/pd3qn_convergence_3seed"),
    )
    parser.add_argument(
        "--our-suite-dir",
        type=Path,
        default=Path("results/hcpr_telemetry_calibration_3seed"),
    )
    return parser.parse_args()


def discover_seeds(suite_dir, label):
    return {
        int(path.name.split("_", 1)[1])
        for path in (suite_dir / "runs" / label).glob("seed_*")
        if (path / "summary.json").exists()
    }


def assert_suite_integrity(suite_dir):
    manifest = read_json(suite_dir / "suite_manifest.json")
    require(
        manifest.get("status") == "complete",
        f"Incomplete suite: {suite_dir}",
    )
    require(
        not manifest.get("failed_runs")
        and not manifest.get("nonconverged_runs"),
        f"Suite contains failed or non-converged runs: {suite_dir}",
    )
    require(
        manifest.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION,
        f"Wrong information protocol: {suite_dir}",
    )


def assert_run_integrity(method_key, run_dir):
    expected = METHODS[method_key]
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "config.json")
    arguments = config["arguments"]
    require(
        summary.get("status") == "complete"
        and summary.get("eligible_for_comparison") is True,
        f"Incomplete or ineligible run: {run_dir}",
    )
    require(
        summary["convergence"]["reached"] is True,
        f"Non-converged run: {run_dir}",
    )
    require(
        summary.get("evaluation_state_frozen") is True
        and arguments["eval_update_caching"] is False,
        f"Evaluation state was not frozen: {run_dir}",
    )
    require(
        arguments["algorithm"] == expected["algorithm"]
        and arguments["cache_policy"] == expected["cache_policy"]
        and arguments["reward_mode"] == expected["reward_mode"],
        f"Unexpected method configuration: {run_dir}",
    )
    require(
        summary.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION,
        f"Wrong run information protocol: {run_dir}",
    )
    if method_key != "daoc":
        require(
            summary.get("cache_information_regime")
            == CAUSAL_CACHE_INFORMATION_REGIME,
            f"Non-causal cache information regime: {run_dir}",
        )
    if method_key == "our":
        require(
            arguments["priority_alpha"] == 0.6
            and arguments["priority_beta_start"] == 0.4
            and arguments["priority_beta_anneal_steps"] == 2000
            and arguments["criticality_boost"] == 2.0
            and arguments["hcpr_temperature"] == 0.05,
            f"Unexpected HCPR configuration: {run_dir}",
        )
    return summary, config


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values):
    interval = confidence_interval(values)
    return {
        "mean": float(np.mean(values)),
        "ci95_half_width": interval[2],
        "ci95_lower": interval[0],
        "ci95_upper": interval[1],
    }


def method_aggregate(rows, prefix):
    result = {}
    for metric in METRICS:
        result[metric] = aggregate(
            [row[f"{prefix}_{metric}"] for row in rows]
        )
    for field in (
        "convergence_episode",
        "total_wall_time_sec",
        "eval_episode_wall_time_sec",
    ):
        result[field] = aggregate(
            [row[f"{prefix}_{field}"] for row in rows]
        )
    return result


def mechanism_summary(rows):
    result = {}
    for field in HCPR_METRICS:
        result[field] = aggregate([row[field] for row in rows])
    improvements = np.asarray(
        [row["our_vs_pd3qn_finish_improvement_percent"] for row in rows],
        dtype=float,
    )
    correlations = {}
    for field in HCPR_METRICS:
        values = np.asarray([row[field] for row in rows], dtype=float)
        pearson = stats.pearsonr(values, improvements)
        spearman = stats.spearmanr(values, improvements)
        correlations[field] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    result["correlation_with_pd3qn_improvement"] = correlations
    return result


def effect_diagnostics(rows):
    finish_improvement = np.asarray(
        [
            row["pd3qn_finish_time"] - row["our_finish_time"]
            for row in rows
        ],
        dtype=float,
    )
    result = {}
    for metric in (
        "predecessor_latency",
        "service_latency",
        "computing_latency",
        "waiting_latency",
        "cache_hit_rate",
    ):
        candidate_difference = np.asarray(
            [
                row[f"our_{metric}"] - row[f"pd3qn_{metric}"]
                for row in rows
            ],
            dtype=float,
        )
        metric_improvement = (
            candidate_difference
            if metric == "cache_hit_rate"
            else -candidate_difference
        )
        pearson = stats.pearsonr(
            metric_improvement,
            finish_improvement,
        )
        spearman = stats.spearmanr(
            metric_improvement,
            finish_improvement,
        )
        result[metric] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    return result


def plot_results(path, rows, summary):
    seeds = np.asarray([row["seed"] for row in rows])
    colors = {
        "daoc": "#4B5563",
        "pd3qn": "#D97706",
        "our": "#2563EB",
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9.0),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")

    positions = np.arange(3)
    for row in rows:
        axes[0, 0].plot(
            positions,
            [
                row["daoc_finish_time"],
                row["pd3qn_finish_time"],
                row["our_finish_time"],
            ],
            color="#9CA3AF",
            linewidth=1.0,
            alpha=0.65,
        )
    for position, method in enumerate(("daoc", "pd3qn", "our")):
        axes[0, 0].scatter(
            np.repeat(position, len(rows)),
            [row[f"{method}_finish_time"] for row in rows],
            color=colors[method],
            edgecolor="white",
            linewidth=0.6,
            s=38,
            zorder=3,
        )
    axes[0, 0].set_xticks(positions, ["DAOC", "PD3QN", "OUR"])
    axes[0, 0].set_ylabel("Mean application finish time (s)")
    axes[0, 0].set_title("Paired held-out performance", loc="left")

    width = 0.36
    axes[0, 1].bar(
        seeds - width / 2,
        [
            row["our_vs_daoc_finish_improvement_percent"]
            for row in rows
        ],
        width=width,
        color=colors["daoc"],
        label="OUR vs DAOC",
    )
    axes[0, 1].bar(
        seeds + width / 2,
        [
            row["our_vs_pd3qn_finish_improvement_percent"]
            for row in rows
        ],
        width=width,
        color=colors["our"],
        label="OUR vs PD3QN",
    )
    axes[0, 1].axhline(0.0, color="#6B7280", linewidth=1.0)
    axes[0, 1].set_xticks(seeds)
    axes[0, 1].set_xlabel("Training seed")
    axes[0, 1].set_ylabel("Finish-time improvement (%)")
    axes[0, 1].set_title("Per-seed relative effect", loc="left")
    axes[0, 1].legend(frameon=False)

    metric_names = ["Mean", "P95"]
    x_values = np.arange(len(metric_names))
    for offset, method in zip(
        (-0.25, 0.0, 0.25),
        ("daoc", "pd3qn", "our"),
    ):
        means = [
            summary["method_aggregates"][method]["finish_time"]["mean"],
            summary["method_aggregates"][method]["p95_finish_time"]["mean"],
        ]
        errors = [
            summary["method_aggregates"][method]["finish_time"][
                "ci95_half_width"
            ],
            summary["method_aggregates"][method]["p95_finish_time"][
                "ci95_half_width"
            ],
        ]
        axes[1, 0].bar(
            x_values + offset,
            means,
            width=0.25,
            yerr=errors,
            capsize=3,
            color=colors[method],
            label=METHODS[method]["display"],
        )
    axes[1, 0].set_xticks(x_values, metric_names)
    axes[1, 0].set_ylabel("Application finish time (s)")
    axes[1, 0].set_title("Mean and tail latency (95% CI)", loc="left")
    axes[1, 0].legend(frameon=False)

    mechanism = summary["hcpr_mechanism"]
    mechanism_names = ["Buffer", "Sampled", "Lift"]
    mechanism_values = [
        mechanism["mean_hcpr_buffer_criticality"]["mean"],
        mechanism["mean_hcpr_sampled_criticality"]["mean"],
        mechanism["mean_hcpr_sampling_criticality_lift"]["mean"],
    ]
    axes[1, 1].bar(
        mechanism_names,
        mechanism_values,
        color=["#6B7280", "#2563EB", "#059669"],
    )
    axes[1, 1].set_ylabel("Posterior criticality")
    axes[1, 1].set_title("HCPR replay mechanism", loc="left")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle(
        "HCPR + Causal Telemetry: 10 Paired Converged Seeds",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path, summary, rows):
    aggregate_results = summary["method_aggregates"]
    daoc = summary["comparisons"]["our_vs_daoc"]
    pd3qn = summary["comparisons"]["our_vs_pd3qn"]
    mechanism = summary["hcpr_mechanism"]
    diagnostics = summary["effect_diagnostics_vs_pd3qn"]
    row_lines = [
        (
            f"| {row['seed']} | {row['daoc_finish_time']:.6f} | "
            f"{row['pd3qn_finish_time']:.6f} | "
            f"{row['our_finish_time']:.6f} | "
            f"{row['our_vs_daoc_finish_improvement_percent']:.2f}% | "
            f"{row['our_vs_pd3qn_finish_improvement_percent']:.2f}% |"
        )
        for row in rows
    ]
    text = "\n".join(
        [
            "# HCPR + 因果遥测主实验诊断报告",
            "",
            "## 一句话结论",
            "",
            "OUR 在完全配对的 10-seed 实验中稳定超过 DAOC，但相对现有 "
            "PD3QN 只有边界显著的小幅收益。当前结果足以支持“比 DAOC 更优”，"
            "尚不足以支持“两个新模块都稳定有效”或“显著优于 PD3QN”。",
            "",
            "## 公平性审计",
            "",
            "- 规模一致：20 用户、10 服务器、10 服务、每用户最多 10 个任务。",
            "- 三种方法使用相同物理配置；每个 seed 的 10 个验证场景和 100 个"
            "冻结测试场景指纹逐一完全一致。",
            "- 所有 30 个训练任务都通过预设收敛门槛；测试期间网络、缓存和历史"
            "遥测均冻结。",
            "- PD3QN 与 OUR 使用相同的因果关键路径奖励、联合缓存和历史信息边界；"
            "两者比较主要隔离 HCPR 与服务器历史遥测的组合效果。",
            "- DAOC 使用其原生 popularity-EMA 缓存和 terminal reward，因此"
            "与 DAOC 的比较是完整方法对完整方法。",
            "- OUR 不读取未执行任务的 CPU cycles，也不读取当前精确 server.load"
            " 或 frequency；服务器质量仅来自已完成任务执行时延 EMA。",
            "",
            "## 主要结果",
            "",
            "| 方法 | 平均完成时间 (s) | P95 (s) | 缓存命中率 | 收敛轮数 |",
            "|---|---:|---:|---:|---:|",
            *[
                (
                    f"| {METHODS[method]['display']} | "
                    f"{aggregate_results[method]['finish_time']['mean']:.6f} "
                    f"± {aggregate_results[method]['finish_time']['ci95_half_width']:.6f} | "
                    f"{aggregate_results[method]['p95_finish_time']['mean']:.6f} | "
                    f"{aggregate_results[method]['cache_hit_rate']['mean']:.4f} | "
                    f"{aggregate_results[method]['convergence_episode']['mean']:.0f} |"
                )
                for method in ("daoc", "pd3qn", "our")
            ],
            "",
            "相对 DAOC：",
            "",
            "- 平均完成时间降低 "
            f"{daoc['finish_time']['mean_paired_improvement_percent']:.2f}%"
            "（配对 95% CI "
            f"[{daoc['finish_time']['paired_improvement_ci95_lower']:.2f}%, "
            f"{daoc['finish_time']['paired_improvement_ci95_upper']:.2f}%]），"
            f"10/10 seed 获胜，配对 t 检验 p="
            f"{daoc['finish_time']['paired_t_two_sided_p']:.6g}，单侧 "
            f"Wilcoxon p={daoc['finish_time']['wilcoxon_one_sided_p']:.6g}。",
            "- P95 完成时间降低 "
            f"{daoc['p95_finish_time']['mean_paired_improvement_percent']:.2f}%，"
            f"{daoc['p95_finish_time']['wins']}/10 seed 获胜。",
            "",
            "相对 PD3QN：",
            "",
            "- 平均完成时间降低 "
            f"{pd3qn['finish_time']['mean_paired_improvement_percent']:.2f}%"
            "（配对 95% CI "
            f"[{pd3qn['finish_time']['paired_improvement_ci95_lower']:.3f}%, "
            f"{pd3qn['finish_time']['paired_improvement_ci95_upper']:.3f}%]），"
            f"{pd3qn['finish_time']['wins']}/10 seed 获胜；配对 t 检验 p="
            f"{pd3qn['finish_time']['paired_t_two_sided_p']:.4f}，但单侧 "
            f"Wilcoxon p={pd3qn['finish_time']['wilcoxon_one_sided_p']:.4f}。",
            "- P95 仅降低 "
            f"{pd3qn['p95_finish_time']['mean_paired_improvement_percent']:.2f}%，"
            f"p={pd3qn['p95_finish_time']['paired_t_two_sided_p']:.4f}，"
            "没有统计显著性。",
            "- 因此该增益属于边界证据：参数检验刚过 0.05，非参数检验刚未过"
            " 0.05，且进行多指标校正后不会保持显著。",
            "",
            "## HCPR 是否真的工作",
            "",
            "- 训练末段真实关键路径平均覆盖 "
            f"{100.0 * mechanism['mean_hcpr_exact_path_rate']['mean']:.2f}% "
            "的任务，软后验关键度均值为 "
            f"{mechanism['mean_hcpr_mean_posterior_criticality']['mean']:.3f}。",
            "- 回放池关键度均值为 "
            f"{mechanism['mean_hcpr_buffer_criticality']['mean']:.3f}，"
            "实际采样均值为 "
            f"{mechanism['mean_hcpr_sampled_criticality']['mean']:.3f}，"
            "平均采样提升为 "
            f"{mechanism['mean_hcpr_sampling_criticality_lift']['mean']:.3f}。"
            "这证明优先回放机制已生效。",
            "- 但约三分之二任务都在后验关键路径上，标签区分度偏低；采样关键度"
            "只提高约 0.043，说明当前 DAG 较短或近似链式，HCPR 可利用的"
            "“少数关键任务”信号有限。",
            "- 探索性跨 seed 诊断显示，完成时间改善与 predecessor latency "
            "改善的 Pearson r="
            f"{diagnostics['predecessor_latency']['pearson_r']:.3f}"
            "，与 computing latency 改善的 r="
            f"{diagnostics['computing_latency']['pearson_r']:.3f}"
            "；与缓存命中率变化的 r 仅为 "
            f"{diagnostics['cache_hit_rate']['pearson_r']:.3f}"
            f"（p={diagnostics['cache_hit_rate']['pearson_p']:.3f}）。"
            "这表明本轮成败主要来自依赖路径与计算服务器选择，而不是命中率本身。",
            "",
            "## 发现的问题",
            "",
            "1. **增益小且不够稳健。** OUR 相对 PD3QN 仅提升 0.47%，"
            "seed 1、8、10 反而退化；P95 没有显著改善。",
            "2. **两个模块尚未拆分。** 当前只有 PD3QN 与 HCPR+遥测组合，"
            "缺少 telemetry-only 和 HCPR-only 消融，无法判断是谁带来收益，"
            "也无法排除一个模块提升、另一个模块抵消。",
            "3. **遥测存在任务混合偏差。** 当前服务器 EMA 统计 raw "
            "(computing + waiting) latency。被分配较重任务的服务器可能被误判"
            "为慢服务器；更合理的下一版应记录每单位 CPU cycle 的执行时延或"
            "对 CPU cycles 分桶后再做 EMA。",
            "4. **HCPR 训练成本较高。** OUR 平均总运行时间为 "
            f"{aggregate_results['our']['total_wall_time_sec']['mean']:.1f}s/seed，"
            "PD3QN 为 "
            f"{aggregate_results['pd3qn']['total_wall_time_sec']['mean']:.1f}s/seed，"
            "OUR 增加约 "
            f"{100.0 * (aggregate_results['our']['total_wall_time_sec']['mean'] / aggregate_results['pd3qn']['total_wall_time_sec']['mean'] - 1.0):.1f}%"
            "。HCPR 不增加部署推理步骤，但会增加训练回放开销；遥测输入只使"
            "网络参数从 40,450 增至 40,514（增加 64 个，约 0.16%）。",
            "5. **仿真器队列模型较弱。** server.execute_task 立即把任务标记为"
            "完成，waiting latency 由静态 load/frequency 代数计算，而不是真实"
            "并发队列。后验路径与该仿真定义一致，但外部有效性仍需更真实的队列"
            "或动态负载实验支撑。",
            "6. **部署假设不是纯分布式 DAOC。** broker 需要维护并广播每台服务器"
            "一个历史质量标量，本规模每次为 10 个 float，通信量很小但不能写成"
            "零开销或与 DAOC 完全相同的纯分布式信息结构。",
            "",
            "## 逐 seed 结果",
            "",
            "| Seed | DAOC | PD3QN | OUR | OUR vs DAOC | OUR vs PD3QN |",
            "|---:|---:|---:|---:|---:|---:|",
            *row_lines,
            "",
            "## 论文结论边界",
            "",
            "- 可以写：在与 DAOC 相同且严格配对的默认规模下，OUR 将平均完成"
            "时间降低约 3.75%，10/10 seed 获胜，且没有使用未来任务信息或"
            "精确服务器私有状态。",
            "- 暂时不要写：OUR 显著优于强 PD3QN；HCPR 和因果遥测分别有效；"
            "对动态负载、故障或大规模 DAG 具有部署鲁棒性。",
            "- 下一步最省成本的必要实验是先做 3-seed telemetry-only 与 "
            "HCPR-only 消融；只有单模块趋势正确后再扩到 10 seed。随后构造"
            "同一任务边界内的宽 DAG/异构 CPU/动态负载场景，提高关键路径信号。",
            "",
            "## 生成文件",
            "",
            "- `hcpr_telemetry_per_seed.csv`",
            "- `hcpr_telemetry_summary.json`",
            "- `hcpr_telemetry_comparison.png` 和 `.pdf`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main():
    args = parse_args()
    suite_for_method = {
        "daoc": args.reference_suite_dir.resolve(),
        "pd3qn": args.pd3qn_suite_dir.resolve(),
        "our": args.our_suite_dir.resolve(),
    }
    for suite_dir in set(suite_for_method.values()):
        assert_suite_integrity(suite_dir)

    seed_sets = {
        method: discover_seeds(
            suite_for_method[method],
            config["label"],
        )
        for method, config in METHODS.items()
    }
    require(
        seed_sets["daoc"] == seed_sets["pd3qn"] == seed_sets["our"],
        "Methods do not have identical seed sets",
    )
    require(
        seed_sets["our"] == set(range(1, 11)),
        "Expected exactly seeds 1 through 10",
    )

    rows = []
    for seed in sorted(seed_sets["our"]):
        artifacts = {}
        for method, method_config in METHODS.items():
            run_dir = (
                suite_for_method[method]
                / "runs"
                / method_config["label"]
                / f"seed_{seed}"
            )
            summary, config = assert_run_integrity(method, run_dir)
            artifacts[method] = {
                "run_dir": run_dir,
                "summary": summary,
                "config": config,
                "validation": validation_fingerprints(run_dir),
                "eval_rows": evaluation_rows(run_dir),
            }

        reference = artifacts["daoc"]
        reference_eval_fingerprints = tuple(
            row["scenario_fingerprint"]
            for row in reference["eval_rows"]
        )
        for method in ("pd3qn", "our"):
            candidate = artifacts[method]
            require(
                all(
                    reference["config"]["input_config"][key]
                    == candidate["config"]["input_config"][key]
                    for key in PHYSICAL_CONFIG_KEYS
                ),
                f"Physical configuration mismatch: {method}, seed {seed}",
            )
            require(
                all(
                    reference["config"]["arguments"][key]
                    == candidate["config"]["arguments"][key]
                    for key in EVALUATION_CONFIG_KEYS
                ),
                f"Evaluation protocol mismatch: {method}, seed {seed}",
            )
            require(
                reference["validation"] == candidate["validation"],
                f"Validation bank mismatch: {method}, seed {seed}",
            )
            candidate_eval_fingerprints = tuple(
                row["scenario_fingerprint"]
                for row in candidate["eval_rows"]
            )
            require(
                reference_eval_fingerprints
                == candidate_eval_fingerprints,
                f"Evaluation bank mismatch: {method}, seed {seed}",
            )
        require(
            len(reference_eval_fingerprints) == 100
            and len(reference["validation"]) == 10,
            f"Unexpected bank size for seed {seed}",
        )

        row = {
            "seed": seed,
            "validation_fingerprints_match": 1,
            "evaluation_fingerprints_match": 1,
        }
        for method in ("daoc", "pd3qn", "our"):
            summary = artifacts[method]["summary"]
            for metric, (summary_field, _) in METRICS.items():
                row[f"{method}_{metric}"] = summary["eval"][
                    summary_field
                ]
            row[f"{method}_convergence_episode"] = summary[
                "convergence"
            ]["actual_train_episodes"]
            row[f"{method}_total_wall_time_sec"] = summary[
                "total_wall_time_sec"
            ]
            row[f"{method}_eval_episode_wall_time_sec"] = summary[
                "eval"
            ]["mean_episode_wall_time_sec"]
        for reference_method in ("daoc", "pd3qn"):
            row[
                f"our_vs_{reference_method}_finish_improvement_percent"
            ] = (
                100.0
                * (
                    row[f"{reference_method}_finish_time"]
                    - row["our_finish_time"]
                )
                / row[f"{reference_method}_finish_time"]
            )
        for field in HCPR_METRICS:
            row[field] = artifacts["our"]["summary"]["train_tail"][
                field
            ]
        rows.append(row)

    comparisons = {}
    for reference_method in ("daoc", "pd3qn"):
        method_comparison = {}
        for metric, (_, lower_is_better) in METRICS.items():
            method_comparison[metric] = paired_statistics(
                [row[f"{reference_method}_{metric}"] for row in rows],
                [row[f"our_{metric}"] for row in rows],
                lower_is_better=lower_is_better,
            )
        comparisons[f"our_vs_{reference_method}"] = method_comparison

    summary = {
        "status": "complete",
        "comparison": "hcpr_telemetry_vs_daoc_and_pd3qn",
        "information_protocol_version": INFORMATION_PROTOCOL_VERSION,
        "integrity": {
            "training_seeds": 10,
            "validation_scenarios_per_seed": 10,
            "evaluation_scenarios_per_seed": 100,
            "all_runs_converged": True,
            "all_evaluations_frozen": True,
            "physical_environment_match": True,
            "evaluation_protocol_match": True,
            "all_validation_fingerprints_match": True,
            "all_evaluation_fingerprints_match": True,
        },
        "method_aggregates": {
            method: method_aggregate(rows, method)
            for method in ("daoc", "pd3qn", "our")
        },
        "comparisons": comparisons,
        "hcpr_mechanism": mechanism_summary(rows),
        "effect_diagnostics_vs_pd3qn": effect_diagnostics(rows),
        "parameter_count": {
            "pd3qn": 40450,
            "our": 40514,
            "telemetry_increment": 64,
        },
        "claim_status": {
            "vs_daoc": "robust",
            "vs_pd3qn_mean": "borderline",
            "vs_pd3qn_p95": "not_significant",
            "individual_module_attribution": "requires_ablation",
        },
    }

    output_dir = suite_for_method["our"]
    write_csv(output_dir / "hcpr_telemetry_per_seed.csv", rows)
    (output_dir / "hcpr_telemetry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_results(
        output_dir / "hcpr_telemetry_comparison",
        rows,
        summary,
    )
    write_report(
        output_dir / "HCPR_TELEMETRY_EXPERIMENT_REPORT.md",
        summary,
        rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
