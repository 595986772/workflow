#!/usr/bin/env python3
"""Audit and analyze the 2x2 telemetry/HCPR PD3QN ablation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    "base": {
        "display": "PD3QN",
        "label": "pd3qn",
        "algorithm": "causal_task_serverPD3QN",
        "telemetry": False,
        "hcpr": False,
    },
    "telemetry": {
        "display": "Telemetry-only",
        "label": "telemetry_pd3qn",
        "algorithm": "causal_telemetryPD3QN",
        "telemetry": True,
        "hcpr": False,
    },
    "hcpr": {
        "display": "HCPR-only",
        "label": "hcpr_pd3qn",
        "algorithm": "causal_task_serverHCPRPD3QN",
        "telemetry": False,
        "hcpr": True,
    },
    "full": {
        "display": "OUR",
        "label": "hcpr_telemetry_pd3qn",
        "algorithm": "causal_telemetryHCPRPD3QN",
        "telemetry": True,
        "hcpr": True,
    },
}

METRICS = {
    "finish_time": ("mean_average_finish_time", True),
    "p95_finish_time": ("mean_p95_finish_time", True),
    "predecessor_latency": ("mean_predecessor_latency", True),
    "computing_latency": ("mean_computing_latency", True),
    "data_transfer_latency": ("mean_data_transfer_latency", True),
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
        description="Analyze the converged 3-seed 2x2 HCPR ablation."
    )
    parser.add_argument(
        "--base-suite-dir",
        type=Path,
        default=Path("results/pd3qn_convergence_3seed"),
    )
    parser.add_argument(
        "--ablation-suite-dir",
        type=Path,
        default=Path("results/hcpr_factorial_ablation_3seed"),
    )
    parser.add_argument(
        "--full-suite-dir",
        type=Path,
        default=Path("results/hcpr_telemetry_calibration_3seed"),
    )
    return parser.parse_args()


def assert_suite_integrity(suite_dir):
    manifest = read_json(suite_dir / "suite_manifest.json")
    require(
        manifest.get("status") == "complete"
        and not manifest.get("failed_runs")
        and not manifest.get("nonconverged_runs"),
        f"Incomplete suite: {suite_dir}",
    )
    require(
        manifest.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION,
        f"Wrong information protocol: {suite_dir}",
    )


def assert_run_integrity(method, run_dir):
    expected = METHODS[method]
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "config.json")
    arguments = config["arguments"]
    require(
        summary.get("status") == "complete"
        and summary.get("eligible_for_comparison") is True
        and summary["convergence"]["reached"] is True,
        f"Incomplete or non-converged run: {run_dir}",
    )
    require(
        summary.get("evaluation_state_frozen") is True
        and arguments["eval_update_caching"] is False,
        f"Evaluation was not frozen: {run_dir}",
    )
    require(
        arguments["algorithm"] == expected["algorithm"],
        f"Unexpected algorithm: {run_dir}",
    )
    require(
        arguments["reward_mode"] == "causal_critical_path"
        and arguments["cache_policy"] == "critical_path_joint",
        f"Reward or cache policy mismatch: {run_dir}",
    )
    require(
        summary.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION
        and summary.get("cache_information_regime")
        == CAUSAL_CACHE_INFORMATION_REGIME,
        f"Non-causal information regime: {run_dir}",
    )
    if expected["hcpr"]:
        require(
            arguments["priority_alpha"] == 0.6
            and arguments["priority_beta_start"] == 0.4
            and arguments["priority_beta_anneal_steps"] == 2000
            and arguments["criticality_boost"] == 2.0
            and arguments["hcpr_temperature"] == 0.05,
            f"HCPR parameter mismatch: {run_dir}",
        )
        require(
            summary["train_tail"][
                "mean_hcpr_sampling_criticality_lift"
            ]
            > 0.0,
            f"HCPR did not alter replay sampling: {run_dir}",
        )
        require(
            summary["eval"]["mean_hcpr_exact_path_rate"] == 0.0
            and summary["eval"][
                "mean_hcpr_sampling_criticality_lift"
            ]
            == 0.0,
            f"HCPR labels changed during evaluation: {run_dir}",
        )
    else:
        require(
            all(
                summary["train_tail"].get(field, 0.0) == 0.0
                for field in HCPR_METRICS
            ),
            f"Non-HCPR method contains HCPR activity: {run_dir}",
        )
    return summary, config


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values):
    values = np.asarray(values, dtype=float)
    interval = confidence_interval(values)
    return {
        "mean": float(values.mean()),
        "ci95_half_width": interval[2],
        "ci95_lower": interval[0],
        "ci95_upper": interval[1],
    }


def method_aggregate(rows, method):
    result = {
        metric: aggregate(
            [row[f"{method}_{metric}"] for row in rows]
        )
        for metric in METRICS
    }
    for field in (
        "convergence_episode",
        "total_wall_time_sec",
        "eval_episode_wall_time_sec",
    ):
        result[field] = aggregate(
            [row[f"{method}_{field}"] for row in rows]
        )
    return result


def factorial_effects(rows):
    per_seed = []
    for row in rows:
        base = row["base_finish_time"]
        telemetry = row["telemetry_finish_time"]
        hcpr = row["hcpr_finish_time"]
        full = row["full_finish_time"]
        per_seed.append(
            {
                "seed": row["seed"],
                "telemetry_without_hcpr_percent": (
                    100.0 * (base - telemetry) / base
                ),
                "hcpr_without_telemetry_percent": (
                    100.0 * (base - hcpr) / base
                ),
                "full_vs_base_percent": (
                    100.0 * (base - full) / base
                ),
                "hcpr_given_telemetry_percent": (
                    100.0 * (telemetry - full) / base
                ),
                "telemetry_given_hcpr_percent": (
                    100.0 * (hcpr - full) / base
                ),
                "interaction_percent": (
                    100.0
                    * (telemetry + hcpr - base - full)
                    / base
                ),
            }
        )
    result = {
        field: aggregate([row[field] for row in per_seed])
        for field in per_seed[0]
        if field != "seed"
    }
    result["per_seed"] = per_seed
    return result


def mechanism_aggregate(rows, method):
    return {
        field: aggregate([row[f"{method}_{field}"] for row in rows])
        for field in HCPR_METRICS
    }


def component_effects(summary):
    methods = summary["method_aggregates"]
    result = {}
    for method in ("telemetry", "hcpr", "full"):
        result[method] = {}
        for metric in METRICS:
            base = methods["base"][metric]["mean"]
            candidate = methods[method][metric]["mean"]
            lower_is_better = METRICS[metric][1]
            direction = 1.0 if lower_is_better else -1.0
            result[method][metric] = {
                "absolute_difference": candidate - base,
                "improvement_percent": (
                    100.0
                    * direction
                    * (base - candidate)
                    / base
                ),
            }
    return result


def plot_results(path, rows, summary):
    colors = {
        "base": "#4B5563",
        "telemetry": "#059669",
        "hcpr": "#D97706",
        "full": "#2563EB",
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9.0),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")

    order = ("base", "telemetry", "hcpr", "full")
    positions = np.arange(len(order))
    for row in rows:
        axes[0, 0].plot(
            positions,
            [row[f"{method}_finish_time"] for method in order],
            color="#9CA3AF",
            linewidth=1.1,
        )
    for position, method in enumerate(order):
        axes[0, 0].scatter(
            np.repeat(position, len(rows)),
            [row[f"{method}_finish_time"] for row in rows],
            color=colors[method],
            edgecolor="white",
            linewidth=0.6,
            s=42,
            zorder=3,
        )
    axes[0, 0].set_xticks(
        positions,
        [METHODS[method]["display"] for method in order],
        rotation=8,
    )
    axes[0, 0].set_ylabel("Mean application finish time (s)")
    axes[0, 0].set_title("Paired held-out performance", loc="left")

    effects = summary["factorial_effects"]["per_seed"]
    seeds = np.asarray([row["seed"] for row in rows])
    width = 0.24
    axes[0, 1].bar(
        seeds - width,
        [
            effect["telemetry_without_hcpr_percent"]
            for effect in effects
        ],
        width=width,
        color=colors["telemetry"],
        label="Telemetry-only",
    )
    axes[0, 1].bar(
        seeds,
        [
            effect["hcpr_without_telemetry_percent"]
            for effect in effects
        ],
        width=width,
        color=colors["hcpr"],
        label="HCPR-only",
    )
    axes[0, 1].bar(
        seeds + width,
        [effect["full_vs_base_percent"] for effect in effects],
        width=width,
        color=colors["full"],
        label="Full OUR",
    )
    axes[0, 1].axhline(0.0, color="#6B7280", linewidth=1.0)
    axes[0, 1].set_xticks(seeds)
    axes[0, 1].set_xlabel("Training seed")
    axes[0, 1].set_ylabel("Improvement over PD3QN (%)")
    axes[0, 1].set_title("Per-seed module effects", loc="left")
    axes[0, 1].legend(frameon=False)

    for method, marker in (("base", "o"), ("hcpr", "s")):
        without = summary["method_aggregates"][method][
            "finish_time"
        ]["mean"]
        with_telemetry_method = (
            "telemetry" if method == "base" else "full"
        )
        with_telemetry = summary["method_aggregates"][
            with_telemetry_method
        ]["finish_time"]["mean"]
        axes[1, 0].plot(
            [0, 1],
            [without, with_telemetry],
            marker=marker,
            linewidth=2.0,
            markersize=7,
            color=colors[method],
            label=(
                "HCPR off" if method == "base" else "HCPR on"
            ),
        )
    axes[1, 0].set_xticks([0, 1], ["Telemetry off", "Telemetry on"])
    axes[1, 0].set_ylabel("Mean application finish time (s)")
    axes[1, 0].set_title("2x2 interaction", loc="left")
    axes[1, 0].legend(frameon=False)

    mechanism = summary["hcpr_mechanism"]["hcpr"]
    values = [
        mechanism["mean_hcpr_buffer_criticality"]["mean"],
        mechanism["mean_hcpr_sampled_criticality"]["mean"],
        mechanism["mean_hcpr_sampling_criticality_lift"]["mean"],
    ]
    axes[1, 1].bar(
        ["Buffer", "Sampled", "Lift"],
        values,
        color=["#6B7280", colors["hcpr"], colors["telemetry"]],
    )
    axes[1, 1].set_ylabel("Posterior criticality")
    axes[1, 1].set_title("HCPR-only replay mechanism", loc="left")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Telemetry x HCPR Factorial Ablation: 3 Converged Seeds",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path, summary, rows):
    methods = summary["method_aggregates"]
    effects = summary["factorial_effects"]
    components = summary["component_effects_vs_base"]
    comparisons = summary["paired_comparisons"]
    mechanism = summary["hcpr_mechanism"]
    row_lines = [
        (
            f"| {row['seed']} | {row['base_finish_time']:.6f} | "
            f"{row['telemetry_finish_time']:.6f} | "
            f"{row['hcpr_finish_time']:.6f} | "
            f"{row['full_finish_time']:.6f} | "
            f"{100.0 * (row['base_finish_time'] - row['telemetry_finish_time']) / row['base_finish_time']:.2f}% | "
            f"{100.0 * (row['base_finish_time'] - row['hcpr_finish_time']) / row['base_finish_time']:.2f}% |"
        )
        for row in rows
    ]
    text = "\n".join(
        [
            "# Telemetry × HCPR 2×2 消融实验报告",
            "",
            "## 结论",
            "",
            "本轮筛选结果不支持把当前 HCPR 作为独立有效创新点。"
            "Telemetry-only 是四组中平均完成时间最低的方法；HCPR-only "
            "在全部三个 seed 上退化；完整 OUR 没有超过 Telemetry-only。",
            "",
            "## 公平性",
            "",
            "- 四组均为 20 用户、10 服务器、10 服务、每用户最多 10 个任务。",
            "- 每个 seed 的 10 个验证场景和 100 个冻结测试场景指纹在四组间"
            "逐一完全一致。",
            "- 四组均使用 causal critical-path reward、critical-path joint "
            "cache 和 causal history-only information protocol。",
            "- 所有 12 个模型结果均达到预设收敛门槛；本轮四组 seeds 1–3 "
            "均在 7,500 轮停止。",
            "- 测试阶段模型、replay、epsilon、学习率、缓存和历史状态全部冻结。",
            "",
            "## 主结果",
            "",
            "| 方法 | Telemetry | HCPR | 平均完成时间 (s) | P95 (s) | 缓存命中率 |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                (
                    f"| {METHODS[method]['display']} | "
                    f"{'✓' if METHODS[method]['telemetry'] else '×'} | "
                    f"{'✓' if METHODS[method]['hcpr'] else '×'} | "
                    f"{methods[method]['finish_time']['mean']:.6f} | "
                    f"{methods[method]['p95_finish_time']['mean']:.6f} | "
                    f"{methods[method]['cache_hit_rate']['mean']:.4f} |"
                )
                for method in ("base", "telemetry", "hcpr", "full")
            ],
            "",
            "- Telemetry-only 相对 PD3QN 平均改善 "
            f"{effects['telemetry_without_hcpr_percent']['mean']:.2f}%，"
            f"{comparisons['telemetry_vs_base']['finish_time']['wins']}/3 "
            "seed 获胜。",
            "- HCPR-only 相对 PD3QN 平均"
            f"{'改善' if effects['hcpr_without_telemetry_percent']['mean'] >= 0 else '退化'} "
            f"{abs(effects['hcpr_without_telemetry_percent']['mean']):.2f}%，"
            f"{comparisons['hcpr_vs_base']['finish_time']['wins']}/3 seed 获胜。",
            "- 完整 OUR 相对 PD3QN 平均改善 "
            f"{effects['full_vs_base_percent']['mean']:.2f}%，但相对 "
            "Telemetry-only 平均"
            f"{'改善' if effects['hcpr_given_telemetry_percent']['mean'] >= 0 else '退化'} "
            f"{abs(effects['hcpr_given_telemetry_percent']['mean']):.2f}%。",
            "- 交互项为 "
            f"{effects['interaction_percent']['mean']:.2f}%。这是补偿性交互："
            "遥测将 HCPR-only 的明显退化救回到接近 Telemetry-only，"
            "但组合并没有优于最佳单模块，不能表述为有价值的协同增益。",
            "- 统计边界：本轮仅有 3 个 seed，定位是低成本模块筛选。"
            "Telemetry 主效应的 95% CI 为 "
            f"[{effects['telemetry_without_hcpr_percent']['ci95_lower']:.2f}%, "
            f"{effects['telemetry_without_hcpr_percent']['ci95_upper']:.2f}%]，"
            "配对双侧 t 检验 "
            f"p={comparisons['telemetry_vs_base']['finish_time']['paired_t_two_sided_p']:.3f}；"
            "因此当前只能决定保留该方向，不能声称已经取得统计显著提升。",
            "",
            "## 组件诊断",
            "",
            "- Telemetry-only 相对基础 PD3QN：computing latency 改善 "
            f"{components['telemetry']['computing_latency']['improvement_percent']:.2f}%，"
            "waiting latency 改善 "
            f"{components['telemetry']['waiting_latency']['improvement_percent']:.2f}%，"
            "service latency 改善 "
            f"{components['telemetry']['service_latency']['improvement_percent']:.2f}%，"
            "缓存命中率提高 "
            f"{100.0 * components['telemetry']['cache_hit_rate']['absolute_difference']:.2f} "
            "个百分点。",
            "- HCPR-only 相对基础 PD3QN：computing latency 退化 "
            f"{abs(components['hcpr']['computing_latency']['improvement_percent']):.2f}%，"
            "service latency 退化 "
            f"{abs(components['hcpr']['service_latency']['improvement_percent']):.2f}%，"
            "predecessor latency 退化 "
            f"{abs(components['hcpr']['predecessor_latency']['improvement_percent']):.2f}%，"
            "缓存命中率下降 "
            f"{abs(100.0 * components['hcpr']['cache_hit_rate']['absolute_difference']):.2f} "
            "个百分点。",
            "- 因此 HCPR 的主要问题不是代码未生效，而是改变 replay 分布后"
            "学出了更差的计算服务器和服务局部性决策。",
            "",
            "## HCPR 机制",
            "",
            "- HCPR-only 的真实关键路径平均覆盖 "
            f"{100.0 * mechanism['hcpr']['mean_hcpr_exact_path_rate']['mean']:.2f}% "
            "任务。",
            "- replay buffer 关键度均值为 "
            f"{mechanism['hcpr']['mean_hcpr_buffer_criticality']['mean']:.3f}，"
            "采样均值为 "
            f"{mechanism['hcpr']['mean_hcpr_sampled_criticality']['mean']:.3f}，"
            "采样提升为 "
            f"{mechanism['hcpr']['mean_hcpr_sampling_criticality_lift']['mean']:.3f}。",
            "- 机制确实提高了关键路径样本概率，但约三分之二任务都被标为"
            "后验关键路径任务，标签过宽，无法区分路径中真正的瓶颈任务。",
            "",
            "## 逐 Seed",
            "",
            "| Seed | PD3QN | Telemetry-only | HCPR-only | OUR | Telemetry 效应 | HCPR 效应 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *row_lines,
            "",
            "## 决策",
            "",
            "1. 保留因果历史遥测方向。它在当前筛选中贡献了主要收益，但后续仍应"
            "把 raw execution-latency EMA 改成 workload-normalized telemetry，"
            "减少任务混合偏差。",
            "2. 不把当前 HCPR 版本扩到 10 seeds，也不把它作为论文中已经成立的"
            "第二创新点。",
            "3. 将 HCPR 改为 bottleneck-contribution replay：后验路径任务的"
            "权重还要乘以其 realized local latency / path makespan，并只增强"
            "每个 DAG 中贡献最高的一小部分任务，避免 66% 任务一起被放大。",
            "4. 新 HCPR 先做相同 seeds 1–3 的低成本筛选；只有平均超过 "
            "Telemetry-only 且至少 2/3 seed 获胜，再扩到 10 seeds。",
            "",
            "## 文件",
            "",
            "- `hcpr_factorial_per_seed.csv`",
            "- `hcpr_factorial_summary.json`",
            "- `hcpr_factorial_comparison.png` 和 `.pdf`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main():
    args = parse_args()
    suite_for_method = {
        "base": args.base_suite_dir.resolve(),
        "telemetry": args.ablation_suite_dir.resolve(),
        "hcpr": args.ablation_suite_dir.resolve(),
        "full": args.full_suite_dir.resolve(),
    }
    for suite_dir in set(suite_for_method.values()):
        assert_suite_integrity(suite_dir)

    rows = []
    for seed in (1, 2, 3):
        artifacts = {}
        for method, expected in METHODS.items():
            run_dir = (
                suite_for_method[method]
                / "runs"
                / expected["label"]
                / f"seed_{seed}"
            )
            summary, config = assert_run_integrity(method, run_dir)
            artifacts[method] = {
                "summary": summary,
                "config": config,
                "validation": validation_fingerprints(run_dir),
                "eval_rows": evaluation_rows(run_dir),
            }

        reference = artifacts["base"]
        reference_eval = tuple(
            row["scenario_fingerprint"]
            for row in reference["eval_rows"]
        )
        for method in ("telemetry", "hcpr", "full"):
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
                f"Validation fingerprints mismatch: {method}, seed {seed}",
            )
            require(
                reference_eval
                == tuple(
                    row["scenario_fingerprint"]
                    for row in candidate["eval_rows"]
                ),
                f"Evaluation fingerprints mismatch: {method}, seed {seed}",
            )
        require(
            len(reference["validation"]) == 10
            and len(reference_eval) == 100,
            f"Unexpected scenario-bank size for seed {seed}",
        )

        row = {
            "seed": seed,
            "validation_fingerprints_match": 1,
            "evaluation_fingerprints_match": 1,
        }
        for method in METHODS:
            method_summary = artifacts[method]["summary"]
            for metric, (field, _) in METRICS.items():
                row[f"{method}_{metric}"] = method_summary["eval"][
                    field
                ]
            row[f"{method}_convergence_episode"] = method_summary[
                "convergence"
            ]["actual_train_episodes"]
            row[f"{method}_total_wall_time_sec"] = method_summary[
                "total_wall_time_sec"
            ]
            row[
                f"{method}_eval_episode_wall_time_sec"
            ] = method_summary["eval"]["mean_episode_wall_time_sec"]
            for field in HCPR_METRICS:
                row[f"{method}_{field}"] = method_summary[
                    "train_tail"
                ].get(field, 0.0)
        rows.append(row)

    paired_comparisons = {}
    for method in ("telemetry", "hcpr", "full"):
        paired_comparisons[f"{method}_vs_base"] = {
            metric: paired_statistics(
                [row[f"base_{metric}"] for row in rows],
                [row[f"{method}_{metric}"] for row in rows],
                lower_is_better=lower_is_better,
            )
            for metric, (_, lower_is_better) in METRICS.items()
        }
    paired_comparisons["full_vs_telemetry"] = {
        metric: paired_statistics(
            [row[f"telemetry_{metric}"] for row in rows],
            [row[f"full_{metric}"] for row in rows],
            lower_is_better=lower_is_better,
        )
        for metric, (_, lower_is_better) in METRICS.items()
    }

    summary = {
        "status": "complete",
        "comparison": "telemetry_x_hcpr_factorial_ablation",
        "information_protocol_version": INFORMATION_PROTOCOL_VERSION,
        "integrity": {
            "training_seeds": 3,
            "validation_scenarios_per_seed": 10,
            "evaluation_scenarios_per_seed": 100,
            "all_runs_converged": True,
            "all_convergence_episodes_equal": all(
                row[f"{method}_convergence_episode"] == 7500
                for row in rows
                for method in METHODS
            ),
            "all_evaluations_frozen": True,
            "physical_environment_match": True,
            "evaluation_protocol_match": True,
            "all_validation_fingerprints_match": True,
            "all_evaluation_fingerprints_match": True,
            "module_isolation_verified": True,
        },
        "method_aggregates": {
            method: method_aggregate(rows, method)
            for method in METHODS
        },
        "factorial_effects": factorial_effects(rows),
        "paired_comparisons": paired_comparisons,
        "hcpr_mechanism": {
            method: mechanism_aggregate(rows, method)
            for method in ("hcpr", "full")
        },
        "screening_decision": {
            "telemetry": "retain_and_refine",
            "current_hcpr": "reject_and_redesign",
            "full_vs_best_single_module": "does_not_improve",
            "expand_current_hcpr_to_10_seeds": False,
        },
    }
    summary["component_effects_vs_base"] = component_effects(summary)

    output_dir = args.ablation_suite_dir.resolve()
    write_csv(output_dir / "hcpr_factorial_per_seed.csv", rows)
    (output_dir / "hcpr_factorial_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_results(
        output_dir / "hcpr_factorial_comparison",
        rows,
        summary,
    )
    write_report(
        output_dir / "HCPR_FACTORIAL_ABLATION_REPORT.md",
        summary,
        rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
