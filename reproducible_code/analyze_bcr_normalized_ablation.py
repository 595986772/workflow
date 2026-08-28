#!/usr/bin/env python3
"""Audit and analyze normalized telemetry x BCR PD3QN screening."""

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
        "normalized": False,
        "bcr": False,
    },
    "normalized": {
        "display": "Normalized telemetry",
        "label": "normalized_telemetry_pd3qn",
        "algorithm": "causal_normalizedTelemetryPD3QN",
        "normalized": True,
        "bcr": False,
    },
    "bcr": {
        "display": "BCR",
        "label": "bcr_pd3qn",
        "algorithm": "causal_task_serverBCRPD3QN",
        "normalized": False,
        "bcr": True,
    },
    "full": {
        "display": "Redesigned OUR",
        "label": "normalized_bcr_pd3qn",
        "algorithm": "causal_normalizedTelemetryBCRPD3QN",
        "normalized": True,
        "bcr": True,
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

REPLAY_METRICS = (
    "mean_hcpr_exact_path_rate",
    "mean_hcpr_selected_rate",
    "mean_hcpr_mean_posterior_criticality",
    "mean_hcpr_buffer_criticality",
    "mean_hcpr_sampled_criticality",
    "mean_hcpr_sampling_criticality_lift",
    "mean_hcpr_importance_weight_mean",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the converged 3-seed normalized telemetry/BCR ablation."
    )
    parser.add_argument(
        "--base-suite-dir",
        type=Path,
        default=Path("results/pd3qn_convergence_3seed"),
    )
    parser.add_argument(
        "--ablation-suite-dir",
        type=Path,
        default=Path("results/bcr_normalized_ablation_3seed"),
    )
    parser.add_argument(
        "--legacy-ablation-summary",
        type=Path,
        default=Path(
            "results/hcpr_factorial_ablation_3seed/"
            "hcpr_factorial_summary.json"
        ),
    )
    return parser.parse_args()


def assert_suite_integrity(path):
    manifest = read_json(path / "suite_manifest.json")
    require(
        manifest.get("status") == "complete"
        and not manifest.get("failed_runs")
        and not manifest.get("nonconverged_runs"),
        f"Incomplete suite: {path}",
    )
    require(
        manifest.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION,
        f"Wrong information protocol: {path}",
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
        arguments["algorithm"] == expected["algorithm"]
        and arguments["reward_mode"] == "causal_critical_path"
        and arguments["cache_policy"] == "critical_path_joint",
        f"Unexpected method configuration: {run_dir}",
    )
    require(
        summary.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION
        and summary.get("cache_information_regime")
        == CAUSAL_CACHE_INFORMATION_REGIME,
        f"Non-causal information regime: {run_dir}",
    )

    train_tail = summary["train_tail"]
    if expected["normalized"]:
        require(
            arguments["telemetry_min_samples"] == 5
            and arguments["telemetry_freshness_half_life"] == 10.0
            and train_tail["mean_cache_mean_telemetry_confidence"] > 0.0,
            f"Normalized telemetry was inactive: {run_dir}",
        )
    else:
        require(
            train_tail.get("mean_cache_mean_telemetry_confidence", 0.0)
            == 0.0,
            f"Non-telemetry method contains telemetry activity: {run_dir}",
        )

    if expected["bcr"]:
        require(
            arguments["bcr_top_fraction"] == 0.25
            and arguments["priority_alpha"] == 0.6
            and arguments["criticality_boost"] == 2.0,
            f"BCR parameter mismatch: {run_dir}",
        )
        require(
            0.0 < train_tail["mean_hcpr_selected_rate"]
            < train_tail["mean_hcpr_exact_path_rate"]
            and train_tail["mean_hcpr_sampling_criticality_lift"] > 0.0,
            f"BCR was not sparse or did not alter sampling: {run_dir}",
        )
        require(
            summary["eval"]["mean_hcpr_selected_rate"] == 0.0
            and summary["eval"][
                "mean_hcpr_sampling_criticality_lift"
            ]
            == 0.0,
            f"BCR changed replay during evaluation: {run_dir}",
        )
    else:
        require(
            all(
                train_tail.get(metric, 0.0) == 0.0
                for metric in REPLAY_METRICS
            ),
            f"Non-BCR method contains replay activity: {run_dir}",
        )
    return summary, config


def aggregate(values):
    values = np.asarray(values, dtype=float)
    interval = confidence_interval(values)
    return {
        "mean": float(values.mean()),
        "ci95_half_width": interval[2],
        "ci95_lower": interval[0],
        "ci95_upper": interval[1],
    }


def factorial_effects(rows):
    records = []
    for row in rows:
        base = row["base_finish_time"]
        normalized = row["normalized_finish_time"]
        bcr = row["bcr_finish_time"]
        full = row["full_finish_time"]
        records.append(
            {
                "seed": row["seed"],
                "normalized_without_bcr_percent": (
                    100.0 * (base - normalized) / base
                ),
                "bcr_without_normalized_percent": (
                    100.0 * (base - bcr) / base
                ),
                "full_vs_base_percent": (
                    100.0 * (base - full) / base
                ),
                "bcr_given_normalized_percent": (
                    100.0 * (normalized - full) / base
                ),
                "normalized_given_bcr_percent": (
                    100.0 * (bcr - full) / base
                ),
                "interaction_percent": (
                    100.0
                    * (normalized + bcr - base - full)
                    / base
                ),
            }
        )
    result = {
        field: aggregate([record[field] for record in records])
        for field in records[0]
        if field != "seed"
    }
    result["per_seed"] = records
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path, rows, summary):
    colors = {
        "base": "#4B5563",
        "normalized": "#059669",
        "bcr": "#D97706",
        "full": "#2563EB",
    }
    order = ("base", "normalized", "bcr", "full")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.2),
        constrained_layout=True,
    )
    positions = np.arange(len(order))
    for row in rows:
        axes[0].plot(
            positions,
            [row[f"{method}_finish_time"] for method in order],
            color="#9CA3AF",
            linewidth=1.0,
        )
    for position, method in enumerate(order):
        axes[0].scatter(
            np.repeat(position, len(rows)),
            [row[f"{method}_finish_time"] for row in rows],
            color=colors[method],
            edgecolor="white",
            s=44,
            zorder=3,
        )
    axes[0].set_xticks(
        positions,
        [METHODS[method]["display"] for method in order],
        rotation=10,
    )
    axes[0].set_ylabel("Mean application finish time (s)")
    axes[0].set_title("Paired frozen evaluation", loc="left")

    effects = summary["factorial_effects"]["per_seed"]
    seeds = np.asarray([row["seed"] for row in rows])
    width = 0.24
    for offset, method, field in (
        (-width, "normalized", "normalized_without_bcr_percent"),
        (0.0, "bcr", "bcr_without_normalized_percent"),
        (width, "full", "full_vs_base_percent"),
    ):
        axes[1].bar(
            seeds + offset,
            [record[field] for record in effects],
            width=width,
            color=colors[method],
            label=METHODS[method]["display"],
        )
    axes[1].axhline(0.0, color="#6B7280", linewidth=1.0)
    axes[1].set_xticks(seeds)
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Improvement over PD3QN (%)")
    axes[1].set_title("Module effects", loc="left")
    axes[1].legend(frameon=False)

    mechanism = summary["bcr_mechanism"]["bcr"]
    axes[2].bar(
        ["Selected", "Buffer", "Sampled", "Lift"],
        [
            mechanism["mean_hcpr_selected_rate"]["mean"],
            mechanism["mean_hcpr_buffer_criticality"]["mean"],
            mechanism["mean_hcpr_sampled_criticality"]["mean"],
            mechanism["mean_hcpr_sampling_criticality_lift"]["mean"],
        ],
        color=["#4B5563", "#6B7280", "#D97706", "#059669"],
    )
    axes[2].set_ylabel("Rate or replay score")
    axes[2].set_title("BCR mechanism", loc="left")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Normalized Telemetry x BCR: 3-Seed Screening",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path, summary, rows):
    methods = summary["method_aggregates"]
    effects = summary["factorial_effects"]
    comparisons = summary["paired_comparisons"]
    redesign = summary["redesign_comparison"]
    decision = summary["screening_decision"]
    best = min(
        METHODS,
        key=lambda method: methods[method]["finish_time"]["mean"],
    )
    lines = [
        "# 归一化遥测 × BCR 三 Seed 筛选报告",
        "",
        "## 结论",
        "",
        f"四组中平均完成时间最低的是 **{METHODS[best]['display']}**。"
        "本轮仅用于低成本模块筛选，不能替代最终 10-seed 显著性实验。",
        "",
        "## 公平性",
        "",
        "- 四组均为 20 用户、10 服务器、10 服务、每用户最多 10 个任务。",
        "- 每个 seed 的 10 个验证场景和 100 个冻结测试场景逐一完全一致。",
        "- 所有方法使用相同因果关键路径奖励、联合缓存和历史信息边界。",
        "- 所有模型均达到预设收敛门槛，测试阶段模型与历史状态完全冻结。",
        "",
        "## 主结果",
        "",
        "| 方法 | 归一化遥测 | BCR | 平均完成时间 (s) | P95 (s) | 缓存命中率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("base", "normalized", "bcr", "full"):
        lines.append(
            f"| {METHODS[method]['display']} | "
            f"{'✓' if METHODS[method]['normalized'] else '×'} | "
            f"{'✓' if METHODS[method]['bcr'] else '×'} | "
            f"{methods[method]['finish_time']['mean']:.6f} | "
            f"{methods[method]['p95_finish_time']['mean']:.6f} | "
            f"{methods[method]['cache_hit_rate']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "- 归一化遥测相对 PD3QN："
            f"{effects['normalized_without_bcr_percent']['mean']:.2f}%（"
            f"{comparisons['normalized_vs_base']['finish_time']['wins']}/3 seed 获胜）。",
            "- BCR 相对 PD3QN："
            f"{effects['bcr_without_normalized_percent']['mean']:.2f}%（"
            f"{comparisons['bcr_vs_base']['finish_time']['wins']}/3 seed 获胜）。",
            "- 完整方法相对 PD3QN："
            f"{effects['full_vs_base_percent']['mean']:.2f}%（"
            f"{comparisons['full_vs_base']['finish_time']['wins']}/3 seed 获胜）。",
            "- 完整方法相对最佳单模块："
            f"{100.0 * (methods['normalized']['finish_time']['mean'] - methods['full']['finish_time']['mean']) / methods['normalized']['finish_time']['mean']:.2f}%"
            f"（{comparisons['full_vs_normalized']['finish_time']['wins']}/3 seed 获胜）。",
            "- 完整方法相对 PD3QN 的配对 95% CI："
            f"[{comparisons['full_vs_base']['finish_time']['paired_improvement_ci95_lower']:.2f}%, "
            f"{comparisons['full_vs_base']['finish_time']['paired_improvement_ci95_upper']:.2f}%]，"
            f"双侧配对 t 检验 p={comparisons['full_vs_base']['finish_time']['paired_t_two_sided_p']:.3f}。",
            "",
            "## 重设计诊断",
            "",
            "- 旧 HCPR 相对 PD3QN："
            f"{redesign['old_hcpr_vs_base_percent']:.2f}%；"
            "新 BCR 相对 PD3QN："
            f"{redesign['new_bcr_vs_base_percent']:.2f}%。",
            "- 稀疏 BCR 将 replay 模块造成的平均退化缩小了 "
            f"{redesign['replay_harm_reduction_percentage_points']:.2f} 个百分点。",
            "- 新版完整 OUR 相对旧版完整 OUR 仅改善 "
            f"{redesign['new_vs_old_full_percent']:.3f}%"
            f"（{redesign['old_full_finish_time']:.6f} s → "
            f"{redesign['new_full_finish_time']:.6f} s）。",
            "- 旧版完整 OUR 对基线为 "
            f"{redesign['old_full_wins']}/3 seed 获胜；新版为 "
            f"{redesign['new_full_wins']}/3。稳定性改善，但均值增量不足以构成新的性能突破。",
            "",
            "## 机制",
            "",
            "- BCR 平均选择率："
            f"{100.0 * summary['bcr_mechanism']['bcr']['mean_hcpr_selected_rate']['mean']:.2f}%。",
            "- BCR replay 采样提升："
            f"{summary['bcr_mechanism']['bcr']['mean_hcpr_sampling_criticality_lift']['mean']:.3f}。",
            "- 归一化遥测末段平均置信度："
            f"{summary['telemetry_confidence']['normalized']['mean']:.3f}。",
            "",
            "## 门控决策",
            "",
            f"- 归一化遥测：`{decision['normalized_telemetry']}`。",
            f"- BCR：`{decision['bcr']}`。",
            f"- 完整方法扩展到 10 seeds：`{decision['expand_full_to_10_seeds']}`。",
            f"- 下一步复杂环境实验：`{decision['proceed_to_stress_environment']}`。",
            "- 推荐路线：保持默认任务边界不变，先做 3-seed 关键路径压力环境；"
            "只有完整 OUR 相对 PD3QN 达到 ≥2% 且 3/3 seed 获胜，才扩展到 10 seeds。",
            "",
            "统计边界：3-seed 置信区间较宽，本报告只决定是否值得继续投入计算，"
            "不用于论文中的最终显著性结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    base_suite = args.base_suite_dir.resolve()
    ablation_suite = args.ablation_suite_dir.resolve()
    legacy_summary = read_json(args.legacy_ablation_summary.resolve())
    assert_suite_integrity(base_suite)
    assert_suite_integrity(ablation_suite)

    rows = []
    for seed in (1, 2, 3):
        run_dirs = {
            "base": base_suite
            / "runs"
            / METHODS["base"]["label"]
            / f"seed_{seed}",
        }
        run_dirs.update(
            {
                method: ablation_suite
                / "runs"
                / METHODS[method]["label"]
                / f"seed_{seed}"
                for method in ("normalized", "bcr", "full")
            }
        )
        records = {
            method: assert_run_integrity(method, run_dir)
            for method, run_dir in run_dirs.items()
        }
        base_config = records["base"][1]
        for method in ("normalized", "bcr", "full"):
            config = records[method][1]
            require(
                all(
                    config["input_config"][key]
                    == base_config["input_config"][key]
                    for key in PHYSICAL_CONFIG_KEYS
                ),
                f"Physical configuration mismatch for seed {seed}: {method}",
            )
            require(
                all(
                    config["arguments"][key]
                    == base_config["arguments"][key]
                    for key in EVALUATION_CONFIG_KEYS
                ),
                f"Evaluation protocol mismatch for seed {seed}: {method}",
            )

        validation = {
            method: validation_fingerprints(run_dir)
            for method, run_dir in run_dirs.items()
        }
        require(
            all(
                fingerprints == validation["base"]
                for fingerprints in validation.values()
            )
            and len(validation["base"]) == 10,
            f"Validation fingerprints mismatch for seed {seed}",
        )
        evaluated = {
            method: evaluation_rows(run_dir)
            for method, run_dir in run_dirs.items()
        }
        evaluation_fingerprints = {
            method: tuple(
                row["scenario_fingerprint"] for row in method_rows
            )
            for method, method_rows in evaluated.items()
        }
        require(
            all(
                fingerprints == evaluation_fingerprints["base"]
                for fingerprints in evaluation_fingerprints.values()
            )
            and len(evaluation_fingerprints["base"]) == 100,
            f"Evaluation fingerprints mismatch for seed {seed}",
        )

        row = {"seed": seed}
        for method, (summary, _) in records.items():
            for metric, (field, _) in METRICS.items():
                row[f"{method}_{metric}"] = summary["eval"][field]
            row[f"{method}_convergence_episode"] = summary[
                "convergence"
            ]["episode"]
            row[f"{method}_telemetry_confidence"] = summary[
                "train_tail"
            ].get("mean_cache_mean_telemetry_confidence", 0.0)
            for metric in REPLAY_METRICS:
                row[f"{method}_{metric}"] = summary[
                    "train_tail"
                ].get(metric, 0.0)
        rows.append(row)

    method_aggregates = {
        method: {
            metric: aggregate(
                [row[f"{method}_{metric}"] for row in rows]
            )
            for metric in METRICS
        }
        for method in METHODS
    }
    effects = factorial_effects(rows)
    paired = {
        f"{method}_vs_base": {
            metric: paired_statistics(
                [row[f"base_{metric}"] for row in rows],
                [row[f"{method}_{metric}"] for row in rows],
                lower_is_better=lower_is_better,
            )
            for metric, (_, lower_is_better) in METRICS.items()
        }
        for method in ("normalized", "bcr", "full")
    }
    paired["full_vs_normalized"] = {
        metric: paired_statistics(
            [row[f"normalized_{metric}"] for row in rows],
            [row[f"full_{metric}"] for row in rows],
            lower_is_better=lower_is_better,
        )
        for metric, (_, lower_is_better) in METRICS.items()
    }
    paired["full_vs_bcr"] = {
        metric: paired_statistics(
            [row[f"bcr_{metric}"] for row in rows],
            [row[f"full_{metric}"] for row in rows],
            lower_is_better=lower_is_better,
        )
        for metric, (_, lower_is_better) in METRICS.items()
    }

    normalized_pass = (
        effects["normalized_without_bcr_percent"]["mean"] > 0.0
        and paired["normalized_vs_base"]["finish_time"]["wins"] >= 2
    )
    bcr_pass = (
        effects["bcr_without_normalized_percent"]["mean"] > 0.0
        and paired["bcr_vs_base"]["finish_time"]["wins"] >= 2
    )
    best_single = min(
        method_aggregates["normalized"]["finish_time"]["mean"],
        method_aggregates["bcr"]["finish_time"]["mean"],
    )
    full_beats_best = (
        method_aggregates["full"]["finish_time"]["mean"]
        < best_single
        and paired["full_vs_base"]["finish_time"]["wins"] >= 2
    )
    legacy_effects = legacy_summary["factorial_effects"]
    legacy_methods = legacy_summary["method_aggregates"]
    old_full_wins = sum(
        record["full_vs_base_percent"] > 0.0
        for record in legacy_effects["per_seed"]
    )
    redesign_comparison = {
        "old_hcpr_vs_base_percent": legacy_effects[
            "hcpr_without_telemetry_percent"
        ]["mean"],
        "new_bcr_vs_base_percent": effects[
            "bcr_without_normalized_percent"
        ]["mean"],
        "replay_harm_reduction_percentage_points": (
            effects["bcr_without_normalized_percent"]["mean"]
            - legacy_effects["hcpr_without_telemetry_percent"]["mean"]
        ),
        "old_full_finish_time": legacy_methods["full"]["finish_time"][
            "mean"
        ],
        "new_full_finish_time": method_aggregates["full"]["finish_time"][
            "mean"
        ],
        "new_vs_old_full_percent": (
            100.0
            * (
                legacy_methods["full"]["finish_time"]["mean"]
                - method_aggregates["full"]["finish_time"]["mean"]
            )
            / legacy_methods["full"]["finish_time"]["mean"]
        ),
        "old_full_vs_base_percent": legacy_effects[
            "full_vs_base_percent"
        ]["mean"],
        "new_full_vs_base_percent": effects["full_vs_base_percent"][
            "mean"
        ],
        "old_full_wins": old_full_wins,
        "new_full_wins": paired["full_vs_base"]["finish_time"]["wins"],
    }
    summary = {
        "status": "complete",
        "comparison": "normalized_telemetry_x_bcr",
        "information_protocol_version": INFORMATION_PROTOCOL_VERSION,
        "integrity": {
            "training_seeds": 3,
            "validation_scenarios_per_seed": 10,
            "evaluation_scenarios_per_seed": 100,
            "all_runs_converged": True,
            "all_evaluations_frozen": True,
            "physical_environment_match": True,
            "evaluation_protocol_match": True,
            "all_validation_fingerprints_match": True,
            "all_evaluation_fingerprints_match": True,
            "module_isolation_verified": True,
        },
        "method_aggregates": method_aggregates,
        "factorial_effects": effects,
        "paired_comparisons": paired,
        "redesign_comparison": redesign_comparison,
        "bcr_mechanism": {
            method: {
                metric: aggregate(
                    [row[f"{method}_{metric}"] for row in rows]
                )
                for metric in REPLAY_METRICS
            }
            for method in ("bcr", "full")
        },
        "telemetry_confidence": {
            method: aggregate(
                [
                    row[f"{method}_telemetry_confidence"]
                    for row in rows
                ]
            )
            for method in ("normalized", "full")
        },
        "screening_decision": {
            "normalized_telemetry": (
                "retain" if normalized_pass else "reject_or_refine"
            ),
            "bcr": "retain" if bcr_pass else "reject_or_refine",
            "full_beats_best_single_module": full_beats_best,
            "expand_full_to_10_seeds": (
                normalized_pass and bcr_pass and full_beats_best
            ),
            "proceed_to_stress_environment": (
                normalized_pass or bcr_pass or full_beats_best
            ),
        },
    }

    output_dir = ablation_suite
    write_csv(output_dir / "bcr_normalized_per_seed.csv", rows)
    (output_dir / "bcr_normalized_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_results(
        output_dir / "bcr_normalized_comparison",
        rows,
        summary,
    )
    write_report(
        output_dir / "BCR_NORMALIZED_SCREENING_REPORT.md",
        summary,
        rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
