#!/usr/bin/env python3
"""Audit, summarize and plot the P14 three-seed sensitivity results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_line_style as line_style
from evaluate_pegasus_service_size_sensitivity import multiplier_token
from pegasus_service_sensitivity_protocol import (
    ACTIVE_SERVICE_COUNTS,
    DAOC,
    DISPLAY_NAMES as INTERNAL_DISPLAY_NAMES,
    DQN_COORD_CACHE,
    METHODS,
    OUR,
    RESULT_ROOT,
    SEEDS,
    SERVICE_SIZE_MULTIPLIERS,
    active_service_run,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = RESULT_ROOT / "analysis"
FIGURE_DIR = ROOT / "paper_drafts/figures_topconf"
STYLE_PATH = (
    ROOT.parent
    / ".codex/skills/scientific-visualization/assets/publication.mplstyle"
)
DISPLAY_NAMES = {
    **INTERNAL_DISPLAY_NAMES,
    "greedy": "SA-Nearest",
    DQN_COORD_CACHE: "DDQN + DCC",
    "coord_cache_discrete_sac": "SAC + DCC",
    OUR: "OUR",
}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def service_size_run(multiplier: float, method: str, seed: int) -> Path:
    return (
        RESULT_ROOT
        / "service_size"
        / multiplier_token(multiplier)
        / "runs"
        / method
        / f"seed_{seed}"
    )


def metric_record(summary: dict) -> dict:
    evaluation = summary["eval"]
    return {
        "mean_finish_time": float(evaluation["mean_average_finish_time"]),
        "p95_finish_time": float(evaluation["mean_p95_finish_time"]),
        "cache_hit_rate": float(evaluation["mean_cache_hit_rate"]),
        "remote_loading_rate": float(
            evaluation["mean_cache_remote_loading_rate"]
        ),
        "service_latency": float(evaluation["mean_service_latency"]),
        "waiting_latency": float(evaluation["mean_waiting_latency"]),
    }


def collect_rows() -> tuple[list[dict], list[dict]]:
    active_rows = []
    for active_services in ACTIVE_SERVICE_COUNTS:
        for method in METHODS:
            for seed in SEEDS:
                summary = read_json(
                    active_service_run(active_services, method, seed)
                    / "summary.json"
                )
                active_rows.append(
                    {
                        "experiment": "active_services",
                        "setting": active_services,
                        "method": method,
                        "display_name": DISPLAY_NAMES[method],
                        "seed": seed,
                        **metric_record(summary),
                    }
                )

    size_rows = []
    for multiplier in SERVICE_SIZE_MULTIPLIERS:
        for method in METHODS:
            for seed in SEEDS:
                summary = read_json(
                    service_size_run(multiplier, method, seed)
                    / "summary.json"
                )
                size_rows.append(
                    {
                        "experiment": "service_size",
                        "setting": multiplier,
                        "method": method,
                        "display_name": DISPLAY_NAMES[method],
                        "seed": seed,
                        **metric_record(summary),
                    }
                )
    return active_rows, size_rows


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "mean_finish_time",
        "p95_finish_time",
        "cache_hit_rate",
        "remote_loading_rate",
        "service_latency",
        "waiting_latency",
    )
    grouped = {}
    for row in rows:
        grouped.setdefault((row["setting"], row["method"]), []).append(row)
    results = []
    for (setting, method), records in grouped.items():
        result = {
            "setting": setting,
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "seeds": len(records),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in records], dtype=float)
            result[f"mean_{metric}"] = float(values.mean())
            result[f"std_{metric}"] = float(values.std(ddof=1))
            result[f"min_{metric}"] = float(values.min())
            result[f"max_{metric}"] = float(values.max())
        results.append(result)
    return sorted(results, key=lambda row: (float(row["setting"]), METHODS.index(row["method"])))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_for(aggregates: list[dict], method: str):
    return [row for row in aggregates if row["method"] == method]


def configure_style() -> None:
    line_style.configure_style(STYLE_PATH)


def plot(active_agg: list[dict], size_agg: list[dict]) -> tuple[Path, Path]:
    configure_style()
    mm = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * mm, 78 * mm),
    )
    panels = (
        (
            axes[0],
            active_agg,
            np.asarray(ACTIVE_SERVICE_COUNTS, dtype=float),
            "Number of active services",
            "(a) Cache pressure",
        ),
        (
            axes[1],
            size_agg,
            np.asarray(SERVICE_SIZE_MULTIPLIERS, dtype=float),
            "Service image size multiplier",
            "(b) Service loading cost",
        ),
    )
    for axis, aggregates, x_values, x_label, title in panels:
        for method in METHODS:
            records = rows_for(aggregates, method)
            means = np.asarray(
                [row["mean_mean_finish_time"] for row in records]
            )
            errors = np.asarray(
                [row["std_mean_finish_time"] for row in records]
            )
            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                **line_style.errorbar_kwargs(
                    method,
                    label=DISPLAY_NAMES[method],
                ),
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Mean DAG completion time (s)")
        axis.set_title(title, loc="left", fontweight="bold", pad=4)
        axis.set_xticks(x_values)
        line_style.style_axis(axis, grid_axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        handlelength=2.35,
        columnspacing=1.0,
        handletextpad=0.55,
        labelspacing=0.45,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.18,
        top=0.78,
        wspace=0.23,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "fig_service_cache_pressure_sensitivity.png"
    pdf = FIGURE_DIR / "fig_service_cache_pressure_sensitivity.pdf"
    svg = FIGURE_DIR / "fig_service_cache_pressure_sensitivity.svg"
    fig.savefig(png, dpi=400, bbox_inches=None, facecolor="white")
    fig.savefig(pdf, bbox_inches=None, facecolor="white")
    fig.savefig(svg, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return png, pdf


def relative_improvement(ours: float, baseline: float) -> float:
    return 100.0 * (baseline - ours) / baseline


def comparison_table(aggregates: list[dict]) -> list[dict]:
    by_key = {
        (row["setting"], row["method"]): row for row in aggregates
    }
    rows = []
    settings = sorted({row["setting"] for row in aggregates})
    for setting in settings:
        ours = by_key[(setting, OUR)]["mean_mean_finish_time"]
        row = {
            "setting": setting,
            "our_mean_finish_time": ours,
        }
        for baseline in METHODS:
            if baseline == OUR:
                continue
            value = by_key[(setting, baseline)]["mean_mean_finish_time"]
            row[f"{baseline}_mean_finish_time"] = value
            row[f"our_vs_{baseline}_improvement_percent"] = (
                relative_improvement(ours, value)
            )
        rows.append(row)
    return rows


def report_markdown(
    active_comparisons: list[dict],
    size_comparisons: list[dict],
    png: Path,
) -> str:
    lines = [
        "# Pegasus 服务压力与镜像大小敏感性实验",
        "",
        "> 该组实验仅使用 3 个独立 seed（51–53），用于描述趋势和机制证据，不单独承担显著性结论。",
        "",
        "## 方法身份",
        "",
        "- `DDQN + DCC` 的内部标签为 `our_flat_ddqn`：普通 Double DQN 调度器 + 本文 DCC，不是 `DAOC + DCC`。",
        "- 所有方法共享同一 DAG、拓扑、服务容量分配和评估 seed。",
        "- 活跃服务数实验对 Q=4/6/8 从头训练，Q=10 复用已审计的收敛 checkpoint。",
        "- 镜像大小实验冻结神经网络，保留每个场景内的原生缓存更新，场景间重置状态以消除顺序污染。",
        "",
        "## 活跃服务数",
        "",
        "| 活跃服务 | OUR (s) | 相对 DAOC | 相对 DDQN + DCC | 相对 SAC + DCC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in active_comparisons:
        lines.append(
            f"| {int(row['setting'])} | {row['our_mean_finish_time']:.4f} | "
            f"{row['our_vs_daoc_paper_improvement_percent']:+.2f}% | "
            f"{row['our_vs_our_flat_ddqn_improvement_percent']:+.2f}% | "
            f"{row['our_vs_coord_cache_discrete_sac_improvement_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 服务镜像大小",
            "",
            "| 镜像倍率 | OUR (s) | 相对 DAOC | 相对 DDQN + DCC | 相对 SAC + DCC |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in size_comparisons:
        lines.append(
            f"| {row['setting']:g}x | {row['our_mean_finish_time']:.4f} | "
            f"{row['our_vs_daoc_paper_improvement_percent']:+.2f}% | "
            f"{row['our_vs_our_flat_ddqn_improvement_percent']:+.2f}% | "
            f"{row['our_vs_coord_cache_discrete_sac_improvement_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 图文用途",
            "",
            f"- 主图：`{png}`",
            "- 左图验证当活跃服务数逐渐超过总缓存预算8时，协调副本放置的收益是否保持。",
            "- 右图验证远程服务加载代价增大时，OUR 减少错配缓存和远程加载的优势是否放大。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    active_audit = read_json(RESULT_ROOT / "ACTIVE_SERVICE_AUDIT.json")
    size_audit = read_json(RESULT_ROOT / "service_size/AUDIT.json")
    if active_audit.get("status") != "complete":
        raise RuntimeError("Active-service audit is incomplete")
    if size_audit.get("status") != "complete":
        raise RuntimeError("Service-size audit is incomplete")

    active_rows, size_rows = collect_rows()
    active_agg = aggregate(active_rows)
    size_agg = aggregate(size_rows)
    active_comparisons = comparison_table(active_agg)
    size_comparisons = comparison_table(size_agg)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ANALYSIS_DIR / "active_service_seed_metrics.csv", active_rows)
    write_csv(ANALYSIS_DIR / "service_size_seed_metrics.csv", size_rows)
    write_csv(ANALYSIS_DIR / "active_service_aggregate.csv", active_agg)
    write_csv(ANALYSIS_DIR / "service_size_aggregate.csv", size_agg)
    write_csv(
        ANALYSIS_DIR / "active_service_our_improvements.csv",
        active_comparisons,
    )
    write_csv(
        ANALYSIS_DIR / "service_size_our_improvements.csv",
        size_comparisons,
    )
    png, pdf = plot(active_agg, size_agg)
    report = report_markdown(active_comparisons, size_comparisons, png)
    (ANALYSIS_DIR / "SERVICE_SENSITIVITY_REPORT_CN.md").write_text(
        report,
        encoding="utf-8",
    )
    write_json(
        ANALYSIS_DIR / "analysis_manifest.json",
        {
            "status": "complete",
            "seeds": list(SEEDS),
            "active_service_counts": list(ACTIVE_SERVICE_COUNTS),
            "service_size_multipliers": list(SERVICE_SIZE_MULTIPLIERS),
            "methods": list(METHODS),
            "figure_png": str(png.resolve()),
            "figure_pdf": str(pdf.resolve()),
            "claim_scope": "three_seed_descriptive_sensitivity",
        },
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
