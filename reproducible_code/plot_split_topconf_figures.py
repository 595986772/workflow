#!/usr/bin/env python3
"""Render standalone vector figures split from formal Figures 3 and 5."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import paper_line_style as line_style
import plot_topconf_paper_figures as base


MECHANISM_METHODS = (
    "greedy",
    "daoc_paper",
    "discrete_sac_std_cache",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pegasus_workflows() -> None:
    families = ("CyberShake", "Epigenomics", "Inspiral", "Montage", "Sipht")
    data = base.family_values()
    figure, axis = plt.subplots(
        figsize=(base.DOUBLE_COLUMN, line_style.LINE_FIGURE_HEIGHT),
        layout="constrained",
    )
    x = np.arange(len(families))
    table_rows = []
    for method in base.MAIN_METHODS:
        summaries = [base.ci95(data[method][family]) for family in families]
        means, errors = zip(*summaries)
        for family, (mean, error) in zip(families, summaries):
            table_rows.append(
                {
                    "method": base.DISPLAY[method],
                    "method_key": method,
                    "dataset_or_workflow": family,
                    "mean_completion_time_s": f"{mean:.9f}",
                    "ci95_half_width_s": f"{error:.9f}",
                    "independent_seeds": len(data[method][family]),
                    "evaluation_mode": "frozen_in_dataset_evaluation",
                }
            )
        axis.errorbar(
            x,
            means,
            yerr=errors,
            **line_style.errorbar_kwargs(
                method,
                label=base.DISPLAY[method],
            ),
        )
    axis.set_xticks(x)
    axis.set_xticklabels(families, rotation=12)
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.set_ylim(bottom=0)
    line_style.style_axis(axis, grid_axis="y")
    line_style.legend_above(axis, ncol=3)
    write_csv(
        base.OUTPUT_DIR / "fig03a_pegasus_workflows_data.csv",
        table_rows,
    )
    base.save_figure(
        figure,
        "fig03a_pegasus_workflows",
        additional_transformations=(
            "Pegasus-only workflow comparison separated from cross-dataset evidence",
            "95% Student-t CI over ten independent seed means",
        ),
    )


def plot_alibaba_controlled() -> None:
    data = base.family_values()
    figure, axis = plt.subplots(
        figsize=(base.SINGLE_COLUMN, 86 * base.MM),
        layout="constrained",
    )
    y = np.arange(len(base.MAIN_METHODS), dtype=float)
    rows = []
    for index, method in enumerate(base.MAIN_METHODS):
        values = np.asarray(data[method]["Alibaba-CP100"], dtype=float)
        mean, error = base.ci95(values)
        rows.append(
            {
                "method": base.DISPLAY[method],
                "method_key": method,
                "dataset": "Alibaba-CP100",
                "mean_completion_time_s": f"{mean:.9f}",
                "ci95_half_width_s": f"{error:.9f}",
                "independent_seeds": len(values),
                "evaluation_mode": "frozen_cross_dataset_controlled_pressure_test",
            }
        )
        axis.errorbar(
            mean,
            index,
            xerr=error,
            fmt=base.MARKERS[method],
            color=base.COLORS[method],
            markersize=6.0 if method == "lean_our" else 4.4,
            markerfacecolor=(
                base.COLORS[method] if method == "lean_our" else "white"
            ),
            markeredgecolor=base.COLORS[method],
            markeredgewidth=0.8,
            capsize=2.2,
            elinewidth=1.0,
            zorder=4 if method == "lean_our" else 2,
        )
    axis.set_yticks(y)
    axis.set_yticklabels([base.DISPLAY[method] for method in base.MAIN_METHODS])
    axis.invert_yaxis()
    axis.set_xlabel("Mean DAG completion time (s; lower is better)")
    axis.set_title("Alibaba-CP100 controlled pressure test", loc="left")
    base.style_axis(axis, grid="x")
    write_csv(base.OUTPUT_DIR / "fig03c_alibaba_controlled_data.csv", rows)
    base.save_figure(
        figure,
        "fig03c_alibaba_controlled",
        additional_transformations=(
            "Alibaba-CP100 separated from the five Pegasus workflow families",
            "frozen Pegasus checkpoints; controlled pressure test, not an unbiased holdout",
        ),
        additional_raw_data=(base.P10_SUMMARY_PATH,),
    )


def plot_latency_composition() -> None:
    from regenerate_critical_path_latency_figure import (
        DEFAULT_ANALYSIS_DIR,
        render_figure,
    )

    summary_path = DEFAULT_ANALYSIS_DIR / "critical_path_method_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Corrected critical-path latency data are missing; run "
            "regenerate_critical_path_latency_figure.py first"
        )
    render_figure(DEFAULT_ANALYSIS_DIR, base.OUTPUT_DIR, MECHANISM_METHODS)


def nature_context():
    return mpl.rc_context(
        fname=str(base.NATURE_STYLE),
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 450,
            "axes.unicode_minus": False,
            "font.family": "Arial",
            "font.size": 7.5,
            "text.color": "#3D4348",
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelcolor": "#3D4348",
            "axes.titlecolor": "#30363A",
            "axes.edgecolor": "#596168",
            "axes.linewidth": 0.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.color": "#4B5359",
            "ytick.color": "#4B5359",
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "legend.fontsize": 6.5,
        },
    )


def plot_service_availability() -> None:
    with nature_context():
        figure = plt.figure(figsize=(118 * base.MM, 55 * base.MM))
        axis = figure.add_axes([0.25, 0.30, 0.72, 0.55])
        coverage = np.vstack(
            [base.cluster_service_coverage(method) for method in base.PRIMARY_METHODS]
        )
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "vermilion_sequential",
            ("#FFFBF8", "#FAE3DC", "#F1B4A3", "#E18570", "#C45C49"),
        )
        axis.pcolormesh(
            np.arange(coverage.shape[1] + 1) - 0.5,
            np.arange(coverage.shape[0] + 1) - 0.5,
            coverage,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            shading="flat",
            edgecolors="#FFFDFC",
            linewidth=0.42,
        )
        axis.set_xlim(-0.5, coverage.shape[1] - 0.5)
        axis.set_ylim(coverage.shape[0] - 0.5, -0.5)
        for row_index in range(coverage.shape[0]):
            for column_index in range(coverage.shape[1]):
                value = coverage[row_index, column_index]
                red, green, blue, _ = cmap(value)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column_index,
                    row_index,
                    f"{int(round(100 * value))}%",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#FFFDFC" if luminance < 0.48 else "#343A3F",
                )
        axis.set_xticks(np.arange(10))
        axis.set_xticklabels(np.arange(1, 11))
        axis.set_yticks(np.arange(len(base.PRIMARY_METHODS)))
        axis.set_yticklabels(
            [base.DISPLAY[method] for method in base.PRIMARY_METHODS]
        )
        axis.set_xlabel("Service ID")
        axis.set_title("Service availability (%)", loc="left", pad=3)
        figure.text(
            0.25,
            0.09,
            "Cell: percentage of seeds with at least one replica (n=10).  "
            "B=8; 4xK=0, 4xK=1, 2xK=2.",
            fontsize=5.8,
            color="#596168",
        )
        rows = []
        for method_index, method in enumerate(base.PRIMARY_METHODS):
            for service_index, value in enumerate(coverage[method_index], start=1):
                rows.append(
                    {
                        "method": base.DISPLAY[method],
                        "method_key": method,
                        "service_id": service_index,
                        "availability": f"{value:.9f}",
                        "independent_seeds": len(base.FINAL_SEEDS),
                    }
                )
        write_csv(base.OUTPUT_DIR / "fig05a_service_availability_data.csv", rows)
        base.save_figure(
            figure,
            "fig05a_service_availability",
            additional_transformations=(
                "standalone extraction of cluster-level service availability",
                "direct percentage annotations",
            ),
        )


def cache_metric_values(method: str, metric: str) -> np.ndarray:
    values = []
    for seed in base.FINAL_SEEDS:
        rows = base.evaluation_rows(method, seed)
        if len(rows) != 100:
            raise RuntimeError(
                f"Expected 100 evaluation rows for {method} seed {seed}, "
                f"found {len(rows)}"
            )
        per_episode = np.asarray([float(row[metric]) for row in rows])
        if np.any((per_episode < 0.0) | (per_episode > 1.0)):
            raise RuntimeError(f"Invalid rate for {method} seed {seed}: {metric}")
        values.append(float(np.mean(per_episode)))
    return np.asarray(values, dtype=float)


def plot_cache_effectiveness() -> None:
    with mpl.rc_context():
        figure, axis = plt.subplots(
            figsize=(base.DOUBLE_COLUMN, 78 * base.MM),
            layout="constrained",
        )
        metrics = (
            ("cache_hit_rate", "Cache hit\n(higher better)"),
            ("cache_service_coverage", "Service coverage\n(higher better)"),
            ("cache_remote_loading_rate", "Remote loading\n(lower better)"),
        )
        x = np.arange(len(metrics), dtype=float)
        width = 0.125
        rows = []
        method_count = len(MECHANISM_METHODS)
        for method_index, method in enumerate(MECHANISM_METHODS):
            means, errors = [], []
            for metric, label in metrics:
                values = cache_metric_values(method, metric)
                mean, half = base.ci95(values)
                means.append(mean)
                errors.append(half)
                rows.append(
                    {
                        "method": base.DISPLAY[method],
                        "method_key": method,
                        "metric": metric,
                        "metric_label": label.replace("\n", " "),
                        "mean_rate": f"{mean:.9f}",
                        "ci95_half_width": f"{half:.9f}",
                        "independent_seeds": len(values),
                    }
                )
            positions = x + (
                method_index - (method_count - 1) / 2.0
            ) * width
            bars = axis.bar(
                positions,
                means,
                width=width,
                color=base.COLORS[method],
                edgecolor="#303030",
                linewidth=0.65,
                yerr=errors,
                error_kw={
                    "ecolor": "#202427",
                    "elinewidth": 0.75,
                    "capsize": 2.0,
                    "capthick": 0.75,
                },
                label=base.DISPLAY[method],
                zorder=3,
            )
            for bar, half in zip(bars, errors):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + half + 0.025,
                    f"{100 * bar.get_height():.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color="#303030",
                )
        axis.set_xticks(x)
        axis.set_xticklabels([label for _, label in metrics])
        axis.set_ylabel("Rate (%)")
        axis.set_ylim(0.0, 1.12)
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.legend(
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
            handlelength=1.5,
            columnspacing=0.9,
            labelspacing=0.35,
        )
        base.style_axis(axis, grid="y")
        write_csv(base.OUTPUT_DIR / "fig05b_cache_effectiveness_data.csv", rows)
        base.save_figure(
            figure,
            "fig05b_cache_effectiveness",
            additional_transformations=(
                "standalone extraction of cache-effectiveness metrics",
                "method set aligned exactly with the latency-composition figure",
                "solid method colors aligned with the formal paper palette",
                "95% Student-t CI over independent seed means",
            ),
        )


def main() -> None:
    base.validate_inputs()
    base.configure_style()
    p6_summary = base.read_json(base.P6_SUMMARY_PATH)
    p7_summary = base.read_json(base.P7_SUMMARY_PATH)
    p8_summary = base.read_json(base.P8_SUMMARY_PATH)
    base.validate_formal_results(p6_summary, p7_summary, p8_summary)
    base.validate_cross_dataset_results()
    plot_pegasus_workflows()
    plot_alibaba_controlled()
    plot_latency_composition()
    plot_service_availability()
    plot_cache_effectiveness()
    print("Wrote five standalone PNG/PDF/SVG figure sets")


if __name__ == "__main__":
    main()
