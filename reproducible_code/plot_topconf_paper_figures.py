#!/usr/bin/env python3
"""Render the frozen Pegasus results as publication-ready conference figures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy import stats

import paper_line_style as line_style

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
SCI_VIS_SKILL = WORKSPACE_ROOT / ".codex/skills/scientific-visualization"
SCI_VIS_STYLE = SCI_VIS_SKILL / "assets/publication.mplstyle"
NATURE_STYLE = SCI_VIS_SKILL / "assets/nature.mplstyle"
SCI_VIS_EXPORTER = SCI_VIS_SKILL / "scripts/figure_export.py"
SCI_VIS_REPOSITORY = "K-Dense-AI/scientific-agent-skills"
SCI_VIS_COMMIT = "991bd993aca4e90891d1f9908ba82ef45d77b6f0"
OUTPUT_DIR = ROOT / "paper_drafts" / "figures_topconf"
PREVIOUS_FIGURE_V2_DIR = ROOT / "paper_drafts" / "figures_scivis_v2"
PREVIOUS_FIGURE_V3_DIR = ROOT / "paper_drafts" / "figures_scivis_v3"
P3_FINAL = ROOT / "results/pegasus_pscale/p3_paper_closure/final"
P4_SUPPLEMENT = ROOT / "results/pegasus_pscale/p4_paper_supplement"
P5_RESULTS = ROOT / "results/pegasus_pscale/p5_baseline_extension"
P6_RESULTS = ROOT / "results/pegasus_pscale/p6_baselines_ablation"
P7_RESULTS = ROOT / "results/pegasus_pscale/p7_std_cache_discrete_sac"
P8_RESULTS = ROOT / "results/pegasus_pscale/p8_daoc_our_coord_cache"
P10_ALIBABA = ROOT / "results/pegasus_pscale/p10_alibaba_cp100_cross_dataset"
P2_SENSITIVITY = ROOT / "results/pegasus_pscale/p2/sensitivity"

P3_SUMMARY_PATH = P3_FINAL / "analysis/pegasus_paper_closure_summary.json"
P4_SUMMARY_PATH = (
    P4_SUPPLEMENT / "analysis/paper_supplement_summary.json"
)
P5_SUMMARY_PATH = P5_RESULTS / "analysis/baseline_extension_summary.json"
P6_SUMMARY_PATH = P6_RESULTS / "analysis/pegasus_p6_summary.json"
P7_SUMMARY_PATH = P7_RESULTS / "analysis/sac_std_cache_extension_summary.json"
P8_SUMMARY_PATH = P8_RESULTS / "analysis/daoc_coord_cache_extension_summary.json"
P10_SUMMARY_PATH = P10_ALIBABA / "protocol_summary.json"
P2_SUMMARY_PATH = (
    P2_SENSITIVITY / "analysis/pegasus_pscale_sensitivity_summary.json"
)
ORACLE_SEED_PATH = P3_FINAL / "oracle/oracle_floor_per_seed.csv"

sys.path.insert(0, str(SCI_VIS_SKILL / "assets"))
sys.path.insert(0, str(SCI_VIS_SKILL / "scripts"))
from color_palettes import TOL_BRIGHT  # noqa: E402
from figure_export import export_figure  # noqa: E402

MM = 1.0 / 25.4
SINGLE_COLUMN = 88 * MM
DOUBLE_COLUMN = 180 * MM

FINAL_SEEDS = tuple(range(51, 61))
MAIN_METHODS = (
    "random",
    "nearest",
    "greedy",
    "dqn_wdsa_std_cache",
    "discrete_sac_std_cache",
    "daoc_paper",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
HEURISTIC_METHODS = MAIN_METHODS[:3]
LEARNING_METHODS = MAIN_METHODS[3:]
PRIMARY_METHODS = (
    "daoc_paper",
    "daoc_our_coord_cache",
    "lean_our",
)
CROSS_ENV_METHODS = (
    "daoc_paper",
    "lean_our",
)
ABLATION_METHODS = (
    "lean_our",
    "our_flat_ddqn",
    "our_no_coord_cache",
    "our_no_task_dependency",
    "our_terminal_reward",
)
DISPLAY = {
    "daoc_paper": "DAOC",
    "random": "Random",
    "nearest": "Nearest",
    "greedy": "SA-Nearest",
    "dqn_wdsa_std_cache": "DQN-NoDSA",
    "discrete_sac_std_cache": "SAC + DAOC-Cache",
    "coord_cache_discrete_sac": "SAC + DCC",
    "lean_our": "OUR",
    "base_ddqn_std_cache": "DDQN + DAOC-Cache",
    "our_flat_ddqn": "OUR w/o Pairwise Q",
    "our_no_coord_cache": "OUR w/o DCC",
    "daoc_our_coord_cache": "DAOC + DCC",
    "our_no_task_dependency": "OUR w/o TaskDep",
    "our_no_dependency_cache": "OUR w/o DepCache",
    "our_terminal_reward": "OUR w/ Terminal Reward",
    "oracle_floor": "Capacity-Aware Oracle",
    "perfect_cache_floor": "Perfect-Cache Lower Bound",
}

# Nature-inspired editorial palette shared with the finalized convergence
# figure. Color is reinforced by markers, hatches, and line styles.
# Marker and hatch redundancy remains necessary for grayscale reproduction.
COLORS = {
    "random": "#C9CED2",
    "nearest": "#9BA6AC",
    "greedy": "#5F6A70",
    "dqn_wdsa_std_cache": "#B45F57",
    "discrete_sac_std_cache": "#9A7B62",
    "daoc_paper": "#7B708E",
    "daoc_our_coord_cache": "#5E897E",
    "coord_cache_discrete_sac": "#5F8FA8",
    "lean_our": "#1F77A8",
    "base_ddqn_std_cache": "#B45F57",
    "our_flat_ddqn": "#5F8FA8",
    "our_no_coord_cache": "#9A7B62",
    "our_no_task_dependency": "#B45F57",
    "our_no_dependency_cache": "#9A7B62",
    "our_terminal_reward": "#7B708E",
    "oracle_floor": "#3F6B8A",
    "perfect_cache_floor": "#B45F57",
}
MARKERS = {
    "daoc_paper": "o",
    "random": "v",
    "nearest": "s",
    "greedy": "P",
    "dqn_wdsa_std_cache": "^",
    "discrete_sac_std_cache": "h",
    "coord_cache_discrete_sac": "X",
    "lean_our": "*",
    "base_ddqn_std_cache": "s",
    "our_flat_ddqn": "D",
    "our_no_coord_cache": "^",
    "daoc_our_coord_cache": "P",
}
HATCHES = {
    "daoc_paper": "//",
    "random": "..",
    "nearest": "xx",
    "greedy": "++",
    "dqn_wdsa_std_cache": "--",
    "discrete_sac_std_cache": "**",
    "coord_cache_discrete_sac": "oo",
    "lean_our": "",
    "base_ddqn_std_cache": "xx",
    "our_flat_ddqn": "\\\\",
    "our_no_coord_cache": "//",
    "daoc_our_coord_cache": "++",
}
LINESTYLES = {
    "random": (0, (1.2, 1.5)),
    "nearest": (0, (4.0, 1.8)),
    "greedy": (0, (6.0, 1.8, 1.2, 1.8)),
    "dqn_wdsa_std_cache": (0, (2.0, 1.2)),
    "discrete_sac_std_cache": (0, (4.0, 1.4, 1.0, 1.4)),
    "daoc_paper": "-",
    "daoc_our_coord_cache": (0, (5.0, 1.5)),
    "coord_cache_discrete_sac": (0, (3.0, 1.2)),
    "lean_our": "-",
}

# Formal trend figures share one canonical method identity across scripts.
COLORS.update(line_style.COLORS)
MARKERS.update(line_style.MARKERS)
LINESTYLES.update(line_style.LINESTYLES)

# These plots were not redesigned around Central-Greedy replacement. Keep the
# previously approved artwork instead of restyling the whole figure suite.
PRESERVED_PREVIOUS_FIGURES = {
    "fig06_heterogeneity_budget": PREVIOUS_FIGURE_V3_DIR,
    "fig07_scalability_overhead": PREVIOUS_FIGURE_V3_DIR,
}

REPORTED_EXTENSION_METHODS = (
    "coord_cache_discrete_sac",
    "lean_our",
)
REPORTED_EXTENSION_END_EPISODE = 26_000
REPORTED_EXTENSION_DROP_RANGE = (0.005, 0.010)
REPORTED_EXTENSION_DROP_MIDPOINT = 0.0075


def configure_style() -> None:
    line_style.configure_style(SCI_VIS_STYLE)
    mpl.rcParams.update(
        {
            "patch.linewidth": 0.55,
            "hatch.linewidth": 0.65,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0,
            "axes.unicode_minus": False,
        }
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ci95(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, 0.0
    half = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    return mean, half


def style_axis(axis, *, grid: str = "y") -> None:
    if grid:
        line_style.style_axis(axis, grid_axis=grid)
    else:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def panel_label(axis, label: str, x: float = -0.12, y: float = 1.04) -> None:
    del x, y
    axis.set_title(
        label,
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=3,
    )


def save_figure(
    figure,
    stem: str,
    *,
    additional_transformations: tuple[str, ...] = (),
    additional_raw_data: tuple[Path, ...] = (),
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(
        figure,
        OUTPUT_DIR / stem,
        formats=("pdf", "png", "svg"),
        dpi=400,
        transparent=False,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        font_mode="truetype",
        overwrite=True,
        mkdir=True,
        metadata={
            "Creator": "plot_topconf_paper_figures.py + scientific-visualization",
            "Title": stem,
        },
        provenance={
            "raw_data": [
                str(P3_SUMMARY_PATH.relative_to(ROOT)),
                str(P4_SUMMARY_PATH.relative_to(ROOT)),
                str(P5_SUMMARY_PATH.relative_to(ROOT)),
                str(P6_SUMMARY_PATH.relative_to(ROOT)),
                str(P7_SUMMARY_PATH.relative_to(ROOT)),
                str(P8_SUMMARY_PATH.relative_to(ROOT)),
                str(P2_SUMMARY_PATH.relative_to(ROOT)),
                *(str(path.relative_to(ROOT)) for path in additional_raw_data),
            ],
            "transformations": [
                "seed-level aggregation",
                "no smoothing or selective filtering in formal-result panels",
                *additional_transformations,
            ],
            "uncertainty": "95% Student-t CI over independent seed-level means",
            "missing_data": "no missing formal-result cells",
            "skill": "scientific-visualization",
            "skill_commit": SCI_VIS_COMMIT,
        },
        write_manifest=True,
    )
    plt.close(figure)


def restore_previous_figures() -> None:
    for stem, source_directory in PRESERVED_PREVIOUS_FIGURES.items():
        for suffix in ("png", "pdf"):
            source = source_directory / f"{stem}.{suffix}"
            if not source.exists():
                raise FileNotFoundError(f"Missing approved previous figure: {source}")
            shutil.copy2(source, OUTPUT_DIR / source.name)

        # The approved previous set predates SVG export. Remove any newly
        # rendered SVG so no alternate styling is mistaken for the same figure.
        svg_path = OUTPUT_DIR / f"{stem}.svg"
        if svg_path.exists():
            svg_path.unlink()
        export_report = OUTPUT_DIR / f"{stem}.export.json"
        if export_report.exists():
            export_report.unlink()


def method_values(summary, method: str, metric: str) -> np.ndarray:
    return np.asarray(
        [row["methods"][method][metric] for row in summary["per_seed"]],
        dtype=float,
    )


def final_method_values(
    p6_summary,
    p7_summary,
    p8_summary,
    method: str,
    metric: str,
) -> np.ndarray:
    if method == "discrete_sac_std_cache":
        source = p7_summary
    elif method == "daoc_our_coord_cache":
        source = p8_summary
    else:
        source = p6_summary
    return method_values(source, method, metric)


def draw_bar_with_points(axis, methods, values_by_method, *, ylabel: str) -> None:
    x = np.arange(len(methods), dtype=float)
    rng = np.random.default_rng(20260808)
    means = []
    errors = []
    for method in methods:
        mean, half = ci95(values_by_method[method])
        means.append(mean)
        errors.append(half)
    bars = axis.bar(
        x,
        means,
        width=0.72,
        color=[COLORS[method] for method in methods],
        edgecolor="#303030",
        yerr=errors,
        error_kw={"elinewidth": 0.75, "capsize": 2.5, "capthick": 0.75},
        zorder=2,
    )
    for bar, method in zip(bars, methods):
        bar.set_hatch(HATCHES[method])
    for index, method in enumerate(methods):
        values = np.asarray(values_by_method[method], dtype=float)
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=10,
            facecolor="white",
            edgecolor="#303030",
            linewidth=0.45,
            zorder=3,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([DISPLAY[method] for method in methods], rotation=24, ha="right")
    axis.set_ylabel(ylabel)
    style_axis(axis)


def plot_main_baselines(p6_summary, p7_summary, p8_summary) -> None:
    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN, line_style.LINE_FIGURE_HEIGHT),
        layout="constrained",
    )
    specifications = (
        ("mean_finish_time", "Mean", ""),
        ("mean_p95_finish_time", "P95", "//"),
    )
    rng = np.random.default_rng(20260811)
    x = np.arange(len(MAIN_METHODS), dtype=float)
    width = 0.34
    offsets = (-width / 1.7, width / 1.7)
    legend_handles = []
    for offset, (metric, label, hatch) in zip(offsets, specifications):
        values = {
            method: final_method_values(
                p6_summary,
                p7_summary,
                p8_summary,
                method,
                metric,
            )
            for method in MAIN_METHODS
        }
        means, errors = zip(*(ci95(values[method]) for method in MAIN_METHODS))
        bars = axis.bar(
            x + offset,
            means,
            width=width,
            color=[COLORS[method] for method in MAIN_METHODS],
            edgecolor="#303030",
            yerr=errors,
            error_kw={"elinewidth": 0.72, "capsize": 2.2, "capthick": 0.72},
            zorder=2,
        )
        for bar in bars:
            bar.set_hatch(hatch)
        for index, method in enumerate(MAIN_METHODS):
            observations = values[method]
            jitter = rng.uniform(-0.075, 0.075, size=len(observations))
            axis.scatter(
                np.full(len(observations), x[index] + offset) + jitter,
                observations,
                s=10,
                facecolor="white",
                edgecolor="#303030",
                linewidth=0.45,
                zorder=3,
            )
        legend_handles.append(
            mpl.patches.Patch(
                facecolor="#D8D8D8",
                edgecolor="#303030",
                hatch=hatch,
                label=label,
            )
        )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [DISPLAY[method] for method in MAIN_METHODS],
        rotation=27,
        ha="right",
    )
    axis.set_ylabel("DAG completion time (s)")
    axis.set_ylim(bottom=0)
    axis.legend(handles=legend_handles, ncol=2, loc="upper right")
    axis.set_title(
        "Bars: seed mean; error bars: 95% CI; points: 10 seeds",
        loc="left",
        fontsize=6.7,
        color="#505050",
        pad=3,
    )
    style_axis(axis, grid="y")

    # The full-range bars necessarily compress the two strongest methods.
    # Show their paired seed-level difference without truncating the main axis.
    inset = axis.inset_axes([0.685, 0.49, 0.295, 0.31])
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.96)
    rng_inset = np.random.default_rng(20260812)
    inset_rows = (
        ("mean_finish_time", 1.0, "Mean"),
        ("mean_p95_finish_time", 0.0, "P95"),
    )
    all_reductions = []
    tick_labels = []
    for metric, row, label in inset_rows:
        strong = final_method_values(
            p6_summary,
            p7_summary,
            p8_summary,
            "coord_cache_discrete_sac",
            metric,
        )
        our = final_method_values(
            p6_summary,
            p7_summary,
            p8_summary,
            "lean_our",
            metric,
        )
        reductions = np.asarray(strong, dtype=float) - np.asarray(our, dtype=float)
        all_reductions.extend(reductions)
        mean, half = ci95(reductions)
        p_value = float(stats.wilcoxon(reductions, alternative="greater").pvalue)
        improvement = 100.0 * (float(np.mean(strong)) - float(np.mean(our))) / float(
            np.mean(strong)
        )
        significance = "p<.001" if p_value < 0.001 else f"p={p_value:.3f}"
        tick_labels.append(f"{label}: {improvement:.1f}% ({significance})")
        jitter = rng_inset.uniform(-0.10, 0.10, size=len(reductions))
        inset.scatter(
            reductions,
            np.full(len(reductions), row) + jitter,
            s=8,
            facecolor="white",
            edgecolor=COLORS["lean_our"],
            linewidth=0.5,
            zorder=3,
        )
        inset.errorbar(
            mean,
            row,
            xerr=half,
            fmt="D",
            markersize=3.3,
            color=COLORS["lean_our"],
            markeredgecolor="#303030",
            markeredgewidth=0.4,
            capsize=2.0,
            elinewidth=0.9,
            zorder=4,
        )
    limit = max(abs(float(np.min(all_reductions))), abs(float(np.max(all_reductions))))
    inset.set_xlim(-1.18 * limit, 1.18 * limit)
    inset.axvline(0.0, color="#303030", linewidth=0.65, linestyle=(0, (3, 2)))
    inset.set_ylim(-0.42, 1.42)
    inset.set_yticks([1.0, 0.0])
    inset.set_yticklabels(tick_labels)
    inset.set_xlabel(
        "Paired reduction (s; + favors OUR)", fontsize=5.2, labelpad=1.5
    )
    inset.set_title(
        "OUR vs SAC + DCC",
        loc="left",
        fontsize=5.6,
        pad=2,
    )
    style_axis(inset, grid="x")
    inset.tick_params(axis="both", labelsize=5.0, length=2.0, width=0.55)
    save_figure(figure, "fig01_main_baselines")


def plot_seed_robustness(p6_summary, p7_summary, p8_summary) -> None:
    baselines = tuple(method for method in MAIN_METHODS if method != "lean_our")
    our = final_method_values(
        p6_summary,
        p7_summary,
        p8_summary,
        "lean_our",
        "mean_finish_time",
    )
    figure, axis = plt.subplots(
        figsize=(SINGLE_COLUMN, 86 * MM),
        layout="constrained",
    )
    y = np.arange(len(baselines))
    rng = np.random.default_rng(20260808)
    annotations = []
    for index, method in enumerate(baselines):
        baseline = final_method_values(
            p6_summary,
            p7_summary,
            p8_summary,
            method,
            "mean_finish_time",
        )
        reductions = baseline - our
        mean, half = ci95(reductions)
        jitter = rng.uniform(-0.10, 0.10, size=len(reductions))
        axis.scatter(
            reductions,
            np.full(len(reductions), index) + jitter,
            s=14,
            marker=MARKERS[method],
            facecolor="white",
            edgecolor=COLORS[method],
            linewidth=0.75,
            zorder=3,
        )
        axis.errorbar(
            mean,
            index,
            xerr=half,
            fmt="D",
            markersize=4.2,
            color=COLORS[method],
            markeredgecolor="#202020",
            markeredgewidth=0.45,
            capsize=2.5,
            elinewidth=1.1,
            zorder=4,
        )
        wins = int(np.sum(reductions > 0))
        aggregate = float(np.mean(baseline))
        aggregate_our = float(np.mean(our))
        improvement = 100.0 * (aggregate - aggregate_our) / aggregate
        annotations.append((index, wins, improvement))
    axis.axvline(0.0, color="#303030", linewidth=0.75, linestyle=(0, (3, 2)))
    axis.set_yticks(y)
    axis.set_yticklabels([DISPLAY[method] for method in baselines])
    axis.invert_yaxis()
    axis.set_xlabel("Paired reduction (s; + favors OUR)")
    style_axis(axis, grid="x")
    x_left, x_right = axis.get_xlim()
    axis.set_xlim(x_left, x_right + 0.19 * (x_right - x_left))
    annotation_x = axis.get_xlim()[1]
    for index, wins, improvement in annotations:
        axis.text(
            annotation_x,
            index,
            f" {wins}/10 | {improvement:.1f}%",
            ha="right",
            va="center",
            fontsize=6.4,
        )
    save_figure(figure, "fig02_seed_robustness")


def run_directory(method: str, seed: int) -> Path:
    if method in HEURISTIC_METHODS:
        return P6_RESULTS / "heuristics" / "runs" / method / f"seed_{seed}"
    if method == "dqn_wdsa_std_cache":
        return P6_RESULTS / "learning" / "runs" / method / f"seed_{seed}"
    if method == "discrete_sac_std_cache":
        return P7_RESULTS / "final" / "runs" / method / f"seed_{seed}"
    if method == "daoc_our_coord_cache":
        return P8_RESULTS / "final" / "runs" / method / f"seed_{seed}"
    if method == "coord_cache_discrete_sac":
        return P5_RESULTS / "sac_final" / "runs" / method / f"seed_{seed}"
    return P3_FINAL / "runs" / method / f"seed_{seed}"


def evaluation_rows(method: str, seed: int) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(run_directory(method, seed) / "episodes.csv")
        if row["phase"] == "eval"
    ]


def family_values() -> dict[str, dict[str, list[float]]]:
    result = {
        method: defaultdict(list)
        for method in MAIN_METHODS
    }
    for method in MAIN_METHODS:
        for seed in FINAL_SEEDS:
            scenarios = read_json(
                run_directory(method, seed) / "evaluation_scenarios.json"
            )
            family_by_episode = {
                int(row["episode"]): row["workflow_family"]
                for row in scenarios
            }
            grouped = defaultdict(list)
            for row in evaluation_rows(method, seed):
                grouped[family_by_episode[int(row["episode"])]] .append(
                    float(row["average_finish_time"])
                )
            for family, values in grouped.items():
                result[method][family].append(float(np.mean(values)))
            alibaba_summary = read_json(
                P10_ALIBABA
                / "runs"
                / method
                / f"seed_{seed}"
                / "summary.json"
            )
            result[method]["Alibaba-CP100"].append(
                float(alibaba_summary["eval"]["mean_average_finish_time"])
            )
    return result


def latency_shares(
    methods: tuple[str, ...] = PRIMARY_METHODS,
) -> dict[str, dict[str, list[float]]]:
    repair_path = (
        ROOT
        / "results"
        / "pegasus_pscale"
        / "p15_latency_composition_repair"
        / "critical_path_seed_summary.csv"
    )
    if not repair_path.exists():
        raise FileNotFoundError(
            "Corrected critical-path latency data are missing; run "
            "regenerate_critical_path_latency_figure.py first"
        )
    rows = read_csv(repair_path)
    by_method_seed = {
        (row["method_key"], int(row["seed"])): row
        for row in rows
    }
    components = (
        "Computation",
        "Data transfer",
        "Service loading",
        "Waiting",
    )
    result = {
        method: {name: [] for name in components}
        for method in methods
    }
    for method in methods:
        for seed in FINAL_SEEDS:
            row = by_method_seed.get((method, seed))
            if row is None:
                raise RuntimeError(
                    f"Missing corrected latency row for {method} seed {seed}"
                )
            denominator = float(row["completion_time_s"])
            values = {
                "Computation": float(row["computation_s"]),
                "Data transfer": (
                    float(row["user_input_transfer_s"])
                    + float(row["dependency_transfer_s"])
                ),
                "Service loading": float(row["service_loading_s"]),
                "Waiting": float(row["waiting_s"]),
            }
            if not np.isclose(sum(values.values()), denominator, atol=1e-9):
                raise RuntimeError(
                    f"Corrected components do not sum for {method} seed {seed}"
                )
            for name, value in values.items():
                result[method][name].append(value / denominator)
    return result


def plot_workflow_latency() -> None:
    families = (
        "CyberShake",
        "Epigenomics",
        "Inspiral",
        "Montage",
        "Sipht",
        "Alibaba-CP100",
    )
    family_data = family_values()
    shares = latency_shares()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN, 82 * MM),
        gridspec_kw={"width_ratios": (1.28, 1.0)},
        layout="constrained",
    )

    x = np.arange(len(families))
    table_rows = []
    for method in MAIN_METHODS:
        summaries = [ci95(family_data[method][family]) for family in families]
        means, errors = zip(*summaries)
        for family, (mean, error) in zip(families, summaries):
            table_rows.append(
                {
                    "method": DISPLAY[method],
                    "method_key": method,
                    "dataset_or_workflow": family,
                    "mean_completion_time_s": f"{mean:.9f}",
                    "ci95_half_width_s": f"{error:.9f}",
                    "independent_seeds": len(family_data[method][family]),
                    "evaluation_mode": (
                        "frozen_cross_dataset_zero_shot"
                        if family == "Alibaba-CP100"
                        else "frozen_in_dataset_evaluation"
                    ),
                }
            )
        axes[0].errorbar(
            x,
            means,
            yerr=errors,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.45 if method == "lean_our" else 1.0,
            markersize=5.2 if method == "lean_our" else 3.8,
            markerfacecolor="white" if method != "lean_our" else COLORS[method],
            markeredgecolor=COLORS[method],
            markeredgewidth=0.75,
            capsize=1.6,
            elinewidth=0.65,
            zorder=4 if method == "lean_our" else 2,
            label=DISPLAY[method],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        (
            "CyberShake",
            "Epigen.",
            "Inspiral",
            "Montage",
            "Sipht",
            "Alibaba-\nCP100*",
        ),
        rotation=18,
    )
    axes[0].set_ylabel("Mean completion time (s)")
    axes[0].legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.85,
        handlelength=2.2,
    )
    style_axis(axes[0])
    panel_label(axes[0], "(a)")

    component_colors = {
        "Computation": TOL_BRIGHT[0],
        "Data transfer": TOL_BRIGHT[4],
        "Service loading": TOL_BRIGHT[3],
        "Waiting": TOL_BRIGHT[1],
    }
    component_hatches = {
        "Computation": "",
        "Data transfer": "//",
        "Service loading": "xx",
        "Waiting": "..",
    }
    positions = np.arange(len(PRIMARY_METHODS))
    bottom = np.zeros(len(PRIMARY_METHODS))
    for component in component_colors:
        values = np.asarray(
            [
                np.mean(shares[method][component]) * 100.0
                for method in PRIMARY_METHODS
            ]
        )
        bars = axes[1].bar(
            positions,
            values,
            bottom=bottom,
            width=0.62,
            color=component_colors[component],
            edgecolor="#303030",
            label=component,
        )
        for bar in bars:
            bar.set_hatch(component_hatches[component])
        for index, value in enumerate(values):
            if value >= 8.0:
                axes[1].text(
                    index,
                    bottom[index] + value / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                )
        bottom += values
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([DISPLAY[method] for method in PRIMARY_METHODS])
    axes[1].set_ylabel("Average latency share (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0,
    )
    style_axis(axes[1])
    panel_label(axes[1], "(b)")
    with (OUTPUT_DIR / "fig03_workflow_latency_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    save_figure(figure, "fig03_workflow_latency")


def plot_ablation(p6_summary, p7_summary, p8_summary) -> None:
    scheduler_groups = (
        ("DAOC\nagent", "daoc_paper", "daoc_our_coord_cache"),
        ("Flat\nDDQN", "base_ddqn_std_cache", "our_flat_ddqn"),
        ("Discrete\nSAC", "discrete_sac_std_cache", "coord_cache_discrete_sac"),
        ("Pairwise\nPD3QN", "our_no_coord_cache", "lean_our"),
    )
    metric = "mean_finish_time"
    standard_values = [
        final_method_values(p6_summary, p7_summary, p8_summary, standard, metric)
        for _, standard, _ in scheduler_groups
    ]
    coordinated_values = [
        final_method_values(p6_summary, p7_summary, p8_summary, coordinated, metric)
        for _, _, coordinated in scheduler_groups
    ]
    standard_stats = [ci95(values) for values in standard_values]
    coordinated_stats = [ci95(values) for values in coordinated_values]

    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN, 70 * MM),
        layout="constrained",
    )
    x = np.arange(len(scheduler_groups), dtype=float)
    width = 0.34
    standard_color = COLORS["daoc_paper"]
    coordinated_color = COLORS["daoc_our_coord_cache"]
    our_color = COLORS["lean_our"]
    edge_color = "#303030"
    standard_means, standard_errors = zip(*standard_stats)
    coordinated_means, coordinated_errors = zip(*coordinated_stats)
    standard_bars = axis.bar(
        x - width / 2,
        standard_means,
        width=width,
        color=standard_color,
        edgecolor=edge_color,
        linewidth=0.75,
        hatch="//",
        yerr=standard_errors,
        error_kw={
            "ecolor": "#202427",
            "elinewidth": 0.85,
            "capsize": 3.0,
            "capthick": 0.85,
        },
        label="DAOC-Cache",
        zorder=2,
    )
    coordinated_bar_colors = [
        coordinated_color,
        coordinated_color,
        coordinated_color,
        our_color,
    ]
    coordinated_bars = axis.bar(
        x + width / 2,
        coordinated_means,
        width=width,
        color=coordinated_bar_colors,
        edgecolor=edge_color,
        linewidth=0.75,
        yerr=coordinated_errors,
        error_kw={
            "ecolor": "#202427",
            "elinewidth": 0.85,
            "capsize": 3.0,
            "capthick": 0.85,
        },
        label="DCC",
        zorder=2,
    )
    coordinated_bars[-1].set_edgecolor("#174D69")
    coordinated_bars[-1].set_linewidth(1.35)

    annotation_tops = []
    for index in range(len(scheduler_groups)):
        standard_top = standard_means[index] + standard_errors[index]
        coordinated_top = coordinated_means[index] + coordinated_errors[index]
        pair_top = max(standard_top, coordinated_top)
        annotation_tops.append(pair_top + 0.045)
        for bar, mean, error, color in (
            (
                standard_bars[index],
                standard_means[index],
                standard_errors[index],
                standard_color,
            ),
            (
                coordinated_bars[index],
                coordinated_means[index],
                coordinated_errors[index],
                coordinated_bar_colors[index],
            ),
        ):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + error + 0.018,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=color,
            )

    axis.annotate(
        "OUR",
        xy=(x[-1] + width / 2, coordinated_means[-1]),
        xytext=(28, 15),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=our_color,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": our_color,
            "linewidth": 0.75,
        },
    )
    axis.set_xticks(x)
    axis.set_xticklabels([label for label, _, _ in scheduler_groups])
    axis.set_ylabel("Mean completion time (s)")
    axis.set_ylim(0, max(annotation_tops) * 1.10)
    axis.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.55,
        columnspacing=1.2,
        borderaxespad=0.35,
    )
    style_axis(axis, grid="y")
    save_figure(figure, "fig04_ablation")


def plot_daoc_style_ablation_preview(p6_summary, p7_summary, p8_summary) -> None:
    """Draw a DAOC Fig. 5-style scheduler/cache replacement study."""
    scheduler_groups = (
        ("DAOC\nagent", "daoc_paper", "daoc_our_coord_cache"),
        ("Flat\nDDQN", "base_ddqn_std_cache", "our_flat_ddqn"),
        ("Discrete\nSAC", "discrete_sac_std_cache", "coord_cache_discrete_sac"),
        ("Pairwise\nPD3QN", "our_no_coord_cache", "lean_our"),
    )
    metric = "mean_finish_time"
    standard_values = [
        final_method_values(p6_summary, p7_summary, p8_summary, standard, metric)
        for _, standard, _ in scheduler_groups
    ]
    coordinated_values = [
        final_method_values(p6_summary, p7_summary, p8_summary, coordinated, metric)
        for _, _, coordinated in scheduler_groups
    ]

    figure, axis = plt.subplots(
        figsize=(145 * MM, 76 * MM),
        layout="constrained",
    )
    x = np.arange(len(scheduler_groups), dtype=float)
    width = 0.34
    cache_series = (
        (
            "DAOC-Cache",
            standard_values,
            -width / 2,
            COLORS["discrete_sac_std_cache"],
            "//",
        ),
        ("DCC", coordinated_values, width / 2, COLORS["lean_our"], ""),
    )
    rng = np.random.default_rng(20260817)
    all_observations = []
    for label, series, offset, color, hatch in cache_series:
        means, errors = zip(*(ci95(values) for values in series))
        bars = axis.bar(
            x + offset,
            means,
            width=width,
            color=color,
            edgecolor="#303030",
            linewidth=0.75,
            hatch=hatch,
            yerr=errors,
            error_kw={
                "ecolor": "#202427",
                "elinewidth": 0.85,
                "capsize": 2.8,
                "capthick": 0.85,
            },
            label=label,
            zorder=2,
        )
        del bars
        for position, observations in zip(x + offset, series):
            jitter = rng.uniform(-0.045, 0.045, size=len(observations))
            axis.scatter(
                position + jitter,
                observations,
                s=12,
                facecolor="white",
                edgecolor="#3F464A",
                linewidth=0.55,
                zorder=3,
            )
            all_observations.extend(float(value) for value in observations)

    hero_mean, hero_error = ci95(coordinated_values[-1])
    axis.annotate(
        "OUR",
        xy=(x[-1] + width / 2, hero_mean + hero_error),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color="#1F5D7A",
        arrowprops={
            "arrowstyle": "-",
            "color": "#1F5D7A",
            "linewidth": 0.75,
        },
    )
    axis.set_xticks(x)
    axis.set_xticklabels([label for label, _, _ in scheduler_groups])
    axis.set_ylabel("Mean completion time (s)")
    upper = max(all_observations)
    axis.set_ylim(0, upper * 1.18)
    axis.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.55,
        columnspacing=1.2,
        borderaxespad=0.35,
    )
    style_axis(axis, grid="y")

    rows = []
    for (label, standard, coordinated), standard_obs, coordinated_obs in zip(
        scheduler_groups, standard_values, coordinated_values
    ):
        rows.append(
            {
                "scheduler": label.replace("\n", " "),
                "standard_method": standard,
                "standard_cache_mean_sec": float(np.mean(standard_obs)),
                "coordinated_method": coordinated,
                "dcc_mean_sec": float(np.mean(coordinated_obs)),
                "dcc_improvement_percent": float(
                    100.0
                    * (np.mean(standard_obs) - np.mean(coordinated_obs))
                    / np.mean(standard_obs)
                ),
            }
        )
    with (OUTPUT_DIR / "fig04_daoc_style_ablation_preview_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_figure(
        figure,
        "fig04_daoc_style_ablation_preview",
        additional_transformations=(
            "fully crossed four-scheduler by two-cache replacement study",
        ),
    )


def plot_daoc_style_dumbbell_preview(p6_summary, p7_summary, p8_summary) -> None:
    """Draw the scheduler/cache replacement study as paired improvement arrows."""
    scheduler_groups = (
        ("DAOC agent", "daoc_paper", "daoc_our_coord_cache"),
        ("Flat DDQN", "base_ddqn_std_cache", "our_flat_ddqn"),
        ("Discrete SAC", "discrete_sac_std_cache", "coord_cache_discrete_sac"),
        ("Pairwise PD3QN", "our_no_coord_cache", "lean_our"),
    )
    metric = "mean_finish_time"
    rows = []
    for label, standard_method, coordinated_method in scheduler_groups:
        standard = final_method_values(
            p6_summary, p7_summary, p8_summary, standard_method, metric
        )
        coordinated = final_method_values(
            p6_summary, p7_summary, p8_summary, coordinated_method, metric
        )
        standard_mean, standard_error = ci95(standard)
        coordinated_mean, coordinated_error = ci95(coordinated)
        rows.append(
            {
                "label": label,
                "standard_mean": standard_mean,
                "standard_error": standard_error,
                "coordinated_mean": coordinated_mean,
                "coordinated_error": coordinated_error,
                "improvement": 100.0
                * (standard_mean - coordinated_mean)
                / standard_mean,
            }
        )

    figure, axis = plt.subplots(
        figsize=(148 * MM, 76 * MM),
        layout="constrained",
    )
    y = np.arange(len(rows), dtype=float) * 1.15
    standard_color = COLORS["discrete_sac_std_cache"]
    coordinated_color = COLORS["lean_our"]
    connector_color = "#9CA6AA"

    for index, (position, row) in enumerate(zip(y, rows)):
        if index % 2 == 0:
            axis.axhspan(
                position - 0.43,
                position + 0.43,
                color="#F5F7F8",
                zorder=0,
            )
        axis.annotate(
            "",
            xy=(row["coordinated_mean"], position),
            xytext=(row["standard_mean"], position),
            arrowprops={
                "arrowstyle": "-|>",
                "color": connector_color,
                "linewidth": 1.6,
                "mutation_scale": 10,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=1,
        )
        axis.errorbar(
            row["standard_mean"],
            position,
            xerr=row["standard_error"],
            fmt="s",
            markersize=6.2,
            markerfacecolor=standard_color,
            markeredgecolor="#3F464A",
            markeredgewidth=0.7,
            ecolor=standard_color,
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )
        coordinated_marker = "*" if index == len(rows) - 1 else "o"
        coordinated_size = 9.2 if coordinated_marker == "*" else 6.5
        axis.errorbar(
            row["coordinated_mean"],
            position,
            xerr=row["coordinated_error"],
            fmt=coordinated_marker,
            markersize=coordinated_size,
            markerfacecolor=coordinated_color,
            markeredgecolor="#24495B",
            markeredgewidth=0.7,
            ecolor=coordinated_color,
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=4,
        )
        midpoint = (row["standard_mean"] + row["coordinated_mean"]) / 2.0
        axis.text(
            midpoint,
            position - 0.18,
            f'{row["improvement"]:.1f}% lower',
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#566167",
            fontweight="semibold",
        )
        axis.text(
            row["standard_mean"],
            position + 0.20,
            f'{row["standard_mean"]:.3f}',
            ha="center",
            va="top",
            fontsize=6.7,
            color=standard_color,
        )
        axis.text(
            row["coordinated_mean"],
            position + 0.20,
            f'{row["coordinated_mean"]:.3f}',
            ha="center",
            va="top",
            fontsize=6.7,
            color=coordinated_color,
        )

    standard_handle = mpl.lines.Line2D(
        [],
        [],
        marker="s",
        linestyle="none",
        markersize=6,
        markerfacecolor=standard_color,
        markeredgecolor="#3F464A",
        label="DAOC-Cache",
    )
    coordinated_handle = mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=6,
        markerfacecolor=coordinated_color,
        markeredgecolor="#24495B",
        label="DCC",
    )
    axis.legend(
        handles=[standard_handle, coordinated_handle],
        loc="upper right",
        frameon=False,
        ncol=2,
        handletextpad=0.45,
        columnspacing=1.2,
        borderaxespad=0.3,
    )
    axis.set_yticks(y)
    axis.set_yticklabels([row["label"] for row in rows])
    axis.invert_yaxis()
    lower = min(row["coordinated_mean"] - row["coordinated_error"] for row in rows)
    upper = max(row["standard_mean"] + row["standard_error"] for row in rows)
    axis.set_xlim(max(0, lower - 0.10), upper + 0.16)
    axis.set_xlabel("Mean completion time (s)")
    axis.text(
        0.995,
        0.025,
        "Lower is better  ←",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#68737A",
    )
    style_axis(axis, grid="x")
    save_figure(
        figure,
        "fig04_daoc_style_dumbbell_preview",
        additional_transformations=(
            "paired scheduler-cache means rendered as improvement arrows",
        ),
    )


def first_evaluation_row(directory: Path) -> dict[str, str]:
    for row in read_csv(directory / "episodes.csv"):
        if row["phase"] == "eval":
            return row
    raise RuntimeError(f"No evaluation rows in {directory}")


def average_cache_heatmap(method: str) -> tuple[np.ndarray, list[int]]:
    matrices = []
    capacity_order = None
    for seed in FINAL_SEEDS:
        directory = run_directory(method, seed)
        summary = read_json(directory / "summary.json")
        capacities = {
            int(server): int(capacity)
            for server, capacity in summary["server_capacities"].items()
        }
        order = sorted(capacities, key=lambda server: (capacities[server], server))
        capacity_order = [capacities[server] for server in order]
        cache = json.loads(first_evaluation_row(directory)["cache_matrix_json"])
        matrix = np.zeros((10, 10), dtype=float)
        for row_index, server in enumerate(order):
            for service in cache[str(server)]:
                service = int(service)
                if service > 0:
                    matrix[row_index, service - 1] = 1.0
        matrices.append(matrix)
    return np.mean(matrices, axis=0), capacity_order


def cluster_service_coverage(method: str) -> np.ndarray:
    """Return the probability that each service has at least one replica."""
    coverage_rows = []
    for seed in FINAL_SEEDS:
        directory = run_directory(method, seed)
        cache = json.loads(first_evaluation_row(directory)["cache_matrix_json"])
        covered = {
            int(service)
            for services in cache.values()
            for service in services
            if int(service) > 0
        }
        coverage_rows.append(
            [1.0 if service in covered else 0.0 for service in range(1, 11)]
        )
    return np.mean(coverage_rows, axis=0)


def plot_cache_mechanism(p8_summary) -> None:
    with mpl.rc_context(fname=str(NATURE_STYLE)):
        mpl.rcParams.update(
            {
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
            }
        )
        figure = plt.figure(figsize=(195 * MM, 76 * MM))
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=(1.00, 1.32),
            left=0.155,
            right=0.99,
            bottom=0.23,
            top=0.82,
            wspace=0.30,
        )
        coverage_axis = figure.add_subplot(grid[0, 0])
        metric_axis = figure.add_subplot(grid[0, 1])

        coverage = np.vstack(
            [cluster_service_coverage(method) for method in PRIMARY_METHODS]
        )
        coverage_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "vermilion_sequential",
            ("#FFFBF8", "#FAE3DC", "#F1B4A3", "#E18570", "#C45C49"),
        )
        coverage_axis.imshow(
            coverage,
            vmin=0.0,
            vmax=1.0,
            cmap=coverage_cmap,
            aspect="auto",
            interpolation="nearest",
        )
        coverage_axis.set_box_aspect(0.40)
        coverage_axis.set_anchor("N")
        for row_index in range(coverage.shape[0]):
            for column_index in range(coverage.shape[1]):
                value = coverage[row_index, column_index]
                red, green, blue, _ = coverage_cmap(value)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                coverage_axis.text(
                    column_index,
                    row_index,
                    f"{int(round(100 * value))}%",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="#FFFDFC" if luminance < 0.48 else "#343A3F",
                )
        coverage_axis.set_xticks(np.arange(10))
        coverage_axis.set_xticklabels(np.arange(1, 11))
        coverage_axis.set_yticks(np.arange(len(PRIMARY_METHODS)))
        coverage_axis.set_yticklabels([DISPLAY[method] for method in PRIMARY_METHODS])
        coverage_axis.tick_params(axis="y", labelsize=6.2)
        coverage_axis.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        coverage_axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        coverage_axis.grid(which="minor", color="#FFFDFC", linewidth=0.42)
        coverage_axis.tick_params(which="minor", bottom=False, left=False)
        coverage_axis.set_xlabel("Service ID")
        coverage_axis.set_title("Service availability (%)", loc="left", pad=3)
        coverage_axis.text(
            -0.13,
            1.06,
            "a",
            transform=coverage_axis.transAxes,
            fontsize=8.6,
            fontweight="bold",
        )
        coverage_axis.text(
            0.0,
            -0.44,
            "Cell value: percentage of seeds with at least one replica (n=10).\n"
            "B=8; capacity profile: 4xK=0, 4xK=1, 2xK=2.",
            transform=coverage_axis.transAxes,
            fontsize=5.9,
            color="#596168",
            va="top",
        )

        metrics = (
            ("mean_cache_hit_rate", "Cache hit\n(higher better)"),
            ("mean_cache_service_coverage", "Service coverage\n(higher better)"),
            ("mean_remote_loading_rate", "Remote loading\n(lower better)"),
        )
        x = np.arange(len(metrics), dtype=float)
        width = 0.23
        legend_handles = []
        for method_index, method in enumerate(PRIMARY_METHODS):
            means, errors = [], []
            for metric, _ in metrics:
                values = np.asarray(
                    [
                        row["methods"][method][metric]
                        for row in p8_summary["per_seed"]
                    ],
                    dtype=float,
                )
                mean, half = ci95(values)
                means.append(mean)
                errors.append(half)
            positions = x + (method_index - 1) * width
            bars = metric_axis.bar(
                positions,
                means,
                width=width,
                color=COLORS[method],
                edgecolor="#596168",
                linewidth=0.35,
                yerr=errors,
                error_kw={
                    "ecolor": "#515960",
                    "elinewidth": 0.6,
                    "capsize": 1.6,
                    "capthick": 0.6,
                },
                label=DISPLAY[method],
                zorder=3,
            )
            legend_handles.append(bars[0])
            for bar, half in zip(bars, errors):
                metric_axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + half + 0.025,
                    f"{100 * bar.get_height():.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    color="#3D4348",
                )
        metric_axis.set_xticks(x)
        metric_axis.set_xticklabels([label for _, label in metrics])
        metric_axis.set_ylabel("Rate (%)")
        metric_axis.set_ylim(0.0, 1.08)
        metric_axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        metric_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        metric_axis.set_title("Cache effectiveness", loc="left", pad=3)
        metric_axis.text(
            -0.09,
            1.06,
            "b",
            transform=metric_axis.transAxes,
            fontsize=8.6,
            fontweight="bold",
        )
        metric_axis.grid(
            axis="y",
            color="#E1E4E6",
            linewidth=0.4,
            linestyle=(0, (2.5, 2.5)),
            zorder=0,
        )
        figure.legend(
            legend_handles,
            [DISPLAY[method] for method in PRIMARY_METHODS],
            loc="upper right",
            bbox_to_anchor=(0.99, 0.98),
            ncol=3,
            fontsize=6.5,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.0,
        )
        save_figure(
            figure,
            "fig05_cache_mechanism",
            additional_transformations=(
                "Nature visual preset",
                "cluster-level binary service availability averaged over seeds",
                "direct percentage annotations in panel (a)",
            ),
        )


def plot_heterogeneity_budget(p4_summary, p2_summary) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN, 67 * MM),
        layout="constrained",
    )
    profiles = ("uniform_b10", "heterogeneous_b10")
    x = np.arange(len(profiles))
    width = 0.28
    rng = np.random.default_rng(20260808)
    for offset, method in enumerate(CROSS_ENV_METHODS):
        method_offset = (offset - (len(CROSS_ENV_METHODS) - 1) / 2.0) * width
        means, errors = [], []
        profile_values = []
        for profile in profiles:
            values = np.asarray(
                [
                    row["methods"][method]["mean_finish_time"]
                    for row in p4_summary["heterogeneity"]["profiles"][profile]["per_seed"]
                ]
            )
            profile_values.append(values)
            mean, half = ci95(values)
            means.append(mean)
            errors.append(half)
        bars = axes[0].bar(
            x + method_offset,
            means,
            width=width,
            color=COLORS[method],
            edgecolor="#303030",
            yerr=errors,
            error_kw={"elinewidth": 0.7, "capsize": 2.0},
            label=DISPLAY[method],
        )
        for bar in bars:
            bar.set_hatch(HATCHES[method])
        for profile_index, values in enumerate(profile_values):
            jitter = rng.uniform(-0.025, 0.025, size=len(values))
            axes[0].scatter(
                np.full(len(values), x[profile_index] + method_offset) + jitter,
                values,
                s=10,
                facecolor="white",
                edgecolor="#303030",
                linewidth=0.4,
                zorder=3,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(("Uniform B=10", "Heterogeneous B=10"))
    axes[0].set_ylabel("Mean completion time (s)")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    style_axis(axes[0])
    panel_label(axes[0], "(a)")

    budgets = (5, 8, 10)
    sensitivity_methods = {
        "daoc_paper": "guided_full",
        "lean_our": "lean_our",
    }
    for method in CROSS_ENV_METHODS:
        source_method = sensitivity_methods[method]
        means, errors = [], []
        for budget in budgets:
            profile = f"B{budget}"
            values = [
                row["methods"][source_method]["mean_finish_time"]
                for row in p2_summary["per_profile_seed"][profile]
            ]
            mean, half = ci95(values)
            means.append(mean)
            errors.append(half)
        axes[1].errorbar(
            budgets,
            means,
            yerr=errors,
            color=COLORS[method],
            marker=MARKERS[method],
            markerfacecolor="white" if method != "lean_our" else COLORS[method],
            capsize=2.2,
            elinewidth=0.8,
            label=DISPLAY[method],
        )
    axes[1].set_xticks(budgets)
    axes[1].set_xlabel("Total cache budget B")
    axes[1].set_ylabel("Mean completion time (s)")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    axes[1].text(
        0.98,
        0.05,
        "n=3 seeds",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        color="#505050",
    )
    style_axis(axes[1])
    panel_label(axes[1], "(b)")
    save_figure(figure, "fig06_heterogeneity_budget")


def scaling_values(p4_summary, users: int, method: str, metric: str) -> np.ndarray:
    return np.asarray(
        [
            row["methods"][method][metric]
            for row in p4_summary["scaling"]["results"][str(users)]["per_seed"]
        ],
        dtype=float,
    )


def plot_scalability_overhead(p4_summary) -> None:
    users = tuple(p4_summary["scaling"]["user_counts"])
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(DOUBLE_COLUMN, 112 * MM),
        layout="constrained",
    )
    for axis, metric, ylabel, label in (
        (axes[0, 0], "mean_finish_time", "Mean completion time (s)", "(a)"),
        (axes[0, 1], "mean_p95_finish_time", "P95 completion time (s)", "(b)"),
    ):
        for method in CROSS_ENV_METHODS:
            means, errors = [], []
            for count in users:
                mean, half = ci95(scaling_values(p4_summary, count, method, metric))
                means.append(mean)
                errors.append(half)
            axis.errorbar(
                users,
                means,
                yerr=errors,
                color=COLORS[method],
                marker=MARKERS[method],
                markerfacecolor="white" if method != "lean_our" else COLORS[method],
                capsize=2.0,
                elinewidth=0.75,
                label=DISPLAY[method],
            )
        axis.set_xticks(users)
        axis.set_xlabel("Concurrent users")
        axis.set_ylabel(ylabel)
        axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
        style_axis(axis)
        panel_label(axis, label)

    def draw_overhead(axis, metric, ylabel, label, *, scale="linear", divisor=1.0):
        x = np.arange(len(CROSS_ENV_METHODS))
        values = {
            method: scaling_values(p4_summary, 20, method, metric) / divisor
            for method in CROSS_ENV_METHODS
        }
        means, errors = zip(
            *(ci95(values[method]) for method in CROSS_ENV_METHODS)
        )
        bars = axis.bar(
            x,
            means,
            width=0.62,
            color=[COLORS[method] for method in CROSS_ENV_METHODS],
            edgecolor="#303030",
            yerr=errors,
            error_kw={"elinewidth": 0.7, "capsize": 2.0},
            zorder=2,
        )
        for bar, method in zip(bars, CROSS_ENV_METHODS):
            bar.set_hatch(HATCHES[method])
        rng = np.random.default_rng(20260811)
        for index, method in enumerate(CROSS_ENV_METHODS):
            observations = values[method]
            jitter = rng.uniform(-0.10, 0.10, size=len(observations))
            axis.scatter(
                np.full(len(observations), index) + jitter,
                observations,
                s=8,
                facecolor="white",
                edgecolor="#303030",
                linewidth=0.4,
                zorder=3,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [DISPLAY[method] for method in CROSS_ENV_METHODS],
            rotation=12,
        )
        axis.set_ylabel(ylabel)
        axis.set_yscale(scale)
        style_axis(axis)
        panel_label(axis, label)

    draw_overhead(
        axes[0, 2],
        "inference_ms_per_task",
        "Policy inference (ms/task)",
        "(c) Inference",
        scale="log",
    )
    draw_overhead(
        axes[1, 0],
        "tasks_per_wall_second",
        "Evaluation throughput (tasks/s)",
        "(d) Throughput",
    )
    draw_overhead(
        axes[1, 1],
        "cache_decision_mean_ms",
        "Cache decision (ms/call)",
        "(e) Cache decision",
        scale="log",
    )
    draw_overhead(
        axes[1, 2],
        "coordination_bytes",
        "Coordination payload (KiB/call)",
        "(f) Communication",
        divisor=1024.0,
    )
    save_figure(figure, "fig07_scalability_overhead")


def convergence_curve(method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    for seed in FINAL_SEEDS:
        summary = read_json(run_directory(method, seed) / "summary.json")
        curves.append(
            {
                int(row["episode"]): float(row["mean_average_finish_time"])
                for row in summary["checkpoint_validation"]
            }
        )
    episodes = sorted(set.intersection(*(set(curve) for curve in curves)))
    means, lows, highs = [], [], []
    for episode in episodes:
        mean, half = ci95([curve[episode] for curve in curves])
        means.append(mean)
        lows.append(mean - half)
        highs.append(mean + half)
    return (
        np.asarray(episodes),
        np.asarray(means),
        np.asarray(lows),
        np.asarray(highs),
    )


def plot_convergence(
    p6_summary,
    p7_summary,
    p8_summary,
    *,
    reported_extension: bool = False,
) -> None:
    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN, 82 * MM),
        layout="constrained",
    )
    validation_curves = {
        method: convergence_curve(method) for method in LEARNING_METHODS
    }
    x_max = max(curves[0][-1] for curves in validation_curves.values())
    reference_x = np.linspace(0, x_max, 241)

    for method in HEURISTIC_METHODS:
        values = final_method_values(
            p6_summary,
            p7_summary,
            p8_summary,
            method,
            "mean_finish_time",
        )
        mean, _ = ci95(values)
        axis.plot(
            reference_x,
            np.full_like(reference_x, mean),
            markevery=24,
            **line_style.line_kwargs(method, label=DISPLAY[method]),
        )

    for method in LEARNING_METHODS:
        episodes, means, lows, highs = validation_curves[method]
        del lows, highs
        axis.plot(
            episodes,
            means,
            markevery=max(1, len(episodes) // 9),
            **line_style.line_kwargs(method, label=DISPLAY[method]),
        )
        if reported_extension and method in REPORTED_EXTENSION_METHODS:
            extension_end = max(
                int(x_max),
                REPORTED_EXTENSION_END_EPISODE,
            )
            extension_episodes = np.arange(
                int(episodes[-1]),
                extension_end + 1,
                500,
                dtype=float,
            )
            if extension_episodes[-1] != extension_end:
                extension_episodes = np.append(
                    extension_episodes,
                    float(extension_end),
                )
            progress = (
                (extension_episodes - extension_episodes[0])
                / (extension_episodes[-1] - extension_episodes[0])
            )
            start = float(means[-1])
            drop_low, drop_high = REPORTED_EXTENSION_DROP_RANGE
            extension_mid = start * (
                1.0 - REPORTED_EXTENSION_DROP_MIDPOINT * progress
            )
            extension_band_low = start * (1.0 - drop_high * progress)
            extension_band_high = start * (1.0 - drop_low * progress)
            axis.fill_between(
                extension_episodes,
                extension_band_low,
                extension_band_high,
                color=COLORS[method],
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            axis.plot(
                extension_episodes,
                extension_mid,
                color=COLORS[method],
                linestyle=(0, (7.0, 2.5)),
                linewidth=1.65,
                label=f"{DISPLAY[method]} continuation (reported)",
                zorder=4,
            )

    axis.set_xlim(0, x_max)
    axis.set_ylim(bottom=0)
    axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5000))
    axis.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}")
    )
    axis.set_xlabel(r"Training episode ($\times 10^3$)")
    axis.set_ylabel("Frozen-validation DAG completion time (s)")
    line_style.style_axis(axis, grid_axis="y")
    line_style.legend_above(axis, ncol=3)
    note = "10-seed mean; fixed validation scenarios; no smoothing"
    if reported_extension:
        note = (
            "Marked segments: observed fixed-validation checkpoints\n"
            "Marker-free tails: reported 0.5-1.0% continuations "
            "(0.75% midpoint shown)"
        )
    axis.text(
        0.03,
        0.04,
        note,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#505050",
    )
    if reported_extension:
        save_figure(
            figure,
            "fig08b_convergence_reported_extension",
            additional_transformations=(
                "marker-free dashed segments for OUR and SAC + DCC "
                "are deterministic visual reconstructions of the user-reported "
                "0.5%-1.0% declines, not raw per-checkpoint data",
                "each dashed midpoint ends 0.75% below its method's last observed "
                "common checkpoint at episode 26000",
            ),
        )
    else:
        save_figure(figure, "fig08_convergence")


def plot_oracle_gap(p3_summary, p8_summary) -> None:
    oracle_rows = read_csv(ORACLE_SEED_PATH)
    methods = (
        "perfect_cache_floor",
        "oracle_floor",
        "lean_our",
        "daoc_our_coord_cache",
        "daoc_paper",
    )
    values = {
        "perfect_cache_floor": np.asarray([float(row["perfect_cache_floor"]) for row in oracle_rows]),
        "oracle_floor": np.asarray([float(row["oracle_floor"]) for row in oracle_rows]),
        "lean_our": method_values(p3_summary, "lean_our", "mean_finish_time"),
        "daoc_our_coord_cache": method_values(
            p8_summary, "daoc_our_coord_cache", "mean_finish_time"
        ),
        "daoc_paper": method_values(p3_summary, "daoc_paper", "mean_finish_time"),
    }
    figure, axis = plt.subplots(
        figsize=(SINGLE_COLUMN, 70 * MM),
        layout="constrained",
    )
    y = np.arange(len(methods))
    means, errors = zip(*(ci95(values[method]) for method in methods))
    bars = axis.barh(
        y,
        means,
        xerr=errors,
        height=0.62,
        color=[COLORS[method] for method in methods],
        edgecolor="#303030",
        error_kw={"elinewidth": 0.7, "capsize": 2.0},
    )
    bars[0].set_hatch("xx")
    bars[1].set_hatch("//")
    axis.set_yticks(y)
    axis.set_yticklabels([DISPLAY[method] for method in methods])
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Mean completion time (s, log scale)")
    style_axis(axis, grid="x")
    save_figure(figure, "fig09_oracle_gap_appendix")


def write_manifest() -> None:
    # The final manifest is maintained as a protocol-facing artifact because it
    # also indexes split figures produced by plot_split_topconf_figures.py.
    # Do not overwrite it when regenerating the legacy nine-figure bundle.
    return


def write_legacy_manifest_content() -> None:
    content = f"""# Final Experimental Figure Set

Generated with the project-local `scientific-visualization` skill from
`{SCI_VIS_REPOSITORY}` at commit `{SCI_VIS_COMMIT}`. The plotting script loads
the skill's publication style and atomic figure exporter. It uses a restrained,
method-family-aware palette; color is reinforced with labels, markers, or
hatches for grayscale use.

Central-Greedy is excluded from the formal set. Every coordinated-cache control
shown here uses the measured DAOC + DCC results from P8 (seeds 51--60),
not relabeled Central-Greedy values.

Figures 1, 2, 3, 4, 5, 8, and 9 use the unified paper method names. Figures 6
and 7 retain their approved `figures_scivis_v3` artwork.

All error bars and confidence bands are 95% Student-t confidence intervals over
independent seed-level means. Scenario-level samples are never treated as
independent statistical units.

| File | Suggested caption | Statistical scope |
|---|---|---|
| `fig01_main_baselines` | Mean and P95 DAG completion time for all formal baselines; inset shows paired OUR reductions over SAC + DCC. | 10 seeds, 100 paired scenarios/seed |
| `fig02_seed_robustness` | Seed-level paired reduction of OUR over each baseline. | 10 paired seed means |
| `fig03_workflow_latency` | Performance of all formal baselines across five Pegasus families and frozen-checkpoint Alibaba-CP100 cross-dataset evaluation; panel (b) retains the primary-method latency composition. | 10 seeds, 100 paired scenarios/seed |
| `fig04_ablation` | Controlled scheduler replacement under DAOC-Cache and DCC, covering DAOC, Flat DDQN, Discrete SAC, and Pairwise PD3QN. | 10 paired seeds |
| `fig05_cache_mechanism` | Cluster-level service-availability matrix and cache-effectiveness metrics. | 10 seeds |
| `fig06_heterogeneity_budget` | Capacity heterogeneity and cache-budget sensitivity. | 3 development seeds; trend evidence only |
| `fig07_scalability_overhead` | Frozen-policy scalability and deployment overhead. | 10 seeds, 50 paired scenarios/scale |
| `fig08_convergence` | Online episode-latency comparison over the common 26k training horizon; heuristic methods are horizontal online-tail references. | 10-seed mean; 500-episode trailing mean |
| `fig09_oracle_gap_appendix` | Gap to optimistic oracle references. | Appendix diagnostic only; not a formal lower-bound claim |

Newly rendered figures use exact 88-mm single-column or 180-mm double-column
widths and are emitted as PDF/SVG plus 400-dpi PNG. Preserved Figures 6 and 7
retain their approved PDF/PNG artifacts.
Exports use `bbox_inches=None`, so their physical dimensions are not silently
changed. The source results remain unchanged.
"""
    (OUTPUT_DIR / "FIGURE_MANIFEST.md").write_text(content, encoding="utf-8")
    usage = """# 最终实验结果图使用说明

本图集共九张正式图。所有 `DAOC + DCC` 数值都来自P8的seeds 51--60真实重训与评估结果；不是将Central-Greedy旧数值改名。

本次统一替换图中方法名称，不改变方法键、数据、排序或统计方式。Fig. 6和7沿用 `figures_scivis_v3` 的图形。

## 主文建议

1. `fig01_main_baselines`：主结果图，回答OUR是否优于启发式、DAOC、DQN和Discrete SAC；右上局部面板专门展示OUR与同缓存SAC + DCC的逐seed配对差值。
2. `fig03_workflow_latency`：左图纳入全部九种正式方法，回答优势是否覆盖五类Pegasus工作流，并用同一B8环境下的Alibaba-CP100冻结checkpoint测试补充跨数据集证据；右图保持三种核心方法的时延构成。Alibaba-CP100带星号，表示其为选定压力集上的零样本测试，不是无偏holdout，也不是旧预算20实验。
3. `fig04_ablation`：参照DAOC Fig. 5的方法替换型消融，将DAOC、Flat DDQN、Discrete SAC和Pairwise PD3QN分别与DAOC-Cache、DCC成对组合，同时检验调度器选择与协调缓存的贡献；Pairwise PD3QN + DCC为完整OUR。
4. `fig05_cache_mechanism`：用逐服务的集群可用性矩阵，以及命中率、覆盖率和远程加载率解释协调缓存为什么有效。
5. `fig06_heterogeneity_budget`：对应论文的异构缓存背景，并展示缓存预算变化趋势。该图只有3个开发seed，只能作为趋势证据。
6. `fig07_scalability_overhead`：回答部署规模、推理时间、缓存协调时间和通信量问题。

## 补充材料建议

7. `fig08_convergence`：参考DAOC训练曲线的单图结构，使用所有学习方法的实测26k episode在线训练日志；曲线为10 seed平均的500-episode trailing mean，启发式方法使用各seed训练尾部均值水平线。

## 补充材料建议

8. `fig02_seed_robustness`：展示逐seed配对改善及95%置信区间，支撑统计稳健性。
9. `fig09_oracle_gap_appendix`：展示与乐观Oracle参考的距离，仅作诊断，不能写成严格可达下界。

## 不再使用

- Central-Greedy相关图和数值只作历史审计，不进入正式对比、消融或结论。
- 含人工延长尾段的 `fig08b_convergence_reported_extension` 不属于正式图集。
- 旧HCPR、smoke、screen和失败诊断图不与最终Pegasus十seed结果混用。

图中误差条均为以独立seed均值为统计单位的双侧95% Student-t置信区间。主基线来自P6公平重跑结果；SAC + DAOC-Cache来自P7；DAOC + DCC来自P8。Central-Greedy仅保留为历史审计结果，不进入正式论文图表。
"""
    (OUTPUT_DIR / "FIGURE_USAGE_ZH.md").write_text(usage, encoding="utf-8")
    input_paths = (
        P3_SUMMARY_PATH,
        P4_SUMMARY_PATH,
        P5_SUMMARY_PATH,
        P6_SUMMARY_PATH,
        P7_SUMMARY_PATH,
        P8_SUMMARY_PATH,
        P10_SUMMARY_PATH,
        P2_SUMMARY_PATH,
        ORACLE_SEED_PATH,
    )
    provenance = {
        "generator": str(Path(__file__).relative_to(WORKSPACE_ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "skill": {
            "name": "scientific-visualization",
            "repository": SCI_VIS_REPOSITORY,
            "commit": SCI_VIS_COMMIT,
            "path": str(SCI_VIS_SKILL.relative_to(WORKSPACE_ROOT)),
            "style": str(SCI_VIS_STYLE.relative_to(WORKSPACE_ROOT)),
            "palette": "nature-inspired editorial palette with redundant encodings",
            "exporter": str(SCI_VIS_EXPORTER.relative_to(WORKSPACE_ROOT)),
        },
        "statistical_unit": "independent seed-level mean",
        "uncertainty": "two-sided 95% Student-t confidence interval",
        "data_policy": (
            "fig08 is generated separately from measured 26k online-training "
            "logs using a declared 500-episode trailing mean; "
            "no selective filtering or scenario-level pseudo-replication"
        ),
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_paths
        },
        "preserved_previous_figures": {
            stem: {
                suffix: {
                    "source": str(
                        (source_directory / f"{stem}.{suffix}").relative_to(ROOT)
                    ),
                    "sha256": hashlib.sha256(
                        (source_directory / f"{stem}.{suffix}").read_bytes()
                    ).hexdigest(),
                }
                for suffix in ("png", "pdf")
            }
            for stem, source_directory in PRESERVED_PREVIOUS_FIGURES.items()
        },
        "physical_width_mm": {"single_column": 88, "double_column": 180},
        "exports": {
            "pdf": "vector with embedded TrueType fonts",
            "svg": "editable vector for newly rendered figures",
            "png": "400 dpi for newly rendered figures; approved metadata retained for preserved figures",
            "bbox_inches": None,
        },
    }
    (OUTPUT_DIR / "FIGURE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_inputs() -> None:
    required = (
        P3_SUMMARY_PATH,
        P4_SUMMARY_PATH,
        P6_SUMMARY_PATH,
        P7_SUMMARY_PATH,
        P8_SUMMARY_PATH,
        P10_SUMMARY_PATH,
        P2_SUMMARY_PATH,
        ORACLE_SEED_PATH,
        SCI_VIS_STYLE,
        SCI_VIS_EXPORTER,
        NATURE_STYLE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal-result inputs:\n" + "\n".join(missing))


def validate_formal_results(p6_summary, p7_summary, p8_summary) -> None:
    for label, summary in (("P6", p6_summary), ("P7", p7_summary), ("P8", p8_summary)):
        failed = [key for key, value in summary["integrity"].items() if value is not True]
        if failed:
            raise RuntimeError(f"{label} integrity checks failed: {failed}")
        seeds = tuple(int(row["seed"]) for row in summary["per_seed"])
        if seeds != FINAL_SEEDS:
            raise RuntimeError(f"{label} seed order mismatch: {seeds}")

    p6_sha256 = hashlib.sha256(P6_SUMMARY_PATH.read_bytes()).hexdigest()
    p7_sha256 = hashlib.sha256(P7_SUMMARY_PATH.read_bytes()).hexdigest()
    if p7_summary["p6_summary_sha256"] != p6_sha256:
        raise RuntimeError("P7 does not reference the current frozen P6 summary")
    if p8_summary["parent_summary_hashes"] != {"p6": p6_sha256, "p7": p7_sha256}:
        raise RuntimeError("P8 parent summary hashes do not match frozen P6/P7")


def validate_cross_dataset_results() -> None:
    summary = read_json(P10_SUMMARY_PATH)
    expected = {
        "status": "complete",
        "protocol_version": "pegasus_b8_frozen_alibaba_cp100_v1",
        "dataset_sha256": (
            "2903ff2478f5c55fe445bd2a7b6fbe595aecf6ea6383913b85ea5efd92ee2d89"
        ),
        "only_changed_factor": "DAG dataset",
        "paired_scenarios_across_methods": True,
        "cache_budget": 8,
        "checkpoint_weights_frozen": True,
        "cache_and_history_frozen": True,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"P10 cross-dataset protocol mismatch: {mismatches}")
    if tuple(summary.get("methods", ())) != MAIN_METHODS:
        raise RuntimeError("P10 method set does not match the formal baselines")
    if tuple(summary.get("seeds", ())) != FINAL_SEEDS:
        raise RuntimeError("P10 seed set does not match the formal experiment")
    if summary.get("evaluation_scenarios_per_seed") != 100:
        raise RuntimeError("P10 must contain 100 paired scenarios per seed")


def main() -> None:
    validate_inputs()
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p3_summary = read_json(P3_SUMMARY_PATH)
    p4_summary = read_json(P4_SUMMARY_PATH)
    p6_summary = read_json(P6_SUMMARY_PATH)
    p7_summary = read_json(P7_SUMMARY_PATH)
    p8_summary = read_json(P8_SUMMARY_PATH)
    p2_summary = read_json(P2_SUMMARY_PATH)
    validate_formal_results(p6_summary, p7_summary, p8_summary)
    validate_cross_dataset_results()

    plot_main_baselines(p6_summary, p7_summary, p8_summary)
    plot_seed_robustness(p6_summary, p7_summary, p8_summary)
    plot_workflow_latency()
    plot_ablation(p6_summary, p7_summary, p8_summary)
    plot_cache_mechanism(p8_summary)
    plot_oracle_gap(p3_summary, p8_summary)
    from plot_common_horizon_online_curves import (
        collect_curves,
        collect_heuristic_levels,
        draw_topconf_comparison,
    )

    episodes, curves, _ = collect_curves(window=500, stride=1)
    heuristic_levels, _ = collect_heuristic_levels()
    draw_topconf_comparison(
        OUTPUT_DIR,
        episodes,
        curves,
        heuristic_levels,
        output_stem="fig08_convergence",
    )
    restore_previous_figures()
    write_manifest()
    print(
        f"Wrote 9 measured publication figures to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
