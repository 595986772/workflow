#!/usr/bin/env python3
"""Create the single-table, single-figure minimal E2/E3 ablation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_strict_environment_suite import paired_statistics


LABELS = (
    "lean_our",
    "our_no_telemetry",
    "our_no_coord_cache",
)
DISPLAY = {
    "lean_our": "Full OUR",
    "our_no_telemetry": "No telemetry",
    "our_no_coord_cache": "No coordinated cache",
}


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def eval_rows(path):
    with (path / "episodes.csv").open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        return [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]


def effective_delay(summary):
    delay = summary["adaptation"]["adaptation_delay_windows"]
    if delay is not None:
        return float(delay)
    protocol = summary["protocol"]
    return float(
        protocol["episodes"] - protocol["shift_episode"] + 2
    )


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    rows = []
    values = {
        label: {
            "e2_finish_time": [],
            "e2_remote_loading_rate": [],
            "e3_recovery_delay": [],
            "e3_oracle_regret": [],
        }
        for label in LABELS
    }
    for seed in args.seeds:
        for label in LABELS:
            path = suite_dir / "runs" / label / f"seed_{seed}"
            static_rows = eval_rows(path)
            online = read_json(
                path / "online_stream_e3_load_shift_summary.json"
            )
            record = {
                "seed": seed,
                "label": label,
                "e2_finish_time": float(
                    np.mean(
                        [
                            float(row["average_finish_time"])
                            for row in static_rows
                        ]
                    )
                ),
                "e2_remote_loading_rate": float(
                    np.mean(
                        [
                            float(row["cache_remote_loading_rate"])
                            for row in static_rows
                        ]
                    )
                ),
                "e3_recovery_delay": effective_delay(online),
                "e3_oracle_regret": float(
                    online["adaptation"][
                        "cumulative_oracle_regret"
                    ]
                ),
            }
            rows.append(record)
            for metric in values[label]:
                values[label][metric].append(record[metric])

    comparisons = {}
    for ablation in LABELS[1:]:
        comparisons[ablation] = {}
        for metric in values["lean_our"]:
            comparisons[ablation][metric] = paired_statistics(
                values[ablation][metric],
                values["lean_our"][metric],
                lower_is_better=True,
            )
    summary = {
        "status": "complete",
        "seeds": args.seeds,
        "means": {
            label: {
                metric: float(np.mean(metric_values))
                for metric, metric_values
                in label_values.items()
            }
            for label, label_values in values.items()
        },
        "full_our_vs_ablation": comparisons,
        "interpretation": {
            "e2": "Full OUR versus no coordinated cache isolates caching.",
            "e3": "Full OUR versus no telemetry isolates online adaptation.",
            "selection_use": False,
        },
    }
    (suite_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (suite_dir / "ablation_table.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.3),
        constrained_layout=True,
    )
    x = np.arange(len(LABELS))
    colors = ("#2563EB", "#6B7280", "#DC2626")
    for axis, metric, title, ylabel in (
        (
            axes[0],
            "e2_finish_time",
            "E2 static heterogeneous cache",
            "Mean finish time (s)",
        ),
        (
            axes[1],
            "e3_oracle_regret",
            "E3 load-shift adaptation",
            "Cumulative Oracle regret",
        ),
    ):
        means = [summary["means"][label][metric] for label in LABELS]
        errors = [
            float(np.std(values[label][metric], ddof=1))
            / np.sqrt(len(args.seeds))
            if len(args.seeds) > 1
            else 0.0
            for label in LABELS
        ]
        axis.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=colors,
            width=0.68,
        )
        axis.set_xticks(
            x,
            [DISPLAY[label] for label in LABELS],
            rotation=12,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.savefig(suite_dir / "ablation_figure.png", dpi=220)
    figure.savefig(suite_dir / "ablation_figure.pdf")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
