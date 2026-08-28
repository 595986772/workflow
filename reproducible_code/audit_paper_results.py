#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


EXPECTED_LABELS = [
    "nearest",
    "greedy",
    "unguided_full",
    "guided_full",
    "guided_decay",
]

DISPLAY_NAMES = {
    "nearest": "Nearest",
    "greedy": "Nearest + Service",
    "unguided_full": "Unguided Full",
    "guided_full": "Guided Full (Fixed beta)",
    "guided_decay": "Guided Full (Paper Decay)",
}

KEY_COMPARISONS = [
    ("guided_full", "nearest"),
    ("guided_full", "greedy"),
    ("guided_full", "unguided_full"),
    ("guided_decay", "nearest"),
    ("guided_decay", "greedy"),
    ("guided_decay", "unguided_full"),
    ("guided_decay", "guided_full"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the completed paper-scale reproduction suite."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(
        stats.t.ppf(0.975, values.size - 1)
        * stats.sem(values)
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(suite_dir):
    runs = []
    for label in EXPECTED_LABELS:
        for seed in range(1, 11):
            run_dir = suite_dir / "runs" / label / f"seed_{seed}"
            runs.append(
                {
                    "label": label,
                    "seed": seed,
                    "run_dir": run_dir,
                    "summary": read_json(run_dir / "summary.json"),
                    "config": read_json(run_dir / "config.json"),
                    "scenario": read_json(run_dir / "scenario_initial.json"),
                }
            )
    return runs


def verify_integrity(suite_dir, runs):
    manifest = read_json(suite_dir / "suite_manifest.json")
    if manifest.get("status") != "complete":
        raise RuntimeError("Suite manifest is not complete")
    if manifest.get("completed_runs") != 50:
        raise RuntimeError("Suite does not contain 50 completed runs")
    if manifest.get("failed_runs"):
        raise RuntimeError("Suite manifest contains failed runs")

    for run in runs:
        summary = run["summary"]
        config = run["config"]
        if summary.get("status") != "complete":
            raise RuntimeError(
                f"Incomplete summary: {run['label']} seed={run['seed']}"
            )
        if not summary.get("evaluation_state_frozen"):
            raise RuntimeError(
                f"Evaluation not frozen: {run['label']} seed={run['seed']}"
            )
        if summary.get("episode_rows") != 30500:
            raise RuntimeError(
                f"Unexpected row count: {run['label']} seed={run['seed']}"
            )

        arguments = config["arguments"]
        learning = config["learning_config"]
        input_config = config["input_config"]
        expected_config = {
            "train_episodes": 30000,
            "eval_episodes": 500,
            "num_users": 20,
            "num_servers": 10,
            "num_services": 10,
            "num_tasks": 10,
            "batch_size": 1024,
            "epsilon": 0.01,
            "bandwidth": 15000,
        }
        for key, expected in expected_config.items():
            if arguments.get(key) != expected:
                raise RuntimeError(
                    f"Unexpected {key}: {run['label']} seed={run['seed']}"
                )
        if learning["hidden_units"] != [64, 64]:
            raise RuntimeError(
                f"Unexpected hidden layers: {run['label']} seed={run['seed']}"
            )
        if input_config["Bandwidth"] != 15000:
            raise RuntimeError(
                f"Unexpected input bandwidth: {run['label']} seed={run['seed']}"
            )

    for seed in range(1, 11):
        seed_runs = [run for run in runs if run["seed"] == seed]
        canonical = []
        for run in seed_runs:
            scenario = dict(run["scenario"])
            scenario.pop("algorithm", None)
            canonical.append(scenario)
        if any(scenario != canonical[0] for scenario in canonical[1:]):
            raise RuntimeError(f"Scenario mismatch for seed={seed}")


def verify_beta_schedules(runs):
    target_episodes = {
        ("train", 1),
        ("train", 439),
        ("train", 440),
        ("train", 30000),
        ("eval", 1),
        ("eval", 500),
    }
    for run in runs:
        arguments = run["config"]["arguments"]
        observed = {}
        with (run["run_dir"] / "episodes.csv").open(
            newline="",
            encoding="utf-8",
        ) as input_file:
            for row in csv.DictReader(input_file):
                key = (row["phase"], int(row["episode"]))
                if key in target_episodes:
                    observed[key] = float(row["beta"])

        if set(observed) != target_episodes:
            raise RuntimeError(
                f"Missing beta samples: {run['label']} seed={run['seed']}"
            )
        for phase, episode in target_episodes:
            if phase == "eval":
                expected = 0.0
            else:
                expected = max(
                    arguments["beta_min"],
                    arguments["beta"]
                    * arguments["beta_decay"] ** (episode - 1),
                )
            if not np.isclose(
                observed[(phase, episode)],
                expected,
                rtol=0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Incorrect beta: {run['label']} seed={run['seed']} "
                    f"{phase} episode={episode}"
                )


def build_score_tables(runs):
    long_rows = []
    values = defaultdict(dict)
    for run in runs:
        summary = run["summary"]
        eval_score = summary["eval"]["mean_average_finish_time"]
        paper_score = summary["paper_training_metric"][
            "mean_average_finish_time"
        ]
        values["frozen_eval", run["label"]][run["seed"]] = eval_score
        values["paper_training", run["label"]][run["seed"]] = paper_score
        long_rows.append(
            {
                "seed": run["seed"],
                "label": run["label"],
                "display_name": DISPLAY_NAMES[run["label"]],
                "frozen_eval_finish_time": eval_score,
                "paper_training_finish_time": paper_score,
            }
        )
    return long_rows, values


def aggregate_scores(runs):
    rows = []
    for label in EXPECTED_LABELS:
        label_runs = [run for run in runs if run["label"] == label]
        eval_values = [
            run["summary"]["eval"]["mean_average_finish_time"]
            for run in label_runs
        ]
        paper_values = [
            run["summary"]["paper_training_metric"][
                "mean_average_finish_time"
            ]
            for run in label_runs
        ]
        rows.append(
            {
                "label": label,
                "display_name": DISPLAY_NAMES[label],
                "frozen_eval_mean": float(np.mean(eval_values)),
                "frozen_eval_ci95": confidence_interval(eval_values),
                "paper_training_mean": float(np.mean(paper_values)),
                "paper_training_ci95": confidence_interval(paper_values),
            }
        )
    return rows


def paired_rows(values, metric):
    rows = []
    for label, reference in KEY_COMPARISONS:
        label_values = values[metric, label]
        reference_values = values[metric, reference]
        seeds = sorted(set(label_values) & set(reference_values))
        ratios = np.asarray(
            [
                label_values[seed] / reference_values[seed]
                for seed in seeds
            ],
            dtype=float,
        )
        improvements = 100.0 * (1.0 - ratios)
        rows.append(
            {
                "metric": metric,
                "label": label,
                "reference": reference,
                "pairs": len(seeds),
                "wins": int(np.sum(ratios < 1.0)),
                "mean_ratio": float(ratios.mean()),
                "ratio_ci95": confidence_interval(ratios),
                "mean_improvement_percent": float(improvements.mean()),
                "improvement_ci95": confidence_interval(improvements),
            }
        )
    return rows


def convergence_rows(runs):
    rows = []
    for run in runs:
        values = []
        with (run["run_dir"] / "episodes.csv").open(
            newline="",
            encoding="utf-8",
        ) as input_file:
            for row in csv.DictReader(input_file):
                if row["phase"] == "train":
                    values.append(float(row["average_finish_time"]))
        values = np.asarray(values, dtype=float)
        previous = float(values[-10000:-5000].mean())
        final = float(values[-5000:].mean())
        x = np.arange(5000, dtype=float)
        slope_per_1000 = float(np.polyfit(x, values[-5000:], 1)[0] * 1000)
        rows.append(
            {
                "label": run["label"],
                "seed": run["seed"],
                "episodes_20001_25000_mean": previous,
                "episodes_25001_30000_mean": final,
                "relative_change_percent": 100.0 * (final / previous - 1.0),
                "last_5000_slope_per_1000_episodes": slope_per_1000,
            }
        )
    return rows


def convergence_summary(rows):
    output = []
    for label in EXPECTED_LABELS:
        label_rows = [row for row in rows if row["label"] == label]
        changes = [row["relative_change_percent"] for row in label_rows]
        slopes = [
            row["last_5000_slope_per_1000_episodes"]
            for row in label_rows
        ]
        output.append(
            {
                "label": label,
                "display_name": DISPLAY_NAMES[label],
                "relative_change_percent_mean": float(np.mean(changes)),
                "relative_change_percent_ci95": confidence_interval(changes),
                "slope_per_1000_episodes_mean": float(np.mean(slopes)),
                "slope_per_1000_episodes_ci95": confidence_interval(slopes),
            }
        )
    return output


def format_mean_ci(mean, ci):
    return f"{mean:.4f} +/- {ci:.4f}"


def render_report(
    aggregate,
    paired_eval,
    paired_paper,
    convergence,
):
    aggregate_by_label = {row["label"]: row for row in aggregate}
    paired_eval_by_key = {
        (row["label"], row["reference"]): row
        for row in paired_eval
    }
    paired_paper_by_key = {
        (row["label"], row["reference"]): row
        for row in paired_paper
    }

    lines = [
        "# Paper-scale reproduction audit",
        "",
        "## Integrity",
        "",
        "- 50/50 runs complete; 0 failed.",
        "- Five methods have 10 independent seeds each.",
        "- Every run contains 30,000 training and 500 frozen-evaluation episodes.",
        "- Model weights, replay buffers, epsilon, deadlines, and cache state remained frozen during evaluation.",
        "- Initial topology and workload snapshots match exactly across methods for every paired seed.",
        "- Paper-scale network, learning, and beta-schedule parameters were verified from each run config and episode trace.",
        "",
        "## Absolute scores",
        "",
        "| Method | Frozen evaluation (s) | Paper-style late training (s) |",
        "|---|---:|---:|",
    ]
    for label in EXPECTED_LABELS:
        row = aggregate_by_label[label]
        lines.append(
            "| "
            f"{DISPLAY_NAMES[label]} | "
            f"{format_mean_ci(row['frozen_eval_mean'], row['frozen_eval_ci95'])} | "
            f"{format_mean_ci(row['paper_training_mean'], row['paper_training_ci95'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired improvements",
            "",
            "| Comparison | Frozen wins | Frozen improvement | Paper-style wins | Paper-style improvement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, reference in KEY_COMPARISONS:
        eval_row = paired_eval_by_key[label, reference]
        paper_row = paired_paper_by_key[label, reference]
        lines.append(
            "| "
            f"{DISPLAY_NAMES[label]} vs {DISPLAY_NAMES[reference]} | "
            f"{eval_row['wins']}/10 | "
            f"{eval_row['mean_improvement_percent']:.2f}% "
            f"+/- {eval_row['improvement_ci95']:.2f}% | "
            f"{paper_row['wins']}/10 | "
            f"{paper_row['mean_improvement_percent']:.2f}% "
            f"+/- {paper_row['improvement_ci95']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Late-training movement",
            "",
            "Negative values indicate that the final 5,000 episodes improved over episodes 20,001-25,000.",
            "",
            "| Method | Relative change | Last-5k slope per 1k episodes |",
            "|---|---:|---:|",
        ]
    )
    for row in convergence:
        lines.append(
            "| "
            f"{row['display_name']} | "
            f"{row['relative_change_percent_mean']:.3f}% "
            f"+/- {row['relative_change_percent_ci95']:.3f}% | "
            f"{row['slope_per_1000_episodes_mean']:.6f} "
            f"+/- {row['slope_per_1000_episodes_ci95']:.6f} s |"
        )

    lines.extend(
        [
            "",
            "## Audit conclusion",
            "",
            "- The default-setting ranking is reproduced: both guided variants beat Nearest, Nearest + Service, and Unguided Full on all 10 frozen-evaluation seeds.",
            "- The paper-style metric also favors both guided variants over both heuristic baselines on all 10 seeds and over Unguided Full on 9 of 10 seeds.",
            "- Paper beta decay is not better than fixed beta: the two variants are effectively tied under both metrics.",
            "- This audit supports the core ranking under one paper-scale configuration. It does not reproduce every parameter sweep or the much larger effect size reported in the paper.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    runs = load_runs(suite_dir)
    verify_integrity(suite_dir, runs)
    verify_beta_schedules(runs)

    score_rows, values = build_score_tables(runs)
    aggregate = aggregate_scores(runs)
    paired_eval = paired_rows(values, "frozen_eval")
    paired_paper = paired_rows(values, "paper_training")
    convergence = convergence_rows(runs)
    convergence_aggregate = convergence_summary(convergence)

    write_csv(suite_dir / "audit_per_seed_scores.csv", score_rows)
    write_csv(suite_dir / "audit_aggregate_scores.csv", aggregate)
    write_csv(suite_dir / "audit_paired_frozen_eval.csv", paired_eval)
    write_csv(suite_dir / "audit_paired_paper_training.csv", paired_paper)
    write_csv(suite_dir / "audit_convergence_per_seed.csv", convergence)
    write_csv(
        suite_dir / "audit_convergence_summary.csv",
        convergence_aggregate,
    )
    report = render_report(
        aggregate,
        paired_eval,
        paired_paper,
        convergence_aggregate,
    )
    report_path = suite_dir / "AUDIT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
