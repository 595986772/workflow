#!/usr/bin/env python3
"""Audit whether historical telemetry PD3QN results can represent Lean OUR."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from information_protocol import INFORMATION_PROTOCOL_VERSION
from run_reproduction_suite import (
    ALGORITHMS,
    PROFILES,
    effective_method_profile,
)


OLD_LABEL = "telemetry_pd3qn"
CURRENT_LABEL = "lean_our"

AUDIT_GROUPS = {
    "network": (
        "algorithm",
        "gamma",
        "n_step",
        "num_quantiles",
        "risk_tail_fraction",
        "entropy_coefficient",
        "priority_alpha",
        "priority_beta_start",
        "priority_beta_anneal_steps",
        "criticality_boost",
    ),
    "reward": (
        "reward_mode",
        "potential_reward_weight",
        "training_objective",
    ),
    "cache": (
        "cache_policy",
        "cache_server_quality",
    ),
    "information": (
        "historical_feedback_guidance",
        "adaptive_guidance_gate",
        "active_modules",
        "excluded_modules",
    ),
}

PROFILE_FIELDS = (
    "train_episodes",
    "checkpoint_every",
    "validation_scenarios",
    "convergence_mode",
    "convergence_min_episodes",
    "convergence_window",
    "convergence_patience",
    "num_users",
    "num_servers",
    "num_services",
    "num_tasks",
    "bandwidth",
    "server_capacity",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit E0 result reuse eligibility."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    return parser.parse_args()


def algorithm_config(label):
    for config in ALGORITHMS:
        if config["label"] == label:
            return dict(config)
    raise KeyError(label)


def normalized_field(config, field):
    defaults = {
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "potential_reward_weight": 0.0,
        "cache_server_quality": True,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [],
        "excluded_modules": [],
        "training_objective": None,
    }
    return config.get(field, defaults.get(field))


def compare_group(old, current, fields):
    rows = []
    for field in fields:
        old_value = normalized_field(old, field)
        current_value = normalized_field(current, field)
        rows.append(
            {
                "field": field,
                "historical": old_value,
                "lean_our": current_value,
                "equal": old_value == current_value,
            }
        )
    return {
        "equal": all(row["equal"] for row in rows),
        "fields": rows,
    }


def profile_snapshot(label):
    profile = effective_method_profile(
        PROFILES["strict_stress_converged_3seed"],
        label,
    )
    return {
        field: profile.get(field)
        for field in PROFILE_FIELDS
    }


def historical_configs(root):
    matches = []
    if not root.exists():
        return matches
    for path in root.rglob("config.json"):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        arguments = config.get("arguments", config)
        if arguments.get("label") != OLD_LABEL:
            continue
        matches.append(
            {
                "path": str(path),
                "experiment_config_sha256": config.get(
                    "experiment_config_sha256"
                ),
                "scenario_protocol_present": bool(
                    config.get("experiment_config_sha256")
                ),
            }
        )
    return matches


def build_audit(historical_root):
    old = algorithm_config(OLD_LABEL)
    current = algorithm_config(CURRENT_LABEL)
    groups = {
        name: compare_group(old, current, fields)
        for name, fields in AUDIT_GROUPS.items()
    }
    old_profile = profile_snapshot(OLD_LABEL)
    current_profile = profile_snapshot(CURRENT_LABEL)
    profile_fields = [
        {
            "field": field,
            "historical": old_profile[field],
            "lean_our": current_profile[field],
            "equal": old_profile[field] == current_profile[field],
        }
        for field in PROFILE_FIELDS
    ]
    profile_equal = all(row["equal"] for row in profile_fields)
    historical = historical_configs(historical_root)
    historical_protocol_complete = bool(historical) and all(
        row["scenario_protocol_present"] for row in historical
    )
    behavior_equivalent = (
        all(group["equal"] for group in groups.values())
        and profile_equal
        and historical_protocol_complete
    )
    return {
        "status": "complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "historical_label": OLD_LABEL,
        "current_label": CURRENT_LABEL,
        "information_protocol_version": INFORMATION_PROTOCOL_VERSION,
        "groups": groups,
        "training_protocol": {
            "equal": profile_equal,
            "fields": profile_fields,
        },
        "historical_artifacts": historical,
        "historical_protocol_complete": historical_protocol_complete,
        "behavior_equivalent": behavior_equivalent,
        "reuse_allowed": behavior_equivalent,
        "decision": (
            "reuse_historical_e0"
            if behavior_equivalent
            else "run_new_e0_daoc_vs_lean_our"
        ),
    }


def markdown_lines(audit, chinese=False):
    if chinese:
        lines = [
            "# E0兼容性审计",
            "",
            f"- 旧方法：`{audit['historical_label']}`。",
            f"- 当前方法：`{audit['current_label']}`。",
            f"- 行为等价：{audit['behavior_equivalent']}。",
            f"- 旧结果允许复用：{audit['reuse_allowed']}。",
            f"- 结论：`{audit['decision']}`。",
            "",
            "## 字段比较",
            "",
        ]
    else:
        lines = [
            "# E0 Compatibility Audit",
            "",
            f"- Historical method: `{audit['historical_label']}`.",
            f"- Current method: `{audit['current_label']}`.",
            f"- Behavior equivalent: {audit['behavior_equivalent']}.",
            f"- Historical reuse allowed: {audit['reuse_allowed']}.",
            f"- Decision: `{audit['decision']}`.",
            "",
            "## Field Comparison",
            "",
        ]
    for name, group in audit["groups"].items():
        lines.append(f"### {name}")
        lines.append("")
        for row in group["fields"]:
            if not row["equal"]:
                lines.append(
                    f"- `{row['field']}`: historical="
                    f"`{row['historical']}`, lean_our="
                    f"`{row['lean_our']}`."
                )
        if group["equal"]:
            lines.append("- All audited fields match.")
        lines.append("")
    return lines


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args.historical_root.resolve())
    (output_dir / "E0_COMPATIBILITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    (output_dir / "E0_COMPATIBILITY_AUDIT.md").write_text(
        "\n".join(markdown_lines(audit)) + "\n",
        encoding="utf-8",
    )
    (output_dir / "E0_COMPATIBILITY_AUDIT_ZH.md").write_text(
        "\n".join(markdown_lines(audit, chinese=True)) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
