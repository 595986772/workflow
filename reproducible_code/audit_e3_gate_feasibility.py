#!/usr/bin/env python3
"""Audit whether the strict E3 recovery-win gate is attainable."""

import argparse
import csv
import json
import math
from pathlib import Path


MINIMUM_RECOVERY_DELAY = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether any candidate can strictly beat DAOC on the "
            "required number of E3 recovery-delay seed pairs."
        )
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def build_feasibility(rows):
    if not rows:
        raise ValueError("E3 per-seed results are empty")
    required_wins = math.ceil(2 * len(rows) / 3)
    pairs = []
    for row in rows:
        daoc_delay = float(row["daoc_adaptation_delay"])
        our_delay = float(row["our_adaptation_delay"])
        if daoc_delay < MINIMUM_RECOVERY_DELAY:
            raise ValueError("DAOC recovery delay is below its lower bound")
        if our_delay < MINIMUM_RECOVERY_DELAY:
            raise ValueError("OUR recovery delay is below its lower bound")
        pairs.append(
            {
                "seed": int(row["seed"]),
                "daoc_recovery_delay": daoc_delay,
                "our_recovery_delay": our_delay,
                "current_strict_win": our_delay < daoc_delay,
                "candidate_can_strictly_win": (
                    daoc_delay > MINIMUM_RECOVERY_DELAY
                ),
            }
        )

    maximum_possible_wins = sum(
        pair["candidate_can_strictly_win"] for pair in pairs
    )
    current_wins = sum(pair["current_strict_win"] for pair in pairs)
    attainable = maximum_possible_wins >= required_wins
    return {
        "status": "complete",
        "gate_definition": (
            "candidate_recovery_delay_strictly_lower_than_daoc_on_at_"
            "least_two_thirds_of_seed_pairs"
        ),
        "recovery_delay_lower_bound": MINIMUM_RECOVERY_DELAY,
        "seed_pairs": len(pairs),
        "required_strict_wins": required_wins,
        "current_strict_wins": current_wins,
        "maximum_possible_strict_wins": maximum_possible_wins,
        "strict_recovery_gate_attainable": attainable,
        "protocol_decision": (
            "continue_single_module_revision"
            if attainable
            else "stop_without_algorithm_revision"
        ),
        "reason": (
            "At least two thirds of DAOC delays leave strict-win headroom."
            if attainable
            else (
                "Too many DAOC delays already equal the zero-window lower "
                "bound; no algorithm can achieve the required number of "
                "strictly lower paired delays."
            )
        ),
        "pairs": pairs,
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_reports(suite_dir, result):
    pairs = ", ".join(
        (
            f"seed {pair['seed']}: "
            f"{pair['daoc_recovery_delay']:g}/"
            f"{pair['our_recovery_delay']:g}"
        )
        for pair in result["pairs"]
    )
    english = [
        "# E3 Strict Recovery-Gate Feasibility Audit",
        "",
        f"- DAOC/OUR delays: {pairs}.",
        (
            "- Required strict wins: "
            f"{result['required_strict_wins']}/"
            f"{result['seed_pairs']}."
        ),
        (
            "- Maximum mathematically possible strict wins: "
            f"{result['maximum_possible_strict_wins']}."
        ),
        (
            "- Gate attainable: "
            f"{result['strict_recovery_gate_attainable']}."
        ),
        f"- Protocol decision: `{result['protocol_decision']}`.",
        "",
        result["reason"],
        "",
        (
            "This audit does not relax the registered gate or reinterpret "
            "ties as wins."
        ),
    ]
    chinese = [
        "# E3严格恢复门槛可达性审计",
        "",
        f"- DAOC/OUR恢复延迟：{pairs}。",
        (
            "- 要求严格获胜："
            f"{result['required_strict_wins']}/"
            f"{result['seed_pairs']}。"
        ),
        (
            "- 数学上最多可能严格获胜："
            f"{result['maximum_possible_strict_wins']}。"
        ),
        (
            "- 门槛是否可达："
            f"{result['strict_recovery_gate_attainable']}。"
        ),
        f"- 协议决定：`{result['protocol_decision']}`。",
        "",
        (
            "DAOC有过多seed已经达到0窗口下限，任何算法都不可能"
            "获得协议要求数量的严格更低恢复延迟。"
        ),
        "",
        "本审计不放宽既定门槛，也不把平局改算为获胜。",
    ]
    (suite_dir / "E3_GATE_FEASIBILITY.md").write_text(
        "\n".join(english) + "\n",
        encoding="utf-8",
    )
    (suite_dir / "E3_GATE_FEASIBILITY_ZH.md").write_text(
        "\n".join(chinese) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    rows = read_rows(
        suite_dir / "online_stream_e3_load_shift_per_seed.csv"
    )
    result = build_feasibility(rows)
    write_json(suite_dir / "E3_GATE_FEASIBILITY.json", result)
    write_reports(suite_dir, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
