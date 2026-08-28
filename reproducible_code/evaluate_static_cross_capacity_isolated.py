#!/usr/bin/env python3
"""Evaluate each cross-capacity run in a fresh Python process."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from evaluate_static_cross_capacity import (
    DEFAULT_LABELS,
    aggregate_output,
    evaluate_one,
    expected_complete,
    parse_int_list,
    parse_list,
)
from static_heterogeneity_protocol import (
    CAPACITY_PROFILES,
    MAIN_PROFILE,
    STATIC_HETEROGENEITY_PROTOCOL_VERSION,
)


ISOLATION_PROTOCOL_VERSION = "cross_capacity_process_isolation_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot capacity evaluation with one fresh process "
            "per profile-method-seed."
        )
    )
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=parse_list,
        default=list(DEFAULT_LABELS),
    )
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument(
        "--profiles",
        type=parse_list,
        default=list(CAPACITY_PROFILES),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--revision-id", default="hsr1g1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile")
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        raise ValueError("episodes and workers must be positive")
    if args.worker:
        if (
            args.label is None
            or args.seed is None
            or args.profile is None
        ):
            raise ValueError(
                "worker mode requires label, seed, and profile"
            )
    elif not args.seeds:
        raise ValueError("parent mode requires at least one seed")
    unknown = set(args.profiles) - set(CAPACITY_PROFILES)
    if args.profile is not None:
        unknown |= {args.profile} - set(CAPACITY_PROFILES)
    if unknown:
        raise ValueError(f"Unknown capacity profiles: {sorted(unknown)}")
    return args


def canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def build_specs(labels, seeds, profiles):
    return [
        (profile, label, seed)
        for profile in profiles
        for label in labels
        for seed in seeds
    ]


def target_run(output_dir, profile, label, seed):
    return output_dir / profile / "runs" / label / f"seed_{seed}"


def worker_command(
    script,
    source_suite,
    output_dir,
    label,
    seed,
    profile,
    episodes,
    resume,
):
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--source-suite-dir",
        str(source_suite),
        "--output-dir",
        str(output_dir),
        "--label",
        label,
        "--seed",
        str(seed),
        "--profile",
        profile,
        "--episodes",
        str(episodes),
    ]
    if resume:
        command.append("--resume")
    return command


def run_worker(args):
    torch.set_num_threads(1)
    result = evaluate_one(
        (
            args.source_suite_dir.resolve(),
            args.output_dir.resolve(),
            args.label,
            args.seed,
            args.profile,
            args.episodes,
            args.resume,
        )
    )
    print(json.dumps(result))


def source_audit(source_suite):
    revision_dir = source_suite.parent
    final_lock_path = revision_dir / "FINAL_STATIC_LOCK.json"
    frozen_path = revision_dir / "FROZEN_STATIC_ALGORITHM.json"
    final_manifest_path = source_suite / "static_stage_manifest.json"
    for path in (
        final_lock_path,
        frozen_path,
        final_manifest_path,
    ):
        if not path.exists():
            raise RuntimeError(f"Missing frozen source artifact: {path}")
    final_lock = read_json(final_lock_path)
    frozen = read_json(frozen_path)
    final_manifest = read_json(final_manifest_path)
    if (
        final_lock.get("status") != "complete"
        or final_lock.get("gate_passed") is not True
        or frozen.get("status") != "frozen"
        or final_manifest.get("status") != "complete"
        or final_manifest.get("gate_passed") is not True
    ):
        raise RuntimeError("Source H3 final artifacts are not eligible")
    if (
        final_manifest.get("algorithm_source_sha256")
        != frozen.get("algorithm_source_sha256")
    ):
        raise RuntimeError("Frozen algorithm hash mismatch")
    return {
        "source_revision_id": frozen["revision_id"],
        "source_algorithm_sha256": (
            frozen["algorithm_source_sha256"]
        ),
        "source_protocol_sha256": (
            frozen["protocol_source_sha256"]
        ),
        "source_frozen_configuration_sha256": (
            frozen["frozen_configuration_sha256"]
        ),
        "source_final_lock_sha256": file_hash(final_lock_path),
        "source_final_manifest_sha256": file_hash(
            final_manifest_path
        ),
    }


def run_child(spec, args, script):
    profile, label, seed = spec
    run_dir = target_run(
        args.output_dir,
        profile,
        label,
        seed,
    )
    summary_path = run_dir / "summary.json"
    if args.resume and expected_complete(
        summary_path,
        profile,
        args.episodes,
    ):
        return {
            "status": "skipped",
            "profile": profile,
            "label": label,
            "seed": seed,
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = run_dir / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    command = worker_command(
        script=script,
        source_suite=args.source_suite_dir,
        output_dir=args.output_dir,
        label=label,
        seed=seed,
        profile=profile,
        episodes=args.episodes,
        resume=args.resume,
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(matplotlib_dir)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (run_dir / "isolated_process.log").write_text(
        result.stdout,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Isolated run failed: {profile} {label} seed={seed}"
        )
    return {
        "status": "complete",
        "profile": profile,
        "label": label,
        "seed": seed,
        "wall_time_sec": time.perf_counter() - started,
        "command": command,
        "python_hash_seed": 0,
    }


def validate_parent_args(args):
    if len(args.labels) != 2 and not args.skip_aggregate:
        raise ValueError(
            "Formal aggregation requires exactly two method labels"
        )
    if (
        MAIN_PROFILE not in args.profiles
        and not args.skip_aggregate
    ):
        raise ValueError(
            f"Formal aggregation requires source profile {MAIN_PROFILE}"
        )


def run_parent(args):
    validate_parent_args(args)
    source_suite = args.source_suite_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.source_suite_dir = source_suite
    args.output_dir = output_dir
    script = Path(__file__).resolve()
    audit = source_audit(source_suite)
    specification = {
        "isolation_protocol_version": ISOLATION_PROTOCOL_VERSION,
        "static_protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "revision_id": args.revision_id,
        "source_suite_dir": str(source_suite),
        "labels": args.labels,
        "seeds": args.seeds,
        "profiles": args.profiles,
        "episodes": args.episodes,
        "workers": args.workers,
        "one_fresh_process_per_run": True,
        "python_hash_seed": 0,
        "script_sha256": file_hash(script),
        **audit,
    }
    specification["configuration_sha256"] = canonical_hash(
        specification
    )
    manifest_path = output_dir / "process_isolation_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if (
            not args.resume
            or existing.get("configuration_sha256")
            != specification["configuration_sha256"]
        ):
            raise RuntimeError(
                "Output directory belongs to another isolation run"
            )
    manifest = {
        "status": "running",
        **specification,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_runs": 0,
        "failed_runs": [],
    }
    write_json(manifest_path, manifest)
    specs = build_specs(args.labels, args.seeds, args.profiles)
    failures = []
    records = []
    try:
        with ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            future_to_spec = {
                executor.submit(run_child, spec, args, script): spec
                for spec in specs
            }
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    record = future.result()
                except Exception as error:
                    failure = {
                        "profile": spec[0],
                        "label": spec[1],
                        "seed": spec[2],
                        "error": repr(error),
                    }
                    failures.append(failure)
                    print(f"[failed] {failure}", flush=True)
                else:
                    records.append(record)
                    print(
                        f"[{len(records)}/{len(specs)}] "
                        f"{record['status']} {spec[0]} {spec[1]} "
                        f"seed={spec[2]}",
                        flush=True,
                    )
                manifest["completed_runs"] = len(records)
                manifest["failed_runs"] = failures
                write_json(manifest_path, manifest)
        if failures:
            raise RuntimeError(
                f"{len(failures)} isolated evaluations failed"
            )
        if args.skip_aggregate:
            summary = {
                "status": "complete",
                "skip_aggregate": True,
                "runs": records,
            }
            write_json(
                output_dir / "isolated_worker_summary.json",
                summary,
            )
        else:
            summary = aggregate_output(
                output_dir,
                args.profiles,
                args.labels,
                args.seeds,
            )
            if (
                summary["all_methods_scenario_paired"] is not True
                or summary[
                    "all_profiles_base_scenario_paired"
                ]
                is not True
            ):
                raise RuntimeError(
                    "Process-isolated pairing audit failed"
                )
        manifest["status"] = "complete"
        manifest["summary_gate_passed"] = (
            args.skip_aggregate
            or (
                summary["all_methods_scenario_paired"]
                and summary["all_profiles_base_scenario_paired"]
            )
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        write_json(manifest_path, manifest)


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
