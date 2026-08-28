#!/usr/bin/env python3
"""Run the locked P14 analyzer with paired Q=10 evaluation results."""

import json
from pathlib import Path

import analyze_pegasus_service_sensitivity as analysis
from pegasus_service_sensitivity_protocol import SERVICE_STATE_DIMENSION
from repair_pegasus_q10_pairing import paired_q10_run


locked_active_service_run = analysis.active_service_run


def paired_active_service_run(active_services, method, seed):
    if active_services == SERVICE_STATE_DIMENSION:
        return paired_q10_run(method, seed)
    return locked_active_service_run(active_services, method, seed)


analysis.active_service_run = paired_active_service_run


def annotate_pairfix_outputs():
    """Record the Q=10 evaluation-only repair in the human and machine reports."""
    analysis_dir = Path("results/pegasus_pscale/p14_service_sensitivity/analysis")
    report_path = analysis_dir / "SERVICE_SENSITIVITY_REPORT_CN.md"
    report = report_path.read_text(encoding="utf-8")
    old = "- 活跃服务数实验对 Q=4/6/8 从头训练，Q=10 复用已审计的收敛 checkpoint。"
    new = (
        "- 活跃服务数实验对 Q=4/6/8 从头训练，Q=10 复用已审计的收敛 checkpoint；"
        "Q=10 只在与其他设置统一的 100 个配对场景上冻结复评，没有重训。\n"
        "- 图中曲线为 3 个独立 seed 的均值，误差带为样本标准差（mean ± SD）。"
    )
    if old not in report:
        raise RuntimeError("Expected report sentence was not found; refusing silent rewrite")
    report_path.write_text(report.replace(old, new), encoding="utf-8")

    manifest_path = analysis_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["q10_pairing_repair"] = {
        "checkpoint_retrained": False,
        "evaluation_scenarios": 100,
        "networks_frozen": True,
        "audit": "results/pegasus_pscale/p14_service_sensitivity/ACTIVE_SERVICE_AUDIT_PAIRFIX.json",
        "diagnostic": "results/pegasus_pscale/p14_service_sensitivity/Q10_PAIRING_DIAGNOSTIC.json",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    analysis.main()
    annotate_pairfix_outputs()
