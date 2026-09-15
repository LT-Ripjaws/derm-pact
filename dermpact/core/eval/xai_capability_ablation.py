"""Assemble measured-XAI capability evidence without overstating clinician benefit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json

ROOT = Path(__file__).resolve().parents[3]
COMPONENT = ROOT / "research/manifests/dermpact_phase7/agent_xai_ablation10_v1/results.json"
FEASIBILITY = ROOT / "research/experiments/dermpact_phase6/xai_feasibility_v1/feasibility.json"
OUT = ROOT / "research/manifests/dermpact_phase7/xai_capability_ablation_v1"


def build(out_dir: Path = OUT) -> dict[str, Any]:
    interaction = json.loads(COMPONENT.read_text(encoding="utf-8"))["summaries"]
    full = interaction["full_bounded_agent"]
    absent = interaction["without_measured_xai"]
    feasibility = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
    result = {
        "schema_version": "dermpact-phase7-xai-capability-ablation-v1",
        "status": "PASS",
        "interaction_subset": {
            "n_tasks_per_arm": full["n_tasks"],
            "n_unique_cases_per_arm": full["n_unique_cases"],
            "full_protocol_completion_rate": full["frozen_protocol_success_rate"],
            "without_xai_protocol_completion_rate": absent["frozen_protocol_success_rate"],
            "full_measured_receipt_available_rate": full["measured_receipt_available_rate"],
            "without_xai_measured_receipt_available_rate": absent["measured_receipt_available_rate"],
            "without_xai_safe_unavailability_resolution_rate": absent["safe_unavailability_resolution_rate"],
            "full_final_hazard_rate": full["final_hazard_rate"],
            "without_xai_final_hazard_rate": absent["final_hazard_rate"],
        },
        "measured_computation_feasibility": {
            "n_real_development_cases": 1,
            "ensemble_members_measured": feasibility["members_measured"],
            "no_op_max_abs_probability_difference": feasibility["no_op_parity"]["max_abs_diff"],
            "prediction_seconds": feasibility["latencies"]["predict_seconds"],
            "region_sweep_seconds": feasibility["latencies"]["region_sweep_seconds"],
            "clinical_contribution_seconds": feasibility["latencies"]["clinical_contribution_seconds"],
        },
        "clinician_facing_function": (
            "Shows a host-verified intervention/sensitivity receipt for image regions or the clinical-photo contribution; "
            "when unavailable, the system must say that no measured attribution is available."
        ),
        "scope": {
            "official_or_final_rows_loaded": 0,
            "independent_clinician_benefit_measured": False,
            "causal_attribution_claimed": False,
            "interpretation": "engineering capability and fail-closed interaction evidence only",
        },
    }
    _atomic_json(Path(out_dir) / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
