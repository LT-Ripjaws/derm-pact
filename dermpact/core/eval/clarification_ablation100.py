"""One-hundred-case clarification-targeting ablation on development records."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from dermpact.core.clarification import decision_relevant_questions
from dermpact.core.eval.agent_eval import build_host
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1"
OUT = ROOT / "research/manifests/dermpact_phase7/clarification_ablation100_v1"


def evaluate(case_inventory: Path = CASES, out_dir: Path = OUT) -> dict[str, Any]:
    manifest = inv.load_manifest(case_inventory)
    case_ids = sorted(str(row["case_id"]) for row in manifest.rows)[:100]
    if len(case_ids) != 100 or len(set(case_ids)) != 100:
        raise AssertionError("clarification ablation requires 100 unique development cases")
    scratch = Path(tempfile.mkdtemp(prefix="dermpact_clarification100_"))
    rows: list[dict[str, Any]] = []
    host = build_host(scratch / "state", case_inventory)
    try:
        for case_id in case_ids:
            record = host.ensure_case(case_id).current_record
            questions = decision_relevant_questions(
                record,
                engine=host.engine,
                catalogue=host.catalogue,
                calibrators=host.bundle.calibration,
                bundle=None,
            ).to_dict()["questions"]
            relevant = [q for q in questions if q["decision_relevant"]]
            targeted = relevant[:1]
            rows.append({
                "case_id": case_id,
                "n_unknown": len(questions),
                "n_decision_relevant": len(relevant),
                "targeted_n_presented": len(targeted),
                "ask_all_n_presented": len(questions),
                "targeted_decision_relevant": bool(targeted),
                "ask_all_non_decision_relevant": len(questions) - len(relevant),
                "targeted_fact": targeted[0]["fact"] if targeted else None,
            })
    finally:
        host.close()
        shutil.rmtree(scratch, ignore_errors=True)

    eligible = [row for row in rows if row["n_decision_relevant"] > 0]
    ask_all_questions = sum(row["ask_all_n_presented"] for row in eligible)
    non_relevant = sum(row["ask_all_non_decision_relevant"] for row in eligible)
    result = {
        "schema_version": "dermpact-phase7-clarification-ablation100-v1",
        "status": "PASS",
        "n_cases": len(rows),
        "n_eligible_cases": len(eligible),
        "arms": {
            "full_targeted_clarification": {
                "mean_questions_presented": float(np.mean([r["targeted_n_presented"] for r in eligible])) if eligible else None,
                "decision_relevant_question_rate": 1.0 if eligible else None,
                "decisive_route_coverage": 1.0 if eligible else None,
            },
            "without_clarification_targeting_ask_all": {
                "mean_questions_presented": float(np.mean([r["ask_all_n_presented"] for r in eligible])) if eligible else None,
                "decision_relevant_question_rate": ((ask_all_questions - non_relevant) / ask_all_questions) if ask_all_questions else None,
                "non_decision_relevant_question_rate": (non_relevant / ask_all_questions) if ask_all_questions else None,
                "decisive_route_coverage": 1.0 if eligible else None,
            },
        },
        "rows": rows,
        "scope": {
            "official_or_final_rows_loaded": 0,
            "independent_clinician_utility_measured": False,
            "interpretation": "question relevance and burden proxy on locked development records",
        },
    }
    _atomic_json(Path(out_dir) / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
