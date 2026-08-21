"""Leave-One-Criterion-Out (LOCO) causal influence analysis on Derm7pt test.

For every test case, marks each criterion INDETERMINATE in turn and checks
whether the melanoma-threshold verdict survives.  Because aggregation is
deterministic, this is a true causal influence measurement — not a post-hoc
approximation like SHAP or LIME.

Output: research/evidence/derm7pt_v1/loco_influence_analysis.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import leave_one_criterion_out, score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.pipeline.run_cohort import _load_features, fit_readers

SEED = 20260810
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "loco_influence_analysis.json"


def main() -> None:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    features = _load_features()
    _, heads = fit_readers(train, features)

    x_test = np.stack([features[str(c.case_num)] for c in test])

    head_predictions = {}
    for name, head in heads.items():
        head_predictions[name] = head.predict(x_test).tolist()

    per_case = []
    verdict_flip_counts = Counter()
    decidability_loss_counts = Counter()
    influence_matrix = {name: {"flips_verdict": 0, "loses_decidability": 0, "no_effect": 0}
                        for name in CRITERION_NAMES}
    melanoma_only_influence = {name: {"flips_verdict": 0, "loses_decidability": 0, "no_effect": 0}
                                for name in CRITERION_NAMES}

    for i, case in enumerate(test):
        criteria = {name: head_predictions[name][i] for name in CRITERION_NAMES}
        base_score = score_criteria(criteria)
        loco = leave_one_criterion_out(criteria)

        case_result = {
            "case_id": f"derm7pt-{case.case_num}",
            "diagnosis_group": case.diagnosis_group,
            "is_melanoma": case.diagnosis_group == "MEL",
            "base_score_min": base_score.score_min,
            "base_score_max": base_score.score_max,
            "base_decidable": base_score.decidable,
            "base_meets_threshold": base_score.meets_threshold if base_score.decidable else None,
            "criterion_influence": {},
        }

        for name in CRITERION_NAMES:
            info = loco[name]
            case_result["criterion_influence"][name] = {
                "points_contributed": info["points_contributed"],
                "verdict_changed": info["verdict_changed"],
                "still_decidable": info["still_decidable"],
                "score_range_without": list(info["score_range"]),
            }

            if info["verdict_changed"]:
                verdict_flip_counts[name] += 1
                influence_matrix[name]["flips_verdict"] += 1
                if case.diagnosis_group == "MEL":
                    melanoma_only_influence[name]["flips_verdict"] += 1
            elif not info["still_decidable"]:
                decidability_loss_counts[name] += 1
                influence_matrix[name]["loses_decidability"] += 1
                if case.diagnosis_group == "MEL":
                    melanoma_only_influence[name]["loses_decidability"] += 1
            else:
                influence_matrix[name]["no_effect"] += 1
                if case.diagnosis_group == "MEL":
                    melanoma_only_influence[name]["no_effect"] += 1

        per_case.append(case_result)

    n = len(test)
    n_mel = sum(1 for c in test if c.diagnosis_group == "MEL")

    any_flip = sum(1 for c in per_case if any(
        c["criterion_influence"][name]["verdict_changed"] for name in CRITERION_NAMES
    ))

    result = {
        "protocol": "loco_causal_influence_v1",
        "n_cases": n,
        "n_melanoma": n_mel,
        "cases_with_any_verdict_flip": any_flip,
        "cases_with_any_verdict_flip_rate": round(any_flip / n, 4),
        "summary": {
            "description": (
                "Per-criterion causal influence: removing each criterion and checking "
                "whether the melanoma-threshold verdict changes. Because aggregation is "
                "deterministic (Argenziano 7-point rule), this is an exact causal "
                "measurement, not an approximation."
            ),
            "threshold": MELANOMA_SUSPICION_THRESHOLD,
        },
        "influence_matrix_all_cases": {
            name: {
                "flips_verdict": influence_matrix[name]["flips_verdict"],
                "flips_verdict_rate": round(influence_matrix[name]["flips_verdict"] / n, 4),
                "loses_decidability": influence_matrix[name]["loses_decidability"],
                "loses_decidability_rate": round(influence_matrix[name]["loses_decidability"] / n, 4),
                "no_effect": influence_matrix[name]["no_effect"],
                "no_effect_rate": round(influence_matrix[name]["no_effect"] / n, 4),
                "category": "major" if name in ("pigment_network", "blue_whitish_veil", "vascular_structures") else "minor",
                "points": 2 if name in ("pigment_network", "blue_whitish_veil", "vascular_structures") else 1,
            }
            for name in CRITERION_NAMES
        },
        "influence_matrix_melanoma_only": {
            name: {
                "flips_verdict": melanoma_only_influence[name]["flips_verdict"],
                "flips_verdict_rate": round(melanoma_only_influence[name]["flips_verdict"] / n_mel, 4) if n_mel else 0,
                "loses_decidability": melanoma_only_influence[name]["loses_decidability"],
                "loses_decidability_rate": round(melanoma_only_influence[name]["loses_decidability"] / n_mel, 4) if n_mel else 0,
                "no_effect": melanoma_only_influence[name]["no_effect"],
                "no_effect_rate": round(melanoma_only_influence[name]["no_effect"] / n_mel, 4) if n_mel else 0,
            }
            for name in CRITERION_NAMES
        },
        "ranking_by_verdict_flip_rate": sorted(
            [{"criterion": name, "flip_rate": round(influence_matrix[name]["flips_verdict"] / n, 4)}
             for name in CRITERION_NAMES],
            key=lambda x: -x["flip_rate"],
        ),
        "ranking_by_melanoma_flip_rate": sorted(
            [{"criterion": name, "flip_rate": round(melanoma_only_influence[name]["flips_verdict"] / n_mel, 4) if n_mel else 0}
             for name in CRITERION_NAMES],
            key=lambda x: -x["flip_rate"],
        ),
        "per_case": per_case,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  {n} cases, {n_mel} melanomas")
    print(f"  Cases where any criterion removal flips verdict: {any_flip} ({any_flip/n:.1%})")
    print(f"\n  Verdict flip ranking (all cases):")
    for r in result["ranking_by_verdict_flip_rate"]:
        print(f"    {r['criterion']:25s}  {r['flip_rate']:.4f}")
    print(f"\n  Verdict flip ranking (melanoma only):")
    for r in result["ranking_by_melanoma_flip_rate"]:
        print(f"    {r['criterion']:25s}  {r['flip_rate']:.4f}")


if __name__ == "__main__":
    main()
