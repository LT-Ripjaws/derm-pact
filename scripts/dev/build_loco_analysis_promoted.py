"""Leave-One-Criterion-Out (LOCO) causal influence analysis on the PROMOTED criteria heads.

The frozen ``loco_influence_analysis.json`` was computed on the pre-promotion criterion heads
(512-d dermlip, unbalanced LR C=10 -- ``fit_readers``). The paper now describes the promoted
perception (``run_cohort(promoted=True)``: criteria = dermlip_tta+metadata14, lr_balanced,
per-head C), so the influence analysis must be re-run on the heads the promoted path actually
reads. Criteria are read by ARGMAX, exactly how the promoted cohort path reads them
(``_head_values_and_margins`` argmax; no abstention gating -- the criterion_abstention_v1 cutoffs
were fit on the pre-promotion head and are not valid here).

This is NOT a new confirmatory look at test: the promoted configuration was already evaluated on
the sealed TEST split once, in ``perception_phase2_safety_preserving.json``. This script
re-derives downstream quantities of that same frozen configuration (deterministic refit on TRAIN,
argmax predictions, deterministic Argenziano aggregation) -- no hyperparameter is selected here.

Output: research/evidence/derm7pt_v1/loco_influence_analysis_promoted.json
(the pre-promotion ``loco_influence_analysis.json`` is kept untouched; its summary is embedded
below as ``base_arm_reference`` so the two readings can be compared side by side).

Run::
    venv_win/Scripts/python.exe scripts/dev/build_loco_analysis_promoted.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import leave_one_criterion_out, score_criteria  # noqa: E402
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    PROMOTED_CRITERIA_C,
    _promoted_feature_matrices,
    fit_readers_promoted,
)

OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "loco_influence_analysis_promoted.json"
BASE_LOCO = REPO / "research" / "evidence" / "derm7pt_v1" / "loco_influence_analysis.json"


def main() -> None:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]

    diagnosis_features, criteria_features = _promoted_feature_matrices(cases)
    _, heads = fit_readers_promoted(train, diagnosis_features, criteria_features)

    x_test = np.stack([criteria_features[str(c.case_num)] for c in test])

    # ARGMAX readings -- identical to how run_cohort(promoted=True) reads the heads.
    head_predictions = {name: [str(p) for p in head.predict(x_test)] for name, head in heads.items()}

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

    # Pre-promotion summary, embedded for side-by-side comparison. The full artifact stays on
    # disk untouched; only its summary-level numbers are copied here, explicitly labeled.
    base = json.loads(BASE_LOCO.read_text(encoding="utf-8"))
    base_arm_reference = {
        "label": "PRE-PROMOTION (base) criterion heads -- dermlip 512-d, unbalanced LR C=10",
        "source": "research/evidence/derm7pt_v1/loco_influence_analysis.json (kept, unmodified)",
        "cases_with_any_verdict_flip": base["cases_with_any_verdict_flip"],
        "cases_with_any_verdict_flip_rate": base["cases_with_any_verdict_flip_rate"],
        "ranking_by_verdict_flip_rate": base["ranking_by_verdict_flip_rate"],
        "ranking_by_melanoma_flip_rate": base["ranking_by_melanoma_flip_rate"],
    }

    result = {
        "protocol": "loco_causal_influence_v1_promoted",
        "perception": {
            "criteria_heads": "PROMOTED: dermlip_tta+metadata14 (526-d), lr_balanced, per-head C",
            "criteria_C_per_head": dict(PROMOTED_CRITERIA_C),
            "read_rule": "argmax (no abstention gating) -- matching run_cohort(promoted=True)",
            "source_config": "research/evidence/derm7pt_v1/perception_phase2_safety_preserving.json",
        },
        "boundaries": {
            "fit_split": "train",
            "note": (
                "Re-derivation of the frozen promoted configuration already evaluated on TEST in "
                "perception_phase2_safety_preserving.json; deterministic refit on TRAIN, argmax "
                "predictions, deterministic aggregation. No hyperparameter is selected here and "
                "no new confirmatory look is taken."
            ),
        },
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
        "base_arm_reference": base_arm_reference,
        "per_case": per_case,
    }

    # Ranking-change note, computed rather than eyeballed.
    new_rank = [r["criterion"] for r in result["ranking_by_verdict_flip_rate"]]
    old_rank = [r["criterion"] for r in base["ranking_by_verdict_flip_rate"]]
    result["ranking_changed_vs_base"] = new_rank != old_rank
    result["ranking_promoted_vs_base"] = {"promoted": new_rank, "base": old_rank}

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  {n} cases, {n_mel} melanomas  [PROMOTED heads, argmax]")
    print(f"  Cases where any criterion removal flips verdict: {any_flip} ({any_flip/n:.1%}) "
          f"[base: {base['cases_with_any_verdict_flip']} ({base['cases_with_any_verdict_flip_rate']:.1%})]")
    print("\n  Verdict flip ranking (all cases), promoted vs base:")
    base_rates = {r["criterion"]: r["flip_rate"] for r in base["ranking_by_verdict_flip_rate"]}
    for r in result["ranking_by_verdict_flip_rate"]:
        print(f"    {r['criterion']:25s}  {r['flip_rate']:.4f}   (base {base_rates[r['criterion']]:.4f})")
    print("\n  Verdict flip ranking (melanoma only):")
    for r in result["ranking_by_melanoma_flip_rate"]:
        print(f"    {r['criterion']:25s}  {r['flip_rate']:.4f}")
    print(f"\n  ranking changed vs base: {result['ranking_changed_vs_base']}")


if __name__ == "__main__":
    main()
