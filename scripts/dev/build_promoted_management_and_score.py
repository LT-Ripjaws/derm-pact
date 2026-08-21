"""Management + 7-point-score + per-criterion metrics on the PROMOTED perception (Derm7pt test).

The frozen numbers in ``test_metrics_complete.json`` / ``paper_metrics_summary.json`` describe the
PRE-promotion perception:

* 7-point score agreement (MAE 1.4152 / exact 0.357 / within-one 0.638) comes from the
  pre-promotion criterion heads (dermlip 512-d, unbalanced C=10, argmax);
* the management head is the registered P2 arm -- LogisticRegression C=100 on the raw 512-d
  dermlip features (QWK 0.5735);
* the per-criterion summaries and criteria_aggregate (binary concept F1 0.6623) come from the
  pre-promotion heads.

The paper now describes the promoted perception (``run_cohort(promoted=True)``), so this script
recomputes all of them on the promoted base and dual-reports the pre-promotion values as
explicitly labeled ``base_arm_reference`` blocks:

* criteria heads: PROMOTED (dermlip_tta+metadata14, lr_balanced, per-head C), read by ARGMAX,
  exactly as the promoted cohort path reads them;
* management head: the registered C=100 LogisticRegression refit on the PROMOTED diagnosis view
  (dermlip+metadata14, unbalanced) -- the only change vs the registered P2 arm is the feature
  view, which is fixed by the promotion; C, seed and class_weight are unchanged, so nothing is
  selected here;
* additionally, the CHECKLIST-IMPLIED management (``recommend_management`` over the promoted
  heads' 7-point score) is reported as a labeled secondary block: it is the checklist's own
  implication and can only ever emit 'clinical follow up' or 'excision' (never 'no further
  examination'), which structurally caps its QWK -- reported for transparency, not as the
  primary management arm.

NOT a new confirmatory look: the promoted configuration was already evaluated once on TEST in
``perception_phase2_safety_preserving.json``; everything here is a deterministic re-derivation of
that same frozen configuration. Wiring is proven by reproducing the frozen pre-promotion numbers
exactly before any promoted number is emitted.

Output: research/evidence/derm7pt_v1/promoted_management_and_score.json

Run::
    venv_win/Scripts/python.exe scripts/dev/build_promoted_management_and_score.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import (  # noqa: E402
    MANAGEMENT_LADDER,
    recommend_management,
    score_criteria,
)
from medguard_v2.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.eval.metrics import classification_summary, quadratic_weighted_kappa  # noqa: E402
from medguard_v2.eval.statistics import clustered_bootstrap_ci  # noqa: E402
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    PROMOTED_CRITERIA_C,
    SEED,
    _load_features,
    _promoted_feature_matrices,
    fit_readers,
    fit_readers_promoted,
)

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
OUT = EVIDENCE / "promoted_management_and_score.json"

MANAGEMENT_C = 100.0            # registered P2 arm (confirmatory.py / build_test_metrics_complete.py)
N_BOOT = 2000
BOOT_SEED = 20260712            # same convention as test_metrics_complete.json

# Frozen pre-promotion references, reproduced below as wiring checks before anything is emitted.
REF_BASE = {
    "seven_point": {"mae": 1.4152, "exact_match_rate": 0.357, "within_one_rate": 0.638},
    "management": {"accuracy": 0.7367, "balanced_accuracy": 0.6818, "macro_f1": 0.6868,
                   "qwk": 0.5735},
    "criteria_binary_f1": 0.6623,
    "promoted_binary_f1": 0.7168,   # perception_phase2_safety_preserving.json (same read)
}


def _round4(x):
    return round(float(x), 4)


def _summary_block(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, object]:
    s = classification_summary(y_true, y_pred, labels)
    return {
        "n": s["n"],
        "labels": s["labels"],
        "accuracy": _round4(s["accuracy"]),
        "balanced_accuracy": _round4(s["balanced_accuracy"]),
        "macro_f1": _round4(s["macro_f1"]),
        "weighted_f1": _round4(s["weighted_f1"]),
        "mcc": _round4(s["mcc"]),
        "cohen_kappa": _round4(s["cohen_kappa"]),
        "confusion_matrix": s["confusion_matrix"],
        "row_normalized_confusion_matrix": [
            [_round4(v) for v in row] for row in s["row_normalized_confusion_matrix"]
        ],
        "per_class": {
            label: {k: (_round4(v) if isinstance(v, float) else v) for k, v in stats.items()}
            for label, stats in s["per_class"].items()
        },
    }


def _qwk_with_ci(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    ladder = {name: i for i, name in enumerate(MANAGEMENT_LADDER)}
    ti = [ladder[v] for v in y_true]
    pi = [ladder[v] for v in y_pred]
    qwk = quadratic_weighted_kappa(ti, pi, n_classes=len(MANAGEMENT_LADDER))
    rows = list(zip(ti, pi))
    interval = clustered_bootstrap_ci(
        rows, [str(i) for i in range(len(rows))],
        lambda s: quadratic_weighted_kappa([r[0] for r in s], [r[1] for r in s],
                                           n_classes=len(MANAGEMENT_LADDER)),
        n_boot=N_BOOT, seed=BOOT_SEED,
    )
    return {
        "quadratic_weighted_kappa": _round4(qwk),
        "quadratic_weighted_kappa_ci95": {
            "estimate": _round4(interval.estimate),
            "ci95_low": _round4(interval.low),
            "ci95_high": _round4(interval.high),
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
        },
    }


def _seven_point_agreement(head_predictions: Dict[str, List[str]], test) -> Dict[str, object]:
    predicted_scores, gold_scores = [], []
    for i, case in enumerate(test):
        values = {name: head_predictions[name][i] for name in CRITERION_NAMES}
        scored = score_criteria(values)
        if not scored.determinate:
            raise AssertionError(f"case {case.case_num}: argmax readings must be determinate")
        predicted_scores.append(scored.score)
        gold_scores.append(case.seven_point_score)
    diffs = [abs(p - g) for p, g in zip(predicted_scores, gold_scores)]
    signed = [p - g for p, g in zip(predicted_scores, gold_scores)]
    return {
        "n": len(diffs),
        "mae": _round4(np.mean(diffs)),
        "exact_match_rate": _round4(sum(1 for d in diffs if d == 0) / len(diffs)),
        "within_one_rate": _round4(sum(1 for d in diffs if d <= 1) / len(diffs)),
        "mean_signed_error": _round4(np.mean(signed)),
        "signed_error_note": (
            "mean(predicted - gold): positive means the heads over-score (call more criteria "
            "present) relative to the annotated 7-point score."
        ),
        "_predicted_scores": predicted_scores,
    }


def _per_criterion_and_aggregate(head_predictions: Dict[str, List[str]], test) -> Dict[str, object]:
    per_criterion: Dict[str, object] = {}
    macro_f1s, accs, kappas, bin_f1 = [], [], [], {}
    for name in CRITERION_NAMES:
        gold = [str(c.criteria[name]) for c in test]
        pred = head_predictions[name]
        vocab = sorted(set(gold) | set(pred))
        s = classification_summary(gold, pred, vocab)
        macro_f1s.append(s["macro_f1"])
        accs.append(s["accuracy"])
        kappas.append(s["cohen_kappa"])
        per_criterion[name] = {
            "accuracy": _round4(s["accuracy"]),
            "balanced_accuracy": _round4(s["balanced_accuracy"]),
            "macro_f1": _round4(s["macro_f1"]),
            "cohen_kappa": _round4(s["cohen_kappa"]),
            "n_classes": len(vocab),
        }
        spec = CRITERIA_BY_NAME[name]
        yt = [1 if v in spec.scoring_values else 0 for v in gold]
        yp = [1 if v in spec.scoring_values else 0 for v in pred]
        bin_f1[name] = _round4(f1_score(yt, yp, average="macro", zero_division=0))
    aggregate = {
        "f1_macro_across_criteria": _round4(np.mean(macro_f1s)),
        "accuracy_macro_across_criteria": _round4(np.mean(accs)),
        "cohen_kappa_macro_across_criteria": _round4(np.mean(kappas)),
        "binary_concept_f1_macro": _round4(np.mean(list(bin_f1.values()))),
        "binary_concept_f1_per_criterion": bin_f1,
        "binary_concept_note": (
            "Binary F1-macro (scoring vs non-scoring values, macro-averaged over both classes). "
            "Matches DermAgent's evaluation protocol (binary concept detection F1-Macro)."
        ),
    }
    return {"per_criterion": per_criterion, "criteria_aggregate": aggregate}


def build() -> Dict[str, object]:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    truth_mgmt = [c.management for c in test]

    # ---- promoted readers -------------------------------------------------------------------
    diagnosis_features, criteria_features = _promoted_feature_matrices(cases)
    _, heads_promoted = fit_readers_promoted(train, diagnosis_features, criteria_features)
    xc_test = np.stack([criteria_features[str(c.case_num)] for c in test])
    pred_promoted = {n: [str(p) for p in h.predict(xc_test)] for n, h in heads_promoted.items()}

    # ---- base (pre-promotion) readers, for wiring checks only -------------------------------
    features_base = _load_features()
    _, heads_base = fit_readers(train, features_base)
    xb_test = np.stack([features_base[str(c.case_num)] for c in test])
    pred_base = {n: [str(p) for p in h.predict(xb_test)] for n, h in heads_base.items()}

    # ---- 1. seven-point score agreement -----------------------------------------------------
    seven_promoted = _seven_point_agreement(pred_promoted, test)
    seven_base = _seven_point_agreement(pred_base, test)
    promoted_scores = seven_promoted.pop("_predicted_scores")
    seven_base.pop("_predicted_scores")

    # ---- 2. per-criterion + aggregate -------------------------------------------------------
    crit_promoted = _per_criterion_and_aggregate(pred_promoted, test)
    crit_base = _per_criterion_and_aggregate(pred_base, test)

    # ---- 3. management: registered head on the promoted diagnosis view ----------------------
    xd_train = np.stack([diagnosis_features[str(c.case_num)] for c in train])
    xd_test = np.stack([diagnosis_features[str(c.case_num)] for c in test])
    mgmt_promoted_model = LogisticRegression(max_iter=5000, C=MANAGEMENT_C, random_state=SEED)
    mgmt_promoted_model.fit(xd_train, [c.management for c in train])
    mgmt_promoted_pred = [str(p) for p in mgmt_promoted_model.predict(xd_test)]
    mgmt_promoted = _summary_block(truth_mgmt, mgmt_promoted_pred, list(MANAGEMENT_LADDER))
    mgmt_promoted.update(_qwk_with_ci(truth_mgmt, mgmt_promoted_pred))

    # base management head (registered P2 arm: raw dermlip, C=100) -- wiring check
    x_train_base = np.stack([features_base[str(c.case_num)] for c in train])
    mgmt_base_model = LogisticRegression(max_iter=5000, C=MANAGEMENT_C, random_state=SEED)
    mgmt_base_model.fit(x_train_base, [c.management for c in train])
    mgmt_base_pred = [str(p) for p in mgmt_base_model.predict(xb_test)]
    mgmt_base = _summary_block(truth_mgmt, mgmt_base_pred, list(MANAGEMENT_LADDER))
    mgmt_base.update(_qwk_with_ci(truth_mgmt, mgmt_base_pred))

    # ---- 4. checklist-implied management (secondary, from promoted heads) -------------------
    implied_pred = []
    for i in range(len(test)):
        values = {name: pred_promoted[name][i] for name in CRITERION_NAMES}
        action, _reason = recommend_management(score_criteria(values))
        implied_pred.append(action)
    mgmt_implied = _summary_block(truth_mgmt, implied_pred, list(MANAGEMENT_LADDER))
    mgmt_implied.update(_qwk_with_ci(truth_mgmt, implied_pred))
    mgmt_implied["structural_note"] = (
        "recommend_management emits only 'clinical follow up' or 'excision' (an argmax reading is "
        "always decidable, and 'no further examination' is never implied by the checklist), so "
        "the 63 'no further examination' gold cases can never be matched. This caps accuracy and "
        "QWK structurally; the block is reported for transparency about what the checklist alone "
        "implies, not as the primary management arm."
    )

    # ---- 5. wiring self-checks: reproduce the frozen pre-promotion numbers ------------------
    checks = {
        "base_seven_point": {
            "got": {k: seven_base[k] for k in ("mae", "exact_match_rate", "within_one_rate")},
            "expected": REF_BASE["seven_point"],
            "source": "test_metrics_complete.json seven_point_score_agreement",
        },
        "base_management": {
            "got": {"accuracy": mgmt_base["accuracy"],
                    "balanced_accuracy": mgmt_base["balanced_accuracy"],
                    "macro_f1": mgmt_base["macro_f1"],
                    "qwk": mgmt_base["quadratic_weighted_kappa"]},
            "expected": REF_BASE["management"],
            "source": "test_metrics_complete.json management (registered P2 arm)",
        },
        "base_criteria_binary_f1": {
            "got": crit_base["criteria_aggregate"]["binary_concept_f1_macro"],
            "expected": REF_BASE["criteria_binary_f1"],
            "source": "test_metrics_complete.json criteria_aggregate",
        },
        "promoted_criteria_binary_f1": {
            "got": crit_promoted["criteria_aggregate"]["binary_concept_f1_macro"],
            "expected": REF_BASE["promoted_binary_f1"],
            "source": "perception_phase2_safety_preserving.json (same frozen read)",
        },
    }
    failures = []

    def _close(g, e, tol=0.0006):
        if isinstance(e, dict):
            return all(_close(g[k], e[k], tol) for k in e)
        return abs(float(g) - float(e)) <= tol

    for name, ch in checks.items():
        ch["reproduced"] = bool(_close(ch["got"], ch["expected"]))
        if not ch["reproduced"]:
            failures.append(name)
    if failures:
        raise SystemExit(
            f"WIRING DRIFT: failed to reproduce frozen references: {failures}\n"
            + json.dumps({k: checks[k] for k in failures}, indent=2)
        )

    return {
        "protocol": "derm7pt-promoted-management-and-score-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": (
            "Management, 7-point-score-agreement, and per-criterion metrics for the PROMOTED "
            "perception on Derm7pt test (n=395). Refreshes the pre-promotion numbers in "
            "test_metrics_complete.json, which are kept and dual-reported here as "
            "base_arm_reference blocks."
        ),
        "boundaries": {
            "fit_split": "train",
            "n_fit_cases": 413,
            "hyperparameters": {
                "seed": SEED,
                "management_C": MANAGEMENT_C,
                "management_class_weight": None,
                "criteria": "promoted per-head C (run_cohort.PROMOTED_CRITERIA_C), lr_balanced",
            },
            "note": (
                "Deterministic re-derivation of the frozen promoted configuration already "
                "evaluated on TEST in perception_phase2_safety_preserving.json; the management "
                "head keeps the registered C=100/seed/class_weight and only swaps in the promoted "
                "diagnosis view (dermlip+metadata14), which is fixed by the promotion. Nothing is "
                "selected on test."
            ),
        },
        "promoted_config": {
            "criteria_heads": "dermlip_tta+metadata14 (526-d), lr_balanced, per-head C, ARGMAX",
            "criteria_C_per_head": dict(PROMOTED_CRITERIA_C),
            "management_head": "dermlip+metadata14 (526-d), unbalanced LR C=100 (registered P2 C)",
        },
        "seven_point_score_agreement": {
            "promoted": seven_promoted,
            "base_arm_reference": {
                "label": "PRE-PROMOTION heads (dermlip 512-d, unbalanced C=10, argmax)",
                **seven_base,
            },
            "honesty_note": (
                "The promoted balanced heads are tuned for melanoma sensitivity: they call "
                "scoring values more often, which raises the predicted 7-point score. Whether "
                "MAE/exact agreement improves or degrades vs base is reported exactly as "
                "measured above -- the union melanoma sensitivity gain (0.7426 -> 0.8614) is the "
                "endpoint the promotion was accepted on, not score agreement."
            ),
        },
        "per_criterion_summary": {
            "promoted": crit_promoted["per_criterion"],
            "base_arm_reference": crit_base["per_criterion"],
        },
        "criteria_aggregate": {
            "promoted": crit_promoted["criteria_aggregate"],
            "base_arm_reference": crit_base["criteria_aggregate"],
        },
        "management": {
            "promoted": {
                "registered_arm": (
                    f"promoted diagnosis view (dermlip+metadata14), C={MANAGEMENT_C}, "
                    "class_weight=None, seed=20260810"
                ),
                **mgmt_promoted,
            },
            "base_arm_reference": {
                "label": "PRE-PROMOTION registered P2 arm (raw dermlip 512-d, C=100)",
                **mgmt_base,
            },
            "checklist_implied_secondary": mgmt_implied,
        },
        "wiring_self_checks": checks,
        "_promoted_seven_point_scores_by_case": {
            str(c.case_num): s for c, s in zip(test, promoted_scores)
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    report = build()
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")

    sp = report["seven_point_score_agreement"]
    mg = report["management"]
    print("\n=== 7-point score agreement (promoted vs base) ===")
    for k in ("mae", "exact_match_rate", "within_one_rate", "mean_signed_error"):
        base_v = sp["base_arm_reference"].get(k)
        print(f"  {k:18s}: {sp['promoted'][k]}   (base {base_v})")
    print("\n=== management (promoted view vs base P2 arm) ===")
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "quadratic_weighted_kappa"):
        print(f"  {k:18s}: {mg['promoted'][k]}   (base {mg['base_arm_reference'][k]})")
    ci = mg["promoted"]["quadratic_weighted_kappa_ci95"]
    print(f"  QWK CI95          : [{ci['ci95_low']}, {ci['ci95_high']}]")
    print(f"\n=== checklist-implied management (secondary) ===")
    print(f"  accuracy {mg['checklist_implied_secondary']['accuracy']}  "
          f"QWK {mg['checklist_implied_secondary']['quadratic_weighted_kappa']}")
    ca = report["criteria_aggregate"]
    print(f"\n=== criteria aggregate ===")
    print(f"  binary concept F1 : {ca['promoted']['binary_concept_f1_macro']} "
          f"(base {ca['base_arm_reference']['binary_concept_f1_macro']})")
    print(f"  multiclass macroF1: {ca['promoted']['f1_macro_across_criteria']} "
          f"(base {ca['base_arm_reference']['f1_macro_across_criteria']})")
    print(f"  all wiring checks reproduced: "
          f"{all(v['reproduced'] for v in report['wiring_self_checks'].values())}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
