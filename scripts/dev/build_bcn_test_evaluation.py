"""BCN20000 test evaluation -- the confirmatory read of the 712-lesion held-out cohort.

This is a one-shot evaluation on a guarded test set: `medguard_v2.data.bcn20000.load_records`
refuses to hand out the test split unless called with `allow_test=True`, and this is the first
script in the project registered to pass it. Everything upstream of this run was decided on BCN
train/valid only (G7/U1, G8, G9/U2 in `medguard_v2/eval/gates/`; see `notes/V2_TRANSFER_BOUNDARY_
2026-08-11.md`), so nothing here is selected against this cohort.

Three arms, all scored at **lesion level** (BCN carries >=2 images per lesion; image-level scoring
would count a lesion repeatedly and over-weight multi-image lesions):

    Arm 1  in-domain classifier      BCN train (2,143 lesions)     -> BCN test (712 lesions)
    Arm 2  cross-centre classifier   Derm7pt train (413 cases)     -> BCN test (712 lesions)
    Arm 3  cross-centre classifier + Derm7pt-trained criterion heads (both out-of-domain)

Arm 2 and Arm 3 answer a harder question than G8 did. G8 kept the diagnosis classifier in-domain
(trained on BCN) and only pushed the criteria path out of domain; the complementarity it found on
valid (0.149 -> 0.019 catches-per-melanoma) collapsed almost entirely under that partial transfer.
Arm 2/3 here push the CLASSIFIER out of domain as well, so this is the honest worst case: nothing
in the v2 pipeline has ever seen a BCN training image.

Why the two Derm7pt-side seeds differ from Arm 1's BCN seed
-------------------------------------------------------------
Arm 1 (BCN classifier) uses seed 20260811, as directed for this script. Arm 2/3's Derm7pt
classifier and criterion heads use C=10.0 / seed=20260810 -- the values REGISTERED for the v2
pipeline (`medguard_v2/pipeline/run_cohort.py`'s `BASE_C`/`SEED`, the same constants
`medguard_v2/eval/confirmatory.py` fit the Derm7pt confirmatory test result with). G8's own
criterion heads used a different, unregistered seed (20260811) as an exploratory convenience on
valid; this script uses the actually-registered configuration instead, since this run is
confirmatory.

Derm7pt test is never read here. `load_cases()` returns all three splits because that is how the
loader works (there is no per-split guard on Derm7pt, unlike BCN), but this script only ever
fits or scores against `split == "train"`; Derm7pt test was already spent by the registered
confirmatory run on 2026-08-11 and reading it again here would not change that, but touching it
for a purpose it was never registered for is exactly the kind of thing this project's rules said
not to do, so it is filtered out and never referenced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.bcn20000 import BCN_TO_GROUP, BcnRecord, lesion_level, load_records
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.metrics import classification_summary, ece, aurc, sensitivity_at_fixed_specificity
from medguard_v2.eval.statistics import binary_auc, clustered_bootstrap_ci, mcnemar_exact
from medguard_v2.eval.taxonomy import assert_labels_match

REPO = Path(__file__).resolve().parents[2]
DERM7PT_FEATURES = REPO / "research/evidence/derm7pt_v1/features/dermlip/features.npz"
OUT = REPO / "research/evidence/bcn_v1/bcn_test_evaluation.json"

BCN_SEED = 20260811                 # directed for this script: BCN classifier CV + fit
BCN_C_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

DERM7PT_SEED = 20260810             # registered (medguard_v2/pipeline/run_cohort.py)
DERM7PT_C = 10.0                    # registered (medguard_v2/pipeline/run_cohort.py, g8)

N_BOOT = 2000
BOOTSTRAP_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Small, dependency-light helpers shared across arms
# ---------------------------------------------------------------------------
def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(sorted(float(v) for v in values))
    return {
        "min": round(float(arr[0]), 4),
        "p25": round(float(np.quantile(arr, 0.25)), 4),
        "median": round(float(np.quantile(arr, 0.5)), 4),
        "p75": round(float(np.quantile(arr, 0.75)), 4),
        "max": round(float(arr[-1]), 4),
        "mean": round(float(arr.mean()), 4),
    }


def _stats(flag: Sequence[bool], truth: Sequence[bool]) -> Dict:
    """Binary confusion + sensitivity/specificity/PPV/Youden's J. Matches g8's helper."""
    tp = sum(1 for f, t in zip(flag, truth) if f and t)
    fn = sum(1 for f, t in zip(flag, truth) if not f and t)
    fp = sum(1 for f, t in zip(flag, truth) if f and not t)
    tn = sum(1 for f, t in zip(flag, truth) if not f and not t)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "sensitivity": round(sens, 4) if sens is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "ppv": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "youden_j": round(sens + spec - 1, 4) if None not in (sens, spec) else None,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def _melanoma_auroc_ci(truth_mel: Sequence[bool], score: Sequence[float],
                        lesion_ids: Sequence[str]) -> Dict:
    """Melanoma AUROC with a 95% CI from a lesion-clustered bootstrap.

    Every row is already exactly one lesion (post `lesion_level` pooling), so clustering by
    `lesion_id` here resamples whole lesions rather than pretending finer-grained structure
    exists -- but it is the methodologically correct call to make explicit rather than to fall
    back to an unclustered percentile bootstrap by omission.
    """
    rows = list(zip([bool(t) for t in truth_mel], [float(s) for s in score]))
    interval = clustered_bootstrap_ci(
        rows,
        list(lesion_ids),
        lambda sample: binary_auc([r[0] for r in sample], [r[1] for r in sample]),
        n_boot=N_BOOT,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    return {
        "estimate": round(interval.estimate, 4),
        "ci95_low": round(interval.low, 4),
        "ci95_high": round(interval.high, 4),
        "cluster_unit": "lesion_id",
        "n_boot": N_BOOT,
    }


# ---------------------------------------------------------------------------
# Arm 1: in-domain BCN classifier
# ---------------------------------------------------------------------------
def _select_bcn_c(x: np.ndarray, y: Sequence[str], groups: Sequence[str]) -> Dict:
    """C by 5-fold grouped CV on BCN train, grouped by lesion_id.

    Post-pooling every row is already one lesion, so grouping by `lesion_id` cannot itself
    prevent leakage here (each group has exactly one member) -- but it is the same defensive
    call G7's `_select_c` makes on un-pooled data, applied consistently rather than dropped
    just because it degenerates to `StratifiedKFold` on this particular input.
    """
    y_arr = np.asarray(y)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=BCN_SEED)
    per_c: Dict[str, float] = {}
    best_c, best_score = BCN_C_GRID[0], -1.0
    for c in BCN_C_GRID:
        correct = 0
        for tr, va in folds.split(x, y_arr, groups=np.asarray(groups)):
            model = LogisticRegression(max_iter=5000, C=c, random_state=BCN_SEED)
            model.fit(x[tr], y_arr[tr])
            correct += int((model.predict(x[va]) == y_arr[va]).sum())
        score = correct / len(y_arr)
        per_c[repr(c)] = round(score, 4)
        if score > best_score:
            best_c, best_score = c, score
    return {
        "selected_C": best_c,
        "cv_accuracy": round(best_score, 4),
        "grid_cv_accuracy": per_c,
        "method": "5-fold StratifiedGroupKFold on BCN train, grouped by lesion_id",
        "seed": BCN_SEED,
    }


# ---------------------------------------------------------------------------
# Shared diagnosis-arm evaluator (used for Arm 1 and Arm 2 -- both are plain
# 5-class LogisticRegression classifiers scored on the same BCN test cohort)
# ---------------------------------------------------------------------------
def _evaluate_diagnosis_arm(model: LogisticRegression, x_test: np.ndarray,
                             test_lesions: Sequence[BcnRecord], labels: Sequence[str]) -> Dict:
    truth_group = [r.group for r in test_lesions]
    lesion_ids = [r.lesion_id for r in test_lesions]
    truth_mel = [g == "MEL" for g in truth_group]

    predicted = list(model.predict(x_test))
    posterior = model.predict_proba(x_test)
    classes = list(model.classes_)
    assert "MEL" in classes, "classifier was never trained on a MEL example"
    mel_col = classes.index("MEL")
    mel_score = posterior[:, mel_col]
    conf = posterior.max(axis=1)
    correct = [p == t for p, t in zip(predicted, truth_group)]
    classifier_flag = [p == "MEL" for p in predicted]

    summary = classification_summary(truth_group, predicted, list(labels))
    cm_total = sum(sum(row) for row in summary["confusion_matrix"])
    assert cm_total == len(test_lesions), (
        f"confusion matrix sums to {cm_total}, expected {len(test_lesions)}"
    )

    report = {
        "n": summary["n"],
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "macro_f1": round(summary["macro_f1"], 4),
        "weighted_f1": round(summary["weighted_f1"], 4),
        "mcc": round(summary["mcc"], 4),
        "cohen_kappa": round(summary["cohen_kappa"], 4),
        "per_class": {
            k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            for k, v in summary["per_class"].items()
        },
        "confusion_matrix": summary["confusion_matrix"],
        "confusion_matrix_labels": summary["labels"],
        "confusion_matrix_sum": cm_total,
        "melanoma": {
            "auroc": _melanoma_auroc_ci(truth_mel, mel_score, lesion_ids),
            "at_argmax": _stats(classifier_flag, truth_mel),
            "sensitivity_at_specificity": {
                "0.90": sensitivity_at_fixed_specificity(truth_mel, [float(s) for s in mel_score], 0.90),
                "0.95": sensitivity_at_fixed_specificity(truth_mel, [float(s) for s in mel_score], 0.95),
            },
        },
        "calibration": {
            "ece": ece(list(conf), correct, n_bins=10),
            "aurc": aurc(list(conf), correct),
        },
    }
    return report, predicted, classifier_flag, correct, mel_score


# ---------------------------------------------------------------------------
# Arm 2/3 shared fit: Derm7pt-trained classifier + criterion heads
# ---------------------------------------------------------------------------
def _fit_derm7pt_classifier_and_heads() -> Tuple[LogisticRegression, Dict[str, LogisticRegression], int]:
    cases = load_cases()
    train = [case for case in cases if case.split == "train"]

    data = np.load(DERM7PT_FEATURES)
    index = {str(case_num): i for i, case_num in enumerate(data["case_num"])}
    x = np.stack([data["features"][index[str(case.case_num)]] for case in train])

    classifier = LogisticRegression(max_iter=5000, C=DERM7PT_C, random_state=DERM7PT_SEED)
    classifier.fit(x, [case.diagnosis_group for case in train])

    heads: Dict[str, LogisticRegression] = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(max_iter=5000, C=DERM7PT_C, random_state=DERM7PT_SEED)
        head.fit(x, [str(case.criteria[name]) for case in train])
        heads[name] = head

    return classifier, heads, len(train)


def _arm3_criteria_analysis(classifier_flag: Sequence[bool], truth_mel: Sequence[bool],
                             x_test: np.ndarray, test_lesions: Sequence[BcnRecord],
                             heads: Dict[str, LogisticRegression]) -> Dict:
    n = len(test_lesions)
    lesion_ids = [r.lesion_id for r in test_lesions]

    head_predictions = {name: list(heads[name].predict(x_test)) for name in CRITERION_NAMES}
    criteria_dicts = [
        {name: head_predictions[name][i] for name in CRITERION_NAMES} for i in range(n)
    ]
    scored = [score_criteria(c) for c in criteria_dicts]
    # score_min rather than .score: both are identical here (see decidable_note below) but
    # score_min never raises, so this stays correct even if that ever changes.
    criteria_scores = [s.score_min for s in scored]
    decidable_flags = [s.decidable for s in scored]
    criteria_flag = [s >= MELANOMA_SUSPICION_THRESHOLD for s in criteria_scores]

    union_flag = [a or b for a, b in zip(classifier_flag, criteria_flag)]
    conflict_flag = [a != b for a, b in zip(classifier_flag, criteria_flag)]

    both = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and a and b)
    clf_only = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and a and not b)
    crit_only = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and b and not a)
    neither = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and not a and not b)

    return {
        "criteria_path_out_of_domain": {
            "trained_on": "Derm7pt train (413 cases) -- OUT OF DOMAIN, not retrained for BCN",
            "melanoma_auroc_from_raw_score": _melanoma_auroc_ci(truth_mel, criteria_scores, lesion_ids),
            "at_threshold_3": _stats(criteria_flag, truth_mel),
            "sensitivity_at_specificity": {
                "0.90": sensitivity_at_fixed_specificity(
                    truth_mel, [float(s) for s in criteria_scores], 0.90),
                "0.95": sensitivity_at_fixed_specificity(
                    truth_mel, [float(s) for s in criteria_scores], 0.95),
            },
            "score_distribution": {
                "overall": _quantiles(criteria_scores),
                "melanoma": _quantiles([s for s, t in zip(criteria_scores, truth_mel) if t]),
                "non_melanoma": _quantiles([s for s, t in zip(criteria_scores, truth_mel) if not t]),
            },
        },
        "contract_level": {
            "decidable_rate": round(sum(decidable_flags) / n, 4),
            "decidable_note": (
                "Criterion heads are LogisticRegression.predict() over Derm7pt's own annotation "
                "vocabulary, which never contains INDETERMINATE as a training label -- a head "
                "can therefore never emit it, and every lesion is decidable by construction. "
                "Reported for contract-level completeness, not as a nontrivial finding."
            ),
            "score_range_observed": {"min": min(criteria_scores), "max": max(criteria_scores)},
            "conflict_rate": round(sum(conflict_flag) / n, 4),
            "conflict_definition": (
                "classifier commits MEL XOR criteria score >= threshold "
                f"({MELANOMA_SUSPICION_THRESHOLD}); matches "
                "medguard_v2.evidence.contract._build_conflict's definition, reproduced here "
                "directly so this script does not need the guideline-binding dependency chain."
            ),
        },
        "union_either_flags": _stats(union_flag, truth_mel),
        "complementarity": {
            "melanomas": sum(truth_mel),
            "both": both,
            "classifier_only": clf_only,
            "criteria_only": crit_only,
            "neither": neither,
        },
    }


def _mcnemar_block(a_correct: Sequence[bool], b_correct: Sequence[bool], label_a: str, label_b: str) -> Dict:
    only_a, only_b, p = mcnemar_exact(a_correct, b_correct)
    return {
        "test": "two-sided exact McNemar",
        f"{label_a}_only_correct": only_a,
        f"{label_b}_only_correct": only_b,
        "p_value": round(p, 6),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run() -> Dict:
    labels = list(assert_labels_match(set(BCN_TO_GROUP.values()), "derm7pt"))

    # --- cohorts -------------------------------------------------------------
    train_records, train_features = load_records(splits=("train",))
    train_lesions, x_train = lesion_level(train_records, train_features)

    test_records, test_features = load_records(splits=("test",), allow_test=True)
    test_lesions, x_test = lesion_level(test_records, test_features)

    assert len(train_lesions) == 2143, f"BCN train cohort is {len(train_lesions)}, expected 2143"
    assert len(test_lesions) == 712, f"BCN test cohort is {len(test_lesions)}, expected 712"

    y_train = [r.group for r in train_lesions]
    lesion_ids_train = [r.lesion_id for r in train_lesions]
    truth_group_test = [r.group for r in test_lesions]
    truth_mel_test = [g == "MEL" for g in truth_group_test]

    # --- Arm 1: in-domain BCN classifier --------------------------------------
    cv1 = _select_bcn_c(x_train, y_train, lesion_ids_train)
    arm1_model = LogisticRegression(max_iter=5000, C=cv1["selected_C"], random_state=BCN_SEED)
    arm1_model.fit(x_train, y_train)
    arm1_report, arm1_pred, arm1_flag, arm1_correct, arm1_mel_score = _evaluate_diagnosis_arm(
        arm1_model, x_test, test_lesions, labels
    )
    arm1_report = {"model_selection": cv1, **arm1_report}

    # --- Arm 2: cross-centre classifier (Derm7pt -> BCN test) -----------------
    arm2_model, heads, n_derm7pt_train = _fit_derm7pt_classifier_and_heads()
    arm2_report, arm2_pred, arm2_flag, arm2_correct, arm2_mel_score = _evaluate_diagnosis_arm(
        arm2_model, x_test, test_lesions, labels
    )
    arm2_report = {
        "trained_on": f"Derm7pt train ({n_derm7pt_train} cases) -- OUT OF DOMAIN for BCN",
        "hyperparameters": {"C": DERM7PT_C, "seed": DERM7PT_SEED, "registered": True},
        **arm2_report,
    }

    # --- Arm 3: Arm 2's classifier + Derm7pt-trained criterion heads ----------
    arm3_report = _arm3_criteria_analysis(arm2_flag, truth_mel_test, x_test, test_lesions, heads)
    arm3_report = {
        "classifier": "identical to arm2_cross_centre_classifier (same model, same predictions)",
        **arm3_report,
    }

    # --- cross-arm tests -------------------------------------------------------
    cross_arm = {
        "mcnemar_5class_accuracy_in_domain_vs_cross_centre": _mcnemar_block(
            arm1_correct, arm2_correct, "in_domain", "cross_centre"
        ),
        "mcnemar_melanoma_call_in_domain_vs_cross_centre": _mcnemar_block(
            [f == t for f, t in zip(arm1_flag, truth_mel_test)],
            [f == t for f, t in zip(arm2_flag, truth_mel_test)],
            "in_domain", "cross_centre",
        ),
        "complementarity_arm3": arm3_report["complementarity"],
    }

    report: Dict = {
        "protocol": "medguard-v2-bcn-test-evaluation-v1",
        "run": "BCN test -- guarded cohort, single read",
        "question": (
            "In-domain vs cross-centre BCN classification, and whether an out-of-domain "
            "criteria path still adds melanoma sensitivity, on the held-out 712-lesion test set."
        ),
        "boundaries": {
            "bcn_test_read": True,
            "bcn_test_guard": "allow_test=True, medguard_v2.data.bcn20000.load_records",
            "derm7pt_fit_split": "train",
            "derm7pt_test_read": False,
            "scoring_unit": "lesion (mean-pooled DermLIP features via lesion_level())",
            "paid_calls": 0,
            "generation_calls": 0,
        },
        "seeds": {
            "bcn_classifier_cv_and_fit": BCN_SEED,
            "derm7pt_classifier_and_criterion_heads": DERM7PT_SEED,
            "derm7pt_C_registered": DERM7PT_C,
        },
        "cohort": {
            "bcn_train_lesions": len(train_lesions),
            "bcn_train_melanomas": sum(1 for g in y_train if g == "MEL"),
            "bcn_train_class_counts": {g: y_train.count(g) for g in labels},
            "bcn_test_lesions": len(test_lesions),
            "bcn_test_melanomas": sum(truth_mel_test),
            "bcn_test_class_counts": {g: truth_group_test.count(g) for g in labels},
            "derm7pt_train_cases": n_derm7pt_train,
        },
        "arm1_in_domain_classifier": arm1_report,
        "arm2_cross_centre_classifier": arm2_report,
        "arm3_full_v2": arm3_report,
        "cross_arm_tests": cross_arm,
    }

    # --- assertions (fail loudly rather than write a silently-wrong result) ---
    assert report["arm1_in_domain_classifier"]["confusion_matrix_sum"] == 712, (
        "arm1 confusion matrix does not sum to 712"
    )
    assert report["arm2_cross_centre_classifier"]["confusion_matrix_sum"] == 712, (
        "arm2 confusion matrix does not sum to 712"
    )
    arm1_auroc = report["arm1_in_domain_classifier"]["melanoma"]["auroc"]["estimate"]
    assert 0.74 <= arm1_auroc <= 0.90, (
        f"BCN in-domain classifier melanoma AUROC {arm1_auroc} is far from the ~0.82 sanity "
        "band expected for this design; investigate before trusting this run"
    )

    # --- verdict: honest, descriptive, no editorializing beyond the numbers ---
    comp = arm3_report["complementarity"]
    report["verdict"] = {
        "in_domain_beats_cross_centre_5class_accuracy": (
            arm1_report["accuracy"] > arm2_report["accuracy"]
        ),
        "in_domain_beats_cross_centre_melanoma_auroc": (
            arm1_report["melanoma"]["auroc"]["estimate"]
            > arm2_report["melanoma"]["auroc"]["estimate"]
        ),
        "criteria_only_melanoma_catches": comp["criteria_only"],
        "criteria_only_share_of_melanomas": (
            round(comp["criteria_only"] / comp["melanomas"], 4) if comp["melanomas"] else None
        ),
        "union_sensitivity_gain_over_cross_centre_classifier_alone": (
            round(
                arm3_report["union_either_flags"]["sensitivity"]
                - arm2_report["melanoma"]["at_argmax"]["sensitivity"], 4
            )
            if None not in (
                arm3_report["union_either_flags"]["sensitivity"],
                arm2_report["melanoma"]["at_argmax"]["sensitivity"],
            ) else None
        ),
        "note": (
            "This is the fully out-of-domain arm: unlike G8 (in-domain BCN classifier, "
            "out-of-domain criteria only), here neither reader has ever seen a BCN training "
            "image. Compare criteria_only_share_of_melanomas against the G8 valid-set trail "
            "(0.148 Derm7pt valid -> 0.149 Derm7pt test -> 0.019 BCN valid, partial transfer) "
            "to see whether full transfer collapses further, holds, or (if the in-domain "
            "classifier's BCN-specific edge was doing a lot of G8's work) recovers."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = run()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
