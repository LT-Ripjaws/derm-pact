"""U2 / G9 -- can the criteria path be adapted to a new centre without criterion labels?

G8 established the problem: criterion heads trained on Derm7pt's 413 cases collapse on BCN
(melanoma AUROC 0.929 -> 0.682, criteria-only catches 0.149 -> 0.019 of melanomas). There is no
second criterion-labelled dataset to train on, so the only route is to manufacture supervision.

Two variants, both fitted on **BCN train** and evaluated on **BCN valid**. BCN test is never read.

**A -- naive self-training.** The Derm7pt heads pseudo-label BCN train; high-confidence
predictions are added to the training set and the heads are refitted. The obvious risk is
confirmation bias: a weak teacher (AUROC 0.682) mostly teaches its own mistakes.

**B -- diagnosis-consistent filtering.** BCN carries something the teacher does not use: the
**diagnosis**. A pseudo-label set is kept only when the 7-point score it produces is consistent
with the known malignancy status of that lesion -- score >= 3 for a melanoma, < 3 otherwise. This
is side information the teacher never saw, so it can correct the teacher rather than echo it.

B is the interesting one. If neither works, the annotation bottleneck in
`V2_TRANSFER_BOUNDARY_2026-08-11.md` is real and not engineerable around, which is itself the
finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.bcn20000 import lesion_level, load_records
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.statistics import binary_auc

REPO = Path(__file__).resolve().parents[3]
DERM7PT_FEATURES = REPO / "research/evidence/derm7pt_v1/features/dermlip/features.npz"
OUT = REPO / "research/evidence/derm7pt_v1/g9_semisupervised_adaptation.json"

SEED = 20260811
BASE_C = 10.0
CONFIDENCE_GRID = (0.7, 0.8, 0.9)


def _derm7pt_training():
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    data = np.load(DERM7PT_FEATURES)
    index = {str(c): i for i, c in enumerate(data["case_num"])}
    x = np.stack([data["features"][index[str(c.case_num)]] for c in train])
    y = {name: [str(c.criteria[name]) for c in train] for name in CRITERION_NAMES}
    return x, y


def _fit_heads(x: np.ndarray, y: Dict[str, Sequence[str]]) -> Dict[str, LogisticRegression]:
    heads = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
        head.fit(x, list(y[name]))
        heads[name] = head
    return heads


def _predict_criteria(heads, x: np.ndarray) -> Tuple[List[Dict[str, str]], np.ndarray]:
    per_case: List[Dict[str, str]] = []
    confidences = np.zeros((x.shape[0], len(CRITERION_NAMES)))
    for j, name in enumerate(CRITERION_NAMES):
        proba = heads[name].predict_proba(x)
        classes = heads[name].classes_
        picks = proba.argmax(axis=1)
        confidences[:, j] = proba.max(axis=1)
        for i, p in enumerate(picks):
            if len(per_case) <= i:
                per_case.append({})
            per_case[i][name] = str(classes[p])
    return per_case, confidences


def _evaluate(heads, x_valid: np.ndarray, truth_mel: Sequence[bool],
              classifier_flag: Sequence[bool]) -> Dict:
    criteria, _ = _predict_criteria(heads, x_valid)
    scores = [score_criteria(c).score for c in criteria]
    flag = [s >= MELANOMA_SUSPICION_THRESHOLD for s in scores]

    tp = sum(1 for f, t in zip(flag, truth_mel) if f and t)
    fn = sum(1 for f, t in zip(flag, truth_mel) if not f and t)
    fp = sum(1 for f, t in zip(flag, truth_mel) if f and not t)
    tn = sum(1 for f, t in zip(flag, truth_mel) if not f and not t)
    crit_only = sum(
        1 for a, b, t in zip(classifier_flag, flag, truth_mel) if t and b and not a
    )
    return {
        "melanoma_auroc_from_score": round(binary_auc(truth_mel, [float(s) for s in scores]), 4),
        "sensitivity_at_threshold_3": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity_at_threshold_3": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "criteria_only_melanoma_catches": crit_only,
        "criteria_only_share_of_melanomas": round(crit_only / sum(truth_mel), 4),
        "mean_score_melanoma": round(
            float(np.mean([s for s, t in zip(scores, truth_mel) if t])), 3
        ),
        "mean_score_non_melanoma": round(
            float(np.mean([s for s, t in zip(scores, truth_mel) if not t])), 3
        ),
    }


def run() -> Dict:
    # --- cohorts ------------------------------------------------------------------
    train_records, train_features = load_records(splits=("train",))
    valid_records, valid_features = load_records(splits=("valid",))
    train_lesions, x_bcn_train = lesion_level(train_records, train_features)
    valid_lesions, x_bcn_valid = lesion_level(valid_records, valid_features)

    y_bcn_train = [r.group for r in train_lesions]
    truth_mel_valid = [r.group == "MEL" for r in valid_lesions]

    # In-domain classifier, identical to G8, for the complementarity comparison.
    classifier = LogisticRegression(max_iter=5000, C=100.0, random_state=SEED)
    classifier.fit(x_bcn_train, y_bcn_train)
    classifier_flag = [p == "MEL" for p in classifier.predict(x_bcn_valid)]

    x_d7, y_d7 = _derm7pt_training()
    baseline_heads = _fit_heads(x_d7, y_d7)

    report: Dict = {
        "gate_id": "G9_semisupervised_adaptation",
        "question": "Can the criteria path be adapted to a new centre without criterion labels?",
        "boundaries": {
            "teacher": "Derm7pt-trained heads (413 cases)",
            "adaptation_cohort": "BCN train (lesion level)",
            "evaluated_on": "BCN valid (lesion level)",
            "bcn_test_read": False,
            "paid_calls": 0,
        },
        "cohort": {
            "bcn_train_lesions": len(train_lesions),
            "bcn_valid_lesions": len(valid_lesions),
            "valid_melanomas": sum(truth_mel_valid),
        },
        "baseline_derm7pt_only": _evaluate(
            baseline_heads, x_bcn_valid, truth_mel_valid, classifier_flag
        ),
        "variant_A_naive_self_training": {},
        "variant_B_diagnosis_consistent": {},
    }

    # Teacher predictions on BCN train, once.
    pseudo, confidence = _predict_criteria(baseline_heads, x_bcn_train)
    pseudo_scores = [score_criteria(c).score for c in pseudo]
    known_mel = [g == "MEL" for g in y_bcn_train]

    # --- variant A: confidence-thresholded self-training --------------------------
    for threshold in CONFIDENCE_GRID:
        keep = [i for i in range(len(pseudo)) if confidence[i].min() >= threshold]
        if len(keep) < 50:
            report["variant_A_naive_self_training"][f"conf>={threshold}"] = {
                "kept": len(keep), "skipped": "too few pseudo-labels"
            }
            continue
        x_aug = np.vstack([x_d7, x_bcn_train[keep]])
        y_aug = {
            name: list(y_d7[name]) + [pseudo[i][name] for i in keep]
            for name in CRITERION_NAMES
        }
        heads = _fit_heads(x_aug, y_aug)
        result = _evaluate(heads, x_bcn_valid, truth_mel_valid, classifier_flag)
        result["pseudo_labels_kept"] = len(keep)
        report["variant_A_naive_self_training"][f"conf>={threshold}"] = result

    # --- variant B: keep only pseudo-labels consistent with the known diagnosis ----
    consistent = [
        i for i in range(len(pseudo))
        if (pseudo_scores[i] >= MELANOMA_SUSPICION_THRESHOLD) == known_mel[i]
    ]
    report["variant_B_diagnosis_consistent"]["selection"] = {
        "candidates": len(pseudo),
        "kept": len(consistent),
        "kept_fraction": round(len(consistent) / len(pseudo), 4),
        "kept_melanomas": sum(1 for i in consistent if known_mel[i]),
        "rationale": (
            "A pseudo-label set is retained only when the 7-point score it implies agrees with "
            "the lesion's known malignancy status. The diagnosis is side information the teacher "
            "never used, so this can correct the teacher rather than echo it."
        ),
    }
    if len(consistent) >= 50:
        x_aug = np.vstack([x_d7, x_bcn_train[consistent]])
        y_aug = {
            name: list(y_d7[name]) + [pseudo[i][name] for i in consistent]
            for name in CRITERION_NAMES
        }
        heads_b = _fit_heads(x_aug, y_aug)
        report["variant_B_diagnosis_consistent"]["result"] = _evaluate(
            heads_b, x_bcn_valid, truth_mel_valid, classifier_flag
        )

    # --- verdict ------------------------------------------------------------------
    base = report["baseline_derm7pt_only"]
    candidates = []
    for key, block in report["variant_A_naive_self_training"].items():
        if isinstance(block, dict) and "melanoma_auroc_from_score" in block:
            candidates.append((f"A::{key}", block))
    if "result" in report["variant_B_diagnosis_consistent"]:
        candidates.append(("B::diagnosis_consistent", report["variant_B_diagnosis_consistent"]["result"]))

    best = max(candidates, key=lambda kv: kv[1]["melanoma_auroc_from_score"], default=None)
    report["verdict"] = {
        "baseline_auroc": base["melanoma_auroc_from_score"],
        "baseline_criteria_only_catches": base["criteria_only_melanoma_catches"],
        "best_variant": best[0] if best else None,
        "best_auroc": best[1]["melanoma_auroc_from_score"] if best else None,
        "auroc_gain": round(best[1]["melanoma_auroc_from_score"] - base["melanoma_auroc_from_score"], 4) if best else None,
        "best_criteria_only_catches": best[1]["criteria_only_melanoma_catches"] if best else None,
        "derm7pt_in_domain_reference": {
            "criteria_only_share_of_melanomas": 0.149,
            "note": "Derm7pt test. Adaptation would have to approach this to matter.",
        },
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
