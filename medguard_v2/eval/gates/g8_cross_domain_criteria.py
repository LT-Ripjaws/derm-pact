"""G8 -- does an OUT-OF-DOMAIN criteria path still add melanoma sensitivity in-domain?

This is the question the whole v2 claim now rests on, and U1 is the reason it is sharp.

U1 showed a classifier trained on HAM loses 0.105 melanoma AUROC on Derm7pt: dermoscopy models do
not transfer across centres. The criterion heads can only ever be trained on Derm7pt, because it is
the only source of criterion labels in existence. So on any other cohort the criteria path is
permanently out-of-domain, while the classifier can be trained in-domain and will therefore be
strong.

That sets up the honest test:

    classifier   -> trained on BCN train (in-domain, 2,143 lesions, 314 melanomas)
    criteria path -> Derm7pt-trained heads applied to BCN (out-of-domain, no retraining)
    evaluated on -> BCN valid (715 lesions, 105 melanomas). Test stays sealed.

If the criteria path still catches melanomas the in-domain classifier misses, the complementarity
claim survives domain shift and is far stronger than the same-dataset version. If it does not,
that is a real boundary condition on criteria-based dermoscopy AI and worth reporting as one.

Scoring is at **lesion** level. BCN has a minimum of two images per lesion, so image-level scoring
would count the same lesion repeatedly and weight multi-image lesions more heavily.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.bcn20000 import lesion_level, load_records, load_split
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.statistics import binary_auc
from medguard_v2.eval.taxonomy import assert_labels_match

REPO = Path(__file__).resolve().parents[3]
DERM7PT_FEATURES = REPO / "research/evidence/derm7pt_v1/features/dermlip/features.npz"
OUT = REPO / "research/evidence/derm7pt_v1/g8_cross_domain_criteria.json"

SEED = 20260811
BASE_C = 10.0          # the registered v2 value for criterion heads
CLASSIFIER_C = 100.0   # in-domain classifier; BCN is large enough to want weak regularisation


def _fit_criterion_heads():
    """Derm7pt-trained heads. These are the out-of-domain component."""
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    data = np.load(DERM7PT_FEATURES)
    index = {str(c): i for i, c in enumerate(data["case_num"])}
    x = np.stack([data["features"][index[str(c.case_num)]] for c in train])
    heads = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
        head.fit(x, [str(c.criteria[name]) for c in train])
        heads[name] = head
    return heads, len(train)


def _stats(flag: Sequence[bool], truth: Sequence[bool]) -> Dict:
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


def _spec_at_sens(score: np.ndarray, truth: Sequence[bool], target: float) -> Dict:
    order = np.argsort(-score)
    n_pos = sum(truth)
    need = int(np.ceil(target * n_pos))
    hits, cut = 0, float(score[order[-1]])
    for i in order:
        if truth[i]:
            hits += 1
            if hits >= need:
                cut = float(score[i])
                break
    pred = [s >= cut for s in score]
    return {"threshold": round(cut, 4), **_stats(pred, truth)}


def run() -> Dict:
    split = load_split()
    labels = list(assert_labels_match(set(split["label_mapping"].values()), "derm7pt"))

    train_records, train_features = load_records(splits=("train",))
    valid_records, valid_features = load_records(splits=("valid",))
    train_lesions, x_train = lesion_level(train_records, train_features)
    valid_lesions, x_valid = lesion_level(valid_records, valid_features)

    y_train = [r.group for r in train_lesions]
    y_valid = [r.group for r in valid_lesions]
    truth_mel = [g == "MEL" for g in y_valid]

    # --- in-domain classifier ----------------------------------------------------
    classifier = LogisticRegression(max_iter=5000, C=CLASSIFIER_C, random_state=SEED)
    classifier.fit(x_train, y_train)
    mel_col = list(classifier.classes_).index("MEL")
    classifier_score = classifier.predict_proba(x_valid)[:, mel_col]
    classifier_pred = list(classifier.predict(x_valid))
    classifier_flag = [p == "MEL" for p in classifier_pred]

    # --- out-of-domain criteria path ---------------------------------------------
    heads, n_derm7pt_train = _fit_criterion_heads()
    criteria_by_case = [
        {name: heads[name].predict(x_valid[i : i + 1])[0] for name in CRITERION_NAMES}
        for i in range(x_valid.shape[0])
    ]
    criteria_scores = [score_criteria(c).score for c in criteria_by_case]
    criteria_flag = [s >= MELANOMA_SUSPICION_THRESHOLD for s in criteria_scores]

    union_flag = [a or b for a, b in zip(classifier_flag, criteria_flag)]

    both = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and a and b)
    clf_only = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and a and not b)
    crit_only = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and b and not a)
    neither = sum(1 for a, b, t in zip(classifier_flag, criteria_flag, truth_mel) if t and not a and not b)

    report = {
        "gate_id": "G8_cross_domain_criteria",
        "question": "Does a Derm7pt-trained criteria path add melanoma sensitivity on top of an in-domain BCN classifier?",
        "boundaries": {
            "classifier_trained_on": "BCN train, lesion level",
            "classifier_train_lesions": len(train_lesions),
            "criteria_heads_trained_on": f"Derm7pt train ({n_derm7pt_train} cases) -- OUT OF DOMAIN, not retrained",
            "evaluated_on": "BCN valid, lesion level",
            "bcn_test_read": False,
            "scoring_unit": "lesion",
            "paid_calls": 0,
        },
        "cohort": {
            "valid_lesions": len(valid_lesions),
            "melanomas": sum(truth_mel),
            "class_counts": {g: y_valid.count(g) for g in labels},
        },
        "classifier_in_domain": {
            "melanoma_auroc": round(binary_auc(truth_mel, [float(s) for s in classifier_score]), 4),
            "at_argmax": _stats(classifier_flag, truth_mel),
            "specificity_at_sensitivity": {
                "0.90": _spec_at_sens(classifier_score, truth_mel, 0.90),
                "0.95": _spec_at_sens(classifier_score, truth_mel, 0.95),
            },
        },
        "criteria_path_out_of_domain": {
            "melanoma_auroc_from_raw_score": round(
                binary_auc(truth_mel, [float(s) for s in criteria_scores]), 4
            ),
            "at_threshold_3": _stats(criteria_flag, truth_mel),
            "score_distribution": {
                "mean_in_melanoma": round(
                    float(np.mean([s for s, t in zip(criteria_scores, truth_mel) if t])), 3
                ),
                "mean_in_non_melanoma": round(
                    float(np.mean([s for s, t in zip(criteria_scores, truth_mel) if not t])), 3
                ),
            },
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

    clf = report["classifier_in_domain"]["at_argmax"]
    union = report["union_either_flags"]
    report["verdict"] = {
        "criteria_path_catches_melanomas_the_classifier_misses": crit_only,
        "sensitivity_gain_from_union": (
            round(union["sensitivity"] - clf["sensitivity"], 4)
            if None not in (union["sensitivity"], clf["sensitivity"]) else None
        ),
        "specificity_cost_from_union": (
            round(union["specificity"] - clf["specificity"], 4)
            if None not in (union["specificity"], clf["specificity"]) else None
        ),
        "survives_domain_shift": bool(crit_only > 0),
        "interpretation": (
            "The criteria path is out-of-domain by construction -- Derm7pt is the only source of "
            "criterion labels anywhere. If it still recovers melanomas an in-domain classifier "
            "misses, the complementarity claim is robust to domain shift. If it does not, that is "
            "a boundary condition on criteria-based dermoscopy AI, and reportable as one."
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
