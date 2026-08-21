"""G5 -- is classifier-vs-criteria disagreement a useful triage signal?

The image classifier and the criteria path are two different ways of looking at the same lesion.
G2 and G3 showed they have different strengths: the classifier discriminates all five classes,
the criteria are a melanoma instrument that beats it on melanoma while ignoring vessel morphology
the rule discards.  When two such views disagree, that disagreement is information -- and nobody
in the corpus has measured whether it is *useful* information.

Three questions, in ascending order of clinical value:

1. **Does disagreement mark error?**  Is the classifier wrong more often on cases where the
   criteria path contradicts it?
2. **Does it beat confidence?**  Confidence is the standard error predictor.  Disagreement is
   only worth surfacing if it adds something confidence does not.
3. **Does it catch the misses that matter?**  Of the melanomas the classifier gets wrong, how
   many does the criteria path flag?  This is the number a clinician cares about.

This gate uses GOLD criteria, so it measures the *ceiling* of the disagreement signal, exactly as
G2 measured the ceiling of the criteria path.  A real assessor can only approach it -- G1 decides
how closely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import SEVEN_CRITERIA, load_cases
from medguard_v2.eval.statistics import binary_auc
from medguard_v2.eval.taxonomy import assert_labels_match

REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "research" / "evidence" / "derm7pt_v1" / "perception_baseline.json"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "g5_disagreement_signal.json"

SEED = 20260810


def _load_classifier_predictions() -> Dict[str, Dict]:
    """Flatten the baseline artifact's nested per-target blocks into one row per case."""
    artifact = json.loads(BASELINE.read_text(encoding="utf-8"))
    flat: Dict[str, Dict] = {}
    for row in artifact["cases"]:
        flat[str(row["case_num"])] = {
            "diagnosis_prediction": row["diagnosis"]["prediction"],
            "diagnosis_confidence": row["diagnosis"]["confidence"],
            "diagnosis_posterior": row["diagnosis"]["posterior"],
            "management_prediction": row["management"]["prediction"],
        }
    return flat


def _criteria_model_predictions(train, evaluate, labels: Sequence[str]) -> List[str]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_tr = encoder.fit_transform(
        [[str(c.criteria[n]) for n in SEVEN_CRITERIA] for c in train]
    )
    x_ev = encoder.transform(
        [[str(c.criteria[n]) for n in SEVEN_CRITERIA] for c in evaluate]
    )
    model = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
    model.fit(x_tr, [c.diagnosis_group for c in train])
    return list(model.predict(x_ev))


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


def run(split: str = "valid") -> Dict:
    cases = load_cases()
    labels = assert_labels_match({c.diagnosis_group for c in cases}, "derm7pt")
    train = [c for c in cases if c.split == "train"]
    target = [c for c in cases if c.split == split]

    predictions = _load_classifier_predictions()
    missing = [c.case_num for c in target if str(c.case_num) not in predictions]
    if missing:
        raise RuntimeError(f"classifier baseline missing {len(missing)} cases, e.g. {missing[:3]}")

    criteria_pred = _criteria_model_predictions(train, target, labels)

    clf_pred: List[str] = []
    clf_conf: List[float] = []
    for case in target:
        row = predictions[str(case.case_num)]
        clf_pred.append(str(row["diagnosis_prediction"]))
        clf_conf.append(float(row["diagnosis_confidence"]))

    truth = [c.diagnosis_group for c in target]
    clf_correct = [p == t for p, t in zip(clf_pred, truth)]
    disagree = [a != b for a, b in zip(clf_pred, criteria_pred)]

    n_dis = sum(disagree)
    n_agree = len(disagree) - n_dis
    err_dis = sum(1 for d, ok in zip(disagree, clf_correct) if d and not ok)
    err_agree = sum(1 for d, ok in zip(disagree, clf_correct) if not d and not ok)

    # Does disagreement beat confidence at predicting a classifier error?
    is_error = [not ok for ok in clf_correct]
    auroc_conf = binary_auc(is_error, [-c for c in clf_conf])          # low conf -> error
    auroc_dis = binary_auc(is_error, [1.0 if d else 0.0 for d in disagree])
    auroc_both = binary_auc(
        is_error, [(-c) + (1.0 if d else 0.0) for c, d in zip(clf_conf, disagree)]
    )

    # --- the clinical question: melanomas the classifier misses -----------------
    mel_truth = [t == "MEL" for t in truth]
    clf_says_mel = [p == "MEL" for p in clf_pred]
    rule_says_mel = [
        score_criteria(c.criteria).score >= MELANOMA_SUSPICION_THRESHOLD for c in target
    ]

    missed = [t and not p for t, p in zip(mel_truth, clf_says_mel)]
    rescued = sum(1 for m, r in zip(missed, rule_says_mel) if m and r)
    n_missed = sum(missed)

    # Cost of the rescue: benign lesions the rule escalates that the classifier did not.
    benign_escalated = sum(
        1 for t, p, r in zip(mel_truth, clf_says_mel, rule_says_mel) if (not t) and (not p) and r
    )
    n_benign_clf_ok = sum(1 for t, p in zip(mel_truth, clf_says_mel) if (not t) and (not p))

    report = {
        "gate_id": "G5_disagreement_signal",
        "question": "Does classifier-vs-criteria disagreement predict error better than confidence?",
        "boundaries": {
            "criteria_source": "GOLD (oracle) -- this is a ceiling, not a system result",
            "classifier": "DermLIP linear probe, fit on train only",
            "evaluated_on": split,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "cohort": {"n": len(target), "melanoma": sum(mel_truth)},
        "agreement": {
            "n_agree": n_agree,
            "n_disagree": n_dis,
            "disagreement_rate": _rate(n_dis, len(target)),
            "classifier_error_rate_when_agreeing": _rate(err_agree, n_agree),
            "classifier_error_rate_when_disagreeing": _rate(err_dis, n_dis),
            "error_rate_ratio": (
                round((err_dis / n_dis) / (err_agree / n_agree), 2)
                if n_dis and n_agree and err_agree
                else None
            ),
        },
        "error_prediction_auroc": {
            "confidence_only": round(auroc_conf, 4),
            "disagreement_only": round(auroc_dis, 4),
            "confidence_plus_disagreement": round(auroc_both, 4),
            "note": (
                "Disagreement is binary, so its AUROC is bounded well below a continuous "
                "score's. The question is whether adding it to confidence helps, not whether "
                "it wins alone."
            ),
        },
        "melanoma_rescue": {
            "melanomas_in_split": sum(mel_truth),
            "missed_by_classifier": n_missed,
            "of_those_flagged_by_the_7point_rule": rescued,
            "rescue_rate": _rate(rescued, n_missed),
            "benign_cases_classifier_called_benign": n_benign_clf_ok,
            "of_those_escalated_by_the_rule": benign_escalated,
            "false_escalation_rate": _rate(benign_escalated, n_benign_clf_ok),
        },
    }

    rescue = report["melanoma_rescue"]["rescue_rate"]
    report["verdict"] = {
        "disagreement_marks_error": bool(
            report["agreement"]["classifier_error_rate_when_disagreeing"]
            and report["agreement"]["classifier_error_rate_when_agreeing"] is not None
            and report["agreement"]["classifier_error_rate_when_disagreeing"]
            > report["agreement"]["classifier_error_rate_when_agreeing"]
        ),
        "adds_to_confidence": bool(auroc_both > auroc_conf),
        "rescues_missed_melanomas": bool(rescue is not None and rescue > 0.5),
        "interpretation": (
            "A rescue rate materially above zero means the two views fail on different cases, "
            "which is the argument for surfacing the conflict rather than silently picking one. "
            "It must be read against the false-escalation rate: rescue is only free if the rule "
            "does not also escalate large numbers of benign lesions."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    report = run(args.split)
    text = json.dumps(report, indent=2)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(text)


if __name__ == "__main__":
    main()
