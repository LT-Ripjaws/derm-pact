"""G2 -- is there enough signal in the 7-point criteria to build on?

The go/no-go for the whole v2 architecture, and it is free: no images are encoded, no model is
called, nothing is paid for.  It asks one question with expert-annotated inputs, which is the
most generous possible setting:

    If an oracle handed you perfect criterion judgements, how well could you predict the
    diagnosis and the management decision?

That number is the **ceiling**.  A real assessor reading criteria off an image can only approach
it.  If the ceiling is not clearly above what metadata alone buys, the architecture has no room
to win and should be abandoned before it is written.

Design
------
Fit on the official ``train`` split, evaluate on ``valid``.  **The official test split is never
read** -- it is the confirmatory partition.

Four predictor sets, so that "criteria carry signal" is separated from "anything carries signal":

* ``majority``   -- predict the most frequent training class
* ``metadata``   -- sex, body location, elevation only (the control)
* ``criteria``   -- the seven gold criterion values, one-hot
* ``combined``   -- criteria + metadata

Plus the published rule applied directly, with no learning at all: score >= 3 implies excision.
That is the bar the agent must approach, and it is the honest reference for the whole paper.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from medguard_v2.criteria.aggregate import MANAGEMENT_LADDER, score_criteria
from medguard_v2.criteria.schema import MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import SEVEN_CRITERIA, load_cases
from medguard_v2.eval.metrics import classification_summary, quadratic_weighted_kappa
from medguard_v2.eval.statistics import binary_auc
from medguard_v2.eval.taxonomy import assert_labels_match

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "g2_criteria_ceiling.json"

METADATA_FIELDS = ("sex", "location", "elevation")
SEED = 20260810


def _features(cases, fields: Sequence[str]) -> List[List[str]]:
    rows = []
    for case in cases:
        row = []
        for field in fields:
            if field in SEVEN_CRITERIA:
                row.append(str(case.criteria[field]))
            else:
                row.append(str(getattr(case, field)))
        rows.append(row)
    return rows


def _fit_predict(
    train_cases, valid_cases, fields: Sequence[str], targets_train, targets_valid
) -> List[str]:
    """One-hot + multinomial logistic regression. Deliberately simple: this measures signal in
    the features, not the cleverness of the model."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train = encoder.fit_transform(_features(train_cases, fields))
    x_valid = encoder.transform(_features(valid_cases, fields))
    model = LogisticRegression(max_iter=5000, random_state=SEED)
    model.fit(x_train, targets_train)
    return list(model.predict(x_valid))


def _summarise(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict:
    summary = classification_summary(list(y_true), list(y_pred), list(labels))
    return {
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "macro_f1": round(summary["macro_f1"], 4),
    }


def _management_extras(y_true: Sequence[str], y_pred: Sequence[str], groups: Sequence[str]) -> Dict:
    """Ordinal quality plus the number that matters clinically."""
    ladder = {name: i for i, name in enumerate(MANAGEMENT_LADDER)}
    qwk = quadratic_weighted_kappa(
        [ladder[v] for v in y_true], [ladder[v] for v in y_pred], n_classes=len(MANAGEMENT_LADDER)
    )
    mel_total = sum(1 for g in groups if g == "MEL")
    mel_under = sum(
        1
        for true_m, pred_m, group in zip(y_true, y_pred, groups)
        if group == "MEL" and ladder[pred_m] < ladder["excision"]
    )
    return {
        "quadratic_weighted_kappa": round(qwk, 4),
        "melanoma_cases": mel_total,
        "melanoma_under_managed": mel_under,
        "melanoma_under_management_rate": round(mel_under / mel_total, 4) if mel_total else None,
    }


def _published_rule(valid_cases) -> Dict:
    """The 7-point rule with no learning at all: the reference bar."""
    scores = [score_criteria(c.criteria).score for c in valid_cases]
    truth_mel = [c.diagnosis_group == "MEL" for c in valid_cases]
    pred_excision = [s >= MELANOMA_SUSPICION_THRESHOLD for s in scores]

    tp = sum(1 for p, t in zip(pred_excision, truth_mel) if p and t)
    fn = sum(1 for p, t in zip(pred_excision, truth_mel) if not p and t)
    fp = sum(1 for p, t in zip(pred_excision, truth_mel) if p and not t)
    tn = sum(1 for p, t in zip(pred_excision, truth_mel) if not p and not t)

    y_true_mgmt = [c.management for c in valid_cases]
    y_pred_mgmt = ["excision" if p else "clinical follow up" for p in pred_excision]

    mel_prevalence = sum(truth_mel) / len(truth_mel)
    return {
        "melanoma_detection": {
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "ppv": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "npv": round(tn / (tn + fn), 4) if (tn + fn) else None,
            "auroc_raw_score": round(binary_auc(truth_mel, [float(s) for s in scores]), 4),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "melanoma_prevalence_in_this_split": round(mel_prevalence, 4),
            "prevalence_caveat": (
                "The official valid split is MEL-enriched (30.0% vs 21.8% train, 25.6% test). "
                "Sensitivity, specificity and AUROC are prevalence-independent and transfer; "
                "PPV and NPV are NOT and must not be quoted against another split."
            ),
        },
        "management_from_rule": {
            **_summarise(y_true_mgmt, y_pred_mgmt, MANAGEMENT_LADDER),
            **_management_extras(
                y_true_mgmt, y_pred_mgmt, [c.diagnosis_group for c in valid_cases]
            ),
        },
    }


def run() -> Dict:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    if any(c.split == "test" for c in (train + valid)):
        raise AssertionError("test split leaked into G2")

    # Pin the label set to the canonical Derm7pt order rather than deriving it alphabetically
    # from the cohort.  Deriving it is fragile twice over: the confusion-matrix rows come out in
    # a different order from every other artifact, and a cohort missing a class would silently
    # shrink the label set and rescale every macro average.  The guard raises instead.
    diag_labels = list(assert_labels_match({c.diagnosis_group for c in cases}, "derm7pt"))
    predictor_sets = {
        "metadata": METADATA_FIELDS,
        "criteria": SEVEN_CRITERIA,
        "combined": tuple(SEVEN_CRITERIA) + METADATA_FIELDS,
    }

    report: Dict = {
        "gate_id": "G2_criteria_ceiling",
        "question": "With oracle criteria, how well can diagnosis and management be predicted?",
        "boundaries": {
            "images_encoded": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "test_split_read": False,
            "fit_on": "train",
            "evaluated_on": "valid",
        },
        "cohort": {
            "train": len(train),
            "valid": len(valid),
            "diagnosis_groups": dict(Counter(c.diagnosis_group for c in cases)),
            "management": dict(Counter(c.management for c in cases)),
        },
        "diagnosis": {},
        "management": {},
        "published_rule_no_learning": _published_rule(valid),
    }

    # --- diagnosis -------------------------------------------------------------
    y_train_d = [c.diagnosis_group for c in train]
    y_valid_d = [c.diagnosis_group for c in valid]
    majority_d = Counter(y_train_d).most_common(1)[0][0]
    report["diagnosis"]["majority"] = _summarise(
        y_valid_d, [majority_d] * len(y_valid_d), diag_labels
    )
    for name, fields in predictor_sets.items():
        pred = _fit_predict(train, valid, fields, y_train_d, y_valid_d)
        report["diagnosis"][name] = _summarise(y_valid_d, pred, diag_labels)

    # --- management ------------------------------------------------------------
    y_train_m = [c.management for c in train]
    y_valid_m = [c.management for c in valid]
    groups_valid = [c.diagnosis_group for c in valid]
    majority_m = Counter(y_train_m).most_common(1)[0][0]
    report["management"]["majority"] = {
        **_summarise(y_valid_m, [majority_m] * len(y_valid_m), MANAGEMENT_LADDER),
        **_management_extras(y_valid_m, [majority_m] * len(y_valid_m), groups_valid),
    }
    for name, fields in predictor_sets.items():
        pred = _fit_predict(train, valid, fields, y_train_m, y_valid_m)
        report["management"][name] = {
            **_summarise(y_valid_m, pred, MANAGEMENT_LADDER),
            **_management_extras(y_valid_m, pred, groups_valid),
        }

    # --- verdict ---------------------------------------------------------------
    d_gain = report["diagnosis"]["criteria"]["balanced_accuracy"] - max(
        report["diagnosis"]["majority"]["balanced_accuracy"],
        report["diagnosis"]["metadata"]["balanced_accuracy"],
    )
    m_gain = report["management"]["criteria"]["quadratic_weighted_kappa"] - max(
        report["management"]["majority"]["quadratic_weighted_kappa"],
        report["management"]["metadata"]["quadratic_weighted_kappa"],
    )
    report["verdict"] = {
        "diagnosis_balanced_accuracy_gain_over_best_control": round(d_gain, 4),
        "management_qwk_gain_over_best_control": round(m_gain, 4),
        "go": bool(d_gain > 0.10 and m_gain > 0.10),
        "criterion": "criteria must beat the better of majority/metadata by >0.10 on both",
        "note": (
            "This is an oracle-criteria ceiling, not a system result. A real assessor reads "
            "criteria off an image and can only approach it."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    report = run()
    text = json.dumps(report, indent=2)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(text)


if __name__ == "__main__":
    main()
