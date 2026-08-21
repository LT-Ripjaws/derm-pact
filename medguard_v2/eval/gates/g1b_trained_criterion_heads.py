"""G1b -- the pre-registered mitigation: can a *trained* head read the criteria?

G1 asked whether a general vision-language model can judge the seven criteria zero-shot. It
cannot: macro accuracy 0.5672 against a train-majority chance of 0.6256, i.e. a **negative** lift,
with three of seven criteria substantially worse than a constant predictor. The v2 plan named the
mitigation in advance -- *"train per-criterion heads (Derm7pt has the labels) instead of a
zero-shot VLM"* -- so this gate runs it before any conclusion is drawn about the architecture.

It is free: the DermLIP features are already encoded and cached, and Derm7pt ships criterion labels
for all 1,011 cases. Seven small multinomial probes, fit on train only, evaluated on valid, scored
against exactly the same chance baseline G1 used so the two are directly comparable.

The question is narrow and decisive: **is the criteria layer unreadable, or was the reader wrong?**
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.metrics import classification_summary

REPO = Path(__file__).resolve().parents[3]
FEATURES = REPO / "research" / "evidence" / "derm7pt_v1" / "features" / "dermlip" / "features.npz"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "g1b_trained_criterion_heads.json"

SEED = 20260810
C_GRID = (0.1, 1.0, 10.0, 100.0)


def _load_features() -> Dict[str, np.ndarray]:
    data = np.load(FEATURES)
    return {str(c): v for c, v in zip(data["case_num"], data["features"])}


def _select_c(x: np.ndarray, y: List[str]) -> float:
    """Pick C by stratified out-of-fold accuracy on TRAIN only."""
    from sklearn.model_selection import StratifiedKFold

    counts = Counter(y)
    n_splits = min(5, min(counts.values()))
    if n_splits < 2:
        return 10.0
    best_c, best_score = 10.0, -1.0
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_arr = np.asarray(y)
    for c in C_GRID:
        correct = 0
        for tr, va in folds.split(x, y_arr):
            model = LogisticRegression(max_iter=5000, C=c, random_state=SEED)
            model.fit(x[tr], y_arr[tr])
            correct += int((model.predict(x[va]) == y_arr[va]).sum())
        score = correct / len(y_arr)
        if score > best_score:
            best_c, best_score = c, score
    return best_c


def run() -> Dict:
    features = _load_features()
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]

    x_train = np.stack([features[str(c.case_num)] for c in train])
    x_valid = np.stack([features[str(c.case_num)] for c in valid])

    per_criterion: Dict[str, Dict] = {}
    predicted_by_case: List[Dict[str, str]] = [{} for _ in valid]

    for name in CRITERION_NAMES:
        y_train = [str(c.criteria[name]) for c in train]
        y_valid = [str(c.criteria[name]) for c in valid]
        labels = sorted(set(y_train) | set(y_valid))

        selected_c = _select_c(x_train, y_train)
        model = LogisticRegression(max_iter=5000, C=selected_c, random_state=SEED)
        model.fit(x_train, y_train)
        predicted = list(model.predict(x_valid))
        for row, value in zip(predicted_by_case, predicted):
            row[name] = value

        summary = classification_summary(y_valid, predicted, labels)

        majority = Counter(y_train).most_common(1)[0][0]
        chance = classification_summary(y_valid, [majority] * len(y_valid), labels)

        per_criterion[name] = {
            "selected_C": selected_c,
            "accuracy": round(summary["accuracy"], 4),
            "balanced_accuracy": round(summary["balanced_accuracy"], 4),
            "macro_f1": round(summary["macro_f1"], 4),
            "cohen_kappa": round(summary["cohen_kappa"], 4),
            "chance_accuracy_train_majority": round(chance["accuracy"], 4),
            "lift_over_chance": round(summary["accuracy"] - chance["accuracy"], 4),
            "train_majority_value": majority,
        }

    macro = {
        "accuracy": round(
            float(np.mean([v["accuracy"] for v in per_criterion.values()])), 4
        ),
        "cohen_kappa": round(
            float(np.mean([v["cohen_kappa"] for v in per_criterion.values()])), 4
        ),
        "chance_accuracy_train_majority": round(
            float(np.mean([v["chance_accuracy_train_majority"] for v in per_criterion.values()])), 4
        ),
    }
    macro["lift_over_chance"] = round(
        macro["accuracy"] - macro["chance_accuracy_train_majority"], 4
    )

    # --- derived 7-point score, exactly as the architecture would compute it ----
    gold_scores = [score_criteria(c.criteria).score for c in valid]
    pred_scores = [score_criteria(row).score for row in predicted_by_case]
    errors = [abs(p - g) for p, g in zip(pred_scores, gold_scores)]

    gold_flag = [s >= MELANOMA_SUSPICION_THRESHOLD for s in gold_scores]
    pred_flag = [s >= MELANOMA_SUSPICION_THRESHOLD for s in pred_scores]
    tp = sum(1 for p, g in zip(pred_flag, gold_flag) if p and g)
    fn = sum(1 for p, g in zip(pred_flag, gold_flag) if not p and g)
    fp = sum(1 for p, g in zip(pred_flag, gold_flag) if p and not g)
    tn = sum(1 for p, g in zip(pred_flag, gold_flag) if not p and not g)

    # Against the actual diagnosis, which is what a clinician cares about.
    mel_truth = [c.diagnosis_group == "MEL" for c in valid]
    mtp = sum(1 for p, t in zip(pred_flag, mel_truth) if p and t)
    mfn = sum(1 for p, t in zip(pred_flag, mel_truth) if not p and t)
    mfp = sum(1 for p, t in zip(pred_flag, mel_truth) if p and not t)
    mtn = sum(1 for p, t in zip(pred_flag, mel_truth) if not p and not t)

    report = {
        "gate_id": "G1b_trained_criterion_heads",
        "question": "Is the criteria layer unreadable, or was the zero-shot VLM the wrong reader?",
        "boundaries": {
            "encoder": "DermLIP ViT-B-16 frozen features (cached)",
            "fit_on": "train",
            "evaluated_on": "valid",
            "paid_calls": 0,
            "test_split_read": False,
        },
        "cohort": {"train": len(train), "valid": len(valid)},
        "per_criterion": per_criterion,
        "macro_across_seven": macro,
        "derived_seven_point_score": {
            "mae": round(float(np.mean(errors)), 4),
            "exact_match_rate": round(float(np.mean([e == 0 for e in errors])), 4),
            "within_one_rate": round(float(np.mean([e <= 1 for e in errors])), 4),
            "mean_signed_error": round(
                float(np.mean([p - g for p, g in zip(pred_scores, gold_scores)])), 4
            ),
        },
        "threshold_vs_gold_score": {
            "accuracy": round((tp + tn) / len(valid), 4),
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        },
        "threshold_vs_melanoma_diagnosis": {
            "sensitivity": round(mtp / (mtp + mfn), 4) if (mtp + mfn) else None,
            "specificity": round(mtn / (mtn + mfp), 4) if (mtn + mfp) else None,
            "confusion": {"tp": mtp, "fn": mfn, "fp": mfp, "tn": mtn},
        },
        "comparison_to_g1_zero_shot_vlm": {
            "g1_macro_accuracy_strict": 0.5672,
            "g1_macro_lift_over_chance": -0.0584,
            "g1_macro_cohen_kappa": 0.2092,
            "g1_score_mae": 1.7143,
            "note": "Same chance baseline definition and the same valid split.",
        },
    }
    report["verdict"] = {
        "beats_chance": bool(macro["lift_over_chance"] > 0),
        "beats_zero_shot_vlm": bool(macro["accuracy"] > 0.5672),
        "interpretation": (
            "If a trained head clears chance while the zero-shot VLM does not, the criteria are "
            "readable and G1 measured the reader, not the layer. The architecture would then need "
            "a trained criterion model rather than a general VLM -- a different system, and a "
            "narrower agentic claim, but a live one."
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
