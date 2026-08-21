"""U1 -- stop training the diagnosis classifier on 413 cases.

The v2 classifier is fit on Derm7pt's 413 training cases because that is where the criterion
labels live. But the classifier does not need criterion labels -- it needs diagnoses, and
**HAM10000 has 8,024 dermoscopic images already encoded with the same frozen DermLIP checkpoint**
by v1 (`research/experiments/derm_perception/v2_tournament/features/dermlip`). Its sealed test
partition is excluded from that feature set (`test_records_loaded: 0`), so nothing sealed is
touched here.

That is roughly 19x more training data for the reader that carries the melanoma endpoint, at zero
cost. This gate measures whether it helps, evaluated on Derm7pt **valid** -- a genuine cross-centre,
cross-device transfer test, which is a harder and more honest question than a within-dataset split.

Label mapping, HAM to the Derm7pt five-group taxonomy:

    melanoma -> MEL      nevus -> NEV      bcc -> BCC      bkl -> SK
    akiec, df, vasc -> MISC

`bkl` (benign keratosis-like lesions) maps to SK because seborrhoeic keratosis is its dominant
constituent; the mapping is declared here rather than buried, because it is a judgement call and a
reader may reasonably disagree with it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.metrics import classification_summary
from medguard_v2.eval.statistics import binary_auc
from medguard_v2.eval.taxonomy import assert_labels_match

REPO = Path(__file__).resolve().parents[3]
HAM_FEATURES = REPO / "research/experiments/derm_perception/v2_tournament/features/dermlip/features.npy"
HAM_SPLIT = REPO / "research/manifests/ham10000_split_v2.json"
DERM7PT_FEATURES = REPO / "research/evidence/derm7pt_v1/features/dermlip/features.npz"
OUT = REPO / "research/evidence/derm7pt_v1/g7_ham_scaled_classifier.json"

SEED = 20260811
C_GRID = (0.1, 1.0, 10.0, 100.0)

HAM_TO_GROUP = {
    "melanoma": "MEL",
    "nevus": "NEV",
    "bcc": "BCC",
    "bkl": "SK",
    "akiec": "MISC",
    "df": "MISC",
    "vasc": "MISC",
}


def load_ham() -> Tuple[np.ndarray, List[str], List[str]]:
    """Features, group labels, lesion cluster ids -- in the manifest's own record order."""
    features = np.load(HAM_FEATURES)
    split = json.loads(HAM_SPLIT.read_text(encoding="utf-8"))
    records = [r for r in split["records"] if r["split"] in ("train", "calibration")]
    if len(records) != features.shape[0]:
        raise RuntimeError(
            f"feature/record mismatch: {features.shape[0]} vectors vs {len(records)} records. "
            "The feature file is ordered by the train+calibration manifest; if that ordering "
            "changed, every label below is wrong."
        )
    unknown = {r["label"] for r in records} - set(HAM_TO_GROUP)
    if unknown:
        raise RuntimeError(f"unmapped HAM labels: {sorted(unknown)}")
    labels = [HAM_TO_GROUP[r["label"]] for r in records]
    clusters = [str(r["cluster_id"]) for r in records]
    return features, labels, clusters


def _select_c(x: np.ndarray, y: Sequence[str], groups: Sequence[str]) -> float:
    """Choose C by lesion-grouped cross-validation, so the same lesion never spans folds."""
    from sklearn.model_selection import StratifiedGroupKFold

    y_arr = np.asarray(y)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_c, best = 10.0, -1.0
    for c in C_GRID:
        correct = 0
        for tr, va in folds.split(x, y_arr, groups=np.asarray(groups)):
            model = LogisticRegression(max_iter=5000, C=c, random_state=SEED)
            model.fit(x[tr], y_arr[tr])
            correct += int((model.predict(x[va]) == y_arr[va]).sum())
        score = correct / len(y_arr)
        if score > best:
            best_c, best = c, score
    return best_c


def _evaluate(model, x, truth: Sequence[str], labels: Sequence[str]) -> Dict:
    predicted = list(model.predict(x))
    summary = classification_summary(list(truth), predicted, list(labels))
    mel_index = list(model.classes_).index("MEL") if "MEL" in model.classes_ else None
    truth_mel = [t == "MEL" for t in truth]
    out = {
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "macro_f1": round(summary["macro_f1"], 4),
        "cohen_kappa": round(summary["cohen_kappa"], 4),
        "per_class": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in summary["per_class"].items()
        },
    }
    if mel_index is not None:
        score = model.predict_proba(x)[:, mel_index]
        out["melanoma_auroc"] = round(binary_auc(truth_mel, [float(s) for s in score]), 4)
        flag = [p == "MEL" for p in predicted]
        tp = sum(1 for f, t in zip(flag, truth_mel) if f and t)
        fn = sum(1 for f, t in zip(flag, truth_mel) if not f and t)
        fp = sum(1 for f, t in zip(flag, truth_mel) if f and not t)
        tn = sum(1 for f, t in zip(flag, truth_mel) if not f and not t)
        out["melanoma_at_argmax"] = {
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        }
        # The clinically relevant operating points for a rule-out task.
        out["specificity_at_sensitivity"] = {}
        order = np.argsort(-score)
        n_pos = sum(truth_mel)
        for target in (0.90, 0.95):
            need = int(np.ceil(target * n_pos))
            hits, cut = 0, float(score[order[-1]])
            for i in order:
                if truth_mel[i]:
                    hits += 1
                    if hits >= need:
                        cut = float(score[i])
                        break
            pred = [s >= cut for s in score]
            tn2 = sum(1 for p, t in zip(pred, truth_mel) if not p and not t)
            fp2 = sum(1 for p, t in zip(pred, truth_mel) if p and not t)
            tp2 = sum(1 for p, t in zip(pred, truth_mel) if p and t)
            out["specificity_at_sensitivity"][f"{target:.2f}"] = {
                "achieved_sensitivity": round(tp2 / n_pos, 4) if n_pos else None,
                "specificity": round(tn2 / (tn2 + fp2), 4) if (tn2 + fp2) else None,
                "threshold": round(cut, 4),
            }
    return out


def run() -> Dict:
    cases = load_cases()
    labels = list(assert_labels_match({c.diagnosis_group for c in cases}, "derm7pt"))
    d7_train = [c for c in cases if c.split == "train"]
    d7_valid = [c for c in cases if c.split == "valid"]

    d7_feat = np.load(DERM7PT_FEATURES)
    index = {str(c): i for i, c in enumerate(d7_feat["case_num"])}
    x_d7_train = np.stack([d7_feat["features"][index[str(c.case_num)]] for c in d7_train])
    x_d7_valid = np.stack([d7_feat["features"][index[str(c.case_num)]] for c in d7_valid])
    y_d7_train = [c.diagnosis_group for c in d7_train]
    y_d7_valid = [c.diagnosis_group for c in d7_valid]

    x_ham, y_ham, clusters_ham = load_ham()

    # Sanity: if the feature/record ordering were wrong, a lesion-grouped CV on HAM would sit
    # near chance. This is the check that the join is real.
    c_ham = _select_c(x_ham, y_ham, clusters_ham)
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_arr = np.asarray(y_ham)
    correct = 0
    for tr, va in folds.split(x_ham, y_arr, groups=np.asarray(clusters_ham)):
        m = LogisticRegression(max_iter=5000, C=c_ham, random_state=SEED)
        m.fit(x_ham[tr], y_arr[tr])
        correct += int((m.predict(x_ham[va]) == y_arr[va]).sum())
    ham_internal_accuracy = correct / len(y_arr)

    # --- the comparison -----------------------------------------------------------
    c_d7 = 10.0  # the registered v2 value
    baseline = LogisticRegression(max_iter=5000, C=c_d7, random_state=SEED)
    baseline.fit(x_d7_train, y_d7_train)

    scaled = LogisticRegression(max_iter=5000, C=c_ham, random_state=SEED)
    scaled.fit(x_ham, y_ham)

    combined_x = np.vstack([x_ham, x_d7_train])
    combined_y = list(y_ham) + list(y_d7_train)
    combined = LogisticRegression(max_iter=5000, C=c_ham, random_state=SEED)
    combined.fit(combined_x, combined_y)

    report = {
        "gate_id": "G7_ham_scaled_classifier",
        "question": "Does training the diagnosis classifier on HAM10000 beat training it on Derm7pt's 413 cases?",
        "boundaries": {
            "ham_features": "v1-encoded DermLIP, train+calibration only; HAM sealed test excluded",
            "ham_records": len(y_ham),
            "derm7pt_train_records": len(d7_train),
            "evaluated_on": "Derm7pt valid (cross-centre, cross-device transfer)",
            "derm7pt_test_read": False,
            "paid_calls": 0,
        },
        "label_mapping": HAM_TO_GROUP,
        "ham_class_distribution": dict(Counter(y_ham)),
        "sanity": {
            "ham_lesion_grouped_cv_accuracy": round(ham_internal_accuracy, 4),
            "selected_C": c_ham,
            "note": "Near-chance here would mean the feature/label join is misaligned.",
        },
        "arms_on_derm7pt_valid": {
            "trained_on_derm7pt_413": _evaluate(baseline, x_d7_valid, y_d7_valid, labels),
            "trained_on_ham_8024": _evaluate(scaled, x_d7_valid, y_d7_valid, labels),
            "trained_on_both": _evaluate(combined, x_d7_valid, y_d7_valid, labels),
        },
    }

    base = report["arms_on_derm7pt_valid"]["trained_on_derm7pt_413"]
    best_name = max(
        report["arms_on_derm7pt_valid"],
        key=lambda k: report["arms_on_derm7pt_valid"][k].get("melanoma_auroc") or 0.0,
    )
    report["verdict"] = {
        "best_arm_by_melanoma_auroc": best_name,
        "baseline_melanoma_auroc": base.get("melanoma_auroc"),
        "best_melanoma_auroc": report["arms_on_derm7pt_valid"][best_name].get("melanoma_auroc"),
        "note": (
            "Selection here is descriptive: this gate asks whether more data helps, on valid. "
            "The arm actually used must be fixed on train before any confirmatory cohort."
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
