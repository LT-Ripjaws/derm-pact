"""Offline MedGuard diagnosis/actionable/selective numbers from the frozen probe.

Joins the 160-lesion frozen predictions with development truth and runs the same
scorers the H2H would use. ZERO models, ZERO API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("E:/Studies/Research/MedGuard-RAG-Journal")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medguard.config import MALIGNANT_LABELS
from medguard.eval.metrics import (
    DERM_CLASS_ORDER,
    classification_summary,
    actionable_confusion,
    sensitivity_at_fixed_specificity,
    aurc,
    selective_accuracy_at_coverage,
    triage_rates,
    is_actionable,
    ece,
)
from medguard.eval.h2h_production import load_development_truth

PRED = PROJECT_ROOT / "research/experiments/medguard_h2h_v1/perception/dermlip_v1/predictions.jsonl"


def main():
    truth = load_development_truth()  # {cluster_id: label}
    rows = [json.loads(l) for l in PRED.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"predictions={len(rows)}  truth_labels={len(truth)}")

    y_true, y_pred, act_true, act_scores, conf, correct = [], [], [], [], [], []
    unmatched = 0
    label_set = set(DERM_CLASS_ORDER)
    bad_labels = set()
    for r in rows:
        cid = r["cluster_id"]
        if cid not in truth:
            unmatched += 1
            continue
        t = str(truth[cid]).lower()
        p = str(r["committed_diagnosis"]).lower()
        if t not in label_set:
            bad_labels.add(("truth", t))
        if p not in label_set:
            bad_labels.add(("pred", p))
        y_true.append(t)
        y_pred.append(p)
        act_true.append(is_actionable(t))
        act_scores.append(float(r["actionable_score"]))
        conf.append(float(r["calibrated_confidence"]))
        correct.append(p == t)
    print(f"joined={len(y_true)}  unmatched={unmatched}  bad_labels={sorted(bad_labels)}")
    if bad_labels:
        print("ABORT: label mismatch (need normalization)")
        return

    # --- Diagnosis (7-class) ---
    cs = classification_summary(y_true, y_pred)
    print("\n=== DIAGNOSIS (7-class, 160 dev lesions) ===")
    print(f"accuracy           {cs['accuracy']:.4f}")
    print(f"balanced_accuracy  {cs['balanced_accuracy']:.4f}")
    print(f"macro_f1           {cs['macro_f1']:.4f}")
    print(f"weighted_f1        {cs['weighted_f1']:.4f}")
    print(f"mcc                {cs['mcc']:.4f}")
    print(f"cohen_kappa        {cs['cohen_kappa']:.4f}")
    print("per-class (precision/recall/f1/support):")
    for lab in cs["labels"]:
        pc = cs["per_class"][lab]
        print(f"  {lab:9s} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")

    # true-class support (how balanced is this set?)
    from collections import Counter
    print("true-class counts:", dict(Counter(y_true)))

    # --- Actionable / malignant 2x2 + AUROC (malignant AUC) ---
    ac = actionable_confusion(y_pred, y_true, scores=act_scores)
    print("\n=== ACTIONABLE / MALIGNANT (positive = {melanoma,bcc,akiec}) ===")
    print(f"2x2 [[TP,FN],[FP,TN]] = {ac['confusion_matrix']}")
    print(f"prevalence   {ac['prevalence']}")
    print(f"sensitivity  {ac['sensitivity']}   specificity {ac['specificity']}")
    print(f"ppv          {ac['ppv']}   npv {ac['npv']}")
    print(f"malignant AUROC (actionable_score)  {ac['auroc']}")
    sens90 = sensitivity_at_fixed_specificity(act_true, act_scores, 0.90)
    print(f"sensitivity @ 0.90 fixed specificity: {sens90}")

    tr = triage_rates(y_pred, y_true)
    print(f"triage: high_risk_detection(malig recall)={tr['high_risk_detection']} "
          f"under_triage(missed cancer)={tr['under_triage']} over_triage={tr['over_triage']}")

    # --- Selective prediction (probe confidence) ---
    print("\n=== SELECTIVE (calibrated_confidence vs correctness) ===")
    print(f"AURC (lower better)              {aurc(conf, correct):.4f}")
    print(f"selective acc @ 0.65 coverage    {selective_accuracy_at_coverage(conf, correct, 0.65)}")
    print(f"selective acc @ 0.80 coverage    {selective_accuracy_at_coverage(conf, correct, 0.80)}")
    print(f"ECE                              {ece(conf, correct)}")

    out = {
        "n": len(y_true),
        "diagnosis": {k: cs[k] for k in ("accuracy","balanced_accuracy","macro_f1","weighted_f1","mcc","cohen_kappa")},
        "actionable": {"sensitivity": ac["sensitivity"], "specificity": ac["specificity"],
                       "malignant_auroc": ac["auroc"], "sens_at_90_spec": sens90,
                       "confusion_2x2": ac["confusion_matrix"]},
        "triage": tr,
        "selective": {"aurc": aurc(conf, correct),
                      "sel_acc_0.65": selective_accuracy_at_coverage(conf, correct, 0.65),
                      "ece": ece(conf, correct)},
    }
    outpath = Path(__file__).parent / "offline_numbers.json"
    outpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
