"""Two-view melanoma-detection ablation + diagnosis metric surface on the PROMOTED perception.

The ablation block the paper used to cite (classifier 0.5941/0.9048, criteria 0.5743/0.8946,
union 0.7426/0.8503, Venn 43/17/15/26, union sensitivity gain +0.1485) describes the
PRE-promotion arms. The promoted system (``run_cohort(promoted=True)``) changes both views:
diagnosis = dermlip+metadata14 unbalanced C=100 (+ AUROC-invariant SK/MISC operating point),
criteria = dermlip_tta+metadata14 lr_balanced per-head C, read by argmax. This script recomputes
the full two-view surface on Derm7pt TEST (n=395, 101 melanomas) for the promoted arms:

* classifier-alone / criteria-alone / union sensitivity, specificity, Youden J, confusion,
  and case-level bootstrap CIs;
* complementarity Venn counts + discordant odds ratio + exact McNemar (classifier vs union);
* the union sensitivity GAIN over the classifier (the decision-relevant, DCA-adjacent statistic)
  with a PAIRED case-level bootstrap CI, plus the promoted-vs-base union sensitivity delta with
  its own paired CI;
* the promoted diagnosis 5-class summaries (argmax AND operating point), melanoma AUROC with
  bootstrap CI, sensitivity at fixed specificity, the full ROC curve, and UNCALIBRATED
  ECE/AURC of the confidence the promoted path actually displays (no temperature-scaling
  artifact exists for the promoted head -- stated, not hidden).

NOT a new confirmatory look: the promoted configuration was already evaluated once on TEST in
``perception_phase2_safety_preserving.json``; this is a deterministic re-derivation of that same
frozen configuration, and the script refuses to emit anything unless it exactly reproduces both
that artifact's promoted numbers AND the frozen pre-promotion references.

Output: research/evidence/derm7pt_v1/promoted_two_view_ablation.json

Run::
    venv_win/Scripts/python.exe scripts/dev/build_promoted_two_view_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import score_criteria  # noqa: E402
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD  # noqa: E402
from medguard_v2.data.derm7pt import DERM7PT_CLASS_ORDER, load_cases  # noqa: E402
from medguard_v2.eval.metrics import (  # noqa: E402
    aurc,
    classification_summary,
    ece,
    sensitivity_at_fixed_specificity,
)
from medguard_v2.eval.statistics import (  # noqa: E402
    binary_auc,
    clustered_bootstrap_ci,
    mcnemar_exact,
    odds_ratio_discordant,
    paired_clustered_bootstrap_difference,
)
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    PROMOTED_CRITERIA_C,
    PROMOTED_DIAGNOSIS_C,
    PROMOTED_OPERATING_POINT,
    SEED,
    _load_features,
    _promoted_feature_matrices,
    apply_operating_point,
    fit_readers,
    fit_readers_promoted,
)

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
OUT = EVIDENCE / "promoted_two_view_ablation.json"

LABELS = list(DERM7PT_CLASS_ORDER)
OP = dict(PROMOTED_OPERATING_POINT)
N_BOOT = 2000
BOOT_SEED = 20260712            # same convention as test_metrics_complete.json
SPECIFICITY_TARGETS = (0.90, 0.95)

# Frozen references reproduced as wiring checks before anything promoted is emitted.
REF_BASE = {
    "classifier": {"sensitivity": 0.5941, "specificity": 0.9048},
    "criteria": {"sensitivity": 0.5743, "specificity": 0.8946},
    "union": {"sensitivity": 0.7426, "specificity": 0.8503},
    "venn": {"both": 43, "classifier_only": 17, "criteria_only": 15, "neither": 26},
}
REF_PROMOTED = {
    "mel_auroc": 0.9151,
    "accuracy_argmax": 0.762,
    "accuracy_op": 0.7696,
    "criteria": {"sensitivity": 0.7921, "specificity": 0.7041},
    "union": {"sensitivity": 0.8614, "specificity": 0.6973},
    "venn": {"both": 58, "classifier_only": 7, "criteria_only": 22, "neither": 14},
    "classifier_sensitivity": 0.6436,   # MEL recall (argmax == operating point)
}


def _round4(x):
    return round(float(x), 4)


def _confusion(flag: Sequence[bool], truth: Sequence[bool]) -> Dict[str, object]:
    tp = sum(1 for f, t in zip(flag, truth) if f and t)
    fn = sum(1 for f, t in zip(flag, truth) if not f and t)
    fp = sum(1 for f, t in zip(flag, truth) if f and not t)
    tn = sum(1 for f, t in zip(flag, truth) if not f and not t)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "sensitivity": _round4(sens) if sens is not None else None,
        "specificity": _round4(spec) if spec is not None else None,
        "youden_j": _round4(sens + spec - 1) if None not in (sens, spec) else None,
    }


def _sens_metric(sample: Sequence[Tuple[bool, bool]]) -> float:
    tp = sum(1 for t, f in sample if t and f)
    fn = sum(1 for t, f in sample if t and not f)
    return tp / (tp + fn) if (tp + fn) else 0.0


def _spec_metric(sample: Sequence[Tuple[bool, bool]]) -> float:
    tn = sum(1 for t, f in sample if not t and not f)
    fp = sum(1 for t, f in sample if not t and f)
    return tn / (tn + fp) if (tn + fp) else 0.0


def _ci(rows, metric) -> Dict[str, object]:
    interval = clustered_bootstrap_ci(
        rows, [str(i) for i in range(len(rows))], metric, n_boot=N_BOOT, seed=BOOT_SEED,
    )
    return {
        "estimate": _round4(interval.estimate),
        "ci95_low": _round4(interval.low),
        "ci95_high": _round4(interval.high),
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
    }


def _arm(truth: Sequence[bool], flag: Sequence[bool]) -> Dict[str, object]:
    rows = list(zip(truth, flag))
    return {
        **_confusion(flag, truth),
        "sensitivity_ci95": _ci(rows, _sens_metric),
        "specificity_ci95": _ci(rows, _spec_metric),
    }


def _venn(truth, clf, chk) -> Dict[str, int]:
    return {
        "melanomas": sum(1 for t in truth if t),
        "both": sum(1 for t, a, b in zip(truth, clf, chk) if t and a and b),
        "classifier_only": sum(1 for t, a, b in zip(truth, clf, chk) if t and a and not b),
        "criteria_only": sum(1 for t, a, b in zip(truth, clf, chk) if t and b and not a),
        "neither": sum(1 for t, a, b in zip(truth, clf, chk) if t and not a and not b),
    }


def _summary_block(y_true, y_pred) -> Dict[str, object]:
    s = classification_summary(y_true, y_pred, LABELS)
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


def _roc_curve(truth: Sequence[bool], scores: Sequence[float]) -> List[Dict[str, object]]:
    values = [float(s) for s in scores]
    thresholds = sorted(set(values) | {max(values) + 1.0}, reverse=True)
    curve = []
    for t in thresholds:
        flag = [s >= t for s in values]
        stats = _confusion(flag, truth)
        spec = stats["specificity"]
        curve.append({
            "threshold": round(t, 6),
            "sensitivity_tpr": stats["sensitivity"],
            "specificity": spec,
            "fpr": _round4(1.0 - spec) if spec is not None else None,
        })
    return curve


def build() -> Dict[str, object]:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    truth_group = [c.diagnosis_group for c in test]
    truth_mel = [g == "MEL" for g in truth_group]

    # ---- promoted readers -------------------------------------------------------------------
    diagnosis_features, criteria_features = _promoted_feature_matrices(cases)
    diagnosis, heads = fit_readers_promoted(train, diagnosis_features, criteria_features)
    xd_test = np.stack([diagnosis_features[str(c.case_num)] for c in test])
    xc_test = np.stack([criteria_features[str(c.case_num)] for c in test])

    posterior = diagnosis.predict_proba(xd_test)
    classes = [str(c) for c in diagnosis.classes_]
    mel_idx = classes.index("MEL")
    mel_scores = [float(p) for p in posterior[:, mel_idx]]
    pred_argmax = [classes[i] for i in posterior.argmax(axis=1)]
    pred_op, conf_op = apply_operating_point(posterior, diagnosis.classes_, OP)

    clf_flag_argmax = [p == "MEL" for p in pred_argmax]
    clf_flag_op = [p == "MEL" for p in pred_op]
    if clf_flag_argmax != clf_flag_op:
        raise SystemExit("operating point unexpectedly changed melanoma flags")

    head_pred = {n: [str(p) for p in h.predict(xc_test)] for n, h in heads.items()}
    chk_scores = []
    for i in range(len(test)):
        values = {name: head_pred[name][i] for name in CRITERION_NAMES}
        chk_scores.append(score_criteria(values).score_min)
    chk_flag = [s >= MELANOMA_SUSPICION_THRESHOLD for s in chk_scores]
    union_flag = [a or b for a, b in zip(clf_flag_argmax, chk_flag)]

    # ---- base (pre-promotion) arms, wiring checks -------------------------------------------
    features_base = _load_features()
    diagnosis_base, heads_base = fit_readers(train, features_base)
    xb_test = np.stack([features_base[str(c.case_num)] for c in test])
    clf_flag_base = [str(p) == "MEL" for p in diagnosis_base.predict(xb_test)]
    head_pred_base = {n: [str(p) for p in h.predict(xb_test)] for n, h in heads_base.items()}
    chk_flag_base = []
    for i in range(len(test)):
        values = {name: head_pred_base[name][i] for name in CRITERION_NAMES}
        chk_flag_base.append(score_criteria(values).score_min >= MELANOMA_SUSPICION_THRESHOLD)
    union_flag_base = [a or b for a, b in zip(clf_flag_base, chk_flag_base)]

    # ---- wiring self-checks -----------------------------------------------------------------
    def _ss(flag):
        c = _confusion(flag, truth_mel)
        return {"sensitivity": c["sensitivity"], "specificity": c["specificity"]}

    auroc_point = binary_auc(truth_mel, mel_scores)
    acc_argmax = _round4(sum(1 for p, t in zip(pred_argmax, truth_group) if p == t) / len(test))
    acc_op = _round4(sum(1 for p, t in zip(pred_op, truth_group) if p == t) / len(test))
    checks = {
        "base_classifier": {"got": _ss(clf_flag_base), "expected": REF_BASE["classifier"]},
        "base_criteria": {"got": _ss(chk_flag_base), "expected": REF_BASE["criteria"]},
        "base_union": {"got": _ss(union_flag_base), "expected": REF_BASE["union"]},
        "base_venn": {
            "got": {k: v for k, v in _venn(truth_mel, clf_flag_base, chk_flag_base).items()
                    if k != "melanomas"},
            "expected": REF_BASE["venn"],
        },
        "promoted_mel_auroc": {"got": _round4(auroc_point), "expected": REF_PROMOTED["mel_auroc"]},
        "promoted_accuracy_argmax": {"got": acc_argmax, "expected": REF_PROMOTED["accuracy_argmax"]},
        "promoted_accuracy_op": {"got": acc_op, "expected": REF_PROMOTED["accuracy_op"]},
        "promoted_criteria": {"got": _ss(chk_flag), "expected": REF_PROMOTED["criteria"]},
        "promoted_union": {"got": _ss(union_flag), "expected": REF_PROMOTED["union"]},
        "promoted_venn": {
            "got": {k: v for k, v in _venn(truth_mel, clf_flag_argmax, chk_flag).items()
                    if k != "melanomas"},
            "expected": REF_PROMOTED["venn"],
        },
        "promoted_classifier_sensitivity": {
            "got": _ss(clf_flag_argmax)["sensitivity"],
            "expected": REF_PROMOTED["classifier_sensitivity"],
        },
    }
    failures = []

    def _close(g, e, tol=0.0006):
        if isinstance(e, dict):
            return all(_close(g[k], e[k], tol) for k in e)
        if isinstance(e, int) and not isinstance(e, bool):
            return g == e
        return abs(float(g) - float(e)) <= tol

    for name, ch in checks.items():
        ch["reproduced"] = bool(_close(ch["got"], ch["expected"]))
        ch["source"] = ("perception_phase2_safety_preserving.json" if name.startswith("promoted")
                        else "test_metrics_complete.json")
        if not ch["reproduced"]:
            failures.append(name)
    if failures:
        raise SystemExit(
            f"WIRING DRIFT: failed to reproduce frozen references: {failures}\n"
            + json.dumps({k: checks[k] for k in failures}, indent=2, default=str)
        )

    # ---- promoted arm surface ---------------------------------------------------------------
    arms = {
        "classifier_alone": _arm(truth_mel, clf_flag_argmax),
        "criteria_alone": _arm(truth_mel, chk_flag),
        "union_rule": _arm(truth_mel, union_flag),
    }
    arms["classifier_alone"]["melanoma_auroc"] = _round4(auroc_point)
    arms["classifier_alone"]["note"] = (
        "Melanoma flag identical at argmax and at the SK/MISC operating point (asserted): "
        "re-ranking SK/MISC never touches the MEL column."
    )
    arms["criteria_alone"]["auroc_note"] = (
        "No AUROC: 'seven-point score >= threshold 3' is a binary rule, not a ranked score."
    )
    arms["union_rule"]["auroc_note"] = (
        "No AUROC: OR of two binary flags, not a ranked score."
    )

    venn = _venn(truth_mel, clf_flag_argmax, chk_flag)
    odds = odds_ratio_discordant(venn["classifier_only"], venn["criteria_only"])

    correct_clf = [f == t for f, t in zip(clf_flag_argmax, truth_mel)]
    correct_union = [f == t for f, t in zip(union_flag, truth_mel)]
    b_disc, c_disc, p_value = mcnemar_exact(correct_clf, correct_union)
    mcnemar = {
        "classifier_only_correct": b_disc,
        "union_only_correct": c_disc,
        "p_value": round(p_value, 6),
        "classifier_binary_accuracy": _round4(sum(correct_clf) / len(correct_clf)),
        "union_binary_accuracy": _round4(sum(correct_union) / len(correct_union)),
    }

    # Union sensitivity gain over the classifier, with a PAIRED case-level bootstrap CI.
    rows_union = list(zip(truth_mel, union_flag))
    rows_clf = list(zip(truth_mel, clf_flag_argmax))
    gain_ci = paired_clustered_bootstrap_difference(
        rows_union, rows_clf, [str(i) for i in range(len(test))], _sens_metric,
        n_boot=N_BOOT, seed=BOOT_SEED,
    )
    # Promoted union vs BASE union sensitivity delta (same cases, paired).
    rows_union_base = list(zip(truth_mel, union_flag_base))
    delta_vs_base_ci = paired_clustered_bootstrap_difference(
        rows_union, rows_union_base, [str(i) for i in range(len(test))], _sens_metric,
        n_boot=N_BOOT, seed=BOOT_SEED,
    )

    def _interval(iv):
        return {"estimate": _round4(iv.estimate), "ci95_low": _round4(iv.low),
                "ci95_high": _round4(iv.high), "n_boot": N_BOOT, "seed": BOOT_SEED}

    # ---- diagnosis 5-class surface ----------------------------------------------------------
    auroc_rows = list(zip(truth_mel, mel_scores))
    auroc_ci = _ci(auroc_rows, lambda s: binary_auc([r[0] for r in s], [r[1] for r in s]))

    clinical_operating_points = {}
    for target in SPECIFICITY_TARGETS:
        result = sensitivity_at_fixed_specificity(truth_mel, mel_scores, target)
        if result is None:
            raise SystemExit(f"no operating point reaches target specificity {target}")
        clinical_operating_points[f"specificity_{target}"] = result

    # Confidence the promoted path actually displays: posterior mass on the label chosen at the
    # operating point (run_cohort promoted). RAW, UNCALIBRATED -- no temperature-scaling artifact
    # exists for the promoted head, unlike perception_baseline.json's calibrated confidence.
    correct_op = [p == t for p, t in zip(pred_op, truth_group)]
    calibration = {
        "confidence_source": (
            "raw promoted-head posterior mass on the operating-point label (what "
            "run_cohort(promoted=True) writes into contracts); UNCALIBRATED"
        ),
        "ece_uncalibrated": _round4(ece(conf_op, correct_op)),
        "aurc": _round4(aurc(conf_op, correct_op)),
        "n_bins": 10,
        "honesty_note": (
            "The pre-promotion calibration figures (ECE 0.0455) used a temperature-calibrated "
            "confidence from perception_baseline.json; no calibration has been fit for the "
            "promoted head, so these are raw-posterior diagnostics and not directly comparable "
            "to the calibrated base figure."
        ),
        "base_arm_reference": {
            "ece_calibrated": 0.0455,
            "source": "test_metrics_complete.json calibration (temperature-calibrated base head)",
        },
    }

    return {
        "protocol": "derm7pt-promoted-two-view-ablation-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cases": len(test),
        "n_melanomas": sum(truth_mel),
        "boundaries": {
            "fit_split": "train",
            "n_fit_cases": len(train),
            "hyperparameters": {
                "seed": SEED,
                "diagnosis": f"dermlip+metadata14, unbalanced LR C={PROMOTED_DIAGNOSIS_C}, "
                             f"operating point {OP}",
                "criteria": "dermlip_tta+metadata14, lr_balanced, per-head C "
                            "(run_cohort.PROMOTED_CRITERIA_C), argmax",
            },
            "note": (
                "Deterministic re-derivation of the frozen promoted configuration already "
                "evaluated on TEST in perception_phase2_safety_preserving.json. Nothing is "
                "selected here; the script refuses to emit unless it exactly reproduces both the "
                "promoted read and the frozen pre-promotion references."
            ),
        },
        "diagnosis_5class": {
            "argmax": _summary_block(truth_group, pred_argmax),
            "operating_point": {
                "policy": {"rule": "argmax(posterior/weight)", "weights": OP},
                **_summary_block(truth_group, pred_op),
            },
        },
        "melanoma_auroc": {
            "point_estimate": _round4(auroc_point),
            "ci95": auroc_ci,
            "source": "promoted diagnosis head raw MEL posterior (AUROC-invariant to the "
                      "operating point)",
        },
        "clinical_operating_points": clinical_operating_points,
        "calibration": calibration,
        "roc_curve": _roc_curve(truth_mel, mel_scores),
        "arms": arms,
        "complementarity_venn": venn,
        "complementarity_odds_ratio": odds,
        "mcnemar_classifier_vs_union": mcnemar,
        "union_sensitivity_gain": {
            "gain_over_classifier": _round4(
                arms["union_rule"]["sensitivity"] - arms["classifier_alone"]["sensitivity"]),
            "paired_bootstrap_ci95": _interval(gain_ci),
            "note": (
                "Union minus classifier melanoma sensitivity on the same 395 cases; paired "
                "case-level bootstrap. The decision-relevant statistic the two-view design is "
                "accepted on."
            ),
        },
        "union_sensitivity_delta_vs_base_union": {
            "delta": _round4(
                arms["union_rule"]["sensitivity"] - _confusion(union_flag_base, truth_mel)["sensitivity"]),
            "paired_bootstrap_ci95": _interval(delta_vs_base_ci),
            "note": "Promoted union minus PRE-promotion union sensitivity, paired on the same cases.",
        },
        "disclosed_specificity_cost": (
            "Union melanoma specificity falls 0.8503 (base) -> "
            f"{arms['union_rule']['specificity']} (promoted): the balanced criteria heads raise "
            "false positives along with true positives. Disclosed, not hidden -- the architecture "
            "surfaces the disagreement for human adjudication, and on a melanoma CDSS a missed "
            "melanoma dominates a benign false alarm."
        ),
        "base_arm_reference": {
            "label": "PRE-PROMOTION arms (dermlip 512-d readers, argmax), reproduced exactly",
            "classifier_alone": {**_confusion(clf_flag_base, truth_mel), "melanoma_auroc": 0.895},
            "criteria_alone": _confusion(chk_flag_base, truth_mel),
            "union_rule": _confusion(union_flag_base, truth_mel),
            "complementarity_venn": _venn(truth_mel, clf_flag_base, chk_flag_base),
            "union_sensitivity_gain_over_classifier": _round4(
                _confusion(union_flag_base, truth_mel)["sensitivity"]
                - _confusion(clf_flag_base, truth_mel)["sensitivity"]),
            "source": "test_metrics_complete.json / ablation_table.json (frozen)",
        },
        "wiring_self_checks": checks,
    }


def main() -> None:
    t0 = time.perf_counter()
    report = build()
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")

    arms = report["arms"]
    base = report["base_arm_reference"]
    print("\n=== promoted two-view arms (test n=395) ===")
    for name in ("classifier_alone", "criteria_alone", "union_rule"):
        a, b = arms[name], base[name]
        print(f"  {name:16s}: sens {a['sensitivity']} spec {a['specificity']} J {a['youden_j']}"
              f"   (base {b['sensitivity']}/{b['specificity']}/J {b['youden_j']})")
    print(f"  venn promoted    : {report['complementarity_venn']}")
    print(f"  venn base        : {base['complementarity_venn']}")
    g = report["union_sensitivity_gain"]
    print(f"  union gain vs clf: +{g['gain_over_classifier']} "
          f"CI [{g['paired_bootstrap_ci95']['ci95_low']}, {g['paired_bootstrap_ci95']['ci95_high']}]"
          f"   (base gain +{base['union_sensitivity_gain_over_classifier']})")
    d = report["union_sensitivity_delta_vs_base_union"]
    print(f"  union delta vs base union: +{d['delta']} "
          f"CI [{d['paired_bootstrap_ci95']['ci95_low']}, {d['paired_bootstrap_ci95']['ci95_high']}]")
    m = report["mcnemar_classifier_vs_union"]
    print(f"  McNemar clf vs union: {m['classifier_only_correct']}/{m['union_only_correct']} "
          f"p={m['p_value']}")
    print(f"  odds ratio (discordant): {report['complementarity_odds_ratio']}")
    print(f"\n  MEL AUROC {report['melanoma_auroc']['point_estimate']} "
          f"CI [{report['melanoma_auroc']['ci95']['ci95_low']}, "
          f"{report['melanoma_auroc']['ci95']['ci95_high']}]")
    print(f"  acc argmax {report['diagnosis_5class']['argmax']['accuracy']} / "
          f"op {report['diagnosis_5class']['operating_point']['accuracy']}")
    print(f"  ECE (uncalibrated, op label) {report['calibration']['ece_uncalibrated']}  "
          f"AURC {report['calibration']['aurc']}")
    print(f"  all wiring checks reproduced: "
          f"{all(v['reproduced'] for v in report['wiring_self_checks'].values())}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
