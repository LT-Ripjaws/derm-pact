"""SUPERSEDED (2026-08-14) by scripts/dev/derm7pt_p2_safety_promote.py -- do not re-run.

This is the MONOLITHIC phase-2 promotion: dermlip_tta+dermlip_clinic_tta, class-balanced. Its
one-shot TEST read is frozen in research/evidence/derm7pt_v1/perception_phase2_test.json and is
retained as the "honest cost" record (5-class acc +0.015 and criteria binary-F1 +0.0545, but MEL
AUROC REGRESSED 0.895 -> 0.867). Because that melanoma-AUROC regression is unacceptable on a
melanoma CDSS, run_cohort(promoted=True) was re-wired to the SAFETY-PRESERVING config
(dermlip+metadata14 unbalanced C=100, MEL AUROC 0.9151). This script's build_test_read() still
carries a self-check pinned to the monolith's valid balanced accuracy (0.7151), which the re-wired
promoted path no longer reproduces -- so main() now refuses to run rather than emit a drifted read.
Candidate (c) (this monolith) is reproduced, alongside (a) and (b), by the 3-candidate table in
scripts/dev/derm7pt_p2_safety_promote.py.

Original purpose: wire the phase-1 winners into the live pipeline, rebuild the VALID contracts
with the promoted heads, and do the ONE-SHOT Derm7pt TEST read.

Nothing is selected here. Every hyperparameter is frozen in
``research/evidence/derm7pt_v1/perception_uplift_phase1.json`` (final_config + operating_point_policy)
and re-declared as constants in ``medguard_v2/pipeline/run_cohort.py`` (PROMOTED_*). This script only
*applies* them: fit on train, score valid (to reproduce the phase-1 numbers as a wiring self-check),
score test ONCE.

CALIBER-safe framing (baked into the JSON): this is NOT an accuracy-SOTA claim and must not be framed
as beating any prior frozen-probe work -- CALIBER/DERM-GATE is our own prior work on the SAME frozen
probe. The value is minority-class (SK/MISC) recall + calibration + downstream contract quality via
metadata, multi-view TTA, and class-balancing.

Writes:
- research/evidence/derm7pt_v1/cdss_contracts_valid_promoted.json   (promoted VALID contracts, alongside
  the frozen cdss_contracts_valid.json -- the pinned one is never overwritten)
- research/evidence/derm7pt_v1/perception_phase2_test.json          (the one-shot test read)

Run::

    venv_win/Scripts/python.exe scripts/dev/derm7pt_p2_promote.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES  # noqa: E402
from medguard_v2.data.derm7pt import (  # noqa: E402
    CRITERION_VOCABULARY,
    DERM7PT_CLASS_ORDER,
    load_cases,
)
from medguard_v2.eval.metrics import classification_summary  # noqa: E402
from medguard_v2.eval.statistics import binary_auc  # noqa: E402
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    PROMOTED_CRITERIA_C,
    PROMOTED_DIAGNOSIS_C,
    PROMOTED_DIAGNOSIS_VIEWS,
    PROMOTED_OPERATING_POINT,
    apply_operating_point,
    fit_readers_promoted,
    _promoted_feature_matrices,
    run_cohort,
)

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
VALID_CONTRACTS_OUT = EVIDENCE / "cdss_contracts_valid_promoted.json"
TEST_READ_OUT = EVIDENCE / "perception_phase2_test.json"

LABELS = list(DERM7PT_CLASS_ORDER)  # BCC, NEV, MEL, SK, MISC

# Phase-1 frozen valid numbers, for the wiring self-check (final_config blocks).
PHASE1_VALID_DIAG_BALACC = 0.7151
PHASE1_VALID_CRIT_MACRO_BINARY_F1 = 0.7287


def _diagnosis_metrics(diagnosis, xd, cases) -> dict:
    """Full 5-class metrics at plain argmax (comparable to the phase-1 selection) plus the
    SK/MISC-lifting operating point as a secondary block. MEL AUROC from the raw posterior."""
    proba = diagnosis.predict_proba(xd)
    classes = [str(c) for c in diagnosis.classes_]
    y_true = [c.diagnosis_group for c in cases]

    argmax_pred = [classes[i] for i in proba.argmax(axis=1)]
    argmax_summary = classification_summary(y_true, argmax_pred, LABELS)

    mel_idx = classes.index("MEL")
    truth_mel = [c.diagnosis_group == "MEL" for c in cases]
    mel_auroc = float(binary_auc(truth_mel, [float(p) for p in proba[:, mel_idx]]))

    op_pred, _op_conf = apply_operating_point(proba, diagnosis.classes_, PROMOTED_OPERATING_POINT)
    op_summary = classification_summary(y_true, op_pred, LABELS)

    def _block(summary: dict) -> dict:
        return {
            "n": int(summary["n"]),
            "accuracy": round(float(summary["accuracy"]), 4),
            "balanced_accuracy": round(float(summary["balanced_accuracy"]), 4),
            "macro_f1": round(float(summary["macro_f1"]), 4),
            "per_class_recall": {
                lbl: round(float(v["recall"]), 4) for lbl, v in summary["per_class"].items()
            },
            "per_class_support": {
                lbl: int(v["support"]) for lbl, v in summary["per_class"].items()
            },
        }

    return {
        "argmax": {**_block(argmax_summary), "mel_auroc": round(mel_auroc, 4)},
        "operating_point": {
            "policy": {"rule": "argmax(posterior / weight)", "weights": dict(PROMOTED_OPERATING_POINT)},
            **_block(op_summary),
            "mel_auroc": round(mel_auroc, 4),
            "note": "MEL AUROC is threshold-free, so re-ranking SK/MISC does not change it.",
        },
    }


def _criteria_metrics(heads, xc, cases) -> dict:
    """Per-criterion binary-concept F1 (DermAgent: scoring vs non-scoring, macro over the 2 classes)
    and multiclass macro-F1, at plain argmax -- exactly the phase-1 protocol."""
    per_criterion = {}
    binary_f1s = []
    multi_f1s = []
    for name in CRITERION_NAMES:
        spec = CRITERIA_BY_NAME[name]
        head = heads[name]
        pred = [str(p) for p in head.predict(xc)]
        y_true = [str(c.criteria[name]) for c in cases]

        yt = [1 if v in spec.scoring_values else 0 for v in y_true]
        yp = [1 if v in spec.scoring_values else 0 for v in pred]
        binary_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))

        multi_summary = classification_summary(y_true, pred, list(CRITERION_VOCABULARY[name]))
        multi_f1 = float(multi_summary["macro_f1"])

        binary_f1s.append(binary_f1)
        multi_f1s.append(multi_f1)
        per_criterion[name] = {
            "scoring_values": sorted(spec.scoring_values),
            "binary_concept_f1": round(binary_f1, 4),
            "multiclass_macro_f1": round(multi_f1, 4),
        }
    return {
        "macro_binary_concept_f1": round(float(np.mean(binary_f1s)), 4),
        "macro_multiclass_f1": round(float(np.mean(multi_f1s)), 4),
        "per_criterion": per_criterion,
    }


def build_test_read() -> dict:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    test = [c for c in cases if c.split == "test"]

    diag_feats, crit_feats = _promoted_feature_matrices(cases)
    diagnosis, heads = fit_readers_promoted(train, diag_feats, crit_feats)

    xd_valid = np.stack([diag_feats[str(c.case_num)] for c in valid])
    xc_valid = np.stack([crit_feats[str(c.case_num)] for c in valid])
    xd_test = np.stack([diag_feats[str(c.case_num)] for c in test])
    xc_test = np.stack([crit_feats[str(c.case_num)] for c in test])

    valid_diag = _diagnosis_metrics(diagnosis, xd_valid, valid)
    valid_crit = _criteria_metrics(heads, xc_valid, valid)
    test_diag = _diagnosis_metrics(diagnosis, xd_test, test)
    test_crit = _criteria_metrics(heads, xc_test, test)

    # --- wiring self-check: the valid numbers MUST reproduce the frozen phase-1 final_config ----
    repro_diag = valid_diag["argmax"]["balanced_accuracy"]
    repro_crit = valid_crit["macro_binary_concept_f1"]
    diag_ok = abs(repro_diag - PHASE1_VALID_DIAG_BALACC) <= 0.002
    crit_ok = abs(repro_crit - PHASE1_VALID_CRIT_MACRO_BINARY_F1) <= 0.002
    if not (diag_ok and crit_ok):
        raise SystemExit(
            "WIRING DRIFT: promoted valid does not reproduce the frozen phase-1 final_config.\n"
            f"  diagnosis balanced_accuracy: got {repro_diag}, phase-1 {PHASE1_VALID_DIAG_BALACC} "
            f"(ok={diag_ok})\n"
            f"  criteria macro binary F1   : got {repro_crit}, phase-1 "
            f"{PHASE1_VALID_CRIT_MACRO_BINARY_F1} (ok={crit_ok})\n"
            "Refusing to trust the test read until valid reproduces. Check feature order / C / "
            "class_weight."
        )

    report = {
        "protocol": "derm7pt-perception-phase2-test-read-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "one_shot": True,
        "authorization": (
            "The sealed TEST split may be re-read: this is evaluation, not training. Heads are fit "
            "on TRAIN only; TEST is scored once, here."
        ),
        "caliber_safe_framing": (
            "NOT an accuracy-SOTA claim and NOT a claim to beat any prior frozen-probe work. "
            "CALIBER/DERM-GATE is our own prior paper on the SAME frozen DermLIP probe, so we do "
            "not (and cannot) position on beating it on accuracy. The publishable value of this "
            "promotion is minority-class recall (SK/MISC) + calibration + downstream decision-"
            "contract quality, delivered by clinical metadata, horizontal-flip TTA multi-view "
            "fusion, and class-balanced heads -- not a higher headline accuracy."
        ),
        "config": {
            "diagnosis": {
                "features": "+".join(PROMOTED_DIAGNOSIS_VIEWS),
                "family": "lr_balanced",
                "C": PROMOTED_DIAGNOSIS_C,
                "operating_point": dict(PROMOTED_OPERATING_POINT),
                "feature_dim": int(xd_test.shape[1]),
            },
            "criteria": {
                "features": "dermlip_tta+metadata14",
                "family": "lr_balanced",
                "C_per_head": dict(PROMOTED_CRITERIA_C),
                "feature_dim": int(xc_test.shape[1]),
            },
            "fit_split": "train",
            "n_fit_cases": len(train),
            "seed": 20260810,
        },
        "valid_reproduction_self_check": {
            "diagnosis_balanced_accuracy": repro_diag,
            "phase1_frozen_diagnosis_balanced_accuracy": PHASE1_VALID_DIAG_BALACC,
            "criteria_macro_binary_concept_f1": repro_crit,
            "phase1_frozen_criteria_macro_binary_concept_f1": PHASE1_VALID_CRIT_MACRO_BINARY_F1,
            "reproduces_phase1": True,
            "note": "If this ever drifts, the promotion wiring is wrong; the script raises rather "
            "than emit an untrustworthy test read.",
        },
        "valid": {"diagnosis": valid_diag, "criteria": valid_crit},
        "test": {
            "n": len(test),
            "diagnosis": test_diag,
            "criteria": test_crit,
        },
        "comparison": {
            "reference_current_test_pre_promotion": {
                "diagnosis_accuracy": 0.7392,
                "mel_auroc": 0.895,
                "criteria_binary_concept_f1": 0.6623,
                "source": "perception_baseline.json / test_metrics_complete.json (frozen; not re-run here)",
            },
            "reference_valid_selected_phase1": {
                "diagnosis_balanced_accuracy": PHASE1_VALID_DIAG_BALACC,
                "criteria_binary_concept_f1": PHASE1_VALID_CRIT_MACRO_BINARY_F1,
            },
            "promoted_test": {
                "diagnosis_accuracy_argmax": test_diag["argmax"]["accuracy"],
                "diagnosis_balanced_accuracy_argmax": test_diag["argmax"]["balanced_accuracy"],
                "diagnosis_macro_f1_argmax": test_diag["argmax"]["macro_f1"],
                "mel_auroc": test_diag["argmax"]["mel_auroc"],
                "recall_SK_argmax": test_diag["argmax"]["per_class_recall"].get("SK"),
                "recall_MISC_argmax": test_diag["argmax"]["per_class_recall"].get("MISC"),
                "recall_SK_operating_point": test_diag["operating_point"]["per_class_recall"].get("SK"),
                "recall_MISC_operating_point": test_diag["operating_point"]["per_class_recall"].get("MISC"),
                "criteria_binary_concept_f1": test_crit["macro_binary_concept_f1"],
                "criteria_multiclass_macro_f1": test_crit["macro_multiclass_f1"],
            },
            "deltas_test": {
                "diagnosis_accuracy_vs_current": round(
                    test_diag["argmax"]["accuracy"] - 0.7392, 4
                ),
                "mel_auroc_vs_current": round(test_diag["argmax"]["mel_auroc"] - 0.895, 4),
                "criteria_binary_concept_f1_vs_current": round(
                    test_crit["macro_binary_concept_f1"] - 0.6623, 4
                ),
            },
            "note": (
                "Expect the TEST gain to be SMALLER than the valid gain (selection optimism): the "
                "phase-1 levers and head/family were chosen on valid. A smaller-but-positive test "
                "gain is the expected, publishable outcome. Diagnosis accuracy is reported at plain "
                "argmax to match how the valid numbers were selected; the operating point trades a "
                "little balanced accuracy for SK/MISC recall and is reported separately."
            ),
        },
        "verdict": {
            "diagnosis_accuracy_transferred": bool(test_diag["argmax"]["accuracy"] > 0.7392),
            "criteria_binary_f1_transferred": bool(test_crit["macro_binary_concept_f1"] > 0.6623),
            "mel_auroc_regressed": bool(test_diag["argmax"]["mel_auroc"] < 0.895),
            "minority_recall_improved": True,
            "summary": (
                "Two of three headline gains transferred to the untouched test split, both smaller "
                "than on valid (selection optimism, as expected): 5-class accuracy "
                f"{test_diag['argmax']['accuracy']} (+{round(test_diag['argmax']['accuracy']-0.7392,4)} "
                "vs current) and criteria binary-concept F1 "
                f"{test_crit['macro_binary_concept_f1']} "
                f"(+{round(test_crit['macro_binary_concept_f1']-0.6623,4)} vs current). SK recall "
                f"rose to {test_diag['argmax']['per_class_recall'].get('SK')} at argmax and "
                f"{test_diag['operating_point']['per_class_recall'].get('SK')} at the SK/MISC "
                "operating point (SK n=19). THE HONEST COST: MEL AUROC regressed to "
                f"{test_diag['argmax']['mel_auroc']} (-{round(0.895-test_diag['argmax']['mel_auroc'],4)} "
                "vs the current 0.895) -- class-balancing plus the lower-AUROC clinic view distorts "
                "the melanoma-score ranking. On valid MEL AUROC was ~flat (0.8909 vs 0.8921), so "
                "this regression only shows up out-of-sample. Notably the SK/MISC operating point "
                f"raised balanced accuracy on test ({test_diag['operating_point']['balanced_accuracy']} "
                f"vs {test_diag['argmax']['balanced_accuracy']} at argmax), the opposite of its "
                "small valid cost. Consistent with the CALIBER-safe framing: the win is minority "
                "recall + contract quality, not a uniformly better classifier -- MEL ranking is a "
                "real trade-off to disclose, not hide."
            ),
        },
    }
    return report


def main() -> None:
    raise SystemExit(
        "SUPERSEDED: this monolithic promotion regressed melanoma AUROC (0.895 -> 0.867) and has "
        "been replaced. run_cohort(promoted=True) now implements the safety-preserving config. To "
        "(re)produce the promotion, the 3-candidate table, and the promoted VALID contracts, run:\n"
        "    venv_win/Scripts/python.exe scripts/dev/derm7pt_p2_safety_promote.py\n"
        "The monolith's frozen one-shot read remains at "
        "research/evidence/derm7pt_v1/perception_phase2_test.json (candidate c)."
    )


def _superseded_monolith_main() -> None:
    started = time.perf_counter()

    # 1. Promoted VALID contracts (alongside the frozen cdss_contracts_valid.json; never overwrite).
    valid_report = run_cohort("valid", promoted=True)
    VALID_CONTRACTS_OUT.write_text(json.dumps(valid_report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {VALID_CONTRACTS_OUT.relative_to(REPO)} "
          f"(protocol={valid_report['protocol']}, n={valid_report['n_cases']})")

    # 2. One-shot TEST read (with valid-reproduction self-check).
    test_report = build_test_read()
    TEST_READ_OUT.write_text(json.dumps(test_report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TEST_READ_OUT.relative_to(REPO)}")

    td = test_report["test"]["diagnosis"]["argmax"]
    tc = test_report["test"]["criteria"]
    op = test_report["test"]["diagnosis"]["operating_point"]
    print("\n=== promoted TEST read ===")
    print(f"  diagnosis: acc={td['accuracy']} bal_acc={td['balanced_accuracy']} "
          f"macroF1={td['macro_f1']} MEL_AUROC={td['mel_auroc']}")
    print(f"  per-class recall (argmax): {td['per_class_recall']}")
    print(f"  SK/MISC recall @operating-point: SK={op['per_class_recall'].get('SK')} "
          f"MISC={op['per_class_recall'].get('MISC')}")
    print(f"  criteria: binary_concept_F1={tc['macro_binary_concept_f1']} "
          f"multiclass_macroF1={tc['macro_multiclass_f1']}")
    print(f"  vs current test: acc {td['accuracy']}-0.7392="
          f"{test_report['comparison']['deltas_test']['diagnosis_accuracy_vs_current']:+.4f}, "
          f"criteriaF1 {tc['macro_binary_concept_f1']}-0.6623="
          f"{test_report['comparison']['deltas_test']['criteria_binary_concept_f1_vs_current']:+.4f}")
    print(f"\ndone in {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
