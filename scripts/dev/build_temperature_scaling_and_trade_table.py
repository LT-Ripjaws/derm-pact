"""M6 (notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md): two report-only deliverables.

A. Temperature scaling of the diagnosis head, fit on VALID, evaluated on TEST.
   The sealed promoted ECE (0.0891, paper_metrics_summary.json -> perception_derm7pt.calibration
   == promoted_two_view_ablation.json -> calibration) is the RAW posterior mass on the
   OPERATING-POINT label, while the sealed base ECE (0.0455, perception_baseline.json /
   test_metrics_complete.json) is TEMPERATURE-CALIBRATED with T fit on out-of-fold TRAIN logits.
   The existing artifact's own honesty note says the two are not comparable. This script makes
   them comparable by fitting a temperature on the SAME split (VALID) for BOTH heads and
   reporting NLL/Brier/ECE/AURC/reliability before vs after, for both heads, on TEST -- nothing
   about the deployed system changes; this is a diagnostic, not a redeployment.

B. Promotion trade table: every promoted-vs-base number in paper_metrics_summary.json (plus the
   M1 ROC-diagnostic finding and the calibration numbers from part A) in one table, so gains
   (MEL AUROC, minority recall, criteria F1, RAGAS, ...) and costs (per-criterion accuracy,
   union specificity, management QWK, retrieval-avoided, conflict rate, ...) sit side by side.

Boundaries: zero paid calls, zero images encoded (linear probes over cached frozen features).
TEST is read for evaluation only -- nothing is fit or selected on it. Does not modify anything
under medguard_v2/, any existing file under research/evidence/, or any existing script. The
diagnosis heads and their C/class_weight are refit here EXACTLY as
scripts/dev/derm7pt_p2_safety_promote.py specifies (imported bit-identically by file path, the
same pattern scripts/dev/build_annotation_learning_curve.py uses) and are verified against the
frozen references before anything downstream is trusted.

Outputs:
  * research/evidence/derm7pt_v1/temperature_scaling_valid.json
  * research/evidence/derm7pt_v1/promotion_trade_table.json
  * notes/TEMPERATURE_SCALING_VALID.md
  * notes/PROMOTION_TRADE_TABLE.md

Run::
    venv_win/Scripts/python.exe scripts/dev/build_temperature_scaling_and_trade_table.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.eval.metrics import aurc, ece, reliability_diagram  # noqa: E402
from medguard_v2.eval.statistics import binary_auc  # noqa: E402
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    PROMOTED_OPERATING_POINT,
    apply_operating_point,
)

# ---- import derm7pt_p2_safety_promote.py's reviewed machinery by file path, so Features/_fit/SEED
# are bit-identical to what produced the frozen promoted/base references (same pattern as
# build_annotation_learning_curve.py / build_correction_value_curves_promoted.py) -------------------
_PROMOTE_PATH = REPO / "scripts" / "dev" / "derm7pt_p2_safety_promote.py"
_spec = importlib.util.spec_from_file_location("_p2_safety_promote", _PROMOTE_PATH)
promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(promote)

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
OUT_TS_JSON = EVIDENCE / "temperature_scaling_valid.json"
OUT_TRADE_JSON = EVIDENCE / "promotion_trade_table.json"
OUT_TS_MD = REPO / "notes" / "TEMPERATURE_SCALING_VALID.md"
OUT_TRADE_MD = REPO / "notes" / "PROMOTION_TRADE_TABLE.md"

SEED = promote.SEED  # 20260810
OP = dict(PROMOTED_OPERATING_POINT)  # {"SK": 0.5, "MISC": 0.7}
N_BINS = 10
ANALYSIS_TYPE = "post-hoc, report-only; T fit on VALID only; deployed system unchanged"

# ---- frozen references reproduced here as wiring self-checks (STOP if any fails) ------------------
REF_PROMOTED_TEST_ACC = 0.762
REF_PROMOTED_TEST_MEL_AUROC = 0.9151
REF_BASE_TEST_ACC = 0.7392
REF_BASE_TEST_MEL_AUROC = 0.895
REF_PROMOTED_RAW_OPPOINT_ECE = 0.0891   # paper_metrics_summary.json / promoted_two_view_ablation.json
REF_BASE_SEALED_TS_ECE = 0.0455         # perception_baseline.json (T fit on OOF-TRAIN; side-by-side only)
REF_M1_CLASSIFIER_SENS_AT_UNION_SPEC = 0.9307  # roc_curve_data.json (peer-review finding M1)
WIRING_TOL = 0.0006


# =====================================================================================================
# generic numeric helpers
# =====================================================================================================

def _round4(x) -> float:
    return round(float(x), 4)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(proba: np.ndarray, y_index: np.ndarray) -> float:
    picked = proba[np.arange(len(y_index)), y_index]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def _brier(proba: np.ndarray, y_index: np.ndarray) -> float:
    """Multiclass Brier score: mean_i sum_k (p_ik - onehot_ik)^2. Range [0, 2]; lower is better."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_index)), y_index] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _class_index(classes: Sequence[str], values: Sequence[str]) -> np.ndarray:
    order = {c: i for i, c in enumerate(classes)}
    missing = sorted(set(values) - set(order))
    if missing:
        raise ValueError(f"labels outside the fitted class vocabulary: {missing}")
    return np.array([order[v] for v in values], dtype=int)


def _fit_temperature(logits: np.ndarray, y_index: np.ndarray) -> Tuple[float, dict]:
    """Scalar T > 0 minimising multiclass NLL of softmax(logits / T).

    Bounded scalar minimisation over log T (deterministic, no starting point). Bounds mirror
    medguard_v2/perception/probe.py's own temperature search; hitting a bound would itself be a
    signal that something is wrong (raised, not silently accepted), exactly as that module does.
    """
    lo, hi = np.log(0.05), np.log(20.0)

    def _objective(log_t: float) -> float:
        return _nll(_softmax(logits / float(np.exp(log_t))), y_index)

    result = minimize_scalar(_objective, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-8})
    t = float(np.exp(float(result.x)))
    if not np.isfinite(t) or t <= 0:
        raise RuntimeError(f"temperature search returned {t!r}")
    if t <= 0.051 or t >= 19.9:
        raise RuntimeError(
            f"temperature {t:.4f} hit the search bounds [0.05, 20]; the logits are not on the "
            "scale this search assumes"
        )
    return t, {"nll_at_optimum": round(float(result.fun), 6), "hit_search_bound": False}


# =====================================================================================================
# Part A: temperature scaling
# =====================================================================================================

def _fit_heads():
    """Fit promoted + base diagnosis heads EXACTLY per the reviewer's wiring spec (M6 step 1)."""
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    test = [c for c in cases if c.split == "test"]
    F = promote.Features()
    promoted_model = promote._fit(F.dermlip_meta_mat, train, C=100.0, class_weight=None)
    base_model = promote._fit(F.dermlip_mat, train, C=10.0, class_weight=None)
    return F, train, valid, test, promoted_model, base_model


def _wiring_self_check_ab(F, test, promoted_model, base_model) -> Dict[str, dict]:
    """STOP if freshly-fit heads do not reproduce the frozen promoted/base TEST references."""
    y_test = [c.diagnosis_group for c in test]
    truth_mel = [g == "MEL" for g in y_test]

    def _acc_auroc(model, feat_fn) -> Tuple[float, float]:
        x = feat_fn(test)
        proba = model.predict_proba(x)
        classes = [str(c) for c in model.classes_]
        pred = [classes[i] for i in proba.argmax(axis=1)]
        acc = sum(1 for a, b in zip(pred, y_test) if a == b) / len(y_test)
        mel_idx = classes.index("MEL")
        auroc = float(binary_auc(truth_mel, proba[:, mel_idx].tolist()))
        return acc, auroc

    acc_p, auroc_p = _acc_auroc(promoted_model, F.dermlip_meta_mat)
    acc_b, auroc_b = _acc_auroc(base_model, F.dermlip_mat)

    checks = {
        "promoted_test_accuracy_argmax": {
            "got": _round4(acc_p), "expected": REF_PROMOTED_TEST_ACC,
            "source": "paper_metrics_summary.json perception_derm7pt.classifier_5class.argmax.accuracy",
        },
        "promoted_test_mel_auroc": {
            "got": _round4(auroc_p), "expected": REF_PROMOTED_TEST_MEL_AUROC,
            "source": "paper_metrics_summary.json perception_derm7pt.melanoma_auroc.point",
        },
        "base_test_accuracy_argmax": {
            "got": _round4(acc_b), "expected": REF_BASE_TEST_ACC,
            "source": "perception_baseline.json metrics.diagnosis.test.accuracy",
        },
        "base_test_mel_auroc": {
            "got": _round4(auroc_b), "expected": REF_BASE_TEST_MEL_AUROC,
            "source": "paper_metrics_summary.json perception_derm7pt.melanoma_auroc.base_arm_reference.point",
        },
    }
    failures = []
    for name, ch in checks.items():
        ch["reproduced"] = bool(abs(ch["got"] - ch["expected"]) <= WIRING_TOL)
        if not ch["reproduced"]:
            failures.append(name)
    if failures:
        raise SystemExit(
            "WIRING DRIFT (M6 part A): freshly-fit heads do not reproduce the frozen promoted/base "
            f"TEST references: {failures}. Refusing to run temperature scaling on an untrustworthy "
            "fit.\n" + json.dumps({k: checks[k] for k in failures}, indent=2)
        )
    return checks


def _head_block(model, feat_fn, valid, test, *, is_promoted: bool) -> dict:
    classes = [str(c) for c in model.classes_]
    x_valid = feat_fn(valid)
    x_test = feat_fn(test)
    y_valid = [c.diagnosis_group for c in valid]
    y_test = [c.diagnosis_group for c in test]
    y_valid_idx = _class_index(classes, y_valid)
    y_test_idx = _class_index(classes, y_test)

    logits_valid = np.asarray(model.decision_function(x_valid), dtype=np.float64)
    logits_test = np.asarray(model.decision_function(x_test), dtype=np.float64)

    # For multinomial LR, predict_proba == softmax(decision_function); verify rather than assume.
    proba_test_before = model.predict_proba(x_test)  # T = 1: the actual deployed raw posterior
    max_abs_diff = float(np.max(np.abs(proba_test_before - _softmax(logits_test))))
    if max_abs_diff >= 1e-6:
        raise SystemExit(
            f"softmax(decision_function(X)) does not match predict_proba(X) (max abs diff "
            f"{max_abs_diff:.3e} >= 1e-6): the multinomial-softmax identity this analysis depends "
            "on does not hold for this model/sklearn version."
        )

    # ---- fit T on VALID only, apply to TEST -------------------------------------------------------
    T_valid, t_diag = _fit_temperature(logits_valid, y_valid_idx)
    proba_test_after = _softmax(logits_test / T_valid)

    pred_idx_before = proba_test_before.argmax(axis=1)
    pred_idx_after = proba_test_after.argmax(axis=1)
    argmax_invariant = bool(np.array_equal(pred_idx_before, pred_idx_after))
    n_mismatches = int(np.sum(pred_idx_before != pred_idx_after))
    if not argmax_invariant:
        raise SystemExit(
            f"temperature scaling changed {n_mismatches} argmax predictions on TEST -- dividing "
            "logits by a positive scalar T must preserve every argmax; this indicates a bug."
        )
    pred_labels = [classes[i] for i in pred_idx_before]  # identical before/after by the check above
    correct_argmax = [p == t for p, t in zip(pred_labels, y_test)]

    nll_before = _nll(proba_test_before, y_test_idx)
    nll_after = _nll(proba_test_after, y_test_idx)
    brier_before = _brier(proba_test_before, y_test_idx)
    brier_after = _brier(proba_test_after, y_test_idx)

    conf_argmax_before = proba_test_before.max(axis=1).tolist()
    conf_argmax_after = proba_test_after.max(axis=1).tolist()
    ece_argmax_before = ece(conf_argmax_before, correct_argmax, n_bins=N_BINS)
    ece_argmax_after = ece(conf_argmax_after, correct_argmax, n_bins=N_BINS)
    aurc_argmax_before = aurc(conf_argmax_before, correct_argmax)
    aurc_argmax_after = aurc(conf_argmax_after, correct_argmax)
    reliab_argmax_before = reliability_diagram(conf_argmax_before, correct_argmax, n_bins=N_BINS)
    reliab_argmax_after = reliability_diagram(conf_argmax_after, correct_argmax, n_bins=N_BINS)

    mel_idx = classes.index("MEL")
    truth_mel = [t == "MEL" for t in y_test]
    mel_auroc_before = float(binary_auc(truth_mel, proba_test_before[:, mel_idx].tolist()))
    mel_auroc_after = float(binary_auc(truth_mel, proba_test_after[:, mel_idx].tolist()))

    block: dict = {
        "features": "dermlip+metadata14 (526-d)" if is_promoted else "dermlip (512-d)",
        "fit_hyperparameters": {"C": 100.0 if is_promoted else 10.0, "class_weight": None, "seed": SEED},
        "classes_order": classes,
        "n_valid": len(valid),
        "n_test": len(test),
        "decision_function_softmax_matches_predict_proba": {
            "max_abs_diff": max_abs_diff, "assert_threshold": 1e-6, "passed": True,
        },
        "temperature": {"T_fit_on_valid": round(T_valid, 6), "fit_diagnostics": t_diag},
        "argmax_invariance_under_T": {
            "identical_predictions": argmax_invariant, "n_mismatches": n_mismatches, "n_test": len(test),
            "why_guaranteed": "dividing every class logit in a row by the same T > 0 preserves the "
                              "row's logit order, and softmax is order-preserving in its inputs, so "
                              "argmax(softmax(z/T)) == argmax(z) for any T > 0.",
        },
        "nll_test": {"before": round(nll_before, 6), "after": round(nll_after, 6),
                     "delta_after_minus_before": round(nll_after - nll_before, 6)},
        "brier_test": {"before": round(brier_before, 6), "after": round(brier_after, 6),
                       "delta_after_minus_before": round(brier_after - brier_before, 6)},
        "ece_argmax_confidence": {"before": ece_argmax_before, "after": ece_argmax_after},
        "aurc_argmax_confidence": {"before": aurc_argmax_before, "after": aurc_argmax_after},
        "reliability_argmax_confidence": {"before": reliab_argmax_before, "after": reliab_argmax_after},
        "mel_auroc": {
            "before": _round4(mel_auroc_before), "after": _round4(mel_auroc_after),
            "delta_after_minus_before": _round4(mel_auroc_after - mel_auroc_before),
            "note": "MEL AUROC is NOT guaranteed invariant to T the way argmax is: it ranks the MEL "
                    "posterior column ACROSS cases, and each case's softmax renormalisation is a "
                    "case-specific nonlinear function of the full logit row, so the cross-case order "
                    "of MEL mass can move slightly under T even though no within-case argmax can.",
        },
    }

    if is_promoted:
        pred_op_before, conf_op_before = apply_operating_point(proba_test_before, model.classes_, OP)
        pred_op_after, conf_op_after = apply_operating_point(proba_test_after, model.classes_, OP)
        correct_op_before = [p == t for p, t in zip(pred_op_before, y_test)]
        correct_op_after = [p == t for p, t in zip(pred_op_after, y_test)]
        ece_op_before = ece(conf_op_before, correct_op_before, n_bins=N_BINS)
        ece_op_after = ece(conf_op_after, correct_op_after, n_bins=N_BINS)
        aurc_op_before = aurc(conf_op_before, correct_op_before)
        aurc_op_after = aurc(conf_op_after, correct_op_after)
        reliab_op_before = reliability_diagram(conf_op_before, correct_op_before, n_bins=N_BINS)
        reliab_op_after = reliability_diagram(conf_op_after, correct_op_after, n_bins=N_BINS)
        n_op_changed = int(sum(1 for a, b in zip(pred_op_before, pred_op_after) if a != b))

        sealed_match = abs(ece_op_before - REF_PROMOTED_RAW_OPPOINT_ECE) <= 0.001
        if not sealed_match:
            raise SystemExit(
                f"WIRING DRIFT (M6 part A): recomputed raw operating-point-label ECE {ece_op_before} "
                f"does not reproduce the sealed {REF_PROMOTED_RAW_OPPOINT_ECE} within 0.001 -- "
                "refusing to trust the calibrated numbers built on top of this confidence definition."
            )

        block["operating_point_label_confidence"] = {
            "policy": {"rule": "argmax(posterior / weight)", "weights": OP},
            "confidence_definition": "raw (un-reweighted) posterior mass on the label chosen after "
                                      "reweighting -- exactly what apply_operating_point() returns "
                                      "and what run_cohort(promoted=True) writes into contracts",
            "ece": {"before": ece_op_before, "after": ece_op_after},
            "aurc": {"before": aurc_op_before, "after": aurc_op_after},
            "reliability": {"before": reliab_op_before, "after": reliab_op_after},
            "n_predictions_changed_by_T": n_op_changed,
            "sealed_reproduction_check": {
                "got": ece_op_before, "expected": REF_PROMOTED_RAW_OPPOINT_ECE, "tolerance": 0.001,
                "reproduced": bool(sealed_match),
                "source": "paper_metrics_summary.json perception_derm7pt.calibration.ece_uncalibrated "
                          "== promoted_two_view_ablation.json calibration.ece_uncalibrated",
            },
        }

    # ---- oracle T fit directly on TEST (leakage; best-case lower bound, NOT usable) ----------------
    T_oracle, oracle_diag = _fit_temperature(logits_test, y_test_idx)
    proba_test_oracle = _softmax(logits_test / T_oracle)
    nll_oracle = _nll(proba_test_oracle, y_test_idx)
    conf_argmax_oracle = proba_test_oracle.max(axis=1).tolist()
    ece_argmax_oracle = ece(conf_argmax_oracle, correct_argmax, n_bins=N_BINS)  # argmax is T-invariant
    block["oracle_T_on_test (not usable)"] = {
        "warning": "T fit directly on TEST logits AND TEST labels: this is test-set leakage and is "
                   "NOT a deployable calibration. Reported only as a best-case lower bound, to show "
                   "how much further NLL/ECE could fall if T could see TEST, i.e. how much the "
                   "honest VALID-fit T above is leaving on the table.",
        "temperature": round(T_oracle, 6),
        "fit_diagnostics": oracle_diag,
        "nll_test_after_oracle_T": round(nll_oracle, 6),
        "ece_argmax_after_oracle_T": ece_argmax_oracle,
    }
    return block


def build_temperature_scaling() -> dict:
    F, train, valid, test, promoted_model, base_model = _fit_heads()
    wiring = _wiring_self_check_ab(F, test, promoted_model, base_model)

    promoted_block = _head_block(promoted_model, F.dermlip_meta_mat, valid, test, is_promoted=True)
    base_block = _head_block(base_model, F.dermlip_mat, valid, test, is_promoted=False)

    base_ts_after = base_block["ece_argmax_confidence"]["after"]
    base_raw_before = base_block["ece_argmax_confidence"]["before"]
    base_sealed_match = abs(base_ts_after - REF_BASE_SEALED_TS_ECE) <= 0.001

    report = {
        "protocol": {
            "analysis_type": ANALYSIS_TYPE,
            "artifact": "derm7pt-temperature-scaling-valid-v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_finding": "notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md #M6",
            "seed": SEED,
            "n_bins": N_BINS,
            "splits": {"train": len(train), "valid": len(valid), "test": len(test)},
            "fitting_source": "scripts/dev/derm7pt_p2_safety_promote.py Features/_fit/SEED, imported "
                              "bit-identically by file path (same pattern as "
                              "build_annotation_learning_curve.py); promoted=_fit(F.dermlip_meta_mat, "
                              "train, C=100.0, class_weight=None), base=_fit(F.dermlip_mat, train, "
                              "C=10.0, class_weight=None)",
            "wiring_self_checks": wiring,
        },
        "promoted_head": promoted_block,
        "base_head": base_block,
        "cross_head_notes": {
            "base_ts_calibrated_ece_vs_sealed": {
                "recomputed_ece_argmax_after_valid_fit_T": base_ts_after,
                "sealed_ece_calibrated": REF_BASE_SEALED_TS_ECE,
                "sealed_source": "perception_baseline.json probe.targets.diagnosis.temperature (T fit "
                                 "on 5-fold OUT-OF-FOLD TRAIN logits) / test_metrics_complete.json "
                                 "calibration.ece",
                "match_within_0_001": bool(base_sealed_match),
                "note": (
                    "NOT forced to agree, per the task's own instruction -- reported side by side "
                    "instead. The sealed 0.0455 used a DIFFERENT temperature, fit on 5-fold "
                    "out-of-fold TRAIN logits (never touching valid or test); this artefact fits T "
                    "on the official VALID split instead, as this analysis was asked to. Both are "
                    "legitimate temperature-scaling procedures fit on disjoint data from TEST; they "
                    "are not expected to produce a bit-identical T or ECE, and the difference "
                    "reported here is exactly that -- a difference, not an error."
                ),
            },
            "base_raw_ece_sanity_check": {
                "recomputed_ece_argmax_before_T_uncalibrated": base_raw_before,
                "perception_baseline_json_ece_uncalibrated": 0.0393,
                "note": "Both use the SAME model spec (C=10, unbalanced, seed 20260810, argmax "
                        "confidence, T=1/no calibration) fit on the same train split, so this is "
                        "expected to closely corroborate perception_baseline.json even though the "
                        "feature-loading path differs (promote.Features().dermlip npz vs "
                        "perception/encode.py's FeatureBundle) -- included as a sanity cross-check, "
                        "not a hard assertion.",
            },
        },
    }
    return report


# =====================================================================================================
# Part B: promotion trade table
# =====================================================================================================

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(metric: str, section: str, dataset: str, base_arm, promoted_arm, direction: str,
         source_path: str, *, note: Optional[str] = None, bonus: bool = False,
         delta: Optional[float] = "auto") -> dict:
    if delta == "auto":
        if base_arm is None or promoted_arm is None:
            delta = None
        else:
            try:
                delta = round(float(promoted_arm) - float(base_arm), 4)
            except (TypeError, ValueError):
                delta = None
    row = {
        "metric": metric, "section": section, "dataset": dataset,
        "base_arm": base_arm, "promoted_arm": promoted_arm, "delta": delta,
        "direction": direction, "source_path": source_path, "bonus": bonus,
    }
    if note:
        row["note"] = note
    return row


def _lower_is_better_direction(promoted_val: float, base_val: float) -> str:
    if promoted_val < base_val:
        return "gain"
    if promoted_val > base_val:
        return "cost"
    return "neutral"


def build_trade_table(ts_report: dict) -> dict:
    summary = _load_json(EVIDENCE / "paper_metrics_summary.json")
    safety = _load_json(EVIDENCE / "perception_phase2_safety_preserving.json")
    roc = _load_json(EVIDENCE / "roc_curve_data.json")
    ablation = _load_json(EVIDENCE / "promoted_two_view_ablation.json")
    baseline = _load_json(EVIDENCE / "perception_baseline.json")

    pd_ = summary["perception_derm7pt"]
    mgmt = summary["management"]
    routing = summary["contracts_and_routing"]
    ragas = summary["ragas"]
    bcn = summary["perception_bcn"]
    ham = summary["perception_ham"]
    final = safety["final_test_metrics"]
    base_per_class = baseline["metrics"]["diagnosis"]["test"]["per_class"]

    # ---- Part-B wiring self-checks (STOP if any fails) --------------------------------------------
    points = roc["promoted_classifier_roc"]["points"]
    union_spec = pd_["operating_point_bootstrap"]["union"]["specificity"]
    m1_recomputed = round(max(p["sensitivity_tpr"] for p in points if p["specificity"] >= union_spec), 4)
    m1_ok = abs(m1_recomputed - REF_M1_CLASSIFIER_SENS_AT_UNION_SPEC) <= WIRING_TOL

    sk_agree = abs(pd_["classifier_5class"]["operating_point"]["per_class"]["SK"]["recall"]
                   - final["minority_recall_operating_point"]["SK"]) <= 1e-9
    misc_agree = abs(pd_["classifier_5class"]["operating_point"]["per_class"]["MISC"]["recall"]
                     - final["minority_recall_operating_point"]["MISC"]) <= 1e-9
    ece_agree = abs(pd_["calibration"]["ece_uncalibrated"] - ablation["calibration"]["ece_uncalibrated"]) <= 1e-9

    wiring = {
        "m1_classifier_alone_sensitivity_at_union_specificity": {
            "got": m1_recomputed, "expected": REF_M1_CLASSIFIER_SENS_AT_UNION_SPEC,
            "tolerance": WIRING_TOL, "reproduced": bool(m1_ok),
            "source": "roc_curve_data.json promoted_classifier_roc.points, max sensitivity_tpr s.t. "
                      "specificity >= perception_derm7pt.operating_point_bootstrap.union.specificity",
        },
        "cross_file_sk_misc_recall_agreement": {
            "sk_reproduced": bool(sk_agree), "misc_reproduced": bool(misc_agree),
            "source": "paper_metrics_summary.json perception_derm7pt vs "
                      "perception_phase2_safety_preserving.json final_test_metrics",
        },
        "cross_file_promoted_raw_ece_agreement": {
            "reproduced": bool(ece_agree),
            "source": "paper_metrics_summary.json perception_derm7pt.calibration vs "
                      "promoted_two_view_ablation.json calibration",
        },
    }
    if not (m1_ok and sk_agree and misc_agree and ece_agree):
        raise SystemExit(
            "WIRING DRIFT (M6 part B): trade-table source files disagree with each other or with the "
            f"reviewer's M1 finding -- refusing to build the table.\n{json.dumps(wiring, indent=2)}"
        )

    rows: List[dict] = []
    ds_test = "derm7pt_test_n395"
    ds_valid = "derm7pt_valid_n203"

    # ============================================================= diagnosis (5-class + melanoma) ===
    mel = pd_["melanoma_auroc"]
    rows.append(_row(
        "MEL AUROC", "diagnosis", ds_test, mel["base_arm_reference"]["point"], mel["point"], "gain",
        "paper_metrics_summary.json perception_derm7pt.melanoma_auroc",
        note=f"CI95 base [{mel['base_arm_reference']['ci95_low']}, {mel['base_arm_reference']['ci95_high']}], "
             f"promoted [{mel['ci95_low']}, {mel['ci95_high']}] -- by construction promoted >= base "
             "(the safety-preserving selection rule required it)."))
    cls = pd_["classifier_5class"]
    rows.append(_row("5-class accuracy (argmax)", "diagnosis", ds_test,
                      cls["base_arm_reference"]["accuracy"], cls["argmax"]["accuracy"], "gain",
                      "paper_metrics_summary.json perception_derm7pt.classifier_5class"))
    rows.append(_row("5-class balanced accuracy (argmax)", "diagnosis", ds_test,
                      cls["base_arm_reference"]["balanced_accuracy"], cls["argmax"]["balanced_accuracy"],
                      "gain", "paper_metrics_summary.json perception_derm7pt.classifier_5class"))
    rows.append(_row("5-class macro-F1 (argmax)", "diagnosis", ds_test,
                      cls["base_arm_reference"]["macro_f1"], cls["argmax"]["macro_f1"], "gain",
                      "paper_metrics_summary.json perception_derm7pt.classifier_5class"))
    rows.append(_row(
        "SK recall", "diagnosis", ds_test,
        base_per_class["SK"]["recall"], cls["operating_point"]["per_class"]["SK"]["recall"], "gain",
        "perception_baseline.json metrics.diagnosis.test.per_class.SK (base) vs "
        "paper_metrics_summary.json classifier_5class.operating_point.per_class.SK (promoted)",
        note="Promoted value uses the SK/MISC operating point (argmax posterior/weight); the base "
             "arm has no operating point, so its value is plain argmax -- this is each arm's own "
             "actually-deployed recall, not a mismatched comparison."))
    rows.append(_row(
        "MISC recall", "diagnosis", ds_test,
        base_per_class["MISC"]["recall"], cls["operating_point"]["per_class"]["MISC"]["recall"], "gain",
        "perception_baseline.json metrics.diagnosis.test.per_class.MISC (base) vs "
        "paper_metrics_summary.json classifier_5class.operating_point.per_class.MISC (promoted)",
        note="Same operating-point caveat as SK recall above."))
    rows.append(_row(
        "MEL recall (argmax)", "diagnosis", ds_test,
        base_per_class["MEL"]["recall"], cls["argmax"]["per_class"]["MEL"]["recall"], "gain",
        "perception_baseline.json metrics.diagnosis.test.per_class.MEL (base) vs "
        "paper_metrics_summary.json classifier_5class.argmax.per_class.MEL (promoted)",
        note="Identical at argmax and at the operating point: the SK/MISC reweighting never touches "
             "the MEL posterior column (AUROC-invariance)."))

    # ============================================================================== criteria ========
    ca = pd_["criteria_aggregate"]
    rows.append(_row("Criteria binary-concept F1 (macro)", "criteria", ds_test,
                      ca["base_arm_reference"]["binary_concept_f1_macro"], ca["binary_concept_f1_macro"],
                      "gain", "paper_metrics_summary.json perception_derm7pt.criteria_aggregate"))
    rows.append(_row(
        "Per-criterion accuracy (macro across 7 criteria)", "criteria", ds_test,
        ca["base_arm_reference"]["accuracy_macro_across_criteria"], ca["accuracy_macro_across_criteria"],
        "cost", "paper_metrics_summary.json perception_derm7pt.criteria_aggregate",
        note="The balanced heads trade raw per-criterion accuracy for recall on rare criterion "
             "values; binary-concept F1 (row above) is the metric the promotion was selected on."))

    # ==================================================================== union melanoma detection ==
    union = pd_["operating_point_bootstrap"]["union"]
    union_base = summary["ablation_perception"]["base_arm_reference"]["union_rule"]
    rows.append(_row("Union melanoma sensitivity (classifier-argmax OR checklist-positive)",
                      "union", ds_test, union_base["sensitivity"], union["sensitivity"], "gain",
                      "paper_metrics_summary.json perception_derm7pt.operating_point_bootstrap.union "
                      "vs ablation_perception.base_arm_reference.union_rule",
                      note="See M1 row below: this gain is relative to the classifier's ARGMAX "
                           "operating point, not to what the classifier's own ROC can already reach."))
    rows.append(_row("Union melanoma specificity", "union", ds_test,
                      union_base["specificity"], union["specificity"], "cost",
                      "paper_metrics_summary.json perception_derm7pt.operating_point_bootstrap.union "
                      "vs ablation_perception.base_arm_reference.union_rule",
                      note="Disclosed cost: the balanced criteria heads flag melanoma suspicion more "
                           "often, raising false positives together with true positives."))
    union_sens_gain_over_base = round(union["sensitivity"] - union_base["sensitivity"], 4)
    rows.append(_row(
        "Classifier-alone sensitivity at the union's own specificity (M1)", "union", ds_test,
        None, m1_recomputed, "context",
        "roc_curve_data.json promoted_classifier_roc.points (max sensitivity_tpr at specificity "
        f">= {union_spec}); reviewer finding M1",
        delta=round(m1_recomputed - union["sensitivity"], 4),
        note=(
            "Not a base-vs-promoted comparison (base_arm left null on purpose). At the SAME "
            f"specificity the union rule operates at ({union_spec}), the promoted classifier ALONE "
            f"-- just by lowering its threshold -- reaches sensitivity {m1_recomputed}, ABOVE the "
            f"union rule's own {union['sensitivity']} (delta shown here is vs the union row, not vs "
            "a base arm). So the union's "
            f"+{union_sens_gain_over_base:.4f} sensitivity gain over the base arm (row above) is a "
            "gain relative to the classifier's ARGMAX operating point only, not relative to what the "
            "classifier's own ROC can already achieve by moving its threshold. The criteria path's "
            "honest value is evidential/complementary (criterion-level audit trail, human-adjudicated "
            "disagreement -- see the Venn/complementarity numbers elsewhere), not an incremental "
            "melanoma-detection gain beyond the classifier's own ROC."
        )))

    # ============================================================================= management =======
    rows.append(_row("Management accuracy", "management", ds_test,
                      mgmt["base_arm_reference"]["accuracy"], mgmt["accuracy"], "gain",
                      "paper_metrics_summary.json management"))
    rows.append(_row(
        "Management QWK (ordinal)", "management", ds_test,
        mgmt["base_arm_reference"]["qwk"], mgmt["qwk"], "cost",
        "paper_metrics_summary.json management",
        note=f"Promoted CI95 {mgmt['qwk_ci']}, base CI95 {mgmt['base_arm_reference']['qwk_ci']} -- "
             "the promoted CI covers the base point estimate, so the dip is not resolvable at n=395 "
             "(dual-reported in the source artifact, not hidden)."))

    # ============================================================================= calibration ======
    ece_a = ts_report["promoted_head"]["ece_argmax_confidence"]["after"]
    ece_b = ts_report["base_head"]["ece_argmax_confidence"]["after"]
    ece_apples_direction = _lower_is_better_direction(ece_a, ece_b)
    rows.append(_row(
        "ECE -- raw, operating-point-label confidence (as originally reported)", "calibration", ds_test,
        None, pd_["calibration"]["ece_uncalibrated"], "context",
        "paper_metrics_summary.json perception_derm7pt.calibration.ece_uncalibrated",
        note="The M6 headline as originally stated (0.0891 promoted). NOT an apples-to-apples base "
             "number: no base-arm equivalent exists because the base arm has no operating point. See "
             "the two rows below for (a) the base arm's own originally-reported sealed number and "
             "(b) the corrected, apples-to-apples pair this analysis produces."))
    rows.append(_row(
        "ECE -- base arm, sealed (T fit on out-of-fold TRAIN, argmax confidence)", "calibration",
        ds_test, pd_["calibration"]["base_arm_reference"]["ece_calibrated"], None, "context",
        "paper_metrics_summary.json perception_derm7pt.calibration.base_arm_reference.ece_calibrated",
        note="The base-arm number the row above was originally paired with in the M6 finding -- "
             "different confidence definition (argmax, not operating-point label) AND different "
             "calibration data (out-of-fold TRAIN, not VALID). Superseded for comparison purposes by "
             "the apples-to-apples row below."))
    rows.append(_row(
        "ECE -- temperature-scaled on VALID, argmax confidence (apples-to-apples)", "calibration",
        ds_test, ece_b, ece_a, ece_apples_direction,
        "research/evidence/derm7pt_v1/temperature_scaling_valid.json "
        "{promoted_head,base_head}.ece_argmax_confidence.after (this analysis, part A)",
        note="Both heads calibrated the SAME way for the first time: T fit on VALID, argmax "
             "confidence, evaluated on TEST. This is the fair comparison the M6 finding asked for."))

    # ============================================================================ contracts/routing =
    rows.append(_row(
        "Retrieval avoided", "contracts_routing", ds_valid,
        routing["base_arm_reference"]["router_and_abstention_valid_n203"]["retrieval_avoided"],
        routing["retrieval_avoided"], "cost",
        "paper_metrics_summary.json contracts_and_routing.retrieval_avoided vs "
        "contracts_and_routing.base_arm_reference.router_and_abstention_valid_n203"))
    rows.append(_row(
        "Conflict rate", "contracts_routing", ds_valid,
        routing["base_arm_reference"]["valid_n203"]["conflict_rate"], routing["conflict_rate"], "cost",
        "paper_metrics_summary.json contracts_and_routing.conflict_rate vs "
        "contracts_and_routing.base_arm_reference.valid_n203",
        note="Same underlying cause as the union-specificity cost above: the balanced criteria heads "
             "disagree with the classifier more often, which is also the mechanism the sensitivity "
             "gain runs through -- a safety/efficiency trade, not a pure regression."))

    # ========================================================================================= ragas
    v2c = ragas["v2_comparison"]
    rows.append(_row("RAGAS faithfulness -- overall", "ragas", "agent_answers",
                      v2c["v2_faithfulness"], v2c["current_faithfulness"], "gain",
                      "paper_metrics_summary.json ragas.v2_comparison"))
    rows.append(_row("RAGAS faithfulness -- guideline-grounded", "ragas", "agent_answers",
                      v2c["v2_guideline"], v2c["current_guideline"], "gain",
                      "paper_metrics_summary.json ragas.v2_comparison"))
    rows.append(_row("RAGAS faithfulness -- mixed-grounded", "ragas", "agent_answers",
                      v2c["v2_mixed"], v2c["current_mixed"], "gain",
                      "paper_metrics_summary.json ragas.v2_comparison",
                      note="Largest v2-vs-current gain of the three grounding scopes."))

    # =============================================================================== bonus rows =====
    aurc_a = pd_["calibration"]["aurc"]
    aurc_b_sealed = pd_["calibration"]["base_arm_reference"]["aurc"]
    rows.append(_row(
        "AURC -- sealed (paper headline; confidence definitions differ, see note)", "calibration",
        ds_test, aurc_b_sealed, aurc_a, _lower_is_better_direction(aurc_a, aurc_b_sealed),
        "paper_metrics_summary.json perception_derm7pt.calibration.aurc", bonus=True,
        note="Same confidence-definition mismatch as the raw ECE headline above (promoted = "
             "operating-point-label mass, base = TS-calibrated argmax mass)."))
    aurc_ts_a = ts_report["promoted_head"]["aurc_argmax_confidence"]["after"]
    aurc_ts_b = ts_report["base_head"]["aurc_argmax_confidence"]["after"]
    rows.append(_row(
        "AURC -- temperature-scaled on VALID, argmax confidence (apples-to-apples)", "calibration",
        ds_test, aurc_ts_b, aurc_ts_a, _lower_is_better_direction(aurc_ts_a, aurc_ts_b),
        "research/evidence/derm7pt_v1/temperature_scaling_valid.json "
        "{promoted_head,base_head}.aurc_argmax_confidence.after (this analysis, part A)", bonus=True))

    sps = pd_["seven_point_score"]
    rows.append(_row("7-point score MAE (lower is better)", "seven_point_score", ds_test,
                      sps["base_arm_reference"]["mae"], sps["mae"], "cost",
                      "paper_metrics_summary.json perception_derm7pt.seven_point_score", bonus=True,
                      note="The promoted balanced heads call scoring values more often (they are "
                           "tuned for melanoma sensitivity, not 7-point score agreement), so MAE "
                           "against the annotated gold score is slightly worse."))
    rows.append(_row("7-point score within-1 agreement rate", "seven_point_score", ds_test,
                      sps["base_arm_reference"]["within_one_rate"], sps["within_one_rate"], "cost",
                      "paper_metrics_summary.json perception_derm7pt.seven_point_score", bonus=True))
    rows.append(_row(
        "Citation rate (different splits -- context only)", "contracts_routing", "mixed_splits",
        routing["base_arm_reference"]["test_n395"]["citation_rate"], routing["citation_rate"], "context",
        "paper_metrics_summary.json contracts_and_routing.citation_rate (valid n=203) vs "
        "contracts_and_routing.base_arm_reference.test_n395.citation_rate (test n=395)", bonus=True,
        note="Promoted number is valid n=203, base number is test n=395 -- different splits, kept "
             "for completeness but not a clean promotion delta."))

    # ============================================================ generalization (BCN / HAM) bonus ==
    bcn_id = bcn["in_domain"]
    bcn_ph = bcn["promoted_recipe_post_hoc"]
    rows.append(_row("BCN in-domain accuracy (promoted recipe, post-hoc)", "generalization",
                      "bcn_in_domain_n712", bcn_id["accuracy"], bcn_ph["accuracy"], "neutral",
                      "paper_metrics_summary.json perception_bcn.{in_domain,promoted_recipe_post_hoc}",
                      bonus=True))
    rows.append(_row("BCN in-domain MEL AUROC (promoted recipe, post-hoc)", "generalization",
                      "bcn_in_domain_n712", bcn_id["mel_auroc"], bcn_ph["mel_auroc"], "gain",
                      "paper_metrics_summary.json perception_bcn.{in_domain,promoted_recipe_post_hoc}",
                      bonus=True))
    bcn_nt = bcn_ph["operating_point_non_transfer"]
    rows.append(_row(
        "BCN boundary: MEL recall if the Derm7pt SK/MISC op-point is naively applied", "generalization",
        "bcn_in_domain_n712", None, bcn_nt["mel_recall_at_op"], "cost",
        "paper_metrics_summary.json perception_bcn.promoted_recipe_post_hoc.operating_point_non_transfer",
        bonus=True, delta=round(bcn_nt["mel_recall_at_op"] - bcn_nt["mel_recall_argmax"], 4),
        note=f"Within-recipe boundary finding, not a base-vs-promoted row: argmax MEL recall is "
             f"{bcn_nt['mel_recall_argmax']}; applying the Derm7pt-tuned operating point collapses it "
             f"to {bcn_nt['mel_recall_at_op']} (delta shown is at-op minus argmax). The operating "
             "point is a Derm7pt-specific policy and is correctly NOT adopted on BCN."))

    ham_e3 = ham["e3_split"]["five_group"]
    ham_e3_ph = ham["promoted_recipe_post_hoc"]["e3_five_group"]
    rows.append(_row("HAM e3 accuracy (5-group, promoted recipe, post-hoc)", "generalization",
                      "ham_e3_n1595", ham_e3["accuracy"], ham_e3_ph["accuracy"], "gain",
                      "paper_metrics_summary.json perception_ham.{e3_split.five_group,"
                      "promoted_recipe_post_hoc.e3_five_group}", bonus=True))
    rows.append(_row("HAM e3 MEL AUROC (5-group, promoted recipe, post-hoc)", "generalization",
                      "ham_e3_n1595", ham_e3["mel_auroc"], ham_e3_ph["mel_auroc"], "gain",
                      "paper_metrics_summary.json perception_ham.{e3_split.five_group,"
                      "promoted_recipe_post_hoc.e3_five_group}", bonus=True))
    ham_e3_nt = ham_e3_ph["operating_point_non_transfer"]
    rows.append(_row(
        "HAM e3 boundary: MEL recall if the Derm7pt SK/MISC op-point is naively applied",
        "generalization", "ham_e3_n1595", None, ham_e3_nt["mel_recall_at_op"], "cost",
        "paper_metrics_summary.json perception_ham.promoted_recipe_post_hoc.e3_five_group."
        "operating_point_non_transfer", bonus=True,
        delta=round(ham_e3_nt["mel_recall_at_op"] - ham_e3_nt["mel_recall_argmax"], 4),
        note=f"argmax MEL recall {ham_e3_nt['mel_recall_argmax']} -> {ham_e3_nt['mel_recall_at_op']} "
             "at the (not-adopted) Derm7pt operating point. Same boundary pattern as BCN."))

    ham_sealed = ham["sealed_test"]
    ham_sealed_ph = ham["promoted_recipe_post_hoc"]["sealed_five_group"]
    rows.append(_row("HAM sealed accuracy (5-group, promoted recipe, post-hoc)", "generalization",
                      "ham_sealed_n1495", ham_sealed["accuracy_5group"], ham_sealed_ph["accuracy"],
                      "gain", "paper_metrics_summary.json perception_ham.{sealed_test,"
                      "promoted_recipe_post_hoc.sealed_five_group}", bonus=True))
    rows.append(_row(
        "HAM sealed MEL AUROC (5-group, promoted recipe, post-hoc)", "generalization",
        "ham_sealed_n1495", ham_sealed["mel_auroc"], ham_sealed_ph["mel_auroc"], "cost",
        "paper_metrics_summary.json perception_ham.{sealed_test,promoted_recipe_post_hoc."
        "sealed_five_group}", bonus=True,
        note="The ONLY dataset/split where the promoted recipe's MEL AUROC REGRESSES vs baseline "
             "(-0.0058) -- flagged, not hidden."))
    ham_sealed_nt = ham_sealed_ph["operating_point_non_transfer"]
    rows.append(_row(
        "HAM sealed boundary: MEL recall if the Derm7pt SK/MISC op-point is naively applied",
        "generalization", "ham_sealed_n1495", None, ham_sealed_nt["mel_recall_at_op"], "cost",
        "paper_metrics_summary.json perception_ham.promoted_recipe_post_hoc.sealed_five_group."
        "operating_point_non_transfer", bonus=True,
        delta=round(ham_sealed_nt["mel_recall_at_op"] - ham_sealed_nt["mel_recall_argmax"], 4),
        note=f"argmax MEL recall {ham_sealed_nt['mel_recall_argmax']} -> "
             f"{ham_sealed_nt['mel_recall_at_op']} at the (not-adopted) Derm7pt operating point."))

    n_gain = sum(1 for r in rows if r["direction"] == "gain")
    n_cost = sum(1 for r in rows if r["direction"] == "cost")
    n_neutral = sum(1 for r in rows if r["direction"] == "neutral")
    n_context = sum(1 for r in rows if r["direction"] == "context")

    report = {
        "protocol": {
            "analysis_type": ANALYSIS_TYPE,
            "artifact": "derm7pt-promotion-trade-table-v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_finding": "notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md #M6 (+ #M1 for the "
                                "classifier-alone-at-union-specificity row)",
            "seed": SEED,
            "wiring_self_checks": wiring,
            "direction_semantics": {
                "gain": "promoted arm is better than the base arm on this metric",
                "cost": "promoted arm is worse than the base arm on this metric",
                "neutral": "no meaningful difference between arms (delta ~ 0)",
                "context": "not a base-vs-promoted comparison; included for interpretive context "
                          "(e.g. the M1 ROC-diagnostic row, or a within-recipe boundary finding)",
            },
            "row_count": {"total": len(rows), "core": sum(1 for r in rows if not r["bonus"]),
                          "bonus": sum(1 for r in rows if r["bonus"]),
                          "gain": n_gain, "cost": n_cost, "neutral": n_neutral, "context": n_context},
            "sections_checked_but_excluded": {
                "agent_eval": "checked (paper_metrics_summary.json agent_eval.endpoints: "
                              "citation_resolvability, refusal_correctness, counterfactual_accuracy, "
                              "absence_disclosure, management_fabrication) -- none of its endpoints "
                              "carry a base_arm_reference, so no row could be built; not a silent "
                              "omission.",
            },
        },
        "rows": rows,
    }
    return report


# =====================================================================================================
# markdown rendering
# =====================================================================================================

def _fmt(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_ts_markdown(ts: dict) -> str:
    p, b = ts["promoted_head"], ts["base_head"]
    wiring = ts["protocol"]["wiring_self_checks"]
    lines: List[str] = []
    lines.append("# Temperature scaling of the diagnosis head (fit on VALID, evaluated on TEST)")
    lines.append("")
    lines.append(
        "Report-only, post-hoc diagnostic (M6, notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md). Nothing "
        "about the deployed system changes: `run_cohort(promoted=True)` still writes raw, uncalibrated "
        "posteriors into contracts. T is fit once per head on the official VALID split (never TEST) "
        "and applied to TEST for evaluation only."
    )
    lines.append("")
    lines.append(
        f"Splits: train {ts['protocol']['splits']['train']}, valid {ts['protocol']['splits']['valid']}, "
        f"test {ts['protocol']['splits']['test']}. Seed {ts['protocol']['seed']}. n_bins {ts['protocol']['n_bins']}."
    )
    lines.append("")
    lines.append("## Wiring self-checks (STOP if any fails -- all four passed before anything below was trusted)")
    lines.append("")
    lines.append("| check | got | expected | reproduced |")
    lines.append("|---|---:|---:|:---:|")
    for name, ch in wiring.items():
        lines.append(f"| {name} | {_fmt(ch['got'])} | {_fmt(ch['expected'])} | {ch['reproduced']} |")
    lines.append("")
    op = p["operating_point_label_confidence"]["sealed_reproduction_check"]
    lines.append(
        f"Additional self-check: raw operating-point-label ECE recomputed here = {op['got']}, sealed "
        f"= {op['expected']} (tolerance {op['tolerance']}) -- reproduced: {op['reproduced']}."
    )
    lines.append("")
    lines.append("## Temperature")
    lines.append("")
    lines.append("| head | T (fit on VALID) | T oracle (fit on TEST, NOT usable) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| promoted (dermlip+metadata14, C=100) | {p['temperature']['T_fit_on_valid']} | "
                  f"{p['oracle_T_on_test (not usable)']['temperature']} |")
    lines.append(f"| base (dermlip, C=10) | {b['temperature']['T_fit_on_valid']} | "
                  f"{b['oracle_T_on_test (not usable)']['temperature']} |")
    lines.append("")
    lines.append(
        "Argmax invariance under T: promoted "
        f"{p['argmax_invariance_under_T']['identical_predictions']} "
        f"({p['argmax_invariance_under_T']['n_mismatches']}/{p['argmax_invariance_under_T']['n_test']} "
        f"mismatches), base {b['argmax_invariance_under_T']['identical_predictions']} "
        f"({b['argmax_invariance_under_T']['n_mismatches']}/{b['argmax_invariance_under_T']['n_test']} "
        "mismatches) -- both exactly 0 by construction (dividing logits by T > 0 cannot change an argmax)."
    )
    lines.append("")
    op_changed = p["operating_point_label_confidence"]["n_predictions_changed_by_T"]
    lines.append(
        f"The SK/MISC OPERATING-POINT label is a different story: it is NOT provably T-invariant "
        f"(it argmaxes posterior/weight, not posterior alone, and the weight is nonlinear in T's "
        f"effect on the posterior). Measured, not assumed: {op_changed}/{p['n_test']} promoted "
        "operating-point predictions actually changed between the raw (T=1) and calibrated "
        "posteriors. `ece_op_before`/`ece_op_after` therefore use their OWN before/after correctness "
        "arrays (not the shared argmax one) -- this is handled correctly in the underlying "
        "computation, called out here so the before/after ECE numbers below are not misread as "
        f"differing only in confidence, when {op_changed} of the underlying decisions differ too."
    )
    lines.append("")
    lines.append("## NLL / Brier on TEST, before vs after")
    lines.append("")
    lines.append("| head | NLL before | NLL after | Brier before | Brier after |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, blk in (("promoted", p), ("base", b)):
        lines.append(f"| {label} | {blk['nll_test']['before']} | {blk['nll_test']['after']} | "
                      f"{blk['brier_test']['before']} | {blk['brier_test']['after']} |")
    lines.append("")
    lines.append("## ECE (n_bins=10), before vs after")
    lines.append("")
    lines.append("| head | confidence definition | ECE before | ECE after |")
    lines.append("|---|---|---:|---:|")
    lines.append(f"| promoted | argmax posterior mass | {p['ece_argmax_confidence']['before']} | "
                  f"{p['ece_argmax_confidence']['after']} |")
    lines.append(f"| promoted | operating-point-label mass | "
                  f"{p['operating_point_label_confidence']['ece']['before']} | "
                  f"{p['operating_point_label_confidence']['ece']['after']} |")
    lines.append(f"| base | argmax posterior mass | {b['ece_argmax_confidence']['before']} | "
                  f"{b['ece_argmax_confidence']['after']} |")
    lines.append("")
    lines.append("## AURC, before vs after")
    lines.append("")
    lines.append("| head | confidence definition | AURC before | AURC after |")
    lines.append("|---|---|---:|---:|")
    lines.append(f"| promoted | argmax posterior mass | {p['aurc_argmax_confidence']['before']} | "
                  f"{p['aurc_argmax_confidence']['after']} |")
    lines.append(f"| promoted | operating-point-label mass | "
                  f"{p['operating_point_label_confidence']['aurc']['before']} | "
                  f"{p['operating_point_label_confidence']['aurc']['after']} |")
    lines.append(f"| base | argmax posterior mass | {b['aurc_argmax_confidence']['before']} | "
                  f"{b['aurc_argmax_confidence']['after']} |")
    lines.append("")
    lines.append("## MEL AUROC drift under T")
    lines.append("")
    lines.append("| head | before | after | delta |")
    lines.append("|---|---:|---:|---:|")
    for label, blk in (("promoted", p), ("base", b)):
        m = blk["mel_auroc"]
        lines.append(f"| {label} | {m['before']} | {m['after']} | {m['delta_after_minus_before']} |")
    lines.append("")
    lines.append(
        "MEL AUROC is not guaranteed invariant to T (unlike argmax): it ranks the MEL posterior "
        "column across cases, and each case's softmax renormalisation is a case-specific nonlinear "
        "function of the full logit row."
    )
    lines.append("")
    lines.append("## Reliability diagram bins (10 bins: mean confidence, accuracy, count)")
    lines.append("")
    for label, blk, key in (("promoted (argmax)", p, "reliability_argmax_confidence"),
                            ("promoted (operating-point label)", p, None),
                            ("base (argmax)", b, "reliability_argmax_confidence")):
        src = blk[key] if key else p["operating_point_label_confidence"]["reliability"]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| bin | conf before | acc before | n before | conf after | acc after | n after |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for i, (bb, ba) in enumerate(zip(src["before"], src["after"])):
            lines.append(f"| {i+1} | {bb['confidence']} | {bb['accuracy']} | {bb['count']} | "
                          f"{ba['confidence']} | {ba['accuracy']} | {ba['count']} |")
        lines.append("")
    lines.append("## Base head: does TS-calibrated ECE reproduce the sealed 0.0455?")
    lines.append("")
    cn = ts["cross_head_notes"]["base_ts_calibrated_ece_vs_sealed"]
    lines.append(f"Recomputed (T fit on VALID, argmax confidence): **{cn['recomputed_ece_argmax_after_valid_fit_T']}**. "
                 f"Sealed (T fit on out-of-fold TRAIN, argmax confidence): **{cn['sealed_ece_calibrated']}**. "
                 f"Match within 0.001: **{cn['match_within_0_001']}**.")
    lines.append("")
    lines.append(cn["note"])
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    ece_a = p["ece_argmax_confidence"]["after"]
    ece_b = b["ece_argmax_confidence"]["after"]
    lines.append(
        f"Fitting T on VALID for BOTH heads (argmax confidence, TEST-evaluated) gives the "
        f"apples-to-apples pair the M6 finding was missing: promoted ECE {ece_a}, base ECE {ece_b} "
        f"(delta {round(ece_a - ece_b, 4)}). The originally-reported 0.0891-vs-0.0455 headline mixed "
        "two different confidence definitions (operating-point-label mass vs argmax mass) and two "
        "different calibration procedures (VALID vs out-of-fold TRAIN); it should be replaced by, or "
        "accompanied by, the pair above. Argmax accuracy is unchanged for both heads by construction; "
        f"MEL AUROC moved by {p['mel_auroc']['delta_after_minus_before']} (promoted) and "
        f"{b['mel_auroc']['delta_after_minus_before']} (base) under T -- negligible."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_trade_markdown(trade: dict) -> str:
    rows = trade["rows"]
    wiring = trade["protocol"]["wiring_self_checks"]
    counts = trade["protocol"]["row_count"]
    lines: List[str] = []
    lines.append("# Promotion trade table: gains vs costs, base arm vs promoted arm")
    lines.append("")
    lines.append(
        "Every promoted-vs-base number in `paper_metrics_summary.json` (plus the M1 ROC-diagnostic "
        "finding and the temperature-scaling numbers from part A) in one table, so the promotion's "
        "gains and costs sit side by side instead of being scattered across the paper "
        "(M6, notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md)."
    )
    lines.append("")
    lines.append(
        f"{counts['total']} rows ({counts['core']} core + {counts['bonus']} bonus/context): "
        f"{counts['gain']} gains, {counts['cost']} costs, {counts['neutral']} neutral, "
        f"{counts['context']} context (not a base-vs-promoted delta)."
    )
    lines.append("")
    lines.append("## Wiring self-checks")
    lines.append("")
    lines.append("| check | reproduced |")
    lines.append("|---|:---:|")
    lines.append(f"| M1 classifier-alone sensitivity at union specificity "
                 f"({wiring['m1_classifier_alone_sensitivity_at_union_specificity']['got']} vs expected "
                 f"{wiring['m1_classifier_alone_sensitivity_at_union_specificity']['expected']}) | "
                 f"{wiring['m1_classifier_alone_sensitivity_at_union_specificity']['reproduced']} |")
    lines.append(f"| SK/MISC recall agree across paper_metrics_summary.json and "
                 f"perception_phase2_safety_preserving.json | "
                 f"{wiring['cross_file_sk_misc_recall_agreement']['sk_reproduced'] and wiring['cross_file_sk_misc_recall_agreement']['misc_reproduced']} |")
    lines.append(f"| Promoted raw ECE agrees across paper_metrics_summary.json and "
                 f"promoted_two_view_ablation.json | "
                 f"{wiring['cross_file_promoted_raw_ece_agreement']['reproduced']} |")
    lines.append("")

    sections = [
        ("diagnosis", "Diagnosis (5-class + melanoma)"),
        ("criteria", "Criteria (7-point checklist heads)"),
        ("union", "Union melanoma detection (classifier OR checklist)"),
        ("management", "Management recommendation"),
        ("calibration", "Calibration (ECE / AURC)"),
        ("contracts_routing", "Contracts, routing, retrieval"),
        ("ragas", "RAGAS faithfulness"),
        ("seven_point_score", "7-point score agreement (supplementary)"),
        ("generalization", "Cross-dataset generalization (BCN / HAM, supplementary)"),
    ]
    by_section: Dict[str, List[dict]] = {}
    for r in rows:
        by_section.setdefault(r["section"], []).append(r)

    for key, title in sections:
        if key not in by_section:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| metric | dataset | base arm | promoted arm | delta | direction | source |")
        lines.append("|---|---|---:|---:|---:|:---:|---|")
        for r in by_section[key]:
            tag = " (bonus)" if r["bonus"] else ""
            lines.append(
                f"| {r['metric']}{tag} | {r['dataset']} | {_fmt(r['base_arm'])} | "
                f"{_fmt(r['promoted_arm'])} | {_fmt(r['delta'])} | {r['direction']} | "
                f"{r['source_path']} |"
            )
        lines.append("")
        for r in by_section[key]:
            if r.get("note"):
                lines.append(f"- **{r['metric']}**: {r['note']}")
        lines.append("")

    lines.append("## Honest reading")
    lines.append("")
    lines.append(
        "The promotion bought real, transferable gains: MEL AUROC rose (0.895 -> 0.9151, preserved "
        "not traded away by the safety-preserving selection rule), 5-class accuracy/balanced-accuracy/"
        "macro-F1 all rose, SK and MISC recall roughly doubled at the operating point, criteria "
        "binary-concept F1 rose +0.0545, and RAGAS faithfulness rose sharply on every grounding scope "
        "versus the v2 system. It also has real, disclosed costs: per-criterion accuracy fell, union "
        "melanoma specificity fell 0.850 -> 0.697 alongside the sensitivity gain, management QWK "
        "dipped (not resolvable at n=395), retrieval-avoided fell 0.70 -> 0.4926 and conflict rate rose "
        "0.1724 -> 0.266 (more cases now need human adjudication), and on HAM's sealed split the "
        "promoted recipe's MEL AUROC is the one place it slightly regresses (-0.0058). Calibration is "
        "genuinely worse for the promoted head under a fair, apples-to-apples read: temperature-scaling "
        "both heads the same way on VALID (see notes/TEMPERATURE_SCALING_VALID.md) still leaves the "
        "promoted head with higher ECE than the base head, so this is a real cost of the promotion, not "
        "an artefact of the mismatched confidence definitions in the original 0.0891-vs-0.0455 "
        "headline. Finally, the union rule's headline sensitivity gain over the base arm is smaller "
        "than it looks: at the union's own specificity, the promoted classifier alone already reaches "
        "0.9307 (row M1) -- above the union rule's 0.8614 -- so the checklist path's honest "
        "contribution is evidential and complementary (an auditable, human-adjudicated disagreement "
        "signal), not an incremental melanoma-detection gain beyond what the classifier's own ROC "
        "already offers."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# =====================================================================================================
# main
# =====================================================================================================

def main() -> None:
    started = time.perf_counter()

    ts_report = build_temperature_scaling()
    OUT_TS_JSON.write_text(json.dumps(ts_report, indent=2) + "\n", encoding="utf-8")
    OUT_TS_MD.write_text(render_ts_markdown(ts_report), encoding="utf-8")
    print(f"wrote {OUT_TS_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_TS_MD.relative_to(REPO)}")

    trade_report = build_trade_table(ts_report)
    OUT_TRADE_JSON.write_text(json.dumps(trade_report, indent=2) + "\n", encoding="utf-8")
    OUT_TRADE_MD.write_text(render_trade_markdown(trade_report), encoding="utf-8")
    print(f"wrote {OUT_TRADE_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_TRADE_MD.relative_to(REPO)}")

    p, b = ts_report["promoted_head"], ts_report["base_head"]
    print("\n=== part A: temperature scaling (VALID-fit, TEST-evaluated) ===")
    print(f"  wiring self-checks    : "
          f"{all(v['reproduced'] for v in ts_report['protocol']['wiring_self_checks'].values())}")
    print(f"  T promoted / base     : {p['temperature']['T_fit_on_valid']} / "
          f"{b['temperature']['T_fit_on_valid']}")
    print(f"  ECE argmax before/after promoted: {p['ece_argmax_confidence']['before']} -> "
          f"{p['ece_argmax_confidence']['after']}")
    print(f"  ECE argmax before/after base    : {b['ece_argmax_confidence']['before']} -> "
          f"{b['ece_argmax_confidence']['after']}")
    op = p["operating_point_label_confidence"]
    print(f"  ECE op-point-label before/after promoted: {op['ece']['before']} -> {op['ece']['after']} "
          f"(sealed reproduction: {op['sealed_reproduction_check']['reproduced']})")
    print(f"  MEL AUROC drift promoted/base   : "
          f"{p['mel_auroc']['delta_after_minus_before']} / {b['mel_auroc']['delta_after_minus_before']}")

    print("\n=== part B: promotion trade table ===")
    counts = trade_report["protocol"]["row_count"]
    print(f"  wiring self-checks    : "
          f"{all(v.get('reproduced', True) for v in trade_report['protocol']['wiring_self_checks'].values())}")
    print(f"  rows                  : {counts}")

    print(f"\ndone in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
