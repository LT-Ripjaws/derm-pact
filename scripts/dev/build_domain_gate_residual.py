"""R-A follow-up (reviewer M5) -- the domain gate's protective value, quantified.

notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md, finding M5:

    "domain_gate_validation.json reports rank separation (valid-vs-BCN AUROC 0.9657) and the
    flagged fraction (46.4% of BCN images auto-downgraded @ 1.27% in-domain spill) -- but no
    performance on the 53.6% of BCN images that pass and therefore receive the Derm7pt
    classifier, whose cross-centre AUROC is 0.7962 overall. If the passed subset's AUROC ~ 0.80
    the gate is not protecting much; if it is markedly higher the gate is doing real work."

This script answers that question directly: it fits the SAME Derm7pt-trained cross-centre
classifier the sealed BCN test evaluation used (scripts/dev/build_bcn_test_evaluation.py,
imported by reference here rather than reimplemented, so there is no copy-paste drift), scores
every BCN image and lesion with it, and splits each population by the registered domain gate
(k=1, threshold=0.835128, fit on Derm7pt VALID p1 -- medguard_v2/perception/domain_check.py,
validated in scripts/dev/validate_domain_gate.py) into FLAGGED (auto-downgraded to the
single-view report) and PASSED (receives the two-view checklist path).

Nothing is refit or reselected here:

  * The CLASSIFIER is the frozen recipe (Derm7pt TRAIN, dermlip 512-d, C=10.0, seed=20260810) --
    imported by reference, not reimplemented, and verified to reproduce the sealed Derm7pt TEST
    numbers (accuracy 0.7392, MEL AUROC 0.895) before anything else runs. A mismatch STOPS the
    script rather than silently reporting numbers from a subtly different classifier.
  * The GATE is the frozen registered gate (k=1, valid-p1 threshold). The threshold is
    recomputed here with the same ``fit_valid_threshold`` the validation script uses, purely to
    verify it reproduces 0.835128 to 1e-6 -- a reproducibility check, not a refit. No new
    operating point is chosen anywhere in this file.

Population discipline mirrors validate_domain_gate.py exactly:

  * PRIMARY = BCN ``load_records(splits=("train","valid"))`` (n=9,846 images / 2,858 lesions) --
    the identical population the gate was validated against. BCN test is NOT read for this part.
  * SECONDARY (step 7, every number tagged ``post_hoc_bcn_test_reread: true``) = BCN TEST
    (n=712 lesions), read here under ``allow_test=True`` with the user's explicit authorisation
    for this one post-hoc re-read. All-lesion numbers are cross-checked against the sealed
    ``arm2_cross_centre_classifier`` result (accuracy 0.5576, MEL AUROC 0.7962,
    research/evidence/bcn_v1/bcn_test_evaluation.json) before the flagged/passed split is
    reported, so a wiring mistake here cannot masquerade as a finding.
  * A COVERAGE SWEEP (Derm7pt-VALID score percentiles 1/2.5/5/10/20 -- p=1 IS the registered
    operating point) turns the single registered threshold into a risk-coverage curve: as the
    gate tightens (smaller passed fraction), does the passed subset's quality keep rising,
    plateau, or is the registered point already near the gate's ceiling?

Everything here is 100% offline: features are read from the frozen caches on disk, no image is
re-encoded, no paid API is called. Run with the feature venv (venv_win\\Scripts\\python.exe).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.data.bcn20000 import BCN_TO_GROUP, BcnRecord, lesion_level, load_records  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.eval.metrics import classification_summary, sensitivity_at_fixed_specificity  # noqa: E402
from medguard_v2.eval.statistics import binary_auc, clustered_bootstrap_ci  # noqa: E402
from medguard_v2.eval.taxonomy import assert_labels_match  # noqa: E402
from medguard_v2.perception.domain_check import (  # noqa: E402
    DOMAIN_GATE_K,
    DOMAIN_GATE_VALID_PERCENTILE,
    _load_dermlip_features,
    build_train_bank,
    domain_scores,
    fit_valid_threshold,
    load_domain_gate,
)

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "domain_gate_residual.json"
OUT_MD = REPO / "notes" / "DOMAIN_GATE_RESIDUAL.md"

N_BOOT = 1000
BOOT_SEED = 20260712
BOOTSTRAP_CONFIDENCE = 0.95
SWEEP_PERCENTILES: Tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 20.0)

#: Sealed numbers the wiring self-check / post-hoc cross-check must reproduce before anything
#: downstream is trusted. Sources named alongside each constant.
REF_DERM7PT_TEST_ACCURACY = 0.7392   # paper_metrics_summary.json -> perception_derm7pt.classifier_5class.base_arm_reference.accuracy
REF_DERM7PT_TEST_MEL_AUROC = 0.895   # paper_metrics_summary.json -> perception_derm7pt.melanoma_auroc.base_arm_reference.point
REF_GATE_THRESHOLD = 0.835128        # research/evidence/derm7pt_v1/domain_gate_validation.json -> registered_gate.threshold
REF_BCN_TEST_ACCURACY = 0.5576       # paper_metrics_summary.json -> perception_bcn.cross_centre.accuracy
REF_BCN_TEST_MEL_AUROC = 0.7962      # paper_metrics_summary.json -> perception_bcn.cross_centre.mel_auroc

#: Canonical Derm7pt 5-group label order, resolved through the taxonomy guard (same call
#: scripts/dev/build_bcn_test_evaluation.py makes) so a mixed-taxonomy label list fails loudly.
LABELS: Tuple[str, ...] = tuple(assert_labels_match(set(BCN_TO_GROUP.values()), "derm7pt"))


# ---------------------------------------------------------------------------
# Reuse the exact classifier-fitting code path (no copy-paste drift)
# ---------------------------------------------------------------------------
def _import_bcn_test_evaluation():
    """Import scripts/dev/build_bcn_test_evaluation.py (not a package) to reuse its exact
    Derm7pt-trained classifier fit, so the cross-centre classifier here is bit-identical to the
    one the sealed BCN test evaluation used (same pattern as
    scripts/dev/promoted_recipe_ham_bcn.py's helper for build_ham_evaluation.py)."""
    spec = importlib.util.spec_from_file_location(
        "build_bcn_test_evaluation", REPO / "scripts/dev/build_bcn_test_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _score(classifier: LogisticRegression, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """5-class posterior -> (predicted label, MEL posterior mass) for every row of ``x``."""
    classes = list(classifier.classes_)
    assert "MEL" in classes, "classifier was never trained on a MEL example"
    mel_col = classes.index("MEL")
    posterior = classifier.predict_proba(x)
    predicted = classifier.predict(x)
    return predicted, posterior[:, mel_col]


# ---------------------------------------------------------------------------
# Metric helpers (mirrors scripts/dev/build_bcn_test_evaluation.py's _melanoma_auroc_ci)
# ---------------------------------------------------------------------------
def _mel_auroc_ci(truth_mel: Sequence[bool], mel_score: Sequence[float],
                   cluster_ids: Sequence[str], *, n_boot: int, seed: int) -> Dict[str, object]:
    rows = list(zip([bool(t) for t in truth_mel], [float(s) for s in mel_score]))
    interval = clustered_bootstrap_ci(
        rows, list(cluster_ids),
        lambda sample: binary_auc([r[0] for r in sample], [r[1] for r in sample]),
        n_boot=n_boot, confidence=BOOTSTRAP_CONFIDENCE, seed=seed,
    )
    return {
        "estimate": round(interval.estimate, 4),
        "ci95_low": round(interval.low, 4),
        "ci95_high": round(interval.high, 4),
        "cluster_unit": "lesion_id",
        "n_boot": n_boot,
        "seed": seed,
    }


def _subset_block(records: Sequence[BcnRecord], predicted: np.ndarray, mel_score: np.ndarray,
                   *, n_boot: int, seed: int) -> Dict[str, object]:
    """n, group composition, 5-class + melanoma metrics for one ALL/FLAGGED/PASSED subset."""
    n = len(records)
    truth_group = [r.group for r in records]
    group_counts = {g: truth_group.count(g) for g in LABELS}
    block: Dict[str, object] = {"n": n, "group_counts": group_counts}
    if n == 0:
        block["note"] = "empty subset"
        return block

    predicted_list = [str(p) for p in predicted]
    summary = classification_summary(truth_group, predicted_list, list(LABELS))
    block["accuracy"] = round(summary["accuracy"], 4)
    block["balanced_accuracy"] = round(summary["balanced_accuracy"], 4)
    block["macro_f1"] = round(summary["macro_f1"], 4)
    block["per_class_recall"] = {g: round(summary["per_class"][g]["recall"], 4) for g in LABELS}

    truth_mel = [g == "MEL" for g in truth_group]
    n_mel = sum(truth_mel)
    classifier_flag = [p == "MEL" for p in predicted_list]
    tp = sum(1 for f, t in zip(classifier_flag, truth_mel) if f and t)
    fn = sum(1 for f, t in zip(classifier_flag, truth_mel) if not f and t)
    block["mel_recall_at_argmax"] = round(tp / (tp + fn), 4) if (tp + fn) else None

    if n_mel == 0 or n_mel == n:
        block["mel_auroc"] = None
        block["mel_sens_at_spec90"] = None
        block["mel_note"] = "AUROC / sens@spec undefined: subset contains only one melanoma class"
    else:
        lesion_ids = [r.lesion_id for r in records]
        block["mel_auroc"] = _mel_auroc_ci(
            truth_mel, [float(s) for s in mel_score], lesion_ids, n_boot=n_boot, seed=seed)
        block["mel_sens_at_spec90"] = sensitivity_at_fixed_specificity(
            truth_mel, [float(s) for s in mel_score], 0.90)
    return block


def _split_and_report(records: Sequence[BcnRecord], predicted: np.ndarray, mel_score: np.ndarray,
                       flagged_mask: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, object]:
    """ALL / FLAGGED / PASSED metric blocks + per-group flagged fraction for one population."""
    flagged_mask = np.asarray(flagged_mask, dtype=bool)
    predicted = np.asarray(predicted)
    mel_score = np.asarray(mel_score, dtype=np.float64)
    n = len(records)

    all_block = _subset_block(records, predicted, mel_score, n_boot=n_boot, seed=seed)
    flagged_records = [r for r, f in zip(records, flagged_mask) if f]
    passed_records = [r for r, f in zip(records, flagged_mask) if not f]
    flagged_block = _subset_block(
        flagged_records, predicted[flagged_mask], mel_score[flagged_mask], n_boot=n_boot, seed=seed)
    passed_block = _subset_block(
        passed_records, predicted[~flagged_mask], mel_score[~flagged_mask], n_boot=n_boot, seed=seed)

    groups = np.asarray([r.group for r in records])
    group_flagged_fraction: Dict[str, Optional[float]] = {}
    for g in LABELS:
        idx = groups == g
        n_g = int(idx.sum())
        group_flagged_fraction[g] = round(float(flagged_mask[idx].mean()), 4) if n_g else None

    return {
        "n_total": n,
        "n_flagged": int(flagged_mask.sum()),
        "n_passed": int((~flagged_mask).sum()),
        "flagged_fraction": round(float(flagged_mask.mean()), 4) if n else None,
        "group_flagged_fraction": group_flagged_fraction,
        "group_flagged_fraction_note": (
            "fraction of each true diagnostic group whose gate score falls below threshold "
            "(auto-downgraded); answers whether melanoma is flagged more or less than others"
        ),
        "all": all_block,
        "flagged": flagged_block,
        "passed": passed_block,
    }


def _sweep_row(records: Sequence[BcnRecord], predicted: np.ndarray, mel_score: np.ndarray,
               gate_score: np.ndarray, threshold_p: float) -> Dict[str, object]:
    """Passed-subset accuracy / MEL AUROC / MEL recall at one coverage-sweep threshold."""
    gate_score = np.asarray(gate_score, dtype=np.float64)
    predicted = np.asarray(predicted)
    mel_score = np.asarray(mel_score, dtype=np.float64)
    passed_mask = gate_score >= threshold_p
    n = len(records)
    n_passed = int(passed_mask.sum())
    row: Dict[str, object] = {
        "threshold": round(float(threshold_p), 6),
        "n_total": n,
        "n_passed": n_passed,
        "passed_fraction": round(n_passed / n, 4) if n else None,
    }
    if n_passed == 0:
        row.update({"accuracy": None, "balanced_accuracy": None, "macro_f1": None,
                     "mel_auroc": None, "mel_recall_at_argmax": None})
        return row

    passed_records = [r for r, m in zip(records, passed_mask) if m]
    truth_group = [r.group for r in passed_records]
    predicted_passed = [str(p) for p in predicted[passed_mask]]
    summary = classification_summary(truth_group, predicted_passed, list(LABELS))
    row["accuracy"] = round(summary["accuracy"], 4)
    row["balanced_accuracy"] = round(summary["balanced_accuracy"], 4)
    row["macro_f1"] = round(summary["macro_f1"], 4)

    truth_mel = [g == "MEL" for g in truth_group]
    n_mel = sum(truth_mel)
    classifier_flag = [p == "MEL" for p in predicted_passed]
    tp = sum(1 for f, t in zip(classifier_flag, truth_mel) if f and t)
    fn = sum(1 for f, t in zip(classifier_flag, truth_mel) if not f and t)
    row["mel_recall_at_argmax"] = round(tp / (tp + fn), 4) if (tp + fn) else None

    if n_mel == 0 or n_mel == len(truth_group):
        row["mel_auroc"] = None
    else:
        mel_score_passed = mel_score[passed_mask]
        row["mel_auroc"] = round(binary_auc(truth_mel, [float(s) for s in mel_score_passed]), 4)
    return row


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build(*, n_boot: int, seed: int) -> Dict[str, object]:
    assert SWEEP_PERCENTILES[0] == DOMAIN_GATE_VALID_PERCENTILE, (
        "sweep's first percentile must equal the registered operating point"
    )

    # ---- frozen classifier: reused by reference, not reimplemented ---------------------------
    bcn_eval_mod = _import_bcn_test_evaluation()
    classifier, _heads, n_derm7pt_train = bcn_eval_mod._fit_derm7pt_classifier_and_heads()
    assert n_derm7pt_train == 413, f"Derm7pt TRAIN cohort is {n_derm7pt_train}, expected 413"
    derm7pt_c = bcn_eval_mod.DERM7PT_C
    derm7pt_seed = bcn_eval_mod.DERM7PT_SEED

    # ---- frozen gate: bank + threshold, recomputed only to verify reproducibility ------------
    cases = load_cases()
    features = _load_dermlip_features()
    bank = build_train_bank(features, cases)
    canonical_gate = load_domain_gate()
    assert bank.bank_hash == canonical_gate.bank_hash, "freshly built TRAIN bank hash mismatch"

    valid_cases = [c for c in cases if c.split == "valid"]
    assert len(valid_cases) == 203, f"Derm7pt VALID cohort is {len(valid_cases)}, expected 203"
    valid_matrix = np.stack([features[str(c.case_num)] for c in valid_cases])
    fit = fit_valid_threshold(bank, valid_matrix, k=DOMAIN_GATE_K)
    threshold = float(fit["threshold"])

    gate_threshold_matches = abs(threshold - REF_GATE_THRESHOLD) < 1e-6
    if not gate_threshold_matches:
        raise AssertionError(
            f"WIRING SELF-CHECK FAILED: recomputed gate threshold {threshold:.6f} != registered "
            f"{REF_GATE_THRESHOLD} (domain_gate_validation.json -> registered_gate.threshold) "
            "to 1e-6. STOPPING -- do not trust anything downstream of this."
        )
    assert abs(threshold - canonical_gate.threshold) < 1e-9, (
        "recomputed threshold disagrees with load_domain_gate()'s canonical threshold"
    )

    # ---- wiring self-check: Derm7pt TEST must reproduce the sealed base-arm numbers ----------
    derm7pt_test_cases = [c for c in cases if c.split == "test"]
    assert len(derm7pt_test_cases) == 395, (
        f"Derm7pt TEST cohort is {len(derm7pt_test_cases)}, expected 395"
    )
    x_derm7pt_test = np.stack([features[str(c.case_num)] for c in derm7pt_test_cases])
    truth_derm7pt_test = [c.diagnosis_group for c in derm7pt_test_cases]
    pred_derm7pt_test, mel_score_derm7pt_test = _score(classifier, x_derm7pt_test)

    wiring_summary = classification_summary(
        truth_derm7pt_test, [str(p) for p in pred_derm7pt_test], list(LABELS))
    wiring_accuracy = round(wiring_summary["accuracy"], 4)
    truth_derm7pt_test_mel = [g == "MEL" for g in truth_derm7pt_test]
    wiring_auroc = round(
        binary_auc(truth_derm7pt_test_mel, [float(s) for s in mel_score_derm7pt_test]), 4)

    accuracy_matches = wiring_accuracy == REF_DERM7PT_TEST_ACCURACY
    auroc_matches = round(wiring_auroc, 3) == REF_DERM7PT_TEST_MEL_AUROC
    if not (accuracy_matches and auroc_matches):
        raise AssertionError(
            "WIRING SELF-CHECK FAILED: Derm7pt TEST reproduction does not match the sealed base "
            f"arm. accuracy {wiring_accuracy} (expected {REF_DERM7PT_TEST_ACCURACY}, match="
            f"{accuracy_matches}); MEL AUROC {wiring_auroc} (expected {REF_DERM7PT_TEST_MEL_AUROC}"
            f", match={auroc_matches}). STOPPING -- do not trust anything downstream of this."
        )

    wiring_self_check = {
        "status": "PASSED",
        "classifier_recipe": {
            "source": "scripts/dev/build_bcn_test_evaluation.py:_fit_derm7pt_classifier_and_heads",
            "fit_on": "Derm7pt TRAIN (dermlip 512-d)",
            "n_train": n_derm7pt_train,
            "C": derm7pt_c,
            "seed": derm7pt_seed,
        },
        "derm7pt_test_n": len(derm7pt_test_cases),
        "accuracy": wiring_accuracy,
        "accuracy_reference": REF_DERM7PT_TEST_ACCURACY,
        "accuracy_matches": accuracy_matches,
        "mel_auroc": wiring_auroc,
        "mel_auroc_reference": REF_DERM7PT_TEST_MEL_AUROC,
        "mel_auroc_matches": auroc_matches,
        "gate_threshold_recomputed": round(threshold, 6),
        "gate_threshold_registered": REF_GATE_THRESHOLD,
        "gate_threshold_matches": gate_threshold_matches,
        "gate_k": DOMAIN_GATE_K,
        "gate_bank_hash": bank.bank_hash,
        "gate_bank_hash_matches_canonical": bank.bank_hash == canonical_gate.bank_hash,
    }

    # ---- in-domain flagged spill: Derm7pt TEST images the gate flags -------------------------
    test_gate_score = domain_scores(bank, x_derm7pt_test, DOMAIN_GATE_K)
    test_flagged_mask = test_gate_score < threshold
    in_domain_flagged_spill = {
        "n": int(test_flagged_mask.sum()),
        "flagged_fraction": round(float(test_flagged_mask.mean()), 4),
        "cases": [
            {
                "case_num": c.case_num,
                "true_group": c.diagnosis_group,
                "predicted_group": str(p),
                "correct": bool(str(p) == c.diagnosis_group),
                "domain_score": round(float(s), 4),
            }
            for c, p, s, f in zip(
                derm7pt_test_cases, pred_derm7pt_test, test_gate_score, test_flagged_mask)
            if f
        ],
    }

    # ---- PRIMARY population: BCN train+valid, image level + lesion level ---------------------
    bcn_records, bcn_images = load_records(splits=("train", "valid"))
    bcn_lesion_records, bcn_lesions = lesion_level(bcn_records, bcn_images)
    assert len(bcn_records) == bcn_images.shape[0] == 9846, (
        f"BCN primary image population is {len(bcn_records)}, expected 9846"
    )
    assert len(bcn_lesion_records) == bcn_lesions.shape[0] == 2858, (
        f"BCN primary lesion population is {len(bcn_lesion_records)}, expected 2858"
    )

    img_pred, img_mel_score = _score(classifier, bcn_images)
    les_pred, les_mel_score = _score(classifier, bcn_lesions)
    img_gate_score = domain_scores(bank, bcn_images, DOMAIN_GATE_K)
    les_gate_score = domain_scores(bank, bcn_lesions, DOMAIN_GATE_K)
    img_flagged = img_gate_score < threshold
    les_flagged = les_gate_score < threshold

    image_level = _split_and_report(
        bcn_records, img_pred, img_mel_score, img_flagged, n_boot=n_boot, seed=seed)
    image_level["note"] = (
        "flagged/passed are gated at IMAGE level (the gate's deployment unit); a lesion with "
        "multiple images can therefore contribute to both the flagged and the passed bucket"
    )
    lesion_level_block = _split_and_report(
        bcn_lesion_records, les_pred, les_mel_score, les_flagged, n_boot=n_boot, seed=seed)

    # ---- coverage sweep: risk-coverage curve over Derm7pt-VALID score percentiles ------------
    sweep_image: List[Dict[str, object]] = []
    sweep_lesion: List[Dict[str, object]] = []
    for p in SWEEP_PERCENTILES:
        fit_p = fit_valid_threshold(bank, valid_matrix, k=DOMAIN_GATE_K, percentile=p)
        threshold_p = float(fit_p["threshold"])
        row_img = {"percentile": p,
                   **_sweep_row(bcn_records, img_pred, img_mel_score, img_gate_score, threshold_p)}
        row_les = {"percentile": p,
                   **_sweep_row(bcn_lesion_records, les_pred, les_mel_score, les_gate_score, threshold_p)}
        sweep_image.append(row_img)
        sweep_lesion.append(row_les)
    assert abs(sweep_image[0]["threshold"] - round(threshold, 6)) < 1e-6, (
        "sweep p=1 threshold disagrees with the registered threshold"
    )
    coverage_sweep = {
        "percentiles": list(SWEEP_PERCENTILES),
        "definition": (
            "threshold = percentile p of the Derm7pt-VALID domain-score distribution "
            "(fit_valid_threshold, same population/formula as the registered gate; p=1 IS the "
            "registered operating point). PASSED = gate score >= threshold. Point estimates "
            "only (no bootstrap CI); see image_level/lesion_level above for the CI at p=1."
        ),
        "image_level": sweep_image,
        "lesion_level": sweep_lesion,
    }

    # ---- SECONDARY (post-hoc): BCN TEST lesions, user-authorised re-read ----------------------
    bcn_test_records, bcn_test_images = load_records(splits=("test",), allow_test=True)
    bcn_test_lesion_records, bcn_test_lesions = lesion_level(bcn_test_records, bcn_test_images)
    assert len(bcn_test_lesion_records) == bcn_test_lesions.shape[0] == 712, (
        f"BCN test lesion population is {len(bcn_test_lesion_records)}, expected 712"
    )
    test_pred, test_mel_score = _score(classifier, bcn_test_lesions)
    truth_bcn_test = [r.group for r in bcn_test_lesion_records]
    test_summary = classification_summary(truth_bcn_test, [str(p) for p in test_pred], list(LABELS))
    test_accuracy = round(test_summary["accuracy"], 4)
    truth_bcn_test_mel = [g == "MEL" for g in truth_bcn_test]
    test_auroc = round(binary_auc(truth_bcn_test_mel, [float(s) for s in test_mel_score]), 4)

    bcn_test_cross_check_matches = (
        test_accuracy == REF_BCN_TEST_ACCURACY and test_auroc == REF_BCN_TEST_MEL_AUROC
    )
    if not bcn_test_cross_check_matches:
        raise AssertionError(
            "POST-HOC CROSS-CHECK FAILED: BCN TEST all-lesion reproduction does not match the "
            f"sealed cross-centre arm. accuracy {test_accuracy} (expected {REF_BCN_TEST_ACCURACY}"
            f"), MEL AUROC {test_auroc} (expected {REF_BCN_TEST_MEL_AUROC}). STOPPING."
        )

    test_gate_score_bcn = domain_scores(bank, bcn_test_lesions, DOMAIN_GATE_K)
    test_flagged_bcn = test_gate_score_bcn < threshold
    post_hoc_lesion_level = _split_and_report(
        bcn_test_lesion_records, test_pred, test_mel_score, test_flagged_bcn,
        n_boot=n_boot, seed=seed)

    post_hoc_bcn_test = {
        "post_hoc_bcn_test_reread": True,
        "authorisation": (
            "user-authorised post-hoc re-read for this analysis only (BCN test is otherwise "
            "sealed/spent -- research/evidence/bcn_v1/bcn_test_evaluation.json)"
        ),
        "n": len(bcn_test_lesion_records),
        "all_lesion_cross_check": {
            "accuracy": test_accuracy,
            "accuracy_reference": REF_BCN_TEST_ACCURACY,
            "mel_auroc": test_auroc,
            "mel_auroc_reference": REF_BCN_TEST_MEL_AUROC,
            "matches": bcn_test_cross_check_matches,
        },
        "lesion_level": post_hoc_lesion_level,
    }

    # ---- honest verdict, whichever way the numbers land ---------------------------------------
    verdict_bits: List[str] = []
    passed_auroc = image_level["passed"].get("mel_auroc")
    flagged_auroc = image_level["flagged"].get("mel_auroc")
    if passed_auroc and flagged_auroc:
        gap = round(passed_auroc["estimate"] - flagged_auroc["estimate"], 4)
        verdict_bits.append(
            f"Image level (primary, n={image_level['n_total']}): PASSED-subset MEL AUROC "
            f"{passed_auroc['estimate']:.4f} [{passed_auroc['ci95_low']:.4f}, "
            f"{passed_auroc['ci95_high']:.4f}] vs FLAGGED-subset {flagged_auroc['estimate']:.4f} "
            f"[{flagged_auroc['ci95_low']:.4f}, {flagged_auroc['ci95_high']:.4f}] -- a gap of "
            f"{gap:+.4f}."
        )
    if passed_auroc:
        delta_vs_sealed_all = round(passed_auroc["estimate"] - REF_BCN_TEST_MEL_AUROC, 4)
        verdict_bits.append(
            f"Against the sealed ALL-BCN-test cross-centre AUROC of {REF_BCN_TEST_MEL_AUROC:.4f} "
            f"(what the classifier gets with no gate at all), the PASSED image subset scores "
            f"{delta_vs_sealed_all:+.4f} -- "
            + ("a material improvement: the gate is doing real protective work, not just "
               "rank-separating noise."
               if delta_vs_sealed_all >= 0.02 else
               "close to the ungated number, so the gate is not buying much melanoma-detection "
               "quality on the images it lets through, whatever its rank-separation AUROC says.")
        )
    mel_flag_frac = image_level["group_flagged_fraction"].get("MEL")
    non_mel_vals = [
        image_level["group_flagged_fraction"][g] for g in LABELS
        if g != "MEL" and image_level["group_flagged_fraction"][g] is not None
    ]
    mean_nonmel_flag = sum(non_mel_vals) / len(non_mel_vals) if non_mel_vals else None
    if mel_flag_frac is not None and mean_nonmel_flag is not None:
        direction = ("MORE" if mel_flag_frac > mean_nonmel_flag
                     else "LESS" if mel_flag_frac < mean_nonmel_flag else "about as")
        verdict_bits.append(
            f"Melanoma is flagged {direction} often than the other four groups: "
            f"{mel_flag_frac:.1%} of MEL images vs a {mean_nonmel_flag:.1%} mean across "
            "NEV/BCC/SK/MISC (image level) -- "
            + ("a safety concern: the class that matters most is disproportionately "
               "auto-downgraded to the single-view report."
               if mel_flag_frac > mean_nonmel_flag + 0.03 else
               "no strong evidence the gate singles out melanoma for downgrade.")
        )
    sweep_auroc_series = [row["mel_auroc"] for row in sweep_image if row["mel_auroc"] is not None]
    if len(sweep_auroc_series) >= 2:
        nondecreasing = all(b >= a - 1e-9 for a, b in zip(sweep_auroc_series, sweep_auroc_series[1:]))
        verdict_bits.append(
            "Coverage sweep (image level, PASSED-subset MEL AUROC as the threshold tightens "
            f"from p=1 to p=20): {sweep_auroc_series} -- "
            + ("rises as coverage shrinks, so a stricter gate keeps buying quality beyond the "
               "registered operating point."
               if nondecreasing else
               "does not rise monotonically, so the registered p=1 operating point is already "
               "close to the ceiling of what this gate can buy; tightening further mostly trades "
               "away coverage without a reliable quality return.")
        )
    pht_passed_auroc = post_hoc_lesion_level["passed"].get("mel_auroc")
    pht_flagged_auroc = post_hoc_lesion_level["flagged"].get("mel_auroc")
    if pht_passed_auroc and pht_flagged_auroc:
        primary_direction_up = bool(
            passed_auroc and flagged_auroc and passed_auroc["estimate"] >= flagged_auroc["estimate"]
        )
        pht_direction_up = pht_passed_auroc["estimate"] >= pht_flagged_auroc["estimate"]
        verdict_bits.append(
            "Post-hoc on the sealed BCN TEST lesions (n=712, post_hoc_bcn_test_reread=true): "
            f"PASSED MEL AUROC {pht_passed_auroc['estimate']:.4f} vs FLAGGED "
            f"{pht_flagged_auroc['estimate']:.4f} -- "
            + ("the same direction as the primary train+valid result above."
               if pht_direction_up == primary_direction_up else
               "the OPPOSITE direction from the primary train+valid result -- flag this for the "
               "paper rather than quoting only one of them.")
        )
    verdict = " ".join(verdict_bits)

    return {
        "protocol": "derm7pt-domain-gate-residual-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis_type": (
            "post-hoc gate-residual analysis; classifier and gate frozen; no threshold refit"
        ),
        "reviewer_finding": {
            "id": "M5",
            "source": "notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md",
            "text": (
                "The domain gate's protective value is unquantified on the images it passes: "
                "domain_gate_validation.json reports rank separation and the flagged fraction "
                "but no performance on the passed subset."
            ),
        },
        "question": (
            "Does the classifier perform materially better on BCN images the domain gate PASSES "
            "than on the ones it FLAGS (auto-downgrades)? And does the gate preferentially flag "
            "melanoma?"
        ),
        "boundaries": {
            "paid_calls": 0,
            "generation_calls": 0,
            "images_encoded": 0,
            "classifier_refit_or_reselected": False,
            "gate_refit_or_reselected": False,
            "classifier_note": (
                "the classifier is imported by reference from "
                "scripts/dev/build_bcn_test_evaluation.py:_fit_derm7pt_classifier_and_heads and "
                "verified (wiring_self_check) to reproduce the sealed Derm7pt TEST numbers "
                "before any BCN scoring is trusted"
            ),
            "gate_note": (
                "k and the valid-p1 threshold are the registered values "
                "(medguard_v2/perception/domain_check.py); the threshold is recomputed here "
                "only to verify reproducibility (assert to 1e-6), never refit or reselected"
            ),
            "primary_population": (
                "BCN load_records(splits=('train','valid')) -- identical to "
                "scripts/dev/validate_domain_gate.py; BCN test is NOT read for this part"
            ),
            "bcn_test_read": True,
            "bcn_test_read_scope": (
                "post_hoc_bcn_test only (step 7), user-authorised post-hoc re-read; every "
                "number in that block is tagged post_hoc_bcn_test_reread=true"
            ),
        },
        "wiring_self_check": wiring_self_check,
        "registered_gate": canonical_gate.summary(),
        "populations": {
            "derm7pt_train_bank": {"n": n_derm7pt_train, "role": "classifier fit + gate bank"},
            "derm7pt_valid": {"n": len(valid_cases), "role": "gate threshold fit (p1)"},
            "derm7pt_test": {
                "n": len(derm7pt_test_cases),
                "role": "wiring self-check + in-domain flagged spill",
            },
            "bcn_image_primary": {"n": int(bcn_images.shape[0]), "role": "PRIMARY, image level"},
            "bcn_lesion_primary": {
                "n": int(bcn_lesions.shape[0]), "role": "PRIMARY, lesion level (mean-pooled)"
            },
            "bcn_test_lesion_post_hoc": {
                "n": len(bcn_test_lesion_records),
                "role": "SECONDARY, post-hoc (post_hoc_bcn_test_reread=true)",
            },
        },
        "image_level": image_level,
        "lesion_level": lesion_level_block,
        "in_domain_flagged_spill": in_domain_flagged_spill,
        "coverage_sweep": coverage_sweep,
        "post_hoc_bcn_test": post_hoc_bcn_test,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# notes/DOMAIN_GATE_RESIDUAL.md
# ---------------------------------------------------------------------------
def _fmt_auroc(block: Optional[Dict[str, object]]) -> str:
    if not block:
        return "n/a"
    return f"{block['estimate']:.4f} [{block['ci95_low']:.4f}, {block['ci95_high']:.4f}]"


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _subset_row(name: str, block: Dict[str, object]) -> str:
    sens90 = block.get("mel_sens_at_spec90")
    sens90_str = f"{sens90['sensitivity']:.4f}" if sens90 else "n/a"
    return (
        f"| {name} | {block['n']} | {block.get('accuracy', 'n/a')} | "
        f"{block.get('balanced_accuracy', 'n/a')} | {_fmt_auroc(block.get('mel_auroc'))} | "
        f"{block.get('mel_recall_at_argmax', 'n/a')} | {sens90_str} |"
    )


def _write_notes(r: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Domain-gate residual analysis (reviewer M5: is the PASSED subset actually safer?)")
    lines.append("")
    lines.append(
        "M5 (`notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md`): `domain_gate_validation.json` "
        "reports that the gate separates in-domain from BCN in RANK (AUROC 0.9657) and flags "
        "46.4% of BCN images, but nothing measured what the Derm7pt-trained classifier does on "
        "the 53.6% of BCN images the gate PASSES. This artifact closes that gap. Classifier and "
        "gate are both frozen (no refit, no reselection); see `domain_gate_residual.json` for "
        "every number."
    )
    lines.append("")
    wsc = r["wiring_self_check"]
    lines.append(
        f"**Wiring self-check: {wsc['status']}.** Derm7pt TEST reproduction -- accuracy "
        f"{wsc['accuracy']} (sealed base {wsc['accuracy_reference']}, match "
        f"{wsc['accuracy_matches']}); MEL AUROC {wsc['mel_auroc']} (sealed base "
        f"{wsc['mel_auroc_reference']}, match {wsc['mel_auroc_matches']}); gate threshold "
        f"{wsc['gate_threshold_recomputed']} (registered {wsc['gate_threshold_registered']}, "
        f"match {wsc['gate_threshold_matches']})."
    )
    lines.append("")
    lines.append(
        f"## Image level (PRIMARY -- the gate's deployment unit; BCN train+valid, "
        f"n={r['image_level']['n_total']})"
    )
    lines.append("")
    lines.append(
        "| subset | n | accuracy | balanced acc | MEL AUROC [95% CI] | MEL recall@argmax | "
        "MEL sens@spec0.90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, key in (("ALL", "all"), ("FLAGGED (downgraded)", "flagged"), ("PASSED (two-view)", "passed")):
        lines.append(_subset_row(name, r["image_level"][key]))
    lines.append("")
    gff = r["image_level"]["group_flagged_fraction"]
    lines.append(
        "Group composition, ALL: " +
        ", ".join(f"{g} {r['image_level']['all']['group_counts'][g]}" for g in gff) +
        ". Fraction of each group flagged: " +
        ", ".join(f"{g} {_fmt_pct(gff[g])}" for g in gff) + "."
    )
    lines.append("")
    lines.append(f"## Lesion level (secondary; mean-pooled, n={r['lesion_level']['n_total']})")
    lines.append("")
    lines.append(
        "| subset | n | accuracy | balanced acc | MEL AUROC [95% CI] | MEL recall@argmax | "
        "MEL sens@spec0.90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, key in (("ALL", "all"), ("FLAGGED (downgraded)", "flagged"), ("PASSED (two-view)", "passed")):
        lines.append(_subset_row(name, r["lesion_level"][key]))
    lines.append("")
    gff_l = r["lesion_level"]["group_flagged_fraction"]
    lines.append(
        "Group composition, ALL: " +
        ", ".join(f"{g} {r['lesion_level']['all']['group_counts'][g]}" for g in gff_l) +
        ". Fraction of each group flagged: " +
        ", ".join(f"{g} {_fmt_pct(gff_l[g])}" for g in gff_l) + "."
    )
    lines.append("")
    lines.append("## In-domain flagged spill (Derm7pt TEST images the gate flags)")
    lines.append("")
    spill = r["in_domain_flagged_spill"]
    lines.append(
        f"n={spill['n']} / {r['populations']['derm7pt_test']['n']} "
        f"({spill['flagged_fraction']:.2%})."
    )
    lines.append("")
    if spill["cases"]:
        lines.append("| case_num | true | predicted | correct | domain score |")
        lines.append("|---:|---|---|---|---:|")
        for c in spill["cases"]:
            lines.append(
                f"| {c['case_num']} | {c['true_group']} | {c['predicted_group']} | "
                f"{c['correct']} | {c['domain_score']:.4f} |"
            )
    lines.append("")
    lines.append("## Coverage sweep (risk-coverage curve; image level, PASSED subset only)")
    lines.append("")
    lines.append(
        "| valid percentile | threshold | passed n | passed fraction | accuracy | MEL AUROC | "
        "MEL recall@argmax |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in r["coverage_sweep"]["image_level"]:
        mark = " (registered)" if row["percentile"] == 1.0 else ""
        lines.append(
            f"| {row['percentile']}{mark} | {row['threshold']:.4f} | {row['n_passed']} | "
            f"{row['passed_fraction']} | {row['accuracy']} | {row['mel_auroc']} | "
            f"{row['mel_recall_at_argmax']} |"
        )
    lines.append("")
    lines.append("## Secondary (post-hoc): BCN TEST lesions (n=712) -- `post_hoc_bcn_test_reread: true`")
    lines.append("")
    pht = r["post_hoc_bcn_test"]
    cc = pht["all_lesion_cross_check"]
    lines.append(
        f"All-lesion cross-check: accuracy {cc['accuracy']} (sealed {cc['accuracy_reference']}), "
        f"MEL AUROC {cc['mel_auroc']} (sealed {cc['mel_auroc_reference']}) -- matches: "
        f"{cc['matches']}."
    )
    lines.append("")
    lines.append(
        "| subset | n | accuracy | balanced acc | MEL AUROC [95% CI] | MEL recall@argmax | "
        "MEL sens@spec0.90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, key in (("ALL", "all"), ("FLAGGED (downgraded)", "flagged"), ("PASSED (two-view)", "passed")):
        lines.append(_subset_row(name, pht["lesion_level"][key]))
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(r["verdict"])
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args()

    t0 = time.perf_counter()
    report = build(n_boot=args.n_boot, seed=args.seed)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_notes(report)
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")

    wsc = report["wiring_self_check"]
    print("\n=== wiring self-check ===")
    print(f"  Derm7pt TEST accuracy {wsc['accuracy']} (ref {wsc['accuracy_reference']}, "
          f"match {wsc['accuracy_matches']})")
    print(f"  Derm7pt TEST MEL AUROC {wsc['mel_auroc']} (ref {wsc['mel_auroc_reference']}, "
          f"match {wsc['mel_auroc_matches']})")
    print(f"  gate threshold {wsc['gate_threshold_recomputed']} (registered "
          f"{wsc['gate_threshold_registered']}, match {wsc['gate_threshold_matches']})")

    for level_name, level_key in (("image", "image_level"), ("lesion", "lesion_level")):
        block = report[level_key]
        print(f"\n=== {level_name} level (n={block['n_total']}) ===")
        for key in ("all", "flagged", "passed"):
            b = block[key]
            auroc = b.get("mel_auroc")
            auroc_str = (
                f"{auroc['estimate']:.4f} [{auroc['ci95_low']:.4f},{auroc['ci95_high']:.4f}]"
                if auroc else "n/a"
            )
            print(f"  {key:8s} n={b['n']:5d}  acc={b.get('accuracy')}  "
                  f"bal_acc={b.get('balanced_accuracy')}  MEL_AUROC={auroc_str}  "
                  f"MEL_recall={b.get('mel_recall_at_argmax')}")

    print("\n=== coverage sweep (image level, passed subset) ===")
    for row in report["coverage_sweep"]["image_level"]:
        print(f"  p={row['percentile']:>5} thr={row['threshold']:.4f}  "
              f"n_passed={row['n_passed']:5d} passed_frac={row['passed_fraction']}  "
              f"acc={row['accuracy']}  MEL_AUROC={row['mel_auroc']}  "
              f"MEL_recall={row['mel_recall_at_argmax']}")

    print("\n=== post-hoc BCN test (n=712, post_hoc_bcn_test_reread=true) ===")
    cc = report["post_hoc_bcn_test"]["all_lesion_cross_check"]
    print(f"  all-lesion cross-check: acc={cc['accuracy']} (ref {cc['accuracy_reference']}) "
          f"MEL_AUROC={cc['mel_auroc']} (ref {cc['mel_auroc_reference']}) matches={cc['matches']}")
    for key in ("all", "flagged", "passed"):
        b = report["post_hoc_bcn_test"]["lesion_level"][key]
        auroc = b.get("mel_auroc")
        auroc_str = (
            f"{auroc['estimate']:.4f} [{auroc['ci95_low']:.4f},{auroc['ci95_high']:.4f}]"
            if auroc else "n/a"
        )
        print(f"  {key:8s} n={b['n']:5d}  acc={b.get('accuracy')}  MEL_AUROC={auroc_str}  "
              f"MEL_recall={b.get('mel_recall_at_argmax')}")

    print("\n=== verdict ===")
    print(report["verdict"])

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")
    print("\n=== full report (every number) ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
