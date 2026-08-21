"""POST-HOC transfer of the promoted Derm7pt perception recipe to HAM10000 and BCN20000.

The Derm7pt safety-preserving promotion (research/evidence/derm7pt_v1/
perception_phase2_safety_preserving.json) won with a three-lever diagnosis recipe:

    1. clinical-metadata concatenation onto the frozen DermLIP embedding,
    2. train-CV-selected C for an UNBALANCED logistic regression,
    3. per-class SK/MISC operating-point weights chosen on train/valid only
       (argmax(posterior/weight), AUROC-invariant).

This script asks whether that recipe transfers to the two other frozen partitions the
paper evaluates on: BCN20000 (lesion-level test, 712 lesions) and HAM10000 (both the
E3 80/20 split, 1,595 test images, and the previously-sealed 1,495-lesion partition).

POST-HOC LABELING (read this first):
  * Every number here is a POST-HOC evaluation on a partition that has already been
    read by its registered evaluation. Nothing here is a one-shot confirmatory read.
    The user explicitly authorized the re-read.
  * The ORIGINAL registered results remain the confirmatory numbers:
      - research/evidence/bcn_v1/bcn_test_evaluation.json      (BCN test)
      - research/evidence/ham_v1/ham_evaluation.json           (HAM E3 80/20)
      - research/evidence/sealed_v2/sealed_v2_evaluation.json  (HAM sealed, arm2)
    They are REPRODUCED here as candidate (a) wiring self-checks -- not replaced.
  * Selection discipline is preserved anyway: every C and every operating-point
    weight below is chosen on train / valid / train-side out-of-fold posteriors
    only. The test partitions are scored, never selected on.

CANDIDATES (frozen per partition before the test read, mirroring the Derm7pt
phase-2 candidate-table style):
  (a) visual baseline + operating point: the recorded baseline configuration
      (visual-only DermLIP features, the partition's own recorded CV-selected C),
      whose ARGMAX row must reproduce the recorded artifact -- plus an operating
      point selected for THIS model on the valid surrogate. Isolates the
      thresholds lever.
  (b) promoted recipe: visual + metadata concat, C re-selected by the same
      train-CV procedure on the augmented features, plus an operating point
      selected for THIS model. (b) argmax vs (a) argmax isolates the metadata
      lever; (x) op vs (x) argmax isolates the thresholds lever.

VALID SURROGATES for operating-point selection:
  * BCN: the real frozen VALID split (714 lesions; model fit on train only).
  * HAM E3 / HAM sealed: HAM has no valid partition, so the operating point is
    selected on TRAIN-SIDE OUT-OF-FOLD posteriors (5-fold StratifiedGroupKFold,
    grouped by lesion, the partition's own seed, at the selected C). For the
    sealed lesion-level read the held-out fold images are lesion-pooled BEFORE
    prediction so the OOF unit mirrors the test-time unit exactly. No test
    contact either way.

OPERATING-POINT GRID (mirrored exactly from scripts/dev/derm7pt_p1_sweep.py):
  w_SK, w_MISC in {1.0, 0.7, 0.5, 0.35, 0.25, 0.15}; rule argmax(posterior/weight);
  chosen = max(recall_SK + recall_MISC) subject to balanced accuracy >= argmax
  balanced accuracy - 0.02, on the selection split. The 7-class HAM heads reuse the
  partition/candidate's 5-group pair with group-tied weights (bkl <- w_SK;
  akiec/df/vasc <- w_MISC) -- no additional selection.

TTA: skipped. No flip/TTA feature caches exist for HAM or BCN (only base dermlip /
panderm / biomedclip / monet), and TTA was never part of the promoted DIAGNOSIS
recipe -- it entered only the Derm7pt criteria heads, which have no HAM/BCN labels.
Nothing is encoded fresh here; every feature file is a pre-existing cache.

Cost: zero paid calls; all offline.

Run::
    venv_win/Scripts/python.exe scripts/dev/promoted_recipe_ham_bcn.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from medguard_v2.data.bcn20000 import lesion_level, load_records  # noqa: E402
from medguard_v2.eval.metrics import classification_summary  # noqa: E402
from medguard_v2.eval.statistics import binary_auc, clustered_bootstrap_ci  # noqa: E402
from medguard_v2.eval.taxonomy import assert_labels_match  # noqa: E402
from medguard_v2.perception.clinical_metadata import encode_bcn, encode_ham  # noqa: E402

OUT = REPO / "research/evidence/derm7pt_v1/promoted_recipe_ham_bcn.json"

# ---- frozen references (recorded artifacts reproduced as wiring checks) -----------------
BCN_BASELINE = REPO / "research/evidence/bcn_v1/bcn_test_evaluation.json"
HAM_E3_BASELINE = REPO / "research/evidence/ham_v1/ham_evaluation.json"
SEALED_BASELINE = REPO / "research/evidence/sealed_v2/sealed_v2_evaluation.json"
DERM7PT_RECIPE = REPO / "research/evidence/derm7pt_v1/perception_phase2_safety_preserving.json"

# ---- data locations (all pre-existing caches; nothing encoded fresh) --------------------
BCN_META_CSV = REPO / "research/data/public/bcn20000/bcn_20k_train.csv"
BCN_MANIFEST = REPO / "research/manifests/bcn20000_external_v1.json"
HAM_SPLIT_MANIFEST = REPO / "research/manifests/ham10000_split_v2.json"
HAM_TRAIN_FEATURES = REPO / "research/experiments/derm_perception/v2_tournament/features/dermlip/features.npy"
SEALED_FEATURE_DIR = REPO / "research/experiments/medguard_h2h_v1/features/dermlip_sealed_v1"

# ---- partition seeds: each partition keeps ITS OWN recorded seed ------------------------
BCN_SEED = 20260811     # scripts/dev/build_bcn_test_evaluation.py
E3_SEED = 20260812      # scripts/dev/build_ham_evaluation.py
SEALED_SEED = 20260810  # scripts/sealed/run_sealed_v2.py

C_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
N_BOOT = 2000
GRID_WEIGHTS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15)  # derm7pt_p1_sweep.operating_point_policy
BALACC_TOLERANCE = 0.02

# The registered 5-group mapping (g7 / ham_evaluation / run_sealed_v2): akiec -> MISC.
# NOTE eval_full_v3.py mapped akiec -> SK; its HAM rows are therefore not label-compatible
# with the registered baselines and are cited only as a directional metadata prior.
HAM_TO_GROUP = {
    "melanoma": "MEL", "nevus": "NEV", "bcc": "BCC",
    "bkl": "SK", "akiec": "MISC", "df": "MISC", "vasc": "MISC",
}
SEVEN_TIED_WEIGHT_CLASSES = {"bkl": "SK", "akiec": "MISC", "df": "MISC", "vasc": "MISC"}

FIVE_GROUP_MINORITY = ("SK", "MISC")


def _round4(x) -> float:
    return round(float(x), 4)


# ==========================================================================================
# operating point: selection (valid/OOF side) + application (test side)
# ==========================================================================================

def apply_operating_point(proba: np.ndarray, classes: Sequence[str],
                          weights: Dict[str, float]) -> List[str]:
    """argmax(posterior / per-class weight); weight < 1 boosts that class's rank."""
    class_list = [str(c) for c in classes]
    wvec = np.array([weights.get(c, 1.0) for c in class_list], dtype=np.float64)
    adjusted = np.asarray(proba, dtype=np.float64) / wvec
    return [class_list[i] for i in adjusted.argmax(axis=1)]


def select_operating_point(proba: np.ndarray, classes: Sequence[str],
                           y_sel: Sequence[str], labels: Sequence[str],
                           selection_split: str) -> dict:
    """Mirror derm7pt_p1_sweep.operating_point_policy on the selection posteriors.

    Grid-search w_SK, w_MISC over GRID_WEIGHTS; chosen = max(recall_SK + recall_MISC)
    subject to balanced accuracy >= argmax balanced accuracy - 0.02.
    """
    class_list = [str(c) for c in classes]
    argmax_pred = [class_list[i] for i in np.asarray(proba).argmax(axis=1)]
    base = classification_summary(list(y_sel), argmax_pred, list(labels))
    base_balacc = _round4(base["balanced_accuracy"])

    rows = []
    for w_sk in GRID_WEIGHTS:
        for w_misc in GRID_WEIGHTS:
            weights = {"SK": w_sk, "MISC": w_misc}
            pred = apply_operating_point(proba, classes, weights)
            summary = classification_summary(list(y_sel), pred, list(labels))
            rows.append({
                "w_SK": w_sk, "w_MISC": w_misc,
                "balanced_accuracy": _round4(summary["balanced_accuracy"]),
                "accuracy": _round4(summary["accuracy"]),
                "recall_SK": _round4(summary["per_class"]["SK"]["recall"]),
                "recall_MISC": _round4(summary["per_class"]["MISC"]["recall"]),
            })
    tolerant = [r for r in rows if r["balanced_accuracy"] >= base_balacc - BALACC_TOLERANCE]
    chosen = max(tolerant, key=lambda r: r["recall_SK"] + r["recall_MISC"]) if tolerant else rows[0]
    return {
        "selection_split": selection_split,
        "rule": (
            "argmax(posterior / per_class_weight); weight<1 boosts that class's rank. "
            "Grid-searched w_SK, w_MISC independently over {1.0,0.7,0.5,0.35,0.25,0.15} on the "
            "selection split (posterior re-ranking only, no refitting). Chosen = "
            "max(recall_SK + recall_MISC) subject to balanced_accuracy staying within 0.02 of "
            "the unweighted argmax. Mirrors scripts/dev/derm7pt_p1_sweep.py exactly."
        ),
        "baseline_argmax": {
            "balanced_accuracy": base_balacc,
            "recall_SK": _round4(base["per_class"]["SK"]["recall"]),
            "recall_MISC": _round4(base["per_class"]["MISC"]["recall"]),
        },
        "grid": rows,
        "chosen": chosen,
    }


# ==========================================================================================
# metric surface for one candidate on one test partition
# ==========================================================================================

def _mel_auroc_block(truth_mel: Sequence[bool], scores: Sequence[float],
                     cluster_ids: Sequence[str]) -> dict:
    est = binary_auc(list(truth_mel), list(scores))
    rows = list(zip([bool(t) for t in truth_mel], [float(s) for s in scores]))
    interval = clustered_bootstrap_ci(
        rows, [str(c) for c in cluster_ids],
        lambda sample: binary_auc([r[0] for r in sample], [r[1] for r in sample]),
        n_boot=N_BOOT,
    )
    return {
        "estimate": _round4(est),
        "ci95_low": _round4(interval.low),
        "ci95_high": _round4(interval.high),
        "n_boot": N_BOOT,
        "cluster_resampled_by": "lesion (cluster_id)",
        "bootstrap_seed": "clustered_bootstrap_ci default (20260712), same as the baselines",
    }


def evaluate_candidate(model: LogisticRegression, x_test: np.ndarray, y_test: Sequence[str],
                       labels: Sequence[str], cluster_ids: Sequence[str],
                       op_weights: Dict[str, float], mel_label: str) -> dict:
    """Argmax + operating-point metric blocks, plus the (op-invariant) MEL AUROC."""
    proba = model.predict_proba(x_test)
    classes = [str(c) for c in model.classes_]
    mel_idx = classes.index(mel_label)
    mel_scores = [float(p) for p in proba[:, mel_idx]]
    truth_mel = [t == mel_label for t in y_test]

    argmax_pred = [classes[i] for i in proba.argmax(axis=1)]
    op_pred = apply_operating_point(proba, classes, op_weights)

    # AUROC invariance: the operating point re-ranks decisions only; the MEL posterior
    # column is untouched, so AUROC computed before and after is definitionally equal.
    auroc_argmax = binary_auc(truth_mel, mel_scores)
    auroc_op = binary_auc(truth_mel, mel_scores)
    assert abs(auroc_argmax - auroc_op) < 1e-12

    def blk(pred):
        s = classification_summary(list(y_test), list(pred), list(labels))
        return {
            "accuracy": _round4(s["accuracy"]),
            "balanced_accuracy": _round4(s["balanced_accuracy"]),
            "macro_f1": _round4(s["macro_f1"]),
            "per_class_recall": {k: _round4(v["recall"]) for k, v in s["per_class"].items()},
        }

    return {
        "n": len(y_test),
        "mel_auroc": _mel_auroc_block(truth_mel, mel_scores, cluster_ids),
        "mel_auroc_invariant_to_operating_point": True,
        "argmax": blk(argmax_pred),
        "operating_point": {
            "policy": {"rule": "argmax(posterior/weight)", "weights": dict(op_weights)},
            **blk(op_pred),
        },
    }


# ==========================================================================================
# wiring self-checks
# ==========================================================================================

def _close(got, expected, tol=0.0006) -> bool:
    if isinstance(expected, dict):
        return all(_close(got[k], expected[k], tol) for k in expected)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return got == expected
    return abs(float(got) - float(expected)) <= tol


def run_wiring_checks(checks: Dict[str, dict], context: str) -> None:
    failures = []
    for name, ch in checks.items():
        ok = _close(ch["got"], ch["expected"])
        ch["reproduced"] = bool(ok)
        if not ok:
            failures.append(name)
    if failures:
        raise SystemExit(
            f"WIRING DRIFT ({context}): failed to reproduce frozen references: {failures}. "
            "Refusing to emit an untrustworthy read.\n"
            + json.dumps({k: checks[k] for k in failures}, indent=2, default=str)
        )


# ==========================================================================================
# BCN20000
# ==========================================================================================

def _bcn_meta_lookup() -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    with open(BCN_META_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case_id = row["bcn_filename"].replace(".jpg", "")
            age_raw = (row.get("age_approx") or "").strip()
            lookup[case_id] = {
                "age": float(age_raw) if age_raw else None,
                "sex": (row.get("sex") or "unknown"),
                "site": (row.get("anatom_site_general") or "unknown") or "unknown",
            }
    return lookup


def _bcn_lesion_matrices(records, features) -> Tuple[list, np.ndarray, np.ndarray, List[str]]:
    """Lesion-level visual (mean-pool + re-norm, via the registered lesion_level()) and
    lesion-level metadata (mean of per-image encodings; 11/3570 lesions have a minor
    age disagreement across images, which the mean absorbs deterministically)."""
    reps, pooled = lesion_level(records, features)
    meta_lookup = _bcn_meta_lookup()

    by_lesion: Dict[str, List[np.ndarray]] = {}
    order: List[str] = []
    for rec in records:
        if rec.lesion_id not in by_lesion:
            by_lesion[rec.lesion_id] = []
            order.append(rec.lesion_id)
        m = meta_lookup.get(rec.case_id)
        if m is None:
            raise SystemExit(f"BCN case {rec.case_id} missing from metadata CSV")
        by_lesion[rec.lesion_id].append(encode_bcn(m["age"], m["sex"], m["site"]))
    assert order == [r.lesion_id for r in reps], "lesion order mismatch vs lesion_level()"
    meta = np.stack([np.mean(by_lesion[lid], axis=0) for lid in order])

    y = [r.group for r in reps]
    lids = [r.lesion_id for r in reps]
    return y, pooled, meta, lids


def _select_c_pooled_counts(x: np.ndarray, y: Sequence[str], groups: Sequence[str],
                            seed: int) -> dict:
    """Pooled correct-count CV scoring (build_bcn_test_evaluation / build_ham_evaluation)."""
    y_arr = np.asarray(y)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    per_c: Dict[str, float] = {}
    best_c, best_score = C_GRID[0], -1.0
    for c in C_GRID:
        correct = 0
        for tr, va in folds.split(x, y_arr, groups=np.asarray(groups)):
            model = LogisticRegression(max_iter=5000, C=c, random_state=seed)
            model.fit(x[tr], y_arr[tr])
            correct += int((model.predict(x[va]) == y_arr[va]).sum())
        score = correct / len(y_arr)
        per_c[repr(c)] = round(score, 4)
        if score > best_score:
            best_c, best_score = c, score
    return {"selected_C": best_c, "cv_accuracy": round(best_score, 4),
            "grid_cv_accuracy": per_c, "seed": seed,
            "method": "5-fold StratifiedGroupKFold, pooled correct-count"}


def _select_c_mean_fold(x: np.ndarray, y: Sequence[str], groups: Sequence[str],
                        seed: int) -> dict:
    """Mean-of-fold-accuracy CV scoring (run_sealed_v2.fit_ham_classifier)."""
    per_c: Dict[str, float] = {}
    best_c, best_score = 10.0, 0.0
    for c in C_GRID:
        scores = []
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in cv.split(x, y, groups):
            m = LogisticRegression(max_iter=5000, C=c, random_state=seed)
            m.fit(x[tr_idx], [y[i] for i in tr_idx])
            scores.append(m.score(x[va_idx], [y[i] for i in va_idx]))
        mean_score = float(np.mean(scores))
        per_c[repr(c)] = round(mean_score, 4)
        if mean_score > best_score:
            best_c, best_score = c, mean_score
    return {"selected_C": best_c, "cv_accuracy": round(best_score, 4),
            "grid_cv_accuracy": per_c, "seed": seed,
            "method": "5-fold StratifiedGroupKFold, mean fold accuracy"}


def eval_bcn() -> dict:
    print("=== BCN20000 (lesion-level) ===", flush=True)
    baseline = json.loads(BCN_BASELINE.read_text(encoding="utf-8"))
    arm1 = baseline["arm1_in_domain_classifier"]

    train_records, train_features = load_records(splits=("train",))
    valid_records, valid_features = load_records(splits=("valid",))
    # POST-HOC re-read of the guarded test split -- user-authorized; see module docstring.
    test_records, test_features = load_records(splits=("test",), allow_test=True)

    y_train, x_train, meta_train, lids_train = _bcn_lesion_matrices(train_records, train_features)
    y_valid, x_valid, meta_valid, lids_valid = _bcn_lesion_matrices(valid_records, valid_features)
    y_test, x_test, meta_test, lids_test = _bcn_lesion_matrices(test_records, test_features)

    labels = list(assert_labels_match(y_test, "derm7pt"))
    xa_train, xa_valid, xa_test = x_train, x_valid, x_test
    xb_train = np.hstack([x_train, meta_train])
    xb_valid = np.hstack([x_valid, meta_valid])
    xb_test = np.hstack([x_test, meta_test])

    out: dict = {
        "partition": "bcn20000 lesion-level test (guarded split, POST-HOC re-read)",
        "n_train_lesions": len(y_train), "n_valid_lesions": len(y_valid),
        "n_test_lesions": len(y_test),
        "selection_split": "bcn valid (real frozen valid partition)",
        "metadata": {"features": ["age_approx", "sex", "anatom_site_general"],
                     "dim": int(meta_train.shape[1]),
                     "lesion_encoding": "mean of per-image encode_bcn vectors"},
        "seed": BCN_SEED,
        "baseline_reference": {
            "source": "research/evidence/bcn_v1/bcn_test_evaluation.json arm1",
            "accuracy": arm1["accuracy"], "balanced_accuracy": arm1["balanced_accuracy"],
            "macro_f1": arm1["macro_f1"],
            "mel_auroc": arm1["melanoma"]["auroc"]["estimate"],
            "selected_C": arm1["model_selection"]["selected_C"],
        },
    }

    candidates = {}
    for key, xt, xv, xs in (
        ("candidate_a_visual_plus_op", xa_train, xa_valid, xa_test),
        ("candidate_b_promoted_recipe", xb_train, xb_valid, xb_test),
    ):
        sel = _select_c_pooled_counts(xt, y_train, lids_train, BCN_SEED)
        model = LogisticRegression(max_iter=5000, C=sel["selected_C"], random_state=BCN_SEED)
        model.fit(xt, y_train)
        op_policy = select_operating_point(
            model.predict_proba(xv), model.classes_, y_valid, labels, "bcn valid (714 lesions)")
        weights = {"SK": op_policy["chosen"]["w_SK"], "MISC": op_policy["chosen"]["w_MISC"]}
        test_block = evaluate_candidate(model, xs, y_test, labels, lids_test, weights, "MEL")
        candidates[key] = {
            "config": ("dermlip lesion-pooled (512-d)" if key.startswith("candidate_a")
                       else "dermlip lesion-pooled + metadata9 (521-d)")
                      + f", UNBALANCED LR C={sel['selected_C']} (train-CV), + operating point",
            "c_selection": sel,
            "operating_point_selection": op_policy,
            "test": test_block,
        }
        print(f"  {key}: C={sel['selected_C']} op={weights} "
              f"acc(argmax)={test_block['argmax']['accuracy']} "
              f"acc(op)={test_block['operating_point']['accuracy']} "
              f"AUROC={test_block['mel_auroc']['estimate']}", flush=True)
    out.update(candidates)

    a_arg = candidates["candidate_a_visual_plus_op"]["test"]["argmax"]
    checks = {
        "baseline_argmax_accuracy": {
            "got": a_arg["accuracy"], "expected": arm1["accuracy"],
            "source": "bcn_test_evaluation.json arm1.accuracy"},
        "baseline_argmax_balanced_accuracy": {
            "got": a_arg["balanced_accuracy"], "expected": arm1["balanced_accuracy"],
            "source": "bcn_test_evaluation.json arm1.balanced_accuracy"},
        "baseline_argmax_macro_f1": {
            "got": a_arg["macro_f1"], "expected": arm1["macro_f1"],
            "source": "bcn_test_evaluation.json arm1.macro_f1"},
        "baseline_mel_auroc": {
            "got": candidates["candidate_a_visual_plus_op"]["test"]["mel_auroc"]["estimate"],
            "expected": arm1["melanoma"]["auroc"]["estimate"],
            "source": "bcn_test_evaluation.json arm1.melanoma.auroc"},
        "baseline_selected_C": {
            "got": candidates["candidate_a_visual_plus_op"]["c_selection"]["selected_C"],
            "expected": arm1["model_selection"]["selected_C"],
            "source": "bcn_test_evaluation.json arm1.model_selection"},
        "baseline_per_class_recall": {
            "got": a_arg["per_class_recall"],
            "expected": {k: _round4(v["recall"]) for k, v in arm1["per_class"].items()},
            "source": "bcn_test_evaluation.json arm1.per_class"},
    }
    run_wiring_checks(checks, "BCN candidate (a) argmax vs recorded baseline")
    out["wiring_self_checks"] = checks

    out["v3_metadata_row_note"] = (
        "research/evidence/v3_metadata_evaluation.json reported BCN v3 MEL AUROC 0.8716 "
        "(v2 0.8644) under a DIFFERENT pipeline: image-mean pooling WITHOUT re-normalisation, "
        "fixed C=10, first-image metadata lookup, seed 42. Cited as a directional prior only; "
        "the promoted-recipe number above is the like-for-like comparison against the recorded "
        "baseline (lesion_level() pooling, train-CV C, seed 20260811)."
    )
    return out


# ==========================================================================================
# HAM10000 shared loading
# ==========================================================================================

def _load_ham_build_module():
    """Import scripts/dev/build_ham_evaluation.py (not a package) to reuse its exact
    split-construction code path, so the E3 partition is reproduced by construction."""
    spec = importlib.util.spec_from_file_location(
        "build_ham_evaluation", REPO / "scripts/dev/build_ham_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ham_train_cal() -> Tuple[np.ndarray, List[dict]]:
    features = np.load(HAM_TRAIN_FEATURES)
    manifest = json.loads(HAM_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["split"] in ("train", "calibration")]
    if len(records) != features.shape[0]:
        raise SystemExit(f"HAM feature/record mismatch: {features.shape[0]} vs {len(records)}")
    return features, records


def _ham_meta_matrix(records: Sequence[dict]) -> np.ndarray:
    rows = []
    for r in records:
        m = r.get("metadata", {}) or {}
        rows.append(encode_ham(m.get("age"), m.get("sex", "unknown"),
                               m.get("localization", "unknown")))
    return np.stack(rows)


def _oof_posteriors_images(x: np.ndarray, y: Sequence[str], groups: Sequence[str],
                           C: float, seed: int) -> np.ndarray:
    """Image-level out-of-fold posteriors on the train side (op selection surrogate)."""
    y_arr = np.asarray(y)
    classes_ref: Optional[List[str]] = None
    oof = np.full((len(y_arr), 5), np.nan)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in folds.split(x, y_arr, groups=np.asarray(groups)):
        m = LogisticRegression(max_iter=5000, C=C, random_state=seed)
        m.fit(x[tr], y_arr[tr])
        cl = [str(c) for c in m.classes_]
        if classes_ref is None:
            classes_ref = cl
            oof = np.full((len(y_arr), len(cl)), np.nan)
        assert cl == classes_ref, "fold class order drifted"
        oof[va] = m.predict_proba(x[va])
    assert not np.isnan(oof).any(), "OOF coverage incomplete"
    return oof, classes_ref


def _oof_posteriors_lesion_pooled(x: np.ndarray, y_img: Sequence[str],
                                  groups: Sequence[str], C: float, seed: int
                                  ) -> Tuple[np.ndarray, List[str], List[str]]:
    """Lesion-level OOF posteriors: held-out fold images are mean-pooled by lesion
    (plain mean, mirroring run_sealed_v2.lesion_pool) BEFORE prediction, so the OOF
    unit matches the sealed test-time unit. Grouped folds keep lesions whole."""
    y_arr = np.asarray(y_img)
    groups_arr = np.asarray(groups)
    lesion_rows: List[np.ndarray] = []
    lesion_labels: List[str] = []
    lesion_ids: List[str] = []
    classes_ref: Optional[List[str]] = None
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in folds.split(x, y_arr, groups=groups_arr):
        m = LogisticRegression(max_iter=5000, C=C, random_state=seed)
        m.fit(x[tr], y_arr[tr])
        cl = [str(c) for c in m.classes_]
        if classes_ref is None:
            classes_ref = cl
        assert cl == classes_ref, "fold class order drifted"
        by_lesion: Dict[str, List[int]] = {}
        for i in va:
            by_lesion.setdefault(str(groups_arr[i]), []).append(int(i))
        for lid in sorted(by_lesion):
            idx = by_lesion[lid]
            pooled = np.mean(x[idx], axis=0, keepdims=True)
            lab = {y_arr[i] for i in idx}
            assert len(lab) == 1, f"lesion {lid} label conflict"
            lesion_rows.append(m.predict_proba(pooled)[0])
            lesion_labels.append(next(iter(lab)))
            lesion_ids.append(lid)
    seen = Counter(lesion_ids)
    assert all(v == 1 for v in seen.values()), "a lesion appeared in more than one fold"
    return np.stack(lesion_rows), lesion_labels, classes_ref


def _tied_seven_weights(w_sk: float, w_misc: float) -> Dict[str, float]:
    tied = {}
    for cls, grp in SEVEN_TIED_WEIGHT_CLASSES.items():
        tied[cls] = w_sk if grp == "SK" else w_misc
    return tied


# ==========================================================================================
# HAM E3 (80/20 lesion-level split of train+cal; image-level metrics)
# ==========================================================================================

def eval_ham_e3() -> dict:
    print("\n=== HAM10000 E3 (80/20 split; image-level metrics) ===", flush=True)
    baseline = json.loads(HAM_E3_BASELINE.read_text(encoding="utf-8"))
    mod = _load_ham_build_module()
    assert dict(mod.HAM_TO_GROUP) == HAM_TO_GROUP, "HAM 5-group mapping drifted"

    features, records = mod.load_records()
    cluster_label = mod.lesion_labels(records)
    train_clusters, test_clusters = mod.make_split(cluster_label)

    rec_split = baseline["split"]
    if train_clusters != rec_split["train_cluster_ids"] or test_clusters != rec_split["test_cluster_ids"]:
        raise SystemExit("E3 split regeneration does not match the recorded partition")

    train_set, test_set = set(train_clusters), set(test_clusters)
    train_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in train_set]
    test_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in test_set]

    meta_all = _ham_meta_matrix(records)
    x_train, x_test = features[train_idx], features[test_idx]
    m_train, m_test = meta_all[train_idx], meta_all[test_idx]
    clusters_train = [str(records[i]["cluster_id"]) for i in train_idx]
    clusters_test = [str(records[i]["cluster_id"]) for i in test_idx]
    raw_train = [str(records[i]["label"]) for i in train_idx]
    raw_test = [str(records[i]["label"]) for i in test_idx]
    y5_train = [HAM_TO_GROUP[v] for v in raw_train]
    y5_test = [HAM_TO_GROUP[v] for v in raw_test]

    labels5 = list(assert_labels_match(y5_test, "derm7pt"))
    labels7 = list(assert_labels_match(raw_test, "ham10000"))

    out: dict = {
        "partition": "ham10000 E3 80/20 lesion-level split (train 6429 / test 1595 images), POST-HOC",
        "split_verification": "regenerated via build_ham_evaluation.make_split (seed 20260812); "
                              "asserted equal to the recorded train/test cluster_id lists",
        "selection_split": "train-side out-of-fold posteriors (HAM has no valid partition); "
                           "5-fold StratifiedGroupKFold grouped by cluster_id, seed 20260812",
        "metadata": {"features": ["age", "sex", "localization"], "dim": int(meta_all.shape[1]),
                     "source": "ham10000_split_v2.json record metadata (same as eval_full_v3)"},
        "seed": E3_SEED,
        "baseline_reference": {
            "source": "research/evidence/ham_v1/ham_evaluation.json",
            "five_group": {
                "accuracy": baseline["five_group_taxonomy"]["test_metrics"]["accuracy"],
                "balanced_accuracy": baseline["five_group_taxonomy"]["test_metrics"]["balanced_accuracy"],
                "mel_auroc": baseline["five_group_taxonomy"]["test_metrics"]["melanoma_auroc"]["point_estimate"],
                "selected_C": baseline["five_group_taxonomy"]["selected_C"],
            },
            "seven_class": {
                "accuracy": baseline["seven_class_taxonomy"]["test_metrics"]["accuracy"],
                "balanced_accuracy": baseline["seven_class_taxonomy"]["test_metrics"]["balanced_accuracy"],
                "selected_C": baseline["seven_class_taxonomy"]["selected_C"],
                "mel_auroc": "not recorded in the baseline artifact; candidate (a) below is the comparator",
            },
        },
    }

    for tax, y_tr, y_te, labels, mel_label in (
        ("five_group", y5_train, y5_test, labels5, "MEL"),
        ("seven_class", raw_train, raw_test, labels7, "melanoma"),
    ):
        tax_out = {}
        for key, xt, xs in (
            ("candidate_a_visual_plus_op", x_train, x_test),
            ("candidate_b_promoted_recipe", np.hstack([x_train, m_train]), np.hstack([x_test, m_test])),
        ):
            sel = _select_c_pooled_counts(xt, y_tr, clusters_train, E3_SEED)
            model = LogisticRegression(max_iter=5000, C=sel["selected_C"], random_state=E3_SEED)
            model.fit(xt, y_tr)

            if tax == "five_group":
                oof, oof_classes = _oof_posteriors_images(xt, y_tr, clusters_train,
                                                          sel["selected_C"], E3_SEED)
                op_policy = select_operating_point(
                    oof, oof_classes, y_tr, labels,
                    "E3 train-side OOF (6429 images, image-level)")
                weights = {"SK": op_policy["chosen"]["w_SK"],
                           "MISC": op_policy["chosen"]["w_MISC"]}
                tax_out.setdefault("_five_group_weights", {})[key] = weights
            else:
                w5 = out["five_group"]["_five_group_weights"][key]
                weights = _tied_seven_weights(w5["SK"], w5["MISC"])
                op_policy = {
                    "selection_split": "reused from the same candidate's 5-group selection",
                    "rule": "group-tied weights: bkl <- w_SK, akiec/df/vasc <- w_MISC; "
                            "no additional selection performed for the 7-class head",
                    "chosen": {"w_SK": w5["SK"], "w_MISC": w5["MISC"]},
                }

            test_block = evaluate_candidate(model, xs, y_te, labels, clusters_test,
                                            weights, mel_label)
            tax_out[key] = {
                "config": ("dermlip (512-d)" if key.startswith("candidate_a")
                           else "dermlip + metadata19 (531-d)")
                          + f", UNBALANCED LR C={sel['selected_C']} (train-CV), + operating point",
                "c_selection": sel,
                "operating_point_selection": op_policy,
                "test": test_block,
            }
            print(f"  [{tax}] {key}: C={sel['selected_C']} op={weights} "
                  f"acc(argmax)={test_block['argmax']['accuracy']} "
                  f"acc(op)={test_block['operating_point']['accuracy']} "
                  f"AUROC={test_block['mel_auroc']['estimate']}", flush=True)
        out[tax] = tax_out
    out["five_group"].pop("_five_group_weights", None)

    b5 = baseline["five_group_taxonomy"]
    b7 = baseline["seven_class_taxonomy"]
    a5 = out["five_group"]["candidate_a_visual_plus_op"]
    a7 = out["seven_class"]["candidate_a_visual_plus_op"]
    checks = {
        "e3_5group_accuracy": {"got": a5["test"]["argmax"]["accuracy"],
                               "expected": b5["test_metrics"]["accuracy"],
                               "source": "ham_evaluation.json five_group"},
        "e3_5group_balanced_accuracy": {"got": a5["test"]["argmax"]["balanced_accuracy"],
                                        "expected": b5["test_metrics"]["balanced_accuracy"],
                                        "source": "ham_evaluation.json five_group"},
        "e3_5group_mel_auroc": {"got": a5["test"]["mel_auroc"]["estimate"],
                                "expected": b5["test_metrics"]["melanoma_auroc"]["point_estimate"],
                                "source": "ham_evaluation.json five_group"},
        "e3_5group_selected_C": {"got": a5["c_selection"]["selected_C"],
                                 "expected": b5["selected_C"],
                                 "source": "ham_evaluation.json five_group"},
        "e3_7class_accuracy": {"got": a7["test"]["argmax"]["accuracy"],
                               "expected": b7["test_metrics"]["accuracy"],
                               "source": "ham_evaluation.json seven_class"},
        "e3_7class_selected_C": {"got": a7["c_selection"]["selected_C"],
                                 "expected": b7["selected_C"],
                                 "source": "ham_evaluation.json seven_class"},
    }
    run_wiring_checks(checks, "HAM E3 candidate (a) argmax vs recorded baseline")
    out["wiring_self_checks"] = checks
    return out


# ==========================================================================================
# HAM sealed (1,495 lesions, natural distribution; lesion-level metrics)
# ==========================================================================================

def eval_ham_sealed() -> dict:
    print("\n=== HAM10000 sealed partition (1,495 lesions; POST-HOC re-read) ===", flush=True)
    baseline = json.loads(SEALED_BASELINE.read_text(encoding="utf-8"))
    arm2 = baseline["arm2_ham_baseline"]["classifier_metrics"]

    features_train, records_train = _ham_train_cal()
    meta_train = _ham_meta_matrix(records_train)
    clusters_train = [str(r["cluster_id"]) for r in records_train]
    raw_train = [str(r["label"]) for r in records_train]
    y5_train = [HAM_TO_GROUP[v] for v in raw_train]

    sealed_manifest = json.loads((SEALED_FEATURE_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    sealed_features = np.load(SEALED_FEATURE_DIR / "features.npy")
    case_ids = sealed_manifest["case_ids"]
    assert sealed_features.shape[0] == len(case_ids)

    ham = json.loads(HAM_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    test_by_id = {r["case_id"]: r for r in ham["records"] if r["split"] == "test"}
    test_records = [test_by_id[cid] for cid in case_ids]

    # Lesion pooling mirrors run_sealed_v2.lesion_pool: plain mean, NO re-normalisation.
    by_lesion: Dict[str, List[int]] = {}
    for i, rec in enumerate(test_records):
        by_lesion.setdefault(str(rec["cluster_id"]), []).append(i)
    lesion_order = sorted(by_lesion)
    x_test = np.stack([np.mean(sealed_features[by_lesion[c]], axis=0) for c in lesion_order])
    meta_test_all = _ham_meta_matrix(test_records)
    m_test = np.stack([np.mean(meta_test_all[by_lesion[c]], axis=0) for c in lesion_order])
    raw_test, y5_test = [], []
    for c in lesion_order:
        labs = {str(test_records[i]["label"]) for i in by_lesion[c]}
        assert len(labs) == 1, f"sealed lesion {c} label conflict"
        raw_test.append(next(iter(labs)))
        y5_test.append(HAM_TO_GROUP[raw_test[-1]])

    labels5 = list(assert_labels_match(y5_test, "derm7pt"))
    labels7 = list(assert_labels_match(raw_test, "ham10000"))

    out: dict = {
        "partition": "ham10000 sealed test partition, 1991 images -> 1495 lesions "
                     "(natural distribution)",
        "post_hoc_label": (
            "POST-HOC evaluation on the PREVIOUSLY-SEALED partition (user-authorized re-read). "
            "The original sealed one-shot numbers (sealed_v2_evaluation.json arm2) remain the "
            "registered confirmatory results; candidate (a) argmax reproduces them as a wiring "
            "check, it does not replace them."
        ),
        "selection_split": "train-side lesion-pooled OOF posteriors over the 8024 train+cal "
                           "images (5-fold StratifiedGroupKFold grouped by cluster_id, seed "
                           "20260810); OOF fold images are lesion-pooled before prediction so "
                           "the selection unit mirrors the lesion-level test unit",
        "metadata": {"features": ["age", "sex", "localization"], "dim": int(meta_train.shape[1]),
                     "lesion_encoding": "mean of per-image encode_ham vectors"},
        "seed": SEALED_SEED,
        "baseline_reference": {
            "source": "research/evidence/sealed_v2/sealed_v2_evaluation.json arm2 "
                      "(HAM-trained within-dataset classifier)",
            "five_group": {
                "accuracy": _round4(arm2["accuracy"]),
                "balanced_accuracy": _round4(arm2["balanced_accuracy"]),
                "macro_f1": _round4(arm2["macro_f1"]),
                "mel_auroc": _round4(arm2["melanoma_auroc"]),
                "selected_C": 100.0,
            },
            "seven_class": "no registered sealed 7-class baseline exists; candidate (a) below "
                           "is a fresh post-hoc comparator, labeled as such",
        },
    }

    for tax, y_tr, y_te, labels, mel_label, raw in (
        ("five_group", y5_train, y5_test, labels5, "MEL", False),
        ("seven_class", raw_train, raw_test, labels7, "melanoma", True),
    ):
        tax_out = {}
        for key, xt, xs in (
            ("candidate_a_visual_plus_op", features_train, x_test),
            ("candidate_b_promoted_recipe",
             np.hstack([features_train, meta_train]), np.hstack([x_test, m_test])),
        ):
            sel = _select_c_mean_fold(xt, y_tr, clusters_train, SEALED_SEED)
            model = LogisticRegression(max_iter=5000, C=sel["selected_C"],
                                       random_state=SEALED_SEED)
            model.fit(xt, y_tr)

            if tax == "five_group":
                oof, oof_y, oof_classes = _oof_posteriors_lesion_pooled(
                    xt, y_tr, clusters_train, sel["selected_C"], SEALED_SEED)
                op_policy = select_operating_point(
                    oof, oof_classes, oof_y, labels,
                    "train-side lesion-pooled OOF (5975 lesions)")
                weights = {"SK": op_policy["chosen"]["w_SK"],
                           "MISC": op_policy["chosen"]["w_MISC"]}
                tax_out.setdefault("_five_group_weights", {})[key] = weights
            else:
                w5 = out["five_group"]["_five_group_weights"][key]
                weights = _tied_seven_weights(w5["SK"], w5["MISC"])
                op_policy = {
                    "selection_split": "reused from the same candidate's 5-group selection",
                    "rule": "group-tied weights: bkl <- w_SK, akiec/df/vasc <- w_MISC; "
                            "no additional selection performed for the 7-class head",
                    "chosen": {"w_SK": w5["SK"], "w_MISC": w5["MISC"]},
                }

            test_block = evaluate_candidate(model, xs, y_te, labels, lesion_order,
                                            weights, mel_label)
            tax_out[key] = {
                "config": ("dermlip (512-d)" if key.startswith("candidate_a")
                           else "dermlip + metadata19 (531-d)")
                          + f", UNBALANCED LR C={sel['selected_C']} (train-CV), fit on 8024 "
                            "train+cal images, scored on lesion-pooled sealed features, "
                            "+ operating point",
                "c_selection": sel,
                "operating_point_selection": op_policy,
                "test": test_block,
            }
            print(f"  [{tax}] {key}: C={sel['selected_C']} op={weights} "
                  f"acc(argmax)={test_block['argmax']['accuracy']} "
                  f"acc(op)={test_block['operating_point']['accuracy']} "
                  f"AUROC={test_block['mel_auroc']['estimate']}", flush=True)
        out[tax] = tax_out
    out["five_group"].pop("_five_group_weights", None)

    a5 = out["five_group"]["candidate_a_visual_plus_op"]
    checks = {
        "sealed_5group_accuracy": {"got": a5["test"]["argmax"]["accuracy"],
                                   "expected": _round4(arm2["accuracy"]),
                                   "source": "sealed_v2_evaluation.json arm2"},
        "sealed_5group_balanced_accuracy": {"got": a5["test"]["argmax"]["balanced_accuracy"],
                                            "expected": _round4(arm2["balanced_accuracy"]),
                                            "source": "sealed_v2_evaluation.json arm2"},
        "sealed_5group_mel_auroc": {"got": a5["test"]["mel_auroc"]["estimate"],
                                    "expected": _round4(arm2["melanoma_auroc"]),
                                    "source": "sealed_v2_evaluation.json arm2"},
        "sealed_5group_selected_C": {"got": a5["c_selection"]["selected_C"],
                                     "expected": 100.0,
                                     "source": "sealed_v2_evaluation.json arm2 classifier_fit"},
        "sealed_5group_per_class_recall": {
            "got": a5["test"]["argmax"]["per_class_recall"],
            "expected": {k: _round4(v["recall"]) for k, v in arm2["per_class"].items()},
            "source": "sealed_v2_evaluation.json arm2.per_class"},
    }
    run_wiring_checks(checks, "HAM sealed candidate (a) argmax vs recorded baseline")
    out["wiring_self_checks"] = checks
    return out


# ==========================================================================================
# summary + main
# ==========================================================================================

def _summary_rows(bcn: dict, e3: dict, sealed: dict) -> List[dict]:
    rows = []

    def row(dataset, taxonomy, baseline_acc, baseline_auroc, cand_a, cand_b):
        a, b = cand_a["test"], cand_b["test"]
        return {
            "dataset": dataset, "taxonomy": taxonomy,
            "baseline_argmax_accuracy": baseline_acc,
            "baseline_mel_auroc": baseline_auroc,
            "a_op_accuracy": a["operating_point"]["accuracy"],
            "b_argmax_accuracy": b["argmax"]["accuracy"],
            "b_op_accuracy": b["operating_point"]["accuracy"],
            "b_mel_auroc": b["mel_auroc"]["estimate"],
            "auroc_delta_b_minus_baseline": (
                _round4(b["mel_auroc"]["estimate"] - baseline_auroc)
                if isinstance(baseline_auroc, float) else None),
            "accuracy_delta_b_argmax_minus_baseline": (
                _round4(b["argmax"]["accuracy"] - baseline_acc)
                if isinstance(baseline_acc, float) else None),
            "minority_recall_baseline": {k: cand_a["test"]["argmax"]["per_class_recall"].get(k)
                                         for k in FIVE_GROUP_MINORITY
                                         if k in cand_a["test"]["argmax"]["per_class_recall"]},
            "minority_recall_a_op": {k: a["operating_point"]["per_class_recall"].get(k)
                                     for k in FIVE_GROUP_MINORITY
                                     if k in a["operating_point"]["per_class_recall"]},
            "minority_recall_b_op": {k: b["operating_point"]["per_class_recall"].get(k)
                                     for k in FIVE_GROUP_MINORITY
                                     if k in b["operating_point"]["per_class_recall"]},
            "mel_recall_b_argmax_vs_op": [
                b["argmax"]["per_class_recall"].get("MEL", b["argmax"]["per_class_recall"].get("melanoma")),
                b["operating_point"]["per_class_recall"].get("MEL", b["operating_point"]["per_class_recall"].get("melanoma")),
            ],
        }

    rows.append(row("bcn20000_test", "five_group",
                    bcn["baseline_reference"]["accuracy"], bcn["baseline_reference"]["mel_auroc"],
                    bcn["candidate_a_visual_plus_op"], bcn["candidate_b_promoted_recipe"]))
    rows.append(row("ham10000_e3", "five_group",
                    e3["baseline_reference"]["five_group"]["accuracy"],
                    e3["baseline_reference"]["five_group"]["mel_auroc"],
                    e3["five_group"]["candidate_a_visual_plus_op"],
                    e3["five_group"]["candidate_b_promoted_recipe"]))
    rows.append(row("ham10000_e3", "seven_class",
                    e3["baseline_reference"]["seven_class"]["accuracy"],
                    e3["seven_class"]["candidate_a_visual_plus_op"]["test"]["mel_auroc"]["estimate"],
                    e3["seven_class"]["candidate_a_visual_plus_op"],
                    e3["seven_class"]["candidate_b_promoted_recipe"]))
    rows.append(row("ham10000_sealed", "five_group",
                    sealed["baseline_reference"]["five_group"]["accuracy"],
                    sealed["baseline_reference"]["five_group"]["mel_auroc"],
                    sealed["five_group"]["candidate_a_visual_plus_op"],
                    sealed["five_group"]["candidate_b_promoted_recipe"]))
    rows.append(row("ham10000_sealed", "seven_class",
                    sealed["seven_class"]["candidate_a_visual_plus_op"]["test"]["argmax"]["accuracy"],
                    sealed["seven_class"]["candidate_a_visual_plus_op"]["test"]["mel_auroc"]["estimate"],
                    sealed["seven_class"]["candidate_a_visual_plus_op"],
                    sealed["seven_class"]["candidate_b_promoted_recipe"]))
    return rows


def main() -> None:
    started = time.perf_counter()
    recipe = json.loads(DERM7PT_RECIPE.read_text(encoding="utf-8"))

    bcn = eval_bcn()
    e3 = eval_ham_e3()
    sealed = eval_ham_sealed()

    report = {
        "protocol": "promoted-recipe-transfer-ham-bcn-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post_hoc": True,
        "one_shot": False,
        "authorization": (
            "POST-HOC evaluation on previously-read frozen partitions (BCN guarded test, HAM E3 "
            "test side, HAM sealed partition), explicitly authorized by the user. The original "
            "registered evaluations (bcn_test_evaluation.json, ham_evaluation.json, "
            "sealed_v2_evaluation.json) remain the confirmatory results and are reproduced here "
            "as candidate (a) wiring self-checks, not replaced. All hyperparameters (C, operating "
            "points) are selected on train / valid / train-side OOF only; the test partitions are "
            "scored, never selected on."
        ),
        "recipe_under_test": {
            "source": "research/evidence/derm7pt_v1/perception_phase2_safety_preserving.json",
            "derm7pt_result": {
                "diagnosis": recipe["final_config"]["diagnosis"],
                "test_mel_auroc": recipe["final_test_metrics"]["diagnosis_mel_auroc"],
                "test_accuracy_op": recipe["final_test_metrics"]["diagnosis_accuracy_operating_point"],
            },
            "levers": ["metadata concat", "train-CV-selected C (unbalanced LR)",
                       "per-class SK/MISC operating point selected on train/valid only"],
            "not_transferred": (
                "clinic view (does not exist for HAM/BCN); TTA features (not cached for HAM/BCN "
                "and not part of the promoted diagnosis recipe -- TTA entered only the Derm7pt "
                "criteria heads, which have no HAM/BCN labels); criteria heads (no HAM/BCN "
                "criterion annotations)."
            ),
        },
        "bcn": bcn,
        "ham_e3": e3,
        "ham_sealed": sealed,
    }
    report["summary_table"] = _summary_rows(bcn, e3, sealed)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    print(f"done in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
