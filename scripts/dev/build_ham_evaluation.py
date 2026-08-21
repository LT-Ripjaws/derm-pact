"""80/20 in-distribution HAM10000 evaluation at scale (lesion-level split).

G7 (``research/evidence/derm7pt_v1/g7_ham_scaled_classifier.json``) asks whether HAM10000's
8,024 train+calibration images help the diagnosis classifier on a CROSS-DATASET transfer target
(Derm7pt valid -- a different clinic, different camera, only 76 lesions). That is the harder,
more honest generalization question, but it is also a narrow one: G7's *within-HAM* signal is a
single number -- a 5-fold lesion-grouped CV accuracy of 0.8443 -- with no confusion matrix, no
melanoma operating point, no calibration curve, and no test set actually held out from selection.

This script asks the complementary, simpler question: how good is the DermLIP-encoded perception
backbone at its OWN task, in distribution, at scale, with a real held-out test set? It builds a
fresh 80/20 train/test split of the same 8,024 HAM10000 train+calibration images -- stratified by
5-group label, done at the LESION (cluster_id) level so no lesion's images cross the split -- fits
a classifier on the 80% train side (C chosen by 5-fold lesion-grouped CV, same grid as G7), and
reports the full metric surface on the untouched 20% test side: confusion matrix, per-class
P/R/F1, MCC, Cohen's kappa, melanoma AUROC with a clustered bootstrap CI, sensitivity at fixed
specificity, and calibration (ECE/AURC). It also trains and evaluates a second classifier on the
original 7-class HAM10000 labels, to compare against DermAgent's reported 61.83% accuracy on a
642-image HAM subset.

Caveat, stated once here rather than buried: this is an IN-DISTRIBUTION held-out split of ONE
dataset. It measures whether the backbone discriminates HAM10000 images it wasn't trained on, not
whether it generalizes across clinics or cameras -- that is what G7's Derm7pt-valid transfer
number is for. The two artifacts answer different questions and neither substitutes for the
other.

HAM10000's sealed test partition is never touched: ``HAM_FEATURES`` (imported from G7) is the
same train+calibration-only encoding G7 uses (``test_records_loaded: 0`` in its own feature
manifest).

Usage:
    python scripts/dev/build_ham_evaluation.py [--write] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from medguard_v2.eval.gates.g7_ham_scaled_classifier import (  # noqa: E402
    HAM_FEATURES,
    HAM_SPLIT,
    HAM_TO_GROUP,
)
from medguard_v2.eval.metrics import (  # noqa: E402
    aurc,
    classification_summary,
    ece,
    sensitivity_at_fixed_specificity,
)
from medguard_v2.eval.statistics import binary_auc, clustered_bootstrap_ci  # noqa: E402
from medguard_v2.eval.taxonomy import assert_labels_match  # noqa: E402

OUT = REPO / "research/evidence/ham_v1/ham_evaluation.json"

SEED = 20260812  # date of authorship; frozen here for reproducibility (see split.seed below)
TEST_FRACTION = 0.20
C_GRID = (0.1, 1.0, 10.0, 100.0)  # same grid as G7, for comparability
DEFAULT_N_BOOT = 2000

# research/evidence/derm7pt_v1/g7_ham_scaled_classifier.json -> sanity.ham_lesion_grouped_cv_accuracy
G7_REFERENCE_CV_ACCURACY = 0.8443
G7_REFERENCE_TOLERANCE = 0.05  # a sanity band, not a hard gate -- see run()

DERMAGENT_HAM_ACCURACY = 0.6183
DERMAGENT_HAM_N = 642


def load_records() -> Tuple[np.ndarray, List[dict]]:
    """Features and their manifest records, in the manifest's own record order.

    Same loading contract as G7's ``load_ham`` (same two files, same split filter), reimplemented
    rather than called because this script also needs the raw 7-class label and cluster_id per
    record, which G7's version discards after mapping to the 5-group label.
    """
    features = np.load(HAM_FEATURES)
    manifest = json.loads(HAM_SPLIT.read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["split"] in ("train", "calibration")]
    if len(records) != features.shape[0]:
        raise RuntimeError(
            f"feature/record mismatch: {features.shape[0]} vectors vs {len(records)} records. "
            "The feature file is ordered by the train+calibration manifest; if that ordering "
            "changed, every label below is wrong."
        )
    unknown = {r["label"] for r in records} - set(HAM_TO_GROUP)
    if unknown:
        raise RuntimeError(f"unmapped HAM labels: {sorted(unknown)}")
    return features, records


def lesion_labels(records: Sequence[dict]) -> Dict[str, str]:
    """cluster_id -> raw 7-class HAM label, asserting every lesion's images agree.

    The task brief states this was already verified in G7; re-checking it here is cheap (one pass
    over 8,024 records) and turns a silent stratification bug -- a lesion whose images disagree
    would get an arbitrary label depending on record order -- into a loud one.
    """
    cluster_label: Dict[str, str] = {}
    for r in records:
        cid = str(r["cluster_id"])
        label = str(r["label"])
        prior = cluster_label.get(cid)
        if prior is not None and prior != label:
            raise RuntimeError(
                f"cluster {cid} has inconsistent labels ({prior!r} vs {label!r}); "
                "the lesion-level split assumes exactly one label per lesion"
            )
        cluster_label[cid] = label
    return cluster_label


def make_split(cluster_label: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Stratified 80/20 split of lesion cluster_ids by 5-group label."""
    clusters = sorted(cluster_label)  # deterministic pre-shuffle order
    groups = [HAM_TO_GROUP[cluster_label[c]] for c in clusters]
    train_clusters, test_clusters = train_test_split(
        clusters, test_size=TEST_FRACTION, random_state=SEED, stratify=groups,
    )
    return sorted(train_clusters), sorted(test_clusters)


def _select_c(x: np.ndarray, y: Sequence[str], groups: Sequence[str], *, seed: int) -> float:
    """Choose C by 5-fold lesion-grouped CV -- same procedure and grid as G7's ``_select_c``.

    Reimplemented rather than imported: G7's version reads ``SEED``/``C_GRID`` as module globals
    of the module it is *defined* in, so importing it would silently run this script's model
    selection under G7's seed (20260811) instead of this script's registered one (20260812).
    Taking ``seed`` as an explicit keyword argument avoids that class of bug outright.
    """
    y_arr = np.asarray(y)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    best_c, best = C_GRID[0], -1.0
    for c in C_GRID:
        correct = 0
        for tr, va in folds.split(x, y_arr, groups=np.asarray(groups)):
            model = LogisticRegression(max_iter=5000, C=c, random_state=seed)
            model.fit(x[tr], y_arr[tr])
            correct += int((model.predict(x[va]) == y_arr[va]).sum())
        score = correct / len(y_arr)
        if score > best:
            best_c, best = c, score
    return best_c


def _round_per_class(per_class: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        cls: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        for cls, row in per_class.items()
    }


def _evaluate_5group(
    model: LogisticRegression,
    x_test: np.ndarray,
    y_test: Sequence[str],
    clusters_test: Sequence[str],
    *,
    n_boot: int,
) -> Dict:
    """Full 5-group metric surface on the test set.

    Melanoma AUROC and sensitivity-at-specificity are melanoma-vs-rest: the 5-group MISC bucket
    mixes a malignant class (akiec) with two benign ones (df, vasc), so an "actionable" binary
    framing would be leaky at the MISC boundary. MEL is a clean 1:1 image of the raw ``melanoma``
    class, so it is the only 5-group class an operating-point analysis is unambiguous for.
    """
    labels = assert_labels_match(y_test, "derm7pt")
    predicted = [str(p) for p in model.predict(x_test)]
    summary = classification_summary(list(y_test), predicted, list(labels))

    if "MEL" not in model.classes_:
        raise RuntimeError("MEL is not among the trained model's classes -- check the split/label join")
    mel_index = list(model.classes_).index("MEL")
    proba = model.predict_proba(x_test)
    mel_scores = [float(s) for s in proba[:, mel_index]]
    truth_mel = [t == "MEL" for t in y_test]

    mel_auc_point = binary_auc(truth_mel, mel_scores)
    rows = list(zip(truth_mel, mel_scores))

    def _auc_metric(sample_rows: Sequence[Tuple[bool, float]]) -> float:
        return binary_auc([r[0] for r in sample_rows], [r[1] for r in sample_rows])

    ci = clustered_bootstrap_ci(
        rows, list(clusters_test), _auc_metric, n_boot=n_boot, confidence=0.95, seed=SEED,
    )

    conf = [float(c) for c in proba.max(axis=1)]
    correct = [p == t for p, t in zip(predicted, y_test)]

    return {
        "n_test": summary["n"],
        "labels": summary["labels"],
        "confusion_matrix": summary["confusion_matrix"],
        "row_normalized_confusion_matrix": summary["row_normalized_confusion_matrix"],
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "macro_f1": round(summary["macro_f1"], 4),
        "weighted_f1": round(summary["weighted_f1"], 4),
        "mcc": round(summary["mcc"], 4),
        "cohen_kappa": round(summary["cohen_kappa"], 4),
        "per_class": _round_per_class(summary["per_class"]),
        "melanoma_auroc": {
            "point_estimate": round(mel_auc_point, 4),
            "ci_low": round(ci.low, 4),
            "ci_high": round(ci.high, 4),
            "confidence": 0.95,
            "n_boot": n_boot,
            "cluster_resampled_by": "cluster_id",
            "seed": SEED,
        },
        "sensitivity_at_specificity": {
            f"{target:.2f}": sensitivity_at_fixed_specificity(truth_mel, mel_scores, target)
            for target in (0.90, 0.95)
        },
        "calibration": {
            "ece": ece(conf, correct, n_bins=10),
            "aurc": aurc(conf, correct),
        },
    }


def _evaluate_7class(model: LogisticRegression, x_test: np.ndarray, y_test: Sequence[str]) -> Dict:
    """7-class metric surface on the original HAM10000 labels, for the DermAgent comparison."""
    labels = assert_labels_match(y_test, "ham10000")
    predicted = [str(p) for p in model.predict(x_test)]
    summary = classification_summary(list(y_test), predicted, list(labels))
    return {
        "n_test": summary["n"],
        "labels": summary["labels"],
        "confusion_matrix": summary["confusion_matrix"],
        "row_normalized_confusion_matrix": summary["row_normalized_confusion_matrix"],
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "macro_f1": round(summary["macro_f1"], 4),
        "mcc": round(summary["mcc"], 4),
        "cohen_kappa": round(summary["cohen_kappa"], 4),
        "per_class": _round_per_class(summary["per_class"]),
    }


def run(n_boot: int = DEFAULT_N_BOOT) -> Dict:
    features, records = load_records()
    cluster_label = lesion_labels(records)  # cluster_id -> raw 7-class label
    train_clusters, test_clusters = make_split(cluster_label)

    train_set, test_set = set(train_clusters), set(test_clusters)
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(f"{len(overlap)} cluster_ids leaked across train/test: {sorted(overlap)[:5]}")
    if train_set | test_set != set(cluster_label):
        raise RuntimeError("split does not cover every lesion cluster_id exactly once")

    train_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in train_set]
    test_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in test_set]
    if len(train_idx) + len(test_idx) != len(records):
        raise RuntimeError("image-level split does not partition every record exactly once")

    x_train, x_test = features[train_idx], features[test_idx]
    clusters_train = [str(records[i]["cluster_id"]) for i in train_idx]
    clusters_test = [str(records[i]["cluster_id"]) for i in test_idx]
    raw_train = [str(records[i]["label"]) for i in train_idx]
    raw_test = [str(records[i]["label"]) for i in test_idx]
    y5_train = [HAM_TO_GROUP[label] for label in raw_train]
    y5_test = [HAM_TO_GROUP[label] for label in raw_test]

    # --- 5-group classifier ---------------------------------------------------------
    c5 = _select_c(x_train, y5_train, clusters_train, seed=SEED)
    model5 = LogisticRegression(max_iter=5000, C=c5, random_state=SEED)
    model5.fit(x_train, y5_train)
    five_group_metrics = _evaluate_5group(model5, x_test, y5_test, clusters_test, n_boot=n_boot)

    # --- 7-class classifier (DermAgent comparison) ------------------------------------
    c7 = _select_c(x_train, raw_train, clusters_train, seed=SEED)
    model7 = LogisticRegression(max_iter=5000, C=c7, random_state=SEED)
    model7.fit(x_train, raw_train)
    seven_class_metrics = _evaluate_7class(model7, x_test, raw_test)

    # --- split statistics -------------------------------------------------------------
    five_group_by_cluster = {c: HAM_TO_GROUP[cluster_label[c]] for c in cluster_label}
    split_stats = {
        "method": "stratified 80/20 by 5-group label, at lesion (cluster_id) level",
        "test_fraction": TEST_FRACTION,
        "seed": SEED,
        "n_clusters_total": len(cluster_label),
        "n_clusters_train": len(train_clusters),
        "n_clusters_test": len(test_clusters),
        "n_images_train": len(train_idx),
        "n_images_test": len(test_idx),
        "per_class_counts": {
            "five_group": {
                "clusters_train": dict(Counter(five_group_by_cluster[c] for c in train_clusters)),
                "clusters_test": dict(Counter(five_group_by_cluster[c] for c in test_clusters)),
                "images_train": dict(Counter(y5_train)),
                "images_test": dict(Counter(y5_test)),
            },
            "seven_class": {
                "images_train": dict(Counter(raw_train)),
                "images_test": dict(Counter(raw_test)),
            },
        },
        # Reproducibility: the full lesion-level assignment. Image membership is a deterministic
        # function of (cluster_id, research/manifests/ham10000_split_v2.json), so only the cluster
        # ids -- not every image/case id -- need to be recorded to reconstruct this split exactly.
        "train_cluster_ids": train_clusters,
        "test_cluster_ids": test_clusters,
    }

    # --- sanity checks ------------------------------------------------------------------
    # The first two are structural invariants and would already have raised above if violated;
    # they are recorded here as booleans so the artifact is self-certifying without re-deriving
    # them. The third is a MODEL-PERFORMANCE expectation, not a data-integrity invariant -- a
    # different 80/20 draw legitimately lands a few points away from G7's 5-fold CV estimate, so
    # it is reported with a tolerance band and a loud stderr warning rather than raising, which
    # would destroy an otherwise-valid, inspectable evaluation over a difference of degree.
    accuracy_gap = round(five_group_metrics["accuracy"] - G7_REFERENCE_CV_ACCURACY, 4)
    within_tolerance = abs(accuracy_gap) <= G7_REFERENCE_TOLERANCE
    sanity = {
        "no_cluster_leakage": True,
        "feature_record_count_match_8024": features.shape[0] == len(records) == 8024,
        "five_group_accuracy_near_g7_internal_cv": {
            "ours_80_20_test_accuracy": five_group_metrics["accuracy"],
            "g7_reference_5fold_cv_accuracy": G7_REFERENCE_CV_ACCURACY,
            "g7_reference_source": (
                "research/evidence/derm7pt_v1/g7_ham_scaled_classifier.json"
                ":sanity.ham_lesion_grouped_cv_accuracy"
            ),
            "abs_diff": abs(accuracy_gap),
            "tolerance": G7_REFERENCE_TOLERANCE,
            "within_tolerance": within_tolerance,
            "note": "Sanity expectation, not a hard gate -- see comment in run().",
        },
    }
    if not sanity["feature_record_count_match_8024"]:
        raise RuntimeError(
            f"feature/record count is {features.shape[0]}/{len(records)}, expected 8,024 -- aborting"
        )
    if not within_tolerance:
        print(
            f"WARNING: 5-group test accuracy {five_group_metrics['accuracy']} is more than "
            f"{G7_REFERENCE_TOLERANCE} away from G7's internal CV accuracy "
            f"{G7_REFERENCE_CV_ACCURACY} (diff={accuracy_gap}); inspect before trusting this run.",
            file=sys.stderr,
        )

    dermagent_comparison = {
        "dermagent_ham_accuracy": DERMAGENT_HAM_ACCURACY,
        "dermagent_ham_n": DERMAGENT_HAM_N,
        "dermagent_source_note": (
            "Externally reported figure (task brief), not independently reproduced here: "
            "DermAgent's own published HAM10000 accuracy on a 642-image subset."
        ),
        "our_7class_accuracy": seven_class_metrics["accuracy"],
        "our_7class_n_test": seven_class_metrics["n_test"],
        "our_5group_accuracy": five_group_metrics["accuracy"],
        "our_5group_n_test": five_group_metrics["n_test"],
        "accuracy_diff_7class_vs_dermagent": round(
            seven_class_metrics["accuracy"] - DERMAGENT_HAM_ACCURACY, 4
        ),
        "caveat": (
            "Not a controlled comparison: different test set, different size "
            f"({seven_class_metrics['n_test']} vs {DERMAGENT_HAM_N}), unknown DermAgent "
            "train/test protocol, and a linear probe on frozen DermLIP features here versus "
            "whatever DermAgent's own architecture is."
        ),
    }

    return {
        "schema_version": 1,
        "analysis_id": "ham_evaluation_80_20_v1",
        "purpose": (
            "In-distribution 80/20 lesion-level HAM10000 evaluation at scale: validates the "
            "DermLIP perception backbone's discriminative power on a real held-out test set, "
            "distinct from G7's cross-dataset Derm7pt-valid transfer question."
        ),
        "boundaries": {
            "ham_features": str(HAM_FEATURES.relative_to(REPO)).replace("\\", "/"),
            "ham_split_manifest": str(HAM_SPLIT.relative_to(REPO)).replace("\\", "/"),
            "ham_records": len(records),
            "included_manifest_splits": ["train", "calibration"],
            "sealed_ham_test_partition_excluded": True,
            "cross_dataset_transfer_evidence": (
                "research/evidence/derm7pt_v1/g7_ham_scaled_classifier.json (G7 -- not this script)"
            ),
            "paid_calls": 0,
        },
        "label_mapping_5group": HAM_TO_GROUP,
        "split": split_stats,
        "five_group_taxonomy": {
            "selected_C": c5,
            "c_grid": list(C_GRID),
            "c_selection": "5-fold StratifiedGroupKFold CV on train only, grouped by cluster_id",
            "test_metrics": five_group_metrics,
        },
        "seven_class_taxonomy": {
            "selected_C": c7,
            "c_grid": list(C_GRID),
            "c_selection": "5-fold StratifiedGroupKFold CV on train only, grouped by cluster_id",
            "test_metrics": seven_class_metrics,
        },
        "dermagent_comparison": dermagent_comparison,
        "sanity_checks": sanity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--write", action="store_true", help="persist the report to research/evidence/ham_v1/")
    parser.add_argument(
        "--n-boot", type=int, default=DEFAULT_N_BOOT,
        help=f"bootstrap resamples for the melanoma AUROC CI (default {DEFAULT_N_BOOT})",
    )
    args = parser.parse_args()

    report = run(n_boot=args.n_boot)
    text = json.dumps(report, indent=2)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(text)


if __name__ == "__main__":
    main()
