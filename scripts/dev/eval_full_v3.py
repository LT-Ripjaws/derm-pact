"""Full v3 evaluation: metadata-augmented classifier + conformal prediction.

Re-evaluates all three test sets (Derm7pt, BCN, HAM) with clinical metadata
features concatenated to DermLIP embeddings.  Also produces conformal
prediction sets calibrated on each dataset's validation/calibration split.

Outputs a single comprehensive artifact comparing v2 (visual-only) and
v3 (visual + metadata) side by side.

Cost: zero (all offline, no API calls).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.eval.metrics import classification_summary
from medguard_v2.eval.statistics import binary_auc, clustered_bootstrap_ci
from medguard_v2.perception.clinical_metadata import (
    augment_features,
    encode_batch_derm7pt,
    encode_bcn,
    encode_ham,
)
from medguard_v2.perception.conformal import ConformalClassifier

SEED = 42
OUT = REPO / "research" / "evidence" / "v3_metadata_evaluation.json"
GROUP_LABELS = ("BCC", "MEL", "MISC", "NEV", "SK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mel_auroc(clf, x, y, cluster_ids=None):
    proba = clf.predict_proba(x)
    classes = list(clf.classes_)
    mel_idx = classes.index("MEL") if "MEL" in classes else None
    if mel_idx is None:
        return {"estimate": None, "note": "MEL not in classes"}
    mel_proba = proba[:, mel_idx]
    mel_true = [1 if label == "MEL" else 0 for label in y]
    est = binary_auc(mel_true, list(mel_proba))
    rows = list(zip(mel_true, [float(p) for p in mel_proba]))
    cids = [str(c) for c in cluster_ids] if cluster_ids is not None else [str(i) for i in range(len(y))]
    ci = clustered_bootstrap_ci(
        rows, cids,
        metric=lambda pairs: binary_auc([p[0] for p in pairs], [p[1] for p in pairs]),
        n_boot=2000, seed=SEED,
    )
    return {"estimate": round(est, 4), "low": round(ci.low, 4), "high": round(ci.high, 4)}


def _eval_classifier(clf, x, y, label, cluster_ids=None):
    pred = [str(p) for p in clf.predict(x)]
    acc = float(np.mean(np.array(pred) == np.array(y)))
    summary = classification_summary(y, pred, labels=GROUP_LABELS)
    auroc = _mel_auroc(clf, x, y, cluster_ids)
    return {
        "label": label,
        "n": len(y),
        "accuracy": round(acc, 4),
        "melanoma_auroc": auroc,
        "class_distribution": dict(Counter(y)),
        "per_class": summary.get("per_class", {}),
    }


def _eval_conformal(clf, x_cal, y_cal, x_test, y_test, alpha=0.10):
    np.random.seed(SEED)
    conf = ConformalClassifier(clf, clf.classes_)
    cal = conf.calibrate(x_cal, y_cal, alpha=alpha, method="lac")
    sets = conf.predict_sets(x_test, y_test)
    coverage = sum(1 for s in sets if s.contains_true) / len(sets)
    mean_size = sum(s.set_size for s in sets) / len(sets)
    size_dist = {}
    for s in sets:
        size_dist[s.set_size] = size_dist.get(s.set_size, 0) + 1

    class_cov = {}
    for s, yt in zip(sets, y_test):
        if yt not in class_cov:
            class_cov[yt] = {"total": 0, "covered": 0}
        class_cov[yt]["total"] += 1
        class_cov[yt]["covered"] += int(s.contains_true)
    per_class_cov = {
        cls: round(d["covered"] / d["total"], 4)
        for cls, d in sorted(class_cov.items())
    }

    return {
        "alpha": alpha,
        "method": "lac",
        "calibration": cal.__dict__,
        "test_coverage": round(coverage, 4),
        "test_mean_set_size": round(mean_size, 3),
        "test_set_size_distribution": dict(sorted(size_dist.items())),
        "test_per_class_coverage": per_class_cov,
        "test_singleton_rate": round(size_dist.get(1, 0) / len(sets), 4),
    }


# ---------------------------------------------------------------------------
# E1: Derm7pt
# ---------------------------------------------------------------------------

def eval_derm7pt():
    from medguard_v2.data.derm7pt import load_cases
    from medguard_v2.perception.criterion_abstention import load_policy

    print("=== Derm7pt ===")
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    test = [c for c in cases if c.split == "test"]

    features = np.load(REPO / "research/evidence/derm7pt_v1/features/dermlip/features.npz")
    feat_dict = {str(c): v for c, v in zip(features["case_num"], features["features"])}

    x_train = np.stack([feat_dict[str(c.case_num)] for c in train])
    x_valid = np.stack([feat_dict[str(c.case_num)] for c in valid])
    x_test = np.stack([feat_dict[str(c.case_num)] for c in test])

    meta_train = encode_batch_derm7pt(train)
    meta_valid = encode_batch_derm7pt(valid)
    meta_test = encode_batch_derm7pt(test)

    x_train_aug = augment_features(x_train, meta_train)
    x_valid_aug = augment_features(x_valid, meta_valid)
    x_test_aug = augment_features(x_test, meta_test)

    y_train = [c.diagnosis_group for c in train]
    y_valid = [c.diagnosis_group for c in valid]
    y_test = [c.diagnosis_group for c in test]

    # v2 baseline
    clf_v2 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
    clf_v2.fit(x_train, y_train)

    # v3 with metadata
    clf_v3 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
    clf_v3.fit(x_train_aug, y_train)

    result = {
        "dataset": "derm7pt",
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "metadata_features": ["sex", "location", "elevation"],
        "metadata_dim": meta_train.shape[1],
        "v2_visual_only": {
            "valid": _eval_classifier(clf_v2, x_valid, y_valid, "v2_valid"),
            "test": _eval_classifier(clf_v2, x_test, y_test, "v2_test"),
        },
        "v3_with_metadata": {
            "valid": _eval_classifier(clf_v3, x_valid_aug, y_valid, "v3_valid"),
            "test": _eval_classifier(clf_v3, x_test_aug, y_test, "v3_test"),
        },
        "conformal_v2": _eval_conformal(clf_v2, x_valid, y_valid, x_test, y_test),
        "conformal_v3": _eval_conformal(clf_v3, x_valid_aug, y_valid, x_test_aug, y_test),
    }

    # Criteria heads (unchanged — no metadata for criteria)
    policy = load_policy()
    heads = {}
    for name in CRITERION_NAMES:
        h = LogisticRegression(max_iter=5000, C=0.1, random_state=SEED)
        h.fit(x_train, [str(c.criteria[name]) for c in train])
        heads[name] = h

    crit_results = {}
    for name in CRITERION_NAMES:
        vals_test, _ = policy.apply(heads[name], x_test, name)
        gt = [str(c.criteria[name]) for c in test]
        correct = sum(1 for p, g in zip(vals_test, gt) if p == g)
        crit_results[name] = {
            "accuracy": round(correct / len(gt), 4),
            "n_indeterminate": sum(1 for v in vals_test if v == "indeterminate"),
        }
    result["criteria_heads"] = crit_results

    delta = result["v3_with_metadata"]["test"]["accuracy"] - result["v2_visual_only"]["test"]["accuracy"]
    print(f"  v2 test: {result['v2_visual_only']['test']['accuracy']:.4f}")
    print(f"  v3 test: {result['v3_with_metadata']['test']['accuracy']:.4f} (delta: {delta:+.4f})")
    print(f"  conformal v3: coverage={result['conformal_v3']['test_coverage']}, "
          f"mean_size={result['conformal_v3']['test_mean_set_size']}")
    return result


# ---------------------------------------------------------------------------
# E2: BCN
# ---------------------------------------------------------------------------

def eval_bcn():
    from medguard_v2.data.bcn20000 import load_records, lesion_level

    print("\n=== BCN ===")
    train_recs, _ = load_records(splits=("train",))
    test_recs, _ = load_records(splits=("test",), allow_test=True)

    features_path = REPO / "research/experiments/derm_perception/v2_tournament/external_bcn_v1/features/dermlip/features.npy"
    all_features = np.load(features_path)

    meta_lookup = {}
    with open(REPO / "research/data/public/bcn20000/bcn_20k_train.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case_id = row["bcn_filename"].replace(".jpg", "")
            age_raw = row.get("age_approx", "")
            meta_lookup[case_id] = {
                "age": float(age_raw) if age_raw and age_raw.strip() else None,
                "sex": row.get("sex", "unknown") or "unknown",
                "site": row.get("anatom_site_general", "unknown") or "unknown",
            }

    def _build_lesion_data(records, all_feats, split_label):
        lesion_feats = {}
        lesion_labels = {}
        lesion_meta = {}
        for r in records:
            lid = r.lesion_id
            feat = all_feats[r.row]
            if lid not in lesion_feats:
                lesion_feats[lid] = []
                lesion_labels[lid] = r.group
                m = meta_lookup.get(r.case_id, {})
                lesion_meta[lid] = m if m else {"age": None, "sex": "unknown", "site": "unknown"}
            lesion_feats[lid].append(feat)

        lids = sorted(lesion_feats.keys())
        x = np.stack([np.mean(lesion_feats[lid], axis=0) for lid in lids])
        y = [lesion_labels[lid] for lid in lids]
        meta = np.stack([
            encode_bcn(
                lesion_meta[lid].get("age"),
                lesion_meta[lid].get("sex", "unknown"),
                lesion_meta[lid].get("site", "unknown"),
            )
            for lid in lids
        ])
        return x, y, meta, lids

    x_train, y_train, meta_train, lids_train = _build_lesion_data(train_recs, all_features, "train")
    x_test, y_test, meta_test, lids_test = _build_lesion_data(test_recs, all_features, "test")

    x_train_aug = augment_features(x_train, meta_train)
    x_test_aug = augment_features(x_test, meta_test)

    # In-domain classifiers
    clf_v2 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
    clf_v2.fit(x_train, y_train)

    clf_v3 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
    clf_v3.fit(x_train_aug, y_train)

    result = {
        "dataset": "bcn",
        "n_train_lesions": len(lids_train),
        "n_test_lesions": len(lids_test),
        "metadata_features": ["age_approx", "sex", "anatom_site_general"],
        "metadata_dim": meta_train.shape[1],
        "in_domain_v2": _eval_classifier(clf_v2, x_test, y_test, "bcn_v2", np.array(lids_test)),
        "in_domain_v3": _eval_classifier(clf_v3, x_test_aug, y_test, "bcn_v3", np.array(lids_test)),
        "conformal_v3": _eval_conformal(clf_v3, x_train_aug[-200:], y_train[-200:], x_test_aug, y_test),
    }

    # Cross-validation on BCN train
    print("  Running 5-fold CV on BCN train...")
    y_arr = np.array(y_train)
    lids_arr = np.array(lids_train)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_v2, cv_v3 = [], []
    for tr, va in cv.split(x_train, y_arr, groups=lids_arr):
        m_v2 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
        m_v2.fit(x_train[tr], y_arr[tr])
        cv_v2.append(float(np.mean(m_v2.predict(x_train[va]) == y_arr[va])))

        m_v3 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
        m_v3.fit(x_train_aug[tr], y_arr[tr])
        cv_v3.append(float(np.mean(m_v3.predict(x_train_aug[va]) == y_arr[va])))

    result["cv_5fold_v2"] = {
        "mean": round(float(np.mean(cv_v2)), 4),
        "std": round(float(np.std(cv_v2)), 4),
        "per_fold": [round(v, 4) for v in cv_v2],
    }
    result["cv_5fold_v3"] = {
        "mean": round(float(np.mean(cv_v3)), 4),
        "std": round(float(np.std(cv_v3)), 4),
        "per_fold": [round(v, 4) for v in cv_v3],
    }

    delta = result["in_domain_v3"]["accuracy"] - result["in_domain_v2"]["accuracy"]
    print(f"  v2 in-domain: {result['in_domain_v2']['accuracy']:.4f} (CV: {result['cv_5fold_v2']['mean']:.4f})")
    print(f"  v3 in-domain: {result['in_domain_v3']['accuracy']:.4f} (CV: {result['cv_5fold_v3']['mean']:.4f})")
    return result


# ---------------------------------------------------------------------------
# E3: HAM
# ---------------------------------------------------------------------------

def eval_ham():
    print("\n=== HAM ===")

    manifest = json.loads(
        (REPO / "research/manifests/ham10000_split_v2.json").read_text(encoding="utf-8")
    )
    records = manifest["records"]

    # Train+cal features (8024, 512) and sealed test features (1991, 512) are in separate files.
    train_features = np.load(
        REPO / "research/experiments/derm_perception/v2_tournament/features/dermlip/features.npy"
    )
    test_features = np.load(
        REPO / "research/experiments/medguard_h2h_v1/features/dermlip_sealed_v1/features.npy"
    )
    test_manifest = json.loads(
        (REPO / "research/experiments/medguard_h2h_v1/features/dermlip_sealed_v1/feature_manifest.json")
        .read_text(encoding="utf-8")
    )

    GROUP_MAP = {
        "melanoma": "MEL", "nevus": "NEV", "bcc": "BCC",
        "bkl": "SK", "akiec": "SK", "df": "MISC", "vasc": "MISC",
    }

    # Build train from the records whose split is train/calibration (matches train_features order)
    train_records = [r for r in records if r["split"] in ("train", "calibration")]
    assert len(train_records) == train_features.shape[0], (
        f"train records {len(train_records)} != features {train_features.shape[0]}"
    )
    y_train, meta_train_list = [], []
    for r in train_records:
        y_train.append(GROUP_MAP[r["label"]])
        m = r.get("metadata", {})
        meta_train_list.append(encode_ham(
            m.get("age"), m.get("sex", "unknown"), m.get("localization", "unknown"),
        ))
    x_train = train_features
    meta_train = np.stack(meta_train_list)

    # Build test by joining sealed features with manifest records via case_id
    test_case_ids = test_manifest["case_ids"]
    record_by_id = {r["case_id"]: r for r in records}
    y_test, meta_test_list, cids_test = [], [], []
    for case_id in test_case_ids:
        r = record_by_id[case_id]
        y_test.append(GROUP_MAP[r["label"]])
        m = r.get("metadata", {})
        meta_test_list.append(encode_ham(
            m.get("age"), m.get("sex", "unknown"), m.get("localization", "unknown"),
        ))
        cids_test.append(r["cluster_id"])
    x_test = test_features
    meta_test = np.stack(meta_test_list)

    x_train_aug = augment_features(x_train, meta_train)
    x_test_aug = augment_features(x_test, meta_test)

    clf_v2 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
    clf_v2.fit(x_train, y_train)

    clf_v3 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
    clf_v3.fit(x_train_aug, y_train)

    # Use last 500 train as calibration for conformal
    cal_n = 500
    x_cal = x_train_aug[-cal_n:]
    y_cal = y_train[-cal_n:]

    result = {
        "dataset": "ham10000",
        "n_train": len(train_records),
        "n_test": len(test_case_ids),
        "metadata_features": ["age", "sex", "localization"],
        "metadata_dim": meta_train.shape[1],
        "v2_visual_only": _eval_classifier(clf_v2, x_test, y_test, "ham_v2", np.array(cids_test)),
        "v3_with_metadata": _eval_classifier(clf_v3, x_test_aug, y_test, "ham_v3", np.array(cids_test)),
        "conformal_v3": _eval_conformal(clf_v3, x_cal, y_cal, x_test_aug, y_test),
    }

    # Cross-validation on train+cal (internal validation)
    print("  Running 5-fold CV on train+cal...")
    y_arr = np.array(y_train)
    cids_arr = np.array([r["cluster_id"] for r in train_records])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_v2, cv_v3 = [], []
    for tr, va in cv.split(x_train, y_arr, groups=cids_arr):
        m_v2 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
        m_v2.fit(x_train[tr], y_arr[tr])
        cv_v2.append(float(np.mean(m_v2.predict(x_train[va]) == y_arr[va])))

        m_v3 = LogisticRegression(max_iter=5000, C=10.0, random_state=SEED)
        m_v3.fit(x_train_aug[tr], y_arr[tr])
        cv_v3.append(float(np.mean(m_v3.predict(x_train_aug[va]) == y_arr[va])))

    result["cv_5fold_v2"] = {
        "mean": round(float(np.mean(cv_v2)), 4),
        "std": round(float(np.std(cv_v2)), 4),
        "per_fold": [round(v, 4) for v in cv_v2],
    }
    result["cv_5fold_v3"] = {
        "mean": round(float(np.mean(cv_v3)), 4),
        "std": round(float(np.std(cv_v3)), 4),
        "per_fold": [round(v, 4) for v in cv_v3],
    }

    delta = result["v3_with_metadata"]["accuracy"] - result["v2_visual_only"]["accuracy"]
    print(f"  v2: {result['v2_visual_only']['accuracy']:.4f} (CV: {result['cv_5fold_v2']['mean']:.4f})")
    print(f"  v3: {result['v3_with_metadata']['accuracy']:.4f} (CV: {result['cv_5fold_v3']['mean']:.4f})")
    print(f"  v2 AUROC: {result['v2_visual_only']['melanoma_auroc']['estimate']}")
    print(f"  v3 AUROC: {result['v3_with_metadata']['melanoma_auroc']['estimate']}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {
        "protocol": "v3_metadata_augmented_evaluation",
        "seed": SEED,
        "description": (
            "Full evaluation comparing v2 (DermLIP visual features only) vs "
            "v3 (DermLIP + clinical metadata: sex, age, anatomical site). "
            "Same frozen backbone, same logistic regression classifier, same test sets."
        ),
    }

    results["derm7pt"] = eval_derm7pt()
    results["bcn"] = eval_bcn()
    results["ham"] = eval_ham()

    # Summary table
    summary = []
    for ds in ["derm7pt", "bcn", "ham"]:
        d = results[ds]
        if ds == "derm7pt":
            v2_acc = d["v2_visual_only"]["test"]["accuracy"]
            v3_acc = d["v3_with_metadata"]["test"]["accuracy"]
            v2_auroc = d["v2_visual_only"]["test"]["melanoma_auroc"]["estimate"]
            v3_auroc = d["v3_with_metadata"]["test"]["melanoma_auroc"]["estimate"]
            n = d["n_test"]
        elif ds == "bcn":
            v2_acc = d["in_domain_v2"]["accuracy"]
            v3_acc = d["in_domain_v3"]["accuracy"]
            v2_auroc = d["in_domain_v2"]["melanoma_auroc"]["estimate"]
            v3_auroc = d["in_domain_v3"]["melanoma_auroc"]["estimate"]
            n = d["n_test_lesions"]
        else:
            v2_acc = d["v2_visual_only"]["accuracy"]
            v3_acc = d["v3_with_metadata"]["accuracy"]
            v2_auroc = d["v2_visual_only"]["melanoma_auroc"]["estimate"]
            v3_auroc = d["v3_with_metadata"]["melanoma_auroc"]["estimate"]
            n = d["n_test"]

        summary.append({
            "dataset": ds,
            "n_test": n,
            "v2_accuracy": v2_acc,
            "v3_accuracy": v3_acc,
            "accuracy_delta": round(v3_acc - v2_acc, 4),
            "v2_mel_auroc": v2_auroc,
            "v3_mel_auroc": v3_auroc,
            "auroc_delta": round((v3_auroc or 0) - (v2_auroc or 0), 4) if v3_auroc and v2_auroc else None,
        })

    results["summary"] = summary

    total_n = sum(s["n_test"] for s in summary)
    print(f"\n=== SUMMARY (total {total_n} test cases) ===")
    for s in summary:
        print(f"  {s['dataset']:10s}  n={s['n_test']:5d}  "
              f"v2={s['v2_accuracy']:.4f}  v3={s['v3_accuracy']:.4f}  "
              f"delta={s['accuracy_delta']:+.4f}  "
              f"AUROC v2={s['v2_mel_auroc']}  v3={s['v3_mel_auroc']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
