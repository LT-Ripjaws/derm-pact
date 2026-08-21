"""V2 pipeline evaluation on the sealed HAM10000 test partition.

Runs the full v2 deterministic pipeline (classifier + criterion heads + HOST
aggregation + router + guideline binding + contracts) on the 1495 sealed HAM
lesions. No model training touches the sealed partition; the classifier and
criterion heads are fit on Derm7pt train (the v2 system) or HAM train (within-
dataset comparison).

Two arms:
  1. v2-system: Derm7pt-trained classifier + Derm7pt criterion heads (cross-centre)
  2. ham-baseline: HAM-trained classifier, no criteria (within-dataset)

The agent is NOT evaluated here (would cost ~$15-30 for 1495 lesions). Agent
evaluation is done on the Derm7pt dev cohort (E5a/E5b).

Usage:
    python scripts/sealed/run_sealed_v2.py --write [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.metrics import (
    aurc,
    classification_summary,
    ece,
    sensitivity_at_fixed_specificity,
)
from medguard_v2.eval.statistics import binary_auc, clustered_bootstrap_ci
from medguard_v2.evidence.contract import DecisionContract, build_contract, contract_to_dict
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.evidence.router import route
from medguard_v2.perception.criterion_abstention import load_policy

HAM_SPLIT = REPO / "research" / "manifests" / "ham10000_split_v2.json"
SEALED_FEATURES = REPO / "research" / "experiments" / "medguard_h2h_v1" / "features" / "dermlip_sealed_v1"
DERM7PT_FEATURES = REPO / "research" / "evidence" / "derm7pt_v1" / "features" / "dermlip" / "features.npz"
HAM_TRAIN_FEATURES = REPO / "research" / "experiments" / "derm_perception" / "v2_tournament" / "features" / "dermlip" / "features.npy"
OUT = REPO / "research" / "evidence" / "sealed_v2" / "sealed_v2_evaluation.json"

SEED = 20260810
BASE_C = 10.0
DEFAULT_N_BOOT = 2000

HAM_TO_GROUP = {
    "melanoma": "MEL", "nevus": "NEV", "bcc": "BCC",
    "bkl": "SK", "akiec": "MISC", "df": "MISC", "vasc": "MISC",
}


def load_sealed_data() -> Tuple[np.ndarray, List[dict], List[str]]:
    """Load sealed DermLIP features, HAM test records, and case IDs."""
    manifest = json.loads((SEALED_FEATURES / "feature_manifest.json").read_text())
    features = np.load(SEALED_FEATURES / "features.npy")
    case_ids = manifest["case_ids"]
    assert features.shape[0] == len(case_ids)

    ham = json.loads(HAM_SPLIT.read_text())
    test_records = [r for r in ham["records"] if r["split"] == "test"]
    test_by_id = {r["case_id"]: r for r in test_records}

    records = []
    for cid in case_ids:
        r = test_by_id.get(cid)
        if r is None:
            raise ValueError(f"Sealed case {cid} not found in HAM manifest test split")
        records.append(r)

    return features, records, case_ids


def lesion_pool(
    features: np.ndarray, records: List[dict],
) -> Tuple[np.ndarray, List[dict]]:
    """Mean-pool image features to lesion level."""
    lesions: Dict[str, dict] = {}
    lesion_features: Dict[str, List[np.ndarray]] = {}
    for feat, rec in zip(features, records):
        cid = rec["cluster_id"]
        if cid not in lesions:
            lesions[cid] = {
                "cluster_id": cid,
                "label": rec["label"],
                "group": HAM_TO_GROUP[rec["label"]],
                "n_images": 0,
                "case_ids": [],
            }
            lesion_features[cid] = []
        lesions[cid]["n_images"] += 1
        lesions[cid]["case_ids"].append(rec["case_id"])
        lesion_features[cid].append(feat)

    order = sorted(lesions.keys())
    pooled = np.stack([np.mean(lesion_features[c], axis=0) for c in order])
    pooled_records = [lesions[c] for c in order]
    return pooled, pooled_records


def fit_derm7pt_readers():
    """Fit classifier + criterion heads on Derm7pt train (the v2 system)."""
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    data = np.load(DERM7PT_FEATURES)
    feat_map = {str(c): v for c, v in zip(data["case_num"], data["features"])}
    x = np.stack([feat_map[str(c.case_num)] for c in train])

    clf = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
    clf.fit(x, [c.diagnosis_group for c in train])

    heads = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
        head.fit(x, [str(c.criteria[name]) for c in train])
        heads[name] = head

    return clf, heads, len(train)


def fit_ham_classifier(features: np.ndarray, records: List[dict]):
    """Fit a 5-group classifier on HAM train+cal features."""
    ham_manifest = json.loads(HAM_SPLIT.read_text())
    train_recs = [r for r in ham_manifest["records"] if r["split"] in ("train", "calibration")]
    train_ids = {r["case_id"] for r in train_recs}
    train_labels = {r["case_id"]: HAM_TO_GROUP[r["label"]] for r in train_recs}

    ham_features = np.load(HAM_TRAIN_FEATURES)
    ham_all_manifest = json.loads(HAM_SPLIT.read_text())
    ham_train_cal_recs = [r for r in ham_all_manifest["records"] if r["split"] in ("train", "calibration")]

    x_train = ham_features
    y_train = [HAM_TO_GROUP[r["label"]] for r in ham_train_cal_recs]

    from sklearn.model_selection import StratifiedGroupKFold
    groups = [r["cluster_id"] for r in ham_train_cal_recs]

    best_c, best_score = 10.0, 0.0
    for c in (0.1, 1.0, 10.0, 100.0):
        scores = []
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr_idx, va_idx in cv.split(x_train, y_train, groups):
            m = LogisticRegression(max_iter=5000, C=c, random_state=SEED)
            m.fit(x_train[tr_idx], [y_train[i] for i in tr_idx])
            scores.append(m.score(x_train[va_idx], [y_train[i] for i in va_idx]))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_c, best_score = c, mean_score

    clf = LogisticRegression(max_iter=5000, C=best_c, random_state=SEED)
    clf.fit(x_train, y_train)
    return clf, len(ham_train_cal_recs), best_c


def evaluate_classifier(
    clf, x_test, y_true, cluster_ids, label: str, n_boot: int,
) -> dict:
    """Full classification metrics for a classifier on the sealed set."""
    y_pred = clf.predict(x_test)
    posterior = clf.predict_proba(x_test)
    conf = posterior.max(axis=1)

    summary = classification_summary(y_true, list(y_pred), sorted(set(y_true)))

    mel_true = np.array([1 if y == "MEL" else 0 for y in y_true])
    mel_idx = list(clf.classes_).index("MEL") if "MEL" in clf.classes_ else None
    mel_auroc = None
    mel_auroc_ci = None
    if mel_idx is not None:
        mel_scores = posterior[:, mel_idx]
        mel_auroc = binary_auc(list(mel_true), list(mel_scores))
        rows = list(zip(mel_true.tolist(), mel_scores.tolist()))

        def _auc_metric(sample_rows: Sequence[Tuple[bool, float]]) -> float:
            return binary_auc([r[0] for r in sample_rows], [r[1] for r in sample_rows])

        mel_auroc_ci = clustered_bootstrap_ci(
            rows, list(cluster_ids), _auc_metric, n_boot=n_boot,
        )

    sens_at_spec = None
    if mel_idx is not None:
        sens_at_spec = {
            "at_0.90": sensitivity_at_fixed_specificity(mel_true, posterior[:, mel_idx], 0.90),
            "at_0.95": sensitivity_at_fixed_specificity(mel_true, posterior[:, mel_idx], 0.95),
        }

    cal_ece = ece(conf, np.array([1 if p == t else 0 for p, t in zip(y_pred, y_true)]))
    cal_aurc = aurc(conf, np.array([1 if p == t else 0 for p, t in zip(y_pred, y_true)]))

    return {
        "label": label,
        "n_lesions": len(y_true),
        "accuracy": summary["accuracy"],
        "balanced_accuracy": summary.get("balanced_accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "mcc": summary.get("mcc"),
        "cohen_kappa": summary.get("cohen_kappa"),
        "per_class": summary.get("per_class"),
        "confusion_matrix": summary.get("confusion_matrix"),
        "melanoma_auroc": mel_auroc,
        "melanoma_auroc_ci": mel_auroc_ci,
        "sensitivity_at_specificity": sens_at_spec,
        "calibration": {
            "ece": cal_ece,
            "aurc": cal_aurc,
        },
    }


def build_v2_contracts(
    clf, heads, x_test, records, catalog,
) -> List[DecisionContract]:
    """Build v2 contracts for each sealed lesion."""
    policy = load_policy()
    posterior = clf.predict_proba(x_test)
    predicted = clf.predict(x_test)

    head_predictions: Dict[str, List[str]] = {}
    for name, head in heads.items():
        values, _margins = policy.apply(head, x_test, name)
        head_predictions[name] = values

    contracts = []
    for i, rec in enumerate(records):
        criteria = {name: head_predictions[name][i] for name in CRITERION_NAMES}
        contracts.append(
            build_contract(
                case_id=rec["cluster_id"],
                classifier_diagnosis=str(predicted[i]),
                classifier_confidence=float(posterior[i].max()),
                criteria=criteria,
                catalog=catalog,
                receipt={
                    "split": "sealed_test",
                    "feature_set": "dermlip/ham_sealed",
                    "base_C": BASE_C,
                    "seed": SEED,
                    "fit_split": "derm7pt_train",
                    "abstention_policy": "criterion_abstention_v1",
                },
            )
        )
    return contracts


def contract_metrics(contracts: List[DecisionContract]) -> dict:
    """Decision-level behaviour from v2 contracts."""
    n = len(contracts)
    catalog = GuidelineCatalog.load()

    router_decisions = [
        route(
            classifier_diagnosis=c.classifier_diagnosis,
            classifier_confidence=c.classifier_confidence,
            score=c.score,
            catalog=catalog,
        )
        for c in contracts
    ]

    withheld = [c for c in contracts if not c.safe_to_display]
    conflicts = [c for c in contracts if c.conflict.present]
    escalations = [c for c in contracts if c.conflict.escalate]
    cited = [c for c in contracts if c.guideline.covered]
    undecidable = [c for c in contracts if not c.score.decidable]

    router_states = Counter(d.state.value for d in router_decisions)
    gated_calls = sum(d.retrieval_calls for d in router_decisions)

    trust_count = sum(1 for c in contracts if c.safe_to_display)

    return {
        "n_contracts": n,
        "safe_to_display": trust_count,
        "safe_to_display_rate": round(trust_count / n, 4) if n else None,
        "conflict_rate": round(len(conflicts) / n, 4) if n else None,
        "escalation_rate": round(len(escalations) / n, 4) if n else None,
        "undecidable_rate": round(len(undecidable) / n, 4) if n else None,
        "citation_rate": round(len(cited) / n, 4) if n else None,
        "router_state_distribution": dict(router_states),
        "gated_retrieval_calls": gated_calls,
        "retrieval_savings": round(1.0 - gated_calls / n, 4) if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = parser.parse_args()

    print("Loading sealed features and labels...")
    features, records, case_ids = load_sealed_data()
    print(f"  {features.shape[0]} images, {features.shape[1]}-d features")

    print("Pooling to lesion level...")
    x_lesion, lesion_records = lesion_pool(features, records)
    y_true = [r["group"] for r in lesion_records]
    cluster_ids = [r["cluster_id"] for r in lesion_records]
    print(f"  {len(lesion_records)} lesions")
    print(f"  Distribution: {Counter(y_true)}")

    # --- Arm 1: v2 system (Derm7pt-trained) ---
    print("\nFitting Derm7pt-trained classifier + criterion heads...")
    d7_clf, d7_heads, n_d7_train = fit_derm7pt_readers()
    print(f"  Fit on {n_d7_train} Derm7pt train cases")

    print("Evaluating v2 system on sealed set...")
    arm1_clf_metrics = evaluate_classifier(
        d7_clf, x_lesion, y_true, cluster_ids, "v2_system_derm7pt_trained", args.n_boot,
    )
    print(f"  Accuracy: {arm1_clf_metrics['accuracy']}")
    print(f"  Melanoma AUROC: {arm1_clf_metrics['melanoma_auroc']}")

    catalog = GuidelineCatalog.load()
    print("Building v2 contracts...")
    contracts = build_v2_contracts(d7_clf, d7_heads, x_lesion, lesion_records, catalog)
    arm1_contract_metrics = contract_metrics(contracts)
    print(f"  Safe to display: {arm1_contract_metrics['safe_to_display']}/{arm1_contract_metrics['n_contracts']}")
    print(f"  Citation rate: {arm1_contract_metrics['citation_rate']}")
    print(f"  Retrieval savings: {arm1_contract_metrics['retrieval_savings']}")

    # --- Arm 2: HAM-trained classifier (within-dataset baseline) ---
    print("\nFitting HAM-trained classifier...")
    ham_clf, n_ham_train, ham_c = fit_ham_classifier(features, records)
    print(f"  Fit on {n_ham_train} HAM train+cal images, C={ham_c}")

    print("Evaluating HAM baseline on sealed set...")
    arm2_clf_metrics = evaluate_classifier(
        ham_clf, x_lesion, y_true, cluster_ids, "ham_trained_within_dataset", args.n_boot,
    )
    print(f"  Accuracy: {arm2_clf_metrics['accuracy']}")
    print(f"  Melanoma AUROC: {arm2_clf_metrics['melanoma_auroc']}")

    # --- Assemble output ---
    output = {
        "protocol": "v2_sealed_evaluation",
        "partition": "ham10000_sealed_test",
        "n_images": features.shape[0],
        "n_lesions": len(lesion_records),
        "class_distribution": dict(Counter(y_true)),
        "note": (
            "V2 pipeline evaluation on the sealed HAM10000 test partition (1495 lesions, "
            "natural distribution). No model was trained on any sealed data. The Derm7pt-"
            "trained classifier is a cross-centre transfer; the HAM-trained classifier is "
            "within-dataset. Criterion heads are Derm7pt-specific and have no HAM ground "
            "truth — they feed the HOST aggregation, router, and contracts."
        ),
        "arm1_v2_system": {
            "description": "Full v2 pipeline: Derm7pt-trained classifier + criterion heads + HOST + router + contracts",
            "classifier_fit": f"Derm7pt train ({n_d7_train} cases)",
            "classifier_metrics": arm1_clf_metrics,
            "contract_metrics": arm1_contract_metrics,
        },
        "arm2_ham_baseline": {
            "description": "HAM-trained classifier only (no criteria, no contracts)",
            "classifier_fit": f"HAM train+cal ({n_ham_train} images), C={ham_c}",
            "classifier_metrics": arm2_clf_metrics,
        },
        "v1_sealed_comparison": {
            "note": "The v1 sealed H2H (notes/SEALED_RESULT_2026-08-10.md) evaluated the "
                    "v1 pipeline on the same 1495 lesions with 11 arms. Key v1 numbers for "
                    "comparison: accuracy=0.8234, ECE=0.0279, citation_integrity=1.0 (1017/1017).",
            "v1_accuracy": 0.8234,
            "v1_ece": 0.0279,
            "v1_citation_integrity": 1.0,
        },
        "boundaries": {
            "sealed_test_access": True,
            "n_boot": args.n_boot,
            "seed": SEED,
            "base_C": BASE_C,
        },
    }

    def _serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(output, indent=2, default=_serialize) + "\n", encoding="utf-8")
        print(f"\nWrote {OUT}")
    else:
        print("\n[dry run] pass --write to save the artifact")

    # Summary
    print("\n=== V2 Sealed Evaluation Summary ===")
    print(f"{'':30s}  {'v2 system':>12s}  {'HAM baseline':>12s}  {'v1 sealed':>12s}")
    print(f"{'Accuracy':30s}  {arm1_clf_metrics['accuracy']:>12.4f}  {arm2_clf_metrics['accuracy']:>12.4f}  {'0.8234':>12s}")
    print(f"{'Melanoma AUROC':30s}  {arm1_clf_metrics['melanoma_auroc'] or 'N/A':>12}  {arm2_clf_metrics['melanoma_auroc'] or 'N/A':>12}  {'0.9426':>12s}")
    print(f"{'ECE':30s}  {arm1_clf_metrics['calibration']['ece']:>12.4f}  {arm2_clf_metrics['calibration']['ece']:>12.4f}  {'0.0279':>12s}")
    if arm1_contract_metrics:
        print(f"{'Contract trust rate':30s}  {arm1_contract_metrics['safe_to_display_rate']:>12.4f}  {'N/A':>12s}  {'1.0':>12s}")
        print(f"{'Citation rate':30s}  {arm1_contract_metrics['citation_rate']:>12.4f}  {'N/A':>12s}  {'—':>12s}")
        print(f"{'Retrieval savings':30s}  {arm1_contract_metrics['retrieval_savings']:>12.4f}  {'N/A':>12s}  {'—':>12s}")


if __name__ == "__main__":
    main()
