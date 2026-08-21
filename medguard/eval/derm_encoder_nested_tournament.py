"""Nested, test-redacted development tournament for frozen dermoscopy encoders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..data.freeze_ham_nested_calibration_v1 import OUTPUT as NESTED_SPLIT
from ..data.freeze_ham_nested_calibration_v1 import load_assignment
from ..data.manifest import load_manifest
from .derm_encoder_tournament import (
    CANDIDATES,
    DEV_MANIFEST,
    FEATURE_ROOT,
    MALIGNANT,
    _fit_probes,
    _metrics,
    _softmax,
    _threshold_at_specificity,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "research/experiments/derm_perception/v4_nested_lesion_tournament/results"
LEGACY_TRAIN_PROBE_ROOT = ROOT / "research/experiments/derm_perception/v2_tournament/results"


def _manifest_sha(path: Path) -> str:
    return path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()


def _lesion_binary_scores(records, indices, malignant_scores):
    grouped: dict[str, dict[str, Any]] = {}
    for index, score in zip(indices, malignant_scores):
        record = records[index]
        group = grouped.setdefault(record.cluster_id, {
            "label": record.label, "score_sum": 0.0, "count": 0,
        })
        if group["label"] != record.label:
            raise ValueError(f"conflicting labels in lesion {record.cluster_id}")
        group["score_sum"] += float(score)
        group["count"] += 1
    ordered = [grouped[cluster_id] for cluster_id in sorted(grouped)]
    labels = [group["label"] in MALIGNANT for group in ordered]
    scores = [group["score_sum"] / group["count"] for group in ordered]
    return labels, scores


def _lesion_rows(records, indices, probabilities, malignant_scores, classes):
    import numpy as np

    grouped: dict[str, dict[str, Any]] = {}
    for index, probability, malignant_score in zip(indices, probabilities, malignant_scores):
        record = records[index]
        group = grouped.setdefault(record.cluster_id, {
            "label": record.label,
            "probability_sum": np.zeros(len(classes), dtype=float),
            "malignant_sum": 0.0,
            "count": 0,
        })
        if group["label"] != record.label:
            raise ValueError(f"conflicting labels in lesion {record.cluster_id}")
        group["probability_sum"] += np.asarray(probability, dtype=float)
        group["malignant_sum"] += float(malignant_score)
        group["count"] += 1

    rows = []
    for cluster_id in sorted(grouped):
        group = grouped[cluster_id]
        probability = group["probability_sum"] / group["count"]
        prediction = classes[int(probability.argmax())]
        rows.append({
            "case_id": f"lesion:{cluster_id}",
            "cluster_id": cluster_id,
            "label": group["label"],
            "prediction": prediction,
            "confidence": float(probability.max()),
            "malignant_true": bool(group["label"] in MALIGNANT),
            "malignant_score": float(group["malignant_sum"] / group["count"]),
        })
    return rows


def _load_reusable_train_probes(key: str, manifest_sha: str, expected_train_images: int):
    """Load only a receipt-proven train-only v2 probe to avoid re-running CV.

    The v2 evaluator fits these probes solely on the train partition; its
    calibration metrics were not admissible for selection, but the resulting
    train-only fitted objects are valid inputs to this nested threshold/holdout
    repair once their receipt is verified.
    """
    import joblib

    source = LEGACY_TRAIN_PROBE_ROOT / key
    receipt_path = source / "result.json"
    probe_path = source / "probes.joblib"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("protocol") != "ham10000-derm-encoder-tournament-v2":
        raise ValueError("reusable probe has an unexpected protocol")
    if receipt.get("model_key") != key:
        raise ValueError("reusable probe model key does not match")
    if receipt.get("development_manifest_sha256") != manifest_sha:
        raise ValueError("reusable probe belongs to a different development manifest")
    if receipt.get("test_records_loaded") != 0:
        raise ValueError("reusable probe receipt is not test-redacted")
    if int(receipt.get("training", {}).get("n_images", -1)) != expected_train_images:
        raise ValueError("reusable probe was not fitted to the expected train partition")
    probes = joblib.load(probe_path)
    expected = {"diagnosis", "malignant", "platt", "temperature", "diagnosis_classes"}
    if not expected.issubset(probes):
        raise ValueError("reusable probe payload is incomplete")
    return probes, {
        "mode": "receipt-verified-v2-train-only-probe",
        "receipt": str(receipt_path.relative_to(ROOT).as_posix()),
        "probe": str(probe_path.relative_to(ROOT).as_posix()),
    }


def evaluate_candidate(
    key: str,
    *,
    manifest: Path = DEV_MANIFEST,
    nested_split: Path = NESTED_SPLIT,
    reuse_train_probes: bool = False,
) -> dict[str, Any]:
    """Fit only on train, tune the threshold once, then score the node holdout."""
    import joblib
    import numpy as np

    if key not in CANDIDATES:
        raise ValueError(f"unknown candidate {key!r}")
    records = load_manifest(manifest)
    if {record.split for record in records} != {"train", "calibration"}:
        raise ValueError("nested tournament requires the test-redacted development manifest")
    assignments, nested_metadata = load_assignment(nested_split)
    manifest_sha = _manifest_sha(manifest)
    if nested_metadata.get("parent_manifest_sha256") != manifest_sha:
        raise ValueError("nested split belongs to a different development manifest")
    calibration_clusters = {record.cluster_id for record in records if record.split == "calibration"}
    if set(assignments) != calibration_clusters:
        raise ValueError("nested split does not exactly cover development calibration clusters")

    feature_dir = FEATURE_ROOT / key
    feature_meta = json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    if feature_meta.get("development_manifest_sha256") != manifest_sha:
        raise ValueError("feature/manifest hash mismatch")
    if feature_meta.get("test_records_loaded") != 0:
        raise ValueError("feature cache is not certified test-redacted")
    features = np.load(feature_dir / "features.npy", mmap_mode="r")
    if len(features) != len(records):
        raise ValueError("feature row count does not match manifest")

    train_idx = np.asarray([index for index, record in enumerate(records) if record.split == "train"])
    threshold_idx = np.asarray([
        index for index, record in enumerate(records)
        if record.split == "calibration" and assignments[record.cluster_id] == "threshold_calibration"
    ])
    validation_idx = np.asarray([
        index for index, record in enumerate(records)
        if record.split == "calibration" and assignments[record.cluster_id] == "node_validation"
    ])
    if not len(train_idx) or not len(threshold_idx) or not len(validation_idx):
        raise ValueError("nested train/threshold/validation partition is empty")
    if reuse_train_probes:
        probes, probe_provenance = _load_reusable_train_probes(
            key, manifest_sha, int(len(train_idx))
        )
    else:
        probes = _fit_probes(
            features[train_idx],
            [records[index].label for index in train_idx],
            [records[index].cluster_id for index in train_idx],
        )
        probe_provenance = {"mode": "fit-in-this-run-train-only"}
    malignant_raw_tune = probes["malignant"].decision_function(features[threshold_idx])
    malignant_tune = probes["platt"].predict_proba(malignant_raw_tune[:, None])[:, 1]
    tune_labels, tune_scores = _lesion_binary_scores(
        records, threshold_idx, malignant_tune
    )
    point_90 = _threshold_at_specificity(tune_labels, tune_scores, 0.90)

    diag = probes["diagnosis"]
    decision = diag.decision_function(features[validation_idx])
    classes = list(probes["diagnosis_classes"])
    order = [list(diag.classes_).index(label) for label in classes]
    probabilities = _softmax(decision[:, order], probes["temperature"])
    malignant_raw = probes["malignant"].decision_function(features[validation_idx])
    malignant_scores = probes["platt"].predict_proba(malignant_raw[:, None])[:, 1]
    rows = _lesion_rows(
        records, validation_idx, probabilities, malignant_scores, classes
    )
    metrics = _metrics(rows, float(point_90["threshold"]))
    metrics["n_lesions"] = metrics.pop("n_images")
    metrics["n_source_images"] = int(len(validation_idx))
    metrics["evaluation_unit"] = "lesion (mean probability across views)"

    output = RESULT_ROOT / key
    output.mkdir(parents=True, exist_ok=True)
    (output / "node_validation_lesion_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    joblib.dump(probes, output / "train_only_probes.joblib")
    result: dict[str, Any] = {
        "protocol": "ham10000-derm-encoder-nested-lesion-tournament-v4",
        "evidence_level": "nested development screening; not final-test evidence",
        "model_key": key,
        "model_id": CANDIDATES[key].model_id,
        "development_manifest_sha256": manifest_sha,
        "nested_calibration_sha256": _manifest_sha(nested_split),
        "test_records_loaded": 0,
        "partition": {
            "train_images": int(len(train_idx)),
            "threshold_calibration_images": int(len(threshold_idx)),
            "threshold_calibration_lesions": int(len(set(
                records[index].cluster_id for index in threshold_idx
            ))),
            "node_validation_images": int(len(validation_idx)),
            "node_validation_lesions": int(len(rows)),
            "threshold_source": "threshold_calibration only",
            "node_selection_metrics_source": "node_validation lesions only",
        },
        "training": {
            "lesion_equal_weighting": True,
            "cv": "5-fold StratifiedGroupKFold by lesion on train only",
            "selected_c": probes["selected_c"],
            "cv_scores": probes["cv_scores"],
            "probability_calibration": {
                "diagnosis": "scalar temperature on train OOF logits",
                "diagnosis_temperature": probes["temperature"],
                "malignant": "Platt scaling on train OOF scores",
            },
            "probe_provenance": probe_provenance,
        },
        "operating_point": {"specificity_90": point_90},
        "node_validation_metrics": metrics,
        "feature_provenance": feature_meta,
        "limitations": [
            "This is an internal development partition and cannot replace the sealed final test.",
            "The 90% specificity operating point is selected on one calibration partition; site shift can still move specificity.",
            "Do not select a fusion weight, architecture, or threshold using BCN after reading BCN outcomes.",
            "All threshold and validation outcomes are lesion-level means so multi-image lesions receive equal weight.",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--manifest", type=Path, default=DEV_MANIFEST)
    parser.add_argument("--nested-split", type=Path, default=NESTED_SPLIT)
    parser.add_argument(
        "--reuse-train-probes", action="store_true",
        help="reuse a receipt-verified v2 train-only probe instead of re-running grouped CV",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate_candidate(
        args.model,
        manifest=args.manifest,
        nested_split=args.nested_split,
        reuse_train_probes=args.reuse_train_probes,
    ), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
