"""Untuned external BCN20000 evaluation for frozen HAM derm encoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

from ..data.manifest import CaseRecord, file_sha256, load_manifest
from .derm_encoders import CANDIDATES, FEATURE_ROOT, load_candidate_encoder
from .metrics import ece, selective_accuracy_at_coverage
from .statistics import clustered_bootstrap_ci


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/manifests/bcn20000_external_v1.json"
EXPERIMENT_ROOT = FEATURE_ROOT.parent / "external_bcn_v1"
ACTIONABLE = {"melanoma", "bcc", "akiec", "scc"}
SHARED_CLASSES = ("akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc")
SEED = 20260713


def _checkpoint_hash(provenance: dict[str, object]) -> str:
    return str(provenance["checkpoint"]["combined_sha256"])


def _validate_external_records(manifest: Path) -> tuple[list[CaseRecord], str]:
    records = load_manifest(manifest)
    if not records or {row.source for row in records} != {
        "BCN20000-v1-labeled-cohort-cleaned"
    }:
        raise ValueError("external evaluator accepts only the frozen BCN20000 manifest")
    if {row.split for row in records} != {"test"}:
        raise ValueError("BCN20000 manifest must be external-test only")
    if set(row.label for row in records) != set(SHARED_CLASSES) | {"scc"}:
        raise ValueError("BCN20000 label set differs from the frozen protocol")
    expected = manifest.with_suffix(manifest.suffix + ".sha256").read_text(
        encoding="ascii"
    ).strip()
    observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError("external manifest hash mismatch")
    return records, expected


def extract_external(
    key: str, *, manifest: Path = MANIFEST, batch_size: int | None = None,
) -> dict[str, object]:
    import numpy as np

    if key not in CANDIDATES:
        raise ValueError(f"unknown candidate {key!r}")
    records, manifest_sha = _validate_external_records(manifest)
    candidate = CANDIDATES[key]
    output = EXPERIMENT_ROOT / "features" / key
    output.mkdir(parents=True, exist_ok=True)
    final = output / "features.npy"
    meta_path = output / "feature_manifest.json"
    if final.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("external_manifest_sha256") != manifest_sha:
            raise ValueError("cached external features use a different manifest")
        if metadata.get("n_records") != len(records) or metadata.get("model_key") != key:
            raise ValueError("cached external feature metadata mismatch")
        return metadata

    encode, provenance = load_candidate_encoder(key)
    development_meta = json.loads(
        (FEATURE_ROOT / key / "feature_manifest.json").read_text(encoding="utf-8")
    )
    if _checkpoint_hash(provenance) != _checkpoint_hash(development_meta["provenance"]):
        raise ValueError("external encoder checkpoint differs from development encoder")
    effective_batch = int(batch_size or candidate.batch_size)
    if effective_batch <= 0:
        raise ValueError("batch_size must be positive")
    partial = output / "features.partial.npy"
    progress_path = output / "progress.json"
    next_index, array = 0, None
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("external_manifest_sha256") != manifest_sha:
            raise ValueError("partial external features use a different manifest")
        next_index = int(progress["next_index"])
        array = np.lib.format.open_memmap(partial, mode="r+")
        if array.shape[0] != len(records):
            raise ValueError("partial external feature row count mismatch")
    started = time.perf_counter()
    for start in range(next_index, len(records), effective_batch):
        stop = min(len(records), start + effective_batch)
        batch = encode([record.image_path for record in records[start:stop]])
        if batch.ndim != 2 or batch.shape[0] != stop - start:
            raise ValueError(f"encoder returned invalid shape {batch.shape}")
        if not np.isfinite(batch).all():
            raise ValueError("encoder returned non-finite external features")
        if array is None:
            array = np.lib.format.open_memmap(
                partial, mode="w+", dtype="float32", shape=(len(records), batch.shape[1])
            )
        if batch.shape[1] != array.shape[1]:
            raise ValueError("external feature dimension changed mid-run")
        array[start:stop] = batch
        array.flush()
        temp = progress_path.with_suffix(".tmp")
        temp.write_text(json.dumps({
            "model_key": key,
            "external_manifest_sha256": manifest_sha,
            "next_index": stop,
            "n_records": len(records),
            "feature_dim": int(array.shape[1]),
        }, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, progress_path)
        if stop == len(records) or stop % max(1, effective_batch * 10) == 0:
            print(
                f"[bcn:{key}] {stop}/{len(records)} images; "
                f"{time.perf_counter() - started:.1f}s", flush=True,
            )
    if array is None:
        raise ValueError("no external features extracted")
    del array
    os.replace(partial, final)
    metadata: dict[str, object] = {
        "protocol": "bcn20000-untuned-external-features-v1",
        "model_key": key,
        "model_id": candidate.model_id,
        "external_manifest_sha256": manifest_sha,
        "n_records": len(records),
        "n_lesions": len({row.cluster_id for row in records}),
        "feature_dim": int(np.load(final, mmap_mode="r").shape[1]),
        "feature_file_sha256": file_sha256(final, chunk_size=8 * 1024 * 1024),
        "batch_size": effective_batch,
        "elapsed_seconds_current_process": time.perf_counter() - started,
        "provenance": provenance,
        "threshold_tuning_on_bcn": False,
    }
    temp = meta_path.with_suffix(".tmp")
    temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, meta_path)
    if progress_path.exists():
        progress_path.unlink()
    return metadata


def _softmax(scores, temperature: float):
    from scipy.special import softmax

    return softmax(scores / float(temperature), axis=1)


def _aggregate_lesions(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    import numpy as np

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)
    aggregated = []
    for lesion, group in sorted(grouped.items()):
        labels = {str(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"external lesion {lesion} has conflicting labels")
        probabilities = np.mean([
            [float(value) for value in row["diagnosis_probabilities"]] for row in group
        ], axis=0)
        aggregated.append({
            "case_id": lesion,
            "cluster_id": lesion,
            "label": next(iter(labels)),
            "prediction": SHARED_CLASSES[int(probabilities.argmax())],
            "confidence": float(probabilities.max()),
            "diagnosis_probabilities": probabilities.tolist(),
            "malignant_score": float(np.mean([
                float(row["malignant_score"]) for row in group
            ])),
            "n_views": len(group),
        })
    return aggregated


def _interval(
    rows: Sequence[dict[str, object]], metric: Callable[[Sequence[dict[str, object]]], float],
    *, n_boot: int,
) -> dict[str, float]:
    result = clustered_bootstrap_ci(
        rows, [str(row["cluster_id"]) for row in rows], metric,
        n_boot=n_boot, seed=SEED,
    )
    return {"estimate": result.estimate, "ci_low": result.low, "ci_high": result.high}


def _binary_metrics(
    rows: Sequence[dict[str, object]], threshold: float, *, n_boot: int,
) -> dict[str, object]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    def auc(sample):
        return float(roc_auc_score(
            [str(row["label"]) in ACTIONABLE for row in sample],
            [float(row["malignant_score"]) for row in sample],
        ))

    def recall(sample):
        positives = [row for row in sample if str(row["label"]) in ACTIONABLE]
        return sum(float(row["malignant_score"]) >= threshold for row in positives) / len(positives)

    def specificity(sample):
        negatives = [row for row in sample if str(row["label"]) not in ACTIONABLE]
        return sum(float(row["malignant_score"]) < threshold for row in negatives) / len(negatives)

    truth = [str(row["label"]) in ACTIONABLE for row in rows]
    scores = [float(row["malignant_score"]) for row in rows]
    return {
        "actionable_auc": _interval(rows, auc, n_boot=n_boot),
        "actionable_recall_at_frozen_threshold": _interval(rows, recall, n_boot=n_boot),
        "specificity_at_frozen_threshold": _interval(rows, specificity, n_boot=n_boot),
        "false_reassurance_at_frozen_threshold": _interval(
            rows, lambda sample: 1.0 - recall(sample), n_boot=n_boot
        ),
        "brier": float(brier_score_loss(truth, scores)),
        "ece": ece(scores, truth),
        "per_actionable_class_recall": {
            label: (
                sum(float(row["malignant_score"]) >= threshold for row in rows if row["label"] == label)
                / sum(row["label"] == label for row in rows)
            )
            for label in sorted(ACTIONABLE)
        },
        "counts": {
            "true_positive": sum(y and score >= threshold for y, score in zip(truth, scores)),
            "false_negative": sum(y and score < threshold for y, score in zip(truth, scores)),
            "true_negative": sum(not y and score < threshold for y, score in zip(truth, scores)),
            "false_positive": sum(not y and score >= threshold for y, score in zip(truth, scores)),
        },
    }


def _diagnosis_metrics(
    rows: Sequence[dict[str, object]], *, n_boot: int,
) -> dict[str, object]:
    from sklearn.metrics import recall_score

    shared = [row for row in rows if str(row["label"]) != "scc"]

    def accuracy(sample):
        return sum(row["prediction"] == row["label"] for row in sample) / len(sample)

    def macro_recall(sample):
        return float(recall_score(
            [row["label"] for row in sample], [row["prediction"] for row in sample],
            labels=list(SHARED_CLASSES), average="macro", zero_division=0,
        ))

    correct = [row["prediction"] == row["label"] for row in shared]
    confidence = [float(row["confidence"]) for row in shared]
    return {
        "n_shared_taxonomy": len(shared),
        "accuracy": _interval(shared, accuracy, n_boot=n_boot),
        "macro_recall": _interval(shared, macro_recall, n_boot=n_boot),
        "ece": ece(confidence, correct),
        "selective_accuracy": {
            str(coverage): selective_accuracy_at_coverage(confidence, correct, coverage)
            for coverage in (0.5, 0.65, 0.8, 0.9)
        },
        "per_class_recall": {
            label: (
                sum(row["prediction"] == label for row in shared if row["label"] == label)
                / sum(row["label"] == label for row in shared)
            )
            for label in SHARED_CLASSES
        },
    }


def evaluate_external(
    key: str, *, manifest: Path = MANIFEST, n_boot: int = 1000,
) -> dict[str, object]:
    import joblib
    import numpy as np

    records, manifest_sha = _validate_external_records(manifest)
    feature_dir = EXPERIMENT_ROOT / "features" / key
    feature_meta = json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    if feature_meta["external_manifest_sha256"] != manifest_sha:
        raise ValueError("external feature/manifest hash mismatch")
    features = np.load(feature_dir / "features.npy", mmap_mode="r")
    if len(features) != len(records):
        raise ValueError("external feature row count mismatch")
    development_result = json.loads(
        (FEATURE_ROOT.parent / "results" / key / "result.json").read_text(encoding="utf-8")
    )
    probes = joblib.load(FEATURE_ROOT.parent / "results" / key / "probes.joblib")
    diag = probes["diagnosis"]
    decision = diag.decision_function(features)
    classes = list(probes["diagnosis_classes"])
    if set(classes) != set(SHARED_CLASSES):
        raise ValueError("development diagnosis classes differ from shared BCN taxonomy")
    order = [list(diag.classes_).index(label) for label in SHARED_CLASSES]
    probabilities = _softmax(decision[:, order], probes["temperature"])
    predictions = [SHARED_CLASSES[index] for index in probabilities.argmax(axis=1)]
    malignant_raw = probes["malignant"].decision_function(features)
    malignant_score = probes["platt"].predict_proba(malignant_raw[:, None])[:, 1]
    threshold = float(development_result["operating_points"]["specificity_90"]["threshold"])
    image_rows = [
        {
            "case_id": record.case_id,
            "cluster_id": record.cluster_id,
            "label": record.label,
            "prediction": prediction,
            "confidence": float(probability.max()),
            "diagnosis_probabilities": probability.tolist(),
            "malignant_score": float(score),
        }
        for record, prediction, probability, score in zip(
            records, predictions, probabilities, malignant_score
        )
    ]
    lesion_rows = _aggregate_lesions(image_rows)
    scc_rows = [row for row in lesion_rows if row["label"] == "scc"]
    result: dict[str, object] = {
        "protocol": "bcn20000-untuned-external-evaluation-v1",
        "evidence_level": "external evaluation with HAM calibration threshold frozen",
        "model_key": key,
        "model_id": CANDIDATES[key].model_id,
        "external_manifest_sha256": manifest_sha,
        "threshold_source": "HAM10000 calibration",
        "frozen_malignant_threshold": threshold,
        "threshold_tuning_on_bcn": False,
        "n_images": len(image_rows),
        "n_lesions": len(lesion_rows),
        "primary_unit": "lesion (mean probability across available views)",
        "lesion_level": {
            "binary": _binary_metrics(lesion_rows, threshold, n_boot=n_boot),
            "diagnosis_shared_taxonomy": _diagnosis_metrics(lesion_rows, n_boot=n_boot),
            "scc_unseen_class": {
                "n_lesions": len(scc_rows),
                "actionable_recall_at_frozen_threshold": (
                    sum(float(row["malignant_score"]) >= threshold for row in scc_rows)
                    / len(scc_rows)
                ),
                "diagnosis_prediction_distribution": dict(Counter(
                    str(row["prediction"]) for row in scc_rows
                )),
            },
        },
        "image_level_sensitivity_analysis": {
            "binary": _binary_metrics(image_rows, threshold, n_boot=n_boot),
            "diagnosis_shared_taxonomy": _diagnosis_metrics(image_rows, n_boot=n_boot),
        },
        "feature_provenance": feature_meta,
    }
    output = EXPERIMENT_ROOT / "results" / key
    output.mkdir(parents=True, exist_ok=True)
    (output / "external_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in image_rows),
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    metadata = extract_external(args.model, batch_size=args.batch_size)
    print(json.dumps({"features": metadata}, indent=2, sort_keys=True), flush=True)
    if not args.extract_only:
        result = evaluate_external(args.model, n_boot=args.bootstrap)
        print(json.dumps({"result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
