"""HAM-only DermLIP/PanDerm fusion experiments with frozen BCN transfer.

The neutral 50/50 probability ensemble is predeclared and has no fitted fusion
weight.  The concatenated probe is trained and calibrated only on HAM train and
calibration records.  The sealed HAM test split is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from medguard.data.manifest import file_sha256, load_manifest

from .derm_encoder_tournament import (
    MALIGNANT,
    _fit_probes,
    _metrics,
    _softmax,
    _threshold_at_specificity,
)
from .derm_encoders import CANDIDATES, DEV_MANIFEST, FEATURE_ROOT
from .derm_external_bcn import (
    EXPERIMENT_ROOT as EXTERNAL_ROOT,
    MANIFEST as EXTERNAL_MANIFEST,
    SHARED_CLASSES,
    _aggregate_lesions,
    _binary_metrics,
    _diagnosis_metrics,
    _validate_external_records,
)


RESULT_ROOT = FEATURE_ROOT.parent / "results"
FUSION_KEYS = ("fusion50", "concat")
COMPONENTS = ("dermlip", "panderm")


def _candidate_outputs(features, probes):
    import numpy as np

    diagnosis = probes["diagnosis"]
    classes = list(probes["diagnosis_classes"])
    if set(classes) != set(SHARED_CLASSES):
        raise ValueError(f"unexpected diagnosis classes: {classes}")
    order = [list(diagnosis.classes_).index(label) for label in SHARED_CLASSES]
    probabilities = _softmax(
        diagnosis.decision_function(features)[:, order], probes["temperature"]
    )
    malignant_raw = probes["malignant"].decision_function(features)
    malignant = probes["platt"].predict_proba(malignant_raw[:, None])[:, 1]
    return np.asarray(probabilities, dtype=float), np.asarray(malignant, dtype=float)


def _mean_probabilities(arrays: Sequence[object]):
    import numpy as np

    values = [np.asarray(array, dtype=float) for array in arrays]
    if len(values) != 2 or values[0].shape != values[1].shape:
        raise ValueError("fusion requires two shape-aligned probability arrays")
    if values[0].ndim not in {1, 2}:
        raise ValueError("fusion probabilities must be one- or two-dimensional")
    return (values[0] + values[1]) / 2.0


def _concatenate_features(arrays: Sequence[object]):
    import numpy as np

    values = [np.asarray(array, dtype=float) for array in arrays]
    if len(values) != 2 or any(value.ndim != 2 for value in values):
        raise ValueError("concatenation requires two feature matrices")
    if values[0].shape[0] != values[1].shape[0]:
        raise ValueError("feature matrices are not row-aligned")
    # Each source embedding is already unit-normalized.  Division preserves a
    # unit norm for the concatenation and makes the existing C grid comparable.
    return np.concatenate(values, axis=1) / math.sqrt(2.0)


def _development_data(manifest: Path = DEV_MANIFEST):
    import numpy as np

    records = load_manifest(manifest)
    if {row.split for row in records} != {"train", "calibration"}:
        raise ValueError("fusion requires the test-redacted HAM manifest")
    manifest_sha = manifest.with_suffix(manifest.suffix + ".sha256").read_text(
        encoding="ascii"
    ).strip()
    features = {}
    metadata = {}
    for key in COMPONENTS:
        directory = FEATURE_ROOT / key
        meta = json.loads((directory / "feature_manifest.json").read_text(encoding="utf-8"))
        if meta["development_manifest_sha256"] != manifest_sha:
            raise ValueError(f"{key}: feature/manifest mismatch")
        matrix = np.load(directory / "features.npy", mmap_mode="r")
        if len(matrix) != len(records):
            raise ValueError(f"{key}: feature row mismatch")
        features[key], metadata[key] = matrix, meta
    train = np.asarray([i for i, row in enumerate(records) if row.split == "train"])
    calibration = np.asarray([i for i, row in enumerate(records) if row.split == "calibration"])
    return records, manifest_sha, features, metadata, train, calibration


def _rows(records, indices, probabilities, malignant_scores):
    predictions = [SHARED_CLASSES[index] for index in probabilities.argmax(axis=1)]
    return [
        {
            "case_id": records[index].case_id,
            "cluster_id": records[index].cluster_id,
            "label": records[index].label,
            "prediction": prediction,
            "confidence": float(probability.max()),
            "diagnosis_probabilities": probability.tolist(),
            "malignant_true": bool(records[index].label in MALIGNANT),
            "malignant_score": float(score),
        }
        for index, prediction, probability, score in zip(
            indices, predictions, probabilities, malignant_scores
        )
    ]


def evaluate_development(method: str, *, manifest: Path = DEV_MANIFEST) -> dict[str, object]:
    import joblib
    import numpy as np

    if method not in FUSION_KEYS:
        raise ValueError(f"unknown fusion method {method!r}")
    records, manifest_sha, features, metadata, train, calibration = _development_data(manifest)
    probes_by_component = {
        key: joblib.load(RESULT_ROOT / key / "probes.joblib") for key in COMPONENTS
    }
    output = RESULT_ROOT / method
    output.mkdir(parents=True, exist_ok=True)

    if method == "fusion50":
        outputs = [
            _candidate_outputs(features[key][calibration], probes_by_component[key])
            for key in COMPONENTS
        ]
        diagnosis_probability = _mean_probabilities([value[0] for value in outputs])
        malignant_score = _mean_probabilities([value[1] for value in outputs])
        training = {
            "method": "unweighted arithmetic mean of independently HAM-trained calibrated probabilities",
            "weights": {"dermlip": 0.5, "panderm": 0.5},
            "weight_selection": "predeclared neutral weight; no BCN weight search",
        }
    else:
        train_features = _concatenate_features([features[key][train] for key in COMPONENTS])
        probes = _fit_probes(
            train_features,
            [records[index].label for index in train],
            [records[index].cluster_id for index in train],
        )
        calibration_features = _concatenate_features([
            features[key][calibration] for key in COMPONENTS
        ])
        diagnosis_probability, malignant_score = _candidate_outputs(
            calibration_features, probes
        )
        joblib.dump(probes, output / "probes.joblib")
        training = {
            "method": "unit-norm DermLIP/PanDerm feature concatenation plus linear probes",
            "feature_scale": "concatenation divided by sqrt(2)",
            "n_images": len(train),
            "n_lesions": len({records[index].cluster_id for index in train}),
            "cv": "5-fold StratifiedGroupKFold by lesion",
            "selected_c": probes["selected_c"],
            "cv_scores": probes["cv_scores"],
            "probability_calibration": {
                "diagnosis": "temperature scaling on training OOF logits",
                "malignant": "Platt scaling on training OOF scores",
            },
        }

    truth = np.asarray([records[index].label in MALIGNANT for index in calibration], dtype=bool)
    point_90 = _threshold_at_specificity(truth, malignant_score, 0.90)
    point_80 = _threshold_at_specificity(truth, malignant_score, 0.80)
    rows = _rows(records, calibration, diagnosis_probability, malignant_score)
    (output / "calibration_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    result: dict[str, object] = {
        "protocol": "ham10000-derm-encoder-fusion-v1",
        "evidence_level": "development/calibration screening; not final-test evidence",
        "model_key": method,
        "model_id": "+".join(CANDIDATES[key].model_id for key in COMPONENTS),
        "components": list(COMPONENTS),
        "development_manifest_sha256": manifest_sha,
        "test_records_loaded": 0,
        "training": training,
        "operating_points": {"specificity_90": point_90, "specificity_80": point_80},
        "calibration_metrics": _metrics(rows, float(point_90["threshold"])),
        "feature_provenance": metadata,
        "selection_note": (
            "fusion was proposed after individual BCN screening; BCN fusion results are architecture-"
            "development evidence, while the sealed HAM test remains untouched"
        ),
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def evaluate_external(method: str, *, manifest: Path = EXTERNAL_MANIFEST,
                      n_boot: int = 2000) -> dict[str, object]:
    import joblib
    import numpy as np

    if method not in FUSION_KEYS:
        raise ValueError(f"unknown fusion method {method!r}")
    records, manifest_sha = _validate_external_records(manifest)
    matrices, metadata = {}, {}
    for key in COMPONENTS:
        directory = EXTERNAL_ROOT / "features" / key
        meta = json.loads((directory / "feature_manifest.json").read_text(encoding="utf-8"))
        if meta["external_manifest_sha256"] != manifest_sha:
            raise ValueError(f"{key}: external feature/manifest mismatch")
        matrix = np.load(directory / "features.npy", mmap_mode="r")
        if len(matrix) != len(records):
            raise ValueError(f"{key}: external feature row mismatch")
        matrices[key], metadata[key] = matrix, meta

    development = json.loads((RESULT_ROOT / method / "result.json").read_text(encoding="utf-8"))
    threshold = float(development["operating_points"]["specificity_90"]["threshold"])
    if method == "fusion50":
        outputs = [
            _candidate_outputs(
                matrices[key], joblib.load(RESULT_ROOT / key / "probes.joblib")
            ) for key in COMPONENTS
        ]
        diagnosis_probability = _mean_probabilities([value[0] for value in outputs])
        malignant_score = _mean_probabilities([value[1] for value in outputs])
    else:
        probes = joblib.load(RESULT_ROOT / method / "probes.joblib")
        diagnosis_probability, malignant_score = _candidate_outputs(
            _concatenate_features([matrices[key] for key in COMPONENTS]), probes
        )

    image_rows = [
        {
            "case_id": record.case_id,
            "cluster_id": record.cluster_id,
            "label": record.label,
            "prediction": SHARED_CLASSES[int(probability.argmax())],
            "confidence": float(probability.max()),
            "diagnosis_probabilities": probability.tolist(),
            "malignant_score": float(score),
        }
        for record, probability, score in zip(records, diagnosis_probability, malignant_score)
    ]
    lesion_rows = _aggregate_lesions(image_rows)
    scc_rows = [row for row in lesion_rows if row["label"] == "scc"]
    result: dict[str, object] = {
        "protocol": "bcn20000-derm-fusion-development-transfer-v1",
        "evidence_level": (
            "external architecture-development evaluation; not confirmatory because fusion followed "
            "individual-model BCN screening"
        ),
        "model_key": method,
        "components": list(COMPONENTS),
        "external_manifest_sha256": manifest_sha,
        "threshold_source": "HAM10000 calibration",
        "frozen_malignant_threshold": threshold,
        "threshold_tuning_on_bcn": False,
        "fusion_weight_tuning_on_bcn": False,
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
            },
        },
        "image_level_sensitivity_analysis": {
            "binary": _binary_metrics(image_rows, threshold, n_boot=n_boot),
            "diagnosis_shared_taxonomy": _diagnosis_metrics(image_rows, n_boot=n_boot),
        },
        "feature_provenance": metadata,
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    output = EXTERNAL_ROOT / "results" / method
    output.mkdir(parents=True, exist_ok=True)
    (output / "external_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in image_rows), encoding="utf-8"
    )
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=FUSION_KEYS, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    development = evaluate_development(args.method)
    payload = {"development": development}
    if not args.development_only:
        payload["external"] = evaluate_external(args.method, n_boot=args.bootstrap)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
