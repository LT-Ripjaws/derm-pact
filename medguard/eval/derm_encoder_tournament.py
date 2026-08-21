"""Train/calibration-only HAM10000 frozen-encoder tournament.

Each candidate uses identical content-hashed images, lesion-grouped CV, equal
lesion weighting, linear probes, and calibration metrics.  The sealed final test
is neither loaded nor scored by this module.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from ..data.manifest import load_manifest
from .derm_encoders import CANDIDATES, DEV_MANIFEST, FEATURE_ROOT, extract_candidate
from .metrics import aurc, ece, selective_accuracy_at_coverage
from .statistics import clustered_bootstrap_ci


SEED = 20260713
MALIGNANT = {"melanoma", "bcc", "akiec"}
# The official Derm1M probe uses C=(feature_dim * classes)/100, i.e. 35.84
# for a 512-D seven-way head.  Include the surrounding decade so the search
# cannot terminate below the official-strength regularization setting.  PanDerm's
# binary probe improved through C=100 in the first sealed development run, so
# include one further decade to distinguish a real optimum from a grid boundary.
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _balanced_lesion_weights(labels: Sequence[str], groups: Sequence[str]):
    import numpy as np

    group_counts = Counter(groups)
    base = np.asarray([1.0 / group_counts[group] for group in groups], dtype=float)
    class_mass: dict[str, float] = {}
    for label, weight in zip(labels, base):
        class_mass[label] = class_mass.get(label, 0.0) + float(weight)
    total, n_classes = float(base.sum()), len(class_mass)
    weights = np.asarray([
        weight * total / (n_classes * class_mass[label])
        for label, weight in zip(labels, base)
    ])
    return weights * (len(weights) / weights.sum())


def _cv_splits(labels, groups):
    from sklearn.model_selection import StratifiedGroupKFold

    return list(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    ).split(range(len(labels)), labels, groups))


def _new_probe(c: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=float(c), max_iter=3000, solver="lbfgs")


def _tune_c(X, labels, groups, weights, *, binary: bool) -> tuple[float, dict[str, float]]:
    import numpy as np
    from sklearn.metrics import recall_score, roc_auc_score

    scores: dict[str, float] = {}
    for c in C_GRID:
        fold_scores = []
        for train_idx, valid_idx in _cv_splits(labels, groups):
            probe = _new_probe(c).fit(X[train_idx], np.asarray(labels)[train_idx],
                                      sample_weight=weights[train_idx])
            if binary:
                probability = probe.predict_proba(X[valid_idx])[:, 1]
                fold_scores.append(roc_auc_score(
                    np.asarray(labels)[valid_idx], probability,
                    sample_weight=weights[valid_idx],
                ))
            else:
                prediction = probe.predict(X[valid_idx])
                fold_scores.append(recall_score(
                    np.asarray(labels)[valid_idx], prediction, average="macro",
                    labels=probe.classes_, sample_weight=weights[valid_idx], zero_division=0,
                ))
        scores[str(c)] = float(sum(fold_scores) / len(fold_scores))
    best = min(C_GRID, key=lambda c: (-scores[str(c)], c))
    return float(best), scores


def _multiclass_temperature(scores, labels, classes, weights):
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    class_index = {label: index for index, label in enumerate(classes)}
    targets = np.asarray([class_index[label] for label in labels], dtype=int)

    def objective(log_temperature: float) -> float:
        scaled = scores / math.exp(float(log_temperature))
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        losses = -log_probs[np.arange(len(targets)), targets]
        return float(np.average(losses, weights=weights))

    fit = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(fit.x))


def _softmax(scores, temperature: float):
    import numpy as np
    from scipy.special import softmax

    return softmax(np.asarray(scores) / float(temperature), axis=1)


def _fit_probes(X_train, train_labels, train_groups):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    train_labels = np.asarray(train_labels)
    train_groups = np.asarray(train_groups)
    weights_diag = _balanced_lesion_weights(train_labels.tolist(), train_groups.tolist())
    y_malignant = np.asarray([int(label in MALIGNANT) for label in train_labels])
    weights_malignant = _balanced_lesion_weights(
        [str(value) for value in y_malignant], train_groups.tolist()
    )

    c_diag, cv_diag = _tune_c(
        X_train, train_labels, train_groups, weights_diag, binary=False
    )
    c_malignant, cv_malignant = _tune_c(
        X_train, y_malignant, train_groups, weights_malignant, binary=True
    )

    diag_classes = sorted(set(train_labels.tolist()))
    oof_diag = np.zeros((len(train_labels), len(diag_classes)), dtype=float)
    oof_malignant = np.zeros(len(train_labels), dtype=float)
    for train_idx, valid_idx in _cv_splits(train_labels, train_groups):
        diag = _new_probe(c_diag).fit(
            X_train[train_idx], train_labels[train_idx], sample_weight=weights_diag[train_idx]
        )
        fold_scores = diag.decision_function(X_train[valid_idx])
        order = [list(diag.classes_).index(label) for label in diag_classes]
        oof_diag[valid_idx] = fold_scores[:, order]
        malignant = _new_probe(c_malignant).fit(
            X_train[train_idx], y_malignant[train_idx],
            sample_weight=weights_malignant[train_idx],
        )
        oof_malignant[valid_idx] = malignant.decision_function(X_train[valid_idx])

    temperature = _multiclass_temperature(
        oof_diag, train_labels, diag_classes, weights_diag
    )
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(
        oof_malignant[:, None], y_malignant, sample_weight=weights_malignant
    )
    diag = _new_probe(c_diag).fit(X_train, train_labels, sample_weight=weights_diag)
    malignant = _new_probe(c_malignant).fit(
        X_train, y_malignant, sample_weight=weights_malignant
    )
    return {
        "diagnosis": diag,
        "malignant": malignant,
        "platt": platt,
        "temperature": temperature,
        "diagnosis_classes": diag_classes,
        "selected_c": {"diagnosis": c_diag, "malignant": c_malignant},
        "cv_scores": {"diagnosis_macro_recall": cv_diag, "malignant_auc": cv_malignant},
    }


def _threshold_at_specificity(y_true, scores, target: float) -> dict[str, float]:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = [
        (float(tpr[i]), float(1.0 - fpr[i]), float(thresholds[i]))
        for i in range(len(thresholds)) if 1.0 - fpr[i] + 1e-12 >= target
    ]
    if not eligible:
        raise ValueError("no operating point satisfies specificity target")
    recall, specificity, threshold = max(eligible, key=lambda row: (row[0], row[1], row[2]))
    return {"target_specificity": target, "threshold": threshold,
            "recall": recall, "specificity": specificity}


def _interval(rows, metric, *, n_boot: int = 1000) -> dict[str, float]:
    interval = clustered_bootstrap_ci(
        rows, [str(row["cluster_id"]) for row in rows], metric,
        n_boot=n_boot, seed=SEED,
    )
    return {"estimate": interval.estimate, "ci_low": interval.low, "ci_high": interval.high}


def _metrics(rows: list[dict[str, object]], threshold_90: float) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, brier_score_loss, recall_score, roc_auc_score

    labels = [str(row["label"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    malignant_true = [bool(row["malignant_true"]) for row in rows]
    malignant_score = [float(row["malignant_score"]) for row in rows]
    malignant_prediction = [score >= threshold_90 for score in malignant_score]
    correct = [label == prediction for label, prediction in zip(labels, predictions)]
    confidence = [float(row["confidence"]) for row in rows]

    def malignant_recall(sample):
        positives = [row for row in sample if bool(row["malignant_true"])]
        return (sum(float(row["malignant_score"]) >= threshold_90 for row in positives)
                / len(positives))

    def specificity(sample):
        negatives = [row for row in sample if not bool(row["malignant_true"])]
        return (sum(float(row["malignant_score"]) < threshold_90 for row in negatives)
                / len(negatives))

    return {
        "n_images": len(rows),
        "n_lesions": len({str(row["cluster_id"]) for row in rows}),
        "diagnosis_accuracy": _interval(
            rows, lambda sample: sum(row["label"] == row["prediction"] for row in sample)
            / len(sample)
        ),
        "diagnosis_macro_recall": _interval(
            rows, lambda sample: recall_score(
                [row["label"] for row in sample], [row["prediction"] for row in sample],
                average="macro", zero_division=0,
            )
        ),
        "melanoma_diagnosis_recall": _interval(
            rows, lambda sample: (
                sum(row["prediction"] == "melanoma" for row in sample if row["label"] == "melanoma")
                / max(1, sum(row["label"] == "melanoma" for row in sample))
            )
        ),
        "malignant_auc": _interval(
            rows, lambda sample: roc_auc_score(
                [bool(row["malignant_true"]) for row in sample],
                [float(row["malignant_score"]) for row in sample],
            )
        ),
        "malignant_recall_at_90_specificity": _interval(rows, malignant_recall),
        "specificity_at_selected_threshold": _interval(rows, specificity),
        "false_reassurance_at_90_specificity": _interval(
            rows, lambda sample: 1.0 - malignant_recall(sample)
        ),
        "malignant_brier": float(brier_score_loss(malignant_true, malignant_score)),
        "malignant_ece": ece(malignant_score, malignant_true),
        "diagnosis_ece": ece(confidence, correct),
        "diagnosis_aurc": aurc(confidence, correct),
        "selective_accuracy": {
            str(coverage): selective_accuracy_at_coverage(confidence, correct, coverage)
            for coverage in (0.5, 0.65, 0.8, 0.9)
        },
        "diagnosis_confusion": {
            true: dict(Counter(
                prediction for label, prediction in zip(labels, predictions) if label == true
            ))
            for true in sorted(set(labels))
        },
        "binary_counts": {
            "true_positive": sum(y and p for y, p in zip(malignant_true, malignant_prediction)),
            "false_negative": sum(y and not p for y, p in zip(malignant_true, malignant_prediction)),
            "true_negative": sum(not y and not p for y, p in zip(malignant_true, malignant_prediction)),
            "false_positive": sum(not y and p for y, p in zip(malignant_true, malignant_prediction)),
        },
    }


def evaluate_candidate(key: str, *, manifest: Path = DEV_MANIFEST) -> dict[str, object]:
    import joblib
    import numpy as np

    if key not in CANDIDATES:
        raise ValueError(f"unknown candidate {key!r}")
    records = load_manifest(manifest)
    if {record.split for record in records} != {"train", "calibration"}:
        raise ValueError("evaluation requires the test-redacted development manifest")
    feature_dir = FEATURE_ROOT / key
    feature_meta = json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    manifest_sha = manifest.with_suffix(manifest.suffix + ".sha256").read_text(
        encoding="ascii"
    ).strip()
    if feature_meta["development_manifest_sha256"] != manifest_sha:
        raise ValueError("feature/manifest hash mismatch")
    features = np.load(feature_dir / "features.npy", mmap_mode="r")
    if len(features) != len(records):
        raise ValueError("feature row count does not match manifest")
    train_idx = np.asarray([i for i, row in enumerate(records) if row.split == "train"])
    cal_idx = np.asarray([i for i, row in enumerate(records) if row.split == "calibration"])
    probes = _fit_probes(
        features[train_idx], [records[i].label for i in train_idx],
        [records[i].cluster_id for i in train_idx],
    )

    diag = probes["diagnosis"]
    decision = diag.decision_function(features[cal_idx])
    order = [list(diag.classes_).index(label) for label in probes["diagnosis_classes"]]
    probabilities = _softmax(decision[:, order], probes["temperature"])
    classes = list(probes["diagnosis_classes"])
    predictions = [classes[index] for index in probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    malignant_raw = probes["malignant"].decision_function(features[cal_idx])
    malignant_score = probes["platt"].predict_proba(malignant_raw[:, None])[:, 1]
    y_malignant = np.asarray([records[i].label in MALIGNANT for i in cal_idx], dtype=bool)
    point_90 = _threshold_at_specificity(y_malignant, malignant_score, 0.90)
    point_80 = _threshold_at_specificity(y_malignant, malignant_score, 0.80)
    rows = [
        {
            "case_id": records[index].case_id,
            "cluster_id": records[index].cluster_id,
            "label": records[index].label,
            "prediction": prediction,
            "confidence": float(conf),
            "malignant_true": bool(records[index].label in MALIGNANT),
            "malignant_score": float(mal_score),
        }
        for index, prediction, conf, mal_score in zip(
            cal_idx, predictions, confidence, malignant_score
        )
    ]
    output = FEATURE_ROOT.parent / "results" / key
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "calibration_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    joblib.dump(probes, output / "probes.joblib")
    result: dict[str, object] = {
        "protocol": "ham10000-derm-encoder-tournament-v2",
        "evidence_level": "development/calibration screening; not final-test evidence",
        "model_key": key,
        "model_id": CANDIDATES[key].model_id,
        "development_manifest_sha256": manifest_sha,
        "test_records_loaded": 0,
        "training": {
            "n_images": len(train_idx),
            "n_lesions": len({records[i].cluster_id for i in train_idx}),
            "lesion_equal_weighting": True,
            "cv": "5-fold StratifiedGroupKFold by lesion",
            "c_grid": list(C_GRID),
            "selected_c": probes["selected_c"],
            "cv_scores": probes["cv_scores"],
            "probability_calibration": {
                "diagnosis": "scalar temperature on training OOF logits",
                "diagnosis_temperature": probes["temperature"],
                "malignant": "Platt scaling on training OOF scores",
            },
        },
        "operating_points": {"specificity_90": point_90, "specificity_80": point_80},
        "calibration_metrics": _metrics(rows, float(point_90["threshold"])),
        "feature_provenance": feature_meta,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    feature_meta = extract_candidate(args.model, batch_size=args.batch_size)
    print(json.dumps({"features": feature_meta}, indent=2, sort_keys=True), flush=True)
    if not args.extract_only:
        result = evaluate_candidate(args.model)
        print(json.dumps({"result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
