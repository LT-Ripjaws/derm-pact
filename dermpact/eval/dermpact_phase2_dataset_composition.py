"""Phase 2.2G hybrid dataset-composition experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import dermpact_phase2_bcn_training_features as bcn_features
from . import dermpact_phase2_hybrid_confirmation as confirmation
from . import dermpact_phase2_hybrid_screen as screen
from .dermpact_phase2_baselines import _read_jsonl, _write_jsonl, sha256_file
from .dermpact_phase2_vector_bootstrap import (
    balanced_accuracy_samples,
    binary_auc_samples,
    interval,
    stratified_bootstrap_weights,
)


ROOT = screen.ROOT
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_2/dataset_composition_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_2/dataset_composition_v1"
PROTOCOL_PATH = OUTPUT_DIR / "dataset_composition_protocol.json"
FOLD_PATH = OUTPUT_DIR / "bcn_training_folds_v1.json"
COMPARISON_PATH = OUTPUT_DIR / "dataset_composition_comparisons.json"
RECEIPT_PATH = OUTPUT_DIR / "dataset_composition_receipt.json"
BCN_FEATURES = bcn_features.OUTPUT_FEATURES
BCN_FEATURE_MANIFEST = bcn_features.OUTPUT_MANIFEST
BCN_FEATURE_PROTOCOL = bcn_features.PROTOCOL_PATH
BCN_FEATURE_RECEIPT = bcn_features.RECEIPT_PATH
REFERENCE_DIR = ROOT / "research/manifests/dermpact_phase2_2/hybrid_duration_repair_v1"
REFERENCE_EXPERIMENT_DIR = (
    ROOT / "research/experiments/dermpact_phase2_2/hybrid_duration_repair_v1"
)
REFERENCE_PROTOCOL = REFERENCE_DIR / "hybrid_duration_repair_protocol.json"
REFERENCE_RECEIPT = REFERENCE_DIR / "hybrid_duration_repair_receipt.json"

CONFIG_ID = "lr0p0003_equal_family"
ARMS = ("derm_only", "derm_ham_bcn")
REFERENCE_ARM = "derm_ham"
ALL_ARMS = ("derm_only", REFERENCE_ARM, "derm_ham_bcn")
SEEDS = confirmation.SEEDS
EPOCHS = 12
N_FOLDS = screen.N_FOLDS
N_BOOT = 2000
BOOTSTRAP_SEED = 20261127
BCN_FOLD_SEED = 20261128
TORCH_THREADS = 12
PROTECTION_MARGIN = 0.02

BCN_TO_GLOBAL = {
    "melanoma": "melanoma",
    "nevus": "nevus",
    "bcc": "basal_cell_carcinoma",
    "scc": "squamous_cell_carcinoma",
    "akiec": "actinic_keratosis_or_intraepithelial_carcinoma",
    "bkl": "seborrheic_keratosis_or_benign_keratosis",
    "df": "dermatofibroma",
    "vasc": "vascular_lesion",
}
BCN_PROBABILITY_MAP = {
    label: (global_label,) for label, global_label in BCN_TO_GLOBAL.items()
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_result_path(seed: int) -> Path:
    return REFERENCE_DIR / f"{CONFIG_ID}_seed_{seed}_result.json"


def _result_path(arm: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{arm}_seed_{seed}_result.json"


def _bcn_fold_assignment(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels_by_lesion: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_lesion[str(row["cluster_id"])].add(str(row["label"]))
    conflicts = [key for key, labels in labels_by_lesion.items() if len(labels) != 1]
    if conflicts:
        raise AssertionError(f"BCN lesions have conflicting labels: {conflicts[:3]}")
    lesions = sorted(labels_by_lesion)
    labels = np.asarray([next(iter(labels_by_lesion[key])) for key in lesions])
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=BCN_FOLD_SEED
    )
    fold_by_lesion: dict[str, int] = {}
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(lesions)), labels)):
        for index in valid:
            fold_by_lesion[lesions[int(index)]] = fold
    if set(fold_by_lesion) != set(lesions):
        raise AssertionError("BCN fold assignment is incomplete")
    row_folds = np.asarray(
        [fold_by_lesion[str(row["cluster_id"])] for row in rows], dtype="int64"
    )
    assignments = [
        {
            "cluster_id": lesion,
            "label": next(iter(labels_by_lesion[lesion])),
            "fold": int(fold_by_lesion[lesion]),
        }
        for lesion in lesions
    ]
    return row_folds, assignments


def _write_fold_manifest(rows: Sequence[Mapping[str, Any]]) -> str:
    row_folds, assignments = _bcn_fold_assignment(rows)
    image_counts = Counter(int(value) for value in row_folds)
    lesion_counts = Counter(int(row["fold"]) for row in assignments)
    label_fold_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in assignments:
        label_fold_counts[str(row["label"])][int(row["fold"])] += 1
    if set(image_counts) != set(range(N_FOLDS)) or set(lesion_counts) != set(range(N_FOLDS)):
        raise AssertionError("BCN fold set changed")
    payload = {
        "protocol": "dermpact-phase2-bcn-training-folds-v1",
        "status": "FROZEN_BEFORE_DATASET_COMPOSITION_TRAINING",
        "method": "lesion-level stratified five-fold assignment",
        "random_seed": BCN_FOLD_SEED,
        "n_images": len(rows),
        "n_lesions": len(assignments),
        "image_count_by_fold": {str(k): image_counts[k] for k in range(N_FOLDS)},
        "lesion_count_by_fold": {str(k): lesion_counts[k] for k in range(N_FOLDS)},
        "lesion_label_count_by_fold": {
            label: {str(k): counts[k] for k in range(N_FOLDS)}
            for label, counts in sorted(label_fold_counts.items())
        },
        "assignments": assignments,
    }
    return screen._write_json(FOLD_PATH, payload)


def _load_bcn_data() -> dict[str, Any]:
    rows, _ = bcn_features._source_and_training_rows()
    manifest = _load_json(BCN_FEATURE_MANIFEST)
    if manifest["allowed_role"] != "training only":
        raise AssertionError("BCN feature role changed")
    if manifest["record_identity_sha256"] != screen._canonical_hash(
        bcn_features._identity(rows)
    ):
        raise AssertionError("BCN training feature order changed")
    if manifest["feature_file_sha256"] != sha256_file(BCN_FEATURES):
        raise AssertionError("BCN training feature hash mismatch")
    features = np.load(BCN_FEATURES, mmap_mode="r")
    if features.shape != (bcn_features.EXPECTED_TRAIN_IMAGES, 512):
        raise AssertionError(f"unexpected BCN training feature shape: {features.shape}")
    folds, assignments = _bcn_fold_assignment(rows)
    global_targets = np.asarray(
        [screen.GLOBAL_INDEX[BCN_TO_GLOBAL[str(row["label"])]] for row in rows],
        dtype="int64",
    )
    melanoma = np.asarray(
        [str(row["label"]) == "melanoma" for row in rows], dtype="float32"
    )
    return {
        "bcn_rows": rows,
        "bcn_features": features,
        "bcn_folds": folds,
        "bcn_fold_assignments": assignments,
        "bcn_global": global_targets,
        "bcn_melanoma": melanoma,
        "bcn_feature_manifest": manifest,
    }


def _data() -> dict[str, Any]:
    return {**screen._data(), **_load_bcn_data()}


def _training_weights(data: Mapping[str, Any], fold: int) -> dict[str, np.ndarray]:
    weights = screen._training_weights(data, fold)
    if "bcn_rows" not in data:
        return weights
    train = data["bcn_folds"] != fold
    groups = np.asarray(
        [str(row["cluster_id"]) for row in data["bcn_rows"]]
    )
    diagnosis_labels = np.asarray(
        [screen.GLOBAL_CLASSES[int(value)] for value in data["bcn_global"]]
    )
    weights["bcn_diagnosis"] = np.ones(len(groups), dtype="float32")
    weights["bcn_melanoma"] = np.ones(len(groups), dtype="float32")
    weights["bcn_diagnosis"][train] = screen._balanced_weights(
        diagnosis_labels[train], groups[train]
    ).astype("float32")
    weights["bcn_melanoma"][train] = screen._balanced_weights(
        data["bcn_melanoma"][train], groups[train]
    ).astype("float32")
    return weights


def _bcn_batch_loss(
    model: screen.HybridSharedHead,
    indices: np.ndarray,
    data: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(
        screen._tensor(
            np.asarray(data["bcn_features"][indices], dtype="float32"),
            dtype=torch.float32,
        )
    )
    return screen.multitask_loss(
        outputs,
        dataset="ham",
        diagnosis=screen._tensor(data["bcn_global"][indices], dtype=torch.long),
        melanoma=screen._tensor(data["bcn_melanoma"][indices], dtype=torch.float32),
        diagnosis_weight=screen._tensor(
            weights["bcn_diagnosis"][indices], dtype=torch.float32
        ),
        melanoma_weight=screen._tensor(
            weights["bcn_melanoma"][indices], dtype=torch.float32
        ),
        loss_weighting="equal_family",
    )


def _bcn_prediction_rows(
    data: Mapping[str, Any],
    indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    classes, probabilities = screen._mapped_probabilities(
        predictions["universal"], BCN_PROBABILITY_MAP, renormalize=True
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for local_index, row_index in enumerate(indices):
        lesion = str(data["bcn_rows"][int(row_index)]["cluster_id"])
        grouped[lesion].append(local_index)
    result = []
    for lesion, local_indices in sorted(grouped.items()):
        source_indices = [int(indices[index]) for index in local_indices]
        labels = {str(data["bcn_rows"][index]["label"]) for index in source_indices}
        if len(labels) != 1:
            raise AssertionError("BCN lesion has conflicting labels")
        mapped = probabilities[local_indices].mean(axis=0)
        universal = predictions["universal"][local_indices].mean(axis=0)
        result.append(
            {
                "id": lesion,
                "label": next(iter(labels)),
                "n_images": len(local_indices),
                "diagnosis_probability": {
                    label: float(mapped[position])
                    for position, label in enumerate(classes)
                },
                "melanoma_probability": float(
                    predictions["melanoma"][local_indices].mean()
                ),
                "universal_true": screen.GLOBAL_CLASSES[
                    int(data["bcn_global"][source_indices[0]])
                ],
                "universal_probability": {
                    label: float(universal[position])
                    for position, label in enumerate(screen.GLOBAL_CLASSES)
                },
                "excluded_other_probability_mass": float(
                    universal[screen.GLOBAL_INDEX["other"]]
                ),
            }
        )
    return result


def _steps_per_epoch(data: Mapping[str, Any], fold: int) -> int:
    # Match the already completed Derm7pt+HAM reference arm exactly.
    return max(
        math.ceil(int(np.sum(data["ham_folds"] != fold)) / screen.BATCH_SIZE),
        math.ceil(int(np.sum(data["derm_folds"] != fold)) / screen.BATCH_SIZE),
    )


def train_fold(
    arm: str,
    *,
    fold: int,
    seed: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    initialization_seed = seed + fold
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    ham_valid = np.flatnonzero(data["ham_folds"] == fold)
    derm_train = np.flatnonzero(data["derm_folds"] != fold)
    derm_valid = np.flatnonzero(data["derm_folds"] == fold)
    if arm == "derm_ham_bcn":
        ham_train = np.flatnonzero(data["ham_folds"] != fold)
        bcn_train = np.flatnonzero(data["bcn_folds"] != fold)
        bcn_valid = np.flatnonzero(data["bcn_folds"] == fold)
    weights = _training_weights(data, fold)
    model = screen.HybridSharedHead()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=screen.WEIGHT_DECAY
    )
    steps = _steps_per_epoch(data, fold)
    scheduler = screen._scheduler(optimizer, EPOCHS * steps)
    history = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        derm_batches = screen._cyclic_batches(
            derm_train,
            steps=steps,
            batch_size=screen.BATCH_SIZE,
            rng=np.random.default_rng(seed + fold * 1000 + epoch * 2 + 2),
        )
        if arm == "derm_ham_bcn":
            ham_batches = screen._cyclic_batches(
                ham_train,
                steps=steps,
                batch_size=screen.BATCH_SIZE,
                rng=np.random.default_rng(seed + fold * 1000 + epoch * 2 + 1),
            )
            bcn_batches = screen._cyclic_batches(
                bcn_train,
                steps=steps,
                batch_size=screen.BATCH_SIZE,
                rng=np.random.default_rng(seed + fold * 1000 + epoch * 2 + 3),
            )
        losses = []
        component_sum: dict[str, float] = defaultdict(float)
        model.train()
        epoch_started = time.perf_counter()
        for step, derm_indices in enumerate(derm_batches):
            optimizer.zero_grad(set_to_none=True)
            derm_loss, derm_components = screen._batch_loss(
                model,
                dataset="derm",
                indices=derm_indices,
                data=data,
                weights=weights,
                loss_weighting="equal_family",
            )
            if arm == "derm_only":
                loss = derm_loss
                components = {f"derm_{k}": v for k, v in derm_components.items()}
            else:
                ham_indices = next(ham_batches)
                bcn_indices = next(bcn_batches)
                ham_loss, ham_components = screen._batch_loss(
                    model,
                    dataset="ham",
                    indices=ham_indices,
                    data=data,
                    weights=weights,
                    loss_weighting="equal_family",
                )
                bcn_loss, bcn_components = _bcn_batch_loss(
                    model, bcn_indices, data, weights
                )
                loss = 0.5 * derm_loss + 0.25 * ham_loss + 0.25 * bcn_loss
                components = {
                    **{f"derm_{k}": v for k, v in derm_components.items()},
                    **{f"ham_{k}": v for k, v in ham_components.items()},
                    **{f"bcn_{k}": v for k, v in bcn_components.items()},
                }
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss: {arm}/seed{seed}/fold{fold}/epoch{epoch}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), screen.GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            for name, value in components.items():
                component_sum[name] += float(value.detach())
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(losses)),
                "last_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "component_means": {
                    key: value / steps for key, value in component_sum.items()
                },
            }
        )

    prediction_sets = {
        "ham": (data["ham_features"], ham_valid),
        "derm7pt": (data["derm_features"], derm_valid),
    }
    if arm == "derm_ham_bcn":
        prediction_sets["bcn_train_oof"] = (data["bcn_features"], bcn_valid)
    predictions = {
        name: confirmation._predict_with_logits(model, features, indices)
        for name, (features, indices) in prediction_sets.items()
    }
    fold_dir = EXPERIMENT_DIR / arm / f"seed_{seed}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / f"epoch_{EPOCHS}_state.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)
    reload_difference = confirmation._reload_max_abs_difference(
        model, checkpoint, data["derm_features"][derm_valid]
    )
    if reload_difference != 0.0:
        raise AssertionError(f"checkpoint reload changed logits by {reload_difference}")
    raw_artifacts = {}
    for name, (_, indices) in prediction_sets.items():
        raw_path = fold_dir / f"{name}_validation_raw_logits.npz"
        values = predictions[name]
        raw_artifacts[name] = {
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": confirmation._save_npz(
                raw_path,
                row_indices=np.asarray(indices, dtype=np.int64),
                diagnosis_logits=np.asarray(values["diagnosis_logits"], dtype=np.float32),
                melanoma_logits=np.asarray(values["melanoma_logits"], dtype=np.float32),
                criterion_logits=np.asarray(values["criterion_logits"], dtype=np.float32),
            ),
        }
    result = {
        "fold": fold,
        "seed": seed,
        "initialization_seed": initialization_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "steps_per_epoch": steps,
        "history": history,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint),
            "reload_max_abs_logit_difference": reload_difference,
        },
        "raw_logits": raw_artifacts,
        "ham_rows": screen._ham_prediction_rows(data, ham_valid, predictions["ham"]),
        "derm_rows": screen._derm_prediction_rows(
            data, derm_valid, predictions["derm7pt"]
        ),
    }
    if arm == "derm_ham_bcn":
        result["bcn_rows"] = _bcn_prediction_rows(
            data, bcn_valid, predictions["bcn_train_oof"]
        )
    return result


def _bcn_metrics(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    truth = [str(row["label"]) for row in rows]
    predicted = confirmation._diagnosis_predictions(rows)
    melanoma_truth = [int(label == "melanoma") for label in truth]
    melanoma_scores = [float(row["melanoma_probability"]) for row in rows]
    weights = stratified_bootstrap_weights(
        truth, n_boot=N_BOOT, seed=BOOTSTRAP_SEED + 100
    )
    balanced = float(balanced_accuracy_score(truth, predicted))
    melanoma_auc = float(roc_auc_score(melanoma_truth, melanoma_scores))
    accuracy = float(accuracy_score(truth, predicted))
    accuracy_samples = (
        weights * (np.asarray(truth) == np.asarray(predicted))[None, :]
    ).sum(axis=1) / weights.sum(axis=1)
    return {
        "n_lesions": len(rows),
        "n_images": sum(int(row["n_images"]) for row in rows),
        "accuracy_interval": interval(accuracy_samples, accuracy, n_boot=N_BOOT),
        "balanced_accuracy_interval": interval(
            balanced_accuracy_samples(truth, predicted, weights),
            balanced,
            n_boot=N_BOOT,
        ),
        "melanoma_auc_interval": interval(
            binary_auc_samples(melanoma_truth, melanoma_scores, weights),
            melanoma_auc,
            n_boot=N_BOOT,
        ),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "seed": seed,
    }


def run_arm_seed(arm: str, seed: int, *, protocol_sha256: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    if seed not in SEEDS:
        raise ValueError(seed)
    result_path = _result_path(arm, seed)
    sidecar = result_path.with_suffix(".json.sha256")
    if result_path.exists() and sidecar.exists():
        expected = sidecar.read_text(encoding="ascii").strip()
        existing = _load_json(result_path)
        if sha256_file(result_path) == expected and existing.get(
            "protocol_sha256"
        ) == protocol_sha256:
            return {
                "arm": arm,
                "seed": seed,
                "status": "reused_verified_result",
                "result_path": str(result_path.relative_to(ROOT)),
                "result_sha256": expected,
            }
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    data = _data() if arm == "derm_ham_bcn" else screen._data()
    frozen_ham = _read_jsonl(screen.FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(screen.FROZEN_DERM_PREDICTIONS)
    ham_rows: list[dict[str, Any]] = []
    derm_rows: list[dict[str, Any]] = []
    bcn_rows: list[dict[str, Any]] = []
    fold_results = []
    started = time.perf_counter()
    for fold in range(N_FOLDS):
        fold_result = train_fold(arm, fold=fold, seed=seed, data=data)
        ham_rows.extend(fold_result.pop("ham_rows"))
        derm_rows.extend(fold_result.pop("derm_rows"))
        bcn_rows.extend(fold_result.pop("bcn_rows", []))
        fold_results.append(fold_result)
        print(
            f"[{arm}/seed{seed}] fold {fold + 1}/{N_FOLDS} complete in "
            f"{fold_result['elapsed_seconds']:.2f}s",
            flush=True,
        )
    ham_rows = screen._order_like_reference(ham_rows, frozen_ham)
    derm_rows = screen._order_like_reference(derm_rows, frozen_derm)
    seed_dir = EXPERIMENT_DIR / arm / f"seed_{seed}"
    predictions = {
        "ham": {
            "path": str((seed_dir / "ham_oof_lesion_predictions.jsonl").relative_to(ROOT))
        },
        "derm7pt": {
            "path": str((seed_dir / "derm7pt_oof_predictions.jsonl").relative_to(ROOT))
        },
    }
    predictions["ham"]["sha256"] = _write_jsonl(
        ROOT / predictions["ham"]["path"], ham_rows
    )
    predictions["derm7pt"]["sha256"] = _write_jsonl(
        ROOT / predictions["derm7pt"]["path"], derm_rows
    )
    metrics = confirmation._seed_metrics(ham_rows, derm_rows, seed=seed)
    if arm == "derm_ham_bcn":
        bcn_rows = sorted(bcn_rows, key=lambda row: str(row["id"]))
        if len(bcn_rows) != bcn_features.EXPECTED_TRAIN_LESIONS:
            raise AssertionError("BCN OOF lesion count changed")
        path = seed_dir / "bcn_training_oof_lesion_predictions.jsonl"
        predictions["bcn_train_oof"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _write_jsonl(path, bcn_rows),
        }
        metrics["bcn_train_oof"] = _bcn_metrics(bcn_rows, seed=seed)
    result = {
        "protocol": "dermpact-phase2-dataset-composition-seed-result-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "evidence_level": "development-only OOF dataset-composition experiment",
        "arm": arm,
        "candidate": screen.CANDIDATE,
        "model_id": screen.MODEL_ID,
        "seed": seed,
        "fixed_training": {
            "epochs": EPOCHS,
            "folds": N_FOLDS,
            "batch_size_per_dataset": screen.BATCH_SIZE,
            "learning_rate": 3e-4,
            "loss_weighting": "equal_family",
            "adapter_dim": screen.ADAPTER_DIM,
            "weight_decay": screen.WEIGHT_DECAY,
            "early_stopping": False,
            "dataset_loss_weights": (
                {"derm7pt": 1.0}
                if arm == "derm_only"
                else {"derm7pt": 0.5, "ham": 0.25, "bcn": 0.25}
            ),
            "optimizer_steps_match_reference_arm": True,
        },
        **metrics,
        "fold_training": fold_results,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": predictions,
        "hard_scope_checks": {
            "bcn_training_images_loaded": (
                bcn_features.EXPECTED_TRAIN_IMAGES if arm == "derm_ham_bcn" else 0
            ),
            "bcn_validation_calibration_feature_or_image_rows_loaded": 0,
            "bcn_final_test_feature_or_image_rows_loaded": 0,
            "bcn_protected_labels_used_for_training_or_scoring": 0,
            "bcn_frozen_split_metadata_verified": True,
            "ham_calibration_records_loaded": 0,
            "other_final_test_records_loaded": 0,
            "derm7pt_official_test_records_loaded": 0,
            "ph2_records_loaded": 0,
        },
    }
    result_hash = screen._write_json(result_path, result)
    return {
        "arm": arm,
        "seed": seed,
        "status": "completed",
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": result_hash,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _arm_result(arm: str, seed: int) -> dict[str, Any]:
    path = _reference_result_path(seed) if arm == REFERENCE_ARM else _result_path(arm, seed)
    sidecar = path.with_suffix(".json.sha256")
    if sha256_file(path) != sidecar.read_text(encoding="ascii").strip():
        raise AssertionError(f"result hash mismatch: {arm}/seed{seed}")
    return _load_json(path)


def _arm_rows(arm: str, seed: int, dataset: str) -> list[dict[str, Any]]:
    result = _arm_result(arm, seed)
    artifact = result["predictions"][dataset]
    path = ROOT / artifact["path"]
    if sha256_file(path) != artifact["sha256"]:
        raise AssertionError(f"prediction hash mismatch: {arm}/seed{seed}/{dataset}")
    return _read_jsonl(path)


def _point(result: Mapping[str, Any], dataset: str, metric: str) -> float:
    if dataset == "derm7pt" and metric == "primary":
        return float(result[dataset]["eight_output_mean_auc_interval"]["estimate"])
    key = "diagnosis_balanced_accuracy_interval" if (
        dataset == "derm7pt" and metric == "balanced_accuracy"
    ) else "balanced_accuracy_interval"
    if metric == "balanced_accuracy":
        return float(result[dataset][key]["estimate"])
    if dataset == "derm7pt" and metric == "melanoma_auc":
        return float(result[dataset]["output_auc"]["MEL"])
    if dataset == "ham" and metric == "melanoma_auc":
        return float(result[dataset]["melanoma_auc_interval"]["estimate"])
    raise ValueError((dataset, metric))


def _comparison(
    candidate: str,
    baseline: str,
    *,
    dataset: str,
    metric: str,
    weights: np.ndarray,
    seed_draws: np.ndarray,
) -> dict[str, Any]:
    candidate_samples = []
    baseline_samples = []
    candidate_points = []
    baseline_points = []
    for seed in SEEDS:
        candidate_rows = _arm_rows(candidate, seed, dataset)
        baseline_rows = _arm_rows(baseline, seed, dataset)
        if [row["id"] for row in candidate_rows] != [row["id"] for row in baseline_rows]:
            raise AssertionError("paired comparison row order changed")
        candidate_samples.append(
            confirmation._metric_samples(candidate_rows, dataset=dataset, weights=weights)[metric]
        )
        baseline_samples.append(
            confirmation._metric_samples(baseline_rows, dataset=dataset, weights=weights)[metric]
        )
        candidate_points.append(_point(_arm_result(candidate, seed), dataset, metric))
        baseline_points.append(_point(_arm_result(baseline, seed), dataset, metric))
    candidate_mean = confirmation._seed_resampled_mean(candidate_samples, seed_draws)
    baseline_mean = confirmation._seed_resampled_mean(baseline_samples, seed_draws)
    point = float(np.mean(candidate_points) - np.mean(baseline_points))
    return interval(candidate_mean - baseline_mean, point, n_boot=N_BOOT)


def _bcn_aggregate(seed_draws: np.ndarray) -> dict[str, Any]:
    rows_by_seed = [_arm_rows("derm_ham_bcn", seed, "bcn_train_oof") for seed in SEEDS]
    reference_ids = [row["id"] for row in rows_by_seed[0]]
    if any([row["id"] for row in rows] != reference_ids for rows in rows_by_seed[1:]):
        raise AssertionError("BCN OOF row order differs across seeds")
    truth = [str(row["label"]) for row in rows_by_seed[0]]
    weights = stratified_bootstrap_weights(
        truth, n_boot=N_BOOT, seed=BOOTSTRAP_SEED + 100
    )
    result = {}
    for metric in ("accuracy", "balanced_accuracy", "melanoma_auc"):
        samples = []
        points = []
        for seed, rows in zip(SEEDS, rows_by_seed):
            predicted = confirmation._diagnosis_predictions(rows)
            if metric == "accuracy":
                correct = (np.asarray(truth) == np.asarray(predicted)).astype(float)
                samples.append((weights * correct[None, :]).sum(axis=1) / weights.sum(axis=1))
                points.append(float(accuracy_score(truth, predicted)))
            elif metric == "balanced_accuracy":
                samples.append(balanced_accuracy_samples(truth, predicted, weights))
                points.append(float(balanced_accuracy_score(truth, predicted)))
            else:
                binary_truth = [int(label == "melanoma") for label in truth]
                scores = [float(row["melanoma_probability"]) for row in rows]
                samples.append(binary_auc_samples(binary_truth, scores, weights))
                points.append(float(roc_auc_score(binary_truth, scores)))
        mean_samples = confirmation._seed_resampled_mean(samples, seed_draws)
        result[metric] = interval(mean_samples, float(np.mean(points)), n_boot=N_BOOT)
        result[metric]["per_seed"] = {
            str(seed): value for seed, value in zip(SEEDS, points)
        }
    return result


def finalize(*, protocol_sha256: str) -> dict[str, Any]:
    _verify_protocol(protocol_sha256)
    derm_reference = _arm_rows(REFERENCE_ARM, SEEDS[0], "derm7pt")
    ham_reference = _arm_rows(REFERENCE_ARM, SEEDS[0], "ham")
    derm_weights = stratified_bootstrap_weights(
        confirmation._derm_bootstrap_strata(derm_reference),
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 1,
    )
    ham_weights = stratified_bootstrap_weights(
        [str(row["label"]) for row in ham_reference],
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 2,
    )
    seed_draws = np.random.default_rng(BOOTSTRAP_SEED + 3).integers(
        0, len(SEEDS), size=(N_BOOT, len(SEEDS))
    )
    comparisons: dict[str, Any] = {
        "protocol": "dermpact-phase2-dataset-composition-comparisons-v1",
        "protocol_sha256": protocol_sha256,
        "direction": "candidate minus baseline",
        "derm_ham_minus_derm_only": {},
        "derm_ham_bcn_minus_derm_ham": {},
    }
    for name, candidate, baseline in (
        ("derm_ham_minus_derm_only", REFERENCE_ARM, "derm_only"),
        ("derm_ham_bcn_minus_derm_ham", "derm_ham_bcn", REFERENCE_ARM),
    ):
        for dataset, metrics, weights in (
            ("derm7pt", ("primary", "balanced_accuracy", "melanoma_auc"), derm_weights),
            ("ham", ("balanced_accuracy", "melanoma_auc"), ham_weights),
        ):
            comparisons[name][dataset] = {
                metric: _comparison(
                    candidate,
                    baseline,
                    dataset=dataset,
                    metric=metric,
                    weights=weights,
                    seed_draws=seed_draws,
                )
                for metric in metrics
            }
    comparisons["bcn_combined_training_oof"] = _bcn_aggregate(seed_draws)
    violations = []
    for dataset, metrics in comparisons["derm_ham_bcn_minus_derm_ham"].items():
        for metric, result in metrics.items():
            if result["estimate"] < -PROTECTION_MARGIN and result["ci_high"] < 0:
                violations.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "delta": result["estimate"],
                        "ci_high": result["ci_high"],
                    }
                )
    comparisons["protection_margin"] = PROTECTION_MARGIN
    comparisons["material_regressions"] = violations
    comparisons["decision"] = (
        "GO_FREEZE_TRANSFER_AND_COMBINED_ARMS"
        if not violations
        else "REDESIGN_COMBINED_MIXTURE_BEFORE_PROTECTED_BCN_READ"
    )
    comparison_hash = screen._write_json(COMPARISON_PATH, comparisons)

    artifacts = {}
    checkpoint_count = raw_count = prediction_count = 0
    for arm in ARMS:
        for seed in SEEDS:
            result_path = _result_path(arm, seed)
            result = _arm_result(arm, seed)
            artifacts[f"{arm}_seed_{seed}"] = {
                "path": str(result_path.relative_to(ROOT)),
                "sha256": sha256_file(result_path),
            }
            if len(result["fold_training"]) != N_FOLDS:
                raise AssertionError("fold count changed")
            for fold in result["fold_training"]:
                if len(fold["history"]) != EPOCHS:
                    raise AssertionError("epoch count changed")
                checkpoint = ROOT / fold["checkpoint"]["path"]
                if sha256_file(checkpoint) != fold["checkpoint"]["sha256"]:
                    raise AssertionError("checkpoint hash mismatch")
                checkpoint_count += 1
                for raw in fold["raw_logits"].values():
                    if sha256_file(ROOT / raw["path"]) != raw["sha256"]:
                        raise AssertionError("raw-logit hash mismatch")
                    raw_count += 1
            for prediction in result["predictions"].values():
                if sha256_file(ROOT / prediction["path"]) != prediction["sha256"]:
                    raise AssertionError("prediction hash mismatch")
                prediction_count += 1
    receipt = {
        "protocol": "dermpact-phase2-dataset-composition-receipt-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "decision": comparisons["decision"],
        "comparison_artifact": {
            "path": str(COMPARISON_PATH.relative_to(ROOT)),
            "sha256": comparison_hash,
        },
        "trained_arms": list(ARMS),
        "reused_reference_arm": REFERENCE_ARM,
        "seeds": list(SEEDS),
        "folds_per_seed": N_FOLDS,
        "epochs": EPOCHS,
        "checkpoints_verified": checkpoint_count,
        "raw_logit_artifacts_verified": raw_count,
        "prediction_artifacts_verified": prediction_count,
        "protected_bcn_feature_or_image_rows_loaded": 0,
        "protected_bcn_labels_used_for_training_or_scoring": 0,
        "bcn_frozen_split_metadata_verified": True,
        "result_artifacts": artifacts,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    receipt_hash = screen._write_json(RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": receipt_hash}


def build_protocol() -> dict[str, Any]:
    data = _data()
    fold_hash = _write_fold_manifest(data["bcn_rows"])
    reference_artifacts = {}
    for seed in SEEDS:
        result_path = _reference_result_path(seed)
        result = _load_json(result_path)
        reference_artifacts[f"reference_result_seed_{seed}"] = {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": sha256_file(result_path),
        }
        for dataset, artifact in result["predictions"].items():
            reference_artifacts[f"reference_{dataset}_seed_{seed}"] = artifact
    protocol = {
        "protocol": "dermpact-phase2-dataset-composition-v1",
        "status": "PRESPECIFIED_BEFORE_DATASET_COMPOSITION_TRAINING",
        "question": "What is the effect of adding HAM and then BCN training data to the fixed hybrid multitask head?",
        "arms": {
            "derm_only": {
                "training_data": ["Derm7pt development"],
                "dataset_loss_weights": {"derm7pt": 1.0},
                "action": "train",
            },
            "derm_ham": {
                "training_data": ["Derm7pt development", "HAM training"],
                "dataset_loss_weights": {"derm7pt": 0.5, "ham": 0.5},
                "action": "reuse Phase 2.2D exactly",
            },
            "derm_ham_bcn": {
                "training_data": [
                    "Derm7pt development",
                    "HAM training",
                    "BCN training only",
                ],
                "dataset_loss_weights": {"derm7pt": 0.5, "ham": 0.25, "bcn": 0.25},
                "action": "train",
            },
        },
        "controlled_factors": {
            "candidate": screen.CANDIDATE,
            "model_id": screen.MODEL_ID,
            "head": "HybridSharedHead",
            "learning_rate": 3e-4,
            "loss_weighting": "equal_family",
            "epochs": EPOCHS,
            "folds": N_FOLDS,
            "seeds": list(SEEDS),
            "batch_size_per_dataset": screen.BATCH_SIZE,
            "optimizer_steps": "match the completed Derm7pt+HAM arm fold by fold",
            "early_stopping": False,
        },
        "primary_internal_comparisons": [
            "Derm7pt+HAM minus Derm7pt-only on matched Derm7pt and HAM OOF endpoints",
            "Derm7pt+HAM+BCN minus Derm7pt+HAM on matched Derm7pt and HAM OOF endpoints",
        ],
        "bcn_internal_read": {
            "permitted": "combined arm OOF metrics within BCN training role only",
            "purpose": "training sanity and preparation, not external performance",
            "metrics": ["accuracy", "balanced accuracy", "melanoma AUROC"],
        },
        "combined_arm_gate": {
            "material_regression": "candidate-minus-reference < -0.02 and its CI excludes zero",
            "pass_action": "freeze both transfer and combined arms for the later single protected BCN read",
            "fail_action": "redesign mixture without opening protected BCN rows",
        },
        "hard_scope": {
            "bcn_training_images_available_to_combined_arm": int(data["bcn_features"].shape[0]),
            "bcn_validation_calibration_feature_or_image_rows_loaded": 0,
            "bcn_final_test_feature_or_image_rows_loaded": 0,
            "bcn_protected_labels_used_for_training_or_scoring": 0,
            "bcn_frozen_split_metadata_verified": True,
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
        "artifacts": {
            "bcn_training_features": {
                "path": str(BCN_FEATURES.relative_to(ROOT)),
                "sha256": sha256_file(BCN_FEATURES),
            },
            "bcn_training_feature_manifest": {
                "path": str(BCN_FEATURE_MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(BCN_FEATURE_MANIFEST),
            },
            "bcn_feature_protocol": {
                "path": str(BCN_FEATURE_PROTOCOL.relative_to(ROOT)),
                "sha256": sha256_file(BCN_FEATURE_PROTOCOL),
            },
            "bcn_feature_receipt": {
                "path": str(BCN_FEATURE_RECEIPT.relative_to(ROOT)),
                "sha256": sha256_file(BCN_FEATURE_RECEIPT),
            },
            "bcn_training_folds": {
                "path": str(FOLD_PATH.relative_to(ROOT)),
                "sha256": fold_hash,
            },
            "reference_protocol": {
                "path": str(REFERENCE_PROTOCOL.relative_to(ROOT)),
                "sha256": sha256_file(REFERENCE_PROTOCOL),
            },
            "reference_receipt": {
                "path": str(REFERENCE_RECEIPT.relative_to(ROOT)),
                "sha256": sha256_file(REFERENCE_RECEIPT),
            },
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "tested_head_implementation": {
                "path": str(Path(screen.__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(screen.__file__).resolve()),
            },
            "tested_training_implementation": {
                "path": str(Path(confirmation.__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(confirmation.__file__).resolve()),
            },
            "requirements_lock": {
                "path": str(screen.LOCK_FILE.relative_to(ROOT)),
                "sha256": sha256_file(screen.LOCK_FILE),
            },
            **reference_artifacts,
        },
    }
    protocol_hash = screen._write_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": protocol_hash}


def _verify_protocol(protocol_sha256: str) -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != protocol_sha256:
        raise AssertionError("dataset-composition protocol hash mismatch")
    if PROTOCOL_PATH.with_suffix(".json.sha256").read_text(encoding="ascii").strip() != protocol_sha256:
        raise AssertionError("dataset-composition protocol sidecar mismatch")
    protocol = _load_json(PROTOCOL_PATH)
    for name, artifact in protocol["artifacts"].items():
        if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"protocol artifact changed: {name}")
    return protocol


def run_all() -> dict[str, Any]:
    protocol = build_protocol()
    completed = [
        run_arm_seed(arm, seed, protocol_sha256=protocol["protocol_sha256"])
        for arm in ARMS
        for seed in SEEDS
    ]
    receipt = finalize(protocol_sha256=protocol["protocol_sha256"])
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "completed": completed,
        "receipt": receipt,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    protocol = build_protocol()
    if args.protocol_only:
        payload = protocol
    elif args.finalize:
        payload = finalize(protocol_sha256=protocol["protocol_sha256"])
    elif args.arm is not None or args.seed is not None:
        if args.arm is None or args.seed is None:
            parser.error("--arm and --seed must be supplied together")
        payload = run_arm_seed(
            args.arm, args.seed, protocol_sha256=protocol["protocol_sha256"]
        )
    else:
        payload = run_all()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
