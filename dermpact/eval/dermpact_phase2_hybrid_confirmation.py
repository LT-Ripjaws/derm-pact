"""DermPact Phase 2.2C frozen-hybrid confirmation."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from . import dermpact_phase2_hybrid_screen as screen
from .dermpact_phase2_baselines import (
    _balanced_accuracy_only,
    _derm_output_aucs,
    _multiclass_metrics,
    _read_jsonl,
    _write_jsonl,
    sha256_file,
)
from .dermpact_phase2_vector_bootstrap import (
    balanced_accuracy_samples,
    binary_auc_samples,
    interval,
    mean_binary_auc_samples,
    stratified_bootstrap_weights,
)


ROOT = screen.ROOT
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_2/hybrid_confirmation_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_2/hybrid_confirmation_v1"
SCREEN_PROTOCOL = (
    ROOT
    / "research/manifests/dermpact_phase2_2/hybrid_screen_v1/hybrid_screen_protocol.json"
)
SCREEN_RECEIPT = (
    ROOT
    / "research/manifests/dermpact_phase2_2/hybrid_screen_v1/hybrid_screen_receipt.json"
)
RESOURCE_SNAPSHOT = (
    ROOT
    / "Planning discussions/phase2_audits/resource_snapshot_phase2_2c_confirmation.json"
)
VECTOR_BOOTSTRAP_MODULE = Path(__file__).with_name(
    "dermpact_phase2_vector_bootstrap.py"
)
VECTOR_BOOTSTRAP_TEST = (
    ROOT / "dermpact/tests/test_dermpact_phase2_vector_bootstrap.py"
)

CONFIG_IDS = (
    "lr0p0003_equal_family",
    "lr0p0003_equal_output",
)
SEEDS = (20260827, 20260828, 20260829)
EPOCHS = 40
N_BOOT = 2000
BOOTSTRAP_SEED = 20260927
TORCH_THREADS = 12
_BOOTSTRAP_WEIGHT_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


def _predict_with_logits(
    model: screen.HybridSharedHead,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    diagnosis = []
    melanoma = []
    criteria = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            outputs = model(
                screen._tensor(
                    np.asarray(features[batch_indices], dtype="float32"),
                    dtype=torch.float32,
                )
            )
            diagnosis.append(outputs["diagnosis"].cpu().numpy())
            melanoma.append(outputs["melanoma"].cpu().numpy())
            criteria.append(outputs["criteria"].cpu().numpy())
    diagnosis_logits = np.concatenate(diagnosis, axis=0)
    melanoma_logits = np.concatenate(melanoma, axis=0)
    criterion_logits = np.concatenate(criteria, axis=0)
    return {
        "diagnosis_logits": diagnosis_logits,
        "melanoma_logits": melanoma_logits,
        "criterion_logits": criterion_logits,
        "universal": torch.softmax(
            torch.from_numpy(diagnosis_logits), dim=1
        ).numpy(),
        "melanoma": torch.sigmoid(torch.from_numpy(melanoma_logits)).numpy(),
        "criteria": torch.sigmoid(torch.from_numpy(criterion_logits)).numpy(),
    }


def _reload_max_abs_difference(
    model: screen.HybridSharedHead, checkpoint: Path, features: np.ndarray
) -> float:
    restored = screen.HybridSharedHead()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(state, strict=True)
    model.eval()
    restored.eval()
    batch = screen._tensor(
        np.asarray(features[: min(16, len(features))], dtype="float32"),
        dtype=torch.float32,
    )
    maximum = 0.0
    with torch.inference_mode():
        original = model(batch)
        reloaded = restored(batch)
        for name in original:
            maximum = max(
                maximum,
                float(torch.max(torch.abs(original[name] - reloaded[name]))),
            )
    return maximum


def train_fold(
    config: screen.ScreenConfig,
    *,
    fold: int,
    seed: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    initialization_seed = seed + fold
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    ham_train = np.flatnonzero(data["ham_folds"] != fold)
    ham_valid = np.flatnonzero(data["ham_folds"] == fold)
    derm_train = np.flatnonzero(data["derm_folds"] != fold)
    derm_valid = np.flatnonzero(data["derm_folds"] == fold)
    weights = screen._training_weights(data, fold)
    model = screen.HybridSharedHead()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=screen.WEIGHT_DECAY,
    )
    steps_per_epoch = max(
        math.ceil(len(ham_train) / screen.BATCH_SIZE),
        math.ceil(len(derm_train) / screen.BATCH_SIZE),
    )
    scheduler = screen._scheduler(optimizer, EPOCHS * steps_per_epoch)
    history = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        ham_batches = screen._cyclic_batches(
            ham_train,
            steps=steps_per_epoch,
            batch_size=screen.BATCH_SIZE,
            rng=np.random.default_rng(seed + fold * 1000 + epoch * 2 + 1),
        )
        derm_batches = screen._cyclic_batches(
            derm_train,
            steps=steps_per_epoch,
            batch_size=screen.BATCH_SIZE,
            rng=np.random.default_rng(seed + fold * 1000 + epoch * 2 + 2),
        )
        losses = []
        component_sum: dict[str, float] = defaultdict(float)
        model.train()
        epoch_started = time.perf_counter()
        for ham_indices, derm_indices in zip(ham_batches, derm_batches):
            optimizer.zero_grad(set_to_none=True)
            ham_loss, ham_components = screen._batch_loss(
                model,
                dataset="ham",
                indices=ham_indices,
                data=data,
                weights=weights,
                loss_weighting=config.loss_weighting,
            )
            derm_loss, derm_components = screen._batch_loss(
                model,
                dataset="derm",
                indices=derm_indices,
                data=data,
                weights=weights,
                loss_weighting=config.loss_weighting,
            )
            loss = 0.5 * (ham_loss + derm_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss: {config.config_id}/seed{seed}/fold{fold}/epoch{epoch}"
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
            for name, value in ham_components.items():
                component_sum[f"ham_{name}"] += float(value.detach())
            for name, value in derm_components.items():
                component_sum[f"derm_{name}"] += float(value.detach())
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(losses)),
                "last_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "component_means": {
                    key: value / steps_per_epoch
                    for key, value in component_sum.items()
                },
            }
        )

    ham_predictions = _predict_with_logits(
        model, data["ham_features"], ham_valid
    )
    derm_predictions = _predict_with_logits(
        model, data["derm_features"], derm_valid
    )
    fold_dir = (
        EXPERIMENT_DIR
        / config.config_id
        / f"seed_{seed}"
        / f"fold_{fold}"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / f"epoch_{EPOCHS}_state.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)
    reload_difference = _reload_max_abs_difference(
        model, checkpoint, data["derm_features"][derm_valid]
    )
    if reload_difference != 0.0:
        raise AssertionError(
            f"checkpoint reload changed logits by {reload_difference}"
        )

    raw_artifacts = {}
    for dataset, indices, predictions in (
        ("ham", ham_valid, ham_predictions),
        ("derm7pt", derm_valid, derm_predictions),
    ):
        raw_path = fold_dir / f"{dataset}_validation_raw_logits.npz"
        raw_hash = _save_npz(
            raw_path,
            row_indices=np.asarray(indices, dtype=np.int64),
            diagnosis_logits=np.asarray(
                predictions["diagnosis_logits"], dtype=np.float32
            ),
            melanoma_logits=np.asarray(
                predictions["melanoma_logits"], dtype=np.float32
            ),
            criterion_logits=np.asarray(
                predictions["criterion_logits"], dtype=np.float32
            ),
        )
        raw_artifacts[dataset] = {
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": raw_hash,
        }
    return {
        "fold": fold,
        "seed": seed,
        "initialization_seed": initialization_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "steps_per_epoch": steps_per_epoch,
        "history": history,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint),
            "reload_max_abs_logit_difference": reload_difference,
        },
        "raw_logits": raw_artifacts,
        "ham_rows": screen._ham_prediction_rows(
            data, ham_valid, ham_predictions
        ),
        "derm_rows": screen._derm_prediction_rows(
            data, derm_valid, derm_predictions
        ),
    }


def _diagnosis_predictions(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    classes = sorted(rows[0]["diagnosis_probability"])
    return [
        max(
            classes,
            key=lambda label: float(row["diagnosis_probability"][label]),
        )
        for row in rows
    ]


def _derm_targets_scores(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    targets = {
        "MEL": [int(row["label"] == "MEL") for row in rows]
    }
    scores = {
        "MEL": [float(row["melanoma_probability"]) for row in rows]
    }
    for criterion in screen.CRITERIA:
        short = screen.CRITERION_SHORT[criterion]
        targets[short] = [
            int(row["criterion_true"][criterion]) for row in rows
        ]
        scores[short] = [
            float(row["criterion_probability"][criterion]) for row in rows
        ]
    return targets, scores


def _derm_bootstrap_strata(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row["label"])
        + "|"
        + "".join(
            "1" if bool(row["criterion_true"][criterion]) else "0"
            for criterion in screen.CRITERIA
        )
        for row in rows
    ]


def _cached_bootstrap_weights(
    cache_name: str, labels: Sequence[str], *, seed: int
) -> np.ndarray:
    key = (cache_name, seed)
    if key not in _BOOTSTRAP_WEIGHT_CACHE:
        _BOOTSTRAP_WEIGHT_CACHE[key] = stratified_bootstrap_weights(
            labels, n_boot=N_BOOT, seed=seed
        )
    matrix = _BOOTSTRAP_WEIGHT_CACHE[key]
    if matrix.shape[1] != len(labels):
        raise AssertionError("cached bootstrap weights have the wrong case count")
    return matrix


def _quiet_open_vocab_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    true = [str(row["universal_true"]) for row in rows]
    predicted = [
        max(
            screen.GLOBAL_CLASSES,
            key=lambda label: float(row["universal_probability"][label]),
        )
        for row in rows
    ]
    recalls = []
    for label in sorted(set(true)):
        indices = [index for index, value in enumerate(true) if value == label]
        recalls.append(
            np.mean([predicted[index] == label for index in indices])
        )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy_over_observed_truth_classes": float(np.mean(recalls)),
        "macro_f1_over_universal_vocabulary": float(
            f1_score(
                true,
                predicted,
                labels=list(screen.GLOBAL_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _seed_metrics(
    ham_rows: Sequence[Mapping[str, Any]],
    derm_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    derm_weights = _cached_bootstrap_weights(
        "seed_metrics_derm",
        _derm_bootstrap_strata(derm_rows),
        seed=BOOTSTRAP_SEED + 1,
    )
    ham_labels = [str(row["label"]) for row in ham_rows]
    ham_weights = _cached_bootstrap_weights(
        "seed_metrics_ham",
        ham_labels,
        seed=BOOTSTRAP_SEED + 2,
    )
    derm_targets, derm_scores = _derm_targets_scores(derm_rows)
    derm_output_auc = _derm_output_aucs(derm_rows)
    derm_primary = float(np.mean(list(derm_output_auc.values())))
    derm_primary_samples = mean_binary_auc_samples(
        derm_targets, derm_scores, derm_weights
    )
    derm_truth = [str(row["label"]) for row in derm_rows]
    derm_predictions = _diagnosis_predictions(derm_rows)
    derm_balanced = _balanced_accuracy_only(derm_rows)
    derm_balanced_samples = balanced_accuracy_samples(
        derm_truth, derm_predictions, derm_weights
    )

    ham_truth = [str(row["label"]) for row in ham_rows]
    ham_predictions = _diagnosis_predictions(ham_rows)
    ham_balanced = _balanced_accuracy_only(ham_rows)
    ham_balanced_samples = balanced_accuracy_samples(
        ham_truth, ham_predictions, ham_weights
    )
    ham_melanoma_truth = [int(label == "melanoma") for label in ham_truth]
    ham_melanoma_scores = [
        float(row["melanoma_probability"]) for row in ham_rows
    ]
    ham_melanoma = float(
        roc_auc_score(ham_melanoma_truth, ham_melanoma_scores)
    )
    ham_melanoma_samples = binary_auc_samples(
        ham_melanoma_truth, ham_melanoma_scores, ham_weights
    )
    return {
        "derm7pt": {
            "n_cases": len(derm_rows),
            "diagnosis": _multiclass_metrics(derm_rows),
            "diagnosis_balanced_accuracy_interval": interval(
                derm_balanced_samples, derm_balanced, n_boot=N_BOOT
            ),
            "output_auc": derm_output_auc,
            "eight_output_mean_auc_interval": interval(
                derm_primary_samples, derm_primary, n_boot=N_BOOT
            ),
            "universal_open_vocab": _quiet_open_vocab_metrics(derm_rows),
        },
        "ham": {
            "n_lesions": len(ham_rows),
            "n_images": sum(int(row["n_images"]) for row in ham_rows),
            "diagnosis": _multiclass_metrics(ham_rows),
            "balanced_accuracy_interval": interval(
                ham_balanced_samples, ham_balanced, n_boot=N_BOOT
            ),
            "melanoma_auc_interval": interval(
                ham_melanoma_samples, ham_melanoma, n_boot=N_BOOT
            ),
            "universal_open_vocab": _quiet_open_vocab_metrics(ham_rows),
            "mean_probability_mass_outside_ham7": float(
                np.mean(
                    [
                        float(row["ham_excluded_probability_mass"])
                        for row in ham_rows
                    ]
                )
            ),
        },
    }


def run_seed(
    config_id: str, seed: int, *, protocol_sha256: str
) -> dict[str, Any]:
    if config_id not in CONFIG_IDS:
        raise ValueError(config_id)
    if seed not in SEEDS:
        raise ValueError(seed)
    result_path = OUTPUT_DIR / f"{config_id}_seed_{seed}_result.json"
    if result_path.exists() and result_path.with_suffix(".json.sha256").exists():
        existing = _load_json(result_path)
        expected = result_path.with_suffix(".json.sha256").read_text(
            encoding="utf-8"
        ).strip()
        if (
            sha256_file(result_path) == expected
            and existing.get("protocol_sha256") == protocol_sha256
        ):
            return {
                "config_id": config_id,
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
    config = screen.CONFIG_BY_ID[config_id]
    data = screen._data()
    frozen_ham = _read_jsonl(screen.FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(screen.FROZEN_DERM_PREDICTIONS)
    fold_results = []
    ham_rows: list[dict[str, Any]] = []
    derm_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(screen.N_FOLDS):
        fold_result = train_fold(
            config, fold=fold, seed=seed, data=data
        )
        ham_rows.extend(fold_result.pop("ham_rows"))
        derm_rows.extend(fold_result.pop("derm_rows"))
        fold_results.append(fold_result)
        print(
            f"[{config_id}/seed{seed}] fold {fold + 1}/{screen.N_FOLDS} "
            f"complete in {fold_result['elapsed_seconds']:.2f}s",
            flush=True,
        )
    ham_rows = screen._order_like_reference(ham_rows, frozen_ham)
    derm_rows = screen._order_like_reference(derm_rows, frozen_derm)
    seed_dir = EXPERIMENT_DIR / config_id / f"seed_{seed}"
    ham_path = seed_dir / "ham_oof_lesion_predictions.jsonl"
    derm_path = seed_dir / "derm7pt_oof_predictions.jsonl"
    ham_hash = _write_jsonl(ham_path, ham_rows)
    derm_hash = _write_jsonl(derm_path, derm_rows)
    metrics = _seed_metrics(ham_rows, derm_rows, seed=seed)
    result = {
        "protocol": "dermpact-phase2-2c-hybrid-confirmation-seed-result-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "evidence_level": "development OOF confirmation seed",
        "candidate": screen.CANDIDATE,
        "model_id": screen.MODEL_ID,
        "configuration": {
            "config_id": config.config_id,
            "learning_rate": config.learning_rate,
            "loss_weighting": config.loss_weighting,
        },
        "seed": seed,
        "fixed_training": {
            "epochs": EPOCHS,
            "folds": screen.N_FOLDS,
            "batch_size_per_dataset": screen.BATCH_SIZE,
            "adapter_dim": screen.ADAPTER_DIM,
            "weight_decay": screen.WEIGHT_DECAY,
            "warmup_ratio": screen.WARMUP_RATIO,
            "gradient_clip_norm": screen.GRADIENT_CLIP_NORM,
            "early_stopping": False,
        },
        **metrics,
        "fold_training": fold_results,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": {
            "ham": {
                "path": str(ham_path.relative_to(ROOT)),
                "sha256": ham_hash,
            },
            "derm7pt": {
                "path": str(derm_path.relative_to(ROOT)),
                "sha256": derm_hash,
            },
        },
        "hard_scope_checks": {
            "ham_calibration_records_loaded": 0,
            "final_test_records_loaded": 0,
            "derm7pt_official_test_records_loaded": 0,
            "bcn20000_records_loaded": 0,
            "ph2_records_loaded": 0,
        },
    }
    result_hash = screen._write_json(result_path, result)
    return {
        "config_id": config_id,
        "seed": seed,
        "status": "completed",
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": result_hash,
        "primary": metrics["derm7pt"]["eight_output_mean_auc_interval"][
            "estimate"
        ],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _metric_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    truth = [str(row["label"]) for row in rows]
    result = {
        "balanced_accuracy": balanced_accuracy_samples(
            truth, _diagnosis_predictions(rows), weights
        )
    }
    if dataset == "derm7pt":
        targets, scores = _derm_targets_scores(rows)
        result["primary"] = mean_binary_auc_samples(targets, scores, weights)
        result["melanoma_auc"] = binary_auc_samples(
            targets["MEL"], scores["MEL"], weights
        )
    elif dataset == "ham":
        melanoma_truth = [int(label == "melanoma") for label in truth]
        melanoma_scores = [
            float(row["melanoma_probability"]) for row in rows
        ]
        result["melanoma_auc"] = binary_auc_samples(
            melanoma_truth, melanoma_scores, weights
        )
    else:
        raise ValueError(dataset)
    return result


def _seed_resampled_mean(
    samples_by_seed: Sequence[np.ndarray], seed_draws: np.ndarray
) -> np.ndarray:
    matrix = np.stack(samples_by_seed, axis=0)
    replicate = np.arange(matrix.shape[1])[:, None]
    return matrix[seed_draws, replicate].mean(axis=1)


def _point_metric(
    result: Mapping[str, Any], dataset: str, metric: str
) -> float:
    if dataset == "derm7pt" and metric == "primary":
        return float(
            result[dataset]["eight_output_mean_auc_interval"]["estimate"]
        )
    if metric == "balanced_accuracy":
        key = (
            "diagnosis_balanced_accuracy_interval"
            if dataset == "derm7pt"
            else "balanced_accuracy_interval"
        )
        return float(result[dataset][key]["estimate"])
    if metric == "melanoma_auc":
        if dataset == "derm7pt":
            return float(result[dataset]["output_auc"]["MEL"])
        return float(result[dataset]["melanoma_auc_interval"]["estimate"])
    raise ValueError((dataset, metric))


def finalize(*, protocol_sha256: str) -> dict[str, Any]:
    frozen_derm = _read_jsonl(screen.FROZEN_DERM_PREDICTIONS)
    frozen_ham = _read_jsonl(screen.FROZEN_HAM_PREDICTIONS)
    derm_weights = stratified_bootstrap_weights(
        _derm_bootstrap_strata(frozen_derm),
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 1000,
    )
    ham_weights = stratified_bootstrap_weights(
        [str(row["label"]) for row in frozen_ham],
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 1001,
    )
    baseline_samples = {
        "derm7pt": _metric_samples(
            frozen_derm, dataset="derm7pt", weights=derm_weights
        ),
        "ham": _metric_samples(
            frozen_ham, dataset="ham", weights=ham_weights
        ),
    }
    frozen_result = _load_json(screen.FROZEN_RESULT)
    baseline_points = {
        "derm7pt": {
            "primary": float(
                frozen_result["derm7pt"]["eight_output_mean_auc_interval"][
                    "estimate"
                ]
            ),
            "balanced_accuracy": float(
                frozen_result["derm7pt"][
                    "diagnosis_balanced_accuracy_interval"
                ]["estimate"]
            ),
            "melanoma_auc": float(
                frozen_result["derm7pt"]["output_auc"]["MEL"]
            ),
        },
        "ham": {
            "balanced_accuracy": float(
                frozen_result["ham"]["balanced_accuracy_interval"]["estimate"]
            ),
            "melanoma_auc": float(
                frozen_result["ham"]["melanoma_auc_interval"]["estimate"]
            ),
        },
    }
    seed_rng = np.random.default_rng(BOOTSTRAP_SEED + 2000)
    seed_draws = seed_rng.integers(
        0, len(SEEDS), size=(N_BOOT, len(SEEDS))
    )
    aggregates: dict[str, Any] = {}
    config_sample_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    result_artifacts: dict[str, Any] = {}
    eligible = []
    for config_id in CONFIG_IDS:
        seed_results = []
        seed_rows: dict[str, list[list[dict[str, Any]]]] = {
            "derm7pt": [],
            "ham": [],
        }
        result_artifacts[config_id] = {}
        for seed in SEEDS:
            result_path = OUTPUT_DIR / f"{config_id}_seed_{seed}_result.json"
            result = _load_json(result_path)
            if result.get("protocol_sha256") != protocol_sha256:
                raise AssertionError("seed result protocol hash mismatch")
            seed_results.append(result)
            result_artifacts[config_id][str(seed)] = {
                "path": str(result_path.relative_to(ROOT)),
                "sha256": sha256_file(result_path),
            }
            seed_rows["derm7pt"].append(
                _read_jsonl(ROOT / result["predictions"]["derm7pt"]["path"])
            )
            seed_rows["ham"].append(
                _read_jsonl(ROOT / result["predictions"]["ham"]["path"])
            )
        per_seed_samples: dict[str, dict[str, list[np.ndarray]]] = {
            "derm7pt": defaultdict(list),
            "ham": defaultdict(list),
        }
        for dataset, weights in (
            ("derm7pt", derm_weights),
            ("ham", ham_weights),
        ):
            for rows in seed_rows[dataset]:
                for metric, samples in _metric_samples(
                    rows, dataset=dataset, weights=weights
                ).items():
                    per_seed_samples[dataset][metric].append(samples)

        aggregate_samples: dict[str, dict[str, np.ndarray]] = {
            "derm7pt": {},
            "ham": {},
        }
        comparisons = {}
        points = {}
        for dataset in ("derm7pt", "ham"):
            points[dataset] = {}
            comparisons[dataset] = {}
            for metric, sample_list in per_seed_samples[dataset].items():
                combined = _seed_resampled_mean(sample_list, seed_draws)
                aggregate_samples[dataset][metric] = combined
                point_values = [
                    _point_metric(result, dataset, metric)
                    for result in seed_results
                ]
                point = float(np.mean(point_values))
                points[dataset][metric] = {
                    "mean": point,
                    "sample_standard_deviation": float(
                        np.std(point_values, ddof=1)
                    ),
                    "minimum": float(np.min(point_values)),
                    "maximum": float(np.max(point_values)),
                    "per_seed": {
                        str(seed): float(value)
                        for seed, value in zip(SEEDS, point_values)
                    },
                    "hierarchical_interval": interval(
                        combined, point, n_boot=N_BOOT
                    ),
                }
                difference = combined - baseline_samples[dataset][metric]
                comparisons[dataset][metric] = interval(
                    difference,
                    point - baseline_points[dataset][metric],
                    n_boot=N_BOOT,
                )
        protected = {
            "derm7pt_balanced_accuracy": comparisons["derm7pt"][
                "balanced_accuracy"
            ],
            "ham_balanced_accuracy": comparisons["ham"][
                "balanced_accuracy"
            ],
            "ham_melanoma_auc": comparisons["ham"]["melanoma_auc"],
        }
        violations = []
        for name, comparison in protected.items():
            deficit = -float(comparison["estimate"])
            if (
                deficit > screen.PROTECTION_MARGIN
                and float(comparison["ci_high"]) < 0.0
            ):
                violations.append(
                    {
                        "endpoint": name,
                        "point_deficit": deficit,
                        "candidate_minus_frozen": comparison,
                    }
                )
        primary_gain = comparisons["derm7pt"]["primary"]
        positive_primary_gain = float(primary_gain["estimate"]) > 0.0
        is_eligible = not violations and positive_primary_gain
        if is_eligible:
            eligible.append(config_id)
        aggregates[config_id] = {
            "points_and_seed_stability": points,
            "candidate_minus_frozen_hybrid": comparisons,
            "protected_violations": violations,
            "positive_primary_point_gain": positive_primary_gain,
            "eligible": is_eligible,
        }
        config_sample_cache[config_id] = aggregate_samples

    ranked = sorted(
        eligible,
        key=lambda config_id: aggregates[config_id][
            "points_and_seed_stability"
        ]["derm7pt"]["primary"]["mean"],
        reverse=True,
    )
    selected = ranked[0] if ranked else None
    direct = {}
    if len(CONFIG_IDS) == 2:
        first, second = CONFIG_IDS
        for dataset in ("derm7pt", "ham"):
            direct[dataset] = {}
            for metric in config_sample_cache[first][dataset]:
                difference = (
                    config_sample_cache[first][dataset][metric]
                    - config_sample_cache[second][dataset][metric]
                )
                point = (
                    aggregates[first]["points_and_seed_stability"][dataset][
                        metric
                    ]["mean"]
                    - aggregates[second]["points_and_seed_stability"][dataset][
                        metric
                    ]["mean"]
                )
                direct[dataset][metric] = interval(
                    difference, point, n_boot=N_BOOT
                )
    comparison_path = OUTPUT_DIR / "hybrid_confirmation_comparisons.json"
    comparison_hash = screen._write_json(
        comparison_path,
        {
            "protocol": "dermpact-phase2-2c-hybrid-confirmation-comparisons-v1",
            "protocol_sha256": protocol_sha256,
            "hierarchical_bootstrap": {
                "case_replicates": N_BOOT,
                "training_seeds": list(SEEDS),
                "seed_resampled_with_replacement": True,
                "derm_case_strata": "diagnosis plus seven-criterion outcome pattern",
                "ham_case_strata": "diagnosis class",
            },
            "aggregates": aggregates,
            "direct_equal_family_minus_equal_output": direct,
        },
    )
    status = "PASS" if selected is not None else "FAIL"
    receipt = {
        "protocol": "dermpact-phase2-2c-hybrid-confirmation-receipt-v1",
        "protocol_sha256": protocol_sha256,
        "phase": "2.2C",
        "status": status,
        "candidate": screen.CANDIDATE,
        "model_id": screen.MODEL_ID,
        "confirmed_configurations": list(CONFIG_IDS),
        "seeds": list(SEEDS),
        "folds_per_seed": screen.N_FOLDS,
        "epochs": EPOCHS,
        "selected_configuration": selected,
        "selection_meaning": (
            "loss scheme selected for the required DermLIP control and later "
            "dataset-combination experiments; not a final perception model"
        ),
        "ranked_eligible_configurations": ranked,
        "comparison_artifact": {
            "path": str(comparison_path.relative_to(ROOT)),
            "sha256": comparison_hash,
        },
        "result_artifacts": result_artifacts,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": TORCH_THREADS,
            "requirements_lock_sha256": sha256_file(screen.LOCK_FILE),
        },
        "hard_scope_checks": {
            "forbidden_records_loaded": 0,
            "final_model_selected": False,
            "ph2_predictions_generated": 0,
            "bcn_predictions_generated": 0,
            "all_checkpoint_reload_differences_zero": True,
        },
        "next_phase": (
            "Run the required DermLIP control with the selected configuration, "
            "then freeze the dataset-combination/BCN-arm protocol"
            if status == "PASS"
            else "Redesign the frozen shared head without using protected data"
        ),
    }
    receipt_path = OUTPUT_DIR / "hybrid_confirmation_receipt.json"
    receipt_hash = screen._write_json(receipt_path, receipt)
    return {
        **receipt,
        "receipt_sha256": receipt_hash,
        "aggregates": aggregates,
        "direct_comparison": direct,
    }


def build_protocol() -> dict[str, Any]:
    screen_protocol = _load_json(SCREEN_PROTOCOL)
    screen_receipt = _load_json(SCREEN_RECEIPT)
    if screen_receipt.get("status") != "PASS":
        raise AssertionError("Phase 2.2B receipt is not PASS")
    if tuple(screen_receipt.get("promoted_configurations", ())) != CONFIG_IDS:
        raise AssertionError("confirmation configurations differ from screen promotion")
    data = screen._data()
    protocol = {
        "protocol": "dermpact-phase2-2c-frozen-hybrid-confirmation-v1",
        "status": "PRESPECIFIED_BEFORE_CONFIRMATION_RESULTS",
        "starts_after": {
            "phase2_2b_protocol_sha256": sha256_file(SCREEN_PROTOCOL),
            "phase2_2b_receipt_sha256": sha256_file(SCREEN_RECEIPT),
            "phase2_2b_promoted_configurations": list(CONFIG_IDS),
            "phase2_2b_recorded_protocol_hash": sha256_file(SCREEN_PROTOCOL),
        },
        "candidate": screen.CANDIDATE,
        "model_id": screen.MODEL_ID,
        "encoder_state": "frozen; cached 512-dimensional Phase 2.1 hybrid features",
        "configurations": [
            {
                "config_id": config_id,
                "learning_rate": screen.CONFIG_BY_ID[config_id].learning_rate,
                "loss_weighting": screen.CONFIG_BY_ID[config_id].loss_weighting,
            }
            for config_id in CONFIG_IDS
        ],
        "fixed_training": {
            "epochs": EPOCHS,
            "evaluation_epoch": EPOCHS,
            "early_stopping": False,
            "folds": screen.N_FOLDS,
            "seeds": list(SEEDS),
            "batch_size_per_dataset": screen.BATCH_SIZE,
            "dataset_schedule": "one HAM batch plus one Derm7pt batch per optimizer step",
            "optimizer": "AdamW",
            "weight_decay": screen.WEIGHT_DECAY,
            "warmup_ratio": screen.WARMUP_RATIO,
            "scheduler": "cosine to 1% of maximum learning rate",
            "gradient_clip_norm": screen.GRADIENT_CLIP_NORM,
            "checkpoint_reload_gate": "maximum absolute logit difference must equal zero",
            "raw_logits_saved": True,
        },
        "statistics": {
            "bootstrap_replicates": N_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "derm_case_strata": "diagnosis plus seven-criterion outcome pattern",
            "ham_case_strata": "diagnosis class",
            "training_seed_resampling": "three seeds resampled with replacement inside each case replicate",
            "primary": "mean across seed-specific Derm7pt eight-output AUROCs",
            "seed_stability": "per-seed values, sample SD, minimum, and maximum",
        },
        "selection_rule": {
            "eligibility": (
                "positive primary point gain versus the frozen hybrid probe and no "
                "protected endpoint with point deficit >0.02 whose hierarchical "
                "95% interval is wholly below zero"
            ),
            "selection": "eligible configuration with highest mean primary endpoint",
            "protected_endpoints": [
                "Derm7pt five-class balanced accuracy",
                "HAM lesion-level seven-class balanced accuracy",
                "HAM lesion-level melanoma AUROC",
            ],
            "claim": "loss-scheme confirmation only; no final model or external claim",
        },
        "allowed_data": [
            "HAM Phase 2.1 training role",
            "Derm7pt locked 616-case development role",
        ],
        "forbidden_data": [
            "HAM calibration",
            "all final-test data",
            "Derm7pt official test",
            "BCN20000",
            "PH2",
        ],
        "hard_scope_counts": {
            "ham_feature_rows": int(data["ham_features"].shape[0]),
            "derm7pt_feature_rows": int(data["derm_features"].shape[0]),
            "forbidden_rows": 0,
        },
        "artifacts": {
            "fold_manifest": {
                "path": str(screen.FOLD_MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(screen.FOLD_MANIFEST),
            },
            "frozen_hybrid_result": {
                "path": str(screen.FROZEN_RESULT.relative_to(ROOT)),
                "sha256": sha256_file(screen.FROZEN_RESULT),
            },
            "resource_snapshot": {
                "path": str(RESOURCE_SNAPSHOT.relative_to(ROOT)),
                "sha256": sha256_file(RESOURCE_SNAPSHOT),
            },
            "requirements_lock": {
                "path": str(screen.LOCK_FILE.relative_to(ROOT)),
                "sha256": sha256_file(screen.LOCK_FILE),
            },
            "confirmation_implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "vector_bootstrap_implementation": {
                "path": str(VECTOR_BOOTSTRAP_MODULE.relative_to(ROOT)),
                "sha256": sha256_file(VECTOR_BOOTSTRAP_MODULE),
            },
            "vector_bootstrap_test": {
                "path": str(VECTOR_BOOTSTRAP_TEST.relative_to(ROOT)),
                "sha256": sha256_file(VECTOR_BOOTSTRAP_TEST),
            },
        },
    }
    path = OUTPUT_DIR / "hybrid_confirmation_protocol.json"
    protocol_hash = screen._write_json(path, protocol)
    return {**protocol, "protocol_sha256": protocol_hash}


def run_all() -> dict[str, Any]:
    protocol = build_protocol()
    completed = []
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            completed.append(
                run_seed(
                    config_id, seed, protocol_sha256=protocol["protocol_sha256"]
                )
            )
    receipt = finalize(protocol_sha256=protocol["protocol_sha256"])
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "completed": completed,
        "receipt": receipt,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--config", choices=CONFIG_IDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    protocol = build_protocol()
    if args.protocol_only:
        payload = protocol
    elif args.finalize:
        payload = finalize(protocol_sha256=protocol["protocol_sha256"])
    elif args.config or args.seed is not None:
        if not args.config or args.seed is None:
            parser.error("--config and --seed must be supplied together")
        payload = run_seed(
            args.config, args.seed, protocol_sha256=protocol["protocol_sha256"]
        )
    else:
        payload = run_all()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
