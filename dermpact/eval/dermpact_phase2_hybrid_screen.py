"""DermPact Phase 2.2B frozen-hybrid configuration screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dermpact_phase2_baselines import (
    CRITERIA,
    CRITERION_SHORT,
    FEATURE_DIR,
    ROOT,
    _balanced_accuracy_only,
    _balanced_weights,
    _derm7pt_records,
    _derm_metrics,
    _derm_output_aucs,
    _ham_primary_metrics,
    _ham_records,
    _read_jsonl,
    _stratified_indices,
    _write_jsonl,
    sha256_file,
)
from .dermpact_phase2_shared_speed import _cyclic_batches


MODULE = "dermpact.eval.dermpact_phase2_hybrid_screen"
CANDIDATE = "panderm"
MODEL_ID = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
SEED = 20260827
N_FOLDS = 5
EPOCHS = 12
BATCH_SIZE = 64
ADAPTER_DIM = 256
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.10
MIN_LR_RATIO = 0.01
GRADIENT_CLIP_NORM = 1.0
TORCH_THREADS = 12
N_BOOT = 2000
PROTECTION_MARGIN = 0.02

PHASE2_1_DIR = ROOT / "research/manifests/dermpact_phase2_1"
FOLD_MANIFEST = PHASE2_1_DIR / "dermpact_phase2_1_development_folds_v1.json"
FROZEN_RESULT = PHASE2_1_DIR / "dermpact_phase2_1_panderm_result_v1.json"
FROZEN_HAM_PREDICTIONS = (
    ROOT
    / "research/experiments/dermpact_phase2_1/results/panderm/ham_oof_lesion_predictions.jsonl"
)
FROZEN_DERM_PREDICTIONS = (
    ROOT
    / "research/experiments/dermpact_phase2_1/results/panderm/derm7pt_oof_predictions.jsonl"
)
RESOURCE_SNAPSHOT = (
    ROOT
    / "Planning discussions/phase2_audits/resource_snapshot_phase2_2_hybrid_screen.json"
)
LOCK_FILE = ROOT / "requirements-p2.lock"
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_2/hybrid_screen_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_2/hybrid_screen_v1"

GLOBAL_CLASSES = (
    "melanoma",
    "nevus",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "actinic_keratosis_or_intraepithelial_carcinoma",
    "seborrheic_keratosis_or_benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
    "other",
)
GLOBAL_INDEX = {label: index for index, label in enumerate(GLOBAL_CLASSES)}

HAM_TO_GLOBAL = {
    "melanoma": "melanoma",
    "nevus": "nevus",
    "bcc": "basal_cell_carcinoma",
    "akiec": "actinic_keratosis_or_intraepithelial_carcinoma",
    "bkl": "seborrheic_keratosis_or_benign_keratosis",
    "df": "dermatofibroma",
    "vasc": "vascular_lesion",
}
HAM_PROBABILITY_MAP = {
    "melanoma": ("melanoma",),
    "nevus": ("nevus",),
    "bcc": ("basal_cell_carcinoma",),
    "akiec": ("actinic_keratosis_or_intraepithelial_carcinoma",),
    "bkl": ("seborrheic_keratosis_or_benign_keratosis",),
    "df": ("dermatofibroma",),
    "vasc": ("vascular_lesion",),
}
DERM_HARMONIZED_PROBABILITY_MAP = {
    "MEL": ("melanoma",),
    "NEV": ("nevus",),
    "BCC": ("basal_cell_carcinoma",),
    "SK": ("seborrheic_keratosis_or_benign_keratosis",),
    "MISC": (
        "squamous_cell_carcinoma",
        "actinic_keratosis_or_intraepithelial_carcinoma",
        "dermatofibroma",
        "vascular_lesion",
        "other",
    ),
}


@dataclass(frozen=True)
class ScreenConfig:
    config_id: str
    learning_rate: float
    loss_weighting: str


CONFIGS = tuple(
    ScreenConfig(
        config_id=f"lr{str(lr).replace('.', 'p').replace('-', 'm')}_{weighting}",
        learning_rate=lr,
        loss_weighting=weighting,
    )
    for lr in (3e-4, 1e-3, 3e-3)
    for weighting in ("equal_family", "equal_output")
)
CONFIG_BY_ID = {config.config_id: config for config in CONFIGS}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    os.replace(temporary, path)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _canonical_hash(payload: Any) -> str:
    body = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def derm_raw_to_global(raw: str) -> str:
    raw = str(raw).strip().lower()
    if raw.startswith("melanoma"):
        return "melanoma"
    if "nevus" in raw:
        return "nevus"
    mapping = {
        "basal cell carcinoma": "basal_cell_carcinoma",
        "seborrheic keratosis": "seborrheic_keratosis_or_benign_keratosis",
        "dermatofibroma": "dermatofibroma",
        "vascular lesion": "vascular_lesion",
        "lentigo": "other",
        "melanosis": "other",
        "miscellaneous": "other",
    }
    if raw not in mapping:
        raise ValueError(f"unmapped Derm7pt diagnosis: {raw!r}")
    return mapping[raw]


class HybridSharedHead(nn.Module):
    def __init__(self, feature_dim: int = 512, adapter_dim: int = ADAPTER_DIM) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.diagnosis = nn.Linear(adapter_dim, len(GLOBAL_CLASSES))
        self.melanoma = nn.Linear(adapter_dim, 1)
        self.criteria = nn.ModuleList(
            nn.Linear(adapter_dim, 1) for _ in CRITERIA
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        shared = self.adapter(features)
        return {
            "diagnosis": self.diagnosis(shared),
            "melanoma": self.melanoma(shared).squeeze(-1),
            "criteria": torch.cat(
                [head(shared) for head in self.criteria], dim=1
            ),
        }


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("loss values and sample weights must be aligned vectors")
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def multitask_loss(
    outputs: Mapping[str, Tensor],
    *,
    dataset: str,
    diagnosis: Tensor,
    melanoma: Tensor,
    diagnosis_weight: Tensor,
    melanoma_weight: Tensor,
    loss_weighting: str,
    criteria: Tensor | None = None,
    criterion_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    diagnosis_loss = _weighted_mean(
        F.cross_entropy(outputs["diagnosis"], diagnosis, reduction="none"),
        diagnosis_weight,
    )
    melanoma_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            outputs["melanoma"], melanoma.float(), reduction="none"
        ),
        melanoma_weight,
    )
    if dataset == "ham":
        if criteria is not None or criterion_weights is not None:
            raise ValueError("HAM criterion labels must remain masked")
        criterion_losses = outputs["criteria"].sum(dim=0) * 0.0
        total = 0.5 * (diagnosis_loss + melanoma_loss)
    elif dataset == "derm7pt":
        if criteria is None or criterion_weights is None:
            raise ValueError("Derm7pt requires criterion labels and weights")
        if criteria.shape != outputs["criteria"].shape:
            raise ValueError("criterion target shape mismatch")
        criterion_losses = torch.stack(
            [
                _weighted_mean(
                    F.binary_cross_entropy_with_logits(
                        outputs["criteria"][:, index],
                        criteria[:, index].float(),
                        reduction="none",
                    ),
                    criterion_weights[:, index],
                )
                for index in range(len(CRITERIA))
            ]
        )
        if loss_weighting == "equal_family":
            total = (
                diagnosis_loss + melanoma_loss + criterion_losses.mean()
            ) / 3.0
        elif loss_weighting == "equal_output":
            total = (
                diagnosis_loss + melanoma_loss + criterion_losses.sum()
            ) / (2.0 + len(CRITERIA))
        else:
            raise ValueError(f"unsupported loss weighting: {loss_weighting}")
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return total, {
        "diagnosis": diagnosis_loss,
        "melanoma": melanoma_loss,
        "criteria_mean": criterion_losses.mean(),
    }


def _folds() -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json(FOLD_MANIFEST)
    ham_rows = _ham_records()
    derm_rows = _derm7pt_records()
    ham_map = {
        str(row["cluster_id"]): int(row["fold"])
        for row in payload["ham"]["folds"]
    }
    derm_map = {
        str(row["group_id"]): int(row["fold"])
        for row in payload["derm7pt"]["folds"]
    }
    if set(ham_map) != {str(row["cluster_id"]) for row in ham_rows}:
        raise AssertionError("HAM folds no longer match usable training lesions")
    if set(derm_map) != {str(row["group_id"]) for row in derm_rows}:
        raise AssertionError("Derm7pt folds no longer match development cases")
    ham_folds = np.asarray(
        [ham_map[str(row["cluster_id"])] for row in ham_rows], dtype="int64"
    )
    derm_folds = np.asarray(
        [derm_map[str(row["group_id"])] for row in derm_rows], dtype="int64"
    )
    if set(ham_folds.tolist()) != set(range(N_FOLDS)):
        raise AssertionError("HAM fold set changed")
    if set(derm_folds.tolist()) != set(range(N_FOLDS)):
        raise AssertionError("Derm7pt fold set changed")
    return ham_folds, derm_folds


def _load_feature_matrix(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    feature_path = FEATURE_DIR / CANDIDATE / dataset / "features.npy"
    manifest_path = FEATURE_DIR / CANDIDATE / dataset / "feature_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest["candidate"] != CANDIDATE:
        raise AssertionError("feature candidate changed")
    if manifest["feature_file_sha256"] != sha256_file(feature_path):
        raise AssertionError(f"{dataset} feature hash mismatch")
    features = np.load(feature_path, mmap_mode="r")
    expected_rows = 7008 if dataset == "ham" else 616
    if features.shape != (expected_rows, 512):
        raise AssertionError(f"unexpected {dataset} feature shape: {features.shape}")
    return features, manifest


def _criterion_matrix(derm_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    from .dermpact_phase2_baselines import _criterion_labels

    return np.stack(
        [_criterion_labels(derm_rows, criterion) for criterion in CRITERIA],
        axis=1,
    ).astype("float32")


def _data() -> dict[str, Any]:
    ham_rows = _ham_records()
    derm_rows = _derm7pt_records()
    ham_features, ham_feature_manifest = _load_feature_matrix("ham")
    derm_features, derm_feature_manifest = _load_feature_matrix("derm7pt")
    ham_folds, derm_folds = _folds()
    ham_global = np.asarray(
        [GLOBAL_INDEX[HAM_TO_GLOBAL[str(row["label"])]] for row in ham_rows],
        dtype="int64",
    )
    derm_global_labels = [
        derm_raw_to_global(str(row["diagnosis"]["raw"])) for row in derm_rows
    ]
    derm_global = np.asarray(
        [GLOBAL_INDEX[label] for label in derm_global_labels], dtype="int64"
    )
    ham_melanoma = np.asarray(
        [str(row["label"]) == "melanoma" for row in ham_rows], dtype="float32"
    )
    derm_melanoma = np.asarray(
        [label == "melanoma" for label in derm_global_labels], dtype="float32"
    )
    return {
        "ham_rows": ham_rows,
        "derm_rows": derm_rows,
        "ham_features": ham_features,
        "derm_features": derm_features,
        "ham_feature_manifest": ham_feature_manifest,
        "derm_feature_manifest": derm_feature_manifest,
        "ham_folds": ham_folds,
        "derm_folds": derm_folds,
        "ham_global": ham_global,
        "derm_global": derm_global,
        "ham_melanoma": ham_melanoma,
        "derm_melanoma": derm_melanoma,
        "derm_criteria": _criterion_matrix(derm_rows),
    }


def _training_weights(data: Mapping[str, Any], fold: int) -> dict[str, np.ndarray]:
    ham_train = data["ham_folds"] != fold
    derm_train = data["derm_folds"] != fold
    ham_groups = [str(row["cluster_id"]) for row in data["ham_rows"]]
    ham_diagnosis_labels = [GLOBAL_CLASSES[int(index)] for index in data["ham_global"]]
    derm_diagnosis_labels = [GLOBAL_CLASSES[int(index)] for index in data["derm_global"]]
    weights = {
        "ham_diagnosis": np.ones(len(data["ham_rows"]), dtype="float32"),
        "ham_melanoma": np.ones(len(data["ham_rows"]), dtype="float32"),
        "derm_diagnosis": np.ones(len(data["derm_rows"]), dtype="float32"),
        "derm_melanoma": np.ones(len(data["derm_rows"]), dtype="float32"),
        "derm_criteria": np.ones(
            (len(data["derm_rows"]), len(CRITERIA)), dtype="float32"
        ),
    }
    weights["ham_diagnosis"][ham_train] = _balanced_weights(
        np.asarray(ham_diagnosis_labels)[ham_train], np.asarray(ham_groups)[ham_train]
    ).astype("float32")
    weights["ham_melanoma"][ham_train] = _balanced_weights(
        data["ham_melanoma"][ham_train], np.asarray(ham_groups)[ham_train]
    ).astype("float32")
    weights["derm_diagnosis"][derm_train] = _balanced_weights(
        np.asarray(derm_diagnosis_labels)[derm_train]
    ).astype("float32")
    weights["derm_melanoma"][derm_train] = _balanced_weights(
        data["derm_melanoma"][derm_train]
    ).astype("float32")
    for index in range(len(CRITERIA)):
        weights["derm_criteria"][derm_train, index] = _balanced_weights(
            data["derm_criteria"][derm_train, index]
        ).astype("float32")
    return weights


def _tensor(values: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    tensor = torch.from_numpy(np.asarray(values))
    return tensor if dtype is None else tensor.to(dtype=dtype)


def _batch_loss(
    model: HybridSharedHead,
    *,
    dataset: str,
    indices: np.ndarray,
    data: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
    loss_weighting: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    features = data[f"{dataset}_features"]
    outputs = model(
        _tensor(np.asarray(features[indices], dtype="float32"), dtype=torch.float32)
    )
    common = {
        "outputs": outputs,
        "dataset": dataset,
        "diagnosis": _tensor(data[f"{dataset}_global"][indices], dtype=torch.long),
        "melanoma": _tensor(data[f"{dataset}_melanoma"][indices], dtype=torch.float32),
        "diagnosis_weight": _tensor(
            weights[f"{dataset}_diagnosis"][indices], dtype=torch.float32
        ),
        "melanoma_weight": _tensor(
            weights[f"{dataset}_melanoma"][indices], dtype=torch.float32
        ),
        "loss_weighting": loss_weighting,
    }
    if dataset == "derm":
        common["dataset"] = "derm7pt"
        common["criteria"] = _tensor(
            data["derm_criteria"][indices], dtype=torch.float32
        )
        common["criterion_weights"] = _tensor(
            weights["derm_criteria"][indices], dtype=torch.float32
        )
    return multitask_loss(**common)


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    warmup_steps = max(1, int(round(total_steps * WARMUP_RATIO)))

    def scale(step: int) -> float:
        current = step + 1
        if current <= warmup_steps:
            return current / warmup_steps
        progress = (current - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def _mapped_probabilities(
    universal: np.ndarray, mapping: Mapping[str, Sequence[str]], *, renormalize: bool,
) -> tuple[list[str], np.ndarray]:
    classes = sorted(mapping)
    matrix = np.stack(
        [
            universal[:, [GLOBAL_INDEX[label] for label in mapping[output]]].sum(axis=1)
            for output in classes
        ],
        axis=1,
    )
    if renormalize:
        matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
    return classes, matrix


def _predict(
    model: HybridSharedHead, features: np.ndarray, indices: np.ndarray,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    universal_blocks = []
    melanoma_blocks = []
    criterion_blocks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            outputs = model(
                _tensor(
                    np.asarray(features[batch_indices], dtype="float32"),
                    dtype=torch.float32,
                )
            )
            universal_blocks.append(torch.softmax(outputs["diagnosis"], dim=1).cpu().numpy())
            melanoma_blocks.append(torch.sigmoid(outputs["melanoma"]).cpu().numpy())
            criterion_blocks.append(torch.sigmoid(outputs["criteria"]).cpu().numpy())
    return {
        "universal": np.concatenate(universal_blocks, axis=0),
        "melanoma": np.concatenate(melanoma_blocks, axis=0),
        "criteria": np.concatenate(criterion_blocks, axis=0),
    }


def _ham_prediction_rows(
    data: Mapping[str, Any], indices: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    ham_classes, ham_probability = _mapped_probabilities(
        predictions["universal"], HAM_PROBABILITY_MAP, renormalize=True
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for local_index, row_index in enumerate(indices):
        grouped[str(data["ham_rows"][int(row_index)]["cluster_id"])].append(local_index)
    result = []
    for lesion, local_indices in sorted(grouped.items()):
        source_indices = [int(indices[index]) for index in local_indices]
        labels = {str(data["ham_rows"][index]["label"]) for index in source_indices}
        if len(labels) != 1:
            raise AssertionError("HAM lesion has conflicting labels")
        mapped = ham_probability[local_indices].mean(axis=0)
        universal = predictions["universal"][local_indices].mean(axis=0)
        result.append({
            "id": lesion,
            "label": next(iter(labels)),
            "n_images": len(local_indices),
            "diagnosis_probability": {
                label: float(mapped[position])
                for position, label in enumerate(ham_classes)
            },
            "melanoma_probability": float(
                predictions["melanoma"][local_indices].mean()
            ),
            "universal_true": GLOBAL_CLASSES[int(data["ham_global"][source_indices[0]])],
            "universal_probability": {
                label: float(universal[position])
                for position, label in enumerate(GLOBAL_CLASSES)
            },
            "ham_excluded_probability_mass": float(
                universal[GLOBAL_INDEX["squamous_cell_carcinoma"]]
                + universal[GLOBAL_INDEX["other"]]
            ),
        })
    return result


def _derm_prediction_rows(
    data: Mapping[str, Any], indices: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    derm_classes, derm_probability = _mapped_probabilities(
        predictions["universal"], DERM_HARMONIZED_PROBABILITY_MAP,
        renormalize=False,
    )
    result = []
    for local_index, row_index in enumerate(indices):
        source = data["derm_rows"][int(row_index)]
        result.append({
            "id": str(source["group_id"]),
            "label": str(source["diagnosis"]["harmonized5"]),
            "diagnosis_probability": {
                label: float(derm_probability[local_index, position])
                for position, label in enumerate(derm_classes)
            },
            "melanoma_probability": float(predictions["melanoma"][local_index]),
            "criterion_true": {
                criterion: bool(data["derm_criteria"][int(row_index), position])
                for position, criterion in enumerate(CRITERIA)
            },
            "criterion_probability": {
                criterion: float(predictions["criteria"][local_index, position])
                for position, criterion in enumerate(CRITERIA)
            },
            "universal_true": GLOBAL_CLASSES[int(data["derm_global"][int(row_index)])],
            "universal_probability": {
                label: float(predictions["universal"][local_index, position])
                for position, label in enumerate(GLOBAL_CLASSES)
            },
        })
    return result


def _open_vocab_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    true = [str(row["universal_true"]) for row in rows]
    predicted = [
        max(
            GLOBAL_CLASSES,
            key=lambda label: float(row["universal_probability"][label]),
        )
        for row in rows
    ]
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
    }


def _order_like_reference(
    rows: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): dict(row) for row in rows}
    expected = [str(row["id"]) for row in reference]
    if set(by_id) != set(expected):
        raise AssertionError("OOF IDs do not match the frozen reference")
    return [by_id[identifier] for identifier in expected]


def train_fold(
    config: ScreenConfig, fold: int, data: Mapping[str, Any]
) -> dict[str, Any]:
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    ham_train = np.flatnonzero(data["ham_folds"] != fold)
    ham_valid = np.flatnonzero(data["ham_folds"] == fold)
    derm_train = np.flatnonzero(data["derm_folds"] != fold)
    derm_valid = np.flatnonzero(data["derm_folds"] == fold)
    weights = _training_weights(data, fold)
    model = HybridSharedHead()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = max(
        math.ceil(len(ham_train) / BATCH_SIZE),
        math.ceil(len(derm_train) / BATCH_SIZE),
    )
    scheduler = _scheduler(optimizer, EPOCHS * steps_per_epoch)
    history = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        ham_batches = _cyclic_batches(
            ham_train,
            steps=steps_per_epoch,
            batch_size=BATCH_SIZE,
            rng=np.random.default_rng(SEED + fold * 1000 + epoch * 2 + 1),
        )
        derm_batches = _cyclic_batches(
            derm_train,
            steps=steps_per_epoch,
            batch_size=BATCH_SIZE,
            rng=np.random.default_rng(SEED + fold * 1000 + epoch * 2 + 2),
        )
        losses = []
        component_sum = defaultdict(float)
        model.train()
        epoch_started = time.perf_counter()
        for ham_indices, derm_indices in zip(ham_batches, derm_batches):
            optimizer.zero_grad(set_to_none=True)
            ham_loss, ham_components = _batch_loss(
                model,
                dataset="ham",
                indices=ham_indices,
                data=data,
                weights=weights,
                loss_weighting=config.loss_weighting,
            )
            derm_loss, derm_components = _batch_loss(
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
                    f"non-finite loss: {config.config_id}/fold{fold}/epoch{epoch}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM
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
        history.append({
            "epoch": epoch + 1,
            "mean_training_loss": float(np.mean(losses)),
            "last_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "component_means": {
                key: value / steps_per_epoch for key, value in component_sum.items()
            },
        })

    ham_predictions = _predict(model, data["ham_features"], ham_valid)
    derm_predictions = _predict(model, data["derm_features"], derm_valid)
    fold_dir = EXPERIMENT_DIR / config.config_id / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / "epoch_12_state.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)
    return {
        "fold": fold,
        "elapsed_seconds": time.perf_counter() - started,
        "steps_per_epoch": steps_per_epoch,
        "history": history,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint),
        },
        "ham_rows": _ham_prediction_rows(data, ham_valid, ham_predictions),
        "derm_rows": _derm_prediction_rows(data, derm_valid, derm_predictions),
    }


def run_config(config_id: str) -> dict[str, Any]:
    if config_id not in CONFIG_BY_ID:
        raise ValueError(config_id)
    config = CONFIG_BY_ID[config_id]
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    data = _data()
    frozen_ham = _read_jsonl(FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(FROZEN_DERM_PREDICTIONS)
    fold_results = []
    ham_rows = []
    derm_rows = []
    started = time.perf_counter()
    for fold in range(N_FOLDS):
        result = train_fold(config, fold, data)
        ham_rows.extend(result.pop("ham_rows"))
        derm_rows.extend(result.pop("derm_rows"))
        fold_results.append(result)
        print(
            f"[{config.config_id}] fold {fold + 1}/{N_FOLDS} complete "
            f"in {result['elapsed_seconds']:.2f}s",
            flush=True,
        )
    ham_rows = _order_like_reference(ham_rows, frozen_ham)
    derm_rows = _order_like_reference(derm_rows, frozen_derm)
    config_dir = EXPERIMENT_DIR / config.config_id
    ham_path = config_dir / "ham_oof_lesion_predictions.jsonl"
    derm_path = config_dir / "derm7pt_oof_predictions.jsonl"
    ham_hash = _write_jsonl(ham_path, ham_rows)
    derm_hash = _write_jsonl(derm_path, derm_rows)
    result = {
        "protocol": "dermpact-phase2-2b-hybrid-screen-result-v1",
        "status": "PASS",
        "evidence_level": "development OOF configuration screen only",
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "configuration": asdict(config),
        "fixed_training": {
            "epochs": EPOCHS,
            "batch_size_per_dataset": BATCH_SIZE,
            "adapter_dim": ADAPTER_DIM,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "scheduler": "cosine to 1% of maximum learning rate",
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "folds": N_FOLDS,
            "seeds": [SEED],
        },
        "label_contract": {
            "global_classes": list(GLOBAL_CLASSES),
            "ham_mapping": HAM_TO_GLOBAL,
            "derm_raw_mapping_hash": _canonical_hash(
                {
                    str(row["diagnosis"]["raw"]): derm_raw_to_global(
                        str(row["diagnosis"]["raw"])
                    )
                    for row in data["derm_rows"]
                }
            ),
            "ham_criterion_labels_fabricated": False,
        },
        "ham": {
            **_ham_primary_metrics(ham_rows),
            "universal_open_vocab": _open_vocab_metrics(ham_rows),
            "mean_probability_mass_outside_ham7": float(
                np.mean([row["ham_excluded_probability_mass"] for row in ham_rows])
            ),
        },
        "derm7pt": {
            **_derm_metrics(derm_rows),
            "universal_open_vocab": _open_vocab_metrics(derm_rows),
        },
        "fold_training": fold_results,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": {
            "ham": {"path": str(ham_path.relative_to(ROOT)), "sha256": ham_hash},
            "derm7pt": {"path": str(derm_path.relative_to(ROOT)), "sha256": derm_hash},
        },
        "hard_scope_checks": {
            "ham_calibration_records_loaded": 0,
            "final_test_records_loaded": 0,
            "derm7pt_official_test_records_loaded": 0,
            "bcn20000_records_loaded": 0,
            "ph2_records_loaded": 0,
        },
    }
    result_path = OUTPUT_DIR / f"{config.config_id}_result.json"
    result_hash = _write_json(result_path, result)
    return {
        "config_id": config.config_id,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": result_hash,
        "primary": result["derm7pt"]["eight_output_mean_auc_interval"]["estimate"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _paired_interval(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    metric,
    *,
    seed_offset: int,
) -> dict[str, float]:
    if [row["id"] for row in candidate] != [row["id"] for row in baseline]:
        raise AssertionError("paired prediction rows are not aligned")
    estimate = float(metric(candidate) - metric(baseline))
    labels = [str(row["label"]) for row in candidate]
    rng = np.random.default_rng(SEED + seed_offset)
    samples = np.empty(N_BOOT, dtype=float)
    for replicate in range(N_BOOT):
        indices = _stratified_indices(labels, rng)
        samples[replicate] = metric(
            [candidate[int(index)] for index in indices]
        ) - metric([baseline[int(index)] for index in indices])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": N_BOOT,
    }


def finalize() -> dict[str, Any]:
    frozen_ham = _read_jsonl(FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(FROZEN_DERM_PREDICTIONS)
    frozen_result = _load_json(FROZEN_RESULT)

    def derm_primary(rows):
        values = _derm_output_aucs(rows)
        return float(sum(values.values()) / len(values))

    def ham_melanoma(rows):
        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(
                [int(row["label"] == "melanoma") for row in rows],
                [float(row["melanoma_probability"]) for row in rows],
            )
        )

    metrics = {
        "derm7pt_eight_output_mean_auc": ("derm", derm_primary, frozen_derm),
        "derm7pt_five_class_balanced_accuracy": (
            "derm", _balanced_accuracy_only, frozen_derm
        ),
        "ham_seven_class_balanced_accuracy": (
            "ham", _balanced_accuracy_only, frozen_ham
        ),
        "ham_melanoma_auc": ("ham", ham_melanoma, frozen_ham),
    }
    comparisons = {}
    eligible = []
    for config_index, config in enumerate(CONFIGS):
        result = _load_json(OUTPUT_DIR / f"{config.config_id}_result.json")
        candidate_rows = {
            "ham": _read_jsonl(
                ROOT / result["predictions"]["ham"]["path"]
            ),
            "derm": _read_jsonl(
                ROOT / result["predictions"]["derm7pt"]["path"]
            ),
        }
        differences = {}
        violations = []
        for metric_index, (name, (dataset, metric, baseline_rows)) in enumerate(metrics.items()):
            difference = _paired_interval(
                candidate_rows[dataset], baseline_rows, metric,
                seed_offset=100 + config_index * 10 + metric_index,
            )
            differences[name] = difference
            if name != "derm7pt_eight_output_mean_auc":
                deficit = -difference["estimate"]
                if deficit > PROTECTION_MARGIN and difference["ci_high"] < 0.0:
                    violations.append({
                        "endpoint": name,
                        "point_deficit": deficit,
                        "candidate_minus_frozen": difference,
                    })
        comparisons[config.config_id] = {
            "candidate_minus_frozen_hybrid": differences,
            "protected_violations": violations,
            "eligible": not violations,
            "primary": float(
                result["derm7pt"]["eight_output_mean_auc_interval"]["estimate"]
            ),
        }
        if not violations:
            eligible.append(config.config_id)
    ranked = sorted(
        eligible,
        key=lambda config_id: comparisons[config_id]["primary"],
        reverse=True,
    )
    promoted = ranked[:2]
    status = "PASS" if len(promoted) == 2 else "FAIL"
    comparison_path = OUTPUT_DIR / "hybrid_screen_comparisons.json"
    comparison_hash = _write_json(comparison_path, {
        "protocol": "dermpact-phase2-2b-hybrid-screen-comparisons-v1",
        "frozen_hybrid_reference": {
            "path": str(FROZEN_RESULT.relative_to(ROOT)),
            "sha256": sha256_file(FROZEN_RESULT),
            "primary": frozen_result["derm7pt"]["eight_output_mean_auc_interval"]["estimate"],
        },
        "comparisons": comparisons,
    })
    result_artifacts = {
        config.config_id: {
            "path": str(
                (OUTPUT_DIR / f"{config.config_id}_result.json").relative_to(ROOT)
            ),
            "sha256": sha256_file(OUTPUT_DIR / f"{config.config_id}_result.json"),
        }
        for config in CONFIGS
    }
    receipt = {
        "protocol": "dermpact-phase2-2b-hybrid-screen-receipt-v1",
        "phase": "2.2B",
        "status": status,
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "screened_configurations": len(CONFIGS),
        "fixed_epoch": EPOCHS,
        "promoted_configurations": promoted,
        "promotion_meaning": (
            "development configurations eligible for a separately prespecified "
            "40-epoch, five-fold, three-seed confirmation; not final model selection"
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
            "requirements_lock_sha256": sha256_file(LOCK_FILE),
        },
        "hard_scope_checks": {
            "forbidden_records_loaded": 0,
            "candidate_promoted_to_final_model": False,
            "ph2_predictions_generated": 0,
            "bcn_predictions_generated": 0,
        },
        "next_phase": (
            "Prespecify and run Phase 2.2C confirmation for the two promoted configurations"
            if status == "PASS" else
            "Redesign the frozen shared head without using final/external data"
        ),
    }
    receipt_path = OUTPUT_DIR / "hybrid_screen_receipt.json"
    receipt_hash = _write_json(receipt_path, receipt)
    return {**receipt, "receipt_sha256": receipt_hash, "comparisons": comparisons}


def build_protocol() -> dict[str, Any]:
    data = _data()
    observed_derm_mapping = {
        raw: derm_raw_to_global(raw)
        for raw in sorted({str(row["diagnosis"]["raw"]) for row in data["derm_rows"]})
    }
    protocol = {
        "protocol": "dermpact-phase2-2b-frozen-hybrid-screen-v1",
        "status": "PRESPECIFIED_BEFORE_SCREEN_RESULTS",
        "dated_amendment": {
            "date": "2026-08-27",
            "decision": "make the DermLIP--PanDerm hybrid the primary Phase 2.2B backbone",
            "effect": (
                "supersedes the Phase 2.1 development preference before Phase 2.2B "
                "results; DermLIP remains the required simpler control, and the screen "
                "does not by itself promote any final model"
            ),
        },
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "encoder_state": "frozen; cached 512-dimensional Phase 2.1 features",
        "architecture": {
            "adapter": f"LayerNorm(512)-Linear(512,{ADAPTER_DIM})-GELU-Dropout(0.1)",
            "diagnosis": "one universal nine-class projection",
            "melanoma": "one shared binary projection",
            "criteria": "seven independent binary projections; Derm7pt labels only",
        },
        "global_classes": list(GLOBAL_CLASSES),
        "label_mapping": {
            "ham": HAM_TO_GLOBAL,
            "derm7pt_raw": observed_derm_mapping,
            "ham_metric_projection": HAM_PROBABILITY_MAP,
            "derm7pt_metric_projection": DERM_HARMONIZED_PROBABILITY_MAP,
            "missing_label_imputation": False,
        },
        "configurations": [asdict(config) for config in CONFIGS],
        "fixed_training": {
            "epochs": EPOCHS,
            "evaluation_epoch": EPOCHS,
            "early_stopping": False,
            "folds": N_FOLDS,
            "seed": SEED,
            "batch_size_per_dataset": BATCH_SIZE,
            "dataset_schedule": "one HAM batch plus one Derm7pt batch per optimizer step",
            "ham_lesion_weighting": "equal lesion mass, then equal diagnosis/task class mass",
            "derm7pt_weighting": "equal case mass, then equal diagnosis/task class mass",
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "scheduler": "cosine to 1% of maximum learning rate",
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "primary_endpoint": "Derm7pt OOF mean AUROC over melanoma and seven criteria",
        "protected_endpoints": [
            "Derm7pt five-class balanced accuracy",
            "HAM lesion-level seven-class balanced accuracy",
            "HAM lesion-level melanoma AUROC",
        ],
        "promotion_rule": {
            "comparison": "same hybrid frozen logistic-probe baseline from Phase 2.1",
            "protection_margin": PROTECTION_MARGIN,
            "violation": (
                "point deficit exceeds 0.02 and paired 95% candidate-minus-frozen "
                "interval excludes zero below zero"
            ),
            "promote": "two eligible configurations with highest fixed-epoch primary endpoint",
            "claim": "configuration screen only",
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
        "artifacts": {
            "fold_manifest": {
                "path": str(FOLD_MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(FOLD_MANIFEST),
            },
            "frozen_result": {
                "path": str(FROZEN_RESULT.relative_to(ROOT)),
                "sha256": sha256_file(FROZEN_RESULT),
            },
            "hybrid_ham_features": {
                "sha256": data["ham_feature_manifest"]["feature_file_sha256"]
            },
            "hybrid_derm_features": {
                "sha256": data["derm_feature_manifest"]["feature_file_sha256"]
            },
            "resource_snapshot": {
                "path": str(RESOURCE_SNAPSHOT.relative_to(ROOT)),
                "sha256": sha256_file(RESOURCE_SNAPSHOT),
            },
            "requirements_lock": {
                "path": str(LOCK_FILE.relative_to(ROOT)),
                "sha256": sha256_file(LOCK_FILE),
            },
        },
    }
    protocol_path = OUTPUT_DIR / "hybrid_screen_protocol.json"
    protocol_hash = _write_json(protocol_path, protocol)
    return {**protocol, "protocol_sha256": protocol_hash}


def run_all() -> dict[str, Any]:
    protocol = build_protocol()
    completed = []
    for config in CONFIGS:
        completed.append(run_config(config.config_id))
    finalized = finalize()
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "completed": completed,
        "receipt": finalized,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--config", choices=tuple(CONFIG_BY_ID))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if args.protocol_only:
        payload = build_protocol()
    elif args.config:
        payload = run_config(args.config)
    elif args.finalize:
        payload = finalize()
    else:
        payload = run_all()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
