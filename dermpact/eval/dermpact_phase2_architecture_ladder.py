"""Phase 2.3A development-only perception-head architecture ladder."""
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import Tensor, nn

from . import dermpact_phase2_dataset_composition as composition
from . import dermpact_phase2_hybrid_confirmation as confirmation
from . import dermpact_phase2_hybrid_screen as screen
from .dermpact_phase2_baselines import CRITERIA, _read_jsonl, _write_jsonl, sha256_file
from .dermpact_phase2_vector_bootstrap import (
    interval,
    mean_binary_auc_samples,
    stratified_bootstrap_weights,
)


ROOT = screen.ROOT
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase2_3/architecture_ladder_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_3/architecture_ladder_v1"
PROTOCOL_PATH = OUTPUT_DIR / "architecture_ladder_protocol.json"
COMPARISON_PATH = OUTPUT_DIR / "architecture_ladder_comparisons.json"
RECEIPT_PATH = OUTPUT_DIR / "architecture_ladder_receipt.json"
RESOURCE_SNAPSHOT = (
    ROOT / "Planning discussions/phase2_audits/resource_snapshot_phase2_next.json"
)

DATA_STRATEGIES = ("derm_ham", "derm_ham_bcn")
ARCHITECTURES = ("criterion_tokens", "relation", "relation_consistency")
BASELINE_ARCHITECTURE = "independent_heads"
SEEDS = confirmation.SEEDS
N_FOLDS = screen.N_FOLDS
EPOCHS = 12
LEARNING_RATE = 3e-4
TOKEN_DIM = 128
RELATION_HEADS = 4
RELATION_FF_DIM = 256
CONSISTENCY_WEIGHT = 0.10
CONSISTENCY_TEMPERATURE = 1.0
N_BOOT = 2000
BOOTSTRAP_SEED = 20261227
TORCH_THREADS = 12
PROTECTION_MARGIN = 0.02

# Order follows ``dermpact_phase2_baselines.CRITERIA``.
CHECKLIST_WEIGHTS = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
CHECKLIST_THRESHOLD = 3.0
MELANOCYTIC_GLOBAL = {
    screen.GLOBAL_INDEX["melanoma"],
    screen.GLOBAL_INDEX["nevus"],
}

REFERENCE_RESULT_DIR = {
    "derm_ham": ROOT / "research/manifests/dermpact_phase2_2/hybrid_duration_repair_v1",
    "derm_ham_bcn": ROOT / "research/manifests/dermpact_phase2_2/dataset_composition_v1",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_path(data_strategy: str, architecture: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{data_strategy}_{architecture}_seed_{seed}_result.json"


def _reference_result_path(data_strategy: str, seed: int) -> Path:
    if data_strategy == "derm_ham":
        return REFERENCE_RESULT_DIR[data_strategy] / f"lr0p0003_equal_family_seed_{seed}_result.json"
    if data_strategy == "derm_ham_bcn":
        return REFERENCE_RESULT_DIR[data_strategy] / f"derm_ham_bcn_seed_{seed}_result.json"
    raise ValueError(data_strategy)


class CriterionTokenHead(nn.Module):
    """Shared diagnosis head plus seven criterion-conditioned visual tokens."""

    def __init__(self, *, relation: bool, feature_dim: int = 512) -> None:
        super().__init__()
        self.relation_enabled = bool(relation)
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, screen.ADAPTER_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.diagnosis = nn.Linear(screen.ADAPTER_DIM, len(screen.GLOBAL_CLASSES))
        self.melanoma = nn.Linear(screen.ADAPTER_DIM, 1)
        self.criterion_queries = nn.Parameter(torch.empty(len(CRITERIA), TOKEN_DIM))
        self.criterion_projections = nn.ModuleList(
            nn.Linear(screen.ADAPTER_DIM, TOKEN_DIM) for _ in CRITERIA
        )
        self.criterion_norm = nn.LayerNorm(TOKEN_DIM)
        if self.relation_enabled:
            layer = nn.TransformerEncoderLayer(
                d_model=TOKEN_DIM,
                nhead=RELATION_HEADS,
                dim_feedforward=RELATION_FF_DIM,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.relation = nn.TransformerEncoder(
                layer, num_layers=1, enable_nested_tensor=False
            )
        else:
            self.relation = nn.Identity()
        self.criterion_readout = nn.Linear(TOKEN_DIM, 1)
        nn.init.normal_(self.criterion_queries, mean=0.0, std=0.02)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        shared = self.adapter(features)
        tokens = torch.stack(
            [projection(shared) for projection in self.criterion_projections], dim=1
        )
        tokens = self.criterion_norm(tokens + self.criterion_queries.unsqueeze(0))
        tokens = self.relation(tokens)
        return {
            "diagnosis": self.diagnosis(shared),
            "melanoma": self.melanoma(shared).squeeze(-1),
            "criteria": self.criterion_readout(tokens).squeeze(-1),
        }


def make_model(architecture: str) -> CriterionTokenHead:
    if architecture not in ARCHITECTURES:
        raise ValueError(architecture)
    return CriterionTokenHead(relation=architecture != "criterion_tokens")


def _bernoulli_js(left: Tensor, right: Tensor) -> Tensor:
    eps = 1e-6
    left = left.clamp(eps, 1.0 - eps)
    right = right.clamp(eps, 1.0 - eps)
    middle = 0.5 * (left + right)

    def kl(value: Tensor, reference: Tensor) -> Tensor:
        return value * torch.log(value / reference) + (1.0 - value) * torch.log(
            (1.0 - value) / (1.0 - reference)
        )

    return 0.5 * (kl(left, middle) + kl(right, middle))


def truth_anchored_consistency_loss(
    outputs: Mapping[str, Tensor],
    *,
    melanoma: Tensor,
    criteria: Tensor,
    diagnosis: Tensor,
) -> tuple[Tensor, int]:
    """Agreement loss only where the labelled diagnosis and checklist agree."""
    weights = CHECKLIST_WEIGHTS.to(device=criteria.device, dtype=criteria.dtype)
    true_rule = (criteria @ weights >= CHECKLIST_THRESHOLD).to(melanoma.dtype)
    melanocytic = torch.zeros_like(melanoma, dtype=torch.bool)
    for index in MELANOCYTIC_GLOBAL:
        melanocytic |= diagnosis == index
    eligible = melanocytic & (true_rule == melanoma)
    count = int(eligible.sum().detach().cpu())
    if count == 0:
        return outputs["criteria"].sum() * 0.0, 0
    soft_score = torch.sigmoid(outputs["criteria"]) @ weights
    rule_probability = torch.sigmoid(
        (soft_score - CHECKLIST_THRESHOLD) / CONSISTENCY_TEMPERATURE
    )
    direct_probability = torch.sigmoid(outputs["melanoma"])
    return _bernoulli_js(
        direct_probability[eligible], rule_probability[eligible]
    ).mean(), count


def _batch_loss(
    model: CriterionTokenHead,
    *,
    architecture: str,
    dataset: str,
    indices: np.ndarray,
    data: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> tuple[Tensor, dict[str, Tensor]]:
    outputs = model(
        screen._tensor(
            np.asarray(data[f"{dataset}_features"][indices], dtype="float32"),
            dtype=torch.float32,
        )
    )
    loss_dataset = "derm7pt" if dataset == "derm" else "ham"
    common: dict[str, Any] = {
        "outputs": outputs,
        "dataset": loss_dataset,
        "diagnosis": screen._tensor(data[f"{dataset}_global"][indices], dtype=torch.long),
        "melanoma": screen._tensor(data[f"{dataset}_melanoma"][indices], dtype=torch.float32),
        "diagnosis_weight": screen._tensor(
            weights[f"{dataset}_diagnosis"][indices], dtype=torch.float32
        ),
        "melanoma_weight": screen._tensor(
            weights[f"{dataset}_melanoma"][indices], dtype=torch.float32
        ),
        "loss_weighting": "equal_family",
    }
    if dataset == "derm":
        common["criteria"] = screen._tensor(
            data["derm_criteria"][indices], dtype=torch.float32
        )
        common["criterion_weights"] = screen._tensor(
            weights["derm_criteria"][indices], dtype=torch.float32
        )
    total, components = screen.multitask_loss(**common)
    consistency = total * 0.0
    eligible = 0
    if architecture == "relation_consistency" and dataset == "derm":
        consistency, eligible = truth_anchored_consistency_loss(
            outputs,
            melanoma=common["melanoma"],
            criteria=common["criteria"],
            diagnosis=common["diagnosis"],
        )
        total = total + CONSISTENCY_WEIGHT * consistency
    return total, {
        **components,
        "consistency": consistency,
        "consistency_eligible": torch.tensor(float(eligible)),
    }


def _bcn_batch_loss(
    model: CriterionTokenHead,
    *,
    architecture: str,
    indices: np.ndarray,
    data: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> tuple[Tensor, dict[str, Tensor]]:
    # BCN has no criterion labels; its criterion and consistency losses remain masked.
    outputs = model(
        screen._tensor(
            np.asarray(data["bcn_features"][indices], dtype="float32"),
            dtype=torch.float32,
        )
    )
    total, components = screen.multitask_loss(
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
    zero = total * 0.0
    return total, {
        **components,
        "consistency": zero,
        "consistency_eligible": zero,
    }


def _predict_with_logits(
    model: CriterionTokenHead,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    blocks: dict[str, list[np.ndarray]] = {
        "diagnosis_logits": [],
        "melanoma_logits": [],
        "criterion_logits": [],
    }
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
            blocks["diagnosis_logits"].append(outputs["diagnosis"].cpu().numpy())
            blocks["melanoma_logits"].append(outputs["melanoma"].cpu().numpy())
            blocks["criterion_logits"].append(outputs["criteria"].cpu().numpy())
    raw = {name: np.concatenate(values, axis=0) for name, values in blocks.items()}
    return {
        **raw,
        "universal": torch.softmax(torch.from_numpy(raw["diagnosis_logits"]), dim=1).numpy(),
        "melanoma": torch.sigmoid(torch.from_numpy(raw["melanoma_logits"])).numpy(),
        "criteria": torch.sigmoid(torch.from_numpy(raw["criterion_logits"])).numpy(),
    }


def _reload_difference(
    model: CriterionTokenHead,
    *,
    architecture: str,
    checkpoint: Path,
    features: np.ndarray,
) -> float:
    restored = make_model(architecture)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    batch = screen._tensor(
        np.asarray(features[: min(16, len(features))], dtype="float32"),
        dtype=torch.float32,
    )
    model.eval()
    restored.eval()
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
    data_strategy: str,
    architecture: str,
    *,
    fold: int,
    seed: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    if data_strategy not in DATA_STRATEGIES or architecture not in ARCHITECTURES:
        raise ValueError((data_strategy, architecture))
    initialization_seed = seed + fold
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    ham_train = np.flatnonzero(data["ham_folds"] != fold)
    ham_valid = np.flatnonzero(data["ham_folds"] == fold)
    derm_train = np.flatnonzero(data["derm_folds"] != fold)
    derm_valid = np.flatnonzero(data["derm_folds"] == fold)
    combined = data_strategy == "derm_ham_bcn"
    if combined:
        bcn_train = np.flatnonzero(data["bcn_folds"] != fold)
        bcn_valid = np.flatnonzero(data["bcn_folds"] == fold)
    weights = composition._training_weights(data, fold)
    model = make_model(architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=screen.WEIGHT_DECAY
    )
    steps = composition._steps_per_epoch(data, fold)
    scheduler = screen._scheduler(optimizer, EPOCHS * steps)
    history = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        ham_batches = screen._cyclic_batches(
            ham_train,
            steps=steps,
            batch_size=screen.BATCH_SIZE,
            rng=np.random.default_rng(seed + fold * 1000 + epoch * 3 + 1),
        )
        derm_batches = screen._cyclic_batches(
            derm_train,
            steps=steps,
            batch_size=screen.BATCH_SIZE,
            rng=np.random.default_rng(seed + fold * 1000 + epoch * 3 + 2),
        )
        if combined:
            bcn_batches = screen._cyclic_batches(
                bcn_train,
                steps=steps,
                batch_size=screen.BATCH_SIZE,
                rng=np.random.default_rng(seed + fold * 1000 + epoch * 3 + 3),
            )
        epoch_losses = []
        component_sum: dict[str, float] = defaultdict(float)
        model.train()
        epoch_started = time.perf_counter()
        for ham_indices, derm_indices in zip(ham_batches, derm_batches):
            optimizer.zero_grad(set_to_none=True)
            ham_loss, ham_components = _batch_loss(
                model,
                architecture=architecture,
                dataset="ham",
                indices=ham_indices,
                data=data,
                weights=weights,
            )
            derm_loss, derm_components = _batch_loss(
                model,
                architecture=architecture,
                dataset="derm",
                indices=derm_indices,
                data=data,
                weights=weights,
            )
            if combined:
                bcn_indices = next(bcn_batches)
                bcn_loss, bcn_components = _bcn_batch_loss(
                    model,
                    architecture=architecture,
                    indices=bcn_indices,
                    data=data,
                    weights=weights,
                )
                loss = 0.5 * derm_loss + 0.25 * ham_loss + 0.25 * bcn_loss
                components = {
                    **{f"derm_{key}": value for key, value in derm_components.items()},
                    **{f"ham_{key}": value for key, value in ham_components.items()},
                    **{f"bcn_{key}": value for key, value in bcn_components.items()},
                }
            else:
                loss = 0.5 * derm_loss + 0.5 * ham_loss
                components = {
                    **{f"derm_{key}": value for key, value in derm_components.items()},
                    **{f"ham_{key}": value for key, value in ham_components.items()},
                }
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss: {data_strategy}/{architecture}/seed{seed}/fold{fold}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), screen.GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach()))
            for name, value in components.items():
                component_sum[name] += float(value.detach())
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(epoch_losses)),
                "last_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "component_means": {
                    name: value / steps for name, value in component_sum.items()
                },
            }
        )

    prediction_sets = {
        "ham": (data["ham_features"], ham_valid),
        "derm7pt": (data["derm_features"], derm_valid),
    }
    if combined:
        prediction_sets["bcn_train_oof"] = (data["bcn_features"], bcn_valid)
    predictions = {
        name: _predict_with_logits(model, features, indices)
        for name, (features, indices) in prediction_sets.items()
    }
    fold_dir = EXPERIMENT_DIR / data_strategy / architecture / f"seed_{seed}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / f"epoch_{EPOCHS}_state.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)
    reload_difference = _reload_difference(
        model,
        architecture=architecture,
        checkpoint=checkpoint,
        features=np.asarray(data["derm_features"][derm_valid], dtype="float32"),
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
    result: dict[str, Any] = {
        "fold": fold,
        "seed": seed,
        "initialization_seed": initialization_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "steps_per_epoch": steps,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
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
    if combined:
        result["bcn_rows"] = composition._bcn_prediction_rows(
            data, bcn_valid, predictions["bcn_train_oof"]
        )
    return result


def _derm_rule_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    weights = np.asarray(CHECKLIST_WEIGHTS, dtype=float)
    eligible = []
    joint_correct = []
    disagreements = []
    for row in rows:
        if str(row["label"]) not in {"MEL", "NEV"}:
            continue
        truth = int(str(row["label"]) == "MEL")
        true_criteria = np.asarray(
            [int(bool(row["criterion_true"][name])) for name in CRITERIA], dtype=float
        )
        true_rule = int(float(true_criteria @ weights) >= CHECKLIST_THRESHOLD)
        if true_rule != truth:
            continue
        predicted_criteria = np.asarray(
            [float(row["criterion_probability"][name]) >= 0.5 for name in CRITERIA],
            dtype=float,
        )
        rule_prediction = int(float(predicted_criteria @ weights) >= CHECKLIST_THRESHOLD)
        direct_prediction = int(float(row["melanoma_probability"]) >= 0.5)
        eligible.append(str(row["id"]))
        joint_correct.append(int(rule_prediction == truth and direct_prediction == truth))
        disagreements.append(int(rule_prediction != direct_prediction))
    if not eligible:
        raise ValueError("no rule-consistent melanocytic cases")
    return {
        "eligible_cases": len(eligible),
        "joint_correct_agreement": float(np.mean(joint_correct)),
        "direct_rule_disagreement": float(np.mean(disagreements)),
    }


def run_seed(
    data_strategy: str,
    architecture: str,
    seed: int,
    *,
    protocol_sha256: str,
) -> dict[str, Any]:
    result_path = _result_path(data_strategy, architecture, seed)
    sidecar = result_path.with_suffix(result_path.suffix + ".sha256")
    if result_path.exists() and sidecar.exists():
        existing = _load_json(result_path)
        expected = sidecar.read_text(encoding="ascii").strip()
        if sha256_file(result_path) == expected and existing.get("protocol_sha256") == protocol_sha256:
            return {
                "data_strategy": data_strategy,
                "architecture": architecture,
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
    data = composition._data()
    frozen_ham = _read_jsonl(screen.FROZEN_HAM_PREDICTIONS)
    frozen_derm = _read_jsonl(screen.FROZEN_DERM_PREDICTIONS)
    folds = []
    ham_rows: list[dict[str, Any]] = []
    derm_rows: list[dict[str, Any]] = []
    bcn_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(N_FOLDS):
        fold_result = train_fold(
            data_strategy,
            architecture,
            fold=fold,
            seed=seed,
            data=data,
        )
        ham_rows.extend(fold_result.pop("ham_rows"))
        derm_rows.extend(fold_result.pop("derm_rows"))
        bcn_rows.extend(fold_result.pop("bcn_rows", []))
        folds.append(fold_result)
        print(
            f"[{data_strategy}/{architecture}/seed{seed}] fold {fold + 1}/{N_FOLDS} "
            f"complete in {fold_result['elapsed_seconds']:.2f}s",
            flush=True,
        )
    ham_rows = screen._order_like_reference(ham_rows, frozen_ham)
    derm_rows = screen._order_like_reference(derm_rows, frozen_derm)
    seed_dir = EXPERIMENT_DIR / data_strategy / architecture / f"seed_{seed}"
    predictions = {}
    for name, rows in (("ham", ham_rows), ("derm7pt", derm_rows)):
        path = seed_dir / f"{name}_oof_predictions.jsonl"
        predictions[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _write_jsonl(path, rows),
        }
    if data_strategy == "derm_ham_bcn":
        bcn_rows = sorted(bcn_rows, key=lambda row: str(row["id"]))
        path = seed_dir / "bcn_training_oof_predictions.jsonl"
        predictions["bcn_train_oof"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _write_jsonl(path, bcn_rows),
        }
    metrics = confirmation._seed_metrics(ham_rows, derm_rows, seed=seed)
    metrics["derm7pt"]["criterion_mean_auc"] = float(
        np.mean(
            [
                metrics["derm7pt"]["output_auc"][screen.CRITERION_SHORT[name]]
                for name in CRITERIA
            ]
        )
    )
    metrics["derm7pt"]["rule_consistency"] = _derm_rule_summary(derm_rows)
    if data_strategy == "derm_ham_bcn":
        metrics["bcn_training_oof"] = composition._bcn_metrics(bcn_rows, seed=seed)
    result = {
        "protocol": "dermpact-phase2-3a-architecture-seed-result-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "evidence_level": "development OOF frozen-feature architecture screen",
        "data_strategy": data_strategy,
        "architecture": architecture,
        "seed": seed,
        "fixed_training": {
            "epochs": EPOCHS,
            "folds": N_FOLDS,
            "learning_rate": LEARNING_RATE,
            "loss_weighting": "equal_family",
            "early_stopping": False,
        },
        **metrics,
        "fold_training": folds,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": predictions,
        "hard_scope_checks": {
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "bcn_validation_or_final_feature_rows_loaded": 0,
            "ph2_rows_loaded": 0,
            "clinical_photo_features_loaded": 0,
            "structured_facts_loaded_into_model": 0,
        },
    }
    result_hash = screen._write_json(result_path, result)
    return {
        "data_strategy": data_strategy,
        "architecture": architecture,
        "seed": seed,
        "status": "completed",
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": result_hash,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _source_result(data_strategy: str, architecture: str, seed: int) -> dict[str, Any]:
    path = (
        _reference_result_path(data_strategy, seed)
        if architecture == BASELINE_ARCHITECTURE
        else _result_path(data_strategy, architecture, seed)
    )
    return _load_json(path)


def _source_rows(
    data_strategy: str, architecture: str, seed: int, dataset: str
) -> list[dict[str, Any]]:
    result = _source_result(data_strategy, architecture, seed)
    key = "bcn_train_oof" if dataset == "bcn_train_oof" else dataset
    return _read_jsonl(ROOT / result["predictions"][key]["path"])


def _rule_vectors(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(CHECKLIST_WEIGHTS, dtype=float)
    eligible = np.zeros(len(rows), dtype=bool)
    joint = np.zeros(len(rows), dtype=float)
    disagreement = np.zeros(len(rows), dtype=float)
    for index, row in enumerate(rows):
        if str(row["label"]) not in {"MEL", "NEV"}:
            continue
        truth = int(str(row["label"]) == "MEL")
        true_values = np.asarray(
            [int(bool(row["criterion_true"][name])) for name in CRITERIA], dtype=float
        )
        if int(float(true_values @ weights) >= CHECKLIST_THRESHOLD) != truth:
            continue
        predicted_values = np.asarray(
            [float(row["criterion_probability"][name]) >= 0.5 for name in CRITERIA],
            dtype=float,
        )
        direct = int(float(row["melanoma_probability"]) >= 0.5)
        rule = int(float(predicted_values @ weights) >= CHECKLIST_THRESHOLD)
        eligible[index] = True
        joint[index] = float(direct == truth and rule == truth)
        disagreement[index] = float(direct != rule)
    if not eligible.any():
        raise ValueError("no eligible rule-consistent cases")
    return eligible, np.stack([joint, disagreement], axis=0)


def _metric_samples(
    rows: Sequence[Mapping[str, Any]], *, dataset: str, weights: np.ndarray
) -> dict[str, np.ndarray]:
    result = confirmation._metric_samples(rows, dataset=dataset, weights=weights)
    if dataset != "derm7pt":
        return result
    targets, scores = confirmation._derm_targets_scores(rows)
    criterion_names = [screen.CRITERION_SHORT[name] for name in CRITERIA]
    result["criterion_mean_auc"] = mean_binary_auc_samples(
        {name: targets[name] for name in criterion_names},
        {name: scores[name] for name in criterion_names},
        weights,
    )
    eligible, values = _rule_vectors(rows)
    eligible_weights = weights[:, eligible]
    denominator = eligible_weights.sum(axis=1)
    if np.any(denominator == 0):
        raise ValueError("a bootstrap replicate lost all rule-consistent cases")
    result["joint_correct_agreement"] = (
        eligible_weights @ values[0, eligible]
    ) / denominator
    result["direct_rule_disagreement"] = (
        eligible_weights @ values[1, eligible]
    ) / denominator
    return result


def _point_metric(
    rows: Sequence[Mapping[str, Any]], *, dataset: str, metric: str
) -> float:
    if dataset == "ham":
        truth = [str(row["label"]) for row in rows]
        if metric == "balanced_accuracy":
            return float(
                balanced_accuracy_score(truth, confirmation._diagnosis_predictions(rows))
            )
        if metric == "melanoma_auc":
            return float(
                roc_auc_score(
                    [int(label == "melanoma") for label in truth],
                    [float(row["melanoma_probability"]) for row in rows],
                )
            )
        raise ValueError((dataset, metric))
    targets, scores = confirmation._derm_targets_scores(rows)
    if metric == "balanced_accuracy":
        return float(
            balanced_accuracy_score(
                [str(row["label"]) for row in rows],
                confirmation._diagnosis_predictions(rows),
            )
        )
    if metric == "melanoma_auc":
        return float(roc_auc_score(targets["MEL"], scores["MEL"]))
    if metric in {"primary", "criterion_mean_auc"}:
        names = list(targets) if metric == "primary" else [
            screen.CRITERION_SHORT[name] for name in CRITERIA
        ]
        return float(np.mean([roc_auc_score(targets[name], scores[name]) for name in names]))
    summary = _derm_rule_summary(rows)
    if metric == "joint_correct_agreement":
        return float(summary["joint_correct_agreement"])
    if metric == "direct_rule_disagreement":
        return float(summary["direct_rule_disagreement"])
    raise ValueError((dataset, metric))


def _comparison(
    data_strategy: str,
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
    per_seed_delta = {}
    for seed in SEEDS:
        candidate_rows = _source_rows(data_strategy, candidate, seed, dataset)
        baseline_rows = _source_rows(data_strategy, baseline, seed, dataset)
        if [row["id"] for row in candidate_rows] != [row["id"] for row in baseline_rows]:
            raise AssertionError("paired comparison row order changed")
        candidate_samples.append(
            _metric_samples(candidate_rows, dataset=dataset, weights=weights)[metric]
        )
        baseline_samples.append(
            _metric_samples(baseline_rows, dataset=dataset, weights=weights)[metric]
        )
        candidate_point = _point_metric(candidate_rows, dataset=dataset, metric=metric)
        baseline_point = _point_metric(baseline_rows, dataset=dataset, metric=metric)
        candidate_points.append(candidate_point)
        baseline_points.append(baseline_point)
        per_seed_delta[str(seed)] = candidate_point - baseline_point
    candidate_mean = confirmation._seed_resampled_mean(candidate_samples, seed_draws)
    baseline_mean = confirmation._seed_resampled_mean(baseline_samples, seed_draws)
    point = float(np.mean(candidate_points) - np.mean(baseline_points))
    return {
        **interval(candidate_mean - baseline_mean, point, n_boot=N_BOOT),
        "per_seed_delta": per_seed_delta,
        "nonnegative_seed_count": sum(value >= 0.0 for value in per_seed_delta.values()),
    }


def _qualifies(
    comparison: Mapping[str, Any],
    overall: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    lower_is_better: bool = False,
) -> tuple[bool, list[dict[str, Any]]]:
    estimate = float(comparison["estimate"])
    directional = estimate < 0.0 if lower_is_better else estimate > 0.0
    stable = int(comparison["nonnegative_seed_count"]) >= 2
    if lower_is_better:
        stable = int(comparison["nonnegative_seed_count"]) <= 1
    violations = []
    for dataset, metrics in overall.items():
        for metric, value in metrics.items():
            if float(value["estimate"]) < -PROTECTION_MARGIN and float(value["ci_high"]) < 0.0:
                violations.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "delta": value["estimate"],
                        "ci_high": value["ci_high"],
                    }
                )
    return bool(directional and stable and not violations), violations


def finalize(*, protocol_sha256: str) -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != protocol_sha256:
        raise AssertionError("architecture protocol hash mismatch")
    reference_derm = _source_rows("derm_ham", BASELINE_ARCHITECTURE, SEEDS[0], "derm7pt")
    reference_ham = _source_rows("derm_ham", BASELINE_ARCHITECTURE, SEEDS[0], "ham")
    derm_weights = stratified_bootstrap_weights(
        confirmation._derm_bootstrap_strata(reference_derm),
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 1,
    )
    ham_weights = stratified_bootstrap_weights(
        [str(row["label"]) for row in reference_ham],
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED + 2,
    )
    seed_draws = np.random.default_rng(BOOTSTRAP_SEED + 3).integers(
        0, len(SEEDS), size=(N_BOOT, len(SEEDS))
    )
    comparisons: dict[str, Any] = {
        "protocol": "dermpact-phase2-3a-architecture-comparisons-v1",
        "protocol_sha256": protocol_sha256,
        "direction": "candidate minus baseline",
        "protection_margin": PROTECTION_MARGIN,
        "data_strategies": {},
    }
    declared = {
        "criterion_tokens": (BASELINE_ARCHITECTURE, "primary"),
        "relation": ("criterion_tokens", "criterion_mean_auc"),
        "relation_consistency": ("relation", "joint_correct_agreement"),
    }
    for data_strategy in DATA_STRATEGIES:
        strategy_result: dict[str, Any] = {"steps": {}, "selected_architecture": BASELINE_ARCHITECTURE}
        for candidate in ARCHITECTURES:
            predecessor, declared_metric = declared[candidate]
            incremental: dict[str, Any] = {}
            for dataset, metrics, weights in (
                (
                    "derm7pt",
                    (
                        "primary",
                        "criterion_mean_auc",
                        "balanced_accuracy",
                        "melanoma_auc",
                        "joint_correct_agreement",
                        "direct_rule_disagreement",
                    ),
                    derm_weights,
                ),
                ("ham", ("balanced_accuracy", "melanoma_auc"), ham_weights),
            ):
                incremental[dataset] = {
                    metric: _comparison(
                        data_strategy,
                        candidate,
                        predecessor,
                        dataset=dataset,
                        metric=metric,
                        weights=weights,
                        seed_draws=seed_draws,
                    )
                    for metric in metrics
                }
            overall = {
                dataset: {
                    metric: _comparison(
                        data_strategy,
                        candidate,
                        BASELINE_ARCHITECTURE,
                        dataset=dataset,
                        metric=metric,
                        weights=weights,
                        seed_draws=seed_draws,
                    )
                    for metric in metrics
                }
                for dataset, metrics, weights in (
                    ("derm7pt", ("primary", "balanced_accuracy", "melanoma_auc"), derm_weights),
                    ("ham", ("balanced_accuracy", "melanoma_auc"), ham_weights),
                )
            }
            qualifies, violations = _qualifies(
                incremental["derm7pt"][declared_metric], overall
            )
            strategy_result["steps"][candidate] = {
                "predecessor": predecessor,
                "declared_endpoint": declared_metric,
                "incremental": incremental,
                "versus_independent_heads": overall,
                "material_regressions": violations,
                "qualifies": qualifies,
            }
            if qualifies:
                strategy_result["selected_architecture"] = candidate
        comparisons["data_strategies"][data_strategy] = strategy_result
    comparisons["decision"] = {
        strategy: value["selected_architecture"]
        for strategy, value in comparisons["data_strategies"].items()
    }
    comparison_hash = screen._write_json(COMPARISON_PATH, comparisons)

    artifacts = {}
    checkpoint_count = raw_count = prediction_count = 0
    for data_strategy in DATA_STRATEGIES:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                path = _result_path(data_strategy, architecture, seed)
                result = _load_json(path)
                if result.get("protocol_sha256") != protocol_sha256:
                    raise AssertionError("seed result protocol hash mismatch")
                artifacts[f"{data_strategy}_{architecture}_seed_{seed}"] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for fold in result["fold_training"]:
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
        "protocol": "dermpact-phase2-3a-architecture-receipt-v1",
        "protocol_sha256": protocol_sha256,
        "status": "PASS",
        "decision": comparisons["decision"],
        "comparison_artifact": {
            "path": str(COMPARISON_PATH.relative_to(ROOT)),
            "sha256": comparison_hash,
        },
        "trained_architectures": list(ARCHITECTURES),
        "data_strategies": list(DATA_STRATEGIES),
        "seeds": list(SEEDS),
        "folds_per_seed": N_FOLDS,
        "epochs": EPOCHS,
        "checkpoints_verified": checkpoint_count,
        "raw_logit_artifacts_verified": raw_count,
        "prediction_artifacts_verified": prediction_count,
        "protected_or_external_rows_loaded": 0,
        "clinical_photo_features_loaded": 0,
        "structured_facts_loaded_into_model": 0,
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
    data = composition._data()
    references = {}
    for strategy in DATA_STRATEGIES:
        for seed in SEEDS:
            path = _reference_result_path(strategy, seed)
            references[f"{strategy}_seed_{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
    parameter_counts = {
        architecture: sum(parameter.numel() for parameter in make_model(architecture).parameters())
        for architecture in ARCHITECTURES
    }
    protocol = {
        "protocol": "dermpact-phase2-3a-architecture-ladder-v1",
        "status": "PRESPECIFIED_BEFORE_ARCHITECTURE_TRAINING",
        "question": (
            "Do criterion-conditioned tokens, criterion-relation learning, and a truth-anchored "
            "diagnosis/checklist consistency term improve development performance without "
            "materially damaging a protected task?"
        ),
        "fixed_backbone": {
            "candidate": screen.CANDIDATE,
            "model_id": screen.MODEL_ID,
            "state": "frozen cached global features",
        },
        "baseline": "selected independent seven-head architecture from Phase 2.2D/2.2G",
        "data_strategies": {
            "derm_ham": {
                "datasets": ["Derm7pt development", "HAM training"],
                "loss_weights": {"derm7pt": 0.5, "ham": 0.5},
            },
            "derm_ham_bcn": {
                "datasets": ["Derm7pt development", "HAM training", "BCN training"],
                "loss_weights": {"derm7pt": 0.5, "ham": 0.25, "bcn": 0.25},
            },
        },
        "architecture_steps": {
            "criterion_tokens": {
                "changed_factor": "replace seven scalar heads with seven criterion-conditioned tokens",
                "relation_layer": False,
                "consistency_loss": False,
                "declared_endpoint": "Derm7pt eight-output mean AUROC",
            },
            "relation": {
                "changed_factor": "add one transformer relation layer between criterion tokens",
                "relation_layer": True,
                "consistency_loss": False,
                "declared_endpoint": "Derm7pt seven-criterion mean AUROC",
            },
            "relation_consistency": {
                "changed_factor": "add a weight-0.10 Jensen-Shannon consistency term",
                "relation_layer": True,
                "consistency_loss": {
                    "weight": CONSISTENCY_WEIGHT,
                    "temperature": CONSISTENCY_TEMPERATURE,
                    "eligibility": (
                        "Derm7pt MEL/NEV training cases where the labelled fixed checklist "
                        "decision agrees with the labelled melanoma target"
                    ),
                    "contradictory_source_labels": "masked from this auxiliary term",
                },
                "declared_endpoint": "jointly-correct direct/checklist agreement",
            },
        },
        "parameter_counts": parameter_counts,
        "controlled_training": {
            "learning_rate": LEARNING_RATE,
            "loss_weighting": "equal_family",
            "epochs": EPOCHS,
            "folds": N_FOLDS,
            "seeds": list(SEEDS),
            "batch_size_per_dataset": screen.BATCH_SIZE,
            "optimizer": "AdamW",
            "weight_decay": screen.WEIGHT_DECAY,
            "early_stopping": False,
        },
        "promotion_rule": {
            "declared_endpoint": "positive candidate-minus-predecessor mean delta",
            "seed_stability": "non-negative in at least two of three seeds",
            "material_regression": (
                "candidate-minus-independent-heads < -0.02 and paired hierarchical 95% CI excludes zero"
            ),
            "protected_endpoints": [
                "Derm7pt eight-output mean AUROC",
                "Derm7pt diagnosis balanced accuracy",
                "Derm7pt melanoma AUROC",
                "HAM diagnosis balanced accuracy",
                "HAM melanoma AUROC",
            ],
            "selection": "deepest qualifying nested architecture; otherwise independent heads",
        },
        "later_prespecified_factor_order": [
            {
                "phase": "2.3B",
                "factor": "paired clinical-photo gated residual",
                "requirements": (
                    "Derm7pt development only for paired-photo learning; 50% modality dropout; "
                    "clinical contribution exactly zero when missing; present-photo and forced-missing modes reported"
                ),
            },
            {
                "phase": "2.3C",
                "factor": "bounded observed-fact diagnosis-logit updater",
                "requirements": (
                    "diagnosis branch only; sex and harmonized anatomical site first; missing values exactly zero; "
                    "no dataset identifier; counterfactual swaps, subgroup checks, and calibration required"
                ),
            },
            {
                "phase": "2.3D",
                "factor": "shared model versus separate diagnosis/checklist models",
                "requirements": "same output contract and matched development predictions",
            },
            {
                "phase": "2.4",
                "factor": "real-image backbone adaptation on NVIDIA/Colab",
                "requirements": (
                    "selected head first; LoRA versus last-block adaptation as separate arms; "
                    "WiSE-FT only if adaptation improves internal results but weakens shift performance"
                ),
            },
        ],
        "hard_scope": {
            "allowed": ["Derm7pt 616-case development", "HAM training", "BCN training"],
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_rows_loaded": 0,
            "bcn_validation_or_final_feature_rows_loaded": 0,
            "ph2_rows_loaded": 0,
            "clinical_photo_features_loaded": 0,
            "structured_facts_loaded_into_model": 0,
            "bcn_training_feature_rows": int(data["bcn_features"].shape[0]),
        },
        "artifacts": {
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "dataset_composition_protocol": {
                "path": str(composition.PROTOCOL_PATH.relative_to(ROOT)),
                "sha256": sha256_file(composition.PROTOCOL_PATH),
            },
            "dataset_composition_receipt": {
                "path": str(composition.RECEIPT_PATH.relative_to(ROOT)),
                "sha256": sha256_file(composition.RECEIPT_PATH),
            },
            "requirements_lock": {
                "path": str(screen.LOCK_FILE.relative_to(ROOT)),
                "sha256": sha256_file(screen.LOCK_FILE),
            },
            "resource_snapshot": {
                "path": str(RESOURCE_SNAPSHOT.relative_to(ROOT)),
                "sha256": sha256_file(RESOURCE_SNAPSHOT),
            },
            **references,
        },
    }
    protocol_sha256 = screen._write_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": protocol_sha256}


def run_all() -> dict[str, Any]:
    protocol = build_protocol()
    completed = [
        run_seed(
            data_strategy,
            architecture,
            seed,
            protocol_sha256=protocol["protocol_sha256"],
        )
        for data_strategy in DATA_STRATEGIES
        for architecture in ARCHITECTURES
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
    parser.add_argument("--build-protocol", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--data-strategy", choices=DATA_STRATEGIES)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args(argv)
    if args.build_protocol:
        print(json.dumps(build_protocol(), indent=2, sort_keys=True))
        return 0
    if args.run_all:
        print(json.dumps(run_all(), indent=2, sort_keys=True))
        return 0
    if args.finalize:
        if not args.protocol_sha256:
            raise SystemExit("--protocol-sha256 is required")
        print(json.dumps(finalize(protocol_sha256=args.protocol_sha256), indent=2, sort_keys=True))
        return 0
    if args.data_strategy and args.architecture and args.seed and args.protocol_sha256:
        print(
            json.dumps(
                run_seed(
                    args.data_strategy,
                    args.architecture,
                    args.seed,
                    protocol_sha256=args.protocol_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    parser.error("choose --build-protocol, --run-all, --finalize, or one complete seed run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
