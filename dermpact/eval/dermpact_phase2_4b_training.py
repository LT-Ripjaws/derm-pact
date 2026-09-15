"""Fold-local Phase 2.4B paired-photo training core."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import dermpact_phase2_4a_attention_factor as sharp_factor
from . import dermpact_phase2_4b_block10_cache as cache_core
from . import dermpact_phase2_4b_inputs as phase24b_inputs
from .dermpact_phase2_4b_paired_model import Phase24BPairedCriterionModel


ROOT = phase24b_inputs.ROOT
SEED = 20260827
FOLDS = tuple(range(5))
EPOCHS = 20
BATCH_SIZE = 64
MODALITY_DROPOUT = 0.5
GRADIENT_CLIP_NORM = 1.0
PROTOCOL_SCHEMA = "dermpact-phase2-4b-paired-training-protocol-v1"
RESUME_SCHEMA = "dermpact-phase2-4b-paired-training-resume-v1"
RESULT_SCHEMA = "dermpact-phase2-4b-paired-fold-training-result-v1"

SHARP_PROTOCOL_PATH = (
    sharp_factor.MANIFEST_DIR / "attention_factor_protocol.json"
)
CLINICAL_CACHE_DIR = cache_core.DEFAULT_OUTPUT_DIR

OPTIMIZER_GROUPS: Mapping[str, Mapping[str, float]] = {
    "clinical_adapters": {"learning_rate": 1e-4, "weight_decay": 1e-4},
    "fusion_projections": {"learning_rate": 3e-4, "weight_decay": 1e-4},
    "fusion_controls": {"learning_rate": 1e-3, "weight_decay": 0.0},
}
SCHEDULER = {
    "name": "warmup_cosine",
    "warmup_ratio": 0.1,
    "minimum_lr_ratio": 0.01,
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    return phase24b_inputs.sha256_file(Path(path))


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def _verify_artifact(path: Path, expected: str | None = None) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _sha256(path)
    sidecar = _sidecar(path)
    if sidecar.is_file():
        recorded = sidecar.read_text(encoding="ascii").strip().split()[0]
        if recorded != observed:
            raise AssertionError(f"SHA-256 sidecar mismatch: {path}")
    elif expected is None:
        raise FileNotFoundError(f"missing SHA-256 sidecar: {sidecar}")
    if expected is not None and observed != expected:
        raise AssertionError(f"artifact SHA-256 mismatch: {path}")
    return observed


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _path_from_binding(raw: object, *, base: Path = ROOT) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact binding lacks a path")
    path = Path(raw)
    return path if path.is_absolute() else base / path


@dataclass(frozen=True)
class SharpFoldBinding:
    fold: int
    protocol: Mapping[str, Any]
    protocol_path: Path
    protocol_sha256: str
    training_receipt: Mapping[str, Any]
    training_receipt_path: Path
    training_receipt_sha256: str
    residual_checkpoint_path: Path
    residual_checkpoint_sha256: str

    def identity(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "protocol": {
                "path": str(self.protocol_path),
                "sha256": self.protocol_sha256,
            },
            "training_receipt": {
                "path": str(self.training_receipt_path),
                "sha256": self.training_receipt_sha256,
            },
            "sharp_residual_checkpoint": {
                "path": str(self.residual_checkpoint_path),
                "sha256": self.residual_checkpoint_sha256,
            },
        }


@dataclass(frozen=True)
class ClinicalCache:
    cache_dir: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    derm_tokens: np.ndarray
    clinical_tokens: np.ndarray
    clinical_available: np.ndarray
    bindings: Mapping[str, Mapping[str, Any]]

    def identity(self) -> dict[str, Any]:
        return {
            "manifest": {
                "path": str(self.cache_dir / "cache_manifest.json"),
                "sha256": self.manifest_sha256,
            },
            "artifacts": {key: dict(value) for key, value in self.bindings.items()},
        }


@dataclass(frozen=True)
class FoldTrainingData:
    derm_tokens: np.ndarray
    clinical_tokens: np.ndarray
    clinical_available: np.ndarray
    criterion_targets: np.ndarray
    folds: np.ndarray

    def validate(self) -> None:
        rows = len(self.folds)
        if self.derm_tokens.shape[0] != rows or self.clinical_tokens.shape != self.derm_tokens.shape:
            raise ValueError("paired token arrays must be row-aligned")
        if self.clinical_available.shape != (rows,) or self.clinical_available.dtype != np.bool_:
            raise ValueError("clinical_available must be a Boolean row mask")
        if self.criterion_targets.shape != (rows, 7):
            raise ValueError("criterion_targets must have shape [rows, 7]")
        if not np.isfinite(self.criterion_targets).all():
            raise ValueError("criterion targets must be finite")
        if not np.isin(self.criterion_targets, (0.0, 1.0)).all():
            raise ValueError("criterion targets must be binary")


def resolve_sharp_fold(
    fold: int, *, protocol_path: Path = SHARP_PROTOCOL_PATH
) -> SharpFoldBinding:
    """Resolve one matching trained sharp fold and fail closed on drift."""
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    protocol_path = Path(protocol_path)
    protocol_sha = _verify_artifact(protocol_path)
    protocol = _read_object(protocol_path)
    if protocol.get("schema_version") != sharp_factor.PROTOCOL_SCHEMA:
        raise AssertionError("sharp protocol schema changed")
    identity = protocol.get("identity", {})
    if identity.get("seed") != SEED or identity.get("folds") != list(FOLDS):
        raise AssertionError("sharp protocol seed/folds changed")
    if "sharp" not in identity.get("arms", []):
        raise AssertionError("sharp arm is absent from the frozen protocol")

    receipt_path = protocol_path.parent / "sharp" / f"seed_{SEED}_fold_{fold}_training.json"
    receipt_sha = _verify_artifact(receipt_path)
    receipt = _read_object(receipt_path)
    expected = (
        sharp_factor.TRAINING_SCHEMA,
        "PASS",
        protocol_sha,
        "sharp",
        SEED,
        fold,
        EPOCHS,
    )
    observed = tuple(
        receipt.get(key)
        for key in (
            "schema_version", "status", "protocol_sha256", "arm", "seed",
            "fold", "completed_epochs",
        )
    )
    if observed != expected:
        raise AssertionError(f"sharp fold receipt identity mismatch: {observed!r}")
    artifact = receipt.get("new_residual_checkpoint")
    if not isinstance(artifact, Mapping):
        raise ValueError("sharp fold receipt lacks its residual checkpoint")
    checkpoint = _path_from_binding(artifact.get("path"))
    checkpoint_sha = _verify_artifact(checkpoint, str(artifact.get("sha256")))
    return SharpFoldBinding(
        fold=fold,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha,
        training_receipt=receipt,
        training_receipt_path=receipt_path,
        training_receipt_sha256=receipt_sha,
        residual_checkpoint_path=checkpoint,
        residual_checkpoint_sha256=checkpoint_sha,
    )


def load_clinical_cache(*, cache_dir: Path = CLINICAL_CACHE_DIR) -> ClinicalCache:
    """Load the completed aligned cache read-only after hash verification."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "cache_manifest.json"
    manifest_sha = _verify_artifact(manifest_path)
    manifest = _read_object(manifest_path)
    if manifest.get("format") != cache_core.FORMAT_VERSION or manifest.get("status") != "PASS":
        raise AssertionError("clinical cache is not a completed Phase 2.4B cache")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("clinical cache manifest lacks artifact bindings")

    def local(name: str, expected_shape: Sequence[int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        binding = artifacts.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"clinical cache lacks {name}")
        path = cache_dir / str(binding.get("file", ""))
        _verify_artifact(path, str(binding.get("sha256")))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if expected_shape is not None and values.shape != tuple(expected_shape):
            raise AssertionError(f"{name} shape changed: {values.shape}")
        return values, dict(binding)

    shape = tuple(manifest.get("contract", {}).get("shape", ()))
    if shape != (phase24b_inputs.N_CASES, 197, 768):
        raise AssertionError("clinical cache production shape changed")
    clinical, clinical_binding = local("clinical_tokens", shape)
    available, available_binding = local("clinical_available", (shape[0],))
    if clinical.dtype != np.float32 or available.dtype != np.bool_:
        raise AssertionError("clinical cache dtype changed")
    if np.any(np.asarray(clinical[~available]) != 0.0):
        raise AssertionError("unavailable clinical cache rows are not exact zero")

    derm_binding_raw = artifacts.get("dermoscopic_tokens")
    if not isinstance(derm_binding_raw, Mapping):
        raise ValueError("clinical cache lacks dermoscopic binding")
    derm_binding = dict(derm_binding_raw)
    derm_path = _path_from_binding(derm_binding.get("token_path"))
    _verify_artifact(derm_path, str(derm_binding.get("token_sha256")))
    derm = np.load(derm_path, mmap_mode="r", allow_pickle=False)
    if derm.shape != shape or derm.dtype != np.float32:
        raise AssertionError("bound dermoscopic cache contract changed")
    return ClinicalCache(
        cache_dir=cache_dir,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        derm_tokens=derm,
        clinical_tokens=clinical,
        clinical_available=available,
        bindings={
            "dermoscopic_tokens": derm_binding,
            "clinical_tokens": clinical_binding,
            "clinical_available": available_binding,
        },
    )


def construct_paired_model(
    binding: SharpFoldBinding,
    *,
    attention_dim: int = 64,
    bottleneck_dim: int = 32,
    maximum_logit_shift: float = 1.0,
) -> Phase24BPairedCriterionModel:
    """Strictly reload the matching sharp comparator and add clinical modules."""
    baseline = sharp_factor._new_model(binding.protocol, "sharp", binding.fold)
    state = torch.load(
        binding.residual_checkpoint_path, map_location="cpu", weights_only=True
    )
    baseline.new_residual.load_state_dict(state, strict=True)
    baseline.eval()
    prototypes = (
        baseline.confirmed_model.criterion_residual.prototype_embeddings
        .detach().cpu().clone()
    )
    model = Phase24BPairedCriterionModel(
        baseline,
        baseline.base_model.visual_tail,
        prototype_embeddings=prototypes,
        attention_dim=attention_dim,
        bottleneck_dim=bottleneck_dim,
        maximum_logit_shift=maximum_logit_shift,
    )
    _clinical_named_parameters(model)
    return model


def generate_protocol_spec(
    *,
    sharp_protocol_path: Path = SHARP_PROTOCOL_PATH,
    cache_dir: Path = CLINICAL_CACHE_DIR,
) -> dict[str, Any]:
    """Return the closed, hash-bound specification without writing anything."""
    folds = [resolve_sharp_fold(fold, protocol_path=sharp_protocol_path) for fold in FOLDS]
    cache = load_clinical_cache(cache_dir=cache_dir)
    sources = {
        "training_core": Path(__file__),
        "paired_model": Path(Phase24BPairedCriterionModel.__module__.replace(".", "/") + ".py"),
    }
    paired_path = ROOT / sources["paired_model"]
    spec: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "SPECIFIED_NOT_FROZEN",
        "identity": {"seed": SEED, "folds": list(FOLDS), "baseline_arm": "sharp"},
        "training": {
            "device": "cpu",
            "epochs": EPOCHS,
            "effective_batch_size": BATCH_SIZE,
            "criterion_loss": "mean of seven independently class-balanced BCE losses",
            "objective": "0.5 * true_pair_bce + 0.5 * deterministic_modality_dropout_bce",
            "clinical_modality_dropout": MODALITY_DROPOUT,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "optimizer": "AdamW",
            "optimizer_groups": {key: dict(value) for key, value in OPTIMIZER_GROUPS.items()},
            "scheduler": dict(SCHEDULER),
            "early_stopping": False,
            "best_epoch_selection": False,
        },
        "model": {
            "class": "Phase24BPairedCriterionModel",
            "attention_dim": 64,
            "bottleneck_dim": 32,
            "maximum_logit_shift": 1.0,
            "trainable_scope": "clinical_tail.adapters and cross_modal_fusion only",
            "protected_outputs": ["diagnosis", "melanoma", "sharp baseline criteria when clinical absent"],
        },
        "artifacts": {
            "sharp_folds": [item.identity() for item in folds],
            "clinical_cache": cache.identity(),
            "sources": {
                "training_core": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
                "paired_model": {"path": str(paired_path), "sha256": _sha256(paired_path)},
            },
        },
        "hard_scope": {
            "derm7pt_development_only": True,
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_or_final_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
    }
    digest = _canonical_hash(spec)
    return {
        **spec,
        "protocol_sha256": digest,
        "signature_payload": {
            "schema_version": PROTOCOL_SCHEMA,
            "artifact_sha256": digest,
        },
    }


def validate_protocol_spec(
    protocol: Mapping[str, Any],
    *,
    binding: SharpFoldBinding | None = None,
    cache: ClinicalCache | None = None,
) -> str:
    """Validate fixed science and, when supplied, live upstream bindings."""
    payload = dict(protocol)
    recorded = payload.pop("protocol_sha256", None)
    signature = payload.pop("signature_payload", None)
    observed = _canonical_hash(payload)
    if recorded != observed:
        raise AssertionError("Phase 2.4B protocol digest does not match its payload")
    if signature is not None and signature != {
        "schema_version": PROTOCOL_SCHEMA,
        "artifact_sha256": observed,
    }:
        raise AssertionError("Phase 2.4B protocol signature payload changed")
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise AssertionError("Phase 2.4B protocol schema changed")
    if payload.get("status") not in {"SPECIFIED_NOT_FROZEN", "FROZEN"}:
        raise AssertionError("Phase 2.4B protocol status is invalid")
    if payload.get("identity") != {
        "seed": SEED, "folds": list(FOLDS), "baseline_arm": "sharp"
    }:
        raise AssertionError("Phase 2.4B closed identity changed")
    expected_training = {
        "device": "cpu",
        "epochs": EPOCHS,
        "effective_batch_size": BATCH_SIZE,
        "criterion_loss": "mean of seven independently class-balanced BCE losses",
        "objective": "0.5 * true_pair_bce + 0.5 * deterministic_modality_dropout_bce",
        "clinical_modality_dropout": MODALITY_DROPOUT,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer": "AdamW",
        "optimizer_groups": {key: dict(value) for key, value in OPTIMIZER_GROUPS.items()},
        "scheduler": dict(SCHEDULER),
        "early_stopping": False,
        "best_epoch_selection": False,
    }
    if payload.get("training") != expected_training:
        raise AssertionError("Phase 2.4B training semantics changed")
    expected_model = {
        "class": "Phase24BPairedCriterionModel",
        "attention_dim": 64,
        "bottleneck_dim": 32,
        "maximum_logit_shift": 1.0,
        "trainable_scope": "clinical_tail.adapters and cross_modal_fusion only",
        "protected_outputs": [
            "diagnosis", "melanoma", "sharp baseline criteria when clinical absent"
        ],
    }
    if payload.get("model") != expected_model:
        raise AssertionError("Phase 2.4B model semantics changed")
    scope = payload.get("hard_scope", {})
    if scope != {
        "derm7pt_development_only": True,
        "derm7pt_official_test_rows_loaded": 0,
        "ham_calibration_or_final_rows_loaded": 0,
        "bcn_validation_or_final_rows_loaded": 0,
        "ph2_rows_loaded": 0,
    }:
        raise AssertionError("Phase 2.4B hard scope changed")
    artifacts = payload.get("artifacts", {})
    sharp_rows = artifacts.get("sharp_folds", []) if isinstance(artifacts, Mapping) else []
    if len(sharp_rows) != len(FOLDS) or [row.get("fold") for row in sharp_rows] != list(FOLDS):
        raise AssertionError("Phase 2.4B sharp fold bindings changed")
    if binding is not None and sharp_rows[binding.fold] != binding.identity():
        raise AssertionError("Phase 2.4B current sharp fold binding changed")
    if cache is not None and artifacts.get("clinical_cache") != cache.identity():
        raise AssertionError("Phase 2.4B clinical cache binding changed")
    return observed


def _clinical_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    values = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not values:
        raise AssertionError("paired model has no trainable clinical parameters")
    allowed = ("clinical_tail.adapters.", "cross_modal_fusion.")
    if any(not name.startswith(allowed) for name in values):
        raise AssertionError(f"non-clinical parameter is trainable: {sorted(values)}")
    return values


_MUTABLE_PREFIXES = ("clinical_tail.adapters.", "cross_modal_fusion.")


def mutable_clinical_state(model: nn.Module) -> dict[str, Tensor]:
    """Return the complete bounded clinical state and no frozen comparator state."""
    state = model.state_dict()
    mutable = {
        name: value.detach().cpu()
        for name, value in state.items()
        if name.startswith(_MUTABLE_PREFIXES)
    }
    trainable = set(_clinical_named_parameters(model))
    if not mutable or not trainable.issubset(mutable):
        missing = sorted(trainable.difference(mutable))
        raise AssertionError(f"mutable clinical state is incomplete: {missing}")
    return mutable


def load_mutable_clinical_state(
    model: nn.Module, state: Mapping[str, Tensor]
) -> None:
    """Strictly restore only the exact clinical adapter/fusion state surface."""
    if not isinstance(state, Mapping):
        raise TypeError("mutable clinical state must be a mapping")
    expected = set(mutable_clinical_state(model))
    observed = set(state)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise AssertionError(
            f"mutable clinical state keys changed; missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, Tensor) for value in state.values()):
        raise TypeError("every mutable clinical state value must be a tensor")
    adapter_prefix = "clinical_tail.adapters."
    fusion_prefix = "cross_modal_fusion."
    adapter_state = {
        name[len(adapter_prefix):]: value
        for name, value in state.items()
        if name.startswith(adapter_prefix)
    }
    fusion_state = {
        name[len(fusion_prefix):]: value
        for name, value in state.items()
        if name.startswith(fusion_prefix)
    }
    model.clinical_tail.adapters.load_state_dict(adapter_state, strict=True)
    model.cross_modal_fusion.load_state_dict(fusion_state, strict=True)


def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    named = _clinical_named_parameters(model)
    adapter = [parameter for name, parameter in named.items() if name.startswith("clinical_tail.adapters.")]
    projection_names = (
        "cross_modal_fusion.prototype_projection.",
        "cross_modal_fusion.derm_projection.",
        "cross_modal_fusion.clinical_projection.",
    )
    projections = [parameter for name, parameter in named.items() if name.startswith(projection_names)]
    controls = [
        parameter for name, parameter in named.items()
        if name in {
            "cross_modal_fusion.log_temperature",
            "cross_modal_fusion.raw_mixture_weight",
            "cross_modal_fusion.raw_output_scale",
            "cross_modal_fusion.gate_bias",
            "cross_modal_fusion.gate_uncertainty_weight",
        }
    ]
    groups = (adapter, projections, controls)
    if any(not group for group in groups):
        raise AssertionError("one or more fixed optimizer groups are empty")
    grouped = {id(parameter) for group in groups for parameter in group}
    if grouped != {id(parameter) for parameter in named.values()}:
        raise AssertionError("fixed optimizer groups do not exactly cover trainable parameters")
    configs = tuple(OPTIMIZER_GROUPS.values())
    return torch.optim.AdamW([
        {
            "params": group,
            "lr": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "group_name": name,
        }
        for name, group, config in zip(OPTIMIZER_GROUPS, groups, configs, strict=True)
    ])


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup = max(1, int(math.ceil(total_steps * float(SCHEDULER["warmup_ratio"]))))
    floor = float(SCHEDULER["minimum_lr_ratio"])

    def scale(step: int) -> float:
        current = step + 1
        if current <= warmup:
            return current / warmup
        progress = min(1.0, (current - warmup) / max(1, total_steps - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def class_balanced_criterion_weights(targets: np.ndarray, train: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype="float32")
    train = np.asarray(train, dtype=bool)
    if targets.ndim != 2 or targets.shape[1] != 7 or train.shape != (len(targets),):
        raise ValueError("invalid criterion target/mask shape")
    if not train.any():
        raise ValueError("training fold is empty")
    result = np.ones_like(targets, dtype="float32")
    for criterion in range(7):
        labels = targets[train, criterion].astype("int64")
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"criterion {criterion} has one class in the training fold")
        class_weights = len(labels) / (2.0 * counts.astype("float64"))
        result[train, criterion] = class_weights[labels].astype("float32")
    return result


def _criterion_bce(logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    if logits.shape != targets.shape or logits.shape != weights.shape or logits.ndim != 2 or logits.shape[1] != 7:
        raise ValueError("criterion logits, targets, and weights must have shape [batch, 7]")
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return torch.stack([
        (raw[:, index] * weights[:, index]).sum() / weights[:, index].sum().clamp_min(1e-12)
        for index in range(7)
    ]).mean()


def paired_training_loss(
    model: nn.Module,
    derm: Tensor,
    clinical: Tensor,
    available: Tensor,
    retained_after_dropout: Tensor,
    targets: Tensor,
    weights: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    true_output = model(derm, clinical, available)
    dropout_mask = available & retained_after_dropout
    dropout_criteria = torch.where(
        dropout_mask.unsqueeze(1),
        true_output["criteria"],
        true_output["base_criteria"],
    )
    true_bce = _criterion_bce(true_output["criteria"], targets, weights)
    dropout_bce = _criterion_bce(dropout_criteria, targets, weights)
    total = 0.5 * true_bce + 0.5 * dropout_bce
    return total, {
        "true_pair_bce": true_bce,
        "modality_dropout_bce": dropout_bce,
        "true_pair_available_fraction": available.float().mean(),
        "dropout_available_fraction": dropout_mask.float().mean(),
    }


def load_training_data(cache: ClinicalCache) -> FoldTrainingData:
    data = sharp_factor._base._derm_data()
    result = FoldTrainingData(
        derm_tokens=cache.derm_tokens,
        clinical_tokens=cache.clinical_tokens,
        clinical_available=np.asarray(cache.clinical_available),
        criterion_targets=np.asarray(data["derm_criteria"], dtype="float32"),
        folds=np.asarray(data["derm_folds"], dtype="int64"),
    )
    result.validate()
    if len(result.folds) != phase24b_inputs.N_CASES:
        raise AssertionError("Phase 2.4B requires all 616 development rows")
    return result


def _epoch_batches(indices: np.ndarray, *, fold: int, epoch: int) -> list[np.ndarray]:
    if len(indices) == 0:
        raise ValueError("training fold is empty")
    rng = np.random.default_rng(SEED + fold * 1000 + epoch * 3 + 1)
    shuffled = rng.permutation(indices)
    steps = math.ceil(len(indices) / BATCH_SIZE)
    needed = steps * BATCH_SIZE
    tiled = np.resize(shuffled, needed)
    return [tiled[start:start + BATCH_SIZE] for start in range(0, needed, BATCH_SIZE)]


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_torch(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    torch.save(_cpu_tree(value), temporary)
    # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = _sha256(path)
    sidecar = _sidecar(path)
    temp_sidecar = Path(str(sidecar) + ".partial")
    temp_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temp_sidecar, sidecar)
    return digest


def _resume_identity(protocol_sha256: str, binding: SharpFoldBinding, cache: ClinicalCache, fold: int) -> dict[str, Any]:
    return {
        "format": RESUME_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "seed": SEED,
        "fold": fold,
        "sharp_protocol_sha256": binding.protocol_sha256,
        "sharp_training_receipt_sha256": binding.training_receipt_sha256,
        "sharp_residual_checkpoint_sha256": binding.residual_checkpoint_sha256,
        "clinical_cache_manifest_sha256": cache.manifest_sha256,
    }


def train_fold(
    fold: int,
    *,
    output_dir: Path,
    protocol_spec: Mapping[str, Any] | None = None,
    sharp_protocol_path: Path = SHARP_PROTOCOL_PATH,
    cache_dir: Path = CLINICAL_CACHE_DIR,
    model_factory: Callable[[SharpFoldBinding], nn.Module] = construct_paired_model,
    training_data: FoldTrainingData | None = None,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Train one fold on CPU, atomically resuming only an identical run."""
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    output_dir = Path(output_dir)
    binding = resolve_sharp_fold(fold, protocol_path=sharp_protocol_path)
    cache = load_clinical_cache(cache_dir=cache_dir)
    generated = generate_protocol_spec(
        sharp_protocol_path=sharp_protocol_path, cache_dir=cache_dir
    ) if protocol_spec is None else dict(protocol_spec)
    protocol_sha = validate_protocol_spec(generated, binding=binding, cache=cache)
    data = load_training_data(cache) if training_data is None else training_data
    data.validate()
    train_mask = np.asarray(data.folds != fold)
    train_indices = np.flatnonzero(train_mask)
    weights = class_balanced_criterion_weights(data.criterion_targets, train_mask)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = model_factory(binding).to(torch.device("cpu"))
    optimizer = build_optimizer(model)
    steps = math.ceil(len(train_indices) / BATCH_SIZE)
    scheduler = build_scheduler(optimizer, EPOCHS * steps)
    resume_path = output_dir / f"fold_{fold}" / "epoch_resume_state.pt"
    expected_identity = _resume_identity(protocol_sha, binding, cache, fold)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume_path.exists() or _sidecar(resume_path).exists():
        _verify_artifact(resume_path)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        observed_identity = {key: payload.get(key) for key in expected_identity}
        if observed_identity != expected_identity:
            raise AssertionError("Phase 2.4B resume identity mismatch")
        start_epoch = int(payload.get("completed_epochs", -1))
        history = list(payload.get("history", []))
        if not 1 <= start_epoch <= EPOCHS or len(history) != start_epoch:
            raise AssertionError("Phase 2.4B resume epoch/history mismatch")
        if "model_state" in payload:
            raise AssertionError(
                "legacy full-model Phase 2.4B resume is not accepted by the bounded-state contract"
            )
        load_mutable_clinical_state(model, payload["mutable_clinical_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])

    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        dropout_rng = np.random.default_rng(SEED + fold * 1000 + epoch * 3 + 2)
        totals: list[float] = []
        component_sums: dict[str, float] = {}
        epoch_started = time.perf_counter()
        for indices in _epoch_batches(train_indices, fold=fold, epoch=epoch):
            derm = torch.from_numpy(np.asarray(data.derm_tokens[indices], dtype="float32").copy())
            clinical = torch.from_numpy(np.asarray(data.clinical_tokens[indices], dtype="float32").copy())
            available = torch.from_numpy(np.asarray(data.clinical_available[indices], dtype=bool).copy())
            retained = torch.from_numpy((dropout_rng.random(len(indices)) >= MODALITY_DROPOUT))
            targets = torch.from_numpy(np.asarray(data.criterion_targets[indices], dtype="float32").copy())
            batch_weights = torch.from_numpy(np.asarray(weights[indices], dtype="float32").copy())
            optimizer.zero_grad(set_to_none=True)
            loss, components = paired_training_loss(
                model, derm, clinical, available, retained, targets, batch_weights
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Phase 2.4B paired loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                _clinical_named_parameters(model).values(), GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite Phase 2.4B clinical gradient")
            if any(parameter.grad is not None for parameter in model.baseline_model.parameters()):
                raise AssertionError("gradient reached frozen sharp baseline")
            optimizer.step()
            scheduler.step()
            totals.append(float(loss.detach()))
            for name, value in components.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value.detach())
        history.append({
            "epoch": epoch + 1,
            "mean_training_loss": float(np.mean(totals)),
            "component_means": {name: value / steps for name, value in component_sums.items()},
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "elapsed_seconds": time.perf_counter() - epoch_started,
        })
        _atomic_torch(resume_path, {
            **expected_identity,
            "completed_epochs": epoch + 1,
            "mutable_clinical_state": mutable_clinical_state(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "history": history,
        })
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            return {
                "schema_version": RESULT_SCHEMA,
                "status": "INCOMPLETE",
                **expected_identity,
                "completed_epochs": epoch + 1,
                "resume": {"path": str(resume_path), "sha256": _sha256(resume_path)},
                "history": history,
            }

    model.to("cpu")
    checkpoint_path = output_dir / f"fold_{fold}" / "final_clinical_state.pt"
    clinical_state = mutable_clinical_state(model)
    checkpoint_sha = _atomic_torch(checkpoint_path, clinical_state)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        **expected_identity,
        "completed_epochs": EPOCHS,
        "effective_batch_size": BATCH_SIZE,
        "steps_per_epoch": steps,
        "training_rows": len(train_indices),
        "optimizer_groups": {key: dict(value) for key, value in OPTIMIZER_GROUPS.items()},
        "scheduler": dict(SCHEDULER),
        "objective": {
            "true_pair_bce_weight": 0.5,
            "modality_dropout_bce_weight": 0.5,
            "modality_dropout_probability": MODALITY_DROPOUT,
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "resume": {"path": str(resume_path), "sha256": _sha256(resume_path)},
        "history": history,
        "training_device": "cpu",
        "elapsed_seconds": time.perf_counter() - started,
        "hard_scope": dict(generated.get("hard_scope", {})),
    }
    result_digest = _canonical_hash(result)
    return {
        **result,
        "result_sha256": result_digest,
        "signature_payload": {
            "schema_version": RESULT_SCHEMA,
            "artifact_sha256": result_digest,
        },
    }


__all__ = [
    "BATCH_SIZE", "CLINICAL_CACHE_DIR", "ClinicalCache", "EPOCHS", "FOLDS",
    "FoldTrainingData", "MODALITY_DROPOUT", "OPTIMIZER_GROUPS", "SEED",
    "SHARP_PROTOCOL_PATH", "SharpFoldBinding", "build_optimizer",
    "build_scheduler", "class_balanced_criterion_weights",
    "construct_paired_model", "generate_protocol_spec", "load_clinical_cache",
    "load_mutable_clinical_state", "load_training_data", "mutable_clinical_state",
    "paired_training_loss", "resolve_sharp_fold", "train_fold",
    "validate_protocol_spec",
]
