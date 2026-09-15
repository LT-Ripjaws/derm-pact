"""Fixed Phase 2.4A localized-criterion residual experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from . import dermpact_phase2_architecture_ladder as ladder
from . import dermpact_phase2_backbone_adaptation as adaptation
from . import dermpact_phase2_backbone_training as legacy_training
from . import dermpact_phase2_hybrid_confirmation as confirmation
from . import dermpact_phase2_hybrid_screen as screen
from .dermpact_phase2_4_model import Phase24LocalizedCriterionModel
from .dermpact_phase2_vector_bootstrap import interval, stratified_bootstrap_weights


ROOT = screen.ROOT
MANIFEST_DIR = ROOT / "research/manifests/dermpact_phase2_4/localized_criterion_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_4/localized_criterion_v1"
IMPLEMENTATION_DIR = ROOT / "research/manifests/dermpact_phase2_4/implementation_v1"
PROTOCOL_PATH = MANIFEST_DIR / "localized_criterion_protocol.json"
COMPARISON_PATH = MANIFEST_DIR / "localized_criterion_comparison.json"
FINAL_RECEIPT_PATH = MANIFEST_DIR / "localized_criterion_final_receipt.json"
STATUS_PATH = EXPERIMENT_DIR / "controller_status.json"

PROTOCOL_SCHEMA = "dermpact-phase2-4a-localized-criterion-protocol-v1"
RESUME_SCHEMA = "dermpact-phase2-4a-epoch-resume-v1"
FOLD_SCHEMA = "dermpact-phase2-4a-fold-training-receipt-v1"
CANONICAL_SCHEMA = "dermpact-phase2-4a-canonical-fold-receipt-v1"
SEED = 20260827
FOLDS = tuple(range(5))
STRATEGY = "derm_ham"
ARM = "lora_last_two"
BOOTSTRAP_SEED = 20270241
RETAINED_ROOT = (
    ROOT / "research/experiments/dermpact_phase2_3/backbone_adaptation_v1"
    / STRATEGY / ARM / f"seed_{SEED}"
)
PROTOTYPES_PATH = IMPLEMENTATION_DIR / "semantic_criterion_embeddings_v1.npy"
SEMANTIC_MANIFEST_PATH = IMPLEMENTATION_DIR / "semantic_embeddings_v1_manifest.json"
EQUIVALENCE_RECEIPT = IMPLEMENTATION_DIR / "phase24a_equivalence_receipt.json"
BENCHMARK_RECEIPT = IMPLEMENTATION_DIR / "phase24a_directml_benchmark_receipt.json"
EXACT_AMD_ADAPTER = "AMD Radeon RX 5700 XT"
CANONICAL_ENVIRONMENT = {
    "python": "3.12.13",
    "numpy": "2.4.4",
    "torch": "2.12.1+cpu",
    "device": "cpu",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"missing artifact or sidecar: {path}")
    digest = sha256_file(path)
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise AssertionError(f"SHA-256 sidecar mismatch: {_relative(path)}")
    return digest


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(body)
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".partial")
    sidecar_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(sidecar_tmp, sidecar)
    return digest


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


def _atomic_torch(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(_cpu_tree(payload), temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _artifact(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _verify_sidecar(path)}


def _require_pass_receipt(path: Path, *, protocol: str) -> dict[str, Any]:
    _verify_sidecar(path)
    value = _load_json(path)
    if value.get("status") != "PASS" or value.get("protocol") != protocol:
        raise RuntimeError(f"required smoke receipt did not PASS: {_relative(path)}")
    return value


def _retained_artifacts(fold: int) -> dict[str, Any]:
    fold_dir = RETAINED_ROOT / f"fold_{fold}"
    checkpoint = fold_dir / "epoch_12_state.pt"
    receipt = fold_dir / "fold_result.json"
    receipt_sha = _verify_sidecar(receipt)
    result = _load_json(receipt)
    if (result.get("strategy"), result.get("arm"), result.get("seed"), result.get("fold")) != (
        STRATEGY, ARM, SEED, fold
    ):
        raise AssertionError(f"retained fold receipt identity mismatch: fold {fold}")
    recorded = result.get("checkpoint", {})
    if ROOT / str(recorded.get("path")) != checkpoint:
        raise AssertionError(f"retained checkpoint path mismatch: fold {fold}")
    checkpoint_sha = sha256_file(checkpoint)
    if recorded.get("sha256") != checkpoint_sha:
        raise AssertionError(f"retained checkpoint receipt hash mismatch: fold {fold}")
    return {
        "fold": fold,
        "checkpoint": {"path": _relative(checkpoint), "sha256": checkpoint_sha},
        "fold_receipt": {"path": _relative(receipt), "sha256": receipt_sha},
    }


def _source_artifacts() -> dict[str, Any]:
    paths = {
        "public_training_api": ROOT / "dermpact/eval/dermpact_phase2_4a_training.py",
        "localized_training": Path(__file__),
        "localized_model": ROOT / "dermpact/eval/dermpact_phase2_4_model.py",
        "criterion_residual": ROOT / "dermpact/perception/criterion_residual.py",
        "legacy_training": ROOT / "dermpact/eval/dermpact_phase2_backbone_training.py",
        "legacy_adaptation": ROOT / "dermpact/eval/dermpact_phase2_backbone_adaptation.py",
        "directml_launcher": ROOT / "research/scripts/dermpact_phase2_4a_directml_train.py",
        "controller": ROOT / "research/scripts/dermpact_phase2_4a_controller.py",
        "full_run_controller": ROOT / "research/scripts/dermpact_phase2_4a_full_run_controller.py",
    }
    return {
        name: {"path": _relative(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def build_protocol() -> dict[str, Any]:
    """Freeze the fixed Stage-1 identity after all prerequisites verify."""
    semantic = np.load(PROTOTYPES_PATH, allow_pickle=False)
    if semantic.shape != (7, 2, 512) or semantic.dtype != np.float32:
        raise AssertionError("semantic criterion embeddings must be float32 [7,2,512]")
    if not np.isfinite(semantic).all():
        raise FloatingPointError("semantic criterion embeddings are non-finite")
    _require_pass_receipt(
        EQUIVALENCE_RECEIPT,
        protocol="dermpact-phase2-4a-zero-residual-equivalence-v1",
    )
    benchmark = _require_pass_receipt(
        BENCHMARK_RECEIPT,
        protocol="dermpact-phase2-4a-cpu-vs-amd-directml-benchmark-v1",
    )
    if benchmark.get("directml", {}).get("selected_adapter_name") != EXACT_AMD_ADAPTER:
        raise RuntimeError("DirectML smoke did not select the exact pinned AMD adapter")
    retained = [_retained_artifacts(fold) for fold in FOLDS]
    fold_manifest = screen.FOLD_MANIFEST
    protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "FROZEN",
        "experiment": "Phase 2.4A Stage 1 localized criterion residual",
        "identity": {
            "strategy": STRATEGY,
            "arm": ARM,
            "adaptation_scale": 1.0,
            "seed": SEED,
            "folds": list(FOLDS),
        },
        "model": {
            "class": "dermpact.eval.dermpact_phase2_4_model.Phase24LocalizedCriterionModel",
            "attention_dim": 128,
            "prototype_shape": [7, 2, 512],
            "attention_shape": ["N", 7, 2, 196],
            "residual_initialization": "exactly zero through residual_scale",
            "bounding_or_clamping": False,
            "dropout": False,
            "frozen_base": ["visual tail", "LoRA", "task head"],
        },
        "training": {
            "dataset": "Derm7pt development training rows only",
            "epochs": 20,
            "effective_batch_size": 64,
            "directml_microbatch_size": 8,
            "dtype": "float32",
            "steps_per_epoch": "ceil(fold_training_derm7pt_rows / 64)",
            "cyclic_batch_seed_formula": "seed + fold*1000 + epoch*3 + 2",
            "early_stopping": False,
            "best_epoch_selection": False,
            "loss": "retained equal-family masked Derm7pt multitask plus truth-anchored consistency",
            "consistency_weight": 0.10,
            "optimizer": {
                "name": "AdamW",
                "projection_group": {
                    "members": ["prototype_projection", "patch_key", "patch_value"],
                    "learning_rate": 0.0003,
                    "weight_decay": 0.0001,
                },
                "scale_group": {
                    "members": ["residual_scale"],
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                },
            },
            "scheduler": {"name": "existing warmup cosine", "warmup_ratio": 0.10, "minimum_lr_ratio": 0.01},
            "gradient_clip_norm": 1.0,
        },
        "canonical_evaluation": {
            "environment": "venv_p2 CPU",
            "expected": dict(CANONICAL_ENVIRONMENT),
            "raw_arrays": [
                "row_indices", "diagnosis_logits", "melanoma_logits",
                "base_criterion_logits", "final_criterion_logits",
                "residual_logits", "criterion_attention",
            ],
            "protected_logit_equality": "diagnosis and melanoma torch.equal for every batch",
            "residual_state_reload": "exact tensor equality",
            "output_reload": {
                "outputs": [
                    "diagnosis", "melanoma", "criteria",
                    "criterion_residual", "criterion_attention",
                ],
                "rtol": 0.0,
                "max_abs_tolerance": 1e-6,
                "nonfinite_allowed": False,
            },
        },
        "bootstrap": {"replicates": 2000, "seed": BOOTSTRAP_SEED, "unit": "Derm7pt case", "paired": True},
        "endpoints": {
            "primary": "seven-criterion mean AUROC",
            "reported": [
                "per-criterion AUROC", "seven-criterion mean AUROC",
                "melanoma-plus-seven eight-output mean AUROC", "balanced accuracy",
                "melanoma AUROC", "direct-rule disagreement",
                "joint-correct agreement", "calibration",
            ],
            "promotion": {
                "primary_delta": "> 0 and paired 95% CI lower > 0",
                "maximum_protected_fall": 0.02,
                "maximum_disagreement_increase": 0.02,
            },
        },
        "roles": {
            "allowed": ["Derm7pt 616-case development"],
            "retained_base_training": "Derm7pt development plus HAM training",
            "never_loaded": ["Derm7pt official test", "HAM calibration", "HAM final", "BCN validation", "BCN final", "PH2"],
        },
        "environments": {
            "training": "venv_dml / DirectML / AMD Radeon RX 5700 XT / microbatch 8",
            "canonical": dict(CANONICAL_ENVIRONMENT),
        },
        "artifacts": {
            "fold_manifest": {"path": _relative(fold_manifest), "sha256": sha256_file(fold_manifest)},
            "semantic_embeddings": _artifact(PROTOTYPES_PATH),
            "semantic_manifest": _artifact(SEMANTIC_MANIFEST_PATH),
            "equivalence_receipt": _artifact(EQUIVALENCE_RECEIPT),
            "amd_benchmark_receipt": _artifact(BENCHMARK_RECEIPT),
            "retained_folds": retained,
            "sources": _source_artifacts(),
            "environment_locks": {
                "venv_p2": {"path": "requirements-p2.lock", "sha256": sha256_file(ROOT / "requirements-p2.lock")},
                "venv_dml": {"path": "requirements-dml.lock", "sha256": sha256_file(ROOT / "requirements-dml.lock")},
            },
        },
        "hard_scope_checks": {
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_or_final_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
    }
    digest = _atomic_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": digest}


def load_protocol() -> tuple[dict[str, Any], str]:
    digest = _verify_sidecar(PROTOCOL_PATH)
    protocol = _load_json(PROTOCOL_PATH)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("status") != "FROZEN":
        raise AssertionError("Phase 2.4A protocol identity changed")
    for artifact in protocol["artifacts"]["sources"].values():
        path = ROOT / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise AssertionError(f"protocol-bound source changed: {artifact['path']}")
    return protocol, digest


def controller_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the public, fixed interruption-safe execution matrix."""
    identity = protocol["identity"]
    return (
        [{"stage": "train", "fold": int(fold), "environment": "venv_dml"} for fold in identity["folds"]]
        + [{"stage": "reevaluate", "fold": int(fold), "environment": "venv_p2"} for fold in identity["folds"]]
        + [{"stage": "finalize", "fold": None, "environment": "venv_p2"}]
    )


def _derm_data() -> dict[str, Any]:
    rows = screen._derm7pt_records()
    fold_payload = _load_json(screen.FOLD_MANIFEST)
    fold_map = {str(row["group_id"]): int(row["fold"]) for row in fold_payload["derm7pt"]["folds"]}
    if set(fold_map) != {str(row["group_id"]) for row in rows}:
        raise AssertionError("Derm7pt development folds changed")
    folds = np.asarray([fold_map[str(row["group_id"])] for row in rows], dtype="int64")
    raw_global = [screen.derm_raw_to_global(str(row["diagnosis"]["raw"])) for row in rows]
    return {
        "derm_rows": rows,
        "derm_folds": folds,
        "derm_global": np.asarray([screen.GLOBAL_INDEX[value] for value in raw_global], dtype="int64"),
        "derm_melanoma": np.asarray([value == "melanoma" for value in raw_global], dtype="float32"),
        "derm_criteria": screen._criterion_matrix(rows),
        "derm_tokens": legacy_training._load_prefix_tokens("derm7pt", rows),
    }


def _derm_weights(data: Mapping[str, Any], fold: int) -> dict[str, np.ndarray]:
    train = np.asarray(data["derm_folds"]) != fold
    diagnosis_labels = np.asarray([screen.GLOBAL_CLASSES[int(value)] for value in data["derm_global"]])
    weights = {
        "derm_diagnosis": np.ones(len(train), dtype="float32"),
        "derm_melanoma": np.ones(len(train), dtype="float32"),
        "derm_criteria": np.ones((len(train), len(screen.CRITERIA)), dtype="float32"),
    }
    weights["derm_diagnosis"][train] = screen._balanced_weights(diagnosis_labels[train]).astype("float32")
    weights["derm_melanoma"][train] = screen._balanced_weights(data["derm_melanoma"][train]).astype("float32")
    for index in range(len(screen.CRITERIA)):
        weights["derm_criteria"][train, index] = screen._balanced_weights(data["derm_criteria"][train, index]).astype("float32")
    return weights


def _retained_state(protocol: Mapping[str, Any], fold: int) -> tuple[Mapping[str, Any], str]:
    record = protocol["artifacts"]["retained_folds"][fold]
    if int(record["fold"]) != fold:
        raise AssertionError("protocol retained-fold order changed")
    path = ROOT / record["checkpoint"]["path"]
    digest = sha256_file(path)
    if digest != record["checkpoint"]["sha256"]:
        raise AssertionError("retained checkpoint changed after protocol freeze")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("arm") != ARM or "model_state" not in payload:
        raise AssertionError("retained checkpoint payload identity changed")
    return payload["model_state"], digest


def _new_model(
    visual: torch.nn.Module,
    *,
    model_state: Mapping[str, Any],
    prototypes: Tensor,
    attention_dim: int,
    initialization_seed: int,
) -> Phase24LocalizedCriterionModel:
    torch.manual_seed(initialization_seed)
    base = adaptation.AdaptedHybridModel(visual, ARM)
    base.load_state_dict(model_state, strict=True)
    base.visual_tail.capture_adapted_state()
    base.visual_tail.set_adaptation_scale(1.0)
    model = Phase24LocalizedCriterionModel(base, prototype_embeddings=prototypes, attention_dim=attention_dim)
    with torch.no_grad():
        model.criterion_residual.residual_scale.zero_()
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise AssertionError("retained Phase 2.3 base is not fully frozen")
    return model


def _optimizer(model: Phase24LocalizedCriterionModel, protocol: Mapping[str, Any]) -> torch.optim.AdamW:
    config = protocol["training"]["optimizer"]
    residual = model.criterion_residual
    projection = [
        *residual.prototype_projection.parameters(),
        *residual.patch_key.parameters(),
        *residual.patch_value.parameters(),
    ]
    return torch.optim.AdamW([
        {"params": projection, "lr": float(config["projection_group"]["learning_rate"]), "weight_decay": float(config["projection_group"]["weight_decay"])},
        {"params": [residual.residual_scale], "lr": float(config["scale_group"]["learning_rate"]), "weight_decay": float(config["scale_group"]["weight_decay"])},
    ])


def _fold_dir(fold: int) -> Path:
    return EXPERIMENT_DIR / f"seed_{SEED}" / f"fold_{fold}"


def _resume_path(fold: int) -> Path:
    return _fold_dir(fold) / "epoch_resume_state.pt"


def _residual_path(fold: int) -> Path:
    return _fold_dir(fold) / "final_residual_state.pt"


def _fold_receipt_path(fold: int) -> Path:
    return _fold_dir(fold) / "fold_training_receipt.json"


def _canonical_receipt_path(fold: int) -> Path:
    return _fold_dir(fold) / "canonical_fold_receipt.json"


def validate_resume_identity(payload: Mapping[str, Any], *, protocol_sha256: str, fold: int, source_sha256: str) -> None:
    """Validate the public immutable identity of a Phase 2.4A resume payload."""
    observed = (
        payload.get("format"), payload.get("protocol_sha256"), payload.get("seed"),
        payload.get("fold"), payload.get("source_checkpoint_sha256"),
    )
    expected = (RESUME_SCHEMA, protocol_sha256, SEED, fold, source_sha256)
    if observed != expected:
        raise AssertionError(f"epoch-resume identity mismatch: {observed!r}")


def _write_resume(
    fold: int, *, protocol_sha: str, source_sha: str, completed_epochs: int,
    model: Phase24LocalizedCriterionModel, optimizer: torch.optim.Optimizer,
    scheduler: Any, history: Sequence[Mapping[str, Any]],
) -> str:
    return _atomic_torch(_resume_path(fold), {
        "format": RESUME_SCHEMA,
        "protocol_sha256": protocol_sha,
        "source_checkpoint_sha256": source_sha,
        "seed": SEED,
        "fold": fold,
        "completed_epochs": completed_epochs,
        "residual_model_state": model.criterion_residual.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "history": list(history),
    })


def _preserve_rejected(path: Path, reason: Exception) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    marker = path.with_name(path.name + f".rejected-{stamp}.json")
    _atomic_json(marker, {"status": "REJECTED_PRESERVED", "source": _relative(path), "reason": str(reason)})
    for candidate in (path, path.with_suffix(path.suffix + ".sha256")):
        if candidate.exists():
            destination = candidate.with_name(candidate.name + f".rejected-{stamp}")
            os.replace(candidate, destination)


def _load_resume(
    fold: int, *, protocol_sha: str, source_sha: str,
    model: Phase24LocalizedCriterionModel, optimizer: torch.optim.Optimizer,
    scheduler: Any, device: Any, epochs: int,
) -> tuple[int, list[dict[str, Any]]] | None:
    path = _resume_path(fold)
    if not path.is_file() and not path.with_suffix(path.suffix + ".sha256").is_file():
        return None
    try:
        _verify_sidecar(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_resume_identity(payload, protocol_sha256=protocol_sha, fold=fold, source_sha256=source_sha)
        completed = int(payload["completed_epochs"])
        history = list(payload["history"])
        if not 1 <= completed <= epochs or len(history) != completed:
            raise AssertionError("epoch-resume completion/history mismatch")
        model.criterion_residual.load_state_dict(payload["residual_model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device)
        scheduler.load_state_dict(payload["scheduler_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        return completed, history
    except (AssertionError, EOFError, KeyError, OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as error:
        _preserve_rejected(path, error)
        return None


def _verified_training_fold(protocol_sha: str, fold: int) -> dict[str, Any] | None:
    path = _fold_receipt_path(fold)
    if not path.is_file() or not path.with_suffix(path.suffix + ".sha256").is_file():
        return None
    try:
        _verify_sidecar(path)
        value = _load_json(path)
        if (value.get("schema_version"), value.get("protocol_sha256"), value.get("seed"), value.get("fold")) != (FOLD_SCHEMA, protocol_sha, SEED, fold):
            raise AssertionError("fold training receipt identity mismatch")
        artifact = value["residual_checkpoint"]
        if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError("residual checkpoint hash mismatch")
        return value
    except (AssertionError, KeyError, OSError, TypeError, ValueError):
        return None


def train_fold(fold: int, *, device: Any, microbatch_size: int) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    if fold not in protocol["identity"]["folds"]:
        raise ValueError(fold)
    if microbatch_size != int(protocol["training"]["directml_microbatch_size"]):
        raise ValueError("training microbatch must match the frozen protocol")
    reused = _verified_training_fold(protocol_sha, fold)
    if reused is not None:
        return {"status": "REUSED_VERIFIED", **reused}
    initialization_seed = int(protocol["identity"]["seed"]) + fold
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    data = _derm_data()
    train_indices = np.flatnonzero(data["derm_folds"] != fold)
    effective_batch = int(protocol["training"]["effective_batch_size"])
    steps = math.ceil(len(train_indices) / effective_batch)
    epochs = int(protocol["training"]["epochs"])
    model_state, source_sha = _retained_state(protocol, fold)
    prototypes = torch.from_numpy(np.load(PROTOTYPES_PATH, allow_pickle=False).copy())
    visual, _, _ = adaptation._load_hybrid_visual()
    model = _new_model(visual, model_state=model_state, prototypes=prototypes, attention_dim=int(protocol["model"]["attention_dim"]), initialization_seed=initialization_seed).to(device)
    optimizer = _optimizer(model, protocol)
    if float(screen.WARMUP_RATIO) != float(protocol["training"]["scheduler"]["warmup_ratio"]) or float(screen.MIN_LR_RATIO) != float(protocol["training"]["scheduler"]["minimum_lr_ratio"]):
        raise AssertionError("retained scheduler constants changed")
    scheduler = screen._scheduler(optimizer, epochs * steps)
    weights = _derm_weights(data, fold)
    resumed = _load_resume(fold, protocol_sha=protocol_sha, source_sha=source_sha, model=model, optimizer=optimizer, scheduler=scheduler, device=device, epochs=epochs)
    start_epoch, history = resumed if resumed is not None else (0, [])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        batches = screen._cyclic_batches(
            train_indices, steps=steps, batch_size=effective_batch,
            rng=np.random.default_rng(SEED + fold * 1000 + epoch * 3 + 2),
        )
        losses: list[float] = []
        components: dict[str, float] = defaultdict(float)
        model.train()
        epoch_started = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss_value, values = legacy_training._microbatched_backward(
                model, dataset="derm", indices=next(batches), data=data, weights=weights,
                coefficient=1.0, microbatch_size=microbatch_size,
            )
            if not np.isfinite(loss_value):
                raise FloatingPointError("non-finite Phase 2.4A loss")
            gradient = torch.nn.utils.clip_grad_norm_(
                model.criterion_residual.parameters(), float(protocol["training"]["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(gradient).detach().cpu()):
                raise FloatingPointError("non-finite Phase 2.4A gradient")
            if any(parameter.grad is not None for parameter in model.base_model.parameters()):
                raise AssertionError("gradient reached the frozen retained base")
            optimizer.step()
            scheduler.step()
            losses.append(loss_value)
            for name, value in values.items():
                components[name] += float(value.detach().cpu())
        history.append({
            "epoch": epoch + 1,
            "mean_training_loss": float(np.mean(losses)),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "component_means": {name: value / steps for name, value in components.items()},
            "elapsed_seconds": time.perf_counter() - epoch_started,
        })
        _write_resume(fold, protocol_sha=protocol_sha, source_sha=source_sha, completed_epochs=epoch + 1, model=model, optimizer=optimizer, scheduler=scheduler, history=history)
        print(f"[phase24a/fold{fold}] epoch {epoch + 1}/{epochs} loss={history[-1]['mean_training_loss']:.6f}", flush=True)
    model.to("cpu")
    residual_path = _residual_path(fold)
    residual_sha = _atomic_torch(residual_path, model.criterion_residual.state_dict())
    receipt = {
        "schema_version": FOLD_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "training_backend_note": "DirectML training is execution evidence only",
        "protocol_sha256": protocol_sha,
        "seed": SEED,
        "fold": fold,
        "source_checkpoint_sha256": source_sha,
        "completed_epochs": epochs,
        "steps_per_epoch": steps,
        "training_rows": len(train_indices),
        "effective_batch_size": effective_batch,
        "microbatch_size": microbatch_size,
        "history": history,
        "residual_checkpoint": {"path": _relative(residual_path), "sha256": residual_sha},
        "training_device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "hard_scope_checks": dict(protocol["hard_scope_checks"]),
    }
    digest = _atomic_json(_fold_receipt_path(fold), receipt)
    return {**receipt, "receipt_sha256": digest}


def _canonical_environment() -> dict[str, Any]:
    observed = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"}
    return {"status": "PASS" if observed == CANONICAL_ENVIRONMENT else "FAIL", "expected": dict(CANONICAL_ENVIRONMENT), "observed": observed}


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


RELOAD_OUTPUT_NAMES = (
    "diagnosis",
    "melanoma",
    "criteria",
    "criterion_residual",
    "criterion_attention",
)


def validate_output_reload(
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
    *,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Require finite reload outputs within an absolute tolerance (rtol=0)."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("reload output tolerance must be finite and nonnegative")
    maximums: dict[str, float] = {}
    for name in RELOAD_OUTPUT_NAMES:
        if name not in before or name not in after:
            raise KeyError(f"reload output missing: {name}")
        left = torch.as_tensor(before[name]).detach().cpu()
        right = torch.as_tensor(after[name]).detach().cpu()
        if left.shape != right.shape:
            raise AssertionError(f"reload output shape changed: {name}")
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            raise FloatingPointError(f"reload output is nonfinite: {name}")
        maximum = float(torch.max(torch.abs(left - right)).item()) if left.numel() else 0.0
        if not math.isfinite(maximum):
            raise FloatingPointError(f"reload output difference is nonfinite: {name}")
        if maximum > tolerance:
            raise AssertionError(
                f"reload output exceeds absolute tolerance for {name}: "
                f"{maximum} > {tolerance}"
            )
        maximums[name] = maximum
    return maximums


def _state_dict_exact(
    expected: Mapping[str, Tensor], actual: Mapping[str, Tensor]
) -> bool:
    return set(expected) == set(actual) and all(
        torch.equal(
            torch.as_tensor(expected[name]).detach().cpu(),
            torch.as_tensor(actual[name]).detach().cpu(),
        )
        for name in expected
    )


def canonical_reevaluate_fold(fold: int) -> dict[str, Any]:
    environment = _canonical_environment()
    if environment["status"] != "PASS":
        raise RuntimeError(f"canonical reevaluation requires frozen venv_p2 CPU: {environment['observed']}")
    protocol, protocol_sha = load_protocol()
    training_receipt = _verified_training_fold(protocol_sha, fold)
    if training_receipt is None:
        raise RuntimeError(f"fold {fold} lacks verified training evidence")
    data = _derm_data()
    valid = np.flatnonzero(data["derm_folds"] == fold)
    model_state, source_sha = _retained_state(protocol, fold)
    visual, _, _ = adaptation._load_hybrid_visual()
    prototypes = torch.from_numpy(np.load(PROTOTYPES_PATH, allow_pickle=False).copy())
    model = _new_model(visual, model_state=model_state, prototypes=prototypes, attention_dim=int(protocol["model"]["attention_dim"]), initialization_seed=SEED + fold)
    residual_path = ROOT / training_receipt["residual_checkpoint"]["path"]
    if sha256_file(residual_path) != training_receipt["residual_checkpoint"]["sha256"]:
        raise AssertionError("residual checkpoint changed before canonical reevaluation")
    residual_state = torch.load(residual_path, map_location="cpu", weights_only=True)
    model.criterion_residual.load_state_dict(residual_state, strict=True)
    model.eval()
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    prediction_batch = int(protocol["training"]["effective_batch_size"])
    with torch.inference_mode():
        for start in range(0, len(valid), prediction_batch):
            selection = valid[start:start + prediction_batch]
            tokens = torch.from_numpy(np.asarray(data["derm_tokens"][selection], dtype="float32").copy())
            base = model.base_model(tokens)
            output = model(tokens)
            if not torch.equal(base["diagnosis"], output["diagnosis"]):
                raise AssertionError("canonical diagnosis logits changed")
            if not torch.equal(base["melanoma"], output["melanoma"]):
                raise AssertionError("canonical melanoma logits changed")
            for name, tensor in (
                ("diagnosis_logits", output["diagnosis"]),
                ("melanoma_logits", output["melanoma"]),
                ("base_criterion_logits", output["base_criteria"]),
                ("final_criterion_logits", output["criteria"]),
                ("residual_logits", output["criterion_residual"]),
                ("criterion_attention", output["criterion_attention"]),
            ):
                arrays[name].append(tensor.cpu().numpy().astype("float32", copy=False))
    combined = {name: np.concatenate(values) for name, values in arrays.items()}
    if combined["criterion_attention"].shape != (len(valid), 7, 2, 196):
        raise AssertionError("criterion attention shape changed")
    raw_path = _fold_dir(fold) / "canonical_derm7pt_oof_raw.npz"
    raw_sha = _save_npz(raw_path, row_indices=valid.astype("int64"), **combined)
    reloaded = _new_model(visual, model_state=model_state, prototypes=prototypes, attention_dim=int(protocol["model"]["attention_dim"]), initialization_seed=SEED + fold)
    reloaded_state = torch.load(residual_path, map_location="cpu", weights_only=True)
    reloaded.criterion_residual.load_state_dict(reloaded_state, strict=True)
    residual_state_reload_exact = _state_dict_exact(
        residual_state, reloaded.criterion_residual.state_dict()
    )
    if not residual_state_reload_exact:
        raise AssertionError("canonical residual state reload is not exact")
    reloaded.eval()
    probe = torch.from_numpy(np.asarray(data["derm_tokens"][valid[: min(8, len(valid))]], dtype="float32").copy())
    with torch.inference_mode():
        left, right = model(probe), reloaded(probe)
    output_reload_tolerance = float(
        protocol["canonical_evaluation"]["output_reload"]["max_abs_tolerance"]
    )
    output_reload_max_abs = validate_output_reload(
        left, right, tolerance=output_reload_tolerance
    )
    receipt = {
        "schema_version": CANONICAL_SCHEMA,
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "seed": SEED,
        "fold": fold,
        "source_checkpoint_sha256": source_sha,
        "residual_checkpoint_sha256": training_receipt["residual_checkpoint"]["sha256"],
        "raw_oof": {"path": _relative(raw_path), "sha256": raw_sha, "rows": len(valid)},
        "protected_logits_equal_base": True,
        "residual_state_reload_exact": residual_state_reload_exact,
        "output_reload_within_tolerance": True,
        "output_reload_tolerance": output_reload_tolerance,
        "output_reload_max_abs": output_reload_max_abs,
        "attention_semantics": "seven criterion-specific searches with frozen positive/negative anchors",
        "criterion_attention_shape": [len(valid), 7, 2, 196],
        "environment": environment,
        "hard_scope_checks": dict(protocol["hard_scope_checks"]),
    }
    digest = _atomic_json(_canonical_receipt_path(fold), receipt)
    return {**receipt, "receipt_sha256": digest}


def _canonical_fold(protocol_sha: str, fold: int) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    path = _canonical_receipt_path(fold)
    _verify_sidecar(path)
    receipt = _load_json(path)
    if (receipt.get("schema_version"), receipt.get("protocol_sha256"), receipt.get("fold"), receipt.get("status")) != (CANONICAL_SCHEMA, protocol_sha, fold, "PASS"):
        raise AssertionError(f"canonical fold receipt identity mismatch: {fold}")
    raw = ROOT / receipt["raw_oof"]["path"]
    if sha256_file(raw) != receipt["raw_oof"]["sha256"] or _verify_sidecar(raw) != receipt["raw_oof"]["sha256"]:
        raise AssertionError(f"canonical raw OOF hash mismatch: {fold}")
    return receipt, np.load(raw, allow_pickle=False)


def _prediction_rows(data: Mapping[str, Any], indices: np.ndarray, diagnosis: np.ndarray, melanoma: np.ndarray, criteria: np.ndarray) -> list[dict[str, Any]]:
    values = {
        "diagnosis_logits": diagnosis,
        "melanoma_logits": melanoma,
        "criterion_logits": criteria,
        "universal": torch.softmax(torch.from_numpy(diagnosis), dim=1).numpy(),
        "melanoma": torch.sigmoid(torch.from_numpy(melanoma)).numpy(),
        "criteria": torch.sigmoid(torch.from_numpy(criteria)).numpy(),
    }
    return screen._derm_prediction_rows(data, indices, values)


def _calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    truth = np.asarray([str(row["label"]) == "MEL" for row in rows], dtype=float)
    probability = np.asarray([float(row["melanoma_probability"]) for row in rows])
    return {"melanoma_brier": float(np.mean((probability - truth) ** 2))}


def finalize() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    data = _derm_data()
    base_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    fold_artifacts = []
    for fold in FOLDS:
        receipt, raw = _canonical_fold(protocol_sha, fold)
        indices = np.asarray(raw["row_indices"], dtype="int64")
        base_rows.extend(_prediction_rows(data, indices, raw["diagnosis_logits"], raw["melanoma_logits"], raw["base_criterion_logits"]))
        final_rows.extend(_prediction_rows(data, indices, raw["diagnosis_logits"], raw["melanoma_logits"], raw["final_criterion_logits"]))
        fold_artifacts.append({"fold": fold, "receipt": _artifact(_canonical_receipt_path(fold)), "raw_oof": receipt["raw_oof"]})
    base_rows = sorted(base_rows, key=lambda row: str(row["id"]))
    final_rows = sorted(final_rows, key=lambda row: str(row["id"]))
    if [row["id"] for row in base_rows] != [row["id"] for row in final_rows] or len(base_rows) != len(data["derm_rows"]):
        raise AssertionError("pooled paired Derm7pt OOF coverage changed")
    weights = stratified_bootstrap_weights(confirmation._derm_bootstrap_strata(base_rows), n_boot=int(protocol["bootstrap"]["replicates"]), seed=int(protocol["bootstrap"]["seed"]))
    metrics = ("criterion_mean_auc", "primary", "balanced_accuracy", "melanoma_auc", "direct_rule_disagreement", "joint_correct_agreement")
    comparisons: dict[str, Any] = {}
    for metric in metrics:
        base_samples = ladder._metric_samples(base_rows, dataset="derm7pt", weights=weights)[metric]
        final_samples = ladder._metric_samples(final_rows, dataset="derm7pt", weights=weights)[metric]
        point = ladder._point_metric(final_rows, dataset="derm7pt", metric=metric) - ladder._point_metric(base_rows, dataset="derm7pt", metric=metric)
        comparisons[metric] = interval(final_samples - base_samples, point, n_boot=int(protocol["bootstrap"]["replicates"]))
    base_targets, base_scores = confirmation._derm_targets_scores(base_rows)
    final_targets, final_scores = confirmation._derm_targets_scores(final_rows)
    criterion_names = [screen.CRITERION_SHORT[name] for name in screen.CRITERIA]
    from sklearn.metrics import roc_auc_score
    per_criterion = {
        name: {
            "base": float(roc_auc_score(base_targets[name], base_scores[name])),
            "localized": float(roc_auc_score(final_targets[name], final_scores[name])),
        }
        for name in criterion_names
    }
    gates = protocol["endpoints"]["promotion"]
    primary = comparisons["criterion_mean_auc"]
    protected = ("primary", "balanced_accuracy", "melanoma_auc", "joint_correct_agreement")
    protected_ok = all(comparisons[name]["estimate"] >= -float(gates["maximum_protected_fall"]) for name in protected)
    disagreement_ok = comparisons["direct_rule_disagreement"]["estimate"] <= float(gates["maximum_disagreement_increase"])
    promote = bool(primary["estimate"] > 0.0 and primary["ci_low"] > 0.0 and protected_ok and disagreement_ok)
    comparison = {
        "schema_version": "dermpact-phase2-4a-localized-criterion-comparison-v1",
        "protocol_sha256": protocol_sha,
        "direction": "localized residual minus retained base from the same wrapper passes",
        "n_cases": len(base_rows),
        "per_criterion_auroc": per_criterion,
        "paired_comparisons": comparisons,
        "calibration": {"base": _calibration(base_rows), "localized": _calibration(final_rows)},
        "promotion_checks": {"primary_strictly_positive": primary["estimate"] > 0.0, "primary_ci_lower_positive": primary["ci_low"] > 0.0, "protected_falls_within_0p02": protected_ok, "disagreement_increase_within_0p02": disagreement_ok},
        "decision": "PROMOTE" if promote else "REJECT",
        "official_test_rows_loaded": 0,
    }
    comparison_sha = _atomic_json(COMPARISON_PATH, comparison)
    receipt = {
        "schema_version": "dermpact-phase2-4a-localized-criterion-final-receipt-v1",
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "scientific_decision": comparison["decision"],
        "comparison": {"path": _relative(COMPARISON_PATH), "sha256": comparison_sha},
        "canonical_folds": fold_artifacts,
        "environment": _canonical_environment(),
        "hard_scope_checks": dict(protocol["hard_scope_checks"]),
    }
    digest = _atomic_json(FINAL_RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": digest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-protocol", action="store_true")
    mode.add_argument("--train-fold", type=int, choices=FOLDS)
    mode.add_argument("--reevaluate-fold", type=int, choices=FOLDS)
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--microbatch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.freeze_protocol:
        payload = build_protocol()
    elif args.train_fold is not None:
        payload = train_fold(args.train_fold, device=torch.device("cpu"), microbatch_size=args.microbatch_size)
    elif args.reevaluate_fold is not None:
        payload = canonical_reevaluate_fold(args.reevaluate_fold)
    elif args.finalize:
        payload = finalize()
    else:
        payload = _load_json(STATUS_PATH) if STATUS_PATH.is_file() else {"status": "NOT_STARTED", "path": _relative(STATUS_PATH)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
