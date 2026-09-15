"""Phase 2.5 Stage 1: frozen backbone, train new modules."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from dermpact.perception.multiclass_criterion_module import (
    MulticlassCriterionModule,
)
from dermpact.perception.unified_perception_tail import (
    UnifiedPerceptionTail,
)

from . import dermpact_phase2_4_localized_training as _base
from . import dermpact_phase2_4a_attention_factor as _af
from . import dermpact_phase2_4b_experiment as phase24b
from . import dermpact_phase2_4b_training as training
from . import dermpact_phase2_baselines as baselines
from . import dermpact_phase2_5_window_cache as window_cache
from .dermpact_phase2_5_model import Phase25Model
from .dermpact_phase2_baselines import CRITERIA, RAW_ALLOWED, POSITIVE_VALUES
from .dermpact_phase2_vector_bootstrap import binary_auc_samples, interval


ROOT = _base.ROOT
SEED = 20260901
FOLDS = tuple(range(5))
STAGE1_EPOCHS = 3
EFFECTIVE_BATCH = 16
MICROBATCH = 2
EVAL_BATCH = 2
GRADIENT_CLIP_NORM = 1.0
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
GD_DDW_LAMBDA = 0.5
BINARY_CRIT_LAMBDA = 0.5
DIAG_LAMBDA = 1.0
MEL_LAMBDA = 1.0
DISTILL_LAMBDA = 0.1
MAX_LOGIT_SHIFT = 2.0
TASK_FEATURE_DIM = 128
N_ATTENTION_HEADS = 4
N_BOOT = 2000
BOOTSTRAP_SEED = 20260901
DEFAULT_OUTPUT_DIR = ROOT / "research/experiments/dermpact_phase2_5/stage1_cefix_v1"
RAW_CE_NORMALIZATION_VERSION = "logical-weight-sum-cefix-v1"

RAW_CLASS_MAPS: dict[str, dict[str, int]] = {}
RAW_N_CLASSES: dict[str, int] = {}
RAW_POSITIVE_INDICES: dict[str, list[int]] = {}

for _c in CRITERIA:
    _classes = sorted(RAW_ALLOWED[_c])
    RAW_CLASS_MAPS[_c] = {label: idx for idx, label in enumerate(_classes)}
    RAW_N_CLASSES[_c] = len(_classes)
    RAW_POSITIVE_INDICES[_c] = [
        RAW_CLASS_MAPS[_c][l] for l in sorted(POSITIVE_VALUES[_c])
    ]


STAGE1_SOURCE_FILES = (
    Path(__file__).resolve(),
    (ROOT / "dermpact/eval/dermpact_phase2_5_model.py").resolve(),
    (ROOT / "dermpact/eval/dermpact_phase2_5_window_cache.py").resolve(),
    (ROOT / "dermpact/perception/unified_perception_tail.py").resolve(),
    (ROOT / "dermpact/perception/multiclass_criterion_module.py").resolve(),
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _stage1_source_bindings() -> list[dict[str, Any]]:
    return window_cache.source_bindings(STAGE1_SOURCE_FILES)


def _stage1_protocol(
    *,
    phase24b_protocol_sha: str,
    clinical_cache: training.ClinicalCache,
    window_manifest: Mapping[str, Any],
    paired_data: training.FoldTrainingData,
    source_bindings: Sequence[Mapping[str, Any]],
    backend: str,
    microbatch: int,
    execution_threads: int,
) -> dict[str, Any]:
    source_payload = {"bindings": list(source_bindings)}
    return {
        "schema_version": "dermpact-phase2-5-stage1-protocol-v3",
        "phase24b_protocol_sha256": phase24b_protocol_sha,
        "clinical_cache": clinical_cache.identity(),
        "window_cache": {
            "schema_version": window_manifest.get("schema_version"),
            "manifest_sha256": _sha256(
                window_cache.DEFAULT_OUTPUT_DIR / "window_cache_manifest.json"
            ),
            "source_bundle_sha256": window_manifest.get("source", {}).get("bundle_sha256"),
            "group_id_order_sha256": window_manifest.get("provenance", {}).get("group_id_order_sha256"),
        },
        "data": {
            "n_cases": int(len(paired_data.folds)),
            "folds_sha256": hashlib.sha256(
                np.ascontiguousarray(paired_data.folds, dtype=np.int64).tobytes()
            ).hexdigest(),
            "official_test_rows_loaded": 0,
        },
        "execution": {
            "seed": SEED,
            "folds": list(FOLDS),
            "epochs": STAGE1_EPOCHS,
            "effective_batch": EFFECTIVE_BATCH,
            "microbatch": microbatch,
            "backend": backend,
            "execution_threads": execution_threads,
            "evaluation_batch": EVAL_BATCH,
            "optimizer": "AdamW",
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "losses": {
            "raw_state": "fold-balanced categorical cross-entropy",
            "raw_ce_normalization_version": RAW_CE_NORMALIZATION_VERSION,
            "binary_criterion_focal": BINARY_CRIT_LAMBDA,
            "diagnosis_cross_entropy": DIAG_LAMBDA,
            "melanoma_focal": MEL_LAMBDA,
            "checklist_melanoma_focal": GD_DDW_LAMBDA,
            "baseline_distillation": DISTILL_LAMBDA,
        },
        "sources": source_payload,
        "sources_sha256": _canonical_hash(source_payload),
    }


def _freeze_stage1_protocol(path: Path, protocol: Mapping[str, Any]) -> str:
    path = Path(path)
    if path.is_file():
        _verify_sidecar(path)
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise AssertionError("frozen Stage 1 protocol differs from current execution")
        return _sha256(path)
    return _atomic_json(path, protocol)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_training_metadata_root(root: Path | None) -> dict[str, Any] | None:
    """Bind readers to an explicitly sanitized training-only metadata bundle."""
    if root is None:
        return None
    root = Path(root).resolve()
    paths = {
        "derm7pt_inventory": root / "research/manifests/dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
        "ham_manifest": root / "research/manifests/ham10000_train_calibration_v2.json",
        "ham_role_ledger": root / "research/manifests/dermpact_phase1_ham_roles_v1.json",
        "fold_manifest": root / "research/manifests/dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json",
        "clinical_photo_audit": root / "research/manifests/dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json",
        "structured_fact_support": root / "research/manifests/dermpact_phase2_3/structured_facts_v1/structured_fact_support.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training-only metadata bundle is incomplete: {missing}")
    derm = json.loads(paths["derm7pt_inventory"].read_text(encoding="utf-8"))
    ham = json.loads(paths["ham_manifest"].read_text(encoding="utf-8"))
    folds = json.loads(paths["fold_manifest"].read_text(encoding="utf-8"))
    derm_rows = derm.get("records", [])
    ham_rows = ham.get("records", [])
    if (
        len(derm_rows) != 616
        or any(row.get("phase1_role") != "development" for row in derm_rows)
        or derm.get("portable_redaction", {}).get("official_test_rows_included") != 0
    ):
        raise AssertionError("Derm7pt metadata is not the 616-row development-only bundle")
    if (
        len(ham_rows) != 7010
        or any(row.get("split") != "train" for row in ham_rows)
        or not ham.get("portable_redaction", {}).get("training_rows_only")
    ):
        raise AssertionError("HAM metadata is not the training-only portable bundle")
    if len(folds.get("derm7pt", {}).get("folds", [])) != 616 or len(
        folds.get("ham", {}).get("folds", [])
    ) != 5228:
        raise AssertionError("portable fold manifest has an unexpected scope")
    baselines.DERM7PT_INVENTORY = paths["derm7pt_inventory"]
    baselines.HAM_MANIFEST = paths["ham_manifest"]
    baselines.HAM_ROLE_LEDGER = paths["ham_role_ledger"]
    _base.screen.FOLD_MANIFEST = paths["fold_manifest"]
    return {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with tmp.open("r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    digest = _sha256(path)
    sidecar_tmp = Path(str(path) + ".sha256.tmp")
    sidecar_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(sidecar_tmp, Path(str(path) + ".sha256"))
    return digest


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _atomic_torch(path: Path, state: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    with tmp.open("wb") as handle:
        torch.save(_cpu_tree(dict(state)), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = _sha256(path)
    sidecar_tmp = Path(str(path) + ".sha256.tmp")
    sidecar_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(sidecar_tmp, Path(str(path) + ".sha256"))
    return digest


def _verify_sidecar(path: Path) -> str:
    path = Path(path)
    digest = _sha256(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise AssertionError(f"SHA-256 sidecar mismatch: {path}")
    return digest


def _atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = _sha256(path)
    sidecar_tmp = Path(str(path) + ".sha256.tmp")
    sidecar_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(sidecar_tmp, Path(str(path) + ".sha256"))
    return digest


def _get_device():
    """Select the verified RX 5700 XT and never fall back silently."""
    try:
        import torch_directml
    except ImportError as error:
        raise RuntimeError("torch-directml is required for local Stage 1") from error
    names = [
        torch_directml.device_name(index).rstrip("\x00")
        for index in range(torch_directml.device_count())
    ]
    expected = "AMD Radeon RX 5700 XT"
    matches = [index for index, name in enumerate(names) if name == expected]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {expected!r} DirectML adapter; observed {names!r}"
        )
    index = matches[0]
    return torch_directml.device(index), names[index]


def _resolve_device(backend: str) -> tuple[Any, str, str]:
    normalized = backend.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "directml"}:
        raise ValueError("backend must be auto, cpu, cuda, or directml")
    if normalized == "cpu":
        return torch.device("cpu"), "CPU", "cpu"
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda:0"), torch.cuda.get_device_name(0), "cuda"
    if normalized == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    device, name = _get_device()
    return device, name, "directml"


def _configure_execution_threads(*, backend: str, cpu_threads: int | None) -> int:
    """Configure CPU execution only when the caller explicitly selected it."""
    if cpu_threads is not None and cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    if backend != "cpu":
        if cpu_threads is not None:
            raise ValueError("cpu_threads may be specified only with backend='cpu'")
        return int(torch.get_num_threads())
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    threads = int(torch.get_num_threads())
    if threads <= 0:
        raise RuntimeError("PyTorch reported a non-positive CPU thread count")
    return threads


def _focal_bce(logits: Tensor, targets: Tensor, *, alpha: float, gamma: float) -> Tensor:
    probs = torch.sigmoid(logits)
    ce = _bce_with_logits_native(logits, targets, reduction="none")
    pt = torch.where(targets == 1.0, probs, 1.0 - probs)
    at = torch.where(targets == 1.0, alpha, 1.0 - alpha)
    return (at * (1.0 - pt).pow(gamma) * ce).mean()


def _bce_with_logits_native(
    logits: Tensor,
    targets: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Stable BCE-with-logits without DirectML's log-sigmoid fallback."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    loss = F.softplus(logits) - logits * targets
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be none, sum, or mean")


def _load_raw_targets(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        c: np.array([RAW_CLASS_MAPS[c][r["criteria_raw"][c]] for r in rows], dtype=np.int64)
        for c in CRITERIA
    }


def _fold_class_weights(
    raw_targets: dict[str, np.ndarray], mask: np.ndarray,
) -> dict[str, Tensor]:
    weights = {}
    for c in CRITERIA:
        targets = raw_targets[c][mask]
        n_cls = RAW_N_CLASSES[c]
        counts = np.maximum(np.bincount(targets, minlength=n_cls).astype(np.float64), 1.0)
        w = np.minimum(len(targets) / (n_cls * counts), 10.0)
        weights[c] = torch.from_numpy(w.astype(np.float32))
    return weights


def _forward_batch(
    model: Phase25Model,
    paired_data: training.FoldTrainingData,
    win_tokens: np.ndarray,
    win_mask: np.ndarray,
    win_pos: Tensor,
    indices: np.ndarray,
    device: Any,
    clinical_present_override: np.ndarray | None = None,
    frozen_cache: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], Tensor]:
    g = torch.from_numpy(
        np.asarray(paired_data.derm_tokens[indices], dtype=np.float32).copy()
    ).to(device)
    w = torch.from_numpy(
        np.asarray(win_tokens[indices], dtype=np.float32).copy()
    ).to(device)
    m = torch.from_numpy(np.asarray(win_mask[indices], dtype=bool).copy()).to(device)
    c = torch.from_numpy(
        np.asarray(paired_data.clinical_tokens[indices], dtype=np.float32).copy()
    ).to(device)
    available = np.asarray(paired_data.clinical_available[indices], dtype=bool)
    if clinical_present_override is not None:
        override = np.asarray(clinical_present_override, dtype=bool)
        if override.shape != available.shape:
            raise ValueError("clinical_present_override must match the selected batch")
        if np.any(override & ~available):
            raise ValueError("clinical override cannot invent an unavailable photograph")
        available = override
    cp = torch.from_numpy(available.copy()).to(device)
    frozen_outputs = None
    if frozen_cache is not None:
        true_cache = frozen_cache["true"]
        missing_cache = frozen_cache["missing"]
        criteria = np.where(
            available[:, None],
            np.asarray(true_cache["criteria"])[indices],
            np.asarray(missing_cache["criteria"])[indices],
        )
        frozen_outputs = {
            "diagnosis": torch.from_numpy(
                np.asarray(true_cache["diagnosis"])[indices].astype(np.float32).copy()
            ).to(device),
            "melanoma": torch.from_numpy(
                np.asarray(true_cache["melanoma"])[indices].astype(np.float32).copy()
            ).to(device),
            "criteria": torch.from_numpy(
                criteria.astype(np.float32).copy()
            ).to(device),
        }
    return model(
        g, w, c, m, win_pos.to(device), cp, frozen_outputs=frozen_outputs
    ), g


def _precompute_frozen_outputs(
    model: Phase25Model,
    paired_data: training.FoldTrainingData,
    *,
    batch_size: int = 64,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute true-pair and forced-missing baseline logits once on CPU."""
    model.frozen_baseline.to("cpu")
    model.frozen_baseline.eval()
    n = len(paired_data.folds)
    n_diagnosis = int(model.diagnosis_delta_head.out_features)
    result = {
        mode: {
            "diagnosis": np.empty((n, n_diagnosis), dtype=np.float32),
            "melanoma": np.empty(n, dtype=np.float32),
            "criteria": np.empty((n, len(CRITERIA)), dtype=np.float32),
        }
        for mode in ("true", "missing")
    }
    with torch.no_grad():
        for start in range(0, n, batch_size):
            indices = np.arange(start, min(start + batch_size, n))
            derm = torch.from_numpy(
                np.asarray(paired_data.derm_tokens[indices], dtype=np.float32).copy()
            )
            clinical = torch.from_numpy(
                np.asarray(paired_data.clinical_tokens[indices], dtype=np.float32).copy()
            )
            true_present = torch.from_numpy(
                np.asarray(paired_data.clinical_available[indices], dtype=bool).copy()
            )
            for mode, present in (
                ("true", true_present),
                ("missing", torch.zeros_like(true_present)),
            ):
                output = model.frozen_baseline(derm, clinical, present)
                result[mode]["diagnosis"][indices] = output["diagnosis"].numpy()
                result[mode]["melanoma"][indices] = output["melanoma"].numpy()
                result[mode]["criteria"][indices] = output["criteria"].numpy()
    for values in result.values():
        if not all(np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError("precomputed frozen outputs are non-finite")
    return result


def _multitask_loss(
    output: Mapping[str, Any],
    indices: np.ndarray,
    raw_targets: Mapping[str, np.ndarray],
    class_weights: Mapping[str, Tensor],
    criterion_targets: np.ndarray,
    derm_data: Mapping[str, Any],
    device: Any,
    class_weight_denominators: Mapping[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    anchor = output["criteria"]
    raw_ce_total = anchor.new_zeros(())
    for criterion in CRITERIA:
        target = torch.from_numpy(raw_targets[criterion][indices].copy()).to(device)
        criterion_loss = _logical_weighted_cross_entropy(
            output["raw_logits"][criterion],
            target,
            class_weights[criterion],
            None if class_weight_denominators is None
            else float(class_weight_denominators[criterion]),
        )
        raw_ce_total = raw_ce_total + criterion_loss
    raw_ce = raw_ce_total / len(CRITERIA)

    binary_target = torch.from_numpy(
        criterion_targets[indices].astype(np.float32).copy()
    ).to(device)
    criterion_focal = _focal_bce(
        output["criteria"], binary_target, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA
    )
    diagnosis_target = torch.from_numpy(
        np.asarray(derm_data["derm_global"])[indices].astype(np.int64).copy()
    ).to(device)
    diagnosis_loss = F.cross_entropy(output["diagnosis"], diagnosis_target)
    melanoma_target = torch.from_numpy(
        np.asarray(derm_data["derm_melanoma"])[indices].astype(np.float32).copy()
    ).to(device)
    melanoma_loss = _focal_bce(
        output["melanoma"], melanoma_target, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA
    )
    checklist_loss = _focal_bce(
        output["checklist_melanoma"],
        melanoma_target,
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
    )
    with torch.no_grad():
        frozen_probability = torch.sigmoid(output["frozen_criteria"])
    distillation = _bce_with_logits_native(
        output["criteria"], frozen_probability, reduction="mean"
    )

    parts = {
        "raw_ce": raw_ce,
        "criterion_focal": criterion_focal,
        "diagnosis": diagnosis_loss,
        "melanoma": melanoma_loss,
        "checklist_melanoma": checklist_loss,
        "distillation": distillation,
    }
    total = (
        raw_ce
        + BINARY_CRIT_LAMBDA * criterion_focal
        + DIAG_LAMBDA * diagnosis_loss
        + MEL_LAMBDA * melanoma_loss
        + GD_DDW_LAMBDA * checklist_loss
        + DISTILL_LAMBDA * distillation
    )
    return total, parts


def _logical_weighted_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor,
    logical_denominator: float | None,
) -> Tensor:
    """Weighted CE whose semantics survive microbatch accumulation."""
    if logical_denominator is None:
        return F.cross_entropy(logits, targets, weight=class_weights)
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    sample_weight = class_weights[targets]
    return (per_sample * sample_weight).sum() / max(logical_denominator, 1e-12)


def _scale_microbatch_loss(total_loss: Tensor, raw_ce: Tensor, fraction: float) -> Tensor:
    """Accumulate mean loss parts while retaining logical-batch raw CE weight."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    return total_loss * fraction + raw_ce * (1.0 - fraction)


def _class_weight_denominators(
    indices: np.ndarray,
    raw_targets: Mapping[str, np.ndarray],
    class_weights: Mapping[str, Tensor],
) -> dict[str, float]:
    """Logical-batch denominators that preserve weighting under accumulation."""
    result = {}
    for criterion in CRITERIA:
        targets = torch.from_numpy(raw_targets[criterion][indices].copy()).long()
        weights = class_weights[criterion].detach().cpu()
        result[criterion] = float(weights[targets].sum())
    return result


def _build_phase25(
    fold: int,
    paired_data: training.FoldTrainingData,
) -> Phase25Model:
    """Construct Phase 2.5 model for one fold."""
    protocol_raw, _ = phase24b.load_protocol()
    bindings, cache = phase24b._live_inputs(protocol_raw)
    frozen = training.construct_paired_model(bindings[fold])
    rec = phase24b._verified_json(phase24b.training_receipt_path(fold))
    state = torch.load(rec[0]["checkpoint"]["path"], map_location="cpu", weights_only=True)
    training.load_mutable_clinical_state(frozen, state)
    frozen.eval()

    baseline_tail = frozen.baseline_model.confirmed_model.base_model.visual_tail
    tail = UnifiedPerceptionTail(
        tail_blocks=baseline_tail.blocks,
        tail_norm=baseline_tail.norm,
        tail_projection=baseline_tail.projection,
        shared_rel_pos_bias=baseline_tail.shared_rel_pos_bias,
    )

    torch.manual_seed(SEED + fold * 1000)
    mc = MulticlassCriterionModule(
        token_dim=768,
        criterion_class_counts=RAW_N_CLASSES,
        criterion_positive_indices=RAW_POSITIVE_INDICES,
        criterion_names=list(CRITERIA),
        task_feature_dim=TASK_FEATURE_DIM,
        n_attention_heads=N_ATTENTION_HEADS,
        max_logit_shift=MAX_LOGIT_SHIFT,
    )

    diagnosis_head = frozen.baseline_model.confirmed_model.base_model.task_head
    n_diag = int(diagnosis_head.diagnosis.out_features)

    return Phase25Model(
        frozen_baseline=frozen,
        unified_tail=tail,
        multiclass_module=mc,
        n_diagnosis_classes=n_diag,
        max_logit_shift=MAX_LOGIT_SHIFT,
    )


def _gpu_smoke_test(
    model: Phase25Model, paired_data: training.FoldTrainingData,
    win_tokens: np.ndarray, win_mask: np.ndarray, win_pos: Tensor,
    raw_targets: Mapping[str, np.ndarray],
    class_weights: Mapping[str, Tensor],
    derm_data: Mapping[str, Any],
    device: Any,
    *,
    output_dir: Path,
    microbatch: int = MICROBATCH,
    stage2: bool = False,
    frozen_cache: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Fail-closed real-model accelerator smoke."""
    if microbatch <= 0:
        raise ValueError("microbatch must be positive")
    if stage2:
        model.unified_tail.unfreeze_tail()
    if frozen_cache is None:
        frozen_cache = _precompute_frozen_outputs(model, paired_data)
    model.to(device)
    model.frozen_baseline.to("cpu")
    model.train()

    available = np.flatnonzero(paired_data.clinical_available)
    missing = np.flatnonzero(~paired_data.clinical_available)
    selected = [int(available[0])]
    if microbatch > 1 and len(missing):
        selected.append(int(missing[0]))
    for value in available[1:]:
        if len(selected) >= microbatch:
            break
        selected.append(int(value))
    idx = np.asarray(selected, dtype=np.int64)

    batch_started = time.perf_counter()
    out, _ = _forward_batch(
        model, paired_data, win_tokens, win_mask, win_pos, idx, device,
        frozen_cache=frozen_cache,
    )
    forward_seconds = time.perf_counter() - batch_started

    # Verify exact baseline preservation at init
    frozen_crit = out["frozen_criteria"]
    new_crit = out["criteria"]
    crit_shift = float((new_crit - frozen_crit).abs().max().detach().cpu())
    if crit_shift > 1e-5:
        raise RuntimeError(
            f"GPU smoke: criterion shift {crit_shift:.6f} at init — not base-preserving"
        )

    frozen_mel = out["frozen_melanoma"]
    new_mel = out["melanoma"]
    mel_shift = float((new_mel - frozen_mel).abs().max().detach().cpu())
    if mel_shift > 1e-5:
        raise RuntimeError(
            f"GPU smoke: melanoma shift {mel_shift:.6f} at init — not base-preserving"
        )

    diag_shift = float(
        (out["diagnosis"] - out["frozen_diagnosis"]).abs().max().detach().cpu()
    )
    if diag_shift > 1e-5:
        raise RuntimeError(
            f"GPU smoke: diagnosis shift {diag_shift:.6f} at init — not base-preserving"
        )

    device_weights = {key: value.to(device) for key, value in class_weights.items()}
    class_weight_denominators = _class_weight_denominators(
        idx, raw_targets, class_weights
    )
    loss, loss_parts = _multitask_loss(
        out,
        idx,
        raw_targets,
        device_weights,
        paired_data.criterion_targets,
        derm_data,
        device,
        class_weight_denominators,
    )
    parameter_groups = [group for group in model.trainable_parameter_groups() if group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    backward_started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - backward_started

    trainable = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in trainable if p.grad is not None]
    if not grads:
        raise RuntimeError("GPU smoke: no gradients produced")
    if any(not torch.isfinite(g).all() for g in grads):
        raise RuntimeError("GPU smoke: non-finite gradients")
    if any(p.grad is not None for p in model.frozen_baseline.parameters()):
        raise RuntimeError("GPU smoke: gradient leaked to frozen baseline")
    named_trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    gradient_names = {
        name for name, parameter in named_trainable.items() if parameter.grad is not None
    }
    required_prefixes = [
        "unified_tail.region_transformer.",
        "unified_tail.cross_attn_",
        "multiclass_module.",
        "diagnosis_delta_head.",
        "melanoma_delta_head.",
    ]
    if stage2:
        required_prefixes.append("unified_tail.blocks.")
    missing_gradients = [
        prefix for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in gradient_names)
    ]
    if missing_gradients:
        raise RuntimeError(f"GPU smoke: branches without gradients: {missing_gradients}")

    torch.nn.utils.clip_grad_norm_(list(named_trainable.values()), GRADIENT_CLIP_NORM)
    started = time.perf_counter()
    optimizer.step()
    step_seconds = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        after_step, _ = _forward_batch(
            model, paired_data, win_tokens, win_mask, win_pos, idx, device,
            frozen_cache=frozen_cache,
        )
        expected_criteria = after_step["criteria"].detach().cpu()
        expected_melanoma = after_step["melanoma"].detach().cpu()

    state_dict = model.state_dict()
    trainable_state = {
        name: state_dict[name].detach().cpu()
        for name in named_trainable
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = output_dir / ("stage2_directml_smoke.pt" if stage2 else "stage1_directml_smoke.pt")
    smoke_sha = _atomic_torch(
        smoke_path,
        {
            "stage": 2 if stage2 else 1,
            "microbatch": int(len(idx)),
            "trainable_state": trainable_state,
            "optimizer_state": optimizer.state_dict(),
        },
    )
    loaded = torch.load(smoke_path, map_location="cpu", weights_only=False)
    # Perturb one parameter so the reload check cannot pass vacuously.
    first_parameter = next(iter(named_trainable.values()))
    with torch.no_grad():
        first_parameter.zero_()
    model.load_state_dict(
        {name: value.to(device) for name, value in loaded["trainable_state"].items()},
        strict=False,
    )
    with torch.no_grad():
        reloaded, _ = _forward_batch(
            model, paired_data, win_tokens, win_mask, win_pos, idx, device,
            frozen_cache=frozen_cache,
        )
    reload_difference = max(
        float((reloaded["criteria"].cpu() - expected_criteria).abs().max()),
        float((reloaded["melanoma"].cpu() - expected_melanoma).abs().max()),
    )
    if reload_difference > 1e-5:
        raise RuntimeError(
            f"GPU smoke: save/reload prediction drift {reload_difference:.6g}"
        )

    model.zero_grad(set_to_none=True)
    return {
        "status": "PASS",
        "stage": 2 if stage2 else 1,
        "microbatch": int(len(idx)),
        "criterion_init_shift": crit_shift,
        "melanoma_init_shift": mel_shift,
        "diagnosis_init_shift": diag_shift,
        "n_trainable": sum(p.numel() for p in trainable),
        "n_grads": len(grads),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "step_seconds": step_seconds,
        "train_step_seconds": forward_seconds + backward_seconds + step_seconds,
        "reload_max_abs_difference": reload_difference,
        "loss": float(loss.detach().cpu()),
        "loss_parts": {
            key: float(value.detach().cpu()) for key, value in loss_parts.items()
        },
        "checkpoint": {"path": str(smoke_path), "sha256": smoke_sha},
    }


def _eval_on_indices(
    model: nn.Module,
    paired_data: training.FoldTrainingData,
    win_tokens: np.ndarray,
    win_mask: np.ndarray,
    win_pos: Tensor,
    indices: np.ndarray,
    device: Any,
    frozen_cache: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return binary, diagnosis, and raw-state logits for given indices."""
    n = len(indices)
    crit_logits = np.empty((n, len(CRITERIA)), dtype=np.float64)
    mel_logits = np.empty(n, dtype=np.float64)
    diagnosis_batches: list[np.ndarray] = []
    raw_batches: dict[str, list[np.ndarray]] = {criterion: [] for criterion in CRITERIA}

    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH):
            sel = indices[start:start + EVAL_BATCH]
            k = len(sel)
            out, _ = _forward_batch(
                model, paired_data, win_tokens, win_mask, win_pos, sel, device,
                frozen_cache=frozen_cache,
            )
            crit_logits[start:start + k] = out["criteria"].cpu().numpy()
            mel_logits[start:start + k] = out["melanoma"].cpu().numpy()
            diagnosis_batches.append(out["diagnosis"].cpu().numpy())
            for criterion in CRITERIA:
                raw_batches[criterion].append(
                    out["raw_logits"][criterion].cpu().numpy()
                )

    return (
        crit_logits,
        mel_logits,
        np.concatenate(diagnosis_batches, axis=0),
        {
            criterion: np.concatenate(blocks, axis=0)
            for criterion, blocks in raw_batches.items()
        },
    )


def _compute_8out_aurocs(
    crit_logits: np.ndarray, mel_logits: np.ndarray,
    crit_targets: np.ndarray, mel_targets: np.ndarray,
) -> list[float]:
    aucs = []
    try:
        aucs.append(float(roc_auc_score(mel_targets, mel_logits)))
    except ValueError:
        aucs.append(0.5)
    for c_idx in range(len(CRITERIA)):
        try:
            aucs.append(float(roc_auc_score(crit_targets[:, c_idx], crit_logits[:, c_idx])))
        except ValueError:
            aucs.append(0.5)
    return aucs


def _trainable_state(model: nn.Module) -> dict[str, Tensor]:
    state = model.state_dict()
    return {
        name: state[name].detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])


def _load_resume(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Any,
    fold: int,
    protocol_sha256: str,
    source_bundle_sha256: str,
) -> tuple[int, list[dict[str, Any]]]:
    path = Path(path)
    if not path.is_file():
        return 0, []
    _verify_sidecar(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "dermpact-phase2-5-stage1-resume-v3":
        raise AssertionError("Stage 1 resume schema mismatch")
    expected = (fold, protocol_sha256, source_bundle_sha256)
    observed = (
        int(payload.get("fold", -1)),
        str(payload.get("protocol_sha256", "")),
        str(payload.get("source_bundle_sha256", "")),
    )
    if observed != expected:
        raise AssertionError(f"resume identity mismatch: {observed!r} != {expected!r}")
    model.load_state_dict(payload["trainable_state"], strict=False)
    optimizer.load_state_dict(payload["optimizer_state"])
    _move_optimizer_state(optimizer, device)
    _restore_rng_state(payload["rng_state"])
    completed_epoch = int(payload["completed_epoch"])
    if not 0 <= completed_epoch <= STAGE1_EPOCHS:
        raise AssertionError("resume epoch is outside the Stage 1 schedule")
    return completed_epoch, list(payload.get("history", []))


def train_fold(
    fold: int,
    paired_data: training.FoldTrainingData,
    raw_targets: dict[str, np.ndarray],
    derm_data: Mapping[str, Any],
    win_tokens: np.ndarray,
    win_mask: np.ndarray,
    win_pos: Tensor,
    device: Any,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    protocol_sha256: str,
    source_bundle_sha256: str,
    microbatch: int,
) -> dict[str, Any]:
    """Train one fixed three-epoch outer fold with hash-verified resume."""
    random.seed(SEED + fold)
    np.random.seed(SEED + fold)
    torch.manual_seed(SEED + fold)

    all_folds = paired_data.folds
    train_idx = np.flatnonzero(all_folds != fold)
    outer_val_idx = np.flatnonzero(all_folds == fold)

    print(f"  Train: {len(train_idx)}, Outer val: {len(outer_val_idx)}", flush=True)

    model = _build_phase25(fold, paired_data)
    frozen_cache = _precompute_frozen_outputs(model, paired_data)
    model.to(device)
    model.frozen_baseline.to("cpu")

    trainable = [p for p in model.parameters() if p.requires_grad]
    param_groups = model.trainable_parameter_groups()
    optimizer = torch.optim.AdamW(
        [g for g in param_groups if g["params"]],
        weight_decay=1e-4,
    )

    # Fold-level class weights for balanced CE
    train_mask = all_folds != fold
    fold_cw = _fold_class_weights(raw_targets, train_mask)
    fold_cw_dev = {k: v.to(device) for k, v in fold_cw.items()}

    win_pos_dev = win_pos.to(device)
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(train_idx) / EFFECTIVE_BATCH)
    resume_path = fold_dir / "latest_checkpoint.pt"
    start_epoch, history = _load_resume(
        resume_path,
        model=model,
        optimizer=optimizer,
        device=device,
        fold=fold,
        protocol_sha256=protocol_sha256,
        source_bundle_sha256=source_bundle_sha256,
    )
    if start_epoch:
        print(f"  Resuming after epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, STAGE1_EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + fold * 1000 + epoch)
        perm = rng.permutation(len(train_idx))
        epoch_losses = defaultdict(float)
        started = time.perf_counter()

        for step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            b_start = step * EFFECTIVE_BATCH
            b_end = min(b_start + EFFECTIVE_BATCH, len(train_idx))
            b_idx = train_idx[perm[b_start:b_end]]
            n_b = len(b_idx)
            class_weight_denominators = _class_weight_denominators(
                b_idx, raw_targets, fold_cw
            )
            logical_parts = defaultdict(float)
            for mb_start in range(0, n_b, microbatch):
                mb_idx = b_idx[mb_start:mb_start + microbatch]
                output, _ = _forward_batch(
                    model,
                    paired_data,
                    win_tokens,
                    win_mask,
                    win_pos_dev,
                    mb_idx,
                    device,
                    frozen_cache=frozen_cache,
                )
                loss, parts = _multitask_loss(
                    output,
                    mb_idx,
                    raw_targets,
                    fold_cw_dev,
                    paired_data.criterion_targets,
                    derm_data,
                    device,
                    class_weight_denominators,
                )
                scale = len(mb_idx) / n_b
                _scale_microbatch_loss(loss, parts["raw_ce"], scale).backward()
                for name, value in parts.items():
                    part_scale = 1.0 if name == "raw_ce" else scale
                    logical_parts[name] += float(value.detach().cpu()) * part_scale

            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP_NORM)
            optimizer.step()

            for name, value in logical_parts.items():
                epoch_losses[name] += value * n_b / len(train_idx)

        elapsed = time.perf_counter() - started

        gates = model.multiclass_module.residual_gates().cpu().numpy()
        history.append({
            "epoch": epoch + 1,
            "losses": dict(epoch_losses),
            "gates": gates.tolist(),
            "elapsed": elapsed,
        })

        print(
            f"  [fold{fold}] epoch {epoch+1}/{STAGE1_EPOCHS}  "
            f"raw_ce={epoch_losses['raw_ce']:.4f}  "
            f"gates=[{','.join(f'{g:.3f}' for g in gates)}]  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        # Atomic checkpoint with RNG state
        _atomic_torch(resume_path, {
            "schema_version": "dermpact-phase2-5-stage1-resume-v3",
            "fold": fold,
            "protocol_sha256": protocol_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "completed_epoch": epoch + 1,
            "trainable_state": _trainable_state(model),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
            },
            "history": history,
        })

    # Fixed schedule: the epoch-3 state is the portable Stage 2 initializer.
    final_state = _trainable_state(model)
    ckpt_sha = _atomic_torch(fold_dir / "final_stage1.pt", {
        "schema_version": "dermpact-phase2-5-stage1-final-v3",
        "fold": fold,
        "protocol_sha256": protocol_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "completed_epochs": STAGE1_EPOCHS,
        "trainable_state": final_state,
        "history": history,
    })

    # Evaluate the untouched outer fold once after the fixed schedule.
    model.eval()
    ov_crit, ov_mel, ov_diag, ov_raw = _eval_on_indices(
        model, paired_data, win_tokens, win_mask, win_pos_dev,
        outer_val_idx, device, frozen_cache=frozen_cache,
    )

    model.to("cpu")
    del model

    return {
        "fold": fold,
        "outer_val_idx": outer_val_idx,
        "outer_crit_logits": ov_crit,
        "outer_mel_logits": ov_mel,
        "outer_diag_logits": ov_diag,
        "outer_raw_logits": ov_raw,
        "checkpoint": {"path": str(fold_dir / "final_stage1.pt"), "sha256": ckpt_sha},
        "completed_epochs": STAGE1_EPOCHS,
        "n_trainable": sum(parameter.numel() for parameter in trainable),
        "history": history,
    }


def run_experiment(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    backend: str = "auto",
    microbatch: int | None = None,
    cpu_threads: int | None = None,
    training_metadata_root: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, device_name, resolved_backend = _resolve_device(backend)
    execution_threads = _configure_execution_threads(
        backend=resolved_backend, cpu_threads=cpu_threads
    )
    selected_microbatch = microbatch or (8 if resolved_backend == "cuda" else MICROBATCH)
    if selected_microbatch <= 0 or EFFECTIVE_BATCH % selected_microbatch:
        raise ValueError("microbatch must be a positive divisor of effective batch 16")
    print(f"Device: {device_name} (execution_threads={execution_threads})", flush=True)
    if resolved_backend == "cpu" and training_metadata_root is None:
        raise ValueError("CPU runs require --training-metadata-root for fail-closed scope control")
    metadata_bindings = _configure_training_metadata_root(training_metadata_root)

    # Load verified Phase 2.4B clinical cache — no fallback
    print("Loading verified Phase 2.4B paired data ...", flush=True)
    protocol_raw, protocol_sha = phase24b.load_protocol()
    bindings, cache = phase24b._live_inputs(protocol_raw)
    paired_data = training.load_training_data(cache)
    paired_data.validate()

    rows = baselines._derm7pt_records()
    raw_targets = _load_raw_targets(rows)
    derm_data = _base._derm_data()

    print("Loading window cache ...", flush=True)
    validated_window = window_cache.validate_window_cache()
    win_cache = window_cache.load_window_cache()
    win_tokens = win_cache["window_tokens"]
    win_mask = win_cache["patch_valid_mask"]
    win_pos = torch.from_numpy(win_cache["window_positions"])

    authorized = window_cache.phase24b_inputs.resolve_development_inputs(
        inventory_path=Path(metadata_bindings["derm7pt_inventory"]["path"])
        if metadata_bindings is not None
        else window_cache.phase24b_inputs.DERM7PT_INVENTORY,
        fold_manifest_path=Path(metadata_bindings["fold_manifest"]["path"])
        if metadata_bindings is not None
        else window_cache.phase24b_inputs.DEVELOPMENT_FOLDS,
        clinical_photo_audit_path=Path(metadata_bindings["clinical_photo_audit"]["path"])
        if metadata_bindings is not None
        else window_cache.phase24b_inputs.CLINICAL_PHOTO_AUDIT,
        structured_fact_support_path=Path(metadata_bindings["structured_fact_support"]["path"])
        if metadata_bindings is not None
        else window_cache.phase24b_inputs.STRUCTURED_FACT_SUPPORT,
        require_files=False
    )
    authorized_groups = [case.group_id for case in authorized.cases]
    row_groups = [str(row["group_id"]) for row in rows]
    if row_groups != authorized_groups:
        raise AssertionError("Phase 2.5 label/cache row order changed")
    group_hash = hashlib.sha256(
        json.dumps(authorized_groups, sort_keys=False).encode("utf-8")
    ).hexdigest()
    window_manifest = validated_window["manifest"]
    if window_manifest.get("provenance", {}).get("group_id_order_sha256") != group_hash:
        raise AssertionError("window-cache group order is not aligned to Phase 2.4B")

    frozen_sources = _stage1_source_bindings()
    stage_protocol = _stage1_protocol(
        phase24b_protocol_sha=protocol_sha,
        clinical_cache=cache,
        window_manifest=window_manifest,
        paired_data=paired_data,
        source_bindings=frozen_sources,
        backend=resolved_backend,
        microbatch=selected_microbatch,
        execution_threads=execution_threads,
    )
    stage_protocol["training_metadata_bindings"] = metadata_bindings
    stage_protocol_path = output_dir / "stage1_protocol.json"
    source_bundle_sha = str(stage_protocol["sources_sha256"])
    window_cache.assert_source_bindings(frozen_sources)

    # One real fold-0 Stage 1 smoke covers every trainable branch and loss.
    fold0_train_mask = paired_data.folds != 0
    smoke_weights = _fold_class_weights(raw_targets, fold0_train_mask)
    smoke_model = _build_phase25(0, paired_data)
    smoke = _gpu_smoke_test(
        smoke_model,
        paired_data,
        win_tokens,
        win_mask,
        win_pos,
        raw_targets,
        smoke_weights,
        derm_data,
        device,
        output_dir=output_dir / "smoke",
        microbatch=selected_microbatch,
        stage2=False,
    )
    print(
        f"Stage 1 smoke PASS: microbatch={smoke['microbatch']} "
        f"step={smoke['step_seconds']:.3f}s",
        flush=True,
    )
    smoke_model.to("cpu")
    del smoke_model

    stage_protocol_sha = _freeze_stage1_protocol(stage_protocol_path, stage_protocol)

    # Pooled OOF arrays
    n = len(paired_data.folds)
    n_diag = int(np.max(derm_data["derm_global"])) + 1
    oof_crit = np.full((n, len(CRITERIA)), np.nan, dtype=np.float64)
    oof_mel = np.full(n, np.nan, dtype=np.float64)
    oof_diag = np.full((n, n_diag), np.nan, dtype=np.float64)
    oof_raw = {
        criterion: np.full((n, RAW_N_CLASSES[criterion]), np.nan, dtype=np.float64)
        for criterion in CRITERIA
    }
    fold_results = []

    for fold in FOLDS:
        print(f"\n=== Fold {fold} ===", flush=True)
        result = train_fold(
            fold, paired_data, raw_targets, derm_data,
            win_tokens, win_mask, win_pos, device,
            output_dir=output_dir,
            protocol_sha256=stage_protocol_sha,
            source_bundle_sha256=source_bundle_sha,
            microbatch=selected_microbatch,
        )
        idx = result["outer_val_idx"]
        oof_crit[idx] = result["outer_crit_logits"]
        oof_mel[idx] = result["outer_mel_logits"]
        oof_diag[idx] = result["outer_diag_logits"]
        for criterion in CRITERIA:
            oof_raw[criterion][idx] = result["outer_raw_logits"][criterion]
        fold_results.append({
            k: v for k, v in result.items()
            if k not in (
                "outer_val_idx", "outer_crit_logits", "outer_mel_logits",
                "outer_diag_logits", "history",
                "outer_raw_logits",
            )
        })

    if not all(
        np.isfinite(value).all()
        for value in (oof_crit, oof_mel, oof_diag, *oof_raw.values())
    ):
        raise FloatingPointError("pooled OOF predictions are incomplete or non-finite")

    # Full pooled OOF evaluation
    print("\n=== Pooled OOF evaluation ===", flush=True)
    crit_targets = paired_data.criterion_targets
    mel_targets = derm_data["derm_melanoma"]

    # Base OOF from frozen Phase 2.4B
    base_crit = np.empty_like(oof_crit)
    base_mel = np.empty_like(oof_mel)
    base_diag = np.empty_like(oof_diag)
    for fold in FOLDS:
        val_idx = np.flatnonzero(paired_data.folds == fold)
        model = _build_phase25(fold, paired_data)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(val_idx), EVAL_BATCH):
                sel = val_idx[start:start + EVAL_BATCH]
                g = torch.from_numpy(np.asarray(paired_data.derm_tokens[sel], dtype=np.float32).copy())
                c = torch.from_numpy(np.asarray(paired_data.clinical_tokens[sel], dtype=np.float32).copy())
                cp = torch.from_numpy(paired_data.clinical_available[sel].copy())
                out = model.frozen_baseline(g, c, cp)
                base_crit[sel] = out["criteria"].numpy()
                base_mel[sel] = out["melanoma"].numpy()
                base_diag[sel] = out["diagnosis"].numpy()
        del model

    # Bootstrap comparison
    from . import dermpact_phase2_4b_evaluation as p24b_eval
    diag_targets = np.asarray([
        baselines._derm7pt_records()[i]["diagnosis"]["harmonized5"] for i in range(n)
    ])
    weights, _ = p24b_eval.fixed_bootstrap_weights(
        diag_targets, crit_targets, paired_data.clinical_available,
        n_boot=N_BOOT, seed=BOOTSTRAP_SEED,
    )

    base_aucs = _compute_8out_aurocs(base_crit, base_mel, crit_targets.astype(np.int8), mel_targets.astype(np.int8))
    new_aucs = _compute_8out_aurocs(oof_crit, oof_mel, crit_targets.astype(np.int8), mel_targets.astype(np.int8))
    base_primary = float(np.mean(base_aucs))
    new_primary = float(np.mean(new_aucs))

    base_samples = []
    new_samples = []
    all_targets = [mel_targets.astype(np.int8)] + [
        crit_targets[:, index].astype(np.int8) for index in range(len(CRITERIA))
    ]
    base_scores = [base_mel] + [base_crit[:, index] for index in range(len(CRITERIA))]
    new_scores = [oof_mel] + [oof_crit[:, index] for index in range(len(CRITERIA))]
    for target, baseline_score, candidate_score in zip(
        all_targets, base_scores, new_scores, strict=True
    ):
        base_samples.append(binary_auc_samples(target, baseline_score, weights))
        new_samples.append(binary_auc_samples(target, candidate_score, weights))
    base_sample_matrix = np.stack(base_samples, axis=1)
    new_sample_matrix = np.stack(new_samples, axis=1)
    eight_delta_samples = new_sample_matrix.mean(axis=1) - base_sample_matrix.mean(axis=1)
    criterion_delta_samples = (
        new_sample_matrix[:, 1:].mean(axis=1) - base_sample_matrix[:, 1:].mean(axis=1)
    )
    melanoma_delta_samples = new_sample_matrix[:, 0] - base_sample_matrix[:, 0]
    paired_intervals = {
        "eight_output_mean": interval(
            eight_delta_samples, new_primary - base_primary, n_boot=N_BOOT
        ),
        "criterion_mean": interval(
            criterion_delta_samples,
            float(np.mean(new_aucs[1:]) - np.mean(base_aucs[1:])),
            n_boot=N_BOOT,
        ),
        "melanoma": interval(
            melanoma_delta_samples, new_aucs[0] - base_aucs[0], n_boot=N_BOOT
        ),
    }

    diagnosis_truth = np.asarray(derm_data["derm_global"], dtype=np.int64)
    base_diagnosis = np.argmax(base_diag, axis=1)
    new_diagnosis = np.argmax(oof_diag, axis=1)
    diagnosis_metrics = {
        "base_accuracy": float(accuracy_score(diagnosis_truth, base_diagnosis)),
        "stage1_accuracy": float(accuracy_score(diagnosis_truth, new_diagnosis)),
        "base_balanced_accuracy": float(
            balanced_accuracy_score(diagnosis_truth, base_diagnosis)
        ),
        "stage1_balanced_accuracy": float(
            balanced_accuracy_score(diagnosis_truth, new_diagnosis)
        ),
    }
    raw_state_metrics = {}
    for criterion in CRITERIA:
        truth = raw_targets[criterion]
        prediction = np.argmax(oof_raw[criterion], axis=1)
        raw_state_metrics[criterion] = {
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "n_classes": RAW_N_CLASSES[criterion],
        }

    print(f"\n{'='*72}")
    print(f"  PHASE 2.5 STAGE 1 RESULTS (pooled 616-case OOF)")
    print(f"{'='*72}")
    print(f"  {'Output':24s}  {'Base':>8s}  {'Stage1':>8s}  {'Delta':>8s}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}")
    labels = ["melanoma"] + list(CRITERIA)
    for i, label in enumerate(labels):
        d = new_aucs[i] - base_aucs[i]
        print(f"  {label:24s}  {base_aucs[i]:.4f}    {new_aucs[i]:.4f}    {d:+.4f}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'8-OUTPUT MEAN':24s}  {base_primary:.4f}    {new_primary:.4f}    {new_primary - base_primary:+.4f}")
    print(f"{'='*72}")

    experiment = {
        "schema_version": "dermpact-phase2-5-stage1-v3",
        "device": device_name,
        "execution_threads": execution_threads,
        "protocol_sha256": stage_protocol_sha,
        "phase24b_protocol_sha256": protocol_sha,
        "sources_sha256": source_bundle_sha,
        "smoke": smoke,
        "config": {
            "seed": SEED, "stage1_epochs": STAGE1_EPOCHS,
            "effective_batch": EFFECTIVE_BATCH,
            "microbatch": selected_microbatch,
            "backend": resolved_backend,
            "max_logit_shift": MAX_LOGIT_SHIFT,
            "loss_weights": {
                "raw_ce": 1.0, "binary_crit_focal": BINARY_CRIT_LAMBDA,
                "diagnosis": DIAG_LAMBDA, "melanoma": MEL_LAMBDA,
                "gddw": GD_DDW_LAMBDA, "distill": DISTILL_LAMBDA,
            },
        },
        "evaluation": {
            "base_8output_mean": base_primary,
            "stage1_8output_mean": new_primary,
            "delta": new_primary - base_primary,
            "per_output": {
                label: {"base": base_aucs[i], "stage1": new_aucs[i], "delta": new_aucs[i] - base_aucs[i]}
                for i, label in enumerate(labels)
            },
            "paired_intervals": paired_intervals,
            "diagnosis": diagnosis_metrics,
            "raw_states": raw_state_metrics,
        },
        "fold_results": fold_results,
    }

    window_cache.assert_source_bindings(frozen_sources)
    raw_arrays = {
        f"raw_logits__{criterion}": values
        for criterion, values in oof_raw.items()
    }
    oof_sha = _atomic_npz(
        output_dir / "stage1_oof.npz",
        oof_crit_logits=oof_crit, oof_mel_logits=oof_mel,
        oof_diag_logits=oof_diag,
        base_crit_logits=base_crit, base_mel_logits=base_mel,
        base_diag_logits=base_diag,
        crit_targets=crit_targets, mel_targets=mel_targets,
        **raw_arrays,
    )
    experiment["oof_artifact"] = {
        "path": str(output_dir / "stage1_oof.npz"),
        "sha256": oof_sha,
    }
    _atomic_json(output_dir / "stage1_evaluation.json", experiment)
    return experiment


def run_accelerator_smoke(
    *,
    stage: int,
    microbatch: int,
    output_dir: Path,
    backend: str = "auto",
    cpu_threads: int | None = None,
    training_metadata_root: Path | None = None,
) -> dict[str, Any]:
    """Run the real Stage 1 or partial-unfreeze Stage 2 hardware gate."""
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    device, device_name, resolved_backend = _resolve_device(backend)
    execution_threads = _configure_execution_threads(
        backend=resolved_backend, cpu_threads=cpu_threads
    )
    if resolved_backend == "cpu" and training_metadata_root is None:
        raise ValueError("CPU runs require --training-metadata-root for fail-closed scope control")
    _configure_training_metadata_root(training_metadata_root)
    protocol_raw, _ = phase24b.load_protocol()
    _, clinical_cache = phase24b._live_inputs(protocol_raw)
    paired_data = training.load_training_data(clinical_cache)
    paired_data.validate()
    cache = window_cache.load_window_cache()
    rows = baselines._derm7pt_records()
    raw_targets = _load_raw_targets(rows)
    derm_data = _base._derm_data()
    class_weights = _fold_class_weights(raw_targets, paired_data.folds != 0)
    model = _build_phase25(0, paired_data)
    result = _gpu_smoke_test(
        model,
        paired_data,
        cache["window_tokens"],
        cache["patch_valid_mask"],
        torch.from_numpy(cache["window_positions"]),
        raw_targets,
        class_weights,
        derm_data,
        device,
        output_dir=Path(output_dir),
        microbatch=microbatch,
        stage2=stage == 2,
    )
    result = {
        **result,
        "device": device_name,
        "backend": resolved_backend,
        "execution_threads": execution_threads,
    }
    _atomic_json(Path(output_dir) / f"stage{stage}_{resolved_backend}_smoke.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "directml"), default="auto")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--training-metadata-root", type=Path)
    parser.add_argument("--smoke-stage", type=int, choices=(1, 2))
    parser.add_argument("--smoke-microbatch", type=int, default=MICROBATCH)
    args = parser.parse_args(argv)
    if args.smoke_stage is None:
        result = run_experiment(
            output_dir=args.output_dir,
            backend=args.backend,
            microbatch=args.microbatch,
            cpu_threads=args.cpu_threads,
            training_metadata_root=args.training_metadata_root,
        )
    else:
        result = run_accelerator_smoke(
            stage=args.smoke_stage,
            microbatch=args.smoke_microbatch,
            output_dir=args.output_dir,
            backend=args.backend,
            cpu_threads=args.cpu_threads,
            training_metadata_root=args.training_metadata_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_experiment",
    "train_fold",
    "_build_phase25",
    "_configure_execution_threads",
    "_eval_on_indices",
    "_fold_class_weights",
    "_focal_bce",
    "_gpu_smoke_test",
    "_scale_microbatch_loss",
    "run_accelerator_smoke",
]
