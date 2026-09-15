"""Phase 2.4A-T1 residual-only joint hyperparameter tuning."""
from __future__ import annotations

import argparse
import contextlib
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from . import dermpact_phase2_4_localized_training as _stage1


ROOT = _stage1.ROOT
MANIFEST_DIR = ROOT / "research/manifests/dermpact_phase2_4/localized_criterion_tuning_v1"
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_4/localized_criterion_tuning_v1"
LEDGER_PATH = MANIFEST_DIR / "candidate_ledger.json"
PROTOCOL_PATH = MANIFEST_DIR / "localized_criterion_tuning_protocol.json"
RANKING_PATH = MANIFEST_DIR / "candidate_ranking.json"
RANKING_RECEIPT_PATH = MANIFEST_DIR / "candidate_ranking_receipt.json"

STAGE1_DIR = ROOT / "research/manifests/dermpact_phase2_4/localized_criterion_v1"
STAGE1_PROTOCOL_PATH = STAGE1_DIR / "localized_criterion_protocol.json"
STAGE1_COMPARISON_PATH = STAGE1_DIR / "seed_20260827_comparison.json"
STAGE1_FINAL_RECEIPT_PATH = STAGE1_DIR / "seed_20260827_receipt.json"

LEDGER_SCHEMA = "dermpact-phase2-4a-t1-candidate-ledger-v1"
PROTOCOL_SCHEMA = "dermpact-phase2-4a-t1-tuning-protocol-v1"
TRAINING_SCHEMA = "dermpact-phase2-4a-t1-fold-training-receipt-v1"
CANONICAL_SCHEMA = "dermpact-phase2-4a-t1-canonical-fold-receipt-v1"
COMPARISON_SCHEMA = "dermpact-phase2-4a-t1-candidate-comparison-v1"
FINAL_SCHEMA = "dermpact-phase2-4a-t1-candidate-final-receipt-v1"
RANKING_SCHEMA = "dermpact-phase2-4a-t1-ranking-v1"
RANKING_RECEIPT_SCHEMA = "dermpact-phase2-4a-t1-ranking-receipt-v1"
RESUME_SCHEMA = "dermpact-phase2-4a-t1-epoch-resume-v1"

SEED = 20260827
DESIGN_SEED = 20260829
FOLDS = tuple(range(5))
LEVELS = {
    "attention_dim": (64, 128, 256),
    "projection_learning_rate": (0.0001, 0.0003, 0.001),
    "scale_learning_rate": (0.0003, 0.001, 0.003),
    "projection_weight_decay": (0.0, 0.0001, 0.001),
    "consistency_weight": (0.05, 0.10, 0.20),
}
FACTOR_NAMES = tuple(LEVELS)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(config)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".partial")
    sidecar_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(sidecar_tmp, sidecar)
    return digest


def _verify(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise AssertionError(f"SHA-256 mismatch: {_relative(path)}")
    return digest


def _artifact(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _verify(path)}


def _config_from_codes(codes: Sequence[int]) -> dict[str, Any]:
    return {name: LEVELS[name][int(level)] for name, level in zip(FACTOR_NAMES, codes)}


@lru_cache(maxsize=1)
def _design_codes() -> tuple[tuple[int, ...], ...]:
    """Deterministic balanced random-search maximin design with forced center."""
    center = (1, 1, 1, 1, 1)
    rng = random.Random(DESIGN_SEED)
    remainder = [0] * 4 + [1] * 3 + [2] * 4
    best_rows: tuple[tuple[int, ...], ...] | None = None
    best_score: tuple[int, int, int] | None = None
    for _ in range(25000):
        columns = []
        for _factor in range(5):
            column = remainder.copy()
            rng.shuffle(column)
            columns.append(column)
        rows = (center,) + tuple(
            tuple(columns[factor][row] for factor in range(5)) for row in range(11)
        )
        if len(set(rows)) != 12:
            continue
        distances = [
            sum((a - b) ** 2 for a, b in zip(rows[left], rows[right]))
            for left in range(12) for right in range(left + 1, 12)
        ]
        nearest_sum = sum(min(
            sum((a - b) ** 2 for a, b in zip(rows[index], rows[other]))
            for other in range(12) if other != index
        ) for index in range(12))
        score = (min(distances), nearest_sum, sum(distances))
        if best_score is None or score > best_score:
            best_rows, best_score = rows, score
    if best_rows is None:
        raise RuntimeError("balanced maximin search failed to produce 12 unique candidates")
    return best_rows


def generate_candidate_ledger() -> dict[str, Any]:
    candidates = []
    design = _design_codes()
    for index, codes in enumerate(design):
        config = _config_from_codes(codes)
        candidates.append({
            "candidate_id": f"c{index:02d}",
            "forced_center": index == 0,
            "coded_levels": list(codes),
            "config": config,
            "candidate_config_sha256": _config_hash(config),
        })
    pairwise = [
        sum((a - b) ** 2 for a, b in zip(design[left], design[right]))
        for left in range(12) for right in range(left + 1, 12)
    ]
    return {
        "schema_version": LEDGER_SCHEMA,
        "status": "FROZEN",
        "generation": {
            "method": "deterministic seeded balanced random-search maximin over the 3^5 discrete space",
            "seed": DESIGN_SEED,
            "distance": "squared Euclidean distance on coded levels",
            "tie_break": "seeded random number used only after maximin, balance, and total-distance scores",
            "candidate_count": 12,
            "result_based_selection": False,
            "minimum_pairwise_squared_distance": min(pairwise),
            "level_counts": {
                name: [sum(row[factor] == level for row in design) for level in range(3)]
                for factor, name in enumerate(FACTOR_NAMES)
            },
        },
        "factors": {name: list(values) for name, values in LEVELS.items()},
        "candidates": candidates,
    }


def freeze_candidate_ledger() -> dict[str, Any]:
    ledger = generate_candidate_ledger()
    digest = _atomic_json(LEDGER_PATH, ledger)
    return {**ledger, "ledger_sha256": digest}


def candidate_ids() -> tuple[str, ...]:
    return tuple(f"c{index:02d}" for index in range(12))


def _validate_candidate_id(candidate_id: str) -> None:
    if candidate_id not in candidate_ids():
        raise ValueError(f"unknown tuning candidate: {candidate_id}")


def candidate_manifest_dir(candidate_id: str) -> Path:
    _validate_candidate_id(candidate_id)
    return MANIFEST_DIR / candidate_id


def candidate_experiment_dir(candidate_id: str) -> Path:
    _validate_candidate_id(candidate_id)
    return EXPERIMENT_DIR / candidate_id / f"seed_{SEED}"


def training_receipt_path(candidate_id: str, fold: int) -> Path:
    _validate_candidate_id(candidate_id)
    if fold not in FOLDS:
        raise ValueError(fold)
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_fold_{fold}_training.json"


def canonical_receipt_path(candidate_id: str, fold: int) -> Path:
    _validate_candidate_id(candidate_id)
    if fold not in FOLDS:
        raise ValueError(fold)
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_fold_{fold}_canonical.json"


def candidate_comparison_path(candidate_id: str) -> Path:
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_comparison.json"


def candidate_final_receipt_path(candidate_id: str) -> Path:
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_receipt.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger() -> tuple[dict[str, Any], str]:
    digest = _verify(LEDGER_PATH)
    ledger = _load_json(LEDGER_PATH)
    if ledger != generate_candidate_ledger():
        raise AssertionError("candidate ledger differs from its deterministic generation")
    return ledger, digest


def _candidate(candidate_id: str) -> dict[str, Any]:
    ledger, _ = _ledger()
    matches = [row for row in ledger["candidates"] if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise AssertionError(f"candidate ledger identity changed: {candidate_id}")
    row = matches[0]
    if row["candidate_config_sha256"] != _config_hash(row["config"]):
        raise AssertionError("candidate config hash mismatch")
    return row


def _source_artifacts() -> dict[str, dict[str, str]]:
    paths = {
        "tuning_core": Path(__file__),
        "stage1_training": ROOT / "dermpact/eval/dermpact_phase2_4_localized_training.py",
        "architecture_ladder": ROOT / "dermpact/eval/dermpact_phase2_architecture_ladder.py",
        "localized_model": ROOT / "dermpact/eval/dermpact_phase2_4_model.py",
        "criterion_residual": ROOT / "dermpact/perception/criterion_residual.py",
        "directml_launcher": ROOT / "research/scripts/dermpact_phase2_4a_tuning_directml_train.py",
        "full_run_controller": ROOT / "research/scripts/dermpact_phase2_4a_tuning_full_run_controller.py",
    }
    return {name: {"path": _relative(path), "sha256": _sha256(path)} for name, path in paths.items()}


def build_protocol() -> dict[str, Any]:
    ledger = generate_candidate_ledger()
    ledger_sha = _atomic_json(LEDGER_PATH, ledger)
    stage1_protocol_sha = _verify(STAGE1_PROTOCOL_PATH)
    stage1_comparison_sha = _verify(STAGE1_COMPARISON_PATH)
    stage1_receipt_sha = _verify(STAGE1_FINAL_RECEIPT_PATH)
    stage1_receipt = _load_json(STAGE1_FINAL_RECEIPT_PATH)
    if (
        stage1_receipt.get("schema_version")
        != "dermpact-phase2-4a-localized-criterion-final-receipt-v1"
        or stage1_receipt.get("status") != "PASS"
        or stage1_receipt.get("protocol_sha256") != stage1_protocol_sha
        or stage1_receipt.get("comparison", {}).get("sha256") != stage1_comparison_sha
        or ROOT / str(stage1_receipt.get("comparison", {}).get("path"))
        != STAGE1_COMPARISON_PATH
    ):
        raise RuntimeError("completed Stage-1 result is required before T1 protocol freeze")
    stage1_protocol = _load_json(STAGE1_PROTOCOL_PATH)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "FROZEN",
        "experiment": "Phase 2.4A-T1 residual-only joint hyperparameter tuning",
        "seed": SEED,
        "folds": list(FOLDS),
        "candidate_count": 12,
        "candidate_ledger_sha256": ledger_sha,
        "training": {
            "dataset": "Derm7pt development training rows only",
            "epochs": 12,
            "effective_batch_size": 64,
            "directml_microbatch_size": 8,
            "dtype": "float32",
            "retained_core": "fully frozen",
            "candidate_parallelism": 1,
            "execution_order": "sequential; runtime binding is intentionally not thread-safe",
            "scheduler": copy.deepcopy(stage1_protocol["training"]["scheduler"]),
            "gradient_clip_norm": float(stage1_protocol["training"]["gradient_clip_norm"]),
            "scale_weight_decay": 0.0,
            "early_stopping": False,
            "best_epoch_selection": False,
        },
        "canonical_evaluation": copy.deepcopy(stage1_protocol["canonical_evaluation"]),
        "bootstrap": copy.deepcopy(stage1_protocol["bootstrap"]),
        "endpoints": copy.deepcopy(stage1_protocol["endpoints"]),
        "scope": {
            "official_or_external_test_loading": False,
            "never_loaded": copy.deepcopy(stage1_protocol["roles"]["never_loaded"]),
        },
        "ranking": {
            "safe_gate": "protected falls <=0.02 and disagreement increase <=0.02 against identical retained base",
            "order": "descending localized seven-criterion mean AUROC; candidate_id ascending tie-break",
            "advance_count": 3,
            "center_fallback_if_fewer_than_three_safe": True,
            "t2_launch": False,
            "mandatory_stage1_fallback_rule": "T2 must include Stage-1 center/default at 20 epochs; final tuned winner must statistically outperform Stage-1 criterion mean or Stage-1 is retained",
        },
        "hard_scope_checks": copy.deepcopy(stage1_protocol["hard_scope_checks"]),
        "artifacts": {
            "candidate_ledger": {"path": _relative(LEDGER_PATH), "sha256": ledger_sha},
            "stage1_protocol": {"path": _relative(STAGE1_PROTOCOL_PATH), "sha256": stage1_protocol_sha},
            "stage1_comparison": {"path": _relative(STAGE1_COMPARISON_PATH), "sha256": stage1_comparison_sha},
            "stage1_final_receipt": {"path": _relative(STAGE1_FINAL_RECEIPT_PATH), "sha256": stage1_receipt_sha},
            "retained_folds": copy.deepcopy(stage1_protocol["artifacts"]["retained_folds"]),
            "fold_manifest": copy.deepcopy(stage1_protocol["artifacts"]["fold_manifest"]),
            "semantic_embeddings": copy.deepcopy(stage1_protocol["artifacts"]["semantic_embeddings"]),
            "semantic_manifest": copy.deepcopy(stage1_protocol["artifacts"]["semantic_manifest"]),
            "sources": _source_artifacts(),
            "upstream_stage1_sources": copy.deepcopy(stage1_protocol["artifacts"]["sources"]),
            "environment_locks": copy.deepcopy(stage1_protocol["artifacts"]["environment_locks"]),
        },
    }
    digest = _atomic_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": digest}


def load_protocol() -> tuple[dict[str, Any], str]:
    digest = _verify(PROTOCOL_PATH)
    protocol = _load_json(PROTOCOL_PATH)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("status") != "FROZEN":
        raise AssertionError("T1 protocol identity changed")
    ledger, ledger_sha = _ledger()
    if protocol.get("candidate_ledger_sha256") != ledger_sha or len(ledger["candidates"]) != 12:
        raise AssertionError("T1 candidate ledger binding changed")
    for artifact in protocol["artifacts"]["sources"].values():
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"protocol-bound source changed: {artifact['path']}")
    for artifact in protocol["artifacts"]["upstream_stage1_sources"].values():
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"upstream Stage-1 source changed: {artifact['path']}")
    for name in ("stage1_protocol", "stage1_comparison", "stage1_final_receipt"):
        artifact = protocol["artifacts"][name]
        if _verify(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"Stage-1 fallback artifact changed: {name}")
    for name in ("fold_manifest", "semantic_embeddings", "semantic_manifest"):
        artifact = protocol["artifacts"][name]
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"T1 scientific input changed: {name}")
    return protocol, digest


def _candidate_protocol(candidate_id: str) -> tuple[dict[str, Any], str, str]:
    tuning, protocol_sha = load_protocol()
    candidate = _candidate(candidate_id)
    stage1 = _load_json(STAGE1_PROTOCOL_PATH)
    config = candidate["config"]
    expanded = copy.deepcopy(stage1)
    expanded["schema_version"] = PROTOCOL_SCHEMA
    expanded["identity"]["candidate_id"] = candidate_id
    expanded["identity"]["candidate_config_sha256"] = candidate["candidate_config_sha256"]
    expanded["model"]["attention_dim"] = int(config["attention_dim"])
    expanded["training"]["epochs"] = int(tuning["training"]["epochs"])
    expanded["training"]["effective_batch_size"] = int(tuning["training"]["effective_batch_size"])
    expanded["training"]["directml_microbatch_size"] = int(tuning["training"]["directml_microbatch_size"])
    expanded["training"]["consistency_weight"] = float(config["consistency_weight"])
    expanded["training"]["optimizer"]["projection_group"]["learning_rate"] = float(config["projection_learning_rate"])
    expanded["training"]["optimizer"]["projection_group"]["weight_decay"] = float(config["projection_weight_decay"])
    expanded["training"]["optimizer"]["scale_group"]["learning_rate"] = float(config["scale_learning_rate"])
    expanded["training"]["optimizer"]["scale_group"]["weight_decay"] = 0.0
    expanded["artifacts"]["sources"] = copy.deepcopy(tuning["artifacts"]["sources"])
    expanded["hard_scope_checks"] = copy.deepcopy(tuning["hard_scope_checks"])
    return expanded, protocol_sha, candidate["candidate_config_sha256"]


def validate_tuning_resume_identity(
    payload: Mapping[str, Any], *, protocol_sha256: str, fold: int,
    source_sha256: str, ledger_sha256: str, candidate_id: str,
    candidate_config_sha256: str,
) -> None:
    observed = (
        payload.get("format"), payload.get("protocol_sha256"), payload.get("seed"),
        payload.get("fold"), payload.get("source_checkpoint_sha256"),
        payload.get("ledger_sha256"), payload.get("candidate_id"),
        payload.get("candidate_config_sha256"),
    )
    expected = (
        RESUME_SCHEMA, protocol_sha256, SEED, fold, source_sha256,
        ledger_sha256, candidate_id, candidate_config_sha256,
    )
    if observed != expected:
        raise AssertionError(f"epoch-resume candidate identity mismatch: {observed!r}")


@contextlib.contextmanager
def _stage1_binding(candidate_id: str) -> Iterator[tuple[dict[str, Any], str, str]]:
    """Temporarily route Stage-1 routines into one isolated candidate directory."""
    protocol, protocol_sha, config_sha = _candidate_protocol(candidate_id)
    candidate_exp = candidate_experiment_dir(candidate_id)
    comparison = candidate_comparison_path(candidate_id)
    final_receipt = candidate_final_receipt_path(candidate_id)
    _, ledger_sha = _ledger()
    original_write_resume = _stage1._write_resume
    original_validate_resume = _stage1.validate_resume_identity
    original_verified_training = _stage1._verified_training_fold
    original_canonical_fold = _stage1._canonical_fold

    def write_resume(fold: int, **kwargs: Any) -> str:
        original_write_resume(fold, **kwargs)
        path = _stage1._resume_path(fold)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "ledger_sha256": ledger_sha,
            "candidate_id": candidate_id,
            "candidate_config_sha256": config_sha,
        })
        return _stage1._atomic_torch(path, payload)

    def validate_resume(payload: Mapping[str, Any], *, protocol_sha256: str,
                        fold: int, source_sha256: str) -> None:
        validate_tuning_resume_identity(
            payload, protocol_sha256=protocol_sha256, fold=fold,
            source_sha256=source_sha256, ledger_sha256=ledger_sha,
            candidate_id=candidate_id, candidate_config_sha256=config_sha,
        )

    def verified_training(protocol_sha_arg: str, fold: int) -> dict[str, Any] | None:
        receipt = original_verified_training(protocol_sha_arg, fold)
        if receipt is None:
            return None
        expected = (ledger_sha, candidate_id, config_sha)
        observed = (
            receipt.get("ledger_sha256"), receipt.get("candidate_id"),
            receipt.get("candidate_config_sha256"),
        )
        return receipt if observed == expected else None

    def canonical_fold(protocol_sha_arg: str, fold: int):
        receipt, raw = original_canonical_fold(protocol_sha_arg, fold)
        expected = (ledger_sha, candidate_id, config_sha)
        observed = (
            receipt.get("ledger_sha256"), receipt.get("candidate_id"),
            receipt.get("candidate_config_sha256"),
        )
        if observed != expected:
            raise AssertionError(f"canonical candidate identity mismatch: {candidate_id}/fold{fold}")
        return receipt, raw

    attributes = {
        "EXPERIMENT_DIR": candidate_exp.parent,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "COMPARISON_PATH": comparison,
        "FINAL_RECEIPT_PATH": final_receipt,
        "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA,
        "RESUME_SCHEMA": RESUME_SCHEMA,
        "FOLD_SCHEMA": TRAINING_SCHEMA,
        "CANONICAL_SCHEMA": CANONICAL_SCHEMA,
        "_fold_receipt_path": lambda fold: training_receipt_path(candidate_id, fold),
        "_canonical_receipt_path": lambda fold: canonical_receipt_path(candidate_id, fold),
        "load_protocol": lambda: (protocol, protocol_sha),
        "_write_resume": write_resume,
        "validate_resume_identity": validate_resume,
        "_verified_training_fold": verified_training,
        "_canonical_fold": canonical_fold,
    }
    original = {name: getattr(_stage1, name) for name in attributes}
    original_consistency = _stage1.legacy_training.ladder.CONSISTENCY_WEIGHT
    try:
        for name, value in attributes.items():
            setattr(_stage1, name, value)
        _stage1.legacy_training.ladder.CONSISTENCY_WEIGHT = float(
            protocol["training"]["consistency_weight"]
        )
        yield protocol, protocol_sha, config_sha
    finally:
        _stage1.legacy_training.ladder.CONSISTENCY_WEIGHT = original_consistency
        for name, value in original.items():
            setattr(_stage1, name, value)


def _augment_receipt(path: Path, *, candidate_id: str, protocol_sha: str, ledger_sha: str,
                     config_sha: str, schema: str) -> dict[str, Any]:
    receipt = _load_json(path)
    receipt.update({
        "schema_version": schema,
        "protocol_sha256": protocol_sha,
        "ledger_sha256": ledger_sha,
        "candidate_id": candidate_id,
        "candidate_config_sha256": config_sha,
    })
    _atomic_json(path, receipt)
    return receipt


def train_candidate_fold(candidate_id: str, fold: int, training_device: Any) -> dict[str, Any]:
    _validate_candidate_id(candidate_id)
    if fold not in FOLDS:
        raise ValueError(fold)
    _, ledger_sha = _ledger()
    with _stage1_binding(candidate_id) as (protocol, protocol_sha, config_sha):
        device = torch.device(training_device) if isinstance(training_device, str) else training_device
        result = _stage1.train_fold(
            fold,
            device=device,
            microbatch_size=int(protocol["training"]["directml_microbatch_size"]),
        )
        receipt = _augment_receipt(
            training_receipt_path(candidate_id, fold), candidate_id=candidate_id,
            protocol_sha=protocol_sha, ledger_sha=ledger_sha, config_sha=config_sha,
            schema=TRAINING_SCHEMA,
        )
    return {**receipt, "receipt_sha256": _verify(training_receipt_path(candidate_id, fold)),
            "reuse_status": result.get("status")}


def reevaluate_candidate_fold(candidate_id: str, fold: int) -> dict[str, Any]:
    _validate_candidate_id(candidate_id)
    if fold not in FOLDS:
        raise ValueError(fold)
    _, ledger_sha = _ledger()
    with _stage1_binding(candidate_id) as (_, protocol_sha, config_sha):
        result = _stage1.canonical_reevaluate_fold(fold)
        receipt = _augment_receipt(
            canonical_receipt_path(candidate_id, fold), candidate_id=candidate_id,
            protocol_sha=protocol_sha, ledger_sha=ledger_sha, config_sha=config_sha,
            schema=CANONICAL_SCHEMA,
        )
    return {**receipt, "receipt_sha256": _verify(canonical_receipt_path(candidate_id, fold)),
            "execution_status": result.get("status")}


def finalize_candidate(candidate_id: str) -> dict[str, Any]:
    _validate_candidate_id(candidate_id)
    _, ledger_sha = _ledger()
    with _stage1_binding(candidate_id) as (_, protocol_sha, config_sha):
        _stage1.finalize()
        comparison_path = candidate_comparison_path(candidate_id)
        original = _load_json(comparison_path)
        per_criterion = original["per_criterion_auroc"]
        criterion_mean = float(np.mean([float(value["localized"]) for value in per_criterion.values()]))
        checks = original["promotion_checks"]
        safe = bool(checks["protected_falls_within_0p02"] and checks["disagreement_increase_within_0p02"])
        comparison = {
            **original,
            "schema_version": COMPARISON_SCHEMA,
            "protocol_sha256": protocol_sha,
            "ledger_sha256": ledger_sha,
            "candidate_id": candidate_id,
            "candidate_config_sha256": config_sha,
            "criterion_mean": criterion_mean,
            "safety_checks": {
                "protected_falls_within_0p02": bool(checks["protected_falls_within_0p02"]),
                "disagreement_increase_within_0p02": bool(checks["disagreement_increase_within_0p02"]),
            },
            "safety_decision": "SAFE" if safe else "UNSAFE",
            "scientific_decision": "ADVANCE_ELIGIBLE" if safe else "NOT_SAFE",
        }
        comparison_sha = _atomic_json(comparison_path, comparison)
        receipt_path = candidate_final_receipt_path(candidate_id)
        original_receipt = _load_json(receipt_path)
        receipt = {
            **original_receipt,
            "schema_version": FINAL_SCHEMA,
            "status": "PASS",
            "protocol_sha256": protocol_sha,
            "ledger_sha256": ledger_sha,
            "candidate_id": candidate_id,
            "candidate_config_sha256": config_sha,
            "safety_decision": comparison["safety_decision"],
            "scientific_decision": comparison["scientific_decision"],
            "criterion_mean": criterion_mean,
            "comparison": {"path": _relative(comparison_path), "sha256": comparison_sha},
        }
        receipt_sha = _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def select_advanced_candidates(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    safe = sorted(
        (row for row in rows if row.get("safety_decision") == "SAFE"),
        key=lambda row: (-float(row["criterion_mean"]), str(row["candidate_id"])),
    )
    advanced = [str(row["candidate_id"]) for row in safe[:3]]
    if len(advanced) < 3 and "c00" not in advanced:
        advanced.append("c00")
    return advanced


def _mandatory_fallback(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": "stage1_center_epoch20",
        "config_source": "c00",
        "epochs": 20,
        "comparison": copy.deepcopy(protocol["artifacts"]["stage1_comparison"]),
        "final_receipt": copy.deepcopy(protocol["artifacts"]["stage1_final_receipt"]),
        "rule": "T2 must include this arm; final tuned winner must statistically outperform its criterion mean or Stage 1 is retained",
    }


def rank_candidates() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    _, ledger_sha = _ledger()
    rows = []
    artifacts = []
    for candidate_id in candidate_ids():
        path = candidate_final_receipt_path(candidate_id)
        digest = _verify(path)
        receipt = _load_json(path)
        expected = (FINAL_SCHEMA, "PASS", protocol_sha, ledger_sha, candidate_id)
        observed = (
            receipt.get("schema_version"), receipt.get("status"), receipt.get("protocol_sha256"),
            receipt.get("ledger_sha256"), receipt.get("candidate_id"),
        )
        if observed != expected:
            raise AssertionError(f"candidate final receipt identity mismatch: {candidate_id}")
        comparison = receipt["comparison"]
        if _verify(ROOT / comparison["path"]) != comparison["sha256"]:
            raise AssertionError(f"candidate comparison changed: {candidate_id}")
        rows.append({
            "candidate_id": candidate_id,
            "safety_decision": receipt["safety_decision"],
            "criterion_mean": float(receipt["criterion_mean"]),
        })
        artifacts.append({"candidate_id": candidate_id, "final_receipt": {"path": _relative(path), "sha256": digest}})
    safe_order = sorted(rows, key=lambda row: (
        row["safety_decision"] != "SAFE", -row["criterion_mean"], row["candidate_id"]
    ))
    advanced = select_advanced_candidates(rows)
    ranking = {
        "schema_version": RANKING_SCHEMA,
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "ledger_sha256": ledger_sha,
        "ranking_basis": "safe candidates first, descending localized seven-criterion mean AUROC, candidate_id tie-break",
        "ordered_candidates": safe_order,
        "advanced_candidates": advanced,
        "center_fallback_applied": len([row for row in rows if row["safety_decision"] == "SAFE"]) < 3,
        "mandatory_fallback": _mandatory_fallback(protocol),
        "t2_launched": False,
    }
    ranking_sha = _atomic_json(RANKING_PATH, ranking)
    receipt = {
        "schema_version": RANKING_RECEIPT_SCHEMA,
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "ledger_sha256": ledger_sha,
        "ranking": {"path": _relative(RANKING_PATH), "sha256": ranking_sha},
        "advanced_candidates": advanced,
        "mandatory_fallback": _mandatory_fallback(protocol),
        "candidate_final_receipts": artifacts,
        "t2_launched": False,
    }
    receipt_sha = _atomic_json(RANKING_RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def controller_matrix() -> list[dict[str, Any]]:
    jobs = []
    for candidate_id in candidate_ids():
        jobs.extend({"stage": "train", "candidate_id": candidate_id, "fold": fold, "environment": "venv_dml"} for fold in FOLDS)
        jobs.extend({"stage": "reevaluate", "candidate_id": candidate_id, "fold": fold, "environment": "venv_p2"} for fold in FOLDS)
        jobs.append({"stage": "finalize_candidate", "candidate_id": candidate_id, "fold": None, "environment": "venv_p2"})
    jobs.append({"stage": "rank", "candidate_id": None, "fold": None, "environment": "venv_p2"})
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ledger-only", action="store_true")
    mode.add_argument("--protocol-only", action="store_true")
    mode.add_argument("--train-fold", action="store_true")
    mode.add_argument("--reevaluate-fold", action="store_true")
    mode.add_argument("--finalize-candidate", action="store_true")
    mode.add_argument("--rank", action="store_true")
    parser.add_argument("--candidate", choices=candidate_ids())
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    if args.ledger_only:
        payload = freeze_candidate_ledger()
    elif args.protocol_only:
        payload = build_protocol()
    elif args.train_fold:
        if args.candidate is None or args.fold is None:
            parser.error("--train-fold requires --candidate and --fold")
        payload = train_candidate_fold(args.candidate, args.fold, args.device)
    elif args.reevaluate_fold:
        if args.candidate is None or args.fold is None:
            parser.error("--reevaluate-fold requires --candidate and --fold")
        payload = reevaluate_candidate_fold(args.candidate, args.fold)
    elif args.finalize_candidate:
        if args.candidate is None:
            parser.error("--finalize-candidate requires --candidate")
        payload = finalize_candidate(args.candidate)
    else:
        payload = rank_candidates()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "generate_candidate_ledger", "freeze_candidate_ledger", "build_protocol",
    "train_candidate_fold", "reevaluate_candidate_fold", "finalize_candidate",
    "rank_candidates", "select_advanced_candidates", "controller_matrix", "candidate_ids",
    "training_receipt_path", "canonical_receipt_path", "candidate_comparison_path",
    "candidate_final_receipt_path",
    "validate_tuning_resume_identity",
]
