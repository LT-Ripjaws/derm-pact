"""Phase 2.4A-T2 isolated epoch-20 refits and ranking."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from . import dermpact_phase2_4a_tuning as _t1
from . import dermpact_phase2_4_localized_training as _stage1


ROOT = _t1.ROOT
NAMESPACE = "localized_criterion_tuning_t2_epoch20_v1"
MANIFEST_DIR = ROOT / "research/manifests/dermpact_phase2_4" / NAMESPACE
EXPERIMENT_DIR = ROOT / "research/experiments/dermpact_phase2_4" / NAMESPACE
SELECTION_LEDGER_PATH = MANIFEST_DIR / "epoch20_selection_ledger.json"
PROTOCOL_PATH = MANIFEST_DIR / "localized_criterion_t2_epoch20_protocol.json"
RANKING_PATH = MANIFEST_DIR / "epoch20_candidate_ranking.json"
RANKING_RECEIPT_PATH = MANIFEST_DIR / "epoch20_candidate_ranking_receipt.json"

T1_PROTOCOL_PATH = _t1.PROTOCOL_PATH
T1_LEDGER_PATH = _t1.LEDGER_PATH
T1_RANKING_PATH = _t1.RANKING_PATH
T1_RANKING_RECEIPT_PATH = _t1.RANKING_RECEIPT_PATH
STAGE1_COMPARISON_PATH = _t1.STAGE1_COMPARISON_PATH
STAGE1_FINAL_RECEIPT_PATH = _t1.STAGE1_FINAL_RECEIPT_PATH

T1_PROTOCOL_SHA256 = "aa3f806f98c97710b6a907a0f5fddfba097a5e6234b93ba49a4a5fcc724f3737"
T1_LEDGER_SHA256 = "07ce21bb644c59a8216a61b0090ff86dd71d1d78ba024f3d8ff033d2d5950801"
T1_RANKING_SHA256 = "8e1b57cce6b6cd98060c5d5e619eb8b6e5094a361ab7c947f782604de9caf37d"
T1_RANKING_RECEIPT_SHA256 = "6d96e05fda8f5bbdac9f4c1c927289023b8dc8e6556f18ebdc16754e395001b0"

SELECTION_SCHEMA = "dermpact-phase2-4a-t2-epoch20-selection-ledger-v1"
PROTOCOL_SCHEMA = "dermpact-phase2-4a-t2-epoch20-protocol-v1"
TRAINING_SCHEMA = "dermpact-phase2-4a-t2-epoch20-fold-training-receipt-v1"
CANONICAL_SCHEMA = "dermpact-phase2-4a-t2-epoch20-canonical-fold-receipt-v1"
COMPARISON_SCHEMA = "dermpact-phase2-4a-t2-epoch20-candidate-comparison-v1"
FINAL_SCHEMA = "dermpact-phase2-4a-t2-epoch20-candidate-final-receipt-v1"
RANKING_SCHEMA = "dermpact-phase2-4a-t2-epoch20-ranking-v1"
RANKING_RECEIPT_SCHEMA = "dermpact-phase2-4a-t2-epoch20-ranking-receipt-v1"
RESUME_SCHEMA = "dermpact-phase2-4a-t2-epoch20-resume-v1"

SEED = 20260827
FOLDS = tuple(range(5))
ADVANCED_CANDIDATES = ("c05", "c10", "c01")
REFERENCE_CANDIDATE_ID = "c00@20"


def candidate_ids() -> tuple[str, ...]:
    return ADVANCED_CANDIDATES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace(chr(92), "/")


def _artifact(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise AssertionError(f"SHA-256 mismatch: {_relative(path)}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise AssertionError(f"frozen artifact identity changed: {_relative(path)}")
    return {"path": _relative(path), "sha256": digest}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _verified_t1_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _artifact(T1_PROTOCOL_PATH, T1_PROTOCOL_SHA256)
    _artifact(T1_LEDGER_PATH, T1_LEDGER_SHA256)
    _artifact(T1_RANKING_PATH, T1_RANKING_SHA256)
    _artifact(T1_RANKING_RECEIPT_PATH, T1_RANKING_RECEIPT_SHA256)
    protocol = _load_json(T1_PROTOCOL_PATH)
    ledger = _load_json(T1_LEDGER_PATH)
    ranking = _load_json(T1_RANKING_PATH)
    ranking_receipt = _load_json(T1_RANKING_RECEIPT_PATH)
    if ranking.get("advanced_candidates") != list(ADVANCED_CANDIDATES):
        raise AssertionError("T1 advanced candidate set changed")
    if ranking_receipt.get("advanced_candidates") != list(ADVANCED_CANDIDATES):
        raise AssertionError("T1 ranking receipt candidate set changed")
    if (ranking.get("protocol_sha256"), ranking.get("ledger_sha256")) != (
        T1_PROTOCOL_SHA256, T1_LEDGER_SHA256,
    ):
        raise AssertionError("T1 ranking bindings changed")
    if ranking_receipt.get("ranking", {}).get("sha256") != T1_RANKING_SHA256:
        raise AssertionError("T1 ranking receipt binding changed")
    return protocol, ledger, ranking


def generate_selection_ledger() -> dict[str, Any]:
    """Bind the exact T1 advancement decision and immutable Stage-1 arm."""
    _, ledger, ranking = _verified_t1_inputs()
    indexed = {row["candidate_id"]: row for row in ledger["candidates"]}
    candidates = []
    for rank, candidate_id in enumerate(ADVANCED_CANDIDATES, start=1):
        row = indexed[candidate_id]
        candidates.append({
            "candidate_id": candidate_id,
            "config": copy.deepcopy(row["config"]),
            "candidate_config_sha256": row["candidate_config_sha256"],
            "t1_rank": rank,
        })
    fallback = ranking["mandatory_fallback"]
    return {
        "schema_version": SELECTION_SCHEMA,
        "status": "BOUND",
        "seed": SEED,
        "source_t1": {
            "protocol": _artifact(T1_PROTOCOL_PATH, T1_PROTOCOL_SHA256),
            "candidate_ledger": _artifact(T1_LEDGER_PATH, T1_LEDGER_SHA256),
            "ranking": _artifact(T1_RANKING_PATH, T1_RANKING_SHA256),
            "ranking_receipt": _artifact(T1_RANKING_RECEIPT_PATH, T1_RANKING_RECEIPT_SHA256),
        },
        "advanced_candidates": list(ADVANCED_CANDIDATES),
        "candidates": candidates,
        "stage1_reference": {
            "candidate_id": REFERENCE_CANDIDATE_ID,
            "config_source": "c00",
            "epochs": 20,
            "retrained": False,
            "immutable": True,
            "comparison": copy.deepcopy(fallback["comparison"]),
            "final_receipt": copy.deepcopy(fallback["final_receipt"]),
        },
    }


def freeze_selection_ledger() -> dict[str, Any]:
    payload = generate_selection_ledger()
    digest = _atomic_json(SELECTION_LEDGER_PATH, payload)
    return {**payload, "selection_ledger_sha256": digest}


def generate_protocol_spec() -> dict[str, Any]:
    """Return the deterministic scientific design without writing artifacts."""
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "SPECIFIED",
        "experiment": "Phase 2.4A-T2 epoch-20 from-scratch refits",
        "seed": SEED,
        "folds": list(FOLDS),
        "advanced_candidates": list(ADVANCED_CANDIDATES),
        "training": {
            "dataset": "Derm7pt development training rows only",
            "epochs": 20,
            "initialization": "from_scratch",
            "effective_batch_size": 64,
            "directml_microbatch_size": 8,
            "dtype": "float32",
            "precision": "fp32",
            "retained_core": "fully frozen",
            "early_stopping": False,
            "best_epoch_selection": False,
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_ratio": 0.1,
                "minimum_lr_ratio": 0.01,
                "horizon_epochs": 20,
                "restart_from_t1": False,
            },
        },
        "scope": {
            "official_or_external_test_loading": False,
            "never_loaded": [
                "Derm7pt official test", "HAM calibration", "HAM final",
                "BCN validation", "BCN final", "PH2",
            ],
        },
        "ranking": {
            "reference_candidate_id": REFERENCE_CANDIDATE_ID,
            "reference_is_retrained": False,
            "safe_gate": "original protected gates against identical retained base",
            "order": "SAFE first; descending seven-criterion mean AUROC; candidate_id tie-break",
            "epoch30_selection_count": 2,
            "epoch30_launched": False,
            "winner_rule": "winner must have paired candidate-vs-c00@20 95% CI lower > 0; otherwise retain c00@20",
        },
    }


def _validate_candidate(candidate_id: str) -> None:
    if candidate_id not in ADVANCED_CANDIDATES:
        raise ValueError(f"candidate was not advanced by T1: {candidate_id}")


def candidate_manifest_dir(candidate_id: str) -> Path:
    _validate_candidate(candidate_id)
    return MANIFEST_DIR / candidate_id


def candidate_experiment_dir(candidate_id: str) -> Path:
    _validate_candidate(candidate_id)
    return EXPERIMENT_DIR / candidate_id / f"seed_{SEED}"


def training_receipt_path(candidate_id: str, fold: int) -> Path:
    _validate_candidate(candidate_id)
    if fold not in FOLDS:
        raise ValueError(fold)
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_fold_{fold}_training.json"


def canonical_receipt_path(candidate_id: str, fold: int) -> Path:
    _validate_candidate(candidate_id)
    if fold not in FOLDS:
        raise ValueError(fold)
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_fold_{fold}_canonical.json"


def candidate_comparison_path(candidate_id: str) -> Path:
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_comparison.json"


def candidate_final_receipt_path(candidate_id: str) -> Path:
    return candidate_manifest_dir(candidate_id) / f"seed_{SEED}_receipt.json"


def candidate_config(candidate_id: str) -> dict[str, Any]:
    _validate_candidate(candidate_id)
    row = next(
        row for row in generate_selection_ledger()["candidates"]
        if row["candidate_id"] == candidate_id
    )
    return copy.deepcopy(row)


def _source_artifacts() -> dict[str, dict[str, str]]:
    paths = {
        "t2_core": Path(__file__),
        "stage1_training": ROOT / "dermpact/eval/dermpact_phase2_4_localized_training.py",
        "t1_core": ROOT / "dermpact/eval/dermpact_phase2_4a_tuning.py",
        "architecture_ladder": ROOT / "dermpact/eval/dermpact_phase2_architecture_ladder.py",
        "localized_model": ROOT / "dermpact/eval/dermpact_phase2_4_model.py",
        "criterion_residual": ROOT / "dermpact/perception/criterion_residual.py",
        "directml_launcher": ROOT / "research/scripts/dermpact_phase2_4a_t2_epoch20_directml_train.py",
        "full_run_controller": ROOT / "research/scripts/dermpact_phase2_4a_t2_epoch20_full_run_controller.py",
    }
    return {
        name: {"path": _relative(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }


def build_protocol() -> dict[str, Any]:
    """Atomically freeze the isolated T2 protocol after verifying all T1 inputs."""
    t1_protocol, _, _ = _verified_t1_inputs()
    selection = generate_selection_ledger()
    selection_sha = _atomic_json(SELECTION_LEDGER_PATH, selection)
    spec = generate_protocol_spec()
    stage1_comparison = selection["stage1_reference"]["comparison"]
    stage1_receipt = selection["stage1_reference"]["final_receipt"]
    if _artifact(ROOT / stage1_comparison["path"])["sha256"] != stage1_comparison["sha256"]:
        raise AssertionError("Stage1 comparison binding changed")
    if _artifact(ROOT / stage1_receipt["path"])["sha256"] != stage1_receipt["sha256"]:
        raise AssertionError("Stage1 final receipt binding changed")
    protocol = {
        **spec,
        "status": "FROZEN",
        "selection_ledger_sha256": selection_sha,
        "canonical_evaluation": copy.deepcopy(t1_protocol["canonical_evaluation"]),
        "bootstrap": copy.deepcopy(t1_protocol["bootstrap"]),
        "endpoints": copy.deepcopy(t1_protocol["endpoints"]),
        "hard_scope_checks": copy.deepcopy(t1_protocol["hard_scope_checks"]),
        "artifacts": {
            "selection_ledger": {"path": _relative(SELECTION_LEDGER_PATH), "sha256": selection_sha},
            **copy.deepcopy(selection["source_t1"]),
            "stage1_comparison": copy.deepcopy(stage1_comparison),
            "stage1_final_receipt": copy.deepcopy(stage1_receipt),
            "retained_folds": copy.deepcopy(t1_protocol["artifacts"]["retained_folds"]),
            "fold_manifest": copy.deepcopy(t1_protocol["artifacts"]["fold_manifest"]),
            "semantic_embeddings": copy.deepcopy(t1_protocol["artifacts"]["semantic_embeddings"]),
            "semantic_manifest": copy.deepcopy(t1_protocol["artifacts"]["semantic_manifest"]),
            "sources": _source_artifacts(),
            "upstream_stage1_sources": copy.deepcopy(t1_protocol["artifacts"]["upstream_stage1_sources"]),
            "environment_locks": copy.deepcopy(t1_protocol["artifacts"]["environment_locks"]),
        },
    }
    digest = _atomic_json(PROTOCOL_PATH, protocol)
    return {**protocol, "protocol_sha256": digest}


def load_protocol() -> tuple[dict[str, Any], str]:
    protocol, digest = _load_json(PROTOCOL_PATH), _artifact(PROTOCOL_PATH)["sha256"]
    if (protocol.get("schema_version"), protocol.get("status"), protocol.get("seed")) != (
        PROTOCOL_SCHEMA, "FROZEN", SEED,
    ):
        raise AssertionError("T2 protocol identity changed")
    expected = {
        "protocol": T1_PROTOCOL_SHA256,
        "candidate_ledger": T1_LEDGER_SHA256,
        "ranking": T1_RANKING_SHA256,
        "ranking_receipt": T1_RANKING_RECEIPT_SHA256,
    }
    for name, expected_sha in expected.items():
        artifact = protocol["artifacts"][name]
        if _artifact(ROOT / artifact["path"], expected_sha)["sha256"] != artifact["sha256"]:
            raise AssertionError(f"T1 protocol artifact changed: {name}")
    if _artifact(SELECTION_LEDGER_PATH)["sha256"] != protocol["selection_ledger_sha256"]:
        raise AssertionError("T2 selection ledger changed")
    for artifact in protocol["artifacts"]["sources"].values():
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"protocol-bound source changed: {artifact['path']}")
    for artifact in protocol["artifacts"]["upstream_stage1_sources"].values():
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise AssertionError(f"upstream Stage1 source changed: {artifact['path']}")
    return protocol, digest


def _candidate_protocol(candidate_id: str) -> tuple[dict[str, Any], str, str]:
    tuning, protocol_sha = load_protocol()
    candidate = candidate_config(candidate_id)
    stage1 = _load_json(_t1.STAGE1_PROTOCOL_PATH)
    config = candidate["config"]
    expanded = copy.deepcopy(stage1)
    expanded["schema_version"] = PROTOCOL_SCHEMA
    expanded["identity"]["candidate_id"] = candidate_id
    expanded["identity"]["candidate_config_sha256"] = candidate["candidate_config_sha256"]
    expanded["identity"]["t1_ranking_sha256"] = T1_RANKING_SHA256
    expanded["model"]["attention_dim"] = int(config["attention_dim"])
    expanded["training"].update(copy.deepcopy(tuning["training"]))
    expanded["training"]["consistency_weight"] = float(config["consistency_weight"])
    expanded["training"]["optimizer"]["projection_group"]["learning_rate"] = float(config["projection_learning_rate"])
    expanded["training"]["optimizer"]["projection_group"]["weight_decay"] = float(config["projection_weight_decay"])
    expanded["training"]["optimizer"]["scale_group"]["learning_rate"] = float(config["scale_learning_rate"])
    expanded["training"]["optimizer"]["scale_group"]["weight_decay"] = 0.0
    expanded["artifacts"]["sources"] = copy.deepcopy(tuning["artifacts"]["sources"])
    expanded["hard_scope_checks"] = copy.deepcopy(tuning["hard_scope_checks"])
    return expanded, protocol_sha, candidate["candidate_config_sha256"]


def validate_resume_identity(
    payload: Mapping[str, Any], *, protocol_sha256: str, fold: int,
    source_checkpoint_sha256: str, t1_ledger_sha256: str,
    t1_ranking_sha256: str, candidate_id: str,
    candidate_config_sha256: str,
) -> None:
    observed = (
        payload.get("format"), payload.get("protocol_sha256"), payload.get("seed"),
        payload.get("fold"), payload.get("source_checkpoint_sha256"),
        payload.get("t1_ledger_sha256"), payload.get("t1_ranking_sha256"),
        payload.get("candidate_id"), payload.get("candidate_config_sha256"),
    )
    expected = (
        RESUME_SCHEMA, protocol_sha256, SEED, fold, source_checkpoint_sha256,
        t1_ledger_sha256, t1_ranking_sha256, candidate_id,
        candidate_config_sha256,
    )
    if observed != expected:
        raise AssertionError(f"epoch20 resume identity mismatch: {observed!r}")


def _selection() -> tuple[dict[str, Any], str]:
    digest = _artifact(SELECTION_LEDGER_PATH)["sha256"]
    payload = _load_json(SELECTION_LEDGER_PATH)
    if (
        payload.get("schema_version"),
        payload.get("status"),
        payload.get("seed"),
        payload.get("advanced_candidates"),
    ) != (SELECTION_SCHEMA, "BOUND", SEED, list(ADVANCED_CANDIDATES)):
        raise AssertionError("T2 selection ledger identity changed")
    return payload, digest


@contextlib.contextmanager
def _stage1_binding(candidate_id: str) -> Iterator[tuple[dict[str, Any], str, str, str]]:
    """Route the tested Stage-1 engine into one isolated T2 candidate namespace."""
    protocol, protocol_sha, config_sha = _candidate_protocol(candidate_id)
    selection, selection_sha = _selection()
    t1_ledger_sha = selection["source_t1"]["candidate_ledger"]["sha256"]
    t1_ranking_sha = selection["source_t1"]["ranking"]["sha256"]
    candidate_exp = candidate_experiment_dir(candidate_id)
    original_write_resume = _stage1._write_resume
    original_validate_resume = _stage1.validate_resume_identity
    original_verified_training = _stage1._verified_training_fold
    original_canonical_fold = _stage1._canonical_fold

    def write_resume(fold: int, **kwargs: Any) -> str:
        original_write_resume(fold, **kwargs)
        path = _stage1._resume_path(fold)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "t1_ledger_sha256": t1_ledger_sha,
            "t1_ranking_sha256": t1_ranking_sha,
            "candidate_id": candidate_id,
            "candidate_config_sha256": config_sha,
        })
        return _stage1._atomic_torch(path, payload)

    def validate_resume(
        payload: Mapping[str, Any], *, protocol_sha256: str, fold: int,
        source_sha256: str,
    ) -> None:
        validate_resume_identity(
            payload,
            protocol_sha256=protocol_sha256,
            fold=fold,
            source_checkpoint_sha256=source_sha256,
            t1_ledger_sha256=t1_ledger_sha,
            t1_ranking_sha256=t1_ranking_sha,
            candidate_id=candidate_id,
            candidate_config_sha256=config_sha,
        )

    def verified_training(protocol_sha_arg: str, fold: int) -> dict[str, Any] | None:
        receipt = original_verified_training(protocol_sha_arg, fold)
        if receipt is None:
            return None
        expected = (t1_ledger_sha, t1_ranking_sha, candidate_id, config_sha)
        observed = (
            receipt.get("t1_ledger_sha256"), receipt.get("t1_ranking_sha256"),
            receipt.get("candidate_id"), receipt.get("candidate_config_sha256"),
        )
        return receipt if observed == expected else None

    def canonical_fold(protocol_sha_arg: str, fold: int):
        receipt, raw = original_canonical_fold(protocol_sha_arg, fold)
        expected = (t1_ledger_sha, t1_ranking_sha, candidate_id, config_sha)
        observed = (
            receipt.get("t1_ledger_sha256"), receipt.get("t1_ranking_sha256"),
            receipt.get("candidate_id"), receipt.get("candidate_config_sha256"),
        )
        if observed != expected:
            raise AssertionError(
                f"canonical T2 candidate identity mismatch: {candidate_id}/fold{fold}"
            )
        return receipt, raw

    attributes = {
        "EXPERIMENT_DIR": candidate_exp.parent,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "COMPARISON_PATH": candidate_comparison_path(candidate_id),
        "FINAL_RECEIPT_PATH": candidate_final_receipt_path(candidate_id),
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
    originals = {name: getattr(_stage1, name) for name in attributes}
    original_consistency = _stage1.legacy_training.ladder.CONSISTENCY_WEIGHT
    try:
        for name, value in attributes.items():
            setattr(_stage1, name, value)
        _stage1.legacy_training.ladder.CONSISTENCY_WEIGHT = float(
            protocol["training"]["consistency_weight"]
        )
        yield protocol, protocol_sha, config_sha, selection_sha
    finally:
        _stage1.legacy_training.ladder.CONSISTENCY_WEIGHT = original_consistency
        for name, value in originals.items():
            setattr(_stage1, name, value)


def _augment_fold_receipt(
    path: Path, *, candidate_id: str, protocol_sha: str, selection_sha: str,
    config_sha: str, schema: str,
) -> dict[str, Any]:
    receipt = _load_json(path)
    receipt.update({
        "schema_version": schema,
        "protocol_sha256": protocol_sha,
        "selection_ledger_sha256": selection_sha,
        "t1_ledger_sha256": T1_LEDGER_SHA256,
        "t1_ranking_sha256": T1_RANKING_SHA256,
        "candidate_id": candidate_id,
        "candidate_config_sha256": config_sha,
    })
    _atomic_json(path, receipt)
    return receipt


def _validate_seed(seed: int) -> None:
    if seed != SEED:
        raise ValueError(f"T2 seed is frozen at {SEED}, received {seed}")


def train_candidate_fold(
    candidate_id: str, fold: int, training_device: Any, *, seed: int = SEED,
) -> dict[str, Any]:
    _validate_candidate(candidate_id)
    _validate_seed(seed)
    if fold not in FOLDS:
        raise ValueError(fold)
    with _stage1_binding(candidate_id) as (protocol, protocol_sha, config_sha, selection_sha):
        device = (
            torch.device(training_device)
            if isinstance(training_device, str)
            else training_device
        )
        result = _stage1.train_fold(
            fold,
            device=device,
            microbatch_size=int(protocol["training"]["directml_microbatch_size"]),
        )
        receipt = _augment_fold_receipt(
            training_receipt_path(candidate_id, fold),
            candidate_id=candidate_id,
            protocol_sha=protocol_sha,
            selection_sha=selection_sha,
            config_sha=config_sha,
            schema=TRAINING_SCHEMA,
        )
    return {
        **receipt,
        "receipt_sha256": _artifact(training_receipt_path(candidate_id, fold))["sha256"],
        "reuse_status": result.get("status"),
    }


def reevaluate_candidate_fold(
    candidate_id: str, fold: int, *, seed: int = SEED,
) -> dict[str, Any]:
    _validate_candidate(candidate_id)
    _validate_seed(seed)
    if fold not in FOLDS:
        raise ValueError(fold)
    with _stage1_binding(candidate_id) as (_, protocol_sha, config_sha, selection_sha):
        result = _stage1.canonical_reevaluate_fold(fold)
        receipt = _augment_fold_receipt(
            canonical_receipt_path(candidate_id, fold),
            candidate_id=candidate_id,
            protocol_sha=protocol_sha,
            selection_sha=selection_sha,
            config_sha=config_sha,
            schema=CANONICAL_SCHEMA,
        )
    return {
        **receipt,
        "receipt_sha256": _artifact(canonical_receipt_path(candidate_id, fold))["sha256"],
        "execution_status": result.get("status"),
    }


def _stage1_reference_raw_folds() -> dict[int, Mapping[str, np.ndarray]]:
    selection, _ = _selection()
    artifact = selection["stage1_reference"]["final_receipt"]
    _artifact(ROOT / artifact["path"], artifact["sha256"])
    receipt = _load_json(ROOT / artifact["path"])
    records: dict[int, Mapping[str, np.ndarray]] = {}
    for record in receipt["canonical_folds"]:
        fold = int(record["fold"])
        raw_artifact = record["raw_oof"]
        raw_path = ROOT / raw_artifact["path"]
        _artifact(raw_path, raw_artifact["sha256"])
        records[fold] = np.load(raw_path, allow_pickle=False)
    if set(records) != set(FOLDS):
        raise AssertionError("Stage1 c00@20 canonical fold coverage changed")
    return records


def _paired_candidate_vs_reference(
    candidate_raw: Mapping[int, Mapping[str, np.ndarray]],
    reference_raw: Mapping[int, Mapping[str, np.ndarray]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    data = _stage1._derm_data()
    candidate_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        candidate = candidate_raw[fold]
        reference = reference_raw[fold]
        candidate_indices = np.asarray(candidate["row_indices"], dtype="int64")
        reference_indices = np.asarray(reference["row_indices"], dtype="int64")
        if not np.array_equal(candidate_indices, reference_indices):
            raise AssertionError(f"candidate/c00@20 OOF indices differ in fold {fold}")
        candidate_rows.extend(_stage1._prediction_rows(
            data, candidate_indices, candidate["diagnosis_logits"],
            candidate["melanoma_logits"], candidate["final_criterion_logits"],
        ))
        reference_rows.extend(_stage1._prediction_rows(
            data, reference_indices, reference["diagnosis_logits"],
            reference["melanoma_logits"], reference["final_criterion_logits"],
        ))
    candidate_rows.sort(key=lambda row: str(row["id"]))
    reference_rows.sort(key=lambda row: str(row["id"]))
    candidate_ids_ = [str(row["id"]) for row in candidate_rows]
    reference_ids = [str(row["id"]) for row in reference_rows]
    if (
        candidate_ids_ != reference_ids
        or len(candidate_rows) != 616
        or len(set(candidate_ids_)) != 616
    ):
        raise AssertionError("paired candidate/c00@20 OOF case identity is not exactly 616")
    weights = _stage1.stratified_bootstrap_weights(
        _stage1.confirmation._derm_bootstrap_strata(reference_rows),
        n_boot=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    metrics = (
        "criterion_mean_auc", "primary", "balanced_accuracy", "melanoma_auc",
        "direct_rule_disagreement", "joint_correct_agreement",
    )
    candidate_samples = _stage1.ladder._metric_samples(
        candidate_rows, dataset="derm7pt", weights=weights
    )
    reference_samples = _stage1.ladder._metric_samples(
        reference_rows, dataset="derm7pt", weights=weights
    )
    paired = {}
    for metric in metrics:
        point = (
            _stage1.ladder._point_metric(candidate_rows, dataset="derm7pt", metric=metric)
            - _stage1.ladder._point_metric(reference_rows, dataset="derm7pt", metric=metric)
        )
        paired[metric] = _stage1.interval(
            candidate_samples[metric] - reference_samples[metric],
            point,
            n_boot=int(protocol["bootstrap"]["replicates"]),
        )
    return paired, candidate_rows, reference_rows


def finalize_candidate(candidate_id: str, *, seed: int = SEED) -> dict[str, Any]:
    """Aggregate five canonical folds and compare directly with immutable c00@20."""
    _validate_candidate(candidate_id)
    _validate_seed(seed)
    with _stage1_binding(candidate_id) as (
        protocol, protocol_sha, config_sha, selection_sha,
    ):
        _stage1.finalize()
        original_comparison = _load_json(candidate_comparison_path(candidate_id))
        original_receipt = _load_json(candidate_final_receipt_path(candidate_id))
        candidate_raw: dict[int, Mapping[str, np.ndarray]] = {}
        canonical_folds = []
        for fold in FOLDS:
            canonical, raw = _stage1._canonical_fold(protocol_sha, fold)
            candidate_raw[fold] = raw
            canonical_folds.append({
                "fold": fold,
                "receipt": _artifact(canonical_receipt_path(candidate_id, fold)),
                "raw_oof": copy.deepcopy(canonical["raw_oof"]),
            })
        paired, candidate_rows, reference_rows = _paired_candidate_vs_reference(
            candidate_raw, _stage1_reference_raw_folds(), protocol
        )
        candidate_targets, candidate_scores = _stage1.confirmation._derm_targets_scores(
            candidate_rows
        )
        reference_targets, reference_scores = _stage1.confirmation._derm_targets_scores(
            reference_rows
        )
        criterion_names = [
            _stage1.screen.CRITERION_SHORT[name] for name in _stage1.screen.CRITERIA
        ]
        from sklearn.metrics import roc_auc_score
        per_criterion = {
            name: {
                "c00_epoch20": float(
                    roc_auc_score(reference_targets[name], reference_scores[name])
                ),
                "candidate": float(
                    roc_auc_score(candidate_targets[name], candidate_scores[name])
                ),
            }
            for name in criterion_names
        }
        criterion_mean = float(np.mean([
            value["candidate"] for value in per_criterion.values()
        ]))
        checks = original_comparison["promotion_checks"]
        safe = bool(
            checks["protected_falls_within_0p02"]
            and checks["disagreement_increase_within_0p02"]
        )
        direct_primary = paired["criterion_mean_auc"]
        beats_reference = bool(
            direct_primary["estimate"] > 0.0 and direct_primary["ci_low"] > 0.0
        )
        comparison = {
            "schema_version": COMPARISON_SCHEMA,
            "status": "PASS",
            "seed": SEED,
            "protocol_sha256": protocol_sha,
            "selection_ledger_sha256": selection_sha,
            "t1_ledger_sha256": T1_LEDGER_SHA256,
            "t1_ranking_sha256": T1_RANKING_SHA256,
            "candidate_id": candidate_id,
            "candidate_config_sha256": config_sha,
            "reference_candidate_id": REFERENCE_CANDIDATE_ID,
            "n_cases": len(candidate_rows),
            "identical_oof_case_ids": True,
            "per_criterion_auroc": per_criterion,
            "criterion_mean": criterion_mean,
            "paired_candidate_minus_c00_epoch20": paired,
            "safety_checks_against_retained_base": {
                "protected_falls_within_0p02": bool(
                    checks["protected_falls_within_0p02"]
                ),
                "disagreement_increase_within_0p02": bool(
                    checks["disagreement_increase_within_0p02"]
                ),
            },
            "safety_decision": "SAFE" if safe else "UNSAFE",
            "beats_c00_epoch20_with_positive_95ci": beats_reference,
            "scientific_decision": (
                "CANDIDATE_BEATS_C00_EPOCH20"
                if safe and beats_reference else "RETAIN_C00_EPOCH20"
            ),
            "official_test_rows_loaded": 0,
            "external_test_rows_loaded": 0,
        }
        comparison_sha = _atomic_json(
            candidate_comparison_path(candidate_id), comparison
        )
        receipt = {
            **{
                key: value for key, value in original_receipt.items()
                if key not in {"fold", "source_checkpoint_sha256"}
            },
            "schema_version": FINAL_SCHEMA,
            "status": "PASS",
            "seed": SEED,
            "protocol_sha256": protocol_sha,
            "selection_ledger_sha256": selection_sha,
            "t1_ledger_sha256": T1_LEDGER_SHA256,
            "t1_ranking_sha256": T1_RANKING_SHA256,
            "candidate_id": candidate_id,
            "candidate_config_sha256": config_sha,
            "safety_decision": comparison["safety_decision"],
            "scientific_decision": comparison["scientific_decision"],
            "criterion_mean": criterion_mean,
            "comparison": {
                "path": _relative(candidate_comparison_path(candidate_id)),
                "sha256": comparison_sha,
            },
            "canonical_folds": canonical_folds,
            "official_test_accessed": False,
            "external_test_accessed": False,
        }
        receipt_sha = _atomic_json(candidate_final_receipt_path(candidate_id), receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def select_epoch30_candidates(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    safe = sorted(
        (row for row in rows if row.get("safety_decision") == "SAFE"),
        key=lambda row: (-float(row["criterion_mean"]), str(row["candidate_id"])),
    )
    if len(safe) < 2:
        raise RuntimeError("epoch30 requires two SAFE epoch20 configurations")
    return [str(row["candidate_id"]) for row in safe[:2]]


def _verified_candidate_final(
    candidate_id: str, *, protocol_sha: str, selection_sha: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = candidate_final_receipt_path(candidate_id)
    artifact = _artifact(path)
    receipt = _load_json(path)
    expected = (
        FINAL_SCHEMA, "PASS", SEED, protocol_sha, selection_sha,
        T1_LEDGER_SHA256, T1_RANKING_SHA256, candidate_id,
        candidate_config(candidate_id)["candidate_config_sha256"],
    )
    observed = (
        receipt.get("schema_version"), receipt.get("status"), receipt.get("seed"),
        receipt.get("protocol_sha256"), receipt.get("selection_ledger_sha256"),
        receipt.get("t1_ledger_sha256"), receipt.get("t1_ranking_sha256"),
        receipt.get("candidate_id"), receipt.get("candidate_config_sha256"),
    )
    if observed != expected:
        raise AssertionError(f"T2 final receipt identity mismatch: {candidate_id}")
    comparison = receipt["comparison"]
    _artifact(ROOT / comparison["path"], comparison["sha256"])
    folds = receipt.get("canonical_folds")
    if not isinstance(folds, list) or [row.get("fold") for row in folds] != list(FOLDS):
        raise AssertionError(f"T2 final receipt fold aggregation changed: {candidate_id}")
    for row in folds:
        _artifact(ROOT / row["receipt"]["path"], row["receipt"]["sha256"])
        _artifact(ROOT / row["raw_oof"]["path"], row["raw_oof"]["sha256"])
    return receipt, artifact


def rank_candidates(*, seed: int = SEED) -> dict[str, Any]:
    _validate_seed(seed)
    protocol, protocol_sha = load_protocol()
    selection, selection_sha = _selection()
    rows = []
    artifacts = []
    for candidate_id in ADVANCED_CANDIDATES:
        receipt, artifact = _verified_candidate_final(
            candidate_id, protocol_sha=protocol_sha, selection_sha=selection_sha
        )
        comparison = _load_json(ROOT / receipt["comparison"]["path"])
        rows.append({
            "candidate_id": candidate_id,
            "safety_decision": receipt["safety_decision"],
            "criterion_mean": float(receipt["criterion_mean"]),
            "paired_vs_c00_epoch20": copy.deepcopy(
                comparison["paired_candidate_minus_c00_epoch20"]["criterion_mean_auc"]
            ),
            "beats_c00_epoch20_with_positive_95ci": bool(
                comparison["beats_c00_epoch20_with_positive_95ci"]
            ),
        })
        artifacts.append({
            "candidate_id": candidate_id,
            "final_receipt": artifact,
        })
    reference_artifact = selection["stage1_reference"]["comparison"]
    _artifact(ROOT / reference_artifact["path"], reference_artifact["sha256"])
    reference_comparison = _load_json(ROOT / reference_artifact["path"])
    reference_mean = float(np.mean([
        float(value["localized"])
        for value in reference_comparison["per_criterion_auroc"].values()
    ]))
    rows.append({
        "candidate_id": REFERENCE_CANDIDATE_ID,
        "safety_decision": "SAFE",
        "criterion_mean": reference_mean,
        "paired_vs_c00_epoch20": {
            "estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0,
        },
        "beats_c00_epoch20_with_positive_95ci": False,
    })
    ordered = sorted(rows, key=lambda row: (
        row["safety_decision"] != "SAFE",
        -float(row["criterion_mean"]),
        str(row["candidate_id"]),
    ))
    selected = select_epoch30_candidates(ordered)
    eligible = [
        row for row in ordered
        if row["candidate_id"] != REFERENCE_CANDIDATE_ID
        and row["safety_decision"] == "SAFE"
        and row["beats_c00_epoch20_with_positive_95ci"]
    ]
    winner = eligible[0]["candidate_id"] if eligible else REFERENCE_CANDIDATE_ID
    ranking = {
        "schema_version": RANKING_SCHEMA,
        "status": "PASS",
        "seed": SEED,
        "epoch": 20,
        "protocol_sha256": protocol_sha,
        "selection_ledger_sha256": selection_sha,
        "t1_ledger_sha256": T1_LEDGER_SHA256,
        "t1_ranking_sha256": T1_RANKING_SHA256,
        "ranking_basis": (
            "SAFE first; descending seven-criterion mean AUROC; "
            "candidate_id tie-break"
        ),
        "ordered_candidates": ordered,
        "stage1_reference": copy.deepcopy(selection["stage1_reference"]),
        "selected_for_epoch30": selected,
        "winner": winner,
        "winner_rule": protocol["ranking"]["winner_rule"],
        "epoch30_launched": False,
        "official_test_accessed": False,
        "external_test_accessed": False,
    }
    ranking_sha = _atomic_json(RANKING_PATH, ranking)
    receipt = {
        "schema_version": RANKING_RECEIPT_SCHEMA,
        "status": "PASS",
        "seed": SEED,
        "protocol_sha256": protocol_sha,
        "selection_ledger_sha256": selection_sha,
        "t1_ledger_sha256": T1_LEDGER_SHA256,
        "t1_ranking_sha256": T1_RANKING_SHA256,
        "ranking": {"path": _relative(RANKING_PATH), "sha256": ranking_sha},
        "selected_for_epoch30": selected,
        "winner": winner,
        "candidate_final_receipts": artifacts,
        "stage1_reference": copy.deepcopy(selection["stage1_reference"]),
        "epoch30_launched": False,
        "official_test_accessed": False,
        "external_test_accessed": False,
    }
    receipt_sha = _atomic_json(RANKING_RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def controller_matrix() -> list[dict[str, Any]]:
    jobs = []
    for candidate_id in ADVANCED_CANDIDATES:
        jobs.extend({
            "stage": "train", "candidate_id": candidate_id, "fold": fold,
            "seed": SEED, "environment": "venv_dml",
        } for fold in FOLDS)
        jobs.extend({
            "stage": "reevaluate", "candidate_id": candidate_id, "fold": fold,
            "seed": SEED, "environment": "venv_p2",
        } for fold in FOLDS)
        jobs.append({
            "stage": "finalize_candidate", "candidate_id": candidate_id,
            "fold": None, "seed": SEED, "environment": "venv_p2",
        })
    jobs.append({
        "stage": "rank", "candidate_id": None, "fold": None,
        "seed": SEED, "environment": "venv_p2",
    })
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selection-only", action="store_true")
    mode.add_argument("--protocol-only", action="store_true")
    mode.add_argument("--train-fold", action="store_true")
    mode.add_argument("--reevaluate-fold", action="store_true")
    mode.add_argument("--finalize-candidate", action="store_true")
    mode.add_argument("--rank", action="store_true")
    parser.add_argument("--candidate", choices=ADVANCED_CANDIDATES)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    _validate_seed(args.seed)
    if args.selection_only:
        payload = freeze_selection_ledger()
    elif args.protocol_only:
        payload = build_protocol()
    elif args.train_fold:
        if args.candidate is None or args.fold is None:
            parser.error("--train-fold requires --candidate and --fold")
        payload = train_candidate_fold(
            args.candidate, args.fold, args.device, seed=args.seed
        )
    elif args.reevaluate_fold:
        if args.candidate is None or args.fold is None:
            parser.error("--reevaluate-fold requires --candidate and --fold")
        payload = reevaluate_candidate_fold(
            args.candidate, args.fold, seed=args.seed
        )
    elif args.finalize_candidate:
        if args.candidate is None:
            parser.error("--finalize-candidate requires --candidate")
        payload = finalize_candidate(args.candidate, seed=args.seed)
    else:
        payload = rank_candidates(seed=args.seed)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "generate_selection_ledger", "freeze_selection_ledger",
    "generate_protocol_spec", "build_protocol", "load_protocol",
    "train_candidate_fold", "reevaluate_candidate_fold", "finalize_candidate",
    "rank_candidates", "select_epoch30_candidates", "controller_matrix",
    "candidate_ids", "candidate_config", "training_receipt_path",
    "canonical_receipt_path", "candidate_comparison_path",
    "candidate_final_receipt_path", "validate_resume_identity",
]
