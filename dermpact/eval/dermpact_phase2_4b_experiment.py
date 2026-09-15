"""Crash-safe materialisation for the frozen Phase 2.4B paired-photo study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dermpact.perception import clinical_criterion_fusion as fusion_core
from dermpact.perception import clinical_tail_adapter as tail_core

from . import dermpact_phase2_4b_evaluation as evaluation
from . import dermpact_phase2_4b_training as training


ROOT = training.ROOT
SEED = training.SEED
FOLDS = training.FOLDS
CONDITIONS = evaluation.CONDITIONS
DEFAULT_OUTPUT_DIR = (
    ROOT / "research/experiments/dermpact_phase2_4b/paired_photo_v1"
)
PROTOCOL_PATH_NAME = "paired_photo_protocol.json"
FINAL_RAW_NAME = "canonical_oof_raw.json"
FINAL_EVALUATION_NAME = "paired_photo_evaluation.json"
FINAL_RECEIPT_NAME = "paired_photo_final_receipt.json"
TRAINING_RECEIPT_SCHEMA = "dermpact-phase2-4b-fold-training-receipt-v1"
CANONICAL_RECEIPT_SCHEMA = "dermpact-phase2-4b-canonical-fold-receipt-v1"
FINAL_RECEIPT_SCHEMA = "dermpact-phase2-4b-final-receipt-v1"
N_BOOT = 2000


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def _atomic_json(path: Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_bytes(_canonical_bytes(payload))
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = _sha256(path)
    sidecar = _sidecar(path)
    temporary_sidecar = Path(str(sidecar) + ".partial")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _verified_json(path: Path) -> tuple[dict[str, Any], str] | None:
    path = Path(path)
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        digest = _sha256(path)
        if sidecar.read_text(encoding="ascii").strip() != digest:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return (payload, digest) if isinstance(payload, dict) else None


def _artifact(path: Path, expected: str | None = None) -> dict[str, str]:
    path = Path(path).resolve()
    if not path.is_file() or not _sidecar(path).is_file():
        raise FileNotFoundError(f"artifact or SHA-256 sidecar missing: {path}")
    observed = _sha256(path)
    if _sidecar(path).read_text(encoding="ascii").strip() != observed:
        raise AssertionError(f"SHA-256 sidecar mismatch: {path}")
    if expected is not None and observed != expected:
        raise AssertionError(f"artifact SHA-256 mismatch: {path}")
    return {"path": str(path), "sha256": observed}


def _bound_artifact(value: Any, *, require_sidecar: bool = True) -> Path:
    if not isinstance(value, Mapping):
        raise TypeError("artifact binding must be an object")
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    if require_sidecar:
        _artifact(path, expected)
    elif not path.is_file() or _sha256(path) != expected:
        raise AssertionError(f"bound source SHA-256 mismatch: {path}")
    return path


def protocol_path(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(output_dir) / PROTOCOL_PATH_NAME


def training_receipt_path(fold: int, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(output_dir) / f"fold_{fold}" / "training_receipt.json"


def canonical_receipt_path(fold: int, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(output_dir) / f"fold_{fold}" / "canonical_receipt.json"


def canonical_raw_path(fold: int, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(output_dir) / f"fold_{fold}" / "canonical_raw.json"


def _source_bindings(controller_source: Path | None) -> dict[str, dict[str, str]]:
    paths = {
        "materializer": Path(__file__).resolve(),
        "cli": Path(__file__).resolve(),
        "training": Path(training.__file__).resolve(),
        "evaluation": Path(evaluation.__file__).resolve(),
        "paired_model": (
            ROOT
            / Path(training.Phase24BPairedCriterionModel.__module__.replace(".", "/") + ".py")
        ).resolve(),
        "cross_modal_fusion": Path(fusion_core.__file__).resolve(),
        "clinical_tail_adapter": Path(tail_core.__file__).resolve(),
    }
    if controller_source is not None:
        paths["controller"] = Path(controller_source).resolve()
    return {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }


def generate_protocol(
    *,
    sharp_protocol_path: Path = training.SHARP_PROTOCOL_PATH,
    cache_dir: Path = training.CLINICAL_CACHE_DIR,
    controller_source: Path | None = None,
) -> dict[str, Any]:
    """Generate, but do not write, the complete frozen protocol payload."""
    base = training.generate_protocol_spec(
        sharp_protocol_path=sharp_protocol_path, cache_dir=cache_dir
    )
    if base.get("model", {}).get("trainable_scope") != (
        "clinical_tail.adapters and cross_modal_fusion only"
    ):
        raise AssertionError("Phase 2.4B protocol does not describe the production cross-modal state surface")
    base.pop("protocol_sha256", None)
    base.pop("signature_payload", None)
    base["status"] = "FROZEN"
    base["canonical_evaluation"] = {
        "device": "cpu",
        "conditions": list(CONDITIONS),
        "wrong_pair": {
            "algorithm": "cyclic_next_available_within_validation_fold",
            "seed": SEED,
            "random_draws": 0,
        },
        "duplicate_dermoscopy": "dermoscopic_tokens_used_as_clinical_tokens",
        "forced_missing": "all clinical-present masks false",
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "raw_format": "JSON-safe arrays in canonical row order",
        "gates": {
            "source": "dermpact.eval.dermpact_phase2_4b_evaluation.evaluate_phase24b",
            "enforced_without_override": True,
        },
    }
    base["orchestration"] = {
        "matrix": (
            [{"stage": "protocol", "fold": None}]
            + [{"stage": "train", "fold": fold} for fold in FOLDS]
            + [{"stage": "canonical_evaluate", "fold": fold} for fold in FOLDS]
            + [{"stage": "finalize", "fold": None}]
        ),
        "completed_stage_reuse": "SHA-256 and semantic receipt validation required",
    }
    base["artifacts"]["controller_sources"] = _source_bindings(controller_source)
    digest = hashlib.sha256(_canonical_bytes(base)).hexdigest()
    return {
        **base,
        "protocol_sha256": digest,
        "signature_payload": {
            "schema_version": training.PROTOCOL_SCHEMA,
            "artifact_sha256": digest,
        },
    }


def freeze_protocol(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sharp_protocol_path: Path = training.SHARP_PROTOCOL_PATH,
    cache_dir: Path = training.CLINICAL_CACHE_DIR,
    controller_source: Path | None = None,
) -> dict[str, Any]:
    payload = generate_protocol(
        sharp_protocol_path=sharp_protocol_path,
        cache_dir=cache_dir,
        controller_source=controller_source,
    )
    training.validate_protocol_spec(payload)
    path = protocol_path(output_dir)
    existing = _verified_json(path)
    if existing is not None:
        if existing[0] != payload:
            raise AssertionError("a different Phase 2.4B protocol is already frozen")
        return {"execution_status": "REUSED_VERIFIED", **existing[0]}
    digest = _atomic_json(path, payload)
    return {"execution_status": "EXECUTED", **payload, "artifact_sha256": digest}


def load_protocol(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[dict[str, Any], str]:
    record = _verified_json(protocol_path(output_dir))
    if record is None:
        raise RuntimeError("Phase 2.4B protocol is absent or not SHA-verified")
    payload, artifact_sha = record
    protocol_sha = training.validate_protocol_spec(payload)
    for value in payload.get("artifacts", {}).get("controller_sources", {}).values():
        _bound_artifact(value, require_sidecar=False)
    bindings, cache = _live_inputs(payload)
    for binding in bindings:
        training.validate_protocol_spec(payload, binding=binding, cache=cache)
    return payload, protocol_sha


def _live_inputs(
    protocol: Mapping[str, Any],
) -> tuple[list[training.SharpFoldBinding], training.ClinicalCache]:
    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("protocol lacks artifact bindings")
    sharp_rows = artifacts.get("sharp_folds")
    cache_binding = artifacts.get("clinical_cache")
    if not isinstance(sharp_rows, list) or len(sharp_rows) != len(FOLDS):
        raise ValueError("protocol lacks five sharp-fold bindings")
    if not isinstance(cache_binding, Mapping):
        raise ValueError("protocol lacks clinical-cache binding")
    protocol_binding = sharp_rows[0].get("protocol", {})
    sharp_protocol_path = _bound_artifact(protocol_binding)
    manifest_path = _bound_artifact(cache_binding.get("manifest"))
    bindings = [
        training.resolve_sharp_fold(fold, protocol_path=sharp_protocol_path)
        for fold in FOLDS
    ]
    cache = training.load_clinical_cache(cache_dir=manifest_path.parent)
    return bindings, cache


def _hard_scope_valid(value: Any) -> bool:
    return value == {
        "derm7pt_development_only": True,
        "derm7pt_official_test_rows_loaded": 0,
        "ham_calibration_or_final_rows_loaded": 0,
        "bcn_validation_or_final_rows_loaded": 0,
        "ph2_rows_loaded": 0,
    }


def _training_valid(
    payload: Mapping[str, Any], fold: int, protocol_sha: str,
    protocol: Mapping[str, Any] | None = None,
) -> bool:
    if (
        payload.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("seed") != SEED
        or payload.get("fold") != fold
        or payload.get("protocol_sha256") != protocol_sha
        or payload.get("completed_epochs") != training.EPOCHS
        or payload.get("effective_batch_size") != training.BATCH_SIZE
        or payload.get("training_device") != "cpu"
        or not _hard_scope_valid(payload.get("hard_scope"))
    ):
        return False
    if protocol is not None:
        artifacts = protocol.get("artifacts", {})
        try:
            sharp = artifacts["sharp_folds"][fold]
            cache_manifest_sha = artifacts["clinical_cache"]["manifest"]["sha256"]
        except (KeyError, IndexError, TypeError):
            return False
        if any(
            payload.get(name) != expected
            for name, expected in (
                ("sharp_protocol_sha256", sharp["protocol"]["sha256"]),
                ("sharp_training_receipt_sha256", sharp["training_receipt"]["sha256"]),
                ("sharp_residual_checkpoint_sha256", sharp["sharp_residual_checkpoint"]["sha256"]),
                ("clinical_cache_manifest_sha256", cache_manifest_sha),
            )
        ):
            return False
    try:
        _bound_artifact(payload.get("checkpoint"))
        _bound_artifact(payload.get("resume"))
    except (OSError, TypeError, AssertionError, ValueError):
        return False
    # The signed core result is retained verbatim under ``training_result``.
    core = payload.get("training_result")
    if not isinstance(core, Mapping):
        return False
    unsigned = dict(core)
    recorded = unsigned.pop("result_sha256", None)
    unsigned.pop("signature_payload", None)
    return recorded == training._canonical_hash(unsigned)


def train_fold(fold: int, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    protocol, protocol_sha = load_protocol(output_dir)
    receipt_path = training_receipt_path(fold, output_dir)
    existing = _verified_json(receipt_path)
    if existing is not None and _training_valid(existing[0], fold, protocol_sha, protocol):
        return {"execution_status": "REUSED_VERIFIED", **existing[0]}
    bindings, cache = _live_inputs(protocol)
    result = training.train_fold(
        fold, output_dir=Path(output_dir), protocol_spec=protocol,
        sharp_protocol_path=bindings[fold].protocol_path,
        cache_dir=cache.cache_dir,
    )
    if result.get("status") != "PASS":
        raise RuntimeError(f"fold {fold} did not complete all {training.EPOCHS} epochs")
    receipt = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "status": "PASS",
        "seed": SEED,
        "fold": fold,
        "protocol_sha256": protocol_sha,
        "completed_epochs": result["completed_epochs"],
        "effective_batch_size": result["effective_batch_size"],
        "training_device": result["training_device"],
        "sharp_protocol_sha256": result["sharp_protocol_sha256"],
        "sharp_training_receipt_sha256": result["sharp_training_receipt_sha256"],
        "sharp_residual_checkpoint_sha256": result["sharp_residual_checkpoint_sha256"],
        "clinical_cache_manifest_sha256": result["clinical_cache_manifest_sha256"],
        "checkpoint": result["checkpoint"],
        "resume": result["resume"],
        "hard_scope": result["hard_scope"],
        "training_result": result,
    }
    _atomic_json(receipt_path, receipt)
    if not _training_valid(receipt, fold, protocol_sha, protocol):
        raise AssertionError("materialized training receipt failed validation")
    return {"execution_status": "EXECUTED", **receipt}


def _load_clinical_state(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    training.load_mutable_clinical_state(model, state)


def _case_ids(data: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [str(row["group_id"]) for row in data["derm_rows"]], dtype=object
    )


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype="float32").copy())


def _json_array(value: np.ndarray) -> list[Any]:
    return np.asarray(value).tolist()


def _canonical_valid(
    payload: Mapping[str, Any], fold: int, protocol_sha: str,
    output_dir: Path | None = None,
) -> bool:
    if (
        payload.get("schema_version") != CANONICAL_RECEIPT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("seed") != SEED
        or payload.get("fold") != fold
        or payload.get("protocol_sha256") != protocol_sha
        or payload.get("condition_order") != list(CONDITIONS)
        or payload.get("device") != "cpu"
        or not _hard_scope_valid(payload.get("hard_scope"))
    ):
        return False
    try:
        raw_path = _bound_artifact(payload.get("raw"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, AssertionError, ValueError, json.JSONDecodeError):
        return False
    if output_dir is not None:
        training_record = _verified_json(training_receipt_path(fold, output_dir))
        if (
            training_record is None
            or payload.get("training_receipt_sha256") != training_record[1]
        ):
            return False
    return bool(
        isinstance(raw, dict)
        and raw.get("fold") == fold
        and raw.get("protocol_sha256") == protocol_sha
        and list(raw.get("conditions", {})) == list(CONDITIONS)
        and payload.get("rows") == len(raw.get("row_indices", []))
    )


def canonical_evaluate_fold(
    fold: int, *, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    """Materialise all five canonical validation-fold conditions on CPU."""
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    protocol, protocol_sha = load_protocol(output_dir)
    receipt_path = canonical_receipt_path(fold, output_dir)
    existing = _verified_json(receipt_path)
    if existing is not None and _canonical_valid(
        existing[0], fold, protocol_sha, Path(output_dir)
    ):
        return {"execution_status": "REUSED_VERIFIED", **existing[0]}
    trained = _verified_json(training_receipt_path(fold, output_dir))
    if trained is None or not _training_valid(trained[0], fold, protocol_sha, protocol):
        raise RuntimeError(f"fold {fold} lacks a verified training receipt")

    bindings, cache = _live_inputs(protocol)
    binding = bindings[fold]
    training.validate_protocol_spec(protocol, binding=binding, cache=cache)
    data = training.load_training_data(cache)
    raw_data = training.sharp_factor._base._derm_data()
    indices = np.flatnonzero(np.asarray(data.folds) == fold)
    available_positions = np.flatnonzero(np.asarray(data.clinical_available[indices]))
    if len(available_positions) < 2:
        raise AssertionError("wrong-pair control requires two available fold rows")
    wrong_source = indices.copy()
    wrong_source[available_positions] = np.roll(indices[available_positions], -1)

    model = training.construct_paired_model(binding)
    checkpoint = _bound_artifact(trained[0]["checkpoint"])
    _load_clinical_state(model, checkpoint)
    model.eval().to(torch.device("cpu"))
    arrays: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in CONDITIONS
    }
    with torch.inference_mode():
        for start in range(0, len(indices), training.BATCH_SIZE):
            selection = indices[start : start + training.BATCH_SIZE]
            derm = _tensor(data.derm_tokens[selection])
            clinical = _tensor(data.clinical_tokens[selection])
            present = torch.from_numpy(
                np.asarray(data.clinical_available[selection], dtype=bool).copy()
            )
            local_wrong = wrong_source[start : start + len(selection)]
            inputs = {
                "true_pair": (clinical, present),
                "forced_missing": (torch.zeros_like(clinical), torch.zeros_like(present)),
                "wrong_pair": (_tensor(data.clinical_tokens[local_wrong]), present),
                "duplicate_dermoscopy": (derm, present),
            }
            outputs = {name: model(derm, *values) for name, values in inputs.items()}
            reference = outputs["true_pair"]["baseline_outputs"]
            arrays["baseline"]["diagnosis_logits"].append(reference["diagnosis"].cpu().numpy())
            arrays["baseline"]["melanoma_logits"].append(reference["melanoma"].cpu().numpy())
            arrays["baseline"]["criterion_logits"].append(reference["criteria"].cpu().numpy())
            arrays["baseline"]["clinical_residual"].append(np.zeros((len(selection), 7), dtype="float32"))
            for name, output in outputs.items():
                if not torch.equal(output["diagnosis"], reference["diagnosis"]):
                    raise AssertionError(f"{name} changed protected diagnosis logits")
                if not torch.equal(output["melanoma"], reference["melanoma"]):
                    raise AssertionError(f"{name} changed protected melanoma logits")
                arrays[name]["diagnosis_logits"].append(output["diagnosis"].cpu().numpy())
                arrays[name]["melanoma_logits"].append(output["melanoma"].cpu().numpy())
                arrays[name]["criterion_logits"].append(output["criteria"].cpu().numpy())
                arrays[name]["clinical_residual"].append(output["clinical_residual"].cpu().numpy())

    ids = _case_ids(raw_data)[indices]
    diagnosis_names = np.asarray(
        [
            training.sharp_factor._base.screen.GLOBAL_CLASSES[int(value)]
            for value in raw_data["derm_global"]
        ],
        dtype=object,
    )
    melanocytic = np.isin(diagnosis_names, ("melanoma", "nevus"))
    conditions = {}
    for name in CONDITIONS:
        combined = {key: np.concatenate(value) for key, value in arrays[name].items()}
        conditions[name] = {"ids": _json_array(ids), **{
            key: _json_array(value) for key, value in combined.items()
        }}
    raw = {
        "schema_version": "dermpact-phase2-4b-canonical-fold-raw-v1",
        "protocol_sha256": protocol_sha,
        "fold": fold,
        "row_indices": _json_array(indices.astype("int64")),
        "truth": {
            "ids": _json_array(ids),
            "diagnosis_targets": _json_array(diagnosis_names[indices]),
            "melanoma_targets": _json_array(np.asarray(raw_data["derm_melanoma"])[indices]),
            "criterion_targets": _json_array(np.asarray(data.criterion_targets)[indices]),
            "clinical_available": _json_array(np.asarray(data.clinical_available)[indices]),
            "melanocytic_mask": _json_array(melanocytic[indices]),
        },
        "conditions": conditions,
    }
    raw_path = canonical_raw_path(fold, output_dir)
    raw_sha = _atomic_json(raw_path, raw)
    receipt = {
        "schema_version": CANONICAL_RECEIPT_SCHEMA,
        "status": "PASS",
        "seed": SEED,
        "fold": fold,
        "protocol_sha256": protocol_sha,
        "training_receipt_sha256": trained[1],
        "sharp_protocol_sha256": binding.protocol_sha256,
        "sharp_training_receipt_sha256": binding.training_receipt_sha256,
        "sharp_residual_checkpoint_sha256": binding.residual_checkpoint_sha256,
        "clinical_cache_manifest_sha256": cache.manifest_sha256,
        "clinical_checkpoint_sha256": trained[0]["checkpoint"]["sha256"],
        "condition_order": list(CONDITIONS),
        "rows": len(indices),
        "raw": {"path": str(raw_path.resolve()), "sha256": raw_sha},
        "device": "cpu",
        "hard_scope": dict(protocol["hard_scope"]),
    }
    _atomic_json(receipt_path, receipt)
    if not _canonical_valid(receipt, fold, protocol_sha, Path(output_dir)):
        raise AssertionError("materialized canonical receipt failed validation")
    return {"execution_status": "EXECUTED", **receipt}


def _merged_raw(output_dir: Path, protocol_sha: str) -> dict[str, Any]:
    rows: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    for fold in FOLDS:
        record = _verified_json(canonical_receipt_path(fold, output_dir))
        if record is None or not _canonical_valid(
            record[0], fold, protocol_sha, Path(output_dir)
        ):
            raise RuntimeError(f"fold {fold} lacks verified canonical evidence")
        raw_path = _bound_artifact(record[0]["raw"])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        for offset, index in enumerate(raw["row_indices"]):
            rows.append((int(index), raw["truth"], {name: raw["conditions"][name] for name in CONDITIONS}, offset))
    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(evaluation.N_CASES)):
        raise AssertionError("canonical folds are not a disjoint exhaustive 0..615 partition")
    truth_keys = (
        "ids", "diagnosis_targets", "melanoma_targets", "criterion_targets",
        "clinical_available", "melanocytic_mask",
    )
    truth = {key: [row[1][key][row[3]] for row in rows] for key in truth_keys}
    conditions = {
        name: {
            key: [row[2][name][key][row[3]] for row in rows]
            for key in ("ids", "diagnosis_logits", "melanoma_logits", "criterion_logits", "clinical_residual")
        }
        for name in CONDITIONS
    }
    return {"protocol_sha256": protocol_sha, "row_indices": list(range(evaluation.N_CASES)), "truth": truth, "conditions": conditions}


def _final_valid(
    payload: Mapping[str, Any], protocol_sha: str,
    output_dir: Path | None = None,
) -> bool:
    if (
        payload.get("schema_version") != FINAL_RECEIPT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("seed") != SEED
        or payload.get("protocol_sha256") != protocol_sha
        or payload.get("official_test_rows_loaded") != 0
        or payload.get("final_dataset_rows_loaded") != 0
    ):
        return False
    try:
        raw_path = _bound_artifact(payload.get("raw"))
        result_path = _bound_artifact(payload.get("evaluation"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, AssertionError, ValueError, json.JSONDecodeError):
        return False
    if output_dir is not None:
        rows = payload.get("canonical_fold_receipts")
        if not isinstance(rows, list) or [row.get("fold") for row in rows] != list(FOLDS):
            return False
        for fold, row in enumerate(rows):
            record = _verified_json(canonical_receipt_path(fold, output_dir))
            if record is None or row.get("sha256") != record[1]:
                return False
    return bool(
        raw.get("protocol_sha256") == protocol_sha
        and result.get("contract") == "dermpact-phase2-4b-pure-evaluation-v1"
        and result.get("validation", {}).get("status") == "PASS"
        and result.get("promotion", {}).get("decision") == payload.get("decision")
    )


def finalize(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(output_dir)
    final_path = Path(output_dir) / FINAL_RECEIPT_NAME
    existing = _verified_json(final_path)
    if existing is not None and _final_valid(
        existing[0], protocol_sha, Path(output_dir)
    ):
        return {"execution_status": "REUSED_VERIFIED", **existing[0]}
    raw = _merged_raw(Path(output_dir), protocol_sha)
    truth = raw["truth"]
    weights_all, weights_available = evaluation.fixed_bootstrap_weights(
        truth["diagnosis_targets"], truth["criterion_targets"],
        truth["clinical_available"], n_boot=N_BOOT, seed=SEED,
    )
    result = evaluation.evaluate_phase24b(
        truth=truth,
        bootstrap_weights_all=weights_all,
        bootstrap_weights_available=weights_available,
        **raw["conditions"],
    )
    expected_gates = {
        "row_identity_and_availability", "diagnosis_and_melanoma_exact",
        "forced_missing_fallback_max_le_1e-6", "clinical_residual_max_le_1",
        "missing_path_clinical_residual_exact_zero",
        "all_case_criterion_mean_ci_low_gt_0",
        "available_609_criterion_mean_ci_low_gt_0",
        "true_pair_beats_wrong_pair_control",
        "true_pair_beats_duplicate_dermoscopy_control",
        "direct_rule_disagreement_delta_le_0p02",
        "joint_correct_agreement_delta_ge_minus_0p02",
    }
    gates = result.get("promotion", {}).get("gates", {})
    if set(gates) != expected_gates:
        raise AssertionError("Phase 2.4B evaluation gate set changed")
    raw_path = Path(output_dir) / FINAL_RAW_NAME
    raw_sha = _atomic_json(raw_path, raw)
    evaluation_path = Path(output_dir) / FINAL_EVALUATION_NAME
    evaluation_sha = _atomic_json(evaluation_path, result)
    fold_receipts = []
    for fold in FOLDS:
        record = _verified_json(canonical_receipt_path(fold, output_dir))
        assert record is not None
        fold_receipts.append({"fold": fold, "path": str(canonical_receipt_path(fold, output_dir).resolve()), "sha256": record[1]})
    receipt = {
        "schema_version": FINAL_RECEIPT_SCHEMA,
        "status": "PASS",
        "seed": SEED,
        "protocol_sha256": protocol_sha,
        "condition_order": list(CONDITIONS),
        "canonical_fold_receipts": fold_receipts,
        "raw": {"path": str(raw_path.resolve()), "sha256": raw_sha},
        "evaluation": {"path": str(evaluation_path.resolve()), "sha256": evaluation_sha},
        "decision": result["promotion"]["decision"],
        "promoted": result["promotion"]["promoted"],
        "official_test_rows_loaded": 0,
        "final_dataset_rows_loaded": 0,
        "hard_scope": dict(protocol["hard_scope"]),
    }
    _atomic_json(final_path, receipt)
    if not _final_valid(receipt, protocol_sha, Path(output_dir)):
        raise AssertionError("materialized final receipt failed validation")
    return {"execution_status": "EXECUTED", **receipt}


def verify_stage(
    stage: str, *, fold: int | None = None, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[dict[str, str]] | None:
    """Return verified stage artifacts, or ``None`` when reuse is unauthorized."""
    try:
        protocol, protocol_sha = load_protocol(output_dir)
        if stage == "protocol":
            return [_artifact(protocol_path(output_dir))]
        if stage == "train" and fold in FOLDS:
            record = _verified_json(training_receipt_path(int(fold), output_dir))
            if record is not None and _training_valid(record[0], int(fold), protocol_sha, protocol):
                return [_artifact(training_receipt_path(int(fold), output_dir))]
        if stage == "canonical_evaluate" and fold in FOLDS:
            record = _verified_json(canonical_receipt_path(int(fold), output_dir))
            if record is not None and _canonical_valid(
                record[0], int(fold), protocol_sha, Path(output_dir)
            ):
                return [_artifact(canonical_receipt_path(int(fold), output_dir)), record[0]["raw"]]
        if stage == "finalize":
            record = _verified_json(Path(output_dir) / FINAL_RECEIPT_NAME)
            if record is not None and _final_valid(
                record[0], protocol_sha, Path(output_dir)
            ):
                return [_artifact(Path(output_dir) / FINAL_RECEIPT_NAME), record[0]["raw"], record[0]["evaluation"]]
    except (OSError, TypeError, AssertionError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--protocol", action="store_true")
    actions.add_argument("--train-fold", action="store_true")
    actions.add_argument("--canonical-evaluate-fold", action="store_true")
    actions.add_argument("--finalize", action="store_true")
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--controller-source", type=Path)
    args = parser.parse_args(argv)
    if (args.train_fold or args.canonical_evaluate_fold) and args.fold is None:
        parser.error("fold action requires --fold")
    if args.protocol:
        payload = freeze_protocol(output_dir=args.output_dir, controller_source=args.controller_source)
    elif args.train_fold:
        payload = train_fold(args.fold, output_dir=args.output_dir)
    elif args.canonical_evaluate_fold:
        payload = canonical_evaluate_fold(args.fold, output_dir=args.output_dir)
    else:
        payload = finalize(output_dir=args.output_dir)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "canonical_evaluate_fold", "canonical_raw_path", "canonical_receipt_path",
    "finalize", "freeze_protocol", "generate_protocol", "load_protocol",
    "main", "protocol_path", "train_fold", "training_receipt_path",
    "verify_stage",
]
