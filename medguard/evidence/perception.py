"""Verified adapter from the frozen development perception artifact to C1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .contract import (
    FindingOrigin,
    FindingState,
    FindingType,
    ImageFinding,
    PerceptionBinding,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = Path(
    "research/experiments/medguard_h2h_v1/perception/dermlip_v1/predictions.jsonl")
DEFAULT_RECEIPT = Path(
    "research/experiments/medguard_h2h_v1/perception/dermlip_v1/receipt.json")
DEFAULT_MANIFEST = Path("research/manifests/medguard_h2h_development_v1.json")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding_from_prediction_row(
    row: Mapping[str, Any],
    *,
    raw_line: bytes,
    artifact_id: str,
    artifact_sha256: str,
    model_id: str,
    class_order: Sequence[str],
    view_image_sha256s: Sequence[str],
) -> PerceptionBinding:
    """Build one truth-redacted binding from a frozen prediction row.

    Extracted so the development and sealed loaders construct bindings through the
    SAME code. A sealed partition scored by a subtly different construction than the
    development set would produce a difference that looks like a result.
    """
    if set(row).intersection({"label", "truth", "target", "ground_truth"}):
        raise ValueError("truth field appeared in the frozen perception output")
    record_sha = hashlib.sha256(raw_line).hexdigest()
    commitment = ImageFinding(
        finding_id="f_commitment",
        finding_type=FindingType.DIAGNOSTIC_COMMITMENT,
        concept_code=str(row["committed_diagnosis"]),
        state=FindingState.PRESENT,
        confidence=float(row["calibrated_confidence"]),
        origin=FindingOrigin.FROZEN_PERCEPTION,
        source_record_sha256=record_sha,
    )
    binding = PerceptionBinding(
        artifact_id=artifact_id,
        artifact_sha256=artifact_sha256,
        record_sha256=record_sha,
        model_id=str(model_id),
        class_order=tuple(str(item) for item in class_order),
        diagnosis_probabilities=tuple(float(item) for item in row["diagnosis_probabilities"]),
        view_image_sha256s=tuple(str(item) for item in view_image_sha256s),
        findings=(commitment,),
    )
    binding.validate()
    return binding


def load_development_perception_bindings(
    *, predictions_path: str | Path = DEFAULT_PREDICTIONS,
    receipt_path: str | Path = DEFAULT_RECEIPT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> Dict[str, PerceptionBinding]:
    """Load 160 truth-redacted commitments only after verifying frozen lineage."""
    predictions_file = _resolve(predictions_path)
    receipt_file = _resolve(receipt_path)
    manifest_file = _resolve(manifest_path)
    for path in (predictions_file, receipt_file, manifest_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar = receipt_file.with_name(receipt_file.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != _sha256(receipt_file):
        raise ValueError("perception receipt sidecar mismatch")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "medguard-h2h-perception-receipt-v1":
        raise ValueError("unsupported perception receipt schema")
    if (receipt.get("truth_fields_present") is not False
            or int(receipt.get("test_records_loaded", -1)) != 0
            or int(receipt.get("generation_calls", -1)) != 0):
        raise ValueError("perception receipt violates the development boundary")
    prediction_output = next(
        (row for row in receipt.get("outputs", [])
         if str(row.get("path", "")).replace("\\", "/").endswith("predictions.jsonl")), None)
    if prediction_output is None:
        raise ValueError("perception receipt does not bind predictions.jsonl")
    artifact_sha = _sha256(predictions_file)
    if artifact_sha != prediction_output.get("sha256"):
        raise ValueError("perception predictions hash mismatch")
    manifest_input = next(
        (row for row in receipt.get("inputs", [])
         if str(row.get("path", "")).replace("\\", "/").endswith(
             "medguard_h2h_development_v1.json")), None)
    if manifest_input is None or _sha256(manifest_file) != manifest_input.get("sha256"):
        raise ValueError("development manifest is not bound to the perception receipt")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    by_cluster = {}
    for row in manifest.get("records", []):
        if row.get("split") != "calibration":
            raise ValueError("development perception adapter encountered a non-calibration row")
        by_cluster.setdefault(str(row["cluster_id"]), []).append(row)
    for rows in by_cluster.values():
        rows.sort(key=lambda row: str(row["case_id"]))

    bindings: Dict[str, PerceptionBinding] = {}
    for raw_line in predictions_file.read_bytes().splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line.decode("utf-8"))
        if set(row).intersection({"label", "truth", "target", "ground_truth"}):
            raise ValueError("truth field appeared in the frozen perception output")
        cluster_id = str(row["cluster_id"])
        source_rows = by_cluster.get(cluster_id)
        if not source_rows:
            raise ValueError(f"perception record has no development-manifest cluster: {cluster_id}")
        expected_views = [str(item["case_id"]) for item in source_rows]
        if list(row["view_case_ids"]) != expected_views or int(row["view_count"]) != len(source_rows):
            raise ValueError(f"perception view lineage mismatch: {cluster_id}")
        binding = binding_from_prediction_row(
            row, raw_line=raw_line,
            artifact_id=str(DEFAULT_PREDICTIONS).replace("\\", "/"),
            artifact_sha256=artifact_sha,
            model_id=str(receipt["model_id"]),
            class_order=receipt["class_order"],
            view_image_sha256s=[str(item["image_sha256"]) for item in source_rows],
        )
        case_id = str(row["case_id"])
        if case_id in bindings:
            raise ValueError(f"duplicate frozen perception case: {case_id}")
        bindings[case_id] = binding
    if len(bindings) != int(receipt.get("lesion_outputs", -1)):
        raise ValueError("perception binding count disagrees with receipt")
    return bindings
