import json
from pathlib import Path

import numpy as np

from medguard.data.freeze_medguard_h2h_perception import (
    CLASS_ORDER,
    _aggregate_outputs,
    _validate_output_rows,
)
from medguard.data.manifest import CaseRecord, file_sha256


ROOT = Path(__file__).resolve().parents[2]


def test_perception_aggregation_keeps_all_views_and_redacts_truth():
    records = [
        CaseRecord("a", "lesion-1", "melanoma", "dermoscopy", "a.jpg", "fixture"),
        CaseRecord("b", "lesion-1", "melanoma", "dermoscopy", "b.jpg", "fixture"),
    ]
    probabilities = np.asarray([
        [0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.05],
        [0.02, 0.03, 0.05, 0.05, 0.75, 0.05, 0.05],
    ])
    rows = _aggregate_outputs(
        records, [0, 1], probabilities, np.asarray([0.8, 0.9]),
        CLASS_ORDER, 0.5,
    )
    validation = _validate_output_rows(rows, ["lesion-1"])
    assert len(rows) == 1
    assert rows[0]["committed_diagnosis"] == "melanoma"
    assert rows[0]["view_case_ids"] == ["a", "b"]
    assert rows[0]["view_count"] == 2
    assert rows[0]["actionable_prediction"] is True
    assert not {"label", "truth", "target", "malignant_true"}.intersection(rows[0])
    assert validation["truth_fields_present"] is False


def test_repository_perception_artifact_is_truth_redacted_and_receipted():
    directory = ROOT / "research/experiments/medguard_h2h_v1/perception/dermlip_v1"
    predictions = directory / "predictions.jsonl"
    receipt = directory / "receipt.json"
    audit = directory / "audit.json"
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 160
    assert len({row["cluster_id"] for row in rows}) == 160
    assert all(row["view_count"] == len(row["view_case_ids"]) for row in rows)
    assert not any({"label", "truth", "target", "malignant_true"}.intersection(row)
                   for row in rows)
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    audit_data = json.loads(audit.read_text(encoding="utf-8"))
    assert receipt_data["test_records_loaded"] == 0
    assert receipt_data["encoder_calls"] == 0
    assert receipt_data["generation_calls"] == 0
    assert receipt_data["truth_fields_present"] is False
    assert audit_data["valid"] is True
    assert audit_data["lesion_outputs"] == 160
    assert file_sha256(receipt) == receipt.with_suffix(
        receipt.suffix + ".sha256").read_text(encoding="ascii").strip()
