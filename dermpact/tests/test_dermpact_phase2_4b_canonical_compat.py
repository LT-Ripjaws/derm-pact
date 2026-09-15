"""Regression test for the frozen Phase 2.4B JSON-order mismatch."""
from __future__ import annotations

import json
from pathlib import Path

from dermpact.eval import dermpact_phase2_4b_canonical_compat as compat
from dermpact.eval import dermpact_phase2_4b_experiment as experiment


def test_sorted_condition_json_is_repaired_without_semantic_change(tmp_path: Path) -> None:
    output = tmp_path / "study"
    fold_dir = output / "fold_0"
    fold_dir.mkdir(parents=True)
    protocol_sha = "a" * 64
    protocol_path = experiment.protocol_path(output)
    experiment._atomic_json(protocol_path, {"protocol_sha256": protocol_sha})
    conditions = {
        name: {"marker": name} for name in experiment.CONDITIONS
    }
    raw = {
        "fold": 0,
        "protocol_sha256": protocol_sha,
        "row_indices": [0],
        "conditions": conditions,
    }
    raw_path = experiment.canonical_raw_path(0, output)
    raw_sha = experiment._atomic_json(raw_path, raw)
    training_sha = experiment._atomic_json(
        experiment.training_receipt_path(0, output), {"status": "PASS"}
    )
    receipt = {
        "schema_version": experiment.CANONICAL_RECEIPT_SCHEMA,
        "status": "PASS",
        "seed": experiment.SEED,
        "fold": 0,
        "protocol_sha256": protocol_sha,
        "condition_order": list(experiment.CONDITIONS),
        "device": "cpu",
        "hard_scope": {
            "derm7pt_development_only": True,
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_or_final_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
        "rows": 1,
        "training_receipt_sha256": training_sha,
        "raw": {"path": str(raw_path.resolve()), "sha256": raw_sha},
    }
    receipt_path = experiment.canonical_receipt_path(0, output)
    experiment._atomic_json(receipt_path, receipt)
    assert not experiment._canonical_valid(receipt, 0, protocol_sha)

    before = json.loads(raw_path.read_text(encoding="utf-8"))
    result = compat.repair_materialized_fold(
        0,
        output_dir=output,
        protocol={"protocol_sha256": protocol_sha},
        protocol_sha256=protocol_sha,
    )
    repaired_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    after = json.loads(raw_path.read_text(encoding="utf-8"))

    assert experiment._canonical_valid(repaired_receipt, 0, protocol_sha)
    assert list(after["conditions"]) == list(experiment.CONDITIONS)
    assert before["conditions"] == after["conditions"]
    assert result["semantic_digest_before"] == result["semantic_digest_after"]
    assert result["scientific_values_changed"] is False
