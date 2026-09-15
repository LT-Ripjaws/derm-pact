from __future__ import annotations

from collections import Counter

import pytest

from dermpact.eval import dermpact_phase2_bcn_training_features as bcn


def test_frozen_bcn_selection_contains_training_rows_only():
    rows, positions = bcn._source_and_training_rows()
    split = bcn._load_json(bcn.BCN_SPLIT)["lesion_split"]

    assert len(rows) == bcn.EXPECTED_TRAIN_IMAGES
    assert len(positions) == bcn.EXPECTED_TRAIN_IMAGES
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)
    assert all(split[str(row["cluster_id"])] == "train" for row in rows)
    assert len({str(row["cluster_id"]) for row in rows}) == bcn.EXPECTED_TRAIN_LESIONS


def test_frozen_bcn_training_labels_match_expected_counts():
    rows, _ = bcn._source_and_training_rows()
    assert Counter(str(row["label"]) for row in rows) == Counter(
        {
            "akiec": 467,
            "bcc": 1706,
            "bkl": 650,
            "df": 60,
            "melanoma": 1703,
            "nevus": 2538,
            "scc": 209,
            "vasc": 72,
        }
    )


def test_source_position_and_identity_hashes_are_deterministic():
    rows_a, positions_a = bcn._source_and_training_rows()
    rows_b, positions_b = bcn._source_and_training_rows()
    assert bcn._canonical_hash(positions_a) == bcn._canonical_hash(positions_b)
    assert bcn._canonical_hash(bcn._identity(rows_a)) == bcn._canonical_hash(
        bcn._identity(rows_b)
    )


def test_unknown_frozen_split_fails_closed():
    with pytest.raises(ValueError, match="unknown frozen BCN split"):
        bcn._role_for_source_split("calibration-ish")


def test_protocol_hard_scope_and_row_contract(monkeypatch):
    captured = {}

    def fake_write_json(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return bcn._canonical_hash(payload)

    monkeypatch.setattr(bcn.screen, "_write_json", fake_write_json)
    protocol = bcn.build_protocol()

    assert captured["path"] == bcn.PROTOCOL_PATH
    assert protocol["row_contract"]["selected_training_rows"] == 7405
    assert protocol["row_contract"]["selected_training_lesions"] == 2143
    assert len(protocol["row_contract"]["selected_source_positions_sha256"]) == 64
    assert protocol["hard_scope"] == {
        "validation_calibration_rows_indexed_or_copied": 0,
        "final_test_rows_indexed_or_copied": 0,
        "predictions_generated": 0,
        "metrics_computed": 0,
        "source_matrix_hashing_is_integrity_only": True,
    }
