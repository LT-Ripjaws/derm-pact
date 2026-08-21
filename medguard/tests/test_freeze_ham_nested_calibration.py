from __future__ import annotations

import json
import tempfile
from pathlib import Path

from medguard.data.freeze_ham_nested_calibration_v1 import derive, load_assignment
from medguard.data.manifest import CaseRecord, freeze_manifest
from medguard.eval.derm_encoder_nested_tournament import _lesion_binary_scores, _lesion_rows


def _record(case_id: str, cluster_id: str, label: str, split: str) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        cluster_id=cluster_id,
        label=label,
        modality="dermoscopy",
        image_path=f"{case_id}.jpg",
        source="unit-test",
        split=split,
    )


def test_nested_split_is_label_stratified_deterministic_and_test_redacted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / "development.json"
        records = [_record("train", "train-cluster", "nevus", "train")]
        for label in ("melanoma", "nevus"):
            for index in range(3):
                records.append(_record(f"{label}-{index}", f"{label}-cluster-{index}", label, "calibration"))
        freeze_manifest(
            records,
            parent,
            metadata={"test_records_redacted": 10, "purpose": "test"},
        )
        output = root / "nested.json"
        first = derive(parent, output)
        assignments, metadata = load_assignment(output)
        second = derive(parent, output)

        assert first == second
        assert metadata["test_records_loaded"] == 0
        assert set(assignments) == {record.cluster_id for record in records if record.split == "calibration"}
        for label in ("melanoma", "nevus"):
            partitions = {assignments[f"{label}-cluster-{index}"] for index in range(3)}
            assert partitions == {"threshold_calibration", "node_validation"}
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "records" not in payload


def test_nested_split_rejects_parent_that_exposes_test_records():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / "bad-parent.json"
        freeze_manifest(
            [
                _record("train", "train-cluster", "nevus", "train"),
                _record("cal-1", "cal-1", "melanoma", "calibration"),
                _record("cal-2", "cal-2", "melanoma", "calibration"),
                _record("test", "test-cluster", "melanoma", "test"),
            ],
            parent,
            metadata={"test_records_redacted": 1},
        )
        try:
            derive(parent, root / "nested.json")
        except ValueError as error:
            assert "test-redacted" in str(error)
        else:  # pragma: no cover
            raise AssertionError("expected a test-exposing parent to be rejected")


def test_nested_tournament_weights_each_lesion_once_across_views():
    import numpy as np

    records = [
        _record("a-view-1", "lesion-a", "melanoma", "calibration"),
        _record("a-view-2", "lesion-a", "melanoma", "calibration"),
        _record("b-view-1", "lesion-b", "nevus", "calibration"),
    ]
    labels, scores = _lesion_binary_scores(records, [0, 1, 2], [0.2, 0.8, 0.1])
    assert labels == [True, False]
    assert np.allclose(scores, [0.5, 0.1])
    rows = _lesion_rows(
        records,
        [0, 1, 2],
        np.asarray([[0.8, 0.2], [0.2, 0.8], [0.9, 0.1]]),
        np.asarray([0.2, 0.8, 0.1]),
        ["nevus", "melanoma"],
    )
    assert len(rows) == 2
    assert rows[0]["cluster_id"] == "lesion-a"
    assert rows[0]["malignant_score"] == 0.5
