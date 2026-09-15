import numpy as np

from dermpact.core.eval import final_evaluation as final


def test_probability_mapping_preserves_mass():
    probability = np.arange(1, 10, dtype=float)
    probability /= probability.sum()
    mapped = final._map_prob(probability, final.DERM5_COLUMNS)
    assert mapped.shape == (5,)
    assert np.isclose(mapped.sum(), 1.0)
    assert np.isclose(mapped[-1], probability[[3, 4, 6, 7, 8]].sum())


def test_prediction_key_binds_route_image_and_protocol():
    case = {"dataset": "Derm7pt", "group_id": "case-1", "route": "full"}
    image = {"case_id": "case-1", "sha256": "a" * 64, "clinical_sha256": None}
    first = final._prediction_key("p1", case, image)
    assert first != final._prediction_key("p2", case, image)
    assert first != final._prediction_key("p1", {**case, "route": "diagnosis_only"}, image)


def test_final_access_log_is_hash_chained(monkeypatch, tmp_path):
    monkeypatch.setattr(final, "ACCESS_LOG", tmp_path / "access.jsonl")
    first = final._append_access("start", "a" * 64, "test start")
    second = final._append_access("resume", "a" * 64, "test resume")
    assert second["previous_sha256"] == first["entry_sha256"]
    assert final._verify_access_log() == second["entry_sha256"]
