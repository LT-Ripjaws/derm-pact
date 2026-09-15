from __future__ import annotations

import json

import numpy as np
import pytest

from dermpact.eval import dermpact_phase2_4c_r_feature_cache as cache


def _materialize(tmp_path):
    rng = np.random.default_rng(41)
    present = np.ones((cache.N_CASES, 3), dtype=bool)
    present[:7, 1:] = False
    features = rng.normal(
        size=(cache.N_CASES, 3, 8, cache.FEATURE_DIM)
    ).astype(np.float32)
    features[~present] = 0.0
    arrays = {
        "row_ids": np.asarray([f"case-{i}" for i in range(cache.N_CASES)]),
        "folds": np.arange(cache.N_CASES, dtype=np.int64) % 5,
        "stream_features": features,
        "stream_present": present,
        "first_pass_logits": rng.normal(size=(cache.N_CASES, 8)).astype(np.float32),
        "uncertainty": rng.random(size=(cache.N_CASES, 8)).astype(np.float32),
        "semantic_node_embeddings": rng.normal(size=(8, 2, 512)).astype(np.float32),
        "criterion_targets": rng.integers(0, 2, size=(cache.N_CASES, 7)).astype(np.float32),
        "melanoma_targets": rng.integers(0, 2, size=cache.N_CASES).astype(np.float32),
        "diagnosis_targets": np.asarray(["nevus"] * cache.N_CASES),
        "melanocytic_mask": np.ones(cache.N_CASES, dtype=bool),
        "clinical_available": present[:, 1],
    }
    artifact = tmp_path / "feature_graph_oof_cache.npz"
    artifact_sha = cache._atomic_npz(artifact, arrays)
    manifest = {
        "format": cache.FORMAT,
        "status": "PASS",
        "artifact": {"path": str(artifact), "sha256": artifact_sha},
        "hard_scope": {"derm7pt_official_test_rows_loaded": 0},
    }
    cache._atomic_json(tmp_path / "feature_cache_manifest.json", manifest)
    return artifact, manifest


def test_validate_cache_accepts_complete_leakage_safe_artifact(tmp_path) -> None:
    _materialize(tmp_path)
    result = cache.validate_cache(output_dir=tmp_path)
    assert result["status"] == "PASS"
    assert result["rows"] == cache.N_CASES


def test_validate_cache_rejects_nonzero_official_counter(tmp_path) -> None:
    _, manifest = _materialize(tmp_path)
    manifest["hard_scope"]["derm7pt_official_test_rows_loaded"] = 1
    cache._atomic_json(tmp_path / "feature_cache_manifest.json", manifest)
    with pytest.raises(AssertionError, match="hard scope"):
        cache.validate_cache(output_dir=tmp_path)


def test_validate_cache_rejects_nonzero_absent_stream(tmp_path) -> None:
    artifact, manifest = _materialize(tmp_path)
    with np.load(artifact, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    arrays["stream_features"][0, 1, 0, 0] = 1.0
    manifest["artifact"]["sha256"] = cache._atomic_npz(artifact, arrays)
    cache._atomic_json(tmp_path / "feature_cache_manifest.json", manifest)
    with pytest.raises(AssertionError, match="absent streams"):
        cache.validate_cache(output_dir=tmp_path)


def test_validate_fold_cache_accepts_crash_safe_partial(tmp_path) -> None:
    rows = 11
    present = np.ones((rows, 3), dtype=bool)
    present[0, 1:] = False
    features = np.zeros((rows, 3, 8, cache.FEATURE_DIM), dtype=np.float32)
    arrays = {
        "row_indices": np.arange(rows, dtype=np.int64),
        "stream_features": features,
        "stream_present": present,
        "first_pass_logits": np.zeros((rows, 8), dtype=np.float32),
        "uncertainty": np.ones((rows, 8), dtype=np.float32),
    }
    artifact = cache.fold_artifact_path(0, tmp_path)
    artifact_sha = cache._atomic_npz(artifact, arrays)
    receipt = {
        "format": cache.FORMAT,
        "status": "PASS",
        "fold": 0,
        "rows": rows,
        "artifact_sha256": artifact_sha,
        "hard_scope": {"derm7pt_official_test_rows_loaded": 0},
    }
    cache._atomic_json(cache.fold_receipt_path(0, tmp_path), receipt)
    result = cache.validate_fold_cache(0, output_dir=tmp_path)
    assert result["status"] == "PASS"
    assert result["rows"] == rows
