"""The pre-sealed runtime lock: it must catch drift, and it must catch additions."""
from __future__ import annotations

import json

import pytest

from medguard.eval.runtime_lock import (
    RuntimeLockError,
    build_lock,
    verify_judge_specification,
    verify_lock,
)


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A miniature project tree with one file under each locked category."""
    import medguard.eval.runtime_lock as module

    (tmp_path / "medguard" / "evidence").mkdir(parents=True)
    (tmp_path / "research" / "manifests").mkdir(parents=True)
    (tmp_path / "research" / "cards").mkdir(parents=True)
    (tmp_path / "medguard" / "evidence" / "contract.py").write_text("x = 1\n")
    (tmp_path / "research" / "manifests" / "split.json").write_text('{"a": 1}')
    (tmp_path / "research" / "cards" / "c1.json").write_text('{"card": 1}')
    (tmp_path / "gate.json").write_text('{"tau": 0.45}')

    monkeypatch.setattr(module, "LOCKED_SOURCE_GLOBS", ("medguard/evidence/*.py",))
    monkeypatch.setattr(module, "LOCKED_ARTIFACTS", ("research/manifests/split.json",))
    monkeypatch.setattr(module, "LOCKED_CARDS", ("research/cards/c1.json",))
    return tmp_path


def _build(tree):
    return build_lock(
        tree, abstention_gate_path="gate.json",
        judge_specification_sha256="c521f6", spec_id="v6",
        identical_arms_for_noise_floor=["M", "A0"], created="2026-08-08")


def test_a_clean_tree_verifies(tree):
    lock = _build(tree)
    result = verify_lock(tree, lock)
    assert result["verified"] is True and result["source_files"] == 1


def test_an_edited_source_file_is_drift(tree):
    lock = _build(tree)
    (tree / "medguard" / "evidence" / "contract.py").write_text("x = 2\n")
    with pytest.raises(RuntimeLockError, match="changed"):
        verify_lock(tree, lock)


def test_a_NEW_file_under_a_locked_glob_is_also_drift(tree):
    """The subtle one: adding a module can change behaviour as much as editing one."""
    lock = _build(tree)
    (tree / "medguard" / "evidence" / "sneaky.py").write_text("y = 1\n")
    with pytest.raises(RuntimeLockError, match="added"):
        verify_lock(tree, lock)


def test_a_deleted_file_is_drift(tree):
    lock = _build(tree)
    (tree / "research" / "manifests" / "split.json").unlink()
    with pytest.raises(RuntimeLockError, match="missing"):
        verify_lock(tree, lock)


def test_a_changed_manifest_is_drift(tree):
    """A changed manifest silently changes the population under evaluation."""
    lock = _build(tree)
    (tree / "research" / "manifests" / "split.json").write_text('{"a": 2}')
    with pytest.raises(RuntimeLockError, match="changed"):
        verify_lock(tree, lock)


def test_a_changed_implementation_card_is_drift(tree):
    lock = _build(tree)
    (tree / "research" / "cards" / "c1.json").write_text('{"card": 2}')
    with pytest.raises(RuntimeLockError, match="changed"):
        verify_lock(tree, lock)


def test_a_refitted_abstention_gate_is_drift(tree):
    """Re-fitting the gate on sealed confidences is the failure the protocol exists to stop."""
    lock = _build(tree)
    (tree / "gate.json").write_text('{"tau": 0.61}')
    with pytest.raises(RuntimeLockError, match="changed"):
        verify_lock(tree, lock)


def test_a_missing_gate_refuses_to_lock_at_all(tree):
    (tree / "gate.json").unlink()
    with pytest.raises(RuntimeLockError, match="never fit one"):
        _build(tree)


def test_missing_artifacts_refuse_to_lock(tree):
    (tree / "research" / "manifests" / "split.json").unlink()
    with pytest.raises(RuntimeLockError, match="locked artifacts missing"):
        _build(tree)


def test_the_digest_summarises_the_whole_locked_set(tree):
    before = _build(tree)["source_tree"]["digest"]
    (tree / "medguard" / "evidence" / "contract.py").write_text("x = 3\n")
    assert _build(tree)["source_tree"]["digest"] != before


def test_a_changed_judge_specification_is_refused(tree):
    lock = _build(tree)
    verify_judge_specification(lock, "c521f6")          # unchanged: fine
    with pytest.raises(RuntimeLockError, match="judge specification changed"):
        verify_judge_specification(lock, "deadbeef")


def test_append_only_records_are_deliberately_not_locked(tree):
    """Locking the ledger would break the lock for reasons unrelated to the runtime."""
    lock = _build(tree)
    assert "research/RESULTS_LEDGER.json" in lock["not_locked"]
    assert "medguard/tests/" in lock["not_locked"]
    locked_paths = {row["path"] for row in lock["source_tree"]["files"]}
    assert not any(p.startswith("medguard/tests/") for p in locked_paths)


def test_the_lock_authorizes_nothing(tree):
    assert "authorizes nothing" in _build(tree)["authorization"]


def test_the_gate_carries_its_load_dont_refit_rule(tree):
    assert "Re-fitting" in _build(tree)["abstention_gate"]["rule"]


def test_sealed_run_must_remeasure_the_judge_floor(tree):
    scorers = _build(tree)["scorers"]
    assert "may not be carried over" in scorers["judge_nondeterminism"]
    assert scorers["identical_arms_for_noise_floor"] == ["M", "A0"]


# --- the real repository ------------------------------------------------------------
def test_the_repository_runtime_still_matches_its_lock():
    """The lock is live: this fails the moment a locked runtime file changes.

    That is the point. Between the lock and the sealed run nothing under
    medguard/{evidence,pipeline,eval,agents} may move, because a change there silently
    alters what the sealed numbers mean. Tests are excluded from the lock, so adding a
    test that catches a bug before the run is still allowed -- if that test forces a
    source fix, the lock must be rebuilt deliberately and the reason recorded.
    """
    from pathlib import Path

    from medguard.eval.ragas_judge import judge_specification_sha256

    root = Path(__file__).resolve().parents[2]
    lock_path = root / "research" / "RUNTIME_LOCK_V1.json"
    assert lock_path.is_file(), "the pre-sealed runtime lock is missing"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(root, lock)
    verify_judge_specification(lock, judge_specification_sha256())
