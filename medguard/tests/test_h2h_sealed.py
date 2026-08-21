"""Sealed loaders: they must refuse to run without a genuine sealed perception freeze."""
from __future__ import annotations

import json

import pytest

from medguard.eval import h2h_sealed
from medguard.eval.h2h_sealed import (
    EXPECTED_IMAGES,
    EXPECTED_LESIONS,
    SealedArtifactError,
    load_sealed_run_cases,
    load_sealed_truth,
    sealed_partition_summary,
)
from medguard.eval.sealed_access import purposes_used, read_log


def test_the_partition_can_be_sized_without_opening_sealed_access(tmp_path):
    """Planning a run needs its size; counting rows is not evaluating them."""
    before = read_log()
    summary = sealed_partition_summary()
    assert summary == {"images": EXPECTED_IMAGES, "lesions": EXPECTED_LESIONS}
    assert read_log() == before, "sizing the partition must not log an access"


def test_the_registered_partition_shape_is_what_the_protocol_declared():
    """1,991 images over 1,495 lesions -- the preregistered sealed test."""
    assert (EXPECTED_IMAGES, EXPECTED_LESIONS) == (1991, 1495)
    assert sealed_partition_summary()["lesions"] == 1495


def test_loading_cases_without_a_sealed_perception_artifact_refuses(monkeypatch, tmp_path):
    """The dangerous failure is falling back to development perception silently."""
    monkeypatch.setattr(h2h_sealed, "SEALED_PREDICTIONS", tmp_path / "absent.jsonl")
    with pytest.raises(SealedArtifactError, match="Freeze it first"):
        load_sealed_run_cases(detail="attempt without an artifact")


def test_a_development_perception_artifact_is_rejected_as_sealed(monkeypatch, tmp_path):
    """dev receipts assert test_records_loaded == 0; sealed must assert the opposite."""
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"test_records_loaded": 0, "truth_fields_present": False}),
                       encoding="utf-8")
    monkeypatch.setattr(h2h_sealed, "SEALED_PREDICTIONS", predictions)
    monkeypatch.setattr(h2h_sealed, "SEALED_RECEIPT", receipt)
    with pytest.raises(SealedArtifactError, match="development artifact, not a sealed one"):
        load_sealed_run_cases(detail="dev artifact passed off as sealed")


def test_a_perception_artifact_carrying_truth_is_rejected(monkeypatch, tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"test_records_loaded": 1991, "truth_fields_present": True}),
                       encoding="utf-8")
    monkeypatch.setattr(h2h_sealed, "SEALED_PREDICTIONS", predictions)
    monkeypatch.setattr(h2h_sealed, "SEALED_RECEIPT", receipt)
    with pytest.raises(SealedArtifactError, match="truth-redacted"):
        load_sealed_run_cases(detail="artifact with labels")


def test_a_refused_load_does_not_burn_the_one_shot(monkeypatch, tmp_path):
    """A crash during setup must not consume the single sealed evaluation."""
    monkeypatch.setattr(h2h_sealed, "SEALED_PREDICTIONS", tmp_path / "absent.jsonl")
    before = purposes_used()
    with pytest.raises(SealedArtifactError):
        load_sealed_run_cases(detail="failed attempt")
    assert purposes_used() == before, (
        "sealed access must be opened AFTER the artifacts validate, never before"
    )


def test_truth_loading_is_a_separate_logged_purpose(monkeypatch, tmp_path):
    """The log must distinguish 'we generated' from 'we looked at the labels'."""
    log = tmp_path / "log.jsonl"
    monkeypatch.setattr("medguard.eval.sealed_access.DEFAULT_LOG", log)
    import medguard.eval.sealed_access as sa
    monkeypatch.setattr(h2h_sealed, "open_sealed_access",
                        lambda purpose, **kw: sa.open_sealed_access(
                            purpose, log_path=log, **kw))
    truth = load_sealed_truth(detail="scorer truth join, after the run")
    assert len(truth) == EXPECTED_LESIONS
    entries = sa.read_log(log)
    assert [e["purpose"] for e in entries] == ["sealed_scoring"]


def test_perception_readiness_is_checkable_without_opening_access():
    """A dry run must be able to find the missing artifact without spending the shot."""
    from medguard.eval.h2h_sealed import sealed_perception_status
    before = read_log()
    status = sealed_perception_status()
    assert set(status) >= {"ready"}
    assert read_log() == before, "checking readiness must not log an access"


def test_readiness_rejects_a_development_artifact(monkeypatch, tmp_path):
    from medguard.eval.h2h_sealed import sealed_perception_status
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"test_records_loaded": 0, "truth_fields_present": False}),
                       encoding="utf-8")
    monkeypatch.setattr(h2h_sealed, "SEALED_PREDICTIONS", predictions)
    monkeypatch.setattr(h2h_sealed, "SEALED_RECEIPT", receipt)
    status = sealed_perception_status()
    assert status["ready"] is False and "development artifact" in status["reason"]


def test_readiness_rejects_a_partial_freeze(monkeypatch, tmp_path):
    """A freeze that covered only some lesions must not be accepted as complete."""
    from medguard.eval.h2h_sealed import sealed_perception_status
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n" * 100, encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"test_records_loaded": 1991, "truth_fields_present": False}),
                       encoding="utf-8")
    monkeypatch.setattr(h2h_sealed, "SEALED_PREDICTIONS", predictions)
    monkeypatch.setattr(h2h_sealed, "SEALED_RECEIPT", receipt)
    status = sealed_perception_status()
    assert status["ready"] is False
    assert f"expected {EXPECTED_LESIONS}" in status["reason"]


# --- happy path ---------------------------------------------------------------------
# Every other test in this file exercises a REFUSAL. That gap let a call-signature error
# in _load_sealed_perception survive until the confirmatory run crashed on it. These
# tests exercise the path that actually has to work.

def _sealed_artifact_present():
    from medguard.eval.h2h_sealed import sealed_perception_status
    return sealed_perception_status().get("ready") is True


def test_the_real_sealed_perception_artifact_loads_into_bindings():
    """The path the confirmatory run depends on, against the real artifact."""
    if not _sealed_artifact_present():
        pytest.skip("sealed perception artifact not frozen")
    from medguard.eval.h2h_sealed import _load_sealed_perception
    bindings = _load_sealed_perception()
    assert len(bindings) == EXPECTED_LESIONS
    sample = bindings[sorted(bindings)[0]]
    assert sample.findings[0].concept_code
    assert 0.0 < sample.findings[0].confidence <= 1.0
    assert len(sample.diagnosis_probabilities) == 7
    assert sample.view_image_sha256s, "the binding must record the images it came from"


def test_every_sealed_binding_records_its_view_image_hashes():
    """A commitment whose source images are unrecorded cannot be audited later."""
    if not _sealed_artifact_present():
        pytest.skip("sealed perception artifact not frozen")
    from medguard.eval.h2h_sealed import _load_sealed_perception
    bindings = _load_sealed_perception()
    total_views = sum(len(b.view_image_sha256s) for b in bindings.values())
    assert total_views == EXPECTED_IMAGES, (
        f"{total_views} view hashes across {len(bindings)} lesions, expected "
        f"{EXPECTED_IMAGES}"
    )
    for binding in bindings.values():
        assert all(len(h) == 64 for h in binding.view_image_sha256s)


def test_a_view_lineage_mismatch_is_detected(monkeypatch, tmp_path):
    """If the manifest and the freeze disagree about a lesion's views, refuse."""
    if not _sealed_artifact_present():
        pytest.skip("sealed perception artifact not frozen")
    import json as _json
    from medguard.eval.h2h_sealed import SEALED_PREDICTIONS

    lines = SEALED_PREDICTIONS.read_text(encoding="utf-8").splitlines()
    first = _json.loads(lines[0])
    first["view_case_ids"] = ["not-a-real-view"]
    tampered = tmp_path / "predictions.jsonl"
    tampered.write_text("\n".join([_json.dumps(first)] + lines[1:]) + "\n", encoding="utf-8")
    monkeypatch.setattr(h2h_sealed, "SEALED_PREDICTIONS", tampered)
    from medguard.eval.h2h_sealed import _load_sealed_perception
    with pytest.raises(SealedArtifactError, match="view lineage mismatch"):
        _load_sealed_perception()


def test_the_perception_freeze_happened_exactly_once():
    """Updated 2026-08-08, when the freeze was executed.

    The freeze encodes the sealed images and commits the frozen probe over them. It is
    allowed once; a second one would mean the sealed partition was processed twice
    under possibly different conditions.
    """
    used = purposes_used()
    assert used.get("sealed_perception_freeze") == 1, used
    overrides = [e for e in read_log()
                 if e["purpose"] == "sealed_perception_freeze" and e["override"]]
    assert not overrides, "the freeze was re-run under an override; that must be justified"


def test_the_sealed_evaluation_was_authorised_exactly_once():
    """Updated 2026-08-09, when the confirmatory run was launched and then paused.

    The run may be interrupted and continued any number of times; each continuation is
    logged as a resume. What must never happen is a SECOND authorisation of the
    evaluation itself.
    """
    used = purposes_used()
    assert used.get("sealed_h2h_run") == 1, used
    assert used.get("sealed_scoring", 0) <= 1, used
    entries = read_log()
    confirmatory = [e for e in entries if e["purpose"] == "sealed_h2h_run"]
    assert confirmatory and confirmatory[0]["override"] is False


def test_resumes_are_recorded_separately_from_the_authorisation():
    """A paused-and-continued run must not look like repeated evaluations."""
    from medguard.eval.sealed_access import RESUME_PARENTS
    assert RESUME_PARENTS["sealed_h2h_resume"] == "sealed_h2h_run"
    used = purposes_used()
    # However many resumes have happened, the authorisation count stays at one.
    assert used.get("sealed_h2h_run") == 1


def test_the_perception_freeze_did_not_read_any_label():
    """The freeze touches sealed IMAGES. It must never touch sealed OUTCOMES."""
    import json as _json
    from pathlib import Path as _Path
    from medguard.eval.h2h_sealed import SEALED_PREDICTIONS, SEALED_RECEIPT
    if not SEALED_PREDICTIONS.is_file():
        pytest.skip("sealed perception artifact not frozen")
    receipt = _json.loads(SEALED_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["truth_fields_present"] is False
    assert receipt["generation_calls"] == 0
    banned = {"label", "truth", "target", "ground_truth"}
    for line in SEALED_PREDICTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert not banned & set(_json.loads(line)), "a truth field reached the artifact"


def test_the_sealed_freeze_used_the_same_frozen_probe_as_development():
    """A different temperature or threshold would make sealed and dev incomparable."""
    import json as _json
    from pathlib import Path as _Path
    from medguard.eval.h2h_sealed import SEALED_RECEIPT
    if not SEALED_RECEIPT.is_file():
        pytest.skip("sealed perception artifact not frozen")
    sealed = _json.loads(SEALED_RECEIPT.read_text(encoding="utf-8"))
    dev = _json.loads(_Path(
        "research/experiments/medguard_h2h_v1/perception/dermlip_v1/receipt.json"
    ).read_text(encoding="utf-8"))
    assert sealed["diagnosis_temperature"] == dev["diagnosis_temperature"]
    assert sealed["actionable_threshold"] == dev["actionable_threshold"]
    assert sealed["model_id"] == dev["model_id"]
    assert sealed["lesion_outputs"] == EXPECTED_LESIONS
    assert sealed["source_images"] == EXPECTED_IMAGES
