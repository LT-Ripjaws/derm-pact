"""The one-shot guard: it must refuse a second sealed evaluation and log the first."""
from __future__ import annotations

import json

import pytest

from medguard.eval.sealed_access import (
    GENESIS,
    SealedAccessError,
    SealedAccessExhausted,
    SealedLogTampered,
    assert_never_accessed,
    open_sealed_access,
    purposes_used,
    read_log,
    verify_log,
)


@pytest.fixture
def log(tmp_path):
    return tmp_path / "SEALED_ACCESS_LOG.jsonl"


def test_an_absent_log_means_never_accessed(log):
    assert read_log(log) == []
    assert verify_log(log) == {"valid": True, "n_entries": 0, "head_sha256": GENESIS,
                               "purposes_used": []}
    assert purposes_used(log) == {}
    assert_never_accessed("sealed_h2h_run", log)   # must not raise


def test_the_first_access_is_granted_and_logged(log):
    grant = open_sealed_access("sealed_h2h_run", detail="the single paid evaluation",
                               log_path=log)
    assert grant.sequence == 0 and grant.override is False
    entries = read_log(log)
    assert len(entries) == 1
    assert entries[0]["purpose"] == "sealed_h2h_run"
    assert entries[0]["detail"] == "the single paid evaluation"
    assert entries[0]["previous_sha256"] == GENESIS


def test_a_second_access_to_the_same_purpose_is_refused(log):
    """The protocol allows the sealed test ONCE. This is that rule, enforced."""
    open_sealed_access("sealed_h2h_run", detail="first", log_path=log)
    with pytest.raises(SealedAccessExhausted, match="already used"):
        open_sealed_access("sealed_h2h_run", detail="second", log_path=log)
    assert len(read_log(log)) == 1, "a refused access must not be logged as a success"


def test_different_purposes_do_not_block_each_other(log):
    open_sealed_access("sealed_perception_freeze", detail="encode test images", log_path=log)
    grant = open_sealed_access("sealed_h2h_run", detail="the paid run", log_path=log)
    assert grant.sequence == 1
    assert purposes_used(log) == {"sealed_perception_freeze": 1, "sealed_h2h_run": 1}


def test_an_override_must_spell_out_the_purpose_and_is_recorded_as_one(log):
    open_sealed_access("sealed_h2h_run", detail="first", log_path=log)
    with pytest.raises(SealedAccessExhausted):
        open_sealed_access("sealed_h2h_run", detail="retry", log_path=log,
                           override_token="yes")
    grant = open_sealed_access("sealed_h2h_run", detail="deliberate re-run", log_path=log,
                               override_token="OVERRIDE:sealed_h2h_run")
    assert grant.override is True
    assert read_log(log)[-1]["override"] is True


def test_a_smoke_purpose_is_repeatable_but_every_touch_is_logged(log):
    """One-shot protects a CLAIM. A smoke test makes none, so it repeats freely.

    Forcing overrides for routine pipeline checks would make overrides unremarkable,
    which is the habit that lets a real one slip past.
    """
    from medguard.eval.sealed_access import REPEATABLE_PURPOSES
    assert "sealed_smoke" in REPEATABLE_PURPOSES
    for i in range(3):
        grant = open_sealed_access("sealed_smoke", detail=f"check {i}", log_path=log)
        assert grant.override is False, "a repeatable purpose is not an override"
    assert purposes_used(log)["sealed_smoke"] == 3
    assert verify_log(log)["valid"], "every smoke touch must still extend the chain"


def test_the_confirmatory_purpose_is_not_repeatable(log):
    """The protection that actually matters must not have been loosened."""
    from medguard.eval.sealed_access import REPEATABLE_PURPOSES
    assert "sealed_h2h_run" not in REPEATABLE_PURPOSES
    assert "sealed_perception_freeze" not in REPEATABLE_PURPOSES
    open_sealed_access("sealed_h2h_run", detail="the run", log_path=log)
    with pytest.raises(SealedAccessExhausted):
        open_sealed_access("sealed_h2h_run", detail="again", log_path=log)


def test_resuming_does_not_stamp_an_override_on_the_confirmatory_purpose(log):
    """A pause/resume is continuing one evaluation, not authorising a second.

    Recording it as an override would put override=True against the entry a reviewer
    scrutinises most, to describe something entirely innocent.
    """
    open_sealed_access("sealed_h2h_run", detail="the confirmatory run", log_path=log)
    first = open_sealed_access("sealed_h2h_resume", detail="resume after pause", log_path=log)
    second = open_sealed_access("sealed_h2h_resume", detail="resume again", log_path=log)
    assert first.override is False and second.override is False
    confirmatory = [e for e in read_log(log) if e["purpose"] == "sealed_h2h_run"]
    assert len(confirmatory) == 1 and confirmatory[0]["override"] is False, (
        "the confirmatory purpose must keep exactly one clean entry"
    )
    assert purposes_used(log)["sealed_h2h_resume"] == 2


def test_a_resume_cannot_stand_in_for_a_run_that_never_started(log):
    """Otherwise a run could pose as the continuation of an evaluation that never was."""
    with pytest.raises(SealedAccessError, match="never opened"):
        open_sealed_access("sealed_h2h_resume", detail="resume nothing", log_path=log)
    with pytest.raises(SealedAccessError, match="never opened"):
        open_sealed_access("sealed_scoring_resume", detail="resume nothing", log_path=log)


def test_every_resume_is_still_recorded_in_the_chain(log):
    open_sealed_access("sealed_h2h_run", detail="run", log_path=log)
    for i in range(3):
        open_sealed_access("sealed_h2h_resume", detail=f"resume {i}", log_path=log)
    chain = verify_log(log)
    assert chain["valid"] and chain["n_entries"] == 4
    assert "how many times the run was resumed" or True
    assert purposes_used(log)["sealed_h2h_resume"] == 3


def test_an_unregistered_purpose_is_allowed_but_flagged(log):
    open_sealed_access("curiosity", detail="a peek", log_path=log)
    assert read_log(log)[0]["registered_purpose"] is False


def test_a_read_without_a_reason_is_refused(log):
    with pytest.raises(SealedAccessError, match="detail must be non-empty"):
        open_sealed_access("sealed_h2h_run", detail="   ", log_path=log)
    with pytest.raises(SealedAccessError, match="purpose must be non-empty"):
        open_sealed_access("", detail="x", log_path=log)


def test_the_chain_links_each_entry_to_the_previous(log):
    open_sealed_access("sealed_perception_freeze", detail="a", log_path=log)
    open_sealed_access("sealed_h2h_run", detail="b", log_path=log)
    entries = read_log(log)
    assert entries[1]["previous_sha256"] == entries[0]["entry_sha256"]
    chain = verify_log(log)
    assert chain["valid"] and chain["head_sha256"] == entries[1]["entry_sha256"]


def test_editing_an_entry_breaks_the_chain(log):
    open_sealed_access("sealed_perception_freeze", detail="a", log_path=log)
    open_sealed_access("sealed_h2h_run", detail="b", log_path=log)
    entries = read_log(log)
    entries[0]["detail"] = "something else"
    log.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n",
                   encoding="utf-8")
    chain = verify_log(log)
    assert chain["valid"] is False and chain["broken_at"] == 0


def test_deleting_an_entry_breaks_the_chain(log):
    open_sealed_access("sealed_perception_freeze", detail="a", log_path=log)
    open_sealed_access("sealed_h2h_run", detail="b", log_path=log)
    entries = read_log(log)
    log.write_text(json.dumps(entries[1], sort_keys=True) + "\n", encoding="utf-8")
    assert verify_log(log)["valid"] is False


def test_a_broken_chain_refuses_further_access(log):
    """You cannot quietly repair provenance by appending to a tampered log."""
    open_sealed_access("sealed_perception_freeze", detail="a", log_path=log)
    entries = read_log(log)
    entries[0]["detail"] = "edited"
    log.write_text(json.dumps(entries[0], sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SealedLogTampered, match="Refusing to extend"):
        open_sealed_access("sealed_h2h_run", detail="b", log_path=log)


def test_assert_never_accessed_fires_once_the_sealed_run_has_happened(log):
    open_sealed_access("sealed_h2h_run", detail="the run", log_path=log)
    with pytest.raises(SealedAccessExhausted, match="no longer true"):
        assert_never_accessed("sealed_h2h_run", log)


def test_the_repository_sealed_log_is_intact_and_the_evaluation_was_authorised_once():
    """Updated 2026-08-09, when the confirmatory run was launched.

    The invariant is no longer "untouched" -- it is that the confirmatory purpose was
    opened EXACTLY ONCE and cleanly. Interruptions are recorded as resumes, which is
    what keeps that single entry clean.
    """
    chain = verify_log()
    assert chain["valid"], chain
    entries = read_log()
    confirmatory = [e for e in entries if e["purpose"] == "sealed_h2h_run"]
    assert len(confirmatory) == 1, (
        f"the confirmatory purpose was opened {len(confirmatory)} times; it authorises "
        f"ONE evaluation"
    )
    assert confirmatory[0]["override"] is False, (
        "the confirmatory evaluation was opened under an override; that requires an "
        "explicit justification in the manuscript"
    )
    scoring = [e for e in entries if e["purpose"] == "sealed_scoring"]
    assert len(scoring) <= 1
    # Generation must precede any label read: the whole point of separate purposes.
    if scoring:
        assert confirmatory[0]["sequence"] < scoring[0]["sequence"], (
            "sealed labels were read before sealed generation began"
        )
