"""The LLM judge's test-retest floor, measured from byte-identical arms."""
from __future__ import annotations

import pytest

from medguard.eval.judge_noise_floor import (
    JUDGE_SCORED_ENDPOINTS,
    apply_floor,
    measure_floor,
    verify_identical,
)
from medguard.eval.paired_endpoints import CaseSeries, PairedEndpointError


def _report(arms_by_case):
    return {"cases": [
        {"case_id": cid, "cluster_id": f"L{cid}", "arms": arms}
        for cid, arms in arms_by_case.items()
    ]}


def _identical_report():
    arms = {"M": {"contract_sha256": "abc", "evidence_document_ids": ["e1"]},
            "A0": {"contract_sha256": "abc", "evidence_document_ids": ["e1"]},
            "A3": {"contract_sha256": "abc", "evidence_document_ids": ["e1"]}}
    return _report({"c1": arms, "c2": arms})


def test_identity_is_proven_not_assumed():
    proof = verify_identical(_identical_report(), ["M", "A0", "A3"])
    assert proof["identical"] is True and proof["cases"] == 2


def test_a_differing_arm_cannot_form_a_null():
    """A floor measured from arms that actually differ would understate real effects."""
    report = _identical_report()
    report["cases"][0]["arms"]["A3"] = {"contract_sha256": "zzz",
                                        "evidence_document_ids": ["e1"]}
    with pytest.raises(PairedEndpointError, match="not identical"):
        verify_identical(report, ["M", "A0", "A3"])


def test_differing_evidence_also_blocks_the_null():
    report = _identical_report()
    report["cases"][0]["arms"]["A0"] = {"contract_sha256": "abc",
                                        "evidence_document_ids": ["DIFFERENT"]}
    with pytest.raises(PairedEndpointError, match="not identical"):
        verify_identical(report, ["M", "A0", "A3"])


def _series(arm, endpoint, values):
    return CaseSeries(arm, endpoint, values, {cid: f"L{cid}" for cid in values})


def _judgings(a_values, b_values, endpoint="ragas_context_recall"):
    return {
        "M": {endpoint: _series("M", endpoint, a_values)},
        "A3": {endpoint: _series("A3", endpoint, b_values)},
    }


def test_a_perfectly_deterministic_judge_has_a_zero_floor():
    values = {str(i): float(i % 3) / 2 for i in range(12)}
    floor = measure_floor(_judgings(values, dict(values)), n_boot=200)
    block = floor["endpoints"]["ragas_context_recall"]
    assert block["max_abs_null_difference"] == 0.0
    assert block["false_positives"] == 0


def test_a_disagreeing_judge_produces_a_nonzero_floor():
    a = {str(i): 1.0 for i in range(12)}
    b = {str(i): 0.8 for i in range(12)}
    floor = measure_floor(_judgings(a, b), n_boot=300)
    block = floor["endpoints"]["ragas_context_recall"]
    assert block["max_abs_null_difference"] == pytest.approx(0.2, abs=1e-6)
    # Identical systems differing consistently is, by construction, a false positive.
    assert block["false_positives"] == 1
    assert block["minimum_interpretable_effect"] == pytest.approx(0.2, abs=1e-6)


def test_floor_needs_at_least_two_identical_arms():
    with pytest.raises(PairedEndpointError, match="at least two"):
        measure_floor({"M": {}}, n_boot=50)


def test_an_effect_below_the_floor_is_reported_as_judge_noise():
    floor = {"endpoints": {"ragas_context_recall": {"minimum_interpretable_effect": 0.05}}}
    annotated = apply_floor(
        {"metric": "ragas_context_recall", "estimate": -0.0489, "favours": "A3"}, floor)
    assert annotated["above_judge_noise_floor"] is False
    assert annotated["favours"] == "indistinguishable_from_judge_noise"
    assert "judge noise" in annotated["floor_note"]


def test_an_effect_above_the_floor_keeps_its_verdict_and_gains_a_ratio():
    floor = {"endpoints": {"ragas_context_recall": {"minimum_interpretable_effect": 0.05}}}
    annotated = apply_floor(
        {"metric": "ragas_context_recall", "estimate": 0.3451, "favours": "M"}, floor)
    assert annotated["above_judge_noise_floor"] is True
    assert annotated["favours"] == "M"
    assert annotated["effect_to_noise_ratio"] == pytest.approx(6.9, abs=0.1)


def test_host_computed_endpoints_are_exempt_from_the_floor():
    """Citation integrity and over-reliance are deterministic; no judge is involved."""
    annotated = apply_floor(
        {"metric": "over_reliance", "estimate": -0.1125, "favours": "M"},
        {"endpoints": {}})
    assert annotated["above_judge_noise_floor"] is None
    assert "deterministic" in annotated["floor_note"]
    assert annotated["favours"] == "M"


def test_the_floor_covers_every_judge_scored_endpoint():
    assert "ragas_faithfulness" in JUDGE_SCORED_ENDPOINTS
    assert "ragas_context_recall" in JUDGE_SCORED_ENDPOINTS
    assert "over_reliance" not in JUDGE_SCORED_ENDPOINTS
    assert "citation_integrity" not in JUDGE_SCORED_ENDPOINTS
