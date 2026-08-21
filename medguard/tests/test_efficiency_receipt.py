"""Carrying measured cost from the paying run onto the replayed result run."""
from __future__ import annotations

import pytest

from medguard.eval.efficiency_receipt import (
    EfficiencyReceiptError,
    build_efficiency_receipt,
)


def _report(arms):
    return {"per_arm": {
        arm: {"selective": {"coverage": coverage},
              "efficiency": dict(efficiency)}
        for arm, (coverage, efficiency) in arms.items()
    }}


def _cost():
    return _report({
        "M": (0.7625, {"generation_calls": 50, "total_tokens": 135950,
                       "latency_seconds": 907.0, "cache_hits": 72}),
        "B1": (1.0, {"generation_calls": 160, "total_tokens": 276107,
                     "latency_seconds": 2257.3, "cache_hits": 0}),
    })


def _replay():
    return _report({
        "M": (0.7625, {"generation_calls": 0, "total_tokens": 0, "cache_hits": 122}),
        "B1": (1.0, {"generation_calls": 0, "total_tokens": 0, "cache_hits": 160}),
    })


def test_cost_comes_from_the_paying_run_not_the_replay():
    receipt = build_efficiency_receipt(
        _cost(), _replay(), n_cases=160, cost_run_id="v1", result_run_id="v4")
    assert receipt["arms"]["M"]["remote_calls"] == 50      # not the replay's 0
    assert receipt["arms"]["M"]["total_tokens"] == 135950


def test_calls_per_answered_case_is_the_registered_primary():
    receipt = build_efficiency_receipt(
        _cost(), _replay(), n_cases=160, cost_run_id="v1", result_run_id="v4")
    assert receipt["endpoint"] == "calls_per_answered_case"
    # M: 50 gateway calls for 122 answered lesions; B1: 160 calls for 160.
    assert receipt["arms"]["M"]["answered_cases"] == 122
    assert receipt["arms"]["M"]["calls_per_answered_case"] == pytest.approx(0.4098, abs=1e-4)
    assert receipt["arms"]["B1"]["calls_per_answered_case"] == pytest.approx(1.0)


def test_cold_cache_variant_adds_the_replayed_calls_back():
    receipt = build_efficiency_receipt(
        _cost(), _replay(), n_cases=160, cost_run_id="v1", result_run_id="v4")
    # M was itself partly replayed in the cost run (72 hits): 50 + 72 = 122 -> 1.0.
    assert receipt["arms"]["M"]["calls_per_answered_case_cold"] == pytest.approx(1.0)


def test_token_cost_is_normalised_per_call_not_per_answered_case():
    """The trap: M paid tokens on 50 cold calls but answered 122 lesions.

    Dividing its token spend by 122 makes the heavier arm look ~35% cheaper than B1
    when cold it is ~58% more expensive. Per-call normalisation is cache-independent.
    """
    receipt = build_efficiency_receipt(
        _cost(), _replay(), n_cases=160, cost_run_id="v1", result_run_id="v4")
    m, b1 = receipt["arms"]["M"], receipt["arms"]["B1"]
    assert m["tokens_per_call"] == pytest.approx(135950 / 50, abs=0.1)      # 2719.0
    assert b1["tokens_per_call"] == pytest.approx(276107 / 160, abs=0.1)    # 1725.7
    # Cold, one MedGuard answer costs more than one B1 answer -- auditability is not free.
    assert m["tokens_per_answered_case_cold"] > b1["tokens_per_answered_case_cold"]
    assert m["tokens_per_answered_case_cold"] == pytest.approx(2719.0, abs=1.0)


def test_latency_is_also_per_call():
    receipt = build_efficiency_receipt(
        _cost(), _replay(), n_cases=160, cost_run_id="v1", result_run_id="v4")
    # 907.0s over 50 calls, not over 122 answers.
    assert receipt["arms"]["M"]["latency_seconds_per_call"] == pytest.approx(18.14, abs=0.01)
    assert receipt["arms"]["B1"]["latency_seconds_per_call"] == pytest.approx(14.11, abs=0.01)


def test_cold_full_set_total_reflects_what_a_reproduction_would_pay():
    receipt = build_efficiency_receipt(
        _cost(), _replay(), n_cases=160, cost_run_id="v1", result_run_id="v4")
    # M: 122 answers x 2719 tokens; B1: 160 x 1726.
    assert receipt["arms"]["M"]["total_tokens_cold_for_full_set"] == pytest.approx(
        331718, rel=0.01)
    assert receipt["arms"]["B1"]["total_tokens_cold_for_full_set"] == pytest.approx(
        276107, rel=0.01)


def test_coverage_drift_blocks_the_carry_forward():
    """If the runs answered different lesions, the cost is not transferable."""
    drifted = _replay()
    drifted["per_arm"]["M"]["selective"]["coverage"] = 0.9
    with pytest.raises(EfficiencyReceiptError, match="coverage"):
        build_efficiency_receipt(
            _cost(), drifted, n_cases=160, cost_run_id="v1", result_run_id="v4")


def test_roster_mismatch_blocks_the_carry_forward():
    partial = _replay()
    del partial["per_arm"]["B1"]
    with pytest.raises(EfficiencyReceiptError, match="roster"):
        build_efficiency_receipt(
            _cost(), partial, n_cases=160, cost_run_id="v1", result_run_id="v4")


def test_an_arm_that_answers_nothing_has_no_rate_not_a_zero_rate():
    cost = _cost()
    replay = _replay()
    for report in (cost, replay):
        report["per_arm"]["A1"] = {"selective": {"coverage": 0.0},
                                   "efficiency": {"generation_calls": 0}}
    receipt = build_efficiency_receipt(
        cost, replay, n_cases=160, cost_run_id="v1", result_run_id="v4")
    assert receipt["arms"]["A1"]["calls_per_answered_case"] is None


def test_an_alias_arm_is_not_costed_twice():
    """A0 replays M's generations; giving it its own rate would double-count M."""
    cost, replay = _cost(), _replay()
    cost["per_arm"]["A0"] = {"selective": {"coverage": 0.7625},
                             "efficiency": {"generation_calls": 0, "cache_hits": 160}}
    replay["per_arm"]["A0"] = {"alias_of": "M", "selective": {"coverage": 0.7625},
                               "efficiency": {"generation_calls": 0, "cache_hits": 160}}
    receipt = build_efficiency_receipt(
        cost, replay, n_cases=160, cost_run_id="v1", result_run_id="v4")
    a0 = receipt["arms"]["A0"]
    assert a0["cost_attributed_to"] == "M"
    assert a0["calls_per_answered_case"] is None
    assert a0["calls_per_answered_case_cold"] is None
    assert receipt["arms"]["M"]["cost_attributed_to"] == "M"


def test_n_cases_must_be_positive():
    with pytest.raises(EfficiencyReceiptError):
        build_efficiency_receipt(
            _cost(), _replay(), n_cases=0, cost_run_id="v1", result_run_id="v4")
