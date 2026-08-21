"""Paired evidence-family endpoints: joining, pairing, direction, and the gate."""
from __future__ import annotations

import pytest

from medguard.eval.paired_endpoints import (
    CaseSeries,
    PairedEndpointError,
    answered_case_ids,
    architectural_endpoint,
    arm_answered,
    evidence_family_result,
    load_case_series,
    paired_endpoint_difference,
    predicted_diagnosis,
    sealed_scale_projection,
)
from medguard.eval.statistics import (
    bootstrap_two_sided_p,
    clustered_bootstrap_ci,
    clustered_bootstrap_distribution,
)


# --- statistics additions ----------------------------------------------------------
def test_distribution_helper_reproduces_the_existing_interval_exactly():
    """The refactor must not move a single published CI bound."""
    rows = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    clusters = [f"L{i // 2}" for i in range(len(rows))]
    metric = lambda sample: sum(sample) / len(sample)
    legacy = clustered_bootstrap_ci(rows, clusters, metric, n_boot=200)
    interval, samples = clustered_bootstrap_distribution(rows, clusters, metric, n_boot=200)
    assert (interval.estimate, interval.low, interval.high) == (
        legacy.estimate, legacy.low, legacy.high)
    assert len(samples) == 200


def test_bootstrap_p_is_never_zero_and_is_two_sided():
    assert bootstrap_two_sided_p([1.0] * 500) == pytest.approx(2.0 / 501)
    assert bootstrap_two_sided_p([-1.0] * 500) == pytest.approx(2.0 / 501)
    assert bootstrap_two_sided_p([-1.0, 1.0] * 250) == 1.0


def test_bootstrap_p_requires_replicates():
    with pytest.raises(ValueError):
        bootstrap_two_sided_p([])


# --- report re-joining -------------------------------------------------------------
def _case(case_id, cluster, committed, arms):
    return {"case_id": case_id, "cluster_id": cluster,
            "committed_diagnosis": committed, "arms": arms}


def _report():
    """Three lesions across the three arm row shapes the runner actually emits."""
    return {"cases": [
        _case("c2", "L2", "nevus", {
            "M": {"arm_id": "M", "final_action": "ABSTAIN"},
            "B1": {"arm_id": "B1", "answered": True},
            "B0": {"arm_id": "B0", "predicted_diagnosis": "bkl"}}),
        _case("c1", "L1", "melanoma", {
            "M": {"arm_id": "M", "final_action": "PASS"},
            "B1": {"arm_id": "B1", "answered": True},
            "B0": {"arm_id": "B0", "predicted_diagnosis": "melanoma"}}),
        _case("c3", "L3", "bcc", {
            "M": {"arm_id": "M", "final_action": "ANSWER_UNGATED"},
            "B1": {"arm_id": "B1", "answered": True, "generation_error": "boom"},
            "B0": {"arm_id": "B0", "predicted_diagnosis": ""}}),
    ]}


def _scored(arm, n_samples, faithfulness=None, scopes=None):
    scores = {}
    if faithfulness is not None:
        scores["faithfulness"] = {"per_sample": faithfulness,
                                  "per_sample_scope": scopes or ["assessment"] * len(faithfulness)}
    return {"per_arm": {arm: {"ragas_supporting": {"n_samples": n_samples, "scores": scores}}}}


def test_answered_is_read_from_each_arm_row_shape():
    cases = _report()["cases"]
    by_id = {c["case_id"]: c for c in cases}
    assert arm_answered(by_id["c1"], "M") is True            # final_action PASS
    assert arm_answered(by_id["c3"], "M") is True            # ANSWER_UNGATED counts
    assert arm_answered(by_id["c2"], "M") is False           # ABSTAIN
    assert arm_answered(by_id["c1"], "B1") is True           # answered flag
    assert arm_answered(by_id["c3"], "B1") is False          # generation error wins
    assert arm_answered(by_id["c1"], "B0") is True           # has a diagnosis
    assert arm_answered(by_id["c3"], "B0") is False          # empty diagnosis


def test_answered_case_ids_are_in_case_id_order_not_report_order():
    """The RAGAS sample order is case-id sorted; report order is not."""
    assert answered_case_ids(_report(), "M") == ["c1", "c3"]


def test_missing_arm_is_an_error_not_a_silent_skip():
    with pytest.raises(PairedEndpointError):
        arm_answered(_report()["cases"][0], "B9")


def test_only_b0_diagnoses_independently():
    case = _report()["cases"][1]
    assert predicted_diagnosis(case, "M") == "melanoma"     # frozen probe commitment
    assert predicted_diagnosis(case, "B0") == "melanoma"    # its own prediction
    assert predicted_diagnosis(_report()["cases"][0], "B0") == "bkl"


def test_sample_count_drift_raises_rather_than_mis_joining():
    """A judge score attached to the wrong lesion is worse than no score at all."""
    truth = {"L1": "melanoma", "L2": "nevus", "L3": "bcc"}
    with pytest.raises(PairedEndpointError, match="unsafe"):
        load_case_series(_report(), _scored("M", 3, [1.0, 1.0, 1.0]), "M", truth)


def test_per_sample_scores_join_to_the_answered_cases():
    truth = {"L1": "melanoma", "L2": "nevus", "L3": "bcc"}
    series = load_case_series(_report(), _scored("M", 2, [0.9, None]), "M", truth)
    faith = series["ragas_faithfulness"]
    assert faith.values["c1"] == 0.9
    assert faith.values["c3"] is None      # judge returned no score
    assert faith.values["c2"] is None      # abstained, never sampled


def test_over_reliance_counts_answered_and_wrong_over_every_lesion():
    truth = {"L1": "melanoma", "L2": "nevus", "L3": "melanoma"}
    series = load_case_series(_report(), _scored("M", 2, [0.9, 0.9]), "M", truth)
    over = series["over_reliance"].values
    assert over["c1"] == 0.0     # answered, right
    assert over["c3"] == 1.0     # answered (ANSWER_UNGATED), committed bcc != melanoma
    assert over["c2"] == 0.0     # abstained: wrong-but-silent is not over-reliance
    assert len(over) == 3        # abstentions stay in the denominator


def test_missing_label_raises():
    with pytest.raises(PairedEndpointError, match="no development label"):
        load_case_series(_report(), _scored("M", 2, [0.9, 0.9]), "M", {"L1": "melanoma"})


def test_literature_scoped_series_drops_whole_scope_cases():
    truth = {"L1": "melanoma", "L2": "nevus", "L3": "bcc"}
    series = load_case_series(
        _report(), _scored("M", 2, [0.95, 0.10], scopes=["assessment", "whole"]), "M", truth)
    scoped = series["ragas_faithfulness_literature_scoped"].values
    assert scoped["c1"] == 0.95
    assert scoped["c3"] is None    # whole-scope: the text judge had no jurisdiction
    assert series["ragas_faithfulness"].values["c3"] == 0.10   # unrestricted keeps it


# --- pairing -----------------------------------------------------------------------
def _series(arm, endpoint, values):
    clusters = {cid: f"L{cid}" for cid in values}
    return CaseSeries(arm, endpoint, values, clusters)


def test_pairing_uses_only_jointly_defined_cases_and_never_imputes_zero():
    a = _series("M", "ragas_context_recall", {"1": 1.0, "2": 1.0, "3": None})
    b = _series("B1", "ragas_context_recall", {"1": 0.0, "2": None, "3": 0.0})
    result = paired_endpoint_difference(a, b, n_boot=200)
    assert result["n_paired"] == 1                      # only case 1 is defined in both
    assert result["mean_a_unconditional"] == 1.0        # M's own two defined cases
    assert result["n_defined_a"] == 2 and result["n_defined_b"] == 2
    assert result["coverage_conditioned"] is True


def test_no_shared_cases_returns_none_rather_than_a_fabricated_zero():
    a = _series("M", "x", {"1": 1.0, "2": None})
    b = _series("B1", "x", {"1": None, "2": 1.0})
    assert paired_endpoint_difference(a, b, n_boot=50) is None


def test_mismatched_endpoints_cannot_be_paired():
    with pytest.raises(PairedEndpointError):
        paired_endpoint_difference(
            _series("M", "faith", {"1": 1.0}), _series("B1", "recall", {"1": 1.0}), n_boot=50)


def test_lower_is_better_endpoint_credits_the_arm_with_the_smaller_value():
    a = _series("M", "over_reliance", {str(i): 0.0 for i in range(12)})
    b = _series("B1", "over_reliance", {str(i): 1.0 for i in range(12)})
    result = paired_endpoint_difference(a, b, n_boot=300)
    assert result["higher_is_better"] is False
    assert result["estimate"] == pytest.approx(-1.0)
    assert result["improvement_for_arm_a"] == pytest.approx(1.0)
    assert result["favours"] == "M"


def test_an_interval_spanning_zero_is_inconclusive_not_a_win():
    a = _series("M", "ragas_faithfulness", {str(i): float(i % 2) for i in range(20)})
    b = _series("B1", "ragas_faithfulness", {str(i): float((i + 1) % 2) for i in range(20)})
    result = paired_endpoint_difference(a, b, n_boot=500)
    assert result["excludes_zero"] is False
    assert result["favours"] == "inconclusive"


# --- the registered gate -----------------------------------------------------------
def test_architectural_endpoints_are_marked_untestable():
    entry = architectural_endpoint("citation_integrity", "M", "B1", 1.0, None)
    assert entry["testable"] is False
    assert entry["value_b"] is None      # never imputed as 0.0


def _passing(metric, p=0.001):
    return {"metric": metric, "p_value": p, "excludes_zero": True,
            "improvement_for_arm_a": 0.3, "testable": True}


def test_condition_needs_two_registered_endpoints():
    one = evidence_family_result([_passing("ragas_context_recall")])
    assert one["n_endpoints_passing"] == 1 and one["condition_met"] is False
    two = evidence_family_result(
        [_passing("ragas_context_recall"), _passing("ragas_faithfulness")])
    assert two["condition_met"] is True


def test_holm_m_counts_only_the_tests_actually_performed():
    result = evidence_family_result([
        _passing("ragas_context_recall"),
        architectural_endpoint("citation_integrity", "M", "B1", 1.0, None),
        architectural_endpoint("contract_completion", "M", "B1", 1.0, None),
    ])
    assert result["holm_m"] == 1


def test_an_endpoint_outside_the_frozen_family_cannot_satisfy_the_condition():
    """Post-hoc endpoints must not be able to buy the acceptance condition."""
    result = evidence_family_result([
        _passing("ragas_context_recall"),
        _passing("ragas_faithfulness_literature_scoped"),
    ])
    assert result["endpoints_passing"] == ["ragas_context_recall"]
    assert result["condition_met"] is False


def test_a_significant_result_in_the_wrong_direction_does_not_count():
    losing = {"metric": "ragas_faithfulness", "p_value": 0.001, "excludes_zero": True,
              "improvement_for_arm_a": -0.4, "testable": True}
    result = evidence_family_result([_passing("ragas_context_recall"), losing])
    assert result["n_endpoints_passing"] == 1 and result["condition_met"] is False


def test_holm_steps_down_smallest_p_takes_the_largest_multiplier():
    borderline = {"metric": "ragas_faithfulness", "p_value": 0.04, "excludes_zero": True,
                  "improvement_for_arm_a": 0.1, "testable": True}
    result = evidence_family_result([_passing("ragas_context_recall", p=0.001), borderline])
    adjusted = {c["metric"]: c["p_value_holm"] for c in result["comparisons"]
                if "p_value_holm" in c}
    assert adjusted["ragas_context_recall"] == pytest.approx(0.002)   # smallest: x m
    assert adjusted["ragas_faithfulness"] == pytest.approx(0.04)      # largest:  x 1
    assert result["condition_met"] is True


# --- sealed-scale projection -------------------------------------------------------
def _inconclusive(estimate, low, high, n_paired=80):
    return {"metric": "ragas_faithfulness", "estimate": estimate, "low": low,
            "high": high, "n_paired": n_paired, "excludes_zero": False, "testable": True}


def test_projection_separates_an_underpowered_endpoint_from_a_null_one():
    underpowered = sealed_scale_projection(
        _inconclusive(0.0593, -0.0514, 0.1677), dev_lesions=160, sealed_lesions=1495)
    assert underpowered["projected_half_width"] < abs(underpowered["estimate"])
    assert underpowered["would_exclude_zero_at_sealed_scale"] is True

    genuinely_null = sealed_scale_projection(
        _inconclusive(0.002, -0.11, 0.11), dev_lesions=160, sealed_lesions=1495)
    assert genuinely_null["would_exclude_zero_at_sealed_scale"] is False


def test_projection_is_labelled_a_projection_and_lists_its_assumptions():
    projection = sealed_scale_projection(
        _inconclusive(0.06, -0.05, 0.17), dev_lesions=160, sealed_lesions=1495)
    assert projection["status"] == "projection_not_result"
    assert len(projection["assumptions"]) == 3


def test_projection_scales_the_half_width_by_sqrt_of_the_lesion_ratio():
    projection = sealed_scale_projection(
        _inconclusive(0.1, -0.1, 0.3), dev_lesions=100, sealed_lesions=400)
    assert projection["observed_half_width"] == pytest.approx(0.2)
    assert projection["projected_half_width"] == pytest.approx(0.1)   # /sqrt(4)
    assert projection["projected_n_paired"] == 320


def test_projection_declines_untestable_endpoints():
    entry = architectural_endpoint("citation_integrity", "M", "B1", 1.0, None)
    assert sealed_scale_projection(entry, dev_lesions=160, sealed_lesions=1495) is None


def test_projection_rejects_nonsense_lesion_counts():
    with pytest.raises(PairedEndpointError):
        sealed_scale_projection(
            _inconclusive(0.1, -0.1, 0.3), dev_lesions=0, sealed_lesions=1495)


def test_holm_can_withdraw_a_marginally_significant_endpoint():
    """Two endpoints at p=0.03 each: Holm promotes the smaller to 0.06 and it fails."""
    a = {"metric": "ragas_context_recall", "p_value": 0.03, "excludes_zero": True,
         "improvement_for_arm_a": 0.1, "testable": True}
    b = {"metric": "ragas_faithfulness", "p_value": 0.03, "excludes_zero": True,
         "improvement_for_arm_a": 0.1, "testable": True}
    result = evidence_family_result([a, b])
    assert result["n_endpoints_passing"] == 0
    assert result["condition_met"] is False
