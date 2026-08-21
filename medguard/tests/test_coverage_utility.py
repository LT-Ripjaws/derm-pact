"""Coverage-adjusted utility: counts, the surface, and the break-even algebra."""
from __future__ import annotations

import pytest

from medguard.eval.coverage_utility import (
    CoverageUtilityError,
    arm_counts,
    break_even_wrong_cost,
    build_utility_table,
    utility,
)


def test_counts_decompose_coverage_and_answered_accuracy():
    counts = arm_counts(n_cases=160, coverage=0.7625, answered_accuracy=0.8361)
    assert counts == {"answered": 122, "correct": 102, "wrong": 20, "abstained": 38}


def test_a_full_coverage_arm_never_abstains():
    counts = arm_counts(n_cases=160, coverage=1.0, answered_accuracy=0.7625)
    assert counts == {"answered": 160, "correct": 122, "wrong": 38, "abstained": 0}


def test_an_arm_that_answers_nothing_needs_no_accuracy():
    counts = arm_counts(n_cases=160, coverage=0.0, answered_accuracy=None)
    assert counts == {"answered": 0, "correct": 0, "wrong": 0, "abstained": 160}


def test_an_answering_arm_without_an_accuracy_is_an_error():
    with pytest.raises(CoverageUtilityError, match="answered accuracy"):
        arm_counts(n_cases=160, coverage=0.5, answered_accuracy=None)


def test_impossible_coverage_is_rejected():
    with pytest.raises(CoverageUtilityError, match="outside"):
        arm_counts(n_cases=160, coverage=1.4, answered_accuracy=0.8)


def test_at_zero_wrong_cost_utility_is_just_correct_over_total():
    """lambda=0 is the setting least favourable to abstention -- it must be included."""
    counts = {"answered": 122, "correct": 102, "wrong": 20, "abstained": 38}
    assert utility(counts, n_cases=160, wrong_cost=0.0, abstention_cost=0.0) == pytest.approx(
        102 / 160)


def test_abstention_cost_is_charged_per_abstained_case():
    counts = {"answered": 122, "correct": 102, "wrong": 20, "abstained": 38}
    charged = utility(counts, n_cases=160, wrong_cost=0.0, abstention_cost=0.5)
    assert charged == pytest.approx((102 - 0.5 * 38) / 160)


def test_break_even_is_the_lambda_where_the_two_utilities_cross():
    medguard = {"answered": 122, "correct": 102, "wrong": 20, "abstained": 38}
    baseline = {"answered": 160, "correct": 122, "wrong": 38, "abstained": 0}
    lam = break_even_wrong_cost(medguard, baseline, abstention_cost=0.0)
    # (122-102 + 0) / (38-20) = 20/18
    assert lam == pytest.approx(1.1111, abs=1e-4)
    # Verified by evaluating both arms exactly at the crossing point.
    assert utility(medguard, n_cases=160, wrong_cost=lam, abstention_cost=0.0) == pytest.approx(
        utility(baseline, n_cases=160, wrong_cost=lam, abstention_cost=0.0), abs=1e-3)


def test_charging_for_abstention_raises_the_break_even():
    medguard = {"answered": 122, "correct": 102, "wrong": 20, "abstained": 38}
    baseline = {"answered": 160, "correct": 122, "wrong": 38, "abstained": 0}
    free = break_even_wrong_cost(medguard, baseline, abstention_cost=0.0)
    costly = break_even_wrong_cost(medguard, baseline, abstention_cost=0.5)
    assert costly > free, "deferral overhead must make abstention harder to justify"


def test_equal_error_counts_have_no_crossing_and_report_none():
    a = {"answered": 100, "correct": 80, "wrong": 20, "abstained": 60}
    b = {"answered": 120, "correct": 100, "wrong": 20, "abstained": 40}
    assert break_even_wrong_cost(a, b, abstention_cost=0.0) is None


def _arms():
    return {
        "M": {"coverage": 0.7625, "answered_accuracy": 0.8361},
        "B1": {"coverage": 1.0, "answered_accuracy": 0.7625},
        "B0": {"coverage": 1.0, "answered_accuracy": 0.3625},
        "A1": {"coverage": 0.0, "answered_accuracy": None},
    }


def test_table_reports_a_surface_over_both_parameters():
    table = build_utility_table(_arms(), n_cases=160, reference_arm="B1")
    surface = table["utility_surface"]["M"]
    assert "mu=0.0" in surface and "lambda=10.0" in surface["mu=0.0"]
    # Utility must fall as the cost of being wrong rises.
    row = surface["mu=0.0"]
    assert row["lambda=0.0"] > row["lambda=1.0"] > row["lambda=10.0"]


def test_the_sweep_includes_the_setting_least_favourable_to_abstention():
    table = build_utility_table(_arms(), n_cases=160, reference_arm="B1")
    at_zero = table["utility_surface"]
    assert at_zero["B1"]["mu=0.0"]["lambda=0.0"] > at_zero["M"]["mu=0.0"]["lambda=0.0"], (
        "with wrong answers free, the full-coverage baseline must win; a definition "
        "where MedGuard wins everywhere would be a rigged one"
    )


def test_break_evens_are_computed_against_the_registered_comparator():
    table = build_utility_table(_arms(), n_cases=160, reference_arm="B1")
    assert table["reference_arm"] == "B1"
    assert "B1" not in table["break_even_wrong_cost_vs_reference"]
    assert table["break_even_wrong_cost_vs_reference"]["M"]["mu=0.0"] == pytest.approx(
        1.1111, abs=1e-4)


def test_a_never_answering_arm_has_a_utility_that_only_falls():
    table = build_utility_table(_arms(), n_cases=160, reference_arm="B1")
    a1 = table["utility_surface"]["A1"]
    assert a1["mu=0.0"]["lambda=10.0"] == 0.0     # nothing right, nothing wrong
    assert a1["mu=0.5"]["lambda=0.0"] == pytest.approx(-0.5)


def test_missing_reference_arm_is_an_error():
    with pytest.raises(CoverageUtilityError, match="reference arm"):
        build_utility_table(_arms(), n_cases=160, reference_arm="B9")


def test_token_normalised_utility_is_optional_and_penalises_the_expensive_arm():
    table = build_utility_table(
        _arms(), n_cases=160, reference_arm="B1",
        tokens_per_answered_case={"M": 2719.0, "B1": 1725.7, "B0": 569.8, "A1": None})
    assert table["utility_per_1k_tokens"]["A1"] is None
    m = table["utility_per_1k_tokens"]["M"]["lambda=0.0"]
    b1 = table["utility_per_1k_tokens"]["B1"]["lambda=0.0"]
    assert b1 > m, "at lambda=0 the cheaper, fuller-coverage arm must lead per token"


def test_the_table_states_it_is_not_a_clinical_utility_model():
    table = build_utility_table(_arms(), n_cases=160, reference_arm="B1")
    assert "prices decisions, not outcomes" in table["not_a_clinical_utility_model"]
    assert "missed melanoma" in table["not_a_clinical_utility_model"]
