"""Citation-shaped text vs citation that resolves to a hash-pinned source."""
from __future__ import annotations

import pytest

from medguard.eval.citation_resolvability import (
    build_report,
    count_markers,
    summarize_arm,
)


def test_markers_are_counted_not_deduplicated():
    assert count_markers("Excise with margins [1]. Follow up [1][2].") == 3
    assert count_markers("No citation apparatus at all.") == 0
    assert count_markers("") == 0


def test_a_baseline_that_cites_nothing_resolvable_scores_zero_not_none():
    """The whole point: a baseline emits citation SHAPE with zero resolvability."""
    baseline = summarize_arm(
        ["Findings [1]. Management [2][3]."] * 4,
        resolvable_citations=0, cases_with_resolvable_citations=0,
        intact_citations=0, n_cases=4)
    assert baseline["total_citation_markers"] == 12
    assert baseline["citation_resolvability"] == 0.0     # a real zero, not undefined
    assert baseline["integrity_of_resolvable"] is None   # nothing to be intact
    assert baseline["verifiable_citation_coverage_unconditional"] == 0.0


def test_an_arm_that_writes_no_markers_has_undefined_resolvability():
    """B0 writes no citation apparatus; 0/0 is undefined, never 0.0."""
    vlm = summarize_arm(["Looks like melanoma."] * 3, resolvable_citations=0,
                        cases_with_resolvable_citations=0, intact_citations=0, n_cases=3)
    assert vlm["total_citation_markers"] == 0
    assert vlm["citation_resolvability"] is None


def test_resolvable_arm_reports_both_denominators():
    medguard = summarize_arm(
        ["Assessment: excise [1][2]."] * 2 + ["Diagnosis: nevus."] * 2,
        resolvable_citations=4, cases_with_resolvable_citations=2,
        intact_citations=4, n_cases=4, guideline_eligible_cases=2)
    assert medguard["citation_resolvability"] == 1.0
    assert medguard["integrity_of_resolvable"] == 1.0
    # Unconditional 2/4; on the lesions a guideline exists for, 2/2.
    assert medguard["verifiable_citation_coverage_unconditional"] == 0.5
    assert medguard["verifiable_citation_coverage_on_eligible"] == 1.0


def test_eligible_denominator_is_optional_and_absent_means_absent():
    without = summarize_arm(["x [1]"], resolvable_citations=1,
                            cases_with_resolvable_citations=1, intact_citations=1, n_cases=1)
    assert without["verifiable_citation_coverage_on_eligible"] is None


def test_resolvability_cannot_exceed_one_when_the_contract_holds_more_than_it_prints():
    """MedGuard's contract carries citations the rendered answer does not all print."""
    arm = summarize_arm(["Assessment: excise [1]."], resolvable_citations=5,
                        cases_with_resolvable_citations=1, intact_citations=5, n_cases=1)
    assert arm["citation_resolvability"] == 1.0


def test_n_cases_must_be_positive():
    with pytest.raises(ValueError):
        summarize_arm(["x"], resolvable_citations=0, cases_with_resolvable_citations=0,
                      intact_citations=0, n_cases=0)


def test_report_declares_itself_descriptive_rather_than_inferential():
    report = build_report({"M": {}}, n_cases=160, guideline_eligible_cases=92)
    assert report["role"] == "architectural_descriptive_not_inferential"
    assert "settled by architecture" in report["why_not_a_hypothesis_test"]
