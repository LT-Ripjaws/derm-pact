from __future__ import annotations

import json

from medguard.eval.guideline_catalog_feasibility_v1 import (
    COMMITMENT_TO_CONDITION,
    ROOT,
    build_report,
    condition_scoped_candidates,
    summarize,
)


def _recommendations():
    return [
        {
            "recommendation_id": "box-1",
            "condition": "basal_cell_carcinoma",
            "eligible_claim_types": ["assessment", "diagnosis", "visual_finding", "differential"],
        },
        {
            "recommendation_id": "box-5",
            "condition": "basal_cell_carcinoma",
            "eligible_claim_types": ["management", "treatment"],
        },
        {
            "recommendation_id": "recommendation-1",
            "condition": "melanoma",
            "eligible_claim_types": ["diagnosis", "management", "treatment"],
        },
    ]


def test_scoping_is_keyed_on_the_commitment_and_excludes_other_conditions():
    rows = _recommendations()
    bcc = condition_scoped_candidates("bcc", rows)
    assert {row["recommendation_id"] for row in bcc} == {"box-1", "box-5"}
    melanoma = condition_scoped_candidates("melanoma", rows)
    assert {row["recommendation_id"] for row in melanoma} == {"recommendation-1"}


def test_a_commitment_with_no_guideline_condition_reaches_nothing():
    """nevus/bkl/df/vasc have no catalog condition and must reach zero.

    Silently returning cross-condition records here is the failure that would
    make any citation metric meaningless. akiec is excluded from this list because
    the EADO actinic-keratosis guideline gave it a licensed condition.
    """
    rows = _recommendations()
    for diagnosis in ("nevus", "bkl", "df", "vasc"):
        assert condition_scoped_candidates(diagnosis, rows) == []
        assert diagnosis not in COMMITMENT_TO_CONDITION
    # akiec now maps to a condition, but this fixture holds no AK record, so it
    # still reaches nothing rather than falling back to another condition.
    assert COMMITMENT_TO_CONDITION["akiec"] == "actinic_keratosis"
    assert condition_scoped_candidates("akiec", rows) == []


def test_claim_type_filter_rejects_records_that_do_not_declare_the_type():
    rows = _recommendations()
    assert [row["recommendation_id"] for row in
            condition_scoped_candidates("bcc", rows, "diagnosis")] == ["box-1"]
    assert [row["recommendation_id"] for row in
            condition_scoped_candidates("bcc", rows, "management")] == ["box-5"]
    assert condition_scoped_candidates("bcc", rows, "follow_up") == []


def test_summary_counts_lesions_not_records():
    rows = _recommendations()
    commitments = {"c1": "bcc", "c2": "bcc", "c3": "melanoma", "c4": "nevus"}
    summary = summarize(rows, commitments)
    assert summary["lesions"]["total"] == 4
    reach = summary["reachability"]
    assert reach["any_condition_matched_record"] == 3
    assert reach["zero_condition_matched_record"] == 1
    # melanoma's record is diagnosis-tagged but carries no visual claim type
    assert reach["visual_claim_backable"] == 2


def test_repository_probe_reports_the_measured_catalog_boundary():
    """Guards the three numbers a later reader would otherwise have to trust.

    If the catalog or the frozen perception artifact changes, these move and the
    feasibility verdict must be re-argued rather than inherited.
    """
    report = build_report()
    summary = report["summary"]
    assert summary["catalog"]["records"] == 66
    assert summary["catalog"]["by_condition"] == {
        "basal_cell_carcinoma": 27, "melanoma": 26, "actinic_keratosis": 13}
    assert summary["lesions"]["total"] == 160

    reach = summary["reachability"]
    # 29 bcc + 34 melanoma + 29 akiec committed lesions. The actinic-keratosis
    # guideline is what lifted this from 63: akiec was the only scored HAM7 class
    # with no recommendation coverage at all.
    assert reach["any_condition_matched_record"] == 92
    assert reach["zero_condition_matched_record"] == 68
    # All three covered conditions carry visual/assessment records.
    assert reach["visual_claim_backable"] == 92

    melanoma = summary["catalog"]["per_condition_claim_eligibility"]["melanoma"]
    # The B0 tripwire fired, by design: EADO Part 1 gave melanoma method-licensure
    # records, and NICE (NG14 dermoscopy, QS130 dermoscopy) added more. F3 was
    # re-examined and the image-bound boundary re-confirmed (method licensure is
    # not patient-diagnosis proof), so these are expected to be non-zero.
    assert melanoma["assessment_eligible"] == 7
    assert melanoma["visual_finding_eligible"] == 7
    assert report["boundaries"]["generation_calls"] == 0
    assert report["boundaries"]["truth_fields_consumed"] is False


def test_probe_never_consumes_a_truth_field():
    perception = ROOT / "research/experiments/medguard_h2h_v1/perception/dermlip_v1/predictions.jsonl"
    rows = [json.loads(line) for line in
            perception.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "frozen perception artifact is empty"
    for row in rows:
        assert "label" not in row
        assert "benign_malignant" not in row
