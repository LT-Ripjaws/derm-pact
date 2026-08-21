"""Boundary and artifact tests for the offline 160-lesion preflight."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research/experiments/medguard_h2h_v1/development_preflight.json"
FORBIDDEN_REPORT_KEYS = {"label", "truth", "target", "ground_truth", "image_path"}


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_development_preflight_report_has_exact_offline_inventory():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "medguard-development-preflight-v1"
    assert report["boundaries"] == {
        "development_only": True,
        "sealed_test_access": False,
        "test_records_loaded": 0,
        "truth_fields_consumed": False,
        "generation_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "paid_calls": 0,
    }
    assert report["aggregate"]["cases"] == 160
    assert report["aggregate"]["views"] == 244
    assert len(report["per_case"]) == 160
    assert len({row["case_id"] for row in report["per_case"]}) == 160
    assert sum(row["view_count"] for row in report["per_case"]) == 244
    assert all(row["retrieval"]["all_views_preserved"] for row in report["per_case"])
    assert all(row["retrieval"]["final_evidence"] == 5 for row in report["per_case"])
    assert all(row["generation_calls"] == 0 for row in report["per_case"])


def test_development_preflight_report_does_not_expose_truth_or_local_images():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for mapping in _walk(report):
        assert not (set(mapping) & FORBIDDEN_REPORT_KEYS)
    assert report["retrieval"]["bundle_verified"] is True
    assert report["aggregate"]["valid_contracts"] == 160
    # 130 of 160: the image-bound diagnosis is discharged by the frozen perception
    # commitment and corroborated by at least one matching train exemplar. The
    # remaining 30 retain an unresolved uncertainty because no retrieved exemplar
    # carries the committed diagnosis, and abstain on that image-evidence conflict.
    assert report["aggregate"]["structured_pass_eligible"] == 130
    assert report["aggregate"]["contract_complete"] == 160
    assert report["aggregate"]["verification_failure_case_counts"] == {
        "unresolved_uncertainty": 30}
    assert report["aggregate"]["abstained"] == 30
    assert (report["aggregate"]["structured_pass_eligible"]
            + report["aggregate"]["abstained"]) == 160


def test_development_preflight_wires_the_authoritative_guideline_channel():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    guideline = report["retrieval"]["guideline"]
    assert guideline["channel"] == "authoritative_guideline_catalog"
    assert guideline["records"] == 66 and guideline["guideline_k"] == 3
    agg = report["aggregate"]
    # The 92 lesions committed to a covered condition (bcc/melanoma/akiec) each carry
    # exactly GUIDELINE_K source-statement-integrity citations; benign commitments none.
    # Selection is claim-type-stratified (select_guideline_evidence), so citations lead
    # with the assessment method the question asks about and add management context,
    # rather than being dominated by whichever source_id sorts first alphabetically.
    assert agg["lesions_with_guideline_claims"] == 92
    assert agg["guideline_claims_total"] == 276
    assert agg["guideline_claim_type_counts"] == {"assessment": 92, "management": 184}
    # Guideline claims are SUPPORTED provenance but do not change PASS-eligibility,
    # which stays gated on the image-bound diagnosis + exemplar corroboration.
    assert agg["relation_counts"]["supported"] == 276
    assert agg["structured_pass_eligible"] == 130
    # The fused ceiling is unchanged; guidelines are a separate appended tier.
    assert all(row["retrieval"]["final_evidence"] == 5 for row in report["per_case"])
