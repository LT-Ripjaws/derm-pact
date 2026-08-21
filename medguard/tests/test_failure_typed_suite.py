"""The F1-F4 failure-typed suite routes each injected failure to the correct bounded action.

Host-deterministic, zero paid calls. Uses the perception fixture as the lesion binding.
"""
from medguard.remediation import Action
from medguard.tests.c1_fixtures import perception
from medguard.eval.failure_typed_suite_v1 import (
    STRATUM_EXPECTED_ACTION,
    manifest_sha256,
    run_failure_suite,
    score_lesion,
)


def _binding(dx="melanoma"):
    return perception(diagnosis=dx)


def test_each_stratum_routes_to_its_expected_bounded_action():
    for stratum, expected in STRATUM_EXPECTED_ACTION.items():
        row = score_lesion(_binding(), "lesion:test", stratum)
        assert row["routing_correct"], (stratum, row["routed_action"], expected.value)
        assert row["routed_action"] == expected.value


def test_detected_failure_types_match_the_injection():
    assert "missing_evidence" in score_lesion(_binding(), "l", "F1_sparse_missing_evidence")["detected_failures"]
    assert score_lesion(_binding(), "l", "F2_query_evidence_misalignment")["detected_failures"] == ["misaligned_evidence"]
    assert score_lesion(_binding(), "l", "F3_source_contradiction")["detected_failures"] == ["source_contradiction"]
    assert score_lesion(_binding(), "l", "F4_image_evidence_conflict")["detected_failures"] == ["image_evidence_conflict"]


def test_bounded_safe_termination_when_tools_cannot_repair():
    for stratum in STRATUM_EXPECTED_ACTION:
        row = score_lesion(_binding(), "l", stratum)
        # A loop that cannot repair must ABSTAIN (never free-PASS, never exceed budget).
        assert row["bounded_final_action"] == Action.ABSTAIN.value
        assert row["bounded_steps"] <= 4


def test_recovery_when_tools_repair_except_image_conflict_which_abstains():
    # F1/F2/F3 are repairable -> reach PASS; F4 (image conflict) is not -> safe ABSTAIN.
    assert score_lesion(_binding(), "l", "F1_sparse_missing_evidence")["recovered"]
    assert score_lesion(_binding(), "l", "F2_query_evidence_misalignment")["recovered"]
    assert score_lesion(_binding(), "l", "F3_source_contradiction")["recovered"]
    f4 = score_lesion(_binding(), "l", "F4_image_evidence_conflict")
    assert not f4["recovered"] and f4["recovered_final_action"] == Action.ABSTAIN.value


def test_suite_aggregates_and_manifest_is_stable():
    bindings = {f"lesion:c{i}": _binding(dx) for i, dx in enumerate(("melanoma", "bcc", "nevus"))}
    report = run_failure_suite(bindings)
    assert report["lesions"] == 3
    for stratum in STRATUM_EXPECTED_ACTION:
        block = report["per_stratum"][stratum]
        assert block["routing_appropriateness"] == 1.0
        assert block["bounded_abstain_rate"] == 1.0
    # F4 correctly never "recovers" (it abstains); F1-F3 fully recover with fixture tools.
    assert report["per_stratum"]["F4_image_evidence_conflict"]["recovery_rate_fixture_tools"] == 0.0
    assert report["per_stratum"]["F1_sparse_missing_evidence"]["recovery_rate_fixture_tools"] == 1.0
    # Manifest hash is deterministic (frozen injection spec).
    assert manifest_sha256() == report["manifest_sha256"]
    assert len(manifest_sha256()) == 64
