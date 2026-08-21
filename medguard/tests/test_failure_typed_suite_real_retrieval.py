"""Fix 5: F1-F4 failure-typed recovery measured with REAL (live) re-retrieval.

Unlike ``test_failure_typed_suite.py``'s fixture tools (which always "recover" by
scripted construction), these tools rebuild the contract from the same frozen-index
retrieval the H2H runner's method arm uses, so recovery can genuinely fail when a
lesion has no supporting evidence -- that is the point of the fix. Driven entirely by
the fake retrieval/guideline fixtures from ``test_matched_full_h2h_run``: zero paid
calls, no real index load.
"""
from medguard.eval.failure_typed_suite_v1 import (
    STRATA,
    run_failure_suite_real_retrieval,
    score_lesion_real_retrieval,
)
from medguard.tests.test_matched_full_h2h_run import _case, _image_for, _make_deps


def test_f1_f2_f3_recover_via_real_retrieval_when_evidence_exists(tmp_path):
    # c1/melanoma has a matching image exemplar and a melanoma guideline tier
    # (assessment+management+referral) in the shared fake fixtures, so a genuine
    # rebuild from real retrieval finds evidence and reaches PASS.
    deps = _make_deps(tmp_path, image_for=_image_for)
    case = _case("c1", "melanoma")
    for stratum in ("F1_sparse_missing_evidence", "F2_query_evidence_misalignment",
                    "F3_source_contradiction"):
        row = score_lesion_real_retrieval(case, deps, stratum)
        assert row["recovered"] is True, (stratum, row)
        assert row["recovered_final_action"] == "PASS"
        assert row["recovery_calls"], "a real repair tool must have been invoked"


def test_f4_image_conflict_never_attempts_repair_even_with_real_tools(tmp_path):
    deps = _make_deps(tmp_path, image_for=_image_for)
    case = _case("c1", "melanoma")
    row = score_lesion_real_retrieval(case, deps, "F4_image_evidence_conflict")
    assert row["routed_action"] == "ABSTAIN"
    assert row["recovered"] is False
    assert row["recovered_final_action"] == "ABSTAIN"
    assert row["recovery_calls"] == []


def test_recovery_is_genuine_not_scripted_and_can_fail(tmp_path):
    # c2/bcc has a MISMATCHED exemplar under the shared fixtures (_image_for returns
    # "nevus" for any path not containing "c1"), so no retrieved exemplar corroborates
    # the commitment and every rebuild keeps an unresolved uncertainty -> ABSTAIN. This
    # is the behaviour the fixture-tools suite cannot show (it always recovers).
    deps = _make_deps(tmp_path, image_for=_image_for)
    case = _case("c2", "bcc")
    row = score_lesion_real_retrieval(case, deps, "F1_sparse_missing_evidence")
    assert row["recovered"] is False
    assert row["recovered_final_action"] == "ABSTAIN"
    # The routing decision itself is unaffected by which tool later repairs it.
    assert row["routed_action"] == "EXPAND"


def test_suite_aggregates_real_recovery_rate_per_stratum(tmp_path):
    deps = _make_deps(tmp_path, image_for=_image_for)
    cases = [_case("c1", "melanoma"), _case("c2", "bcc")]
    report = run_failure_suite_real_retrieval(cases, deps)
    assert report["lesions"] == 2
    assert len(report["rows"]) == 2 * len(STRATA)
    f1 = report["per_stratum"]["F1_sparse_missing_evidence"]
    # c1 recovers, c2 does not -> a genuine 0.5, unlike the fixture suite's fixed 1.0.
    assert f1["recovery_rate_real_retrieval"] == 0.5
    assert f1["routing_appropriateness"] == 1.0
    f4 = report["per_stratum"]["F4_image_evidence_conflict"]
    assert f4["recovery_rate_real_retrieval"] == 0.0
    assert report["manifest_sha256"] == report["manifest_sha256"]
    assert len(report["manifest_sha256"]) == 64
