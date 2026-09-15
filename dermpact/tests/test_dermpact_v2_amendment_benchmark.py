"""WP-E item 1: tests for the amendment benchmark v2 (dermpact/core/eval)."""
from __future__ import annotations

import json

import pytest

from dermpact.core.eval import amendment_benchmark_v2 as bench
from dermpact.criteria.schema import CRITERIA_BY_NAME, INDETERMINATE




def test_ledger_nonce_for_deterministic_and_distinct():
    n1 = bench.ledger_nonce_for("proto-sha", "case-1", "corrective")
    n2 = bench.ledger_nonce_for("proto-sha", "case-1", "corrective")
    n3 = bench.ledger_nonce_for("proto-sha", "case-1", "harmful")
    n4 = bench.ledger_nonce_for("proto-sha", "case-2", "corrective")
    assert n1 == n2
    assert n1 != n3
    assert n1 != n4


@pytest.mark.parametrize("criterion", list(CRITERIA_BY_NAME))
def test_wrong_value_for_always_changes_points(criterion):
    spec = CRITERIA_BY_NAME[criterion]
    for true_value in sorted(spec.vocabulary):
        wrong = bench.wrong_value_for(criterion, true_value)
        assert wrong in spec.vocabulary
        assert spec.score_for(wrong) != spec.score_for(true_value)
    # indeterminate true value: any wrong value is acceptable (no polarity to violate)
    wrong = bench.wrong_value_for(criterion, INDETERMINATE)
    assert wrong in spec.vocabulary


def test_gold_next_step_for_assertion_matches_R3_R4_walk():
    suspected_true = bench.gold_next_step_for_assertion(True)
    assert suspected_true == {"status": "recommended", "rule_id": "eado-mel-p1-2024:R3",
                              "action_code": "histopathological_confirmation"}
    suspected_false = bench.gold_next_step_for_assertion(False)
    assert suspected_false == {"status": "recommended", "rule_id": "eado-mel-p1-2024:R4",
                               "action_code": "dermatoscopic_assessment"}


def test_case_clustered_bootstrap_point_estimate_and_empty_input():
    by_case = {"a": [1, 1], "b": [0], "c": [1]}
    result = bench.case_clustered_bootstrap(by_case, n_boot=200, seed=1)
    assert result["n_cases"] == 3
    assert result["estimate"] == pytest.approx(3 / 4)
    assert 0.0 <= result["ci_low"] <= result["estimate"] <= result["ci_high"] <= 1.0
    assert result["boundary_case_bound_one_sided_95"] is None

    empty = bench.case_clustered_bootstrap({}, n_boot=200, seed=1)
    assert empty["n_cases"] == 0
    assert empty["estimate"] != empty["estimate"]  # nan
    assert empty["boundary_case_bound_one_sided_95"] is None


def test_case_clustered_bootstrap_reports_non_degenerate_zero_event_bound():
    result = bench.case_clustered_bootstrap({f"c{i}": [0] for i in range(10)}, n_boot=50, seed=2)
    assert result["ci_low"] == result["ci_high"] == 0.0
    bound = result["boundary_case_bound_one_sided_95"]
    assert bound["kind"] == "upper"
    assert 0.0 < bound["value"] < 1.0




LIMIT = 8


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("amendment_benchmark_v2_small")
    rc = bench.main(["--limit", str(LIMIT), "--out-dir", str(out_dir)])
    assert rc == 0
    protocol = json.loads((out_dir / "protocol.json").read_text(encoding="utf-8"))
    curves = json.loads((out_dir / "curves.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    result_md = (out_dir / "RESULT.md").read_text(encoding="utf-8")
    return out_dir, protocol, curves, rows, result_md


def test_protocol_hard_scope_is_zero_and_dataset_size_pinned(small_run):
    _out_dir, protocol, _curves, _rows, _result_md = small_run
    hard = protocol["hard_scope_checks"]
    assert hard["derm7pt_development_rows"] == 616
    assert hard["derm7pt_official_test_rows_loaded"] == 0
    assert hard["ham_rows_loaded"] == 0
    assert hard["bcn_rows_loaded"] == 0
    assert hard["ph2_rows_loaded"] == 0
    assert hard["images_opened"] == 0
    assert hard["models_loaded"] == 0
    assert hard["live_llm_calls_made"] == 0
    assert protocol["status"] == "PRESPECIFIED_BEFORE_RESULTS"
    assert "module_hashes" in protocol and len(protocol["module_hashes"]) > 0


def test_curves_n_cases_matches_limit(small_run):
    _out_dir, _protocol, curves, _rows, _result_md = small_run
    assert curves["n_cases"] == LIMIT


def test_harmful_checklist_corrections_never_change_next_step(small_run):
    _out_dir, _protocol, curves, _rows, _result_md = small_run
    harmful = curves["harmful_checklist_corrections"]
    assert harmful["next_step_changed_count"] == 0
    assert harmful["acknowledgement_required_count"] == 0
    assert harmful["n_committed_opportunities"] > 0
    assert harmful["next_step_changed_rate"]["estimate"] == 0.0
    assert harmful["polarity_check"]["rate"] == pytest.approx(1.0)


def test_structural_blocks_are_all_1(small_run):
    _out_dir, _protocol, curves, _rows, _result_md = small_run
    for name in bench.STRUCTURAL_BLOCK_NAMES:
        row = curves["structural_blocks"][name]
        assert row["estimate"] == pytest.approx(1.0), f"{name}: {row}"


def test_preview_not_commit_head_unchanged(small_run):
    _out_dir, _protocol, curves, _rows, _result_md = small_run
    preview = curves["preview_not_commit"]
    assert preview["head_unchanged_rate"]["estimate"] == pytest.approx(1.0)
    assert preview["n_disk_verified"] == LIMIT
    assert preview["n_in_memory_fallback"] == 0


def test_assertion_route_matches_gold(small_run):
    _out_dir, _protocol, curves, _rows, _result_md = small_run
    route = curves["assertion_route"]
    assert route["truth_aligned_matches_gold"]["estimate"] == pytest.approx(1.0)
    assert route["truth_negated_matches_gold"]["estimate"] == pytest.approx(1.0)


def test_corrective_checklist_curve_never_reports_next_step_fields(small_run):
    _out_dir, _protocol, curves, rows, _result_md = small_run
    ce = curves["checklist_evidence_curve"]
    assert set(ce) == {"decidable_rate", "meets_threshold_matches_truth_rate", "agreement_present_rate"}
    assert [row["k"] for row in ce["decidable_rate"]] == list(range(8))
    for row in rows:
        if row["curve"] == "corrective":
            for pt in row["points"]:
                assert set(pt) >= {"k", "decidable", "meets_threshold", "meets_threshold_matches_truth", "agreement_present"}
            for r in row["rows"]:
                assert "next_step" not in r and "matches_gold" not in r


def test_result_markdown_states_v1_curve_is_superseded(small_run):
    _out_dir, _protocol, _curves, _rows, result_md = small_run
    assert "NOT a current figure" in result_md
    assert "0.667" in result_md and "0.946" in result_md
    assert "single held-out fold" in result_md or "single_member_oof" in result_md
    assert "deployment" in result_md


def test_repro_run_is_byte_identical_except_the_frozen_timestamp(tmp_path):
    bench.main(["--limit", "6", "--out-dir", str(tmp_path)])
    bench.main(["--limit", "6", "--out-dir", str(tmp_path), "--out-suffix", "_repro"])

    assert (tmp_path / "rows.jsonl").read_text(encoding="utf-8") == (tmp_path / "rows_repro.jsonl").read_text(encoding="utf-8")

    curves_a = json.loads((tmp_path / "curves.json").read_text(encoding="utf-8"))
    curves_b = json.loads((tmp_path / "curves_repro.json").read_text(encoding="utf-8"))
    assert curves_a == curves_b

    protocol_a = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    protocol_b = json.loads((tmp_path / "protocol_repro.json").read_text(encoding="utf-8"))
    diffs = {k for k in protocol_a if protocol_a.get(k) != protocol_b.get(k)}
    assert diffs == {"frozen_at_utc"}
