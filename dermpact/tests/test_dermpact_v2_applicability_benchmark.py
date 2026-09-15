"""WP-B: tests for the reviewed-v2 applicability benchmark (dermpact/core/eval)."""
from __future__ import annotations

import itertools
import json
from datetime import date

import pytest

from dermpact.core.eval import applicability_benchmark_v2 as bench




def test_variant_names_are_unique_and_match_the_brief():
    names = [v["variant"] for v in bench.VARIANTS]
    assert len(names) == 22
    assert len(set(names)) == len(names)
    expected = {
        "base", "suspected_true", "suspected_false",
        "equivocal_true_rcm_unknown", "equivocal_true_rcm_true", "equivocal_true_rcm_false",
        "equivocal_false_rcm_true", "equivocal_unknown_rcm_true", "under_assessment_false",
        "high_risk_true_count_unknown", "high_risk_true_high_count_true",
        "high_risk_true_count_250_no_assertion", "high_risk_false",
        "iqa_review_required_with_suspected_true", "no_image_no_assertions", "no_image_suspected_true",
        "model_melanoma_no_assertion", "checklist_high_no_assertion", "stale_2027", "jurisdiction_us",
        "wrong_entity", "scope_query_staging",
    }
    assert set(names) == expected
    for spec in bench.VARIANTS:
        assert spec["family"]
        assert isinstance(spec["overrides"], dict)




@pytest.mark.parametrize("equivocal,rcm_available", list(itertools.product([True, False, None], repeat=2)))
def test_r7_full_3x3_truth_table(equivocal, rcm_available):
    """R7 is an AND of two asserted facts; suspected fixed False so R3 falls through."""
    kw = {"clinician_melanoma_suspected": False, "clinician_equivocal": equivocal, "rcm_available": rcm_available}
    gold = bench.compute_gold_for_rules(kw)
    entry = gold["eado-mel-p1-2024:R7"]
    if False in (equivocal, rcm_available):
        assert entry["state"] == "not_applicable"
    elif None in (equivocal, rcm_available):
        assert entry["state"] == "applicability_unknown"
        expected_fact = "clinician.equivocal" if equivocal is None else "site.rcm_available"
        assert entry["decisive_missing_fact"] == expected_fact
    else:
        assert entry["state"] == "applicable"
        assert entry["action_code"] == "reflectance_confocal_microscopy"


@pytest.mark.parametrize("model", ["melanoma", "nevus", "abstained", None])
@pytest.mark.parametrize("image_present", [True, False])
@pytest.mark.parametrize("assertion", [True, False, None])
def test_r3_gold_depends_only_on_the_explicit_assertion(model, image_present, assertion):
    """Gold R3 must track only the clinician assertion, never the model output or image presence."""
    kw = {
        "clinician_melanoma_suspected": assertion,
        "dermoscopic_present": image_present,
        "provisional_class": model,
    }
    gold = bench.compute_gold_for_rules(kw)
    entry = gold["eado-mel-p1-2024:R3"]
    if assertion is None:
        assert entry["state"] == "applicability_unknown"
        assert entry["decisive_missing_fact"] == "clinician.melanoma_suspected"
    elif assertion:
        assert entry["state"] == "applicable"
        assert entry["action_code"] == "histopathological_confirmation"
    else:
        assert entry["state"] == "not_applicable"


def test_r6_boundary_count_does_not_supply_the_clinician_assessment():
    """A numeric total_nevus_count alone must not satisfy R6 (matches test_phase4_reviewed_v2.test_count_example_does_not_supply_clinician_assessment)."""
    kw = {"high_risk_status": True, "total_nevus_count": 250}
    gold = bench.compute_gold_for_rules(kw)
    entry = gold["eado-mel-p1-2024:R6"]
    assert entry["state"] == "applicability_unknown"
    assert entry["decisive_missing_fact"] == "patient.high_total_nevus_count"

    kw2 = dict(kw, high_total_nevus_count=True)
    gold2 = bench.compute_gold_for_rules(kw2)
    assert gold2["eado-mel-p1-2024:R6"]["state"] == "applicable"




@pytest.mark.parametrize("kw,expected_status,expected_rule", [
    ({}, "recommended", "eado-mel-p1-2024:R4"),
    ({"clinician_melanoma_suspected": True}, "recommended", "eado-mel-p1-2024:R3"),
    ({"clinician_melanoma_suspected": False}, "recommended", "eado-mel-p1-2024:R4"),
    ({"clinician_melanoma_suspected": False, "clinician_equivocal": True, "rcm_available": True},
     "recommended", "eado-mel-p1-2024:R7"),
    ({"clinician_melanoma_suspected": False, "clinician_equivocal": True, "rcm_available": False},
     "recommended", "eado-mel-p1-2024:R4"),
    ({"lesion_under_assessment": False}, "conditional", "eado-mel-p1-2024:R3"),
    ({"high_risk_status": True, "high_total_nevus_count": True}, "recommended", "eado-mel-p1-2024:R4"),
    ({"provisional_class": "melanoma"}, "recommended", "eado-mel-p1-2024:R4"),
])
def test_next_step_walk_examples(kw, expected_status, expected_rule):
    gold = bench.compute_gold_for_rules(kw)
    next_step = bench.next_step_walk(gold)
    assert next_step["status"] == expected_status
    assert next_step["rule_id"] == expected_rule


def test_r3_wins_over_r4_when_both_applicable():
    """no_image_suspected_true: R3 must win over R4 by priority, regardless of missing image."""
    kw = {"dermoscopic_present": False, "clinician_melanoma_suspected": True}
    gold = bench.compute_gold_for_rules(kw)
    assert gold["eado-mel-p1-2024:R3"]["state"] == "applicable"
    assert gold["eado-mel-p1-2024:R4"]["state"] == "applicable"
    next_step = bench.next_step_walk(gold)
    assert next_step["rule_id"] == "eado-mel-p1-2024:R3"


def test_model_and_checklist_distractors_cannot_supply_r3():
    """The model class and a forced-positive checklist must not make R3 applicable or not_applicable."""
    gold_model = bench.compute_gold_for_rules({"provisional_class": "melanoma", "melanoma_probability": 0.97})
    assert gold_model["eado-mel-p1-2024:R3"]["state"] == "applicability_unknown"
    gold_checklist = bench.compute_gold_for_rules({"criterion_values": bench.ALL_POSITIVE_CRITERIA})
    assert gold_checklist["eado-mel-p1-2024:R3"]["state"] == "applicability_unknown"


def test_source_expiry_and_jurisdiction_withhold_every_rule():
    for kw in ({"evaluation_date": date(2027, 1, 1)}, {"jurisdiction": "US"}):
        gold = bench.compute_gold_for_rules(kw)
        assert all(entry["state"] == "no_catalogue_coverage" for entry in gold.values())
        assert bench.next_step_walk(gold)["status"] == "withheld_no_coverage"




@pytest.fixture(scope="module")
def small_runs():
    built = bench.build_runs()
    # a handful of cases across the variants that matter for these tests
    wanted_variants = {"base", "suspected_true", "wrong_entity", "scope_query_staging", "stale_2027", "jurisdiction_us"}
    by_case: dict = {}
    for r in built.runs:
        if r["variant"] in wanted_variants:
            by_case.setdefault(r["case_id"], {})[r["variant"]] = r
        if len(by_case) >= 3 and all(len(v) == len(wanted_variants) for v in by_case.values()):
            break
    return by_case


@pytest.fixture(scope="module")
def engine():
    return bench.ReviewedApplicabilityEngine()


def test_wrong_entity_refuses_on_both_deterministic_arms(small_runs, engine):
    for case_id, variants in small_runs.items():
        run = variants["wrong_entity"]
        assert run["expected_refusal"] is True
        assert run["expected_case_id"] == bench.WRONG_ENTITY_SENTINEL
        out_e = bench.run_deterministic_reviewed_v2(engine, run)
        out_f = bench.run_deterministic_forms_template(run)
        assert out_e["refused"] is True
        assert out_f["refused"] is True


def test_scope_query_staging_is_no_coverage_on_both_arms(small_runs, engine):
    for case_id, variants in small_runs.items():
        run = variants["scope_query_staging"]
        out_e = bench.run_deterministic_reviewed_v2(engine, run)
        out_f = bench.run_deterministic_forms_template(run)
        for out in (out_e, out_f):
            assert out["refused"] is False
            assert out["per_rule"]["eado-mel-p1-2024:R1"]["state"] == "no_catalogue_coverage"
            assert out["next_step"]["status"] == "withheld_no_coverage"


def test_expiry_and_jurisdiction_variants_agree_with_gold(small_runs, engine):
    for case_id, variants in small_runs.items():
        for variant in ("stale_2027", "jurisdiction_us"):
            run = variants[variant]
            out_e = bench.run_deterministic_reviewed_v2(engine, run)
            out_f = bench.run_deterministic_forms_template(run)
            for out in (out_e, out_f):
                s = bench.score_run(run, out)
                assert s["no_coverage_correct"] == 1
                if variant == "stale_2027":
                    assert s["source_expiry_correct"] == 1


def test_deterministic_arms_agree_with_gold_on_a_sample(small_runs, engine):
    for case_id, variants in small_runs.items():
        for variant in ("base", "suspected_true"):
            run = variants[variant]
            out_e = bench.run_deterministic_reviewed_v2(engine, run)
            out_f = bench.run_deterministic_forms_template(run)
            for out in (out_e, out_f):
                s = bench.score_run(run, out)
                assert s["wrong_or_inapplicable_next_step"] == 0
                for _, gold_state, arm_state in s["state_triples"]:
                    assert gold_state == arm_state




def test_llm_arms_are_wired_to_the_fake_client_only(small_runs, engine):
    pages = []
    llm = bench.arms_mod.FakeLLMClient()
    arms = bench.arms_mod.Arms(engine.catalogue, source_pages=pages, llm=llm)
    run = next(iter(small_runs.values()))["base"]
    for name in ("direct_llm", "naive_rag", "llm_traversal"):
        out = arms.run_arm(name, run).to_dict()
        assert out["arm"] == name
        assert out["n_llm_calls"] >= 1
    assert llm.name == "fake-llm-plumbing-stub"




def test_main_end_to_end_writes_expected_artifacts(tmp_path):
    rc = bench.main(["--llm-mode", "fake", "--limit", "3", "--out-dir", str(tmp_path)])
    assert rc == 0

    protocol = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["n_base_cases"] == 616
    assert protocol["n_variants_per_case"] == 22
    assert protocol["hard_scope"]["protected_or_final_rows_loaded"] == 0
    assert protocol["hard_scope"]["images_opened"] == 0
    for entry in protocol["sources"]:
        assert len(entry["sha256"]) == 64

    gold_summary = json.loads((tmp_path / "gold_summary.json").read_text(encoding="utf-8"))
    assert gold_summary["n_runs"] == 616 * 22
    assert gold_summary["protected_or_final_rows_loaded"] == 0

    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 616 * 22

    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert set(results["arms"]) == {
        "deterministic_reviewed_v2", "deterministic_forms_template", "direct_llm", "naive_rag", "llm_traversal",
    }
    det = results["arms"]["deterministic_reviewed_v2"]
    assert det["n_runs"] == 616 * 22
    assert "primary_ci" in det
    llm_arm = results["arms"]["direct_llm"]
    assert llm_arm["n_runs"] == 3
    assert "Deferred by user" not in results["deferred_live_llm_command"]  # command text itself, not the heading
    assert "--llm-mode live" in results["deferred_live_llm_command"]

    result_md = (tmp_path / "RESULT.md").read_text(encoding="utf-8")
    assert "Deferred by user: live LLM comparison" in result_md
    assert "Plumbing" in result_md or "plumbing" in result_md


def test_repro_run_is_byte_identical_except_the_frozen_timestamp(tmp_path):
    bench.main(["--llm-mode", "fake", "--limit", "3", "--out-dir", str(tmp_path)])
    bench.main(["--llm-mode", "fake", "--limit", "3", "--out-dir", str(tmp_path), "--out-suffix", "_repro"])

    assert (tmp_path / "runs.jsonl").read_text(encoding="utf-8") == (tmp_path / "runs_repro.jsonl").read_text(encoding="utf-8")

    gold_a = json.loads((tmp_path / "gold_summary.json").read_text(encoding="utf-8"))
    gold_b = json.loads((tmp_path / "gold_summary_repro.json").read_text(encoding="utf-8"))
    # "artifact.path" legitimately differs (it names the suffixed runs file); everything else must match.
    gold_a["artifact"] = dict(gold_a["artifact"], path=None)
    gold_b["artifact"] = dict(gold_b["artifact"], path=None)
    assert gold_a == gold_b

    results_a = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    results_b = json.loads((tmp_path / "results_repro.json").read_text(encoding="utf-8"))
    assert results_a == results_b  # results.json carries no wall-time field

    protocol_a = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    protocol_b = json.loads((tmp_path / "protocol_repro.json").read_text(encoding="utf-8"))
    diffs = {k for k in protocol_a if protocol_a.get(k) != protocol_b.get(k)}
    assert diffs == {"frozen_at_utc"}  # the one allowed wall-time field


def test_live100_sample_is_case_unique_and_status_balanced():
    from collections import Counter
    from dermpact.core.eval import applicability_live100

    rows = applicability_live100.select_runs()
    assert len(rows) == 100
    assert len({row["case_id"] for row in rows}) == 100
    assert Counter(row["gold_next_step"]["status"] for row in rows) == applicability_live100.QUOTAS


def test_live100_cache_filename_is_windows_safe_and_collision_resistant():
    from dermpact.core.eval import applicability_live100

    first = applicability_live100._cache_filename("case-1::model_melanoma_no_assertion")
    second = applicability_live100._cache_filename("case-1/model_melanoma_no_assertion")
    assert not any(char in first for char in '<>:"/\\|?*')
    assert first != second


def test_live100_transport_retry_does_not_replace_a_received_answer(monkeypatch):
    from dermpact.core.eval import applicability_live100

    class Output:
        def to_dict(self):
            return {"answer": "kept"}

    class Runner:
        calls = 0

        def run_arm(self, _arm, _item):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("gateway call failed after 3 attempts: ConnectError")
            return Output()

    monkeypatch.setattr(applicability_live100.time, "sleep", lambda _seconds: None)
    runner = Runner()
    assert applicability_live100._run_transport_resilient(runner, "direct_llm", {}) == {"answer": "kept"}
    assert runner.calls == 3


def test_clarification_ablation100_uses_unique_development_cases():
    from dermpact.core.eval import clarification_ablation100
    from dermpact.core.prototype import inventory as inv

    manifest = inv.load_manifest(clarification_ablation100.CASES)
    case_ids = sorted(str(row["case_id"]) for row in manifest.rows)[:100]
    assert len(case_ids) == len(set(case_ids)) == 100
