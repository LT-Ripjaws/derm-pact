"""Focused tests for the matched full-arm validate-only runner.

The pure-configuration tests always run.  The tests that read the emitted report
skip when it is absent so this module stays runnable without the frozen
scientific stack.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from medguard.eval.h2h_protocol import ArmId, reject_truth_keys
from medguard.eval.matched_full_h2h_validate import (
    ARMS,
    MEDGUARD_ARMS,
    MEDGUARD_CONFIGS,
    OUTPUT,
    MedGuardArmConfig,
    single_component_deltas,
)


def _report():
    if not Path(OUTPUT).is_file():
        pytest.skip("matched full-arm report has not been generated in this environment")
    return json.loads(Path(OUTPUT).read_text(encoding="utf-8"))


def test_roster_is_the_frozen_card_roster_with_a0_declared_as_an_alias():
    assert [arm.value for arm in ARMS] == [
        "B0", "B1", "B4", "B5", "B6", "M", "A0", "A1", "A2", "A3", "A4",
    ]
    assert [arm.value for arm in MEDGUARD_ARMS] == ["M", "A0", "A1", "A2", "A3", "A4"]
    # A0 has no configuration of its own; it can only ever be M's.
    assert ArmId.A0 not in MEDGUARD_CONFIGS


def test_each_ablation_changes_exactly_one_declared_component():
    deltas = single_component_deltas()
    assert deltas == {
        "A1": ["image_exemplar_retrieval"],
        "A2": ["graph_lineage"],
        "A3": ["bounded_remediation"],
        "A4": ["verification_gate"],
    }
    assert all(MEDGUARD_CONFIGS[ArmId.M].component_flags.values())


def test_a_two_component_ablation_is_rejected(monkeypatch):
    broken = dict(MEDGUARD_CONFIGS)
    broken[ArmId.A1] = MedGuardArmConfig(
        ArmId.A1, image_exemplar_retrieval=False, graph_lineage=False,
    )
    monkeypatch.setattr(
        "medguard.eval.matched_full_h2h_validate.MEDGUARD_CONFIGS", broken,
    )
    with pytest.raises(ValueError, match="exactly one component"):
        single_component_deltas()


def test_two_ablations_may_not_disable_the_same_component(monkeypatch):
    broken = dict(MEDGUARD_CONFIGS)
    broken[ArmId.A2] = MedGuardArmConfig(ArmId.A2, image_exemplar_retrieval=False)
    monkeypatch.setattr(
        "medguard.eval.matched_full_h2h_validate.MEDGUARD_CONFIGS", broken,
    )
    with pytest.raises(ValueError, match="same component"):
        single_component_deltas()


def test_report_covers_the_frozen_case_set_with_zero_generation():
    report = _report()
    assert report["aggregate"]["cases"] == 160
    assert report["aggregate"]["views"] == 244
    assert report["aggregate"]["arms"] == 11
    assert report["aggregate"]["arm_rows"] == 1760
    boundaries = report["boundaries"]
    assert boundaries["generation_calls"] == 0
    assert boundaries["logical_generation_calls"] == 0
    assert boundaries["prompt_tokens"] == 0
    assert boundaries["completion_tokens"] == 0
    assert boundaries["paid_calls"] == 0
    assert boundaries["sealed_test_access"] is False
    assert boundaries["test_records_loaded"] == 0
    assert report["status"] == "validate_only_not_a_performance_result"


def _keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)
    return found


def test_report_carries_no_truth_field_and_reports_no_outcome_metric():
    report = _report()
    reject_truth_keys(report, path="report")
    # Check for outcome *fields*, not substrings: the report's own disclaimer
    # legitimately contains the word "accuracy" while denying any such result.
    keys = _keys(report)
    for forbidden in (
        "accuracy", "macro_f1", "balanced_accuracy", "auroc", "aurc",
        "ragas_faithfulness", "citation_precision_score", "selective_accuracy",
        "safety_improvement",
    ):
        assert forbidden not in keys
    assert report["interpretation"]["not_claimed"]
    assert "no accuracy" in report["interpretation"]["not_claimed"].lower()


def test_every_arm_is_separately_keyed_except_the_declared_m_a0_alias():
    report = _report()
    namespaces = report["protocol"]["cache_namespaces"]
    assert namespaces["A0"] == namespaces["M"]
    assert len(set(namespaces.values())) == len(ARMS) - 1
    for row in report["cases"]:
        keys = {arm.value: row["arms"][arm.value]["request_key"] for arm in ARMS}
        assert keys["A0"] == keys["M"]
        assert len(set(keys.values())) == len(ARMS) - 1


def test_a0_execution_rows_differ_from_m_only_by_the_audit_alias():
    report = _report()
    for row in report["cases"]:
        a0 = copy.deepcopy(row["arms"]["A0"])
        assert a0.pop("alias_of") == "M"
        left = json.dumps(a0, sort_keys=True, separators=(",", ":"))
        right = json.dumps(row["arms"]["M"], sort_keys=True, separators=(",", ":"))
        assert left == right
    assert report["verification"]["a0_execution_payload_equivalent_to_m"] is True
    assert report["verification"][
        "a0_final_report_row_differs_only_by_alias_annotation"
    ] is True


def test_all_arms_receive_identical_ordered_images_and_frozen_perception():
    report = _report()
    total_views = 0
    for row in report["cases"]:
        case_input = row["input"]
        # One shared truth-redacted input object backs every arm on the case, so
        # the ordered views and the frozen commitment cannot diverge per arm.
        assert case_input["view_count"] == len(case_input["ordered_view_sha256s"])
        assert case_input["view_count"] == len(case_input["ordered_view_case_ids"])
        assert case_input["committed_diagnosis"]
        assert len(case_input["perception_artifact_sha256"]) == 64
        assert len(case_input["perception_record_sha256"]) == 64
        assert len(case_input["perception_distribution_sha256"]) == 64
        assert len(case_input["input_lineage_sha256"]) == 64
        total_views += case_input["view_count"]
    assert total_views == 244
    # Perception is frozen: one artifact digest is shared by every case and arm.
    artifacts = {row["input"]["perception_artifact_sha256"] for row in report["cases"]}
    assert len(artifacts) == 1
    assert artifacts == {report["inputs"]["perception_artifact_sha256"]}
    assert report["verification"]["identical_ordered_images_and_frozen_perception"] is True
    assert report["verification"]["truth_fields_in_arm_visible_inputs"] == 0


def test_b5_returns_real_graph_paths_rather_than_flat_provenance():
    report = _report()
    checks = report["verification"]["b5_is_not_flat_provenance"]
    assert checks["node_types_present"] == ["concept", "evidence", "source", "version"]
    assert checks["concept_edges"] > 0
    assert checks["returned_paths_with_concept_head"] == checks["returned_paths_total"]
    assert report["aggregate"]["b5_final_evidence"] == 800
    for row in report["cases"]:
        for evidence in row["arms"]["B5"]["retrieval"]["evidence"]:
            path = evidence["graph_path"]
            # concept -> evidence -> source -> version, with lineage bound.
            assert path["hops"] == 3
            assert len(path["node_ids"]) == 4
            assert len(path["edge_ids"]) == 3
            assert path["source_id"] and path["source_version"]
            assert len(path["content_sha256"]) == 64
            assert len(path["record_sha256"]) == 64
    # Graph membership genuinely constrains the pool below the whole corpus.
    assert report["aggregate"]["b5_mean_reached_evidence"] < checks["corpus_records"]


def test_b6_loop_is_bounded_and_depends_on_the_b5_graph():
    report = _report()
    bounded = report["verification"]["b6_loop_bounded"]
    assert bounded["maximum_attempts"] == 4
    assert bounded["maximum_refinements"] == 3
    assert bounded["observed_maximum_attempts"] <= 4
    assert bounded["observed_maximum_refinements"] <= 3
    assert bounded["reuses_medguard_failure_labels"] is False
    graph_sha256 = report["runtime"]["graph"]["graph_sha256"]
    for row in report["cases"]:
        audit = row["arms"]["B6"]["retrieval"]["audit"]
        assert audit["graph_sha256"] == graph_sha256
        assert 1 <= audit["attempts_used"] <= 4
        assert 0 <= audit["refinements_used"] <= 3
        assert len(audit["attempts"]) == audit["attempts_used"]
        for attempt in audit["attempts"]:
            weights = attempt["sufficiency"]["weights"]
            assert weights == {
                "semantic_relevance": 0.4, "coverage": 0.3, "density": 0.3,
            }
        assert row["arms"]["B6"]["call_accounting"]["reserved_logical_calls"] == 4


def test_medguard_graph_lineage_is_complete_and_a2_keeps_flat_provenance():
    report = _report()
    checks = report["verification"]
    assert checks["graph_lineage_paths_expected"] > 0
    assert checks["graph_lineage_paths_complete"] == checks["graph_lineage_paths_expected"]
    for row in report["cases"]:
        for arm in ("M", "A1", "A3", "A4"):
            assert row["arms"][arm]["graph_lineage"]["enabled"] is True
            for evidence in row["arms"][arm]["retrieval"]["evidence"]:
                path = evidence["graph_path"]
                assert path["source_id"] and path["source_version"]
                assert len(path["content_sha256"]) == 64
                assert len(path["record_sha256"]) == 64
        a2 = row["arms"]["A2"]
        assert a2["graph_lineage"]["enabled"] is False
        assert a2["graph_lineage"]["flat_provenance_retained"] is True
        for evidence in a2["retrieval"]["evidence"]:
            assert "graph_path" not in evidence
            # Flat provenance and citation identity survive the graph ablation.
            assert evidence["source_id"] and evidence["source_version"]
            assert evidence["citation_id"] and evidence["authority"]
            assert len(evidence["content_sha256"]) == 64


def test_a1_removes_only_the_image_channel_and_keeps_the_shared_ceiling():
    report = _report()
    for row in report["cases"]:
        m_channels = [
            evidence["retrieval_channel"]
            for evidence in row["arms"]["M"]["retrieval"]["evidence"]
        ]
        a1_channels = [
            evidence["retrieval_channel"]
            for evidence in row["arms"]["A1"]["retrieval"]["evidence"]
        ]
        assert "image_exemplar" in m_channels
        assert "image_exemplar" not in a1_channels
        # The final evidence ceiling is unchanged; A1 gets no extra budget.
        assert len(a1_channels) == len(m_channels) == 5
        assert row["arms"]["A1"]["retrieval"]["candidate_k"] == (
            row["arms"]["M"]["retrieval"]["candidate_k"]
        )
        # A1 keeps the identical ordered images and frozen perception.
        assert row["input"]["ordered_view_sha256s"]


def test_a3_disables_repair_only_and_a4_records_an_ungated_verifier():
    report = _report()
    for row in report["cases"]:
        a3 = row["arms"]["A3"]
        assert a3["remediation"]["enabled"] is False
        assert a3["remediation"]["applied_actions"] == []
        assert a3["verification"]["executed"] is True
        assert a3["verification"]["gate_active"] is True
        # A3 removes repair, not verification. A case whose contract is already
        # satisfied on the first attempt needs no repair and still passes; only a
        # case that would have required a repair abstains.
        assert a3["final_action"] in {"PASS", "ABSTAIN"}
        if a3["final_action"] == "PASS":
            assert a3["contract"]["structured_pass_eligible"] is True

        a4 = row["arms"]["A4"]
        assert a4["verification"]["executed"] is True
        assert a4["verification"]["gate_active"] is False
        assert a4["final_action"] in {"ANSWER_UNGATED", "STRUCTURED_OUTPUT_ERROR"}
        # The verifier still runs and is recorded; only its authority is removed.
        assert a4["verification"]["executed"] is True


def test_missing_scores_stay_missing_and_are_never_zero():
    report = _report()
    for arm in ARMS:
        summary = report["protocol"]["missing_scores"][arm.value]
        assert summary["total_count"] == 160
        assert summary["observed_count"] == 0
        assert summary["missing_count"] == 160
        assert summary["mean"] is None
    assert report["verification"]["missing_score_accounting"] == (
        "explicit_none_never_zero"
    )
    for row in report["cases"]:
        for arm in ARMS:
            state = row["arms"][arm.value]["score_state"]
            assert state["status"] == "not_executed_validate_only"
            assert all(value is None for value in state["values"].values())


def test_call_and_token_ceilings_are_exact_and_zero_under_validate_only():
    report = _report()
    for arm in ARMS:
        assert report["verification"]["call_ceiling_per_arm"][arm.value] == 0
        assert report["verification"]["token_ceiling_per_arm"][arm.value] == 0
    for row in report["cases"]:
        assert row["usage"] == {
            "actual_generation_calls": 0,
            "actual_prompt_tokens": 0,
            "actual_completion_tokens": 0,
            "logical_generation_calls": 0,
            "logical_prompt_tokens": 0,
            "logical_completion_tokens": 0,
        }
        for arm in ARMS:
            accounting = row["arms"][arm.value]["call_accounting"]
            assert accounting["actual_attempts"] == 0
            assert accounting["logical_calls"] == 0
            assert accounting["prompt_tokens"] == 0
            assert accounting["completion_tokens"] == 0


def test_neither_b5_nor_b6_is_labelled_a_reproduction():
    report = _report()
    for arm in ("B5", "B6"):
        fidelity = report["fidelity"][arm]
        assert fidelity["faithful_reproduction"] is False
        assert fidelity["classification"] == "compatible_adaptation"
        assert "reproduction" not in fidelity["reporting_label"].lower()
