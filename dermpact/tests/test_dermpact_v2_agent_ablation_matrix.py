from __future__ import annotations

from dermpact.core.eval import agent_ablation_matrix as matrix
from dermpact.core.eval import agent_component_ablation_eval as components
from dermpact.core.memory import ComparisonPolicy


def test_host_capability_dependency_map():
    assert matrix._requires_host_capability({"task_id": "case_comparison-p1", "family": "case_comparison"})
    assert matrix._requires_host_capability({"task_id": "review-p1", "family": "explain_deferral"})
    assert not matrix._requires_host_capability({"task_id": "evidence_report-p1", "family": "evidence_report"})


def test_rate_handles_empty_and_nonempty():
    assert matrix._rate([], "ok")["rate"] is None
    assert matrix._rate([{"ok": True}, {"ok": False}], "ok")["rate"] == 0.5


def test_wrong_case_detection():
    assert matrix._wrong_case_hit("derm7pt-case-22", "This concerns derm7pt-case-99.")
    assert not matrix._wrong_case_hit("derm7pt-case-22", "This concerns derm7pt-case-22.")


def test_component_mutators_remove_only_the_named_capability():
    class Host:
        memory_bank = object()
        memory_policy = ComparisonPolicy()
        jobs = object()

    host = Host()
    components._mutator("without_comparison_memory")(host)
    assert host.memory_bank is None and host.memory_policy is None

    host = Host()
    components._mutator("without_memory_counterexample")(host)
    assert host.memory_bank is not None
    assert host.memory_policy.counterexample_min_similarity == 1.0

    host = Host()
    components._mutator("without_measured_xai")(host)
    assert host.jobs is None
