from __future__ import annotations

from dermpact.core.eval import agent_ragas_eval as ragas_eval


def test_aggregate_keeps_arms_and_passes_separate():
    samples = [
        {"arm": arm, "task_id": "t1"}
        for arm in ragas_eval.ARMS
    ]
    passes = [
        {"scores": {"faithfulness": {"per_sample": values}}}
        for values in ([1.0, 0.5, 0.25], [0.5, 0.5, 0.25], [1.0, 1.0, 0.5])
    ]
    result = ragas_eval._aggregate(passes, samples)
    assert result["full_bounded_agent"]["mean_across_passes"] == 2.5 / 3
    assert result["same_backbone_no_tools"]["mean_across_passes"] == 2.0 / 3
    assert result["naive_static_rag"]["mean_across_passes"] == 1.0 / 3


def test_usage_ignores_non_numeric_fields():
    value = ragas_eval._usage([{"succeeded": True, "latency_s": 1.0,
                                "usage": {"prompt_tokens": 10, "details": {"cached": 2}}}])
    assert value["numeric"] == {"prompt_tokens": 10.0}
    assert value["failed_attempts"] == 0


def test_pass_count_must_be_positive(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="n_passes"):
        ragas_eval.run(out_dir=tmp_path, n_passes=0, arms=("full_bounded_agent",))


def test_confirmatory_subset_excludes_adversarial_and_action_tasks():
    assert "evidence_report" in ragas_eval.ELIGIBLE_FAMILIES
    assert "case_comparison" in ragas_eval.ELIGIBLE_FAMILIES


def test_context_projection_renders_tool_propositions_without_adding_facts():
    raw = '{"tool":"read_measured_xai","ok":true,"binding":{"case_id":"c1","record_version":1},"propositions":[{"id":"p1","path_or_ref":"xai.region.delta","value":0.0}]}'
    rendered = ragas_eval._render_context(raw)
    assert "Tool read_measured_xai returned ok=True" in rendered
    assert "Proposition p1: xai.region.delta = 0.0" in rendered


def test_context_projection_excludes_circular_final_summary_proposition():
    raw = '{"tool":"read_measured_xai","ok":true,"binding":{"case_id":"c1","record_version":1},"propositions":[{"id":"p1","path_or_ref":"xai.region.summary.grounded_summary_text","value":"identical final answer"},{"id":"p2","path_or_ref":"xai.region.delta","value":0.0}]}'
    rendered = ragas_eval._render_context(raw)
    assert "identical final answer" not in rendered
    assert "xai.region.delta = 0.0" in rendered


def test_memory_context_projection_preserves_case_similarity_relation():
    raw = (
        '{"tool":"retrieve_case_comparisons","ok":true,'
        '"binding":{"case_id":"query-1","record_version":1},'
        '"data":{"matches":[{"case_id":"reference-7","similarity":0.821068,'
        '"diagnosis":{"status":"match"},"outcome":{"diagnosis":"NEV"},'
        '"matched_fields":["pigmentation"],"different_fields":["streaks"]}],'
        '"counterexample":{"case_id":"reference-8","similarity":0.814968,'
        '"outcome":{"diagnosis":"MEL"}},'
        '"descriptive_reference_label_counts":{"n_retrieved":3,'
        '"counts":{"MEL":1,"NEV":2},'
        '"diagnostic_effect":"not_used_to_alter_current_prediction"}},'
        '"propositions":[]}'
    )
    rendered = ragas_eval._render_context(raw)
    assert "reference-7 has similarity 0.821068 and dataset label NEV" in rendered
    assert "reference-8 has similarity 0.814968 and dataset label MEL" in rendered
    assert "3 returned reference cases have dataset-label counts MEL=1, NEV=2" in rendered


def test_httpx_judge_retries_empty_success_response(monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self.content}}], "usage": {}}

    responses = iter([Response(""), Response("[1]")])

    class Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return next(responses)

    monkeypatch.setenv("DERMPACT_ALLOW_LIVE_LLM", "1")
    monkeypatch.setattr(ragas_eval.config, "load_env", lambda: None)
    monkeypatch.setattr(ragas_eval.config, "GATEWAY_API_KEY", "configured-for-test")
    monkeypatch.setattr(ragas_eval.config, "GATEWAY_BASE_URL", "https://example.invalid")
    monkeypatch.setattr(ragas_eval.httpx, "Client", Client)
    client = ragas_eval.HttpxJudgeClient(retries=1)
    assert client.chat([], max_tokens=10, temperature=0.0, seed=0) == "[1]"
    assert [call["succeeded"] for call in client.call_log] == [False, True]