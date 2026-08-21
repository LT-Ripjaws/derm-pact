"""RAGAS stays supporting, maps fields correctly, and never calls a real judge."""
import pytest

from medguard import config
from medguard.eval.ragas_adapter import (
    RAGAS_ROLE,
    RagasAdapterError,
    RagasJudgeLock,
    RagasSupportingAdapter,
    map_arm_case_to_sample,
)


def _sample():
    return map_arm_case_to_sample(
        case_id="lesion:1", arm_id="M", question="What is the diagnosis?",
        answer="Decision: bcc. Excise with a clear margin.",
        contexts=[{"text": "Excise BCC with a clear margin."}, "plain context string"],
        reference="bcc",
    )


def test_field_mapping_uses_ragas_names_and_drops_empty_contexts():
    sample = map_arm_case_to_sample(
        case_id="lesion:1", arm_id="M", question="q", answer="a",
        contexts=[{"text": "first"}, {"text": "  "}, "second", {"notext": 1}],
    )
    row = sample.to_ragas_row()
    assert set(row) == {"user_input", "response", "retrieved_contexts"}
    assert row["user_input"] == "q"
    assert row["response"] == "a"
    # Empty / textless contexts are dropped; order is preserved.
    assert row["retrieved_contexts"] == ["first", "second"]
    # No reference supplied -> the key is absent, not empty.
    assert "reference" not in row


def test_reference_included_when_supplied():
    assert _sample().to_ragas_row()["reference"] == "bcc"


def test_default_adapter_maps_but_does_not_execute():
    adapter = RagasSupportingAdapter()
    result = adapter.evaluate([_sample()])
    assert result["role"] == RAGAS_ROLE == "supporting"
    assert result["status"] == "mapped_not_executed"
    assert result["scores"] is None
    assert result["n_samples"] == 1
    # The judge lock pins the study's judge model from config.
    assert result["judge_lock"]["judge_model"] == config.JUDGE_MODEL


def test_lock_hash_is_stable_and_sensitive():
    a = RagasJudgeLock(judge_model_version="build-1")
    b = RagasJudgeLock(judge_model_version="build-1")
    c = RagasJudgeLock(judge_model_version="build-2")
    assert a.lock_sha256 == b.lock_sha256
    assert a.lock_sha256 != c.lock_sha256


class _FakeJudge:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def evaluate(self, rows, metrics):
        self.calls += 1
        return {metric: 1.0 for metric in metrics}


def test_injected_judge_must_match_the_lock():
    adapter = RagasSupportingAdapter(judge=_FakeJudge(model="some/other-model"))
    with pytest.raises(RagasAdapterError):
        adapter.evaluate([_sample()])


def test_injected_matching_judge_executes():
    lock = RagasJudgeLock()
    judge = _FakeJudge(model=lock.judge_model)
    adapter = RagasSupportingAdapter(lock, judge=judge)
    result = adapter.evaluate([_sample()])
    assert result["status"] == "executed"
    assert judge.calls == 1
    assert set(result["scores"]) == set(lock.metrics)
