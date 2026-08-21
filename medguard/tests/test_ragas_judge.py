"""The executable RAGAS faithfulness judge, driven by a fake LLM client (zero paid).

Verifies the two-step faithfulness computation (statement extraction -> verification),
the adapter integration (status flips to 'executed'), the lock/model guard, and the
honest-None behaviour on empty context.
"""
import json

import pytest

from medguard import config
from medguard.eval.ragas_adapter import RagasJudgeLock, RagasSupportingAdapter, RagasSample
from medguard.eval.ragas_judge import (
    RagasFaithfulnessJudge,
    assessment_text_for_faithfulness,
)


class FakeJudgeClient:
    """Scripted chat client: statements on extraction, verdicts on verification."""

    def __init__(self, statements, verdicts, *, model=None):
        self.model = model or config.JUDGE_MODEL
        self._statements = statements
        self._verdicts = verdicts
        self.calls = []

    def chat(self, messages, *, model=None, max_tokens=512, temperature=0.2,
             extra_body=None, seed=None):
        system = messages[0]["content"]
        self.calls.append(system)
        if "decompose" in system:
            return json.dumps(self._statements)
        if "fact-checker" in system:
            return json.dumps([{"verdict": v} for v in self._verdicts])
        return "[]"


def _judge(statements, verdicts):
    return RagasFaithfulnessJudge(
        client=FakeJudgeClient(statements, verdicts), model=config.JUDGE_MODEL,
    )


def test_faithfulness_is_supported_over_total():
    judge = _judge(["lesion is melanoma", "excision is advised"], [1, 0])
    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}]
    out = judge.evaluate(rows, ("faithfulness",))
    assert out["faithfulness"]["mean"] == 0.5
    assert out["faithfulness"]["n_scored"] == 1
    assert out["faithfulness"]["per_sample"] == [0.5]


def test_empty_context_is_none_not_zero():
    judge = _judge(["a claim"], [1])
    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": []}]
    out = judge.evaluate(rows, ("faithfulness",))
    assert out["faithfulness"]["per_sample"] == [None]
    assert out["faithfulness"]["mean"] is None
    assert out["faithfulness"]["n_scored"] == 0


def test_claimless_answer_is_none():
    judge = _judge([], [])  # no statements extracted
    rows = [{"user_input": "q", "response": "hmm", "retrieved_contexts": ["ctx"]}]
    out = judge.evaluate(rows, ("faithfulness",))
    assert out["faithfulness"]["per_sample"] == [None]


def test_uncomputed_metrics_stay_not_computed_not_faked():
    # answer_relevancy / context_recall ARE computed now; a metric we still do not
    # implement (context_precision) must stay an honest "not_computed", never a fake 0.
    judge = _judge(["c"], [1])
    out = judge.evaluate(
        [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}],
        ("faithfulness", "answer_relevancy", "context_recall", "context_precision"),
    )
    assert "mean" in out["answer_relevancy"]   # computed block, not a status stub
    assert "mean" in out["context_recall"]
    assert out["context_precision"] == {"status": "not_computed"}


def test_adapter_executes_with_locked_judge():
    lock = RagasJudgeLock()  # judge_model = config.JUDGE_MODEL
    judge = _judge(["a claim"], [1])
    sample = RagasSample(
        case_id="lesion:1", arm_id="M", user_input="q", response="a",
        retrieved_contexts=("ctx",),
    )
    result = RagasSupportingAdapter(lock, judge=judge).evaluate([sample])
    assert result["status"] == "executed"
    assert result["scores"]["faithfulness"]["mean"] == 1.0


def test_adapter_rejects_unlocked_judge_model():
    lock = RagasJudgeLock()
    judge = RagasFaithfulnessJudge(
        client=FakeJudgeClient(["c"], [1], model="other/model"), model="other/model",
    )
    sample = RagasSample(
        case_id="lesion:1", arm_id="M", user_input="q", response="a",
        retrieved_contexts=("ctx",),
    )
    with pytest.raises(Exception):
        RagasSupportingAdapter(lock, judge=judge).evaluate([sample])


def test_assessment_extraction_scores_only_the_literature_section():
    answer = (
        "Visual findings: Asymmetric pigment network with a blue-white veil.\n"
        "Assessment: Excise suspected melanoma completely. [1] Refer to specialist MDT. [2]\n"
        "Diagnosis: melanoma"
    )
    extracted = assessment_text_for_faithfulness(answer)
    assert extracted == (
        "Excise suspected melanoma completely. [1] Refer to specialist MDT. [2]"
    )


def test_assessment_extraction_falls_back_to_citation_sentences():
    answer = "The lesion is melanoma. Excision is recommended [1]. It looks asymmetric."
    assert assessment_text_for_faithfulness(answer) == "Excision is recommended [1]."


def test_assessment_extraction_scores_whole_no_citation_baseline_answer():
    # A no-citation baseline has no Assessment section and no cited sentence, so its
    # WHOLE answer is scored -> it lands low. That fairness is why the rule is not gaming.
    answer = "The lesion appears benign and no further action is needed."
    assert assessment_text_for_faithfulness(answer) == answer


def test_faithfulness_grades_only_the_assessment_section():
    # The judge must decompose only the literature Assessment, not the image line.
    judge = _judge(["excise melanoma completely", "refer to specialist"], [1, 1])
    answer = (
        "Visual findings: Blue-white veil.\n"
        "Assessment: Excise suspected melanoma completely. [1] Refer to specialist. [2]\n"
        "Diagnosis: melanoma"
    )
    rows = [{"user_input": "q", "response": answer, "retrieved_contexts": ["ctx"]}]
    out = judge.evaluate(rows, ("faithfulness",))
    assert out["faithfulness"]["mean"] == 1.0
    assert out["faithfulness"]["coverage"] == 1.0
    assert out["faithfulness"]["overall_mean"] == 1.0


def test_verify_salvages_truncated_verdict_array_instead_of_dropping():
    # The judge returns fewer verdicts than statements (a truncated tail or salvaged
    # partial parse). The old code discarded the whole sample to None -- exactly what
    # sank the long guideline-assessment cases. Now it scores the aligned prefix.
    judge = _judge(["s1", "s2", "s3"], [1, 1])  # 3 statements, only 2 verdicts
    rows = [{"user_input": "q", "response": "Assessment: a [1] b [2] c [3]",
             "retrieved_contexts": ["ctx"]}]
    out = judge.evaluate(rows, ("faithfulness",))
    assert out["faithfulness"]["per_sample"][0] == 1.0  # 2/2 salvaged verdicts
    assert out["faithfulness"]["assessment_mean"] == 1.0
    assert out["faithfulness"]["per_sample_scope"] == ["assessment"]


class VaryingJudgeClient:
    """Response-sensitive fake: faithfulness = K/4, where K is read from the answer.

    Stateless (safe to share across threads). Extraction embeds K into the first
    statement; verification recovers K from the statements JSON and marks K of 4
    supported -- so different rows get different, order-sensitive scores, which is what
    a parallel==sequential equality test must actually distinguish.
    """

    def __init__(self):
        self.model = config.JUDGE_MODEL

    def chat(self, messages, *, model=None, max_tokens=512, temperature=0.2,
             extra_body=None, seed=None):
        import re
        system = messages[0]["content"]
        user = messages[1]["content"]
        match = re.search(r"K=(\d)", user)
        k = int(match.group(1)) if match else 0
        if "decompose" in system:
            return json.dumps([f"K={k}", "s1", "s2", "s3"])  # 4 statements
        if "fact-checker" in system:
            return json.dumps([{"verdict": 1 if i < k else 0} for i in range(4)])
        return "[]"


class FakeARCRClient:
    """Scripted judge for relevancy (single number) + statement/verdict passes."""

    def __init__(self, statements, verdicts, relevancy, *, model=None):
        self.model = model or config.JUDGE_MODEL
        self._s, self._v, self._r = statements, verdicts, relevancy

    def chat(self, messages, *, model=None, max_tokens=512, temperature=0.2,
             extra_body=None, seed=None):
        system = messages[0]["content"].lower()
        if "relevance" in system:
            return json.dumps({"relevancy": self._r})
        if "decompose" in system:
            return json.dumps(self._s)
        if "fact-checker" in system:
            return json.dumps([{"verdict": v} for v in self._v])
        return "[]"


def test_judge_computes_answer_relevancy_and_context_recall():
    judge = RagasFaithfulnessJudge(
        client=FakeARCRClient(["s1", "s2"], [1, 1], 0.8), model=config.JUDGE_MODEL)
    rows = [{
        "user_input": "q", "response": "Assessment: excise it [1]",
        "retrieved_contexts": ["ctx"], "reference": "Diagnosis: melanoma. Excise it.",
    }]
    out = judge.evaluate(rows, ("faithfulness", "answer_relevancy", "context_recall"))
    assert out["answer_relevancy"]["mean"] == 0.8          # from the relevancy pass
    assert out["context_recall"]["mean"] == 1.0            # both reference claims supported
    assert out["answer_relevancy"]["n_scored"] == 1
    # A metric that is NOT requested stays not_computed (no wasted calls).
    out2 = judge.evaluate(rows, ("faithfulness",))
    assert out2.get("answer_relevancy", {}).get("status") is None or "answer_relevancy" not in out2


def test_context_recall_is_none_without_reference_or_context():
    judge = RagasFaithfulnessJudge(
        client=FakeARCRClient(["s1"], [1], 0.5), model=config.JUDGE_MODEL)
    rows = [{"user_input": "q", "response": "a [1]", "retrieved_contexts": ["ctx"],
             "reference": ""}]  # no reference -> CR undefined
    out = judge.evaluate(rows, ("faithfulness", "context_recall"))
    assert out["context_recall"]["per_sample"] == [None]
    assert out["context_recall"]["mean"] is None


def test_judge_parallel_matches_sequential_and_preserves_order():
    rows = [
        {"user_input": "q", "response": f"benign note K={k}", "retrieved_contexts": ["ctx"]}
        for k in (1, 3, 0, 4, 2, 3, 1)
    ]
    sequential = RagasFaithfulnessJudge(
        client=VaryingJudgeClient(), model=config.JUDGE_MODEL, max_workers=1,
    ).evaluate(rows, ("faithfulness",))
    parallel = RagasFaithfulnessJudge(
        client=VaryingJudgeClient(), model=config.JUDGE_MODEL, max_workers=4,
    ).evaluate(rows, ("faithfulness",))

    assert sequential == parallel  # result-invariant
    # Order preserved and scores are the expected K/4 per row.
    assert parallel["faithfulness"]["per_sample"] == [0.25, 0.75, 0.0, 1.0, 0.5, 0.75, 0.25]


def test_no_judge_still_maps_not_executed():
    # Backward-compatible: the default (no judge) path is unchanged.
    sample = RagasSample(
        case_id="lesion:1", arm_id="M", user_input="q", response="a",
        retrieved_contexts=("ctx",),
    )
    result = RagasSupportingAdapter(RagasJudgeLock()).evaluate([sample])
    assert result["status"] == "mapped_not_executed"
    assert result["scores"] is None
