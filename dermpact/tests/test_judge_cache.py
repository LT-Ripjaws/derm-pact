"""The judge cache: judging must survive a dropped connection, and say when it did."""
from __future__ import annotations

import json

import pytest

from dermpact.eval.ragas_judge import JudgeCache, RagasFaithfulnessJudge


class _Client:
    model = "openai/gpt-oss-20b"

    def __init__(self, response='["a statement"]'):
        self.calls = 0
        self.response = response

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.response


def _judge(client, cache):
    return RagasFaithfulnessJudge(
        client, model="openai/gpt-oss-20b", max_tokens=2048, cache=cache)


def test_an_identical_question_is_not_asked_twice(tmp_path):
    cache = JudgeCache(tmp_path, "s1")
    client = _Client()
    judge = _judge(client, cache)
    judge._chat("sys", "prompt")
    judge._chat("sys", "prompt")
    assert client.calls == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_a_different_question_is_a_different_entry(tmp_path):
    cache = JudgeCache(tmp_path, "s1")
    client = _Client()
    judge = _judge(client, cache)
    judge._chat("sys", "a")
    judge._chat("sys", "b")
    judge._chat("other-system", "a")
    assert client.calls == 3


def test_judging_resumes_across_sessions_without_re_asking(tmp_path):
    """The point: a drop mid-phase must cost minutes, not the whole phase."""
    first = _Client()
    _judge(first, JudgeCache(tmp_path, "s1"))._chat("sys", "prompt")
    second = _Client()
    resumed = JudgeCache(tmp_path, "s2")
    _judge(second, resumed)._chat("sys", "prompt")
    assert second.calls == 0, "a resumed session must reuse the earlier verdict"
    assert resumed.hits == 1


def test_the_session_mix_is_recorded_because_drift_is_real(tmp_path):
    """Cached verdicts keep their original session, so a resume MIXES sessions."""
    _judge(_Client(), JudgeCache(tmp_path, "s1"))._chat("sys", "a")
    cache2 = JudgeCache(tmp_path, "s2")
    judge2 = _judge(_Client(), cache2)
    judge2._chat("sys", "a")   # hit, stays s1
    judge2._chat("sys", "b")   # miss, written as s2
    assert cache2.session_composition() == {"s1": 1, "s2": 1}


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    cache = JudgeCache(tmp_path, "s1")
    client = _Client()
    judge = _judge(client, cache)
    judge._chat("sys", "prompt")
    for path in tmp_path.rglob("*.json"):
        path.write_text("{not json", encoding="utf-8")
    judge._chat("sys", "prompt")
    assert client.calls == 2, "a corrupt entry must be re-asked, never raised on"


def test_entries_are_written_immediately_not_at_the_end(tmp_path):
    """An unflushed buffer would defeat the entire purpose."""
    cache = JudgeCache(tmp_path, "s1")
    _judge(_Client(), cache)._chat("sys", "prompt")
    written = list(tmp_path.rglob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8"))["session_id"] == "s1"


def test_a_judge_without_a_cache_still_works(tmp_path):
    """The cache is optional; development paths must be unaffected."""
    client = _Client()
    judge = RagasFaithfulnessJudge(client, model="openai/gpt-oss-20b", max_tokens=2048)
    judge._chat("sys", "prompt")
    judge._chat("sys", "prompt")
    assert client.calls == 2


def test_the_cache_key_covers_decoding_parameters(tmp_path):
    """Same prompt at different settings is a different question."""
    a = JudgeCache._key("m", "sys", "p", 2048, 0.0, 0)
    assert a != JudgeCache._key("m", "sys", "p", 512, 0.0, 0)
    assert a != JudgeCache._key("m", "sys", "p", 2048, 0.7, 0)
    assert a != JudgeCache._key("m", "sys", "p", 2048, 0.0, 7)
    assert a != JudgeCache._key("other", "sys", "p", 2048, 0.0, 0)
