import tempfile
from pathlib import Path

from medguard.pipeline.generation_cache import CachedStructuredGenerator, GenerationCache
from medguard.tests.c1_fixtures import contract


def _contract():
    return contract()


class Delegate:
    def __init__(self):
        self.calls = 0
        self.last_metadata = {}

    def generate_contract(self, **_kwargs):
        self.calls += 1
        self.last_metadata = {"remote_calls": 1, "total_tokens": 10}
        return _contract()


def test_identical_contract_request_hits_cache():
    with tempfile.TemporaryDirectory() as tmp:
        delegate = Delegate()
        cached = CachedStructuredGenerator(
            delegate, GenerationCache(Path(tmp), namespace="model-v1"))
        first = cached.generate_contract(case_id="HAM_1", question="Diagnosis?")
        second = cached.generate_contract(case_id="HAM_1", question="Diagnosis?")
        assert first.to_dict() == second.to_dict()
        assert delegate.calls == 1
        assert cached.last_metadata["cache_hit"] is True


def test_cache_key_changes_with_request():
    with tempfile.TemporaryDirectory() as tmp:
        cache = GenerationCache(Path(tmp), namespace="model-v1")
        assert cache.request_key({"question": "a"}) != cache.request_key({"question": "b"})
