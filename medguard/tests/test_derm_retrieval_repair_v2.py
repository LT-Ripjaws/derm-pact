import unittest

from medguard.eval.derm_retrieval_repair_v2 import _rank_method


class _Dense:
    def search(self, query, k):
        rows = [{"id": str(index), "source": str(index), "text": query, "score": 1.0}
                for index in range(k)]
        return rows, [1.0] * len(rows)


class _Reranker:
    def rerank(self, query, rows, k):
        return list(rows)[:k]


class _Environment:
    dense = _Dense()
    reranker = _Reranker()


class RetrievalRepairTest(unittest.TestCase):
    def test_admission_first_uses_trusted_environment_and_preserves_budget(self):
        query = {"query_top1": "q1", "query_top2": "q2", "probability_margin": 1.0}
        rows, audit = _rank_method(
            query,
            "admission_first_dense_top1",
            requested_environment=object(),
            trusted_environment=_Environment(),
            quarantined_corpus_documents=4,
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual(audit["quarantined"], 4)


if __name__ == "__main__":
    unittest.main()
