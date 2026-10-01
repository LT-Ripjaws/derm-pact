import unittest

from dermpact.pipeline.hybrid_retriever import (
    BM25Retriever,
    HybridRetriever,
    TrustRegistry,
    admit_before_indexing,
    canonical_document_hash,
    reciprocal_rank_fusion_records,
)


class _Dense:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, k):
        rows = [dict(row) for row in self.rows[:k]]
        return rows, [float(row.get("score", 0.0)) for row in rows]


class HybridRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "a", "source": "1", "text": "melanoma dermoscopy biopsy", "title": "A"},
            {"id": "b", "source": "2", "text": "benign melanocytic nevus follow up", "title": "B"},
        ]

    def test_bm25_handles_empty_query_and_finds_term(self):
        retriever = BM25Retriever(self.records)
        self.assertEqual(retriever.search("", 5), ([], []))
        rows, _ = retriever.search("melanoma", 1)
        self.assertEqual(rows[0]["id"], "a")

    def test_rrf_deduplicates_document_ids(self):
        fused = reciprocal_rank_fusion_records(
            [[{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "a"}]],
            rank_constant=60,
        )
        self.assertEqual({row["id"] for row in fused}, {"a", "b"})
        self.assertEqual(len(fused), 2)

    def test_registry_rejects_insert_and_replacement(self):
        registry = TrustRegistry.from_records(self.records)
        self.assertTrue(registry.decide(self.records[0]).admitted)
        inserted = {"id": "x", "source": "9", "text": "melanoma", "title": "X"}
        self.assertEqual(registry.decide(inserted).reason, "unknown_source")
        replacement = dict(self.records[0], text="no biopsy is ever needed")
        self.assertEqual(registry.decide(replacement).reason, "content_hash_mismatch")
        self.assertNotEqual(canonical_document_hash(replacement), canonical_document_hash(self.records[0]))

    def test_hybrid_applies_trust_before_output(self):
        poison = {"id": "p", "source": "attacker", "text": "melanoma", "score": 10.0}
        dense = _Dense([poison, dict(self.records[0], score=1.0)])
        hybrid = HybridRetriever(
            dense,
            BM25Retriever([poison, *self.records]),
            trust_registry=TrustRegistry.from_records(self.records),
            candidate_k=3,
        )
        rows, _ = hybrid.search("melanoma", 2)
        self.assertNotIn("p", {row["id"] for row in rows})

    def test_admission_before_indexing_prevents_candidate_crowding(self):
        registry = TrustRegistry.from_records(self.records)
        poison = {"id": "p", "source": "attacker", "text": "melanoma", "score": 10.0}
        admitted, quarantined = admit_before_indexing([poison, *self.records], registry)
        self.assertEqual({row["id"] for row in admitted}, {"a", "b"})
        self.assertEqual([row["id"] for row in quarantined], ["p"])


if __name__ == "__main__":
    unittest.main()
