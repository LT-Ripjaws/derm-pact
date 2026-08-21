import unittest

from medguard.eval.derm_retrieval_tournament import aggregate, paired_bootstrap_difference


class DermRetrievalTournamentTest(unittest.TestCase):
    def _row(self, cluster, recall, poison):
        return {
            "cluster_id": cluster,
            "concept_recall_at_5": recall,
            "management_recall_at_5": recall,
            "concept_precision_at_5": recall,
            "mrr": recall,
            "poison_retrieval_at_5": poison,
            "poison_fraction_at_5": poison,
            "retrieval_calls": 1,
            "reranker_pairs": 0,
            "quarantined": 0,
            "elapsed_ms_shared_cache": 1,
            "returned_count": 5,
        }

    def test_aggregate_and_paired_difference(self):
        left = [self._row("a", 1, 0), self._row("b", 1, 0)]
        right = [self._row("a", 0, 1), self._row("b", 1, 1)]
        metrics = aggregate(left, seed=1)
        self.assertEqual(metrics["concept_recall_at_5"], 1.0)
        comparison = paired_bootstrap_difference(left, right, "poison_retrieval_at_5", seed=1)
        self.assertEqual(comparison["estimate"], -1.0)
        self.assertEqual(comparison["n_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
