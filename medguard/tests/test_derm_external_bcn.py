import unittest

from medguard.eval.derm_external_bcn import ACTIONABLE, _aggregate_lesions


class BcnExternalEvaluationTest(unittest.TestCase):
    def test_scc_is_actionable(self):
        self.assertIn("scc", ACTIONABLE)

    def test_lesion_aggregation_averages_views(self):
        rows = [
            {
                "case_id": "a", "cluster_id": "l1", "label": "scc",
                "diagnosis_probabilities": [1, 0, 0, 0, 0, 0, 0],
                "malignant_score": 0.2,
            },
            {
                "case_id": "b", "cluster_id": "l1", "label": "scc",
                "diagnosis_probabilities": [0, 1, 0, 0, 0, 0, 0],
                "malignant_score": 0.8,
            },
        ]
        lesion = _aggregate_lesions(rows)[0]
        self.assertEqual(lesion["n_views"], 2)
        self.assertAlmostEqual(lesion["malignant_score"], 0.5)
