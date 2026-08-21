import unittest

from medguard.eval.derm_encoder_compare import _align, _pareto_front


class DermEncoderComparisonTest(unittest.TestCase):
    def test_align_rejects_target_mismatch(self):
        reference = [{
            "case_id": "a", "cluster_id": "lesion-a", "label": "nevus",
            "malignant_true": False,
        }]
        mismatched = [{
            "case_id": "a", "cluster_id": "lesion-a", "label": "melanoma",
            "malignant_true": True,
        }]
        with self.assertRaisesRegex(ValueError, "target/cluster mismatch"):
            _align({"a": reference, "b": mismatched})

    def test_pareto_front_preserves_real_tradeoff_and_removes_dominated(self):
        metrics = {
            "safety": {
                "diagnosis_accuracy": 0.8, "diagnosis_macro_recall": 0.7,
                "malignant_auc": 0.95, "malignant_recall_at_90_specificity": 0.8,
            },
            "utility": {
                "diagnosis_accuracy": 0.82, "diagnosis_macro_recall": 0.75,
                "malignant_auc": 0.94, "malignant_recall_at_90_specificity": 0.78,
            },
            "dominated": {
                "diagnosis_accuracy": 0.7, "diagnosis_macro_recall": 0.6,
                "malignant_auc": 0.9, "malignant_recall_at_90_specificity": 0.7,
            },
        }
        self.assertEqual(_pareto_front(metrics), ["safety", "utility"])
