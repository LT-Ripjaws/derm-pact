import json
from pathlib import Path
import tempfile
import unittest

from medguard.data.freeze_derm_retrieval_benchmark import CLASS_ORDER, freeze_benchmark


class DermRetrievalFreezeTest(unittest.TestCase):
    def test_freeze_is_grouped_and_never_loads_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "out"
            patterns = {
                "akiec": "actinic keratosis biopsy",
                "bcc": "basal cell carcinoma treatment",
                "bkl": "seborrheic keratosis follow-up",
                "df": "dermatofibroma excision",
                "melanoma": "melanoma biopsy",
                "nevus": "melanocytic nevus follow up",
                "vasc": "vascular lesion treatment",
            }
            with corpus.open("w", encoding="utf-8") as handle:
                for index, label in enumerate(CLASS_ORDER):
                    row = {"id": str(index), "source": str(index), "title": label,
                           "abstract": patterns[label], "text": patterns[label]}
                    handle.write(json.dumps(row) + "\n")
            with predictions.open("w", encoding="utf-8") as handle:
                for index, label in enumerate(CLASS_ORDER):
                    probabilities = [0.01] * len(CLASS_ORDER)
                    probabilities[index] = 0.94
                    for duplicate in range(2):
                        row = {
                            "case_id": f"{label}-{duplicate}", "cluster_id": f"cluster-{label}",
                            "label": label, "prediction": label,
                            "diagnosis_probabilities": probabilities,
                        }
                        handle.write(json.dumps(row) + "\n")
            receipt = freeze_benchmark(corpus, predictions, output, cap_per_class=2)
            self.assertEqual(receipt["test_records_loaded"], 0)
            self.assertEqual(receipt["query_count"], len(CLASS_ORDER))
            queries = [json.loads(line) for line in (output / "queries.jsonl").read_text().splitlines()]
            self.assertEqual(len({row["cluster_id"] for row in queries}), len(queries))
            self.assertTrue(all(row["relevant_concept_document_ids"] for row in queries))


if __name__ == "__main__":
    unittest.main()
