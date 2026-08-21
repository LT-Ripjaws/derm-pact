import unittest
from pathlib import Path


class EmbeddingAuditProtocolTest(unittest.TestCase):
    def test_thresholds_are_review_signals_not_automatic_exclusions(self):
        from medguard.data import audit_derm_embedding_duplicates as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn("exclusions require pixel/geometry evidence", source)
        self.assertNotIn('"excluded_case_ids"', source)
