from __future__ import annotations

import unittest
from unittest import mock

from dermpact.eval import dermpact_phase2_clinical_features as clinical


class DermPactPhase2ClinicalFeaturesTest(unittest.TestCase):
    def test_development_pair_count_and_duplicate_mask_are_locked(self) -> None:
        rows, availability = clinical.paired_rows()
        self.assertEqual(len(rows), 616)
        self.assertEqual(int(availability.sum()), 609)
        self.assertEqual(int((~availability).sum()), 7)
        self.assertTrue(all(row["phase1_role"] == "development" for row in rows))

    def test_protocol_keeps_protected_rows_closed(self) -> None:
        with mock.patch.object(clinical.screen, "_write_json", return_value="hash"):
            protocol = clinical.build_protocol()
        self.assertEqual(protocol["hard_scope"]["derm7pt_official_test_rows_loaded"], 0)
        self.assertEqual(protocol["hard_scope"]["bcn_validation_or_final_rows_loaded"], 0)
        self.assertEqual(protocol["row_contract"]["n_distinct_clinical_photos"], 609)
        self.assertEqual(protocol["row_contract"]["n_non_distinct_views_masked"], 7)


if __name__ == "__main__":
    unittest.main()
