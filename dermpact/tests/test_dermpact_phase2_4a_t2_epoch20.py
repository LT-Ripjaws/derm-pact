"""Public contract tests for Phase 2.4A-T2 epoch-20 refits."""
from __future__ import annotations

import unittest

from dermpact.eval import dermpact_phase2_4a_t2_epoch20 as epoch20


class Phase24AT2Epoch20ContractTest(unittest.TestCase):
    def test_generated_spec_is_an_isolated_from_scratch_epoch20_design(self):
        spec = epoch20.generate_protocol_spec()

        self.assertEqual(epoch20.candidate_ids(), ("c05", "c10", "c01"))
        self.assertEqual(spec["seed"], 20260827)
        self.assertEqual(spec["folds"], [0, 1, 2, 3, 4])
        self.assertEqual(spec["training"]["epochs"], 20)
        self.assertEqual(spec["training"]["initialization"], "from_scratch")
        self.assertEqual(spec["training"]["scheduler"]["horizon_epochs"], 20)
        self.assertEqual(spec["training"]["effective_batch_size"], 64)
        self.assertEqual(spec["training"]["directml_microbatch_size"], 8)
        self.assertFalse(spec["scope"]["official_or_external_test_loading"])
        self.assertEqual(spec["ranking"]["reference_candidate_id"], "c00@20")
        self.assertEqual(spec["ranking"]["epoch30_selection_count"], 2)
        self.assertFalse(spec["ranking"]["epoch30_launched"])
        self.assertIn("localized_criterion_tuning_t2_epoch20_v1", epoch20.PROTOCOL_PATH.as_posix())

    def test_epoch30_selection_is_exactly_top_two_safe_epoch20_rows(self):
        rows = [
            {"candidate_id": "c01", "safety_decision": "UNSAFE", "criterion_mean": 0.99},
            {"candidate_id": "c10", "safety_decision": "SAFE", "criterion_mean": 0.81},
            {"candidate_id": "c00@20", "safety_decision": "SAFE", "criterion_mean": 0.80},
            {"candidate_id": "c05", "safety_decision": "SAFE", "criterion_mean": 0.82},
        ]

        self.assertEqual(epoch20.select_epoch30_candidates(rows), ["c05", "c10"])
        with self.assertRaises(RuntimeError):
            epoch20.select_epoch30_candidates(rows[:1])


if __name__ == "__main__":
    unittest.main()
