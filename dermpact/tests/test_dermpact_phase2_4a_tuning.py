"""Public contract tests for Phase 2.4A-T1 residual-only tuning."""
from __future__ import annotations

import unittest

from dermpact.eval import dermpact_phase2_4a_tuning as tuning


class Phase24ATuningContractTest(unittest.TestCase):
    def test_candidate_ledger_is_deterministic_balanced_and_contains_center(self):
        first = tuning.generate_candidate_ledger()
        second = tuning.generate_candidate_ledger()
        self.assertEqual(first, second)
        candidates = first["candidates"]
        self.assertEqual([row["candidate_id"] for row in candidates], [f"c{i:02d}" for i in range(12)])
        self.assertEqual(len({row["candidate_config_sha256"] for row in candidates}), 12)
        self.assertEqual(candidates[0]["coded_levels"], [1, 1, 1, 1, 1])
        self.assertTrue(candidates[0]["forced_center"])
        for factor in range(5):
            counts = [sum(row["coded_levels"][factor] == level for row in candidates) for level in range(3)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_center_candidate_has_stage1_defaults(self):
        center = tuning.generate_candidate_ledger()["candidates"][0]["config"]
        self.assertEqual(center, {
            "attention_dim": 128,
            "projection_learning_rate": 0.0003,
            "scale_learning_rate": 0.001,
            "projection_weight_decay": 0.0001,
            "consistency_weight": 0.10,
        })

    def test_paths_are_isolated_from_stage1(self):
        self.assertIn("localized_criterion_tuning_v1", tuning.PROTOCOL_PATH.as_posix())
        self.assertEqual(tuning.training_receipt_path("c03", 4).name, "seed_20260827_fold_4_training.json")
        self.assertEqual(tuning.canonical_receipt_path("c03", 4).name, "seed_20260827_fold_4_canonical.json")
        self.assertEqual(tuning.candidate_comparison_path("c03").name, "seed_20260827_comparison.json")

    def test_ranking_advances_safe_top_three_and_center_fallback(self):
        rows = [
            {"candidate_id": "c00", "safety_decision": "SAFE", "criterion_mean": 0.70},
            {"candidate_id": "c01", "safety_decision": "UNSAFE", "criterion_mean": 0.99},
            {"candidate_id": "c02", "safety_decision": "SAFE", "criterion_mean": 0.75},
            {"candidate_id": "c03", "safety_decision": "SAFE", "criterion_mean": 0.73},
            {"candidate_id": "c04", "safety_decision": "SAFE", "criterion_mean": 0.74},
        ]
        self.assertEqual(tuning.select_advanced_candidates(rows), ["c02", "c04", "c03"])
        self.assertEqual(
            tuning.select_advanced_candidates([{"candidate_id": "c02", "safety_decision": "SAFE", "criterion_mean": 0.75}]),
            ["c02", "c00"],
        )

    def test_execution_matrix_has_121_jobs(self):
        matrix = tuning.controller_matrix()
        self.assertEqual(len(matrix), 133)
        self.assertEqual(sum(row["stage"] == "train" for row in matrix), 60)
        self.assertEqual(sum(row["stage"] == "reevaluate" for row in matrix), 60)
        self.assertEqual(matrix[-1]["stage"], "rank")

    def test_resume_identity_is_candidate_specific(self):
        payload = {
            "format": tuning.RESUME_SCHEMA,
            "protocol_sha256": "protocol",
            "seed": tuning.SEED,
            "fold": 2,
            "source_checkpoint_sha256": "source",
            "ledger_sha256": "ledger",
            "candidate_id": "c03",
            "candidate_config_sha256": "config",
        }
        tuning.validate_tuning_resume_identity(
            payload, protocol_sha256="protocol", fold=2,
            source_sha256="source", ledger_sha256="ledger",
            candidate_id="c03", candidate_config_sha256="config",
        )
        payload["candidate_id"] = "c04"
        with self.assertRaisesRegex(AssertionError, "candidate identity mismatch"):
            tuning.validate_tuning_resume_identity(
                payload, protocol_sha256="protocol", fold=2,
                source_sha256="source", ledger_sha256="ledger",
                candidate_id="c03", candidate_config_sha256="config",
            )


if __name__ == "__main__":
    unittest.main()
