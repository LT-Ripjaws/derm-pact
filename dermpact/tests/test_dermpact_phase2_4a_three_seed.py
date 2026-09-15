"""Public contract tests for the Phase 2.4A three-seed confirmation."""
from __future__ import annotations

import unittest

from dermpact.eval import dermpact_phase2_4a_three_seed as confirm


class Phase24AThreeSeedContractTest(unittest.TestCase):
    def test_generated_spec_is_the_fixed_three_seed_design(self):
        spec = confirm.generate_protocol_spec()

        self.assertEqual(confirm.SEEDS, (20260827, 20260828, 20260829))
        self.assertEqual(spec["seeds"], [20260827, 20260828, 20260829])
        self.assertEqual(spec["folds"], [0, 1, 2, 3, 4])
        self.assertEqual(spec["configuration"]["candidate_id"], "c10")
        self.assertEqual(spec["configuration"]["attention_dim"], 64)
        self.assertEqual(spec["configuration"]["consistency_weight"], 0.10)
        self.assertEqual(spec["training"]["epochs"], 20)
        self.assertEqual(spec["training"]["initialization"], "from_scratch")
        self.assertEqual(spec["training"]["retained_core"], "fully frozen")
        self.assertEqual(
            spec["comparison"]["direction"],
            "localized residual minus retained base from the same wrapper pass",
        )
        self.assertEqual(spec["bootstrap"]["seed_resampling"], True)
        self.assertEqual(spec["decision_rule"]["minimum_positive_seeds"], 2)
        self.assertEqual(spec["decision_rule"]["maximum_protected_fall"], 0.02)
        self.assertEqual(spec["decision_rule"]["maximum_disagreement_increase"], 0.02)
        self.assertFalse(spec["scope"]["official_or_external_test_loading"])
        self.assertFalse(spec["gates"]["phase2_4b_launched"])
        self.assertIn("three_seed_confirmation_v1", confirm.PROTOCOL_PATH.as_posix())

    def test_seed_20260827_is_reused_not_retrained(self):
        spec = confirm.generate_protocol_spec()
        reuse = spec["seed_execution"]
        self.assertEqual(reuse["20260827"], "REUSED_FROM_T2_EPOCH20_C10")
        self.assertEqual(reuse["20260828"], "TRAINED_IN_THIS_STAGE")
        self.assertEqual(reuse["20260829"], "TRAINED_IN_THIS_STAGE")

    def test_decision_rule_needs_two_positive_seeds_and_a_positive_interval(self):
        def call(deltas, ci_low, protected=None, disagreement=0.0):
            return confirm.confirmation_decision(
                per_seed_primary_deltas=deltas,
                pooled_interval={"estimate": 0.003, "ci_low": ci_low, "ci_high": 0.01},
                protected_deltas=protected if protected is not None else {
                    "balanced_accuracy": 0.0, "melanoma_auc": 0.0,
                    "primary": 0.002, "joint_correct_agreement": 0.0,
                },
                disagreement_delta=disagreement,
            )

        # Three positive seeds and a positive interval confirm.
        self.assertEqual(call([0.004, 0.003, 0.005], 0.001)["decision"], "CONFIRMED")
        # Exactly two positive seeds still confirm.
        self.assertEqual(call([0.004, -0.001, 0.005], 0.001)["decision"], "CONFIRMED")
        # One positive seed fails the stability requirement.
        result = call([0.004, -0.001, -0.002], 0.001)
        self.assertEqual(result["decision"], "NOT_CONFIRMED")
        self.assertIn("positive_seeds", result["failed_criteria"])
        # An interval touching zero is not wholly above zero.
        result = call([0.004, 0.003, 0.005], 0.0)
        self.assertEqual(result["decision"], "NOT_CONFIRMED")
        self.assertIn("pooled_interval_above_zero", result["failed_criteria"])
        # A protected endpoint falling more than 0.02 fails.
        result = call([0.004, 0.003, 0.005], 0.001, protected={
            "balanced_accuracy": -0.021, "melanoma_auc": 0.0,
            "primary": 0.002, "joint_correct_agreement": 0.0,
        })
        self.assertEqual(result["decision"], "NOT_CONFIRMED")
        self.assertIn("protected_balanced_accuracy", result["failed_criteria"])
        # Disagreement increasing beyond the margin fails.
        result = call([0.004, 0.003, 0.005], 0.001, disagreement=0.0201)
        self.assertEqual(result["decision"], "NOT_CONFIRMED")
        self.assertIn("disagreement_increase", result["failed_criteria"])

    def test_per_seed_retained_checkpoints_are_never_shared_across_seeds(self):
        folds = {
            seed: confirm.retained_fold_artifacts(seed) for seed in confirm.SEEDS
        }
        for seed, rows in folds.items():
            self.assertEqual([r["fold"] for r in rows], [0, 1, 2, 3, 4])
            for row in rows:
                self.assertIn(f"seed_{seed}", row["checkpoint"]["path"])
        # No checkpoint hash may appear under two different seeds.
        seen = {}
        for seed, rows in folds.items():
            for row in rows:
                digest = row["checkpoint"]["sha256"]
                self.assertNotIn(digest, seen, f"checkpoint reused across seeds: {digest}")
                seen[digest] = seed
        self.assertEqual(len(seen), 15)


if __name__ == "__main__":
    unittest.main()
