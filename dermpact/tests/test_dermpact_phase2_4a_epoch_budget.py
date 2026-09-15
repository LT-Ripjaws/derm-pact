"""Epoch-budget decision seam: c10@20 versus c10@30 promotion rule."""
from __future__ import annotations

import unittest

from dermpact.eval import dermpact_phase2_4a_epoch_budget as budget


class Phase24AEpochBudgetContractTest(unittest.TestCase):
    def test_generated_spec_pins_both_arms_and_the_preference_rule(self):
        spec = budget.generate_protocol_spec()

        self.assertEqual(spec["seed"], 20260827)
        self.assertEqual(spec["arms"], ["c10@20", "c10@30"])
        self.assertEqual(spec["orientation"], "c10_epoch30_minus_c10_epoch20")
        self.assertEqual(spec["bootstrap"]["replicates"], 2000)
        self.assertEqual(spec["bootstrap"]["seed"], 20270241)
        self.assertFalse(spec["scope"]["official_or_external_test_loading"])
        self.assertIn("keep c10@20", spec["decision_rule"])
        self.assertIn(
            "localized_criterion_epoch_budget_v1", budget.PROTOCOL_PATH.as_posix()
        )

    def test_winner_rule_prefers_the_20_epoch_arm_unless_30_clearly_wins(self):
        def paired(estimate, ci_low, ci_high):
            return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}

        # CI excludes zero on the positive side: the 30-epoch arm wins.
        self.assertEqual(
            budget.epoch_budget_winner(paired(0.004, 0.001, 0.007)), "c10@30"
        )
        # CI excludes zero on the negative side: the 20-epoch arm wins.
        self.assertEqual(
            budget.epoch_budget_winner(paired(-0.004, -0.007, -0.001)), "c10@20"
        )
        # CI includes zero: keep the cheaper 20-epoch arm.
        self.assertEqual(
            budget.epoch_budget_winner(paired(0.002, -0.001, 0.005)), "c10@20"
        )
        self.assertEqual(
            budget.epoch_budget_winner(paired(-0.002, -0.005, 0.001)), "c10@20"
        )
        # Boundary: a CI touching zero does not exclude it.
        self.assertEqual(
            budget.epoch_budget_winner(paired(0.003, 0.0, 0.006)), "c10@20"
        )


if __name__ == "__main__":
    unittest.main()
