"""Public contract tests for the Phase 2.4A LoRA learning-rate-factor stage."""
from __future__ import annotations

import unittest

from dermpact.eval import dermpact_phase2_4a_lora_factor as lora


class Phase24ALoraFactorContractTest(unittest.TestCase):
    def test_generated_spec_is_the_fixed_lora_factor_design(self):
        spec = lora.generate_protocol_spec()

        self.assertEqual(lora.candidate_ids(), ("lora_1e6", "lora_3e6", "lora_1e5"))
        self.assertEqual(
            [spec["arms"][arm]["lora_learning_rate"] for arm in lora.ARM_IDS],
            [1e-6, 3e-6, 1e-5],
        )
        self.assertEqual(spec["seed"], 20260827)
        self.assertEqual(spec["folds"], [0, 1, 2, 3, 4])
        self.assertEqual(spec["training"]["epochs"], 20)
        self.assertEqual(spec["training"]["initialization"], "residual_from_scratch_lora_from_retained")
        self.assertEqual(spec["training"]["effective_batch_size"], 64)
        self.assertEqual(spec["training"]["microbatch_size"], 8)
        self.assertIn("canonical venv_p2 CPU", spec["training"]["training_backend"])
        self.assertEqual(spec["training"]["optimizer"]["lora_group"]["weight_decay"], 1e-4)
        self.assertEqual(spec["training"]["optimizer"]["projection_group"]["learning_rate"], 1e-3)
        self.assertEqual(spec["training"]["optimizer"]["projection_group"]["weight_decay"], 1e-3)
        self.assertEqual(spec["training"]["optimizer"]["scale_group"]["learning_rate"], 3e-3)
        self.assertEqual(spec["training"]["consistency_weight"], 0.10)
        self.assertEqual(spec["ranking"]["reference_candidate_id"], "c10@20")
        self.assertFalse(spec["ranking"]["three_seed_launched"])
        self.assertFalse(spec["scope"]["official_or_external_test_loading"])
        protected = spec["endpoints"]["protected"]
        self.assertIn("ham_balanced_accuracy", protected)
        self.assertIn("ham_melanoma_auc", protected)
        self.assertIn("derm_balanced_accuracy", protected)
        self.assertEqual(spec["endpoints"]["maximum_protected_fall"], 0.02)
        self.assertEqual(spec["endpoints"]["maximum_disagreement_increase"], 0.02)
        self.assertEqual(spec["bootstrap"]["derm"]["seed"], 20270241)
        self.assertEqual(spec["bootstrap"]["ham"]["seed"], 20271242)
        self.assertIn("lora_rate_factor_v1", lora.PROTOCOL_PATH.as_posix())

    def test_winner_rule_requires_safe_and_positive_ci_else_retains_c10_at_20(self):
        def row(candidate_id, safety, mean, beats):
            return {
                "candidate_id": candidate_id,
                "safety_decision": safety,
                "criterion_mean": mean,
                "beats_c10_epoch20_with_positive_95ci": beats,
            }

        # The best SAFE arm that beats the comparator wins.
        rows = [
            row("lora_1e6", "SAFE", 0.830, True),
            row("lora_3e6", "SAFE", 0.832, True),
            row("lora_1e5", "UNSAFE", 0.840, True),
            row("c10@20", "SAFE", 0.8287, False),
        ]
        self.assertEqual(lora.select_winner(rows), "lora_3e6")
        # A higher-mean SAFE arm that does not beat the comparator is skipped.
        rows[1] = row("lora_3e6", "SAFE", 0.832, False)
        self.assertEqual(lora.select_winner(rows), "lora_1e6")
        # No SAFE arm beats the comparator: retain c10@20.
        rows[0] = row("lora_1e6", "SAFE", 0.830, False)
        self.assertEqual(lora.select_winner(rows), "c10@20")
        # UNSAFE arms can never win even with positive intervals.
        rows = [
            row("lora_1e6", "UNSAFE", 0.85, True),
            row("lora_3e6", "UNSAFE", 0.85, True),
            row("lora_1e5", "UNSAFE", 0.85, True),
            row("c10@20", "SAFE", 0.8287, False),
        ]
        self.assertEqual(lora.select_winner(rows), "c10@20")

    def test_safety_gates_apply_protected_and_disagreement_margins(self):
        paired = {
            "derm": {
                "criterion_mean_auc": {"estimate": 0.002, "ci_low": 0.001, "ci_high": 0.003},
                "primary": {"estimate": -0.001},
                "balanced_accuracy": {"estimate": -0.005},
                "melanoma_auc": {"estimate": 0.001},
                "joint_correct_agreement": {"estimate": -0.002},
                "direct_rule_disagreement": {"estimate": 0.004},
            },
            "ham": {
                "balanced_accuracy": {"estimate": -0.010},
                "melanoma_auc": {"estimate": 0.000},
            },
        }
        checks = lora.safety_checks(paired)
        self.assertTrue(all(checks.values()))
        # A protected endpoint falling by more than 0.02 is UNSAFE.
        bad = {**paired, "ham": {**paired["ham"], "balanced_accuracy": {"estimate": -0.021}}}
        self.assertFalse(all(lora.safety_checks(bad).values()))
        # Disagreement increasing by more than 0.02 is UNSAFE.
        bad = {
            **paired,
            "derm": {**paired["derm"], "direct_rule_disagreement": {"estimate": 0.0201}},
        }
        self.assertFalse(all(lora.safety_checks(bad).values()))


if __name__ == "__main__":
    unittest.main()
