from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from dermpact.eval import dermpact_phase2_architecture_ladder as ladder
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


class DermPactPhase2ArchitectureLadderTest(unittest.TestCase):
    def test_every_architecture_obeys_the_locked_output_shapes(self) -> None:
        features = torch.randn(5, 512)
        for architecture in ladder.ARCHITECTURES:
            outputs = ladder.make_model(architecture)(features)
            self.assertEqual(outputs["diagnosis"].shape, (5, len(screen.GLOBAL_CLASSES)))
            self.assertEqual(outputs["melanoma"].shape, (5,))
            self.assertEqual(outputs["criteria"].shape, (5, len(screen.CRITERIA)))

    def test_relation_is_the_only_model_change_between_first_two_steps(self) -> None:
        tokens = ladder.make_model("criterion_tokens")
        relation = ladder.make_model("relation")
        self.assertFalse(tokens.relation_enabled)
        self.assertTrue(relation.relation_enabled)
        self.assertGreater(
            sum(parameter.numel() for parameter in relation.parameters()),
            sum(parameter.numel() for parameter in tokens.parameters()),
        )

    def test_consistency_masks_source_contradictions(self) -> None:
        model = ladder.make_model("relation_consistency")
        outputs = model(torch.randn(4, 512))
        criteria = torch.zeros(4, len(screen.CRITERIA))
        diagnosis = torch.tensor(
            [
                screen.GLOBAL_INDEX["melanoma"],
                screen.GLOBAL_INDEX["nevus"],
                screen.GLOBAL_INDEX["melanoma"],
                screen.GLOBAL_INDEX["nevus"],
            ]
        )
        melanoma = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss, eligible = ladder.truth_anchored_consistency_loss(
            outputs,
            melanoma=melanoma,
            criteria=criteria,
            diagnosis=diagnosis,
        )
        self.assertEqual(eligible, 2)
        self.assertTrue(torch.isfinite(loss))

    def test_consistency_is_zero_when_no_case_is_eligible(self) -> None:
        model = ladder.make_model("relation_consistency")
        outputs = model(torch.randn(2, 512))
        criteria = torch.zeros(2, len(screen.CRITERIA))
        diagnosis = torch.full(
            (2,), screen.GLOBAL_INDEX["basal_cell_carcinoma"], dtype=torch.long
        )
        melanoma = torch.zeros(2)
        loss, eligible = ladder.truth_anchored_consistency_loss(
            outputs,
            melanoma=melanoma,
            criteria=criteria,
            diagnosis=diagnosis,
        )
        self.assertEqual(eligible, 0)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_bcn_loss_never_uses_criterion_or_consistency_targets(self) -> None:
        data = {
            "bcn_features": np.random.default_rng(1).normal(size=(4, 512)).astype("float32"),
            "bcn_global": np.asarray([0, 1, 0, 1], dtype="int64"),
            "bcn_melanoma": np.asarray([1, 0, 1, 0], dtype="float32"),
        }
        weights = {
            "bcn_diagnosis": np.ones(4, dtype="float32"),
            "bcn_melanoma": np.ones(4, dtype="float32"),
        }
        loss, components = ladder._bcn_batch_loss(
            ladder.make_model("relation_consistency"),
            architecture="relation_consistency",
            indices=np.arange(4),
            data=data,
            weights=weights,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(components["criteria_mean"].detach()), 0.0)
        self.assertEqual(float(components["consistency"].detach()), 0.0)

    def test_fixed_checklist_weight_order_matches_the_registered_criteria(self) -> None:
        expected = {
            "pigment_network": 2,
            "streaks": 1,
            "pigmentation": 1,
            "regression_structures": 1,
            "dots_and_globules": 1,
            "blue_whitish_veil": 2,
            "vascular_structures": 2,
        }
        observed = {
            name: int(weight)
            for name, weight in zip(
                screen.CRITERIA, ladder.CHECKLIST_WEIGHTS.tolist()
            )
        }
        self.assertEqual(observed, expected)

    def test_rule_summary_uses_only_source_consistent_melanocytic_cases(self) -> None:
        rows = ladder._source_rows(
            "derm_ham", ladder.BASELINE_ARCHITECTURE, ladder.SEEDS[0], "derm7pt"
        )
        summary = ladder._derm_rule_summary(rows)
        self.assertEqual(summary["eligible_cases"], 430)
        self.assertGreaterEqual(summary["joint_correct_agreement"], 0.0)
        self.assertLessEqual(summary["joint_correct_agreement"], 1.0)

    def test_protocol_keeps_later_factors_and_protected_sets_out_of_2_3a(self) -> None:
        with mock.patch.object(ladder.screen, "_write_json", return_value="protocol-hash"):
            protocol = ladder.build_protocol()
        scope = protocol["hard_scope"]
        self.assertEqual(scope["derm7pt_official_test_rows_loaded"], 0)
        self.assertEqual(scope["ham_calibration_rows_loaded"], 0)
        self.assertEqual(scope["bcn_validation_or_final_feature_rows_loaded"], 0)
        self.assertEqual(scope["ph2_rows_loaded"], 0)
        self.assertEqual(scope["clinical_photo_features_loaded"], 0)
        self.assertEqual(scope["structured_facts_loaded_into_model"], 0)
        phases = [step["phase"] for step in protocol["later_prespecified_factor_order"]]
        self.assertEqual(phases, ["2.3B", "2.3C", "2.3D", "2.4"])


if __name__ == "__main__":
    unittest.main()
