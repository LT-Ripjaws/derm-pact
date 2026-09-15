from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from dermpact.eval import dermpact_phase2_consistency_repair as repair
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


class DermPactPhase2ConsistencyRepairTest(unittest.TestCase):
    def test_only_derm_batches_receive_the_auxiliary_consistency_loss(self) -> None:
        model = screen.HybridSharedHead()
        rng = np.random.default_rng(2)
        data = {
            "ham_features": rng.normal(size=(4, 512)).astype("float32"),
            "ham_global": np.asarray([0, 1, 0, 1], dtype="int64"),
            "ham_melanoma": np.asarray([1, 0, 1, 0], dtype="float32"),
        }
        weights = {
            "ham_diagnosis": np.ones(4, dtype="float32"),
            "ham_melanoma": np.ones(4, dtype="float32"),
        }
        loss, components = repair._batch_loss(
            model,
            dataset="ham",
            indices=np.arange(4),
            data=data,
            weights=weights,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(components["consistency"].detach()), 0.0)

    def test_repair_model_is_the_proven_independent_head(self) -> None:
        model = screen.HybridSharedHead()
        self.assertEqual(len(model.criteria), len(screen.CRITERIA))
        self.assertFalse(any("relation" in name for name, _ in model.named_parameters()))
        self.assertFalse(any("query" in name for name, _ in model.named_parameters()))

    def test_protocol_declares_one_changed_factor_and_closed_sets(self) -> None:
        with mock.patch.object(repair.screen, "_write_json", return_value="hash"):
            protocol = repair.build_protocol()
        self.assertEqual(
            protocol["single_changed_factor"]["baseline"],
            "independent scalar criterion heads",
        )
        self.assertEqual(protocol["hard_scope"]["ph2_rows_loaded"], 0)
        self.assertEqual(
            protocol["hard_scope"]["bcn_validation_or_final_feature_rows_loaded"], 0
        )
        self.assertEqual(
            protocol["frozen_negative_result"]["failed_arms_remain_rejected"],
            ["criterion_tokens", "relation"],
        )


if __name__ == "__main__":
    unittest.main()
