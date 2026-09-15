"""Behavior tests for the Phase 2.4 localized criterion residual."""
from __future__ import annotations

import unittest

import torch

from dermpact.perception.criterion_residual import LocalizedCriterionResidual


class LocalizedCriterionResidualTest(unittest.TestCase):
    def test_zero_initialized_residual_preserves_all_base_criterion_logits(self):
        torch.manual_seed(20260827)
        prototypes = torch.randn(7, 2, 16)
        module = LocalizedCriterionResidual(
            patch_dim=24,
            prototype_embeddings=prototypes,
            attention_dim=8,
        )
        patch_tokens = torch.randn(3, 196, 24)
        base_logits = torch.randn(3, 7)

        result = module(patch_tokens, base_logits)

        self.assertTrue(torch.equal(result.final_logits, base_logits))
        self.assertEqual(torch.count_nonzero(result.residual_logits).item(), 0)


if __name__ == "__main__":
    unittest.main()
