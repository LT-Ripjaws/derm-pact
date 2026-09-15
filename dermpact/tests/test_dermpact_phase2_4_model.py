"""Comparator-preservation tests for the Phase 2.4 model wrapper."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from dermpact.eval.dermpact_phase2_backbone_adaptation import AdaptedHybridModel
from dermpact.eval.dermpact_phase2_4_model import Phase24LocalizedCriterionModel


class _TailBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, values, *, rel_pos_bias=None):
        del rel_pos_bias
        return self.linear(values)


class _SyntheticVisual(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_TailBlock(width) for _ in range(12))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 512)
        self.rel_pos_bias = None


class Phase24LocalizedCriterionModelTest(unittest.TestCase):
    def test_zero_residual_wrapper_preserves_every_legacy_logit(self):
        torch.manual_seed(20260827)
        base = AdaptedHybridModel(_SyntheticVisual(), "last_block").eval()
        wrapper = Phase24LocalizedCriterionModel(
            base,
            prototype_embeddings=torch.randn(7, 2, 512),
            attention_dim=8,
        ).eval()
        prefix_tokens = torch.randn(2, 197, 8)

        with torch.inference_mode():
            expected = base(prefix_tokens)
            actual = wrapper(prefix_tokens)

        for name in ("diagnosis", "melanoma", "criteria"):
            self.assertTrue(torch.equal(actual[name], expected[name]), name)


if __name__ == "__main__":
    unittest.main()
