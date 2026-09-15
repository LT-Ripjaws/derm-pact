"""Behavior tests for normalized-cosine semantic criterion attention."""
from __future__ import annotations

import math
import unittest

import torch

from dermpact.perception.normalized_cosine_criterion_residual import (
    NormalizedCosineCriterionResidual,
)


def _entropy(attention: torch.Tensor) -> torch.Tensor:
    return -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)


class NormalizedCosineCriterionResidualTest(unittest.TestCase):
    def test_zero_scale_is_exact_and_attention_is_finite(self):
        torch.manual_seed(20260830)
        module = NormalizedCosineCriterionResidual(
            patch_dim=8,
            prototype_embeddings=torch.randn(7, 2, 8),
            attention_dim=8,
        )
        patches = torch.randn(3, 11, 8)
        base = torch.randn(3, 7)

        result = module(patches, base)

        self.assertTrue(torch.equal(result.final_logits, base))
        self.assertEqual(result.attention_weights.shape, (3, 7, 2, 11))
        self.assertTrue(torch.isfinite(result.attention_weights).all())
        self.assertTrue(
            torch.allclose(
                result.attention_weights.sum(dim=-1),
                torch.ones(3, 7, 2),
            )
        )

    def test_higher_temperature_reduces_attention_entropy(self):
        prototypes = torch.zeros(7, 2, 2)
        prototypes[:, 0, 0] = 1.0
        prototypes[:, 1, 1] = 1.0
        module = NormalizedCosineCriterionResidual(
            patch_dim=2,
            prototype_embeddings=prototypes,
            attention_dim=2,
            initial_temperature=1.0,
        )
        with torch.no_grad():
            module.prototype_projection.weight.copy_(torch.eye(2))
            module.patch_projection.weight.copy_(torch.eye(2))
        patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
        base = torch.zeros(1, 7)

        with torch.no_grad():
            module.log_temperature.fill_(math.log(1.0))
            broad = module(patches, base).attention_weights
            module.log_temperature.fill_(math.log(30.0))
            sharp = module(patches, base).attention_weights

        self.assertLess(float(_entropy(sharp).mean()), float(_entropy(broad).mean()))
        self.assertGreater(float(sharp.max(dim=-1).values.mean()), 0.99)

    def test_temperature_and_projection_receive_gradients_when_enabled(self):
        torch.manual_seed(20260830)
        module = NormalizedCosineCriterionResidual(
            patch_dim=5,
            prototype_embeddings=torch.randn(7, 2, 6),
            attention_dim=4,
        )
        with torch.no_grad():
            module.residual_scale.fill_(0.2)
        result = module(torch.randn(4, 13, 5), torch.randn(4, 7))
        result.final_logits.square().mean().backward()

        self.assertIsNone(module.prototype_embeddings.grad)
        for parameter in (
            module.log_temperature,
            module.prototype_projection.weight,
            module.patch_projection.weight,
            module.residual_scale,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
