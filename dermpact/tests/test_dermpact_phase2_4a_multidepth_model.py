"""Focused tests for the comparator-preserving multi-depth criterion model."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from dermpact.eval.dermpact_phase2_backbone_adaptation import AdaptedHybridModel
from dermpact.eval.dermpact_phase2_4_model import Phase24LocalizedCriterionModel
from dermpact.eval.dermpact_phase2_4a_multidepth_model import (
    Phase24MultiDepthCriterionModel,
)


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


def _confirmed_model() -> Phase24LocalizedCriterionModel:
    base = AdaptedHybridModel(_SyntheticVisual(), "last_block").eval()
    model = Phase24LocalizedCriterionModel(
        base,
        prototype_embeddings=torch.randn(7, 2, 16),
        attention_dim=8,
    ).eval()
    with torch.no_grad():
        model.criterion_residual.residual_scale.copy_(
            torch.linspace(-0.2, 0.2, 7)
        )
    return model


class Phase24MultiDepthCriterionModelTest(unittest.TestCase):
    def test_zero_multidepth_scale_is_exact_confirmed_model(self):
        torch.manual_seed(20260830)
        confirmed = _confirmed_model()
        model = Phase24MultiDepthCriterionModel(confirmed).eval()
        prefix = torch.randn(3, 197, 8)

        with torch.inference_mode():
            expected = confirmed(prefix)
            actual = model(prefix)

        for name in ("diagnosis", "melanoma", "criteria"):
            self.assertTrue(torch.equal(actual[name], expected[name]), name)
        self.assertEqual(
            torch.count_nonzero(actual["multidepth_criterion_residual"]).item(), 0
        )
        self.assertTrue(
            torch.equal(
                actual["final_depth_criterion_residual"],
                expected["criterion_residual"],
            )
        )

    def test_only_new_multidepth_residual_is_trainable_and_receives_gradients(self):
        torch.manual_seed(20260830)
        model = Phase24MultiDepthCriterionModel(_confirmed_model()).train()
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("multidepth_residual.") for name in trainable)
        )

        output = model(torch.randn(2, 197, 8))
        output["criteria"].sum().backward()
        confirmed_gradients = [
            parameter.grad for parameter in model.confirmed_model.parameters()
        ]
        self.assertTrue(all(gradient is None for gradient in confirmed_gradients))
        scale = model.multidepth_residual.intermediate_residual.residual_scale
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad).all())

    def test_outputs_have_expected_finite_shapes(self):
        torch.manual_seed(20260830)
        model = Phase24MultiDepthCriterionModel(_confirmed_model()).eval()
        output = model(torch.randn(2, 197, 8))

        self.assertEqual(output["base_criteria"].shape, (2, 7))
        self.assertEqual(output["criteria"].shape, (2, 7))
        self.assertEqual(
            output["multidepth_criterion_attention"].shape, (2, 7, 2, 196)
        )
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_state_dict_reload_is_deterministic(self):
        torch.manual_seed(20260830)
        model = Phase24MultiDepthCriterionModel(_confirmed_model()).eval()
        with torch.no_grad():
            model.multidepth_residual.intermediate_residual.residual_scale.fill_(0.1)
        prefix = torch.randn(2, 197, 8)
        with torch.inference_mode():
            expected = model(prefix)

        state = copy.deepcopy(model.state_dict())
        torch.manual_seed(9)
        reloaded = Phase24MultiDepthCriterionModel(_confirmed_model()).eval()
        reloaded.load_state_dict(state)
        with torch.inference_mode():
            actual = reloaded(prefix)

        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)


if __name__ == "__main__":
    unittest.main()
