"""Focused tests for the independent sharp-attention c10 extension."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from dermpact.eval.dermpact_phase2_backbone_adaptation import AdaptedHybridModel
from dermpact.eval.dermpact_phase2_4_model import Phase24LocalizedCriterionModel
from dermpact.eval.dermpact_phase2_4a_sharp_model import (
    Phase24SharpCriterionModel,
)


class _TailBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.calls = 0

    def forward(self, values, *, rel_pos_bias=None):
        del rel_pos_bias
        self.calls += 1
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
            torch.linspace(-0.3, 0.3, 7)
        )
    return model


class Phase24SharpCriterionModelTest(unittest.TestCase):
    def test_zero_sharp_scale_is_exact_trained_c10_model(self):
        torch.manual_seed(20260830)
        confirmed = _confirmed_model()
        model = Phase24SharpCriterionModel(confirmed).eval()
        prefix = torch.randn(3, 197, 8)

        with torch.inference_mode():
            expected = confirmed(prefix)
            for block in confirmed.base_model.visual_tail.blocks:
                block.calls = 0
            actual = model(prefix)

        for name in ("diagnosis", "melanoma", "criteria"):
            self.assertTrue(torch.equal(actual[name], expected[name]), name)
        self.assertTrue(torch.equal(actual["c10_criteria"], expected["criteria"]))
        self.assertTrue(
            torch.equal(
                actual["c10_criterion_residual"],
                expected["criterion_residual"],
            )
        )
        self.assertTrue(
            torch.equal(
                actual["c10_criterion_attention"],
                expected["criterion_attention"],
            )
        )
        self.assertEqual(
            torch.count_nonzero(actual["sharp_criterion_residual"]).item(), 0
        )
        self.assertTrue(
            all(
                block.calls == 1
                for block in confirmed.base_model.visual_tail.blocks
            )
        )

    def test_only_sharp_residual_is_trainable_and_receives_gradient(self):
        torch.manual_seed(20260830)
        model = Phase24SharpCriterionModel(_confirmed_model()).train()
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("sharp_residual.") for name in trainable))

        model(torch.randn(2, 197, 8))["criteria"].sum().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.confirmed_model.parameters()
            )
        )
        self.assertIsNotNone(model.sharp_residual.residual_scale.grad)
        self.assertTrue(
            torch.isfinite(model.sharp_residual.residual_scale.grad).all()
        )

    def test_evidence_shapes_are_finite_and_temperature_changes_attention(self):
        torch.manual_seed(20260830)
        model = Phase24SharpCriterionModel(_confirmed_model()).eval()
        prefix = torch.randn(2, 197, 8)
        with torch.inference_mode():
            broad = model(prefix)
            model.sharp_residual.log_temperature.fill_(
                torch.log(torch.tensor(50.0))
            )
            sharp = model(prefix)

        self.assertEqual(sharp["sharp_criterion_attention"].shape, (2, 7, 2, 196))
        self.assertEqual(sharp["sharp_temperatures"].shape, (7,))
        self.assertTrue(torch.isfinite(sharp["sharp_criterion_attention"]).all())
        self.assertTrue(torch.isfinite(sharp["sharp_temperatures"]).all())
        self.assertFalse(
            torch.equal(
                sharp["sharp_criterion_attention"],
                broad["sharp_criterion_attention"],
            )
        )
        self.assertTrue(torch.equal(sharp["criteria"], broad["criteria"]))

    def test_state_dict_reload_is_deterministic(self):
        torch.manual_seed(20260830)
        model = Phase24SharpCriterionModel(_confirmed_model()).eval()
        with torch.no_grad():
            model.sharp_residual.residual_scale.fill_(0.1)
            model.sharp_residual.log_temperature.add_(0.2)
        prefix = torch.randn(2, 197, 8)
        with torch.inference_mode():
            expected = model(prefix)

        state = copy.deepcopy(model.state_dict())
        torch.manual_seed(17)
        reloaded = Phase24SharpCriterionModel(_confirmed_model()).eval()
        reloaded.load_state_dict(state)
        with torch.inference_mode():
            actual = reloaded(prefix)

        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)


if __name__ == "__main__":
    unittest.main()
