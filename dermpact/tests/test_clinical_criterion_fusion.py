"""Focused interface tests for Phase 2.4B clinical criterion fusion."""
from __future__ import annotations

import io
import inspect

import pytest
import torch

from dermpact.perception.clinical_criterion_fusion import ClinicalCriterionFusion


def _module(
    *,
    patch_dim: int = 6,
    prototype_dim: int = 5,
    maximum_logit_shift: float = 1.0,
) -> ClinicalCriterionFusion:
    generator = torch.Generator().manual_seed(20260830)
    return ClinicalCriterionFusion(
        clinical_patch_dim=patch_dim,
        prototype_embeddings=torch.randn(7, 2, prototype_dim, generator=generator),
        attention_dim=4,
        maximum_logit_shift=maximum_logit_shift,
    )


def _inputs(batch: int = 4, patch_dim: int = 6):
    generator = torch.Generator().manual_seed(20260831)
    tokens = torch.randn(batch, 196, patch_dim, generator=generator)
    logits = torch.randn(batch, 7, generator=generator)
    return tokens, logits


def test_all_missing_is_bit_exact_and_all_clinical_outputs_are_exact_zero() -> None:
    module = _module()
    tokens, logits = _inputs()

    result = module(tokens, logits, torch.zeros(4, dtype=torch.bool))

    assert torch.equal(result.final_logits, logits)
    assert torch.count_nonzero(result.clinical_residual_logits) == 0
    assert torch.count_nonzero(result.criterion_gates) == 0
    assert torch.count_nonzero(result.attention_maps) == 0
    assert torch.count_nonzero(result.semantic_contributions) == 0


def test_all_missing_loss_produces_zero_clinical_gradients() -> None:
    module = _module()
    with torch.no_grad():
        module.raw_output_scale.fill_(0.3)
    tokens, logits = _inputs(batch=2)
    tokens.requires_grad_()

    result = module(tokens, logits, torch.zeros(2, dtype=torch.bool))
    result.final_logits.square().sum().backward()

    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad) == 0
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_all_present_dense_batch_matches_one_row_forwards() -> None:
    module = _module()
    with torch.no_grad():
        module.raw_output_scale.fill_(0.25)
        module.gate_uncertainty_weight.fill_(0.4)
    tokens, logits = _inputs(batch=3)
    batched = module(tokens, logits, torch.ones(3, dtype=torch.bool))
    separate = [
        module(tokens[index : index + 1], logits[index : index + 1], torch.ones(1, dtype=torch.bool))
        for index in range(3)
    ]

    row_fields = (
        "base_logits",
        "final_logits",
        "clinical_residual_logits",
        "criterion_gates",
        "attention_maps",
        "semantic_contributions",
    )
    for field in row_fields:
        expected = torch.cat([getattr(value, field) for value in separate], dim=0)
        assert torch.allclose(getattr(batched, field), expected, rtol=0.0, atol=1e-6)


def test_forward_source_has_no_sparse_row_assembly_ops() -> None:
    source = inspect.getsource(ClinicalCriterionFusion.forward)
    for prohibited in ("index_select", "index_copy", "scatter"):
        assert prohibited not in source


def test_mixed_batch_masks_every_observable_and_preserves_missing_rows() -> None:
    module = _module()
    tokens, logits = _inputs()
    present = torch.tensor([True, False, True, False])
    with torch.no_grad():
        module.raw_output_scale.fill_(0.25)

    result = module(tokens, logits, present)

    assert torch.equal(result.final_logits[~present], logits[~present])
    assert torch.count_nonzero(result.clinical_residual_logits[~present]) == 0
    assert torch.count_nonzero(result.criterion_gates[~present]) == 0
    assert torch.count_nonzero(result.attention_maps[~present]) == 0
    assert torch.count_nonzero(result.semantic_contributions[~present]) == 0
    assert torch.all(result.criterion_gates[present] > 0.0)
    assert torch.any(result.clinical_residual_logits[present] != 0.0)
    assert torch.allclose(
        result.attention_maps[present].sum(dim=-1),
        torch.ones(2, 7, 2),
    )


def test_only_present_rows_contribute_input_and_parameter_gradients() -> None:
    module = _module()
    with torch.no_grad():
        module.raw_output_scale.fill_(0.3)
        module.gate_uncertainty_weight.copy_(torch.linspace(-0.4, 0.4, 7))
    tokens, logits = _inputs()
    tokens.requires_grad_()
    present = torch.tensor([False, True, False, True])

    result = module(tokens, logits, present)
    result.final_logits.square().sum().backward()

    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad[~present]) == 0
    assert torch.count_nonzero(tokens.grad[present]) > 0
    for parameter in (
        module.prototype_projection.weight,
        module.clinical_projection.weight,
        module.log_temperature,
        module.raw_output_scale,
        module.gate_bias,
        module.gate_uncertainty_weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert module.prototype_embeddings.grad is None


def test_shapes_finiteness_temperatures_and_criterion_gates() -> None:
    module = _module()
    tokens, logits = _inputs(batch=3)
    uncertainty = torch.tensor(
        [[0.0] * 7, [0.5] * 7, [1.0] * 7], dtype=logits.dtype
    )
    with torch.no_grad():
        module.gate_bias.copy_(torch.linspace(-1.0, 1.0, 7))
        module.gate_uncertainty_weight.copy_(torch.linspace(1.0, -1.0, 7))

    result = module(
        tokens,
        logits,
        torch.ones(3, dtype=torch.bool),
        base_uncertainty=uncertainty,
    )

    assert result.final_logits.shape == (3, 7)
    assert result.clinical_residual_logits.shape == (3, 7)
    assert result.criterion_gates.shape == (3, 7)
    assert result.attention_maps.shape == (3, 7, 2, 196)
    assert result.temperatures.shape == (7,)
    assert result.semantic_contributions.shape == (3, 7, 2)
    assert result.effective_output_scales.shape == (7,)
    assert result.maximum_logit_shift.shape == ()
    for value in result.__dict__.values():
        assert torch.isfinite(value).all()
    assert torch.all((result.criterion_gates > 0.0) & (result.criterion_gates < 1.0))
    assert torch.all(result.temperatures > 0.0)
    assert torch.all(result.temperatures <= module.maximum_temperature)
    # Per-criterion parameters make this a vector gate, not a broadcast scalar.
    assert torch.unique(result.criterion_gates[0]).numel() > 1


def test_probability_and_equivalent_uncertainty_interfaces_agree() -> None:
    module = _module()
    tokens, logits = _inputs(batch=2)
    probabilities = logits.sigmoid()
    uncertainty = 1.0 - (2.0 * probabilities - 1.0).abs()
    present = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        module.raw_output_scale.fill_(0.2)
        module.gate_uncertainty_weight.fill_(0.7)

    from_probabilities = module(
        tokens, logits, present, base_probabilities=probabilities
    )
    from_uncertainty = module(
        tokens, logits, present, base_uncertainty=uncertainty
    )

    assert torch.equal(from_probabilities.final_logits, from_uncertainty.final_logits)
    assert torch.equal(
        from_probabilities.criterion_gates, from_uncertainty.criterion_gates
    )
    with pytest.raises(ValueError, match="at most one"):
        module(
            tokens,
            logits,
            present,
            base_probabilities=probabilities,
            base_uncertainty=uncertainty,
        )


def test_state_dict_reload_is_deterministic() -> None:
    module = _module(maximum_logit_shift=0.37)
    tokens, logits = _inputs(batch=2)
    present = torch.tensor([True, False])
    with torch.no_grad():
        module.raw_output_scale.copy_(torch.linspace(-0.2, 0.2, 7))
        module.gate_bias.copy_(torch.linspace(-0.5, 0.5, 7))
    expected = module(tokens, logits, present)

    checkpoint = io.BytesIO()
    torch.save(module.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = _module(maximum_logit_shift=0.91)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    observed = reloaded(tokens, logits, present)

    for field in expected.__dataclass_fields__:
        assert torch.equal(getattr(expected, field), getattr(observed, field))


def test_arbitrary_wrong_pair_is_still_present_to_the_module() -> None:
    module = _module()
    wrong_pair_tokens, logits = _inputs(batch=1)
    with torch.no_grad():
        module.raw_output_scale.fill_(0.2)

    result = module(wrong_pair_tokens, logits, torch.tensor([True]))

    assert torch.all(result.criterion_gates > 0.0)
    assert torch.allclose(result.attention_maps.sum(dim=-1), torch.ones(1, 7, 2))
    assert torch.any(result.clinical_residual_logits != 0.0)


def test_extreme_inputs_and_raw_scales_cannot_exceed_logit_shift_bound() -> None:
    maximum_shift = 0.37
    generator = torch.Generator().manual_seed(20260901)
    module = ClinicalCriterionFusion(
        clinical_patch_dim=6,
        prototype_embeddings=torch.randn(7, 2, 5, generator=generator) * 1e20,
        attention_dim=4,
        maximum_logit_shift=maximum_shift,
    )
    with torch.no_grad():
        module.raw_output_scale.copy_(
            torch.tensor([-1e6, -1e3, -100.0, 0.0, 100.0, 1e3, 1e6])
        )
        module.gate_bias.fill_(1e6)
        module.prototype_projection.weight.mul_(1e10)
        module.clinical_projection.weight.mul_(1e10)
    tokens = torch.randn(5, 196, 6, generator=generator) * 1e10
    logits = torch.randn(5, 7, generator=generator)

    result = module(tokens, logits, torch.ones(5, dtype=torch.bool))

    assert torch.isfinite(result.clinical_residual_logits).all()
    assert torch.all(result.clinical_residual_logits.abs() <= maximum_shift)
    assert torch.all(result.effective_output_scales.abs() <= 0.5 * maximum_shift)
    assert result.maximum_logit_shift.item() == pytest.approx(maximum_shift)


def test_bounded_scale_has_gradient_away_from_saturation() -> None:
    module = _module()
    tokens, logits = _inputs(batch=2)
    with torch.no_grad():
        module.raw_output_scale.copy_(torch.linspace(-0.5, 0.5, 7))

    result = module(tokens, logits, torch.ones(2, dtype=torch.bool))
    result.final_logits.square().sum().backward()

    assert module.raw_output_scale.grad is not None
    assert torch.isfinite(module.raw_output_scale.grad).all()
    assert torch.count_nonzero(module.raw_output_scale.grad) == 7


@pytest.mark.parametrize("invalid", (0.0, -1.0, float("inf"), float("nan")))
def test_maximum_logit_shift_must_be_positive_and_finite(invalid: float) -> None:
    with pytest.raises(ValueError, match="maximum_logit_shift"):
        ClinicalCriterionFusion(
            clinical_patch_dim=6,
            prototype_embeddings=torch.randn(7, 2, 5),
            attention_dim=4,
            maximum_logit_shift=invalid,
        )


@pytest.mark.parametrize(
    ("tokens_shape", "logits_shape", "mask", "message"),
    (
        ((2, 195, 6), (2, 7), torch.ones(2, dtype=torch.bool), "clinical_patch_tokens"),
        ((2, 196, 6), (2, 6), torch.ones(2, dtype=torch.bool), "base_criterion_logits"),
        ((2, 196, 6), (2, 7), torch.ones(2), "dtype torch.bool"),
    ),
)
def test_interface_rejects_invalid_shapes_and_mask_dtype(
    tokens_shape, logits_shape, mask, message
) -> None:
    module = _module()
    with pytest.raises((ValueError, TypeError), match=message):
        module(torch.randn(tokens_shape), torch.randn(logits_shape), mask)
