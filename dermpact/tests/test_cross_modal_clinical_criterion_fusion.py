"""Focused tests for localized Phase 2.4B cross-modal criterion fusion."""
from __future__ import annotations

import inspect
import io

import torch

from dermpact.perception.clinical_criterion_fusion import (
    CrossModalClinicalCriterionFusion,
)


def _module(*, maximum_shift: float = 1.0) -> CrossModalClinicalCriterionFusion:
    generator = torch.Generator().manual_seed(2401)
    return CrossModalClinicalCriterionFusion(
        derm_patch_dim=12,
        clinical_patch_dim=12,
        prototype_embeddings=torch.randn(7, 2, 9, generator=generator),
        attention_dim=8,
        maximum_logit_shift=maximum_shift,
    )


def _inputs(batch: int = 3, *, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(2402)
    derm = torch.randn(batch, 196, 12, generator=generator)
    clinical = torch.randn(batch, 196, 12, generator=generator)
    base = torch.randn(batch, 7, generator=generator)
    if requires_grad:
        derm.requires_grad_()
        clinical.requires_grad_()
    return derm, clinical, base


def test_shapes_frozen_prototypes_and_exact_zero_initialization() -> None:
    module = _module()
    derm, clinical, base = _inputs()
    output = module(derm, clinical, base, torch.tensor([True, False, True]))

    assert torch.equal(output.final_logits, base)
    assert output.clinical_residual_logits.shape == (3, 7)
    assert output.criterion_gates.shape == (3, 7)
    assert output.semantic_attention_maps.shape == (3, 2, 7, 2, 196)
    assert output.cross_attention_maps.shape == (3, 2, 7, 2, 196)
    assert output.semantic_contributions.shape == (3, 7, 2)
    assert output.cross_view_contributions.shape == (3, 7, 2)
    assert output.mixture_weights.shape == (7,)
    assert module.prototype_embeddings.requires_grad is False
    assert "prototype_embeddings" not in dict(module.named_parameters())


def test_missing_rows_are_exact_fallback_and_zero_at_every_observable_path() -> None:
    module = _module()
    derm, clinical, base = _inputs()
    present = torch.tensor([False, True, False])
    with torch.no_grad():
        module.raw_output_scale.fill_(0.7)
    output = module(derm, clinical, base, present)

    assert torch.equal(output.final_logits[~present], base[~present])
    for value in (
        output.clinical_residual_logits,
        output.criterion_gates,
        output.semantic_attention_maps,
        output.cross_attention_maps,
        output.semantic_contributions,
        output.cross_view_contributions,
    ):
        assert torch.count_nonzero(value[~present]) == 0


def test_hard_shift_bound_and_bounded_mixture() -> None:
    maximum_shift = 0.31
    module = _module(maximum_shift=maximum_shift)
    derm, clinical, base = _inputs()
    with torch.no_grad():
        module.raw_output_scale.copy_(
            torch.tensor([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6])
        )
        module.gate_bias.fill_(1e6)
        module.raw_mixture_weight.copy_(
            torch.tensor([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6])
        )
    output = module(derm, clinical, base, torch.ones(3, dtype=torch.bool))

    assert torch.all(output.clinical_residual_logits.abs() <= maximum_shift)
    assert torch.all((output.mixture_weights >= 0.0) & (output.mixture_weights <= 1.0))


def test_gradients_are_localized_to_present_rows_and_all_projections_train() -> None:
    module = _module()
    derm, clinical, base = _inputs(requires_grad=True)
    present = torch.tensor([False, True, True])
    with torch.no_grad():
        module.raw_output_scale.fill_(0.4)
        module.gate_uncertainty_weight.fill_(0.2)
    module(derm, clinical, base, present).final_logits.square().sum().backward()

    assert torch.count_nonzero(derm.grad[~present]) == 0
    assert torch.count_nonzero(clinical.grad[~present]) == 0
    assert torch.count_nonzero(derm.grad[present]) > 0
    assert torch.count_nonzero(clinical.grad[present]) > 0
    for projection in (
        module.prototype_projection,
        module.derm_projection,
        module.clinical_projection,
    ):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_residual_is_sensitive_to_each_direction_of_cross_view_interaction() -> None:
    module = _module()
    derm, clinical, base = _inputs(batch=2)
    present = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        module.raw_output_scale.fill_(0.8)
        reference = module(derm, clinical, base, present).clinical_residual_logits
        derm_changed = module(
            derm.roll(17, dims=1) + 0.2,
            clinical,
            base,
            present,
        ).clinical_residual_logits
        clinical_changed = module(
            derm,
            clinical.roll(19, dims=1) - 0.2,
            base,
            present,
        ).clinical_residual_logits

    assert not torch.equal(reference, derm_changed)
    assert not torch.equal(reference, clinical_changed)


def test_state_dict_reload_is_exact_and_forward_has_no_label_or_fact_input() -> None:
    module = _module(maximum_shift=0.43)
    derm, clinical, base = _inputs(batch=2)
    present = torch.tensor([True, False])
    with torch.no_grad():
        module.raw_output_scale.copy_(torch.linspace(-0.3, 0.3, 7))
        module.raw_mixture_weight.copy_(torch.linspace(-0.4, 0.4, 7))
    expected = module(derm, clinical, base, present)
    checkpoint = io.BytesIO()
    torch.save(module.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = _module(maximum_shift=0.9)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    observed = reloaded(derm, clinical, base, present)

    for field in expected.__dataclass_fields__:
        assert torch.equal(getattr(expected, field), getattr(observed, field)), field
    parameter_names = inspect.signature(module.forward).parameters
    assert not any("label" in name or "fact" in name for name in parameter_names)
