"""Focused interface tests for the Phase 2.4B paired criterion wrapper."""
from __future__ import annotations

import io

import pytest
import torch
from torch import Tensor, nn

from dermpact.eval.dermpact_phase2_4b_paired_model import (
    Phase24BPairedCriterionModel,
)


class _Block(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, tokens: Tensor, *, rel_pos_bias=None) -> Tensor:
        del rel_pos_bias
        return tokens * self.scale


class _Tail(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList((_Block(1.01), _Block(0.99)))
        self.norm = nn.LayerNorm(768)
        self.shared_rel_pos_bias = None


class _Baseline(nn.Module):
    def __init__(self, *, malformed: str | None = None) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.9)
        self.diagnosis = nn.Linear(768, 5)
        self.melanoma = nn.Linear(768, 1)
        self.criteria = nn.Linear(768, 7)
        self.malformed = malformed

    def forward(self, tokens: Tensor):
        pooled = self.dropout(tokens[:, 0])
        output = {
            "diagnosis": self.diagnosis(pooled),
            "melanoma": self.melanoma(pooled).squeeze(-1),
            "criteria": self.criteria(pooled),
            "evidence": {"source": "frozen-dermoscopic-comparator"},
            "selection_provenance": "prespecified-seed",
        }
        if self.malformed == "return":
            return output["criteria"]
        if self.malformed == "missing":
            del output["melanoma"]
        elif self.malformed == "criteria_shape":
            output["criteria"] = output["criteria"][:, :6]
        return output


def _model(*, baseline: nn.Module | None = None, maximum_shift: float = 0.4):
    generator = torch.Generator().manual_seed(20260830)
    return Phase24BPairedCriterionModel(
        _Baseline() if baseline is None else baseline,
        _Tail(),
        prototype_embeddings=torch.randn(7, 2, 11, generator=generator),
        attention_dim=8,
        bottleneck_dim=4,
        maximum_logit_shift=maximum_shift,
    )


def _inputs(batch: int = 4, *, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(20260831)
    derm = torch.randn(batch, 197, 768, generator=generator)
    clinical = torch.randn(batch, 197, 768, generator=generator)
    if requires_grad:
        derm.requires_grad_()
        clinical.requires_grad_()
    return derm, clinical


def test_zero_initialization_preserves_paired_and_missing_baseline_outputs() -> None:
    model = _model().eval()
    derm, clinical = _inputs()
    present = torch.tensor([True, False, True, False])
    with torch.inference_mode():
        expected = model.baseline_model(derm)
        output = model(derm, clinical, present)

    assert torch.equal(output["criteria"], expected["criteria"])
    assert torch.equal(output["base_criteria"], expected["criteria"])
    assert torch.equal(output["diagnosis"], expected["diagnosis"])
    assert torch.equal(output["melanoma"], expected["melanoma"])
    assert output["evidence"] == expected["evidence"]
    assert output["selection_provenance"] == expected["selection_provenance"]
    assert output["baseline_outputs"].keys() == expected.keys()
    for key in ("diagnosis", "melanoma", "criteria"):
        assert torch.equal(output["baseline_outputs"][key], expected[key])


def test_mixed_mask_is_exact_for_missing_rows_and_audits_every_clinical_path() -> None:
    model = _model()
    derm, clinical = _inputs()
    present = torch.tensor([False, True, False, True])
    with torch.no_grad():
        model.clinical_fusion.raw_output_scale.fill_(0.3)
        for adapter in model.clinical_tail.adapters:
            adapter.up_projection.weight.normal_(std=0.002)

    output = model(derm, clinical, present)

    base = output["base_criteria"]
    assert torch.equal(output["criteria"][~present], base[~present])
    assert torch.any(output["criteria"][present] != base[present])
    assert torch.count_nonzero(output["clinical_residual"][~present]) == 0
    assert torch.count_nonzero(output["clinical_gates"][~present]) == 0
    assert torch.count_nonzero(output["clinical_attention"][~present]) == 0
    assert torch.count_nonzero(output["cross_modal_attention"][~present]) == 0
    assert torch.count_nonzero(
        output["clinical_semantic_contributions"][~present]
    ) == 0
    assert torch.count_nonzero(output["cross_modal_contributions"][~present]) == 0
    for contribution in output["clinical_adapter_contributions"]:
        assert contribution.shape == (4, 197, 768)
        assert torch.count_nonzero(contribution[~present]) == 0
    assert output["clinical_temperatures"].shape == (7,)
    assert output["clinical_effective_scales"].shape == (7,)
    assert output["cross_modal_mixture_weights"].shape == (7,)
    assert output["clinical_maximum_logit_shift"].shape == ()


def test_only_present_clinical_rows_have_gradients_and_baseline_stays_isolated() -> None:
    model = _model()
    derm, clinical = _inputs(requires_grad=True)
    present = torch.tensor([False, True, False, True])
    with torch.no_grad():
        model.clinical_fusion.raw_output_scale.fill_(0.25)
        for adapter in model.clinical_tail.adapters:
            adapter.up_projection.weight.normal_(std=0.002)

    output = model(derm, clinical, present)
    output["criteria"].square().sum().backward()

    assert derm.grad is None
    assert clinical.grad is not None
    assert torch.count_nonzero(clinical.grad[~present]) == 0
    assert torch.count_nonzero(clinical.grad[present]) > 0
    assert all(parameter.grad is None for parameter in model.baseline_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.baseline_model.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.clinical_tail.frozen_tail.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.clinical_tail.frozen_tail.parameters()
    )
    assert all(parameter.grad is None for parameter in model.derm_tail.parameters())
    assert all(not parameter.requires_grad for parameter in model.derm_tail.parameters())
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(
        name.startswith("clinical_tail.adapters.")
        or name.startswith("cross_modal_fusion.")
        for name in trainable
    )
    assert all(
        parameter.grad is not None
        for parameter in model.clinical_tail.adapters.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.clinical_fusion.parameters()
    )


def test_all_missing_composition_is_exact_and_has_zero_trainable_gradients() -> None:
    model = _model()
    derm, clinical = _inputs(batch=3, requires_grad=True)
    with torch.no_grad():
        model.clinical_fusion.raw_output_scale.fill_(0.3)
        for adapter in model.clinical_tail.adapters:
            adapter.up_projection.weight.normal_(std=0.002)

    output = model(derm, clinical, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(output["criteria"], output["base_criteria"])
    assert torch.count_nonzero(output["clinical_residual"]) == 0
    assert torch.count_nonzero(output["clinical_gates"]) == 0
    assert torch.count_nonzero(output["clinical_attention"]) == 0
    assert all(
        torch.count_nonzero(value) == 0
        for value in output["clinical_adapter_contributions"]
    )

    output["criteria"].sum().backward()
    assert derm.grad is None
    assert clinical.grad is not None
    assert torch.count_nonzero(clinical.grad) == 0
    for parameter in (
        *model.clinical_tail.adapters.parameters(),
        *model.cross_modal_fusion.parameters(),
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_train_never_enables_baseline_or_retained_tail_training_mode() -> None:
    model = _model()
    model.train()

    assert model.training
    assert not model.baseline_model.training
    assert not model.clinical_tail.frozen_tail.training
    assert not model.derm_tail.training
    assert model.clinical_tail.adapters.training
    assert model.clinical_fusion.training


def test_residual_is_bounded_by_exposed_maximum_shift() -> None:
    maximum_shift = 0.37
    model = _model(maximum_shift=maximum_shift)
    derm, clinical = _inputs(batch=3)
    with torch.no_grad():
        model.clinical_fusion.raw_output_scale.copy_(
            torch.tensor([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6])
        )
        model.clinical_fusion.gate_bias.fill_(1e6)
    output = model(derm, clinical, torch.ones(3, dtype=torch.bool))

    assert torch.all(output["clinical_residual"].abs() <= maximum_shift)
    assert output["clinical_maximum_logit_shift"].item() == pytest.approx(
        maximum_shift
    )


def test_state_dict_save_reload_is_deterministic() -> None:
    model = _model()
    derm, clinical = _inputs(batch=2)
    present = torch.tensor([True, False])
    with torch.no_grad():
        model.clinical_fusion.raw_output_scale.copy_(torch.linspace(-0.2, 0.2, 7))
        model.clinical_fusion.gate_bias.copy_(torch.linspace(-0.4, 0.4, 7))
        for adapter in model.clinical_tail.adapters:
            adapter.up_projection.weight.normal_(std=0.001)
    expected = model(derm, clinical, present)

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = _model(maximum_shift=0.9)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    observed = reloaded(derm, clinical, present)

    tensor_fields = (
        "diagnosis",
        "melanoma",
        "base_criteria",
        "criteria",
        "clinical_residual",
        "clinical_gates",
        "clinical_attention",
        "cross_modal_attention",
        "clinical_temperatures",
        "clinical_semantic_contributions",
        "cross_modal_contributions",
        "cross_modal_mixture_weights",
        "clinical_effective_scales",
        "clinical_maximum_logit_shift",
    )
    for field in tensor_fields:
        assert torch.equal(expected[field], observed[field]), field
    for before, after in zip(
        expected["clinical_adapter_contributions"],
        observed["clinical_adapter_contributions"],
        strict=True,
    ):
        assert torch.equal(before, after)


@pytest.mark.parametrize(
    ("baseline", "error", "message"),
    (
        (_Baseline(malformed="return"), TypeError, "return a mapping"),
        (_Baseline(malformed="missing"), KeyError, "melanoma"),
        (_Baseline(malformed="criteria_shape"), ValueError, "criteria"),
    ),
)
def test_baseline_contract_fails_closed(baseline, error, message) -> None:
    model = _model(baseline=baseline)
    derm, clinical = _inputs(batch=2)
    with pytest.raises(error, match=message):
        model(derm, clinical, torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize(
    ("derm", "clinical", "mask", "error", "message"),
    (
        (
            torch.randn(2, 196, 768),
            torch.randn(2, 196, 768),
            torch.ones(2, dtype=torch.bool),
            ValueError,
            "derm_prefix",
        ),
        (
            torch.randn(2, 197, 768),
            torch.randn(1, 197, 768),
            torch.ones(2, dtype=torch.bool),
            ValueError,
            "batch-aligned",
        ),
        (
            torch.randn(2, 197, 768),
            torch.randn(2, 197, 768),
            torch.ones(2),
            TypeError,
            "dtype torch.bool",
        ),
    ),
)
def test_input_contract_rejects_wrong_shapes_and_types(
    derm, clinical, mask, error, message
) -> None:
    with pytest.raises(error, match=message):
        _model()(derm, clinical, mask)
