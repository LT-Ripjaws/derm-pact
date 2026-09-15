"""Focused interface tests for the Phase 2.4B clinical visual tail."""
from __future__ import annotations

import io
import inspect

import pytest
import torch
from torch import Tensor, nn

from dermpact.perception.clinical_tail_adapter import ClinicalTailAdapter


class _CallRecorder:
    bias_calls = 0
    block_calls: list[tuple[int, Tensor | None]] = []

    @classmethod
    def reset(cls) -> None:
        cls.bias_calls = 0
        cls.block_calls = []


class _SyntheticBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.25))

    def forward(self) -> Tensor:
        _CallRecorder.bias_calls += 1
        return self.value.reshape(1, 1, 1)


class _SyntheticBlock(nn.Module):
    def __init__(self, block_id: int, scale: float) -> None:
        super().__init__()
        self.block_id = block_id
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, tokens: Tensor, *, rel_pos_bias: Tensor | None) -> Tensor:
        _CallRecorder.block_calls.append((self.block_id, rel_pos_bias))
        bias = 0.0 if rel_pos_bias is None else rel_pos_bias
        return tokens * self.scale + bias


class _SyntheticTail(nn.Module):
    def __init__(self, *, with_bias: bool = True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_SyntheticBlock(11, 1.01), _SyntheticBlock(12, 0.99)]
        )
        self.norm = nn.LayerNorm(768)
        self.shared_rel_pos_bias = _SyntheticBias() if with_bias else None


def _tokens(batch: int = 4, *, requires_grad: bool = False) -> Tensor:
    generator = torch.Generator().manual_seed(20260830)
    return torch.randn(
        batch, 197, 768, generator=generator, requires_grad=requires_grad
    )


def _plain_tail(tail: _SyntheticTail, tokens: Tensor) -> Tensor:
    bias = tail.shared_rel_pos_bias() if tail.shared_rel_pos_bias is not None else None
    for block in tail.blocks:
        tokens = block(tokens, rel_pos_bias=bias)
    return tail.norm(tokens)[:, 1:, :]


def test_zero_initialization_exactly_matches_the_frozen_tail_on_present_rows() -> None:
    source = _SyntheticTail()
    expected_tail = _SyntheticTail()
    expected_tail.load_state_dict(source.state_dict())
    module = ClinicalTailAdapter(source)
    tokens = _tokens(batch=3)
    present = torch.tensor([True, False, True])

    _CallRecorder.reset()
    expected = _plain_tail(expected_tail, tokens[present])
    _CallRecorder.reset()
    result = module(tokens, present)

    assert torch.equal(result.final_patch_tokens[present], expected)
    assert torch.count_nonzero(result.final_patch_tokens[~present]) == 0
    assert len(result.adapter_contributions) == 2
    for contribution in result.adapter_contributions:
        assert contribution.shape == (3, 197, 768)
        assert torch.count_nonzero(contribution) == 0


def test_only_present_rows_receive_gradients_in_dense_tail() -> None:
    module = ClinicalTailAdapter(_SyntheticTail())
    tokens = _tokens(requires_grad=True)
    present = torch.tensor([False, True, False, True])
    with torch.no_grad():
        for adapter in module.adapters:
            adapter.up_projection.weight.normal_(std=0.01)

    _CallRecorder.reset()
    result = module(tokens, present)
    result.final_patch_tokens.square().sum().backward()

    assert _CallRecorder.bias_calls == 1
    assert [item[0] for item in _CallRecorder.block_calls] == [11, 12]
    assert all(item[1] is _CallRecorder.block_calls[0][1] for item in _CallRecorder.block_calls)
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad[~present]) == 0
    assert torch.count_nonzero(tokens.grad[present]) > 0
    assert all(parameter.grad is None for parameter in module.frozen_tail.parameters())
    assert all(parameter.requires_grad is False for parameter in module.frozen_tail.parameters())
    assert all(parameter.requires_grad for parameter in module.adapters.parameters())
    assert all(parameter.grad is not None for parameter in module.adapters.parameters())
    for contribution in result.adapter_contributions:
        assert torch.count_nonzero(contribution[~present]) == 0


def test_all_missing_dense_path_is_exact_zero_with_zero_gradients() -> None:
    module = ClinicalTailAdapter(_SyntheticTail())
    tokens = _tokens(batch=2, requires_grad=True)
    _CallRecorder.reset()

    result = module(tokens, torch.zeros(2, dtype=torch.bool))
    result.final_patch_tokens.sum().backward()

    assert _CallRecorder.bias_calls == 1
    assert [item[0] for item in _CallRecorder.block_calls] == [11, 12]
    assert torch.count_nonzero(result.final_patch_tokens) == 0
    assert all(torch.count_nonzero(value) == 0 for value in result.adapter_contributions)
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad) == 0
    assert all(parameter.grad is None for parameter in module.frozen_tail.parameters())
    assert all(parameter.grad is not None for parameter in module.adapters.parameters())
    assert all(
        torch.count_nonzero(parameter.grad) == 0
        for parameter in module.adapters.parameters()
    )


def test_all_present_dense_path_matches_independent_frozen_tail() -> None:
    source = _SyntheticTail()
    expected_tail = _SyntheticTail()
    expected_tail.load_state_dict(source.state_dict())
    module = ClinicalTailAdapter(source)
    tokens = _tokens(batch=3)

    expected = _plain_tail(expected_tail, tokens)
    observed = module(tokens, torch.ones(3, dtype=torch.bool))

    assert torch.equal(observed.final_patch_tokens, expected)
    assert all(torch.count_nonzero(value) == 0 for value in observed.adapter_contributions)


def test_forward_source_has_no_sparse_row_assembly_ops() -> None:
    source = inspect.getsource(ClinicalTailAdapter.forward)
    for prohibited in ("index_select", "index_copy", "scatter"):
        assert prohibited not in source


def test_source_tail_is_copied_and_frozen_tail_stays_in_eval_mode() -> None:
    source = _SyntheticTail()
    module = ClinicalTailAdapter(source)
    module.train()

    assert module.training is True
    assert module.adapters.training is True
    assert module.frozen_tail.training is False
    assert module.frozen_tail.blocks.training is False
    assert module.frozen_tail.norm.training is False
    assert module.frozen_tail.blocks[0] is not source.blocks[0]
    with torch.no_grad():
        source.blocks[0].scale.fill_(9.0)
    assert module.frozen_tail.blocks[0].scale.item() != 9.0


def test_state_dict_save_reload_is_bit_deterministic() -> None:
    source = _SyntheticTail()
    module = ClinicalTailAdapter(source)
    tokens = _tokens(batch=2)
    present = torch.tensor([True, False])
    with torch.no_grad():
        for index, adapter in enumerate(module.adapters, start=1):
            adapter.up_projection.weight.normal_(mean=0.01 * index, std=0.001)
            adapter.up_projection.bias.fill_(0.02 * index)
    expected = module(tokens, present)

    checkpoint = io.BytesIO()
    torch.save(module.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = ClinicalTailAdapter(_SyntheticTail())
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    observed = reloaded(tokens, present)

    assert torch.equal(expected.final_patch_tokens, observed.final_patch_tokens)
    for before, after in zip(
        expected.adapter_contributions,
        observed.adapter_contributions,
        strict=True,
    ):
        assert torch.equal(before, after)


@pytest.mark.parametrize(
    ("tokens", "mask", "error", "message"),
    (
        (torch.randn(2, 196, 768), torch.ones(2, dtype=torch.bool), ValueError, "shape"),
        (torch.randn(2, 197, 767), torch.ones(2, dtype=torch.bool), ValueError, "shape"),
        (torch.ones(2, 197, 768, dtype=torch.int64), torch.ones(2, dtype=torch.bool), TypeError, "floating"),
        (torch.randn(2, 197, 768), torch.ones(2, 1, dtype=torch.bool), ValueError, "mask"),
        (torch.randn(2, 197, 768), torch.ones(2), TypeError, "dtype torch.bool"),
    ),
)
def test_interface_rejects_invalid_shapes_and_dtypes(
    tokens: Tensor, mask: Tensor, error: type[Exception], message: str
) -> None:
    module = ClinicalTailAdapter(_SyntheticTail())
    with pytest.raises(error, match=message):
        module(tokens, mask)


def test_nonfinite_input_and_invalid_tail_outputs_fail_closed() -> None:
    module = ClinicalTailAdapter(_SyntheticTail())
    tokens = _tokens(batch=1)
    tokens[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        module(tokens, torch.tensor([True]))

    broken = _SyntheticTail(with_bias=False)
    with torch.no_grad():
        broken.blocks[0].scale.fill_(float("inf"))
    broken_module = ClinicalTailAdapter(broken)
    with pytest.raises(ValueError, match="block output must be finite"):
        broken_module(_tokens(batch=1), torch.tensor([True]))


def test_construction_rejects_noncanonical_tail_or_bottleneck() -> None:
    wrong_tail = _SyntheticTail()
    wrong_tail.blocks.append(_SyntheticBlock(13, 1.0))
    with pytest.raises(ValueError, match="exactly two"):
        ClinicalTailAdapter(wrong_tail)
    with pytest.raises(ValueError, match="positive"):
        ClinicalTailAdapter(_SyntheticTail(), bottleneck_dim=0)
