from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from dermpact.eval import dermpact_phase2_5_training as stage1
from dermpact.eval import dermpact_phase2_5_window_cache as window_cache
from dermpact.perception.multiclass_criterion_module import (
    MulticlassCriterionModule,
    TaskTokenAttention,
    _stable_logit,
)
from dermpact.perception.unified_perception_tail import BoundedCrossAttention


def test_task_attention_zero_gate_is_exact_identity() -> None:
    torch.manual_seed(1)
    module = TaskTokenAttention(feature_dim=8, n_heads=2).eval()
    features = torch.randn(3, 7, 8)

    observed = module(features)

    assert torch.equal(observed, features)


def test_cross_attention_missing_context_is_bit_exact() -> None:
    torch.manual_seed(2)
    module = BoundedCrossAttention(dim=8, heads=2, max_residual=0.25).eval()
    query = torch.randn(3, 5, 8)
    context = torch.randn(3, 6, 8)
    missing = torch.zeros(3, dtype=torch.bool)

    observed = module(query, context, missing)

    assert torch.equal(observed, query)


def test_cross_attention_present_residual_is_hard_bounded() -> None:
    torch.manual_seed(3)
    module = BoundedCrossAttention(dim=8, heads=2, max_residual=0.25).eval()
    with torch.no_grad():
        module.out_proj.weight.fill_(20.0)
        module.out_proj.bias.fill_(20.0)
    query = torch.randn(2, 5, 8)
    context = torch.randn(2, 6, 8)
    present = torch.ones(2, dtype=torch.bool)

    observed = module(query, context, present)

    assert float((observed - query).abs().max().detach()) <= 0.250001


def test_cross_attention_matches_reference_multihead_attention_math() -> None:
    torch.manual_seed(31)
    module = BoundedCrossAttention(dim=8, heads=2, max_residual=0.25).eval()
    with torch.no_grad():
        module.out_proj.weight.normal_(mean=0.0, std=0.1)
        module.out_proj.bias.normal_(mean=0.0, std=0.1)
    query = torch.randn(2, 5, 8)
    context = torch.randn(2, 6, 8)
    present = torch.tensor([True, False])

    observed = module(query, context, present)
    normalized_query = module.query_norm(query)
    normalized_context = module.context_norm(context)
    attended, _ = module.attention(
        normalized_query,
        normalized_context,
        normalized_context,
        need_weights=False,
    )
    residual = module.max_residual * torch.tanh(
        attended / module.max_residual
    )
    expected = torch.where(
        present.view(2, 1, 1), query + residual, query
    )

    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)


def test_compatible_layer_norm_matches_pytorch_on_cpu() -> None:
    torch.manual_seed(32)
    module = BoundedCrossAttention(dim=8, heads=2)
    tokens = torch.randn(3, 5, 8)

    observed = module._compatible_norm(module.query_norm, tokens)
    expected = module.query_norm(tokens)

    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)


def test_multiclass_criterion_gate_zero_preserves_base() -> None:
    names = [f"criterion_{index}" for index in range(7)]
    counts = {name: 3 for name in names}
    positive = {name: [2] for name in names}
    torch.manual_seed(4)
    module = MulticlassCriterionModule(
        token_dim=8,
        criterion_class_counts=counts,
        criterion_positive_indices=positive,
        criterion_names=names,
        task_feature_dim=8,
        n_attention_heads=2,
    ).eval()
    tokens = torch.randn(2, 12, 8)
    base = torch.randn(2, 7)

    output = module(tokens, None, base)

    assert torch.equal(output.final_criterion_logits, base)


def test_manual_logit_matches_torch_without_using_backend_fallback() -> None:
    probability = torch.tensor([1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6])

    observed = _stable_logit(probability)

    assert torch.allclose(observed, torch.logit(probability), atol=1e-6, rtol=1e-6)


def test_native_bce_with_logits_matches_pytorch() -> None:
    logits = torch.tensor([-20.0, -2.0, 0.0, 1.5, 20.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])

    observed = stage1._bce_with_logits_native(logits, targets, reduction="none")
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)
    observed.sum().backward(retain_graph=True)
    observed_gradient = logits.grad.detach().clone()
    logits.grad.zero_()
    expected.sum().backward()
    assert torch.allclose(observed_gradient, logits.grad, atol=1e-6, rtol=1e-6)


def test_source_binding_detects_mid_run_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("version = 1\n", encoding="utf-8")
    frozen = window_cache.source_bindings([source])
    source.write_text("version = 2\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="source drift"):
        window_cache.assert_source_bindings(frozen)


def test_resume_round_trip_restores_model_optimizer_and_epoch(tmp_path: Path) -> None:
    torch.manual_seed(5)
    original = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    loss = original(torch.randn(3, 4)).square().mean()
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in original.state_dict().items()}
    checkpoint = tmp_path / "latest.pt"
    stage1._atomic_torch(
        checkpoint,
        {
            "schema_version": "dermpact-phase2-5-stage1-resume-v3",
            "fold": 2,
            "protocol_sha256": "protocol",
            "source_bundle_sha256": "sources",
            "completed_epoch": 1,
            "trainable_state": expected,
            "optimizer_state": optimizer.state_dict(),
            "rng_state": {
                "python": __import__("random").getstate(),
                "numpy": __import__("numpy").random.get_state(),
                "torch": torch.random.get_rng_state(),
            },
            "history": [{"epoch": 1}],
        },
    )

    restored = nn.Linear(4, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    epoch, history = stage1._load_resume(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
        fold=2,
        protocol_sha256="protocol",
        source_bundle_sha256="sources",
    )

    assert epoch == 1
    assert history == [{"epoch": 1}]
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])


def test_class_weighted_ce_is_invariant_to_microbatch_one() -> None:
    logits = torch.tensor(
        [[2.0, -1.0], [-0.5, 1.5], [0.2, 0.4], [1.0, -0.5]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 1, 0])
    weights = torch.tensor([1.0, 3.0])
    denominator = float(weights[targets].sum())
    full = stage1._logical_weighted_cross_entropy(
        logits, targets, weights, denominator
    )
    accumulated = sum(
        stage1._logical_weighted_cross_entropy(
            logits[index:index + 1],
            targets[index:index + 1],
            weights,
            denominator,
        )
        for index in range(len(targets))
    )

    assert torch.allclose(accumulated, full)
