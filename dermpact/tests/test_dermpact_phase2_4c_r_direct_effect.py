from __future__ import annotations

import torch

from dermpact.eval import dermpact_phase2_4c_r_direct_effect as direct
from dermpact.eval import dermpact_phase2_4c_r_safe_student as safe


def test_direct_effect_penalties_detect_shift_and_base_compression() -> None:
    base = torch.linspace(-2.0, 2.0, 80).reshape(10, 8)
    base = base - base.mean(dim=0, keepdim=True)
    shift = torch.full_like(base, 0.2)
    mean_penalty, correlation_penalty = (
        direct._residual_direct_effect_penalties(shift, base)
    )
    assert mean_penalty > 0.0
    assert correlation_penalty == 0.0
    compression = -0.2 * base
    mean_penalty, correlation_penalty = (
        direct._residual_direct_effect_penalties(compression, base)
    )
    assert mean_penalty < 1e-8
    assert correlation_penalty > 0.99


def test_zero_residual_correlation_penalty_has_finite_gradient() -> None:
    base = torch.randn(20, 8)
    residual = torch.zeros_like(base, requires_grad=True)
    mean_penalty, correlation_penalty = (
        direct._residual_direct_effect_penalties(residual, base)
    )
    (mean_penalty + correlation_penalty).backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


def test_amendment_model_freezes_adjacency_and_halves_residual_limit() -> None:
    evidence = safe.FeatureArrays(
        row_ids=torch.arange(616).numpy().astype(str),
        folds=(torch.arange(616) % 5).numpy(),
        stream_features=torch.randn(616, 3, 8, 128).numpy(),
        stream_present=torch.ones(616, 3, dtype=torch.bool).numpy(),
        first_pass_logits=torch.randn(616, 8).numpy(),
        uncertainty=torch.rand(616, 8).numpy(),
        semantic_node_embeddings=torch.randn(8, 2, 512).numpy(),
        criterion_targets=torch.randint(0, 2, (616, 7)).float().numpy(),
        melanoma_targets=torch.randint(0, 2, (616,)).float().numpy(),
        diagnosis_targets=torch.arange(616).numpy().astype(str),
        melanocytic_mask=torch.ones(616, dtype=torch.bool).numpy(),
        clinical_available=torch.ones(616, dtype=torch.bool).numpy(),
        manifest_sha256="1" * 64,
        artifact_sha256="2" * 64,
    )
    arrays = safe.fold_arrays(evidence, 0)
    model, _ = direct.build_model(arrays)
    assert model.logit_residual_limit == 0.25
    assert model.learned_adjacency_raw.requires_grad is False
