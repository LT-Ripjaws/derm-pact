from __future__ import annotations

import torch

from dermpact.eval import dermpact_phase2_4c_r_feature_only as feature_only
from dermpact.eval import dermpact_phase2_4c_r_safe_student as safe


def _evidence() -> safe.FeatureArrays:
    generator = torch.Generator().manual_seed(61)
    return safe.FeatureArrays(
        row_ids=torch.arange(616).numpy().astype(str),
        folds=(torch.arange(616) % 5).numpy(),
        stream_features=torch.randn(616, 3, 8, 128, generator=generator).numpy(),
        stream_present=torch.ones(616, 3, dtype=torch.bool).numpy(),
        first_pass_logits=torch.randn(616, 8, generator=generator).numpy(),
        uncertainty=torch.rand(616, 8, generator=generator).numpy(),
        semantic_node_embeddings=torch.randn(8, 2, 512, generator=generator).numpy(),
        criterion_targets=torch.randint(0, 2, (616, 7), generator=generator).float().numpy(),
        melanoma_targets=torch.randint(0, 2, (616,), generator=generator).float().numpy(),
        diagnosis_targets=torch.arange(616).numpy().astype(str),
        melanocytic_mask=torch.ones(616, dtype=torch.bool).numpy(),
        clinical_available=torch.ones(616, dtype=torch.bool).numpy(),
        manifest_sha256="1" * 64,
        artifact_sha256="2" * 64,
    )


def test_feature_only_residual_does_not_depend_on_completed_logits() -> None:
    evidence = _evidence()
    model, _ = feature_only.build_model(safe.fold_arrays(evidence, 0))
    model.eval()
    features = torch.from_numpy(evidence.stream_features[:16])
    uncertainty = torch.from_numpy(evidence.uncertainty[:16])
    present = torch.from_numpy(evidence.stream_present[:16])
    left_logits = torch.from_numpy(evidence.first_pass_logits[:16])
    right_logits = left_logits + torch.randn_like(left_logits) * 3.0
    with torch.no_grad():
        model.feature_scale_raw.fill_(0.2)
        model.film_scale_raw.fill_(0.2)
    left = model(features, left_logits, uncertainty, present)
    right = model(features, right_logits, uncertainty, present)
    assert torch.allclose(left.logit_residuals, right.logit_residuals, atol=1e-7)
    assert model.logit_residual_limit == 0.25
    assert model.learned_adjacency_raw.requires_grad is False
