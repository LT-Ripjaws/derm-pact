from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dermpact.eval import dermpact_phase2_4c_r_aligned_cache as aligned


def test_semantic_coordinates_are_rotation_invariant() -> None:
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(5, 3, 8, 2, 64, generator=generator)
    prototypes = F.normalize(
        torch.randn(14, 64, generator=generator), dim=-1
    )
    rotation, _ = torch.linalg.qr(torch.randn(64, 64, generator=generator))
    left = aligned._semantic_coordinates(
        features.flatten(start_dim=3).numpy(), prototypes
    )
    right = aligned._semantic_coordinates(
        torch.einsum("bsipd,df->bsipf", features, rotation)
        .flatten(start_dim=3)
        .numpy(),
        torch.matmul(prototypes, rotation),
    )
    assert np.allclose(left, right, atol=1e-5)
