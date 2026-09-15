"""Normalized-cosine semantic attention for seven Derm7pt criteria."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .criterion_residual import CriterionResidualOutput


class NormalizedCosineCriterionResidual(nn.Module):
    """Attend fixed semantic prototypes with learned criterion temperatures."""

    def __init__(
        self,
        *,
        patch_dim: int,
        prototype_embeddings: Tensor,
        attention_dim: int,
        initial_temperature: float = 10.0,
        maximum_temperature: float = 100.0,
    ) -> None:
        super().__init__()
        if patch_dim <= 0 or attention_dim <= 0:
            raise ValueError("patch_dim and attention_dim must be positive")
        if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive and finite")
        if (
            not math.isfinite(maximum_temperature)
            or maximum_temperature < initial_temperature
        ):
            raise ValueError(
                "maximum_temperature must be finite and at least initial_temperature"
            )
        prototypes = torch.as_tensor(prototype_embeddings).detach().clone().float()
        if prototypes.ndim != 3 or prototypes.shape[:2] != (7, 2):
            raise ValueError(
                "prototype_embeddings must have shape [7, 2, prototype_dim]"
            )
        if prototypes.shape[2] <= 0 or not torch.isfinite(prototypes).all():
            raise ValueError("prototype_embeddings must be non-empty and finite")

        self.patch_dim = int(patch_dim)
        self.attention_dim = int(attention_dim)
        self.maximum_temperature = float(maximum_temperature)
        self.register_buffer("prototype_embeddings", prototypes, persistent=True)
        self.prototype_projection = nn.Linear(
            int(prototypes.shape[2]), self.attention_dim, bias=False
        )
        self.patch_projection = nn.Linear(
            self.patch_dim, self.attention_dim, bias=False
        )
        self.log_temperature = nn.Parameter(
            torch.full((7,), math.log(float(initial_temperature)))
        )
        self.residual_scale = nn.Parameter(torch.zeros(7))

    def temperatures(self) -> Tensor:
        """Return positive, bounded per-criterion inverse temperatures."""
        return self.log_temperature.exp().clamp(max=self.maximum_temperature)

    def forward(
        self,
        patch_tokens: Tensor,
        base_criterion_logits: Tensor,
    ) -> CriterionResidualOutput:
        if patch_tokens.ndim != 3 or patch_tokens.shape[2] != self.patch_dim:
            raise ValueError(
                f"patch_tokens must have shape [batch, patches, {self.patch_dim}]"
            )
        if patch_tokens.shape[1] <= 0:
            raise ValueError("patch_tokens must contain at least one image patch")
        if base_criterion_logits.shape != (patch_tokens.shape[0], 7):
            raise ValueError("base_criterion_logits must have shape [batch, 7]")
        if not torch.isfinite(patch_tokens).all():
            raise ValueError("patch_tokens must be finite")

        queries = F.normalize(
            self.prototype_projection(self.prototype_embeddings), dim=-1
        )
        keys = F.normalize(self.patch_projection(patch_tokens), dim=-1)
        cosine = torch.einsum("cpd,bnd->bcpn", queries, keys)
        scores = cosine * self.temperatures().view(1, 7, 1, 1)
        attention = torch.softmax(scores, dim=-1)
        localized_cosine = (attention * cosine).sum(dim=-1)
        contrast = localized_cosine[:, :, 0] - localized_cosine[:, :, 1]
        residual_logits = contrast * self.residual_scale.unsqueeze(0)
        final_logits = base_criterion_logits + residual_logits
        return CriterionResidualOutput(
            base_logits=base_criterion_logits,
            residual_logits=residual_logits,
            final_logits=final_logits,
            attention_weights=attention,
        )


__all__ = ["NormalizedCosineCriterionResidual"]
