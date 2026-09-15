"""Localized, case-independent semantic residual for seven Derm7pt criteria."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CriterionResidualOutput:
    base_logits: Tensor
    residual_logits: Tensor
    final_logits: Tensor
    attention_weights: Tensor


class LocalizedCriterionResidual(nn.Module):
    """Attend fixed positive/negative criterion prototypes to image patch tokens."""

    def __init__(
        self,
        *,
        patch_dim: int,
        prototype_embeddings: Tensor,
        attention_dim: int,
    ) -> None:
        super().__init__()
        if patch_dim <= 0 or attention_dim <= 0:
            raise ValueError("patch_dim and attention_dim must be positive")
        prototypes = torch.as_tensor(prototype_embeddings).detach().clone().float()
        if prototypes.ndim != 3 or prototypes.shape[:2] != (7, 2):
            raise ValueError(
                "prototype_embeddings must have shape [7, 2, prototype_dim]"
            )
        if prototypes.shape[2] <= 0 or not torch.isfinite(prototypes).all():
            raise ValueError("prototype_embeddings must be non-empty and finite")
        self.patch_dim = int(patch_dim)
        self.attention_dim = int(attention_dim)
        self.register_buffer("prototype_embeddings", prototypes, persistent=True)
        self.prototype_projection = nn.Linear(
            int(prototypes.shape[2]), self.attention_dim, bias=False
        )
        self.patch_key = nn.Linear(self.patch_dim, self.attention_dim, bias=False)
        self.patch_value = nn.Linear(self.patch_dim, self.attention_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(7))

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

        queries = F.normalize(
            self.prototype_projection(self.prototype_embeddings), dim=-1
        )
        keys = self.patch_key(patch_tokens)
        values = self.patch_value(patch_tokens)
        scores = torch.matmul(
            queries.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1, -1),
            keys.transpose(1, 2).unsqueeze(1),
        ) / math.sqrt(float(self.attention_dim))
        attention = torch.softmax(scores, dim=-1)
        localized = torch.matmul(attention, values.unsqueeze(1))
        polarity_scores = (
            localized * queries.unsqueeze(0)
        ).sum(dim=-1)
        contrast = polarity_scores[:, :, 0] - polarity_scores[:, :, 1]
        residual_logits = contrast * self.residual_scale.unsqueeze(0)
        final_logits = base_criterion_logits + residual_logits
        return CriterionResidualOutput(
            base_logits=base_criterion_logits,
            residual_logits=residual_logits,
            final_logits=final_logits,
            attention_weights=attention,
        )
