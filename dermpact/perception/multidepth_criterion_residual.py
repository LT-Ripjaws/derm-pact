"""Intermediate-depth semantic residual fused onto a confirmed final-depth model."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .criterion_residual import LocalizedCriterionResidual


@dataclass(frozen=True)
class MultiDepthCriterionOutput:
    """Auditable decomposition of criterion logits across token depths."""

    base_logits: Tensor
    final_depth_residual_logits: Tensor
    multidepth_residual_logits: Tensor
    final_logits: Tensor
    final_depth_attention_weights: Tensor
    multidepth_attention_weights: Tensor


class MultiDepthCriterionResidual(nn.Module):
    """Add a zero-initialized block-10 residual to a final-depth residual."""

    def __init__(
        self,
        *,
        patch_dim: int,
        prototype_embeddings: Tensor,
        attention_dim: int,
    ) -> None:
        super().__init__()
        self.intermediate_residual = LocalizedCriterionResidual(
            patch_dim=patch_dim,
            prototype_embeddings=prototype_embeddings,
            attention_dim=attention_dim,
        )

    def forward(
        self,
        *,
        intermediate_patch_tokens: Tensor,
        base_criterion_logits: Tensor,
        final_depth_residual_logits: Tensor,
        final_depth_attention_weights: Tensor,
    ) -> MultiDepthCriterionOutput:
        if final_depth_residual_logits.shape != base_criterion_logits.shape:
            raise ValueError(
                "final_depth_residual_logits must match base_criterion_logits"
            )
        confirmed_logits = base_criterion_logits + final_depth_residual_logits
        intermediate = self.intermediate_residual(
            intermediate_patch_tokens, confirmed_logits
        )
        return MultiDepthCriterionOutput(
            base_logits=base_criterion_logits,
            final_depth_residual_logits=final_depth_residual_logits,
            multidepth_residual_logits=intermediate.residual_logits,
            final_logits=intermediate.final_logits,
            final_depth_attention_weights=final_depth_attention_weights,
            multidepth_attention_weights=intermediate.attention_weights,
        )


__all__ = ["MultiDepthCriterionOutput", "MultiDepthCriterionResidual"]
