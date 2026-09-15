"""Comparator-preserving multi-depth extension of the confirmed c10 model."""
from __future__ import annotations

from typing import Mapping

from torch import Tensor, nn
from torch.nn import functional as F

from dermpact.perception.multidepth_criterion_residual import (
    MultiDepthCriterionResidual,
)

from .dermpact_phase2_4_model import Phase24LocalizedCriterionModel


class Phase24MultiDepthCriterionModel(nn.Module):
    """Freeze confirmed c10 and add only a block-10 semantic residual."""

    def __init__(
        self,
        confirmed_model: Phase24LocalizedCriterionModel,
        *,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(confirmed_model, Phase24LocalizedCriterionModel):
            raise TypeError(
                "confirmed_model must be a Phase24LocalizedCriterionModel"
            )
        self.confirmed_model = confirmed_model
        for parameter in self.confirmed_model.parameters():
            parameter.requires_grad = False

        final_residual = self.confirmed_model.criterion_residual
        selected_attention_dim = (
            final_residual.attention_dim
            if attention_dim is None
            else int(attention_dim)
        )
        self.multidepth_residual = MultiDepthCriterionResidual(
            patch_dim=final_residual.patch_dim,
            prototype_embeddings=final_residual.prototype_embeddings,
            attention_dim=selected_attention_dim,
        )
        self.confirmed_model.eval()

    def forward(self, prefix_tokens: Tensor) -> Mapping[str, Tensor]:
        if prefix_tokens.ndim != 3:
            raise ValueError("prefix_tokens must have shape [batch, tokens, width]")
        if prefix_tokens.shape[1] <= 1:
            raise ValueError("prefix_tokens must contain a class token and patches")

        base_model = self.confirmed_model.base_model
        tail = base_model.visual_tail
        shared_bias = (
            tail.shared_rel_pos_bias()
            if tail.shared_rel_pos_bias is not None
            else None
        )
        tokens = prefix_tokens
        for block in tail.blocks:
            tokens = block(tokens, rel_pos_bias=shared_bias)
        normalized_tokens = tail.norm(tokens)
        final_patches = normalized_tokens[:, 1:, :]
        pooled = tail.norm(final_patches.mean(dim=1))
        pooled = F.normalize(tail.projection(pooled), dim=-1)
        base_outputs = base_model.task_head(pooled)
        confirmed = self.confirmed_model.criterion_residual(
            final_patches, base_outputs["criteria"]
        )
        fused = self.multidepth_residual(
            intermediate_patch_tokens=prefix_tokens[:, 1:, :],
            base_criterion_logits=confirmed.base_logits,
            final_depth_residual_logits=confirmed.residual_logits,
            final_depth_attention_weights=confirmed.attention_weights,
        )
        return {
            "diagnosis": base_outputs["diagnosis"],
            "melanoma": base_outputs["melanoma"],
            "criteria": fused.final_logits,
            "base_criteria": fused.base_logits,
            "final_depth_criterion_residual": fused.final_depth_residual_logits,
            "multidepth_criterion_residual": fused.multidepth_residual_logits,
            "final_depth_criterion_attention": fused.final_depth_attention_weights,
            "multidepth_criterion_attention": fused.multidepth_attention_weights,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.confirmed_model.eval()
        self.confirmed_model.base_model.eval()
        self.confirmed_model.criterion_residual.eval()
        self.multidepth_residual.train(mode)
        return self


__all__ = ["Phase24MultiDepthCriterionModel"]
