"""Comparator-preserving sharp-attention extension of confirmed c10."""
from __future__ import annotations

from typing import Mapping

from torch import Tensor, nn
from torch.nn import functional as F

from dermpact.perception.normalized_cosine_criterion_residual import (
    NormalizedCosineCriterionResidual,
)

from .dermpact_phase2_4_model import Phase24LocalizedCriterionModel


class Phase24SharpCriterionModel(nn.Module):
    """Freeze confirmed c10 and add an independent final-depth sharp residual."""

    def __init__(
        self,
        confirmed_model: Phase24LocalizedCriterionModel,
        *,
        attention_dim: int | None = None,
        initial_temperature: float = 10.0,
        maximum_temperature: float = 100.0,
    ) -> None:
        super().__init__()
        if not isinstance(confirmed_model, Phase24LocalizedCriterionModel):
            raise TypeError(
                "confirmed_model must be a Phase24LocalizedCriterionModel"
            )
        self.confirmed_model = confirmed_model
        for parameter in self.confirmed_model.parameters():
            parameter.requires_grad = False

        c10_residual = self.confirmed_model.criterion_residual
        selected_attention_dim = (
            c10_residual.attention_dim
            if attention_dim is None
            else int(attention_dim)
        )
        self.sharp_residual = NormalizedCosineCriterionResidual(
            patch_dim=c10_residual.patch_dim,
            prototype_embeddings=c10_residual.prototype_embeddings,
            attention_dim=selected_attention_dim,
            initial_temperature=initial_temperature,
            maximum_temperature=maximum_temperature,
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

        c10 = self.confirmed_model.criterion_residual(
            final_patches, base_outputs["criteria"]
        )
        sharp = self.sharp_residual(final_patches, c10.final_logits)
        return {
            "diagnosis": base_outputs["diagnosis"],
            "melanoma": base_outputs["melanoma"],
            "criteria": sharp.final_logits,
            "final_criteria": sharp.final_logits,
            "base_criteria": c10.base_logits,
            "c10_criteria": c10.final_logits,
            "c10_criterion_residual": c10.residual_logits,
            "sharp_criterion_residual": sharp.residual_logits,
            "c10_criterion_attention": c10.attention_weights,
            "sharp_criterion_attention": sharp.attention_weights,
            "sharp_temperatures": self.sharp_residual.temperatures(),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.confirmed_model.eval()
        self.confirmed_model.base_model.eval()
        self.confirmed_model.criterion_residual.eval()
        self.sharp_residual.train(mode)
        return self


__all__ = ["Phase24SharpCriterionModel"]
