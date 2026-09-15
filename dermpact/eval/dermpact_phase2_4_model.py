"""Phase 2.4 localized criterion-residual wrapper."""
from __future__ import annotations

from typing import Mapping

from torch import Tensor, nn
from torch.nn import functional as F

from dermpact.perception.criterion_residual import LocalizedCriterionResidual

from .dermpact_phase2_backbone_adaptation import AdaptedHybridModel


class Phase24LocalizedCriterionModel(nn.Module):
    """Freeze a Phase 2.3 model and add localized criterion residuals."""

    def __init__(
        self,
        base_model: AdaptedHybridModel,
        *,
        prototype_embeddings: Tensor,
        attention_dim: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        projection = self.base_model.visual_tail.projection
        if not hasattr(projection, "in_features"):
            raise TypeError("visual tail projection must expose in_features")
        self.criterion_residual = LocalizedCriterionResidual(
            patch_dim=int(projection.in_features),
            prototype_embeddings=prototype_embeddings,
            attention_dim=attention_dim,
        )
        self.base_model.eval()

    def forward(self, prefix_tokens: Tensor) -> Mapping[str, Tensor]:
        tail = self.base_model.visual_tail
        shared_bias = (
            tail.shared_rel_pos_bias()
            if tail.shared_rel_pos_bias is not None
            else None
        )
        tokens = prefix_tokens
        for block in tail.blocks:
            tokens = block(tokens, rel_pos_bias=shared_bias)
        normalized_tokens = tail.norm(tokens)
        patches = normalized_tokens[:, 1:, :]

        pooled = tail.norm(patches.mean(dim=1))
        pooled = tail.projection(pooled)
        pooled = F.normalize(pooled, dim=-1)
        base_outputs = self.base_model.task_head(pooled)
        residual = self.criterion_residual(patches, base_outputs["criteria"])

        return {
            "diagnosis": base_outputs["diagnosis"],
            "melanoma": base_outputs["melanoma"],
            "criteria": residual.final_logits,
            "base_criteria": residual.base_logits,
            "criterion_residual": residual.residual_logits,
            "criterion_attention": residual.attention_weights,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.base_model.visual_tail.eval()
        self.base_model.task_head.eval()
        self.criterion_residual.train(mode)
        return self


__all__ = ["Phase24LocalizedCriterionModel"]
