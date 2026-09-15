"""Auditable class-balanced criterion losses."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CriterionLossResult:
    """Loss values and per-criterion accounting returned by ``CriterionLoss``."""

    total: Tensor
    bce: Tensor
    ranking: Tensor
    ranking_contribution: Tensor
    per_criterion_bce: Tensor
    per_criterion_ranking: Tensor
    bce_numerators: Tensor
    bce_weight_sums: Tensor
    ranking_numerators: Tensor
    ranking_weight_sums: Tensor
    eligible_pair_counts: Tensor
    eligible_criteria: Tensor


class CriterionLoss(nn.Module):
    """Weighted BCE with an optional pairwise logistic ranking surrogate."""

    def __init__(
        self,
        *,
        ranking_weight: float = 0.0,
        pair_chunk_size: int = 65_536,
        denominator_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if not math.isfinite(ranking_weight) or ranking_weight < 0.0:
            raise ValueError("ranking_weight must be finite and nonnegative")
        if not isinstance(pair_chunk_size, int) or isinstance(pair_chunk_size, bool):
            raise TypeError("pair_chunk_size must be an integer")
        if pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0.0:
            raise ValueError("denominator_epsilon must be finite and positive")
        self.ranking_weight = float(ranking_weight)
        self.pair_chunk_size = pair_chunk_size
        self.denominator_epsilon = float(denominator_epsilon)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        sample_weights: Tensor,
    ) -> CriterionLossResult:
        logits, targets, weights = self._validated_inputs(
            logits, targets, sample_weights
        )
        criterion_count = logits.shape[1]

        bce_numerators: list[Tensor] = []
        bce_weight_sums: list[Tensor] = []
        per_criterion_bce: list[Tensor] = []
        ranking_numerators: list[Tensor] = []
        ranking_weight_sums: list[Tensor] = []
        per_criterion_ranking: list[Tensor] = []
        pair_counts: list[int] = []
        eligible: list[bool] = []

        for index in range(criterion_count):
            element_losses = F.binary_cross_entropy_with_logits(
                logits[:, index], targets[:, index], reduction="none"
            )
            bce_numerator = (element_losses * weights[:, index]).sum()
            bce_weight_sum = weights[:, index].sum()
            criterion_bce = bce_numerator / bce_weight_sum.clamp_min(
                self.denominator_epsilon
            )
            bce_numerators.append(bce_numerator)
            bce_weight_sums.append(bce_weight_sum)
            per_criterion_bce.append(criterion_bce)

            (
                ranking_numerator,
                ranking_weight_sum,
                criterion_ranking,
                pair_count,
            ) = self._criterion_ranking(
                logits[:, index], targets[:, index], weights[:, index]
            )
            ranking_numerators.append(ranking_numerator)
            ranking_weight_sums.append(ranking_weight_sum)
            per_criterion_ranking.append(criterion_ranking)
            pair_counts.append(pair_count)
            eligible.append(pair_count > 0)

        bce_values = torch.stack(per_criterion_bce)
        ranking_values = torch.stack(per_criterion_ranking)
        eligible_mask = torch.tensor(eligible, dtype=torch.bool, device=logits.device)
        pair_count_tensor = torch.tensor(
            pair_counts, dtype=torch.long, device=logits.device
        )
        bce = bce_values.mean()
        if any(eligible):
            ranking = ranking_values[eligible_mask].mean()
        else:
            ranking = logits.sum() * 0.0
        if self.ranking_weight == 0.0:
            # Preserve exact weighted-BCE value and graph when ranking is disabled.
            ranking_contribution = ranking * 0.0
            total = bce
        else:
            ranking_contribution = ranking * self.ranking_weight
            total = bce + ranking_contribution

        return CriterionLossResult(
            total=total,
            bce=bce,
            ranking=ranking,
            ranking_contribution=ranking_contribution,
            per_criterion_bce=bce_values,
            per_criterion_ranking=ranking_values,
            bce_numerators=torch.stack(bce_numerators),
            bce_weight_sums=torch.stack(bce_weight_sums),
            ranking_numerators=torch.stack(ranking_numerators),
            ranking_weight_sums=torch.stack(ranking_weight_sums),
            eligible_pair_counts=pair_count_tensor,
            eligible_criteria=eligible_mask,
        )

    def _criterion_ranking(
        self, logits: Tensor, targets: Tensor, weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        positive_mask = (targets == 1) & (weights > 0)
        negative_mask = (targets == 0) & (weights > 0)
        positive_logits = logits[positive_mask]
        negative_logits = logits[negative_mask]
        positive_weights = weights[positive_mask]
        negative_weights = weights[negative_mask]
        pair_count = positive_logits.numel() * negative_logits.numel()
        zero = logits.sum() * 0.0
        if pair_count == 0:
            return zero, zero, zero, 0

        numerator = zero
        normalizer = zero
        # Both axes are blocked so the temporary matrix is always bounded.
        negative_block_size = min(negative_logits.numel(), self.pair_chunk_size)
        positive_block_size = max(1, self.pair_chunk_size // negative_block_size)
        for positive_start in range(0, positive_logits.numel(), positive_block_size):
            positive_stop = positive_start + positive_block_size
            pos_logits = positive_logits[positive_start:positive_stop]
            pos_weights = positive_weights[positive_start:positive_stop]
            for negative_start in range(
                0, negative_logits.numel(), negative_block_size
            ):
                negative_stop = negative_start + negative_block_size
                neg_logits = negative_logits[negative_start:negative_stop]
                neg_weights = negative_weights[negative_start:negative_stop]
                pair_weights = pos_weights[:, None] * neg_weights[None, :]
                pair_losses = F.softplus(neg_logits[None, :] - pos_logits[:, None])
                numerator = numerator + (pair_losses * pair_weights).sum()
                normalizer = normalizer + pair_weights.sum()
        value = numerator / normalizer.clamp_min(self.denominator_epsilon)
        return numerator, normalizer, value, pair_count

    @staticmethod
    def _validated_inputs(
        logits: Tensor, targets: Tensor, sample_weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(logits, Tensor) or not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if logits.ndim != 2 or logits.shape[1] <= 0:
            raise ValueError("logits must have shape [batch, criteria] with criteria > 0")
        targets = torch.as_tensor(targets, device=logits.device, dtype=logits.dtype)
        if targets.shape != logits.shape:
            raise ValueError("targets must have the same shape as logits")
        weights = torch.as_tensor(
            sample_weights, device=logits.device, dtype=logits.dtype
        )
        if weights.ndim == 1 and weights.shape[0] == logits.shape[0]:
            weights = weights[:, None].expand_as(logits)
        elif weights.ndim == 2 and weights.shape == (logits.shape[0], 1):
            weights = weights.expand_as(logits)
        elif weights.shape != logits.shape:
            raise ValueError(
                "sample_weights must have shape [batch], [batch, 1], or [batch, criteria]"
            )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("logits must be finite")
        if not bool(torch.isfinite(targets).all()) or not bool(
            ((targets == 0) | (targets == 1)).all()
        ):
            raise ValueError("targets must contain only finite binary values")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("sample_weights must be finite and nonnegative")
        return logits, targets, weights
