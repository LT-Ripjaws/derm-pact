"""Frozen clinical visual tail with trainable residual bottleneck adapters."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn


_TOKENS_PER_IMAGE = 197
_PATCHES_PER_IMAGE = 196
_HIDDEN_SIZE = 768
_TAIL_BLOCKS = 2


@dataclass(frozen=True)
class ClinicalTailAdapterOutput:
    """Auditable output of the clinical-only visual tail."""

    final_patch_tokens: Tensor
    adapter_contributions: tuple[Tensor, Tensor]


class _BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.down_projection = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_projection = nn.Linear(bottleneck_dim, hidden_size)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.up_projection(self.activation(self.down_projection(tokens)))


class _FrozenClinicalTail(nn.Module):
    """Private frozen copy of blocks 11--12, norm, and shared rel-pos bias."""

    def __init__(self, retained_visual_tail: nn.Module) -> None:
        super().__init__()
        if not isinstance(retained_visual_tail, nn.Module):
            raise TypeError("retained_visual_tail must be a torch.nn.Module")
        if not hasattr(retained_visual_tail, "blocks"):
            raise TypeError("retained_visual_tail must expose blocks")
        if not hasattr(retained_visual_tail, "norm"):
            raise TypeError("retained_visual_tail must expose norm")

        source_blocks = retained_visual_tail.blocks
        if len(source_blocks) != _TAIL_BLOCKS:
            raise ValueError("retained_visual_tail must contain exactly two blocks")
        if not isinstance(retained_visual_tail.norm, nn.Module):
            raise TypeError("retained_visual_tail.norm must be a torch.nn.Module")
        source_bias = getattr(retained_visual_tail, "shared_rel_pos_bias", None)
        if source_bias is not None and not isinstance(source_bias, nn.Module):
            raise TypeError(
                "retained_visual_tail.shared_rel_pos_bias must be a module or None"
            )

        self.blocks = nn.ModuleList(copy.deepcopy(list(source_blocks)))
        self.norm = copy.deepcopy(retained_visual_tail.norm)
        self.shared_rel_pos_bias = copy.deepcopy(source_bias)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def shared_bias(self) -> Tensor | None:
        if self.shared_rel_pos_bias is None:
            return None
        bias = self.shared_rel_pos_bias()
        if not isinstance(bias, Tensor):
            raise TypeError("shared_rel_pos_bias must return a tensor")
        return bias

    def train(self, mode: bool = True):
        super().train(False)
        return self


class ClinicalTailAdapter(nn.Module):
    """Run aligned clinical prefixes through a frozen tail plus two adapters."""

    def __init__(
        self,
        retained_visual_tail: nn.Module,
        *,
        bottleneck_dim: int = 32,
    ) -> None:
        super().__init__()
        if isinstance(bottleneck_dim, bool) or not isinstance(bottleneck_dim, int):
            raise TypeError("bottleneck_dim must be an integer")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        self.frozen_tail = _FrozenClinicalTail(retained_visual_tail)
        self.adapters = nn.ModuleList(
            _BottleneckAdapter(_HIDDEN_SIZE, bottleneck_dim)
            for _ in range(_TAIL_BLOCKS)
        )

    def forward(
        self,
        clinical_prefix_tokens: Tensor,
        clinical_present_mask: Tensor,
    ) -> ClinicalTailAdapterOutput:
        if not isinstance(clinical_prefix_tokens, Tensor):
            raise TypeError("clinical_prefix_tokens must be a torch.Tensor")
        if clinical_prefix_tokens.shape != (
            clinical_prefix_tokens.shape[0] if clinical_prefix_tokens.ndim else 0,
            _TOKENS_PER_IMAGE,
            _HIDDEN_SIZE,
        ):
            raise ValueError(
                "clinical_prefix_tokens must have shape [batch, 197, 768]"
            )
        if not clinical_prefix_tokens.is_floating_point():
            raise TypeError("clinical_prefix_tokens must have a floating dtype")
        if not torch.isfinite(clinical_prefix_tokens).all():
            raise ValueError("clinical_prefix_tokens must be finite")

        batch = clinical_prefix_tokens.shape[0]
        if not isinstance(clinical_present_mask, Tensor):
            raise TypeError("clinical_present_mask must be a torch.Tensor")
        if clinical_present_mask.shape != (batch,):
            raise ValueError("clinical_present_mask must have shape [batch]")
        if clinical_present_mask.dtype is not torch.bool:
            raise TypeError("clinical_present_mask must have dtype torch.bool")
        if clinical_present_mask.device != clinical_prefix_tokens.device:
            raise ValueError(
                "clinical_present_mask must be on the same device as the tokens"
            )

        token_mask = clinical_present_mask.view(batch, 1, 1)
        tokens = torch.where(
            token_mask,
            clinical_prefix_tokens,
            torch.zeros_like(clinical_prefix_tokens),
        )
        shared_bias = self.frozen_tail.shared_bias()
        audited_contributions: list[Tensor] = []
        for block, adapter in zip(
            self.frozen_tail.blocks, self.adapters, strict=True
        ):
            block_output = block(tokens, rel_pos_bias=shared_bias)
            if not isinstance(block_output, Tensor):
                raise TypeError("retained visual blocks must return a tensor")
            if block_output.shape != tokens.shape:
                raise ValueError("retained visual blocks must preserve token shape")
            if not torch.isfinite(block_output).all():
                raise ValueError("retained visual block output must be finite")
            contribution = adapter(block_output)
            if not torch.isfinite(contribution).all():
                raise ValueError("adapter contribution must be finite")
            audited_contribution = torch.where(
                token_mask,
                contribution,
                torch.zeros_like(contribution),
            )
            audited_contributions.append(
                audited_contribution
            )
            tokens = torch.where(
                token_mask,
                block_output + contribution,
                torch.zeros_like(block_output),
            )

        normalized = self.frozen_tail.norm(tokens)
        if not isinstance(normalized, Tensor):
            raise TypeError("retained visual tail norm must return a tensor")
        if normalized.shape != tokens.shape:
            raise ValueError("retained visual tail norm must preserve token shape")
        if not torch.isfinite(normalized).all():
            raise ValueError("retained visual tail norm output must be finite")
        final_patches = torch.where(
            token_mask,
            normalized[:, 1:, :],
            torch.zeros_like(normalized[:, 1:, :]),
        )
        return ClinicalTailAdapterOutput(
            final_patch_tokens=final_patches,
            adapter_contributions=(
                audited_contributions[0],
                audited_contributions[1],
            ),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_tail.eval()
        return self


__all__ = ["ClinicalTailAdapter", "ClinicalTailAdapterOutput"]
