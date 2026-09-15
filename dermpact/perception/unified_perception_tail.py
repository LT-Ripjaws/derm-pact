"""Phase 2.5 unified perception tail."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TOKEN_DIM = 768
N_CRITERIA = 7
N_WINDOWS = 16
GRID_SIZE = 4


@dataclass(frozen=True)
class PerceptionFeatures:
    """Output from the unified tail — all token sources for downstream queries."""
    global_patches: Tensor
    global_pooled: Tensor
    window_patches: Tensor
    window_cls: Tensor
    regional_features: Tensor
    clinical_patches: Tensor
    clinical_pooled: Tensor
    modality_labels: Tensor


@dataclass(frozen=True)
class GlobalPerceptionFeatures:
    """Global-only tail output for diagnosis datasets without paired modalities."""

    patches: Tensor
    pooled: Tensor


class RegionTransformer(nn.Module):
    """Self-attention across window CLS embeddings with 2D grid positional encoding."""

    def __init__(
        self,
        dim: int = TOKEN_DIM,
        layers: int = 2,
        heads: int = 8,
        grid_size: int = GRID_SIZE,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.grid_size = grid_size

        if dim % heads or dim % 2:
            raise ValueError("region dim must be divisible by heads and by two")
        self.row_embed = nn.Embedding(grid_size, dim // 2)
        self.col_embed = nn.Embedding(grid_size, dim // 2)

        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn_qkv": nn.Linear(dim, 3 * dim),
                "attn_proj": nn.Linear(dim, dim),
                "norm2": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
            }))
        self.final_norm = nn.LayerNorm(dim)

    def _positional_encoding(self, positions: Tensor) -> Tensor:
        """Convert [N_WINDOWS, 2] grid positions to [N_WINDOWS, dim] embeddings."""
        row_emb = self.row_embed(positions[:, 0])
        col_emb = self.col_embed(positions[:, 1])
        return torch.cat([row_emb, col_emb], dim=-1)

    def forward(
        self,
        window_cls: Tensor,
        positions: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Process window CLS tokens with position-aware self-attention."""
        batch = window_cls.shape[0]
        pos_emb = self._positional_encoding(positions).unsqueeze(0)
        x = window_cls + pos_emb

        attn_mask = ~valid_mask.unsqueeze(1).expand(-1, valid_mask.shape[1], -1)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.heads, -1, -1)

        for layer in self.layers:
            residual = x
            x = layer["norm1"](x)
            qkv = layer["attn_qkv"](x)
            qkv = qkv.reshape(batch, -1, 3, self.heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = math.sqrt(float(self.head_dim))
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            scores = scores.masked_fill(attn_mask, -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = attn.masked_fill(attn_mask, 0.0)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(batch, -1, self.dim)
            x = residual + layer["attn_proj"](out)

            residual = x
            x = residual + layer["ffn"](layer["norm2"](x))

        return self.final_norm(x)


class BoundedCrossAttention(nn.Module):
    """Bounded cross-attention between dermoscopic and clinical patch tokens."""

    def __init__(
        self,
        dim: int = TOKEN_DIM,
        heads: int = 4,
        *,
        max_residual: float = 0.5,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        if not math.isfinite(max_residual) or max_residual <= 0.0:
            raise ValueError("max_residual must be positive and finite")
        self.max_residual = float(max_residual)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    @property
    def out_proj(self) -> nn.Linear:
        """Compatibility access to the zero-initialized residual projection."""
        return self.attention.out_proj

    @staticmethod
    def _compatible_norm(norm: nn.LayerNorm, tokens: Tensor) -> Tensor:
        """Use fused LayerNorm except on DirectML's failing LN→MHA graph."""
        if tokens.device.type != "privateuseone":
            return norm(tokens)
        centered = tokens - tokens.mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + norm.eps
        )
        if norm.elementwise_affine:
            normalized = normalized * norm.weight + norm.bias
        return normalized

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        """Apply bounded cross-attention."""
        batch = query_tokens.shape[0]
        normalized_query = self._compatible_norm(self.query_norm, query_tokens)
        normalized_context = self._compatible_norm(self.context_norm, context_tokens)
        out, _ = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        residual = self.max_residual * torch.tanh(
            out / self.max_residual
        )

        mask = context_mask.view(batch, 1, 1)
        updated = query_tokens + residual
        return torch.where(mask, updated, query_tokens)


class UnifiedPerceptionTail(nn.Module):
    """Processes global, regional, and clinical block-10 prefix tokens."""

    def __init__(
        self,
        tail_blocks: nn.ModuleList,
        tail_norm: nn.Module,
        tail_projection: nn.Module,
        shared_rel_pos_bias: nn.Module | None,
        *,
        region_layers: int = 2,
        region_heads: int = 8,
        cross_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if len(tail_blocks) != 2:
            raise ValueError("exactly two tail blocks required")

        self.blocks = copy.deepcopy(tail_blocks)
        self.norm = copy.deepcopy(tail_norm)
        self.projection = copy.deepcopy(tail_projection)
        self.shared_rel_pos_bias = (
            copy.deepcopy(shared_rel_pos_bias)
            if shared_rel_pos_bias is not None
            else None
        )
        for p in self.parameters():
            p.requires_grad = False

        self.region_transformer = RegionTransformer(
            dim=TOKEN_DIM, layers=region_layers, heads=region_heads,
        )
        self.cross_attn_block11 = BoundedCrossAttention(
            dim=TOKEN_DIM, heads=cross_attn_heads,
        )
        self.cross_attn_block12 = BoundedCrossAttention(
            dim=TOKEN_DIM, heads=cross_attn_heads,
        )

        self.modality_global = nn.Parameter(torch.zeros(1, 1, TOKEN_DIM))
        self.modality_regional = nn.Parameter(torch.zeros(1, 1, TOKEN_DIM))
        self.modality_clinical = nn.Parameter(torch.zeros(1, 1, TOKEN_DIM))

    def _get_bias(self) -> Tensor | None:
        if self.shared_rel_pos_bias is None:
            return None
        return self.shared_rel_pos_bias()

    def _run_block(self, block: nn.Module, tokens: Tensor, bias: Tensor | None) -> Tensor:
        return block(tokens, rel_pos_bias=bias)

    def forward_global_only(self, global_prefix: Tensor) -> GlobalPerceptionFeatures:
        """Run the shared trainable tail without window or clinical branches."""
        if global_prefix.ndim != 3 or global_prefix.shape[1:] != (197, TOKEN_DIM):
            raise ValueError("global_prefix must have shape [batch, 197, 768]")
        tokens = global_prefix
        bias = self._get_bias()
        for block in self.blocks:
            tokens = self._run_block(block, tokens, bias)
        normalized = self.norm(tokens)
        patches = normalized[:, 1:, :] + self.modality_global
        pooled = self.norm(normalized[:, 1:, :].mean(dim=1))
        pooled = F.normalize(self.projection(pooled), dim=-1)
        return GlobalPerceptionFeatures(patches=patches, pooled=pooled)

    def unfreeze_tail(self) -> list[nn.Parameter]:
        """Unfreeze only the approved Stage 2 tail parameters."""
        unfrozen = []
        for name, parameter in self.blocks.named_parameters():
            approved = (
                "lora_" in name
                or ".norm1." in f".{name}"
                or ".norm2." in f".{name}"
                or name.endswith("gamma_1")
                or name.endswith("gamma_2")
            )
            if approved:
                parameter.requires_grad_(True)
                unfrozen.append(parameter)
        for p in self.norm.parameters():
            p.requires_grad_(True)
            unfrozen.append(p)
        for p in self.projection.parameters():
            p.requires_grad_(True)
            unfrozen.append(p)
        return unfrozen

    def forward(
        self,
        global_prefix: Tensor,
        window_prefixes: Tensor,
        clinical_prefix: Tensor,
        patch_valid_mask: Tensor,
        window_positions: Tensor,
        clinical_present: Tensor,
    ) -> PerceptionFeatures:
        """Process all token streams through shared tail blocks."""
        batch = global_prefix.shape[0]
        bias = self._get_bias()

        global_tokens = self._run_block(self.blocks[0], global_prefix, bias)

        window_tokens_list = []
        for w in range(N_WINDOWS):
            wt = self._run_block(self.blocks[0], window_prefixes[:, w], bias)
            window_tokens_list.append(wt)
        window_tokens_b11 = torch.stack(window_tokens_list, dim=1)

        clinical_tokens = self._run_block(
            self.blocks[0], clinical_prefix, bias
        )

        global_before_fusion = global_tokens
        clinical_before_fusion = clinical_tokens
        global_tokens = self.cross_attn_block11(
            global_before_fusion, clinical_before_fusion, clinical_present,
        )
        clinical_tokens = self.cross_attn_block11(
            clinical_before_fusion, global_before_fusion, clinical_present,
        )

        global_tokens = self._run_block(self.blocks[1], global_tokens, bias)

        window_tokens_list2 = []
        for w in range(N_WINDOWS):
            wt = self._run_block(self.blocks[1], window_tokens_b11[:, w], bias)
            window_tokens_list2.append(wt)
        window_tokens = torch.stack(window_tokens_list2, dim=1)

        clinical_tokens = self._run_block(self.blocks[1], clinical_tokens, bias)

        # Symmetric cross-attention after block 12.
        global_before_fusion = global_tokens
        clinical_before_fusion = clinical_tokens
        global_tokens = self.cross_attn_block12(
            global_before_fusion, clinical_before_fusion, clinical_present,
        )
        clinical_tokens = self.cross_attn_block12(
            clinical_before_fusion, global_before_fusion, clinical_present,
        )

        # Normalize
        global_normed = self.norm(global_tokens)
        global_patches = global_normed[:, 1:, :]
        global_pooled = self.norm(global_patches.mean(dim=1))
        global_pooled = F.normalize(self.projection(global_pooled), dim=-1)

        # Window patches + CLS
        window_normed = self.norm(window_tokens.reshape(-1, 197, TOKEN_DIM))
        window_normed = window_normed.reshape(batch, N_WINDOWS, 197, TOKEN_DIM)
        window_patches = window_normed[:, :, 1:, :]
        window_cls = window_normed[:, :, 0, :]

        # Region transformer
        valid_window_mask = patch_valid_mask.any(dim=2)
        regional = self.region_transformer(
            window_cls, window_positions, valid_window_mask,
        )

        # Clinical
        clinical_normed = self.norm(clinical_tokens)
        clin_patches = clinical_normed[:, 1:, :]
        clin_pooled = self.norm(clin_patches.mean(dim=1))
        clin_pooled = F.normalize(self.projection(clin_pooled), dim=-1)

        mask_3d = clinical_present.view(batch, 1, 1).to(clin_patches.dtype)
        clin_patches = clin_patches * mask_3d
        clin_pooled = clin_pooled * clinical_present.view(batch, 1).to(clin_pooled.dtype)

        # Modality labels [B, n_tokens, 3] one-hot for downstream queries
        n_global = global_patches.shape[1]
        n_window = window_patches.shape[1] * window_patches.shape[2]
        n_clinical = clin_patches.shape[1]
        labels = torch.zeros(
            batch, n_global + n_window + n_clinical, 3,
            device=global_patches.device,
        )
        labels[:, :n_global, 0] = 1.0
        labels[:, n_global:n_global + n_window, 1] = 1.0
        labels[:, n_global + n_window:, 2] = mask_3d.squeeze(-1)

        return PerceptionFeatures(
            global_patches=global_patches + self.modality_global,
            global_pooled=global_pooled,
            window_patches=window_patches + self.modality_regional.view(1, 1, 1, -1),
            window_cls=window_cls,
            regional_features=regional + self.modality_regional,
            clinical_patches=clin_patches + self.modality_clinical * mask_3d,
            clinical_pooled=clin_pooled,
            modality_labels=labels,
        )


__all__ = [
    "BoundedCrossAttention",
    "GlobalPerceptionFeatures",
    "PerceptionFeatures",
    "RegionTransformer",
    "UnifiedPerceptionTail",
]
