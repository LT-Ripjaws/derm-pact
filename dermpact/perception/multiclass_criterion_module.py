"""Phase 2.5 multiclass criterion module."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


N_CRITERIA = 7
N_QUERIES = 8


def _stable_logit(probability: Tensor) -> Tensor:
    """DirectML-native logit for already-clamped probabilities."""
    return torch.log(probability) - torch.log(1.0 - probability)


@dataclass(frozen=True)
class MulticlassCriterionOutput:
    raw_logits: dict[str, Tensor]
    derived_binary_logits: Tensor
    criterion_residual: Tensor
    final_criterion_logits: Tensor
    diagnosis_features: Tensor
    task_token_features: Tensor
    attention_weights: Tensor
    residual_gates: Tensor
    checklist_melanoma_logit: Tensor
    checklist_weights: Tensor


class TwoWayLoss(nn.Module):
    """Two-Way Loss (Kobayashi CVPR 2023 / SkinM2Former)."""

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        class_weights: Tensor | None = None,
    ) -> Tensor:
        batch, n_classes = logits.shape
        if batch == 0:
            return logits.new_zeros(())
        T = self.temperature

        term1 = F.cross_entropy(
            logits / T, targets, weight=class_weights, reduction="mean",
        ) * T

        term2_parts = []
        n_active = 0
        for c in range(n_classes):
            pos_mask = (targets == c)
            n_pos = int(pos_mask.sum().item())
            if n_pos == 0 or n_pos == batch:
                continue
            n_active += 1
            neg_mask = ~pos_mask
            pos_c = logits[pos_mask, c]
            neg_c = logits[neg_mask, c]
            neg_scaled = neg_c / T
            neg_lse_pos = torch.logsumexp(-pos_c / T, dim=0)
            per_neg = F.softplus(neg_scaled + neg_lse_pos)
            loss_c = per_neg.mean()
            w = float(class_weights[c]) if class_weights is not None else 1.0
            term2_parts.append(loss_c * w)

        if term2_parts:
            term2 = torch.stack(term2_parts).sum() / max(n_active, 1)
        else:
            term2 = logits.new_zeros(())

        return 0.5 * (term1 + term2)


class TaskTokenAttention(nn.Module):
    """Multi-head self-attention across task tokens with zero-initialized gate."""

    def __init__(self, feature_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        if feature_dim % n_heads != 0:
            raise ValueError("feature_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        self.qkv = nn.Linear(feature_dim, 3 * feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.gate_raw = nn.Parameter(torch.zeros(1))

    def forward(self, task_features: Tensor) -> Tensor:
        batch, n_tasks, dim = task_features.shape
        residual = task_features
        normalized = self.norm(task_features)
        qkv = self.qkv(normalized)
        qkv = qkv.reshape(batch, n_tasks, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = math.sqrt(float(self.head_dim))
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, n_tasks, dim)
        attn_residual = self.out_proj(out)
        gate = torch.tanh(self.gate_raw)
        return residual + gate * attn_residual


class MulticlassCriterionModule(nn.Module):
    """Raw-state multiclass criterion heads with task-token attention."""

    def __init__(
        self,
        *,
        token_dim: int,
        criterion_class_counts: Mapping[str, int],
        criterion_positive_indices: Mapping[str, Sequence[int]],
        criterion_names: Sequence[str],
        task_feature_dim: int = 128,
        n_attention_heads: int = 4,
        max_logit_shift: float = 2.0,
    ) -> None:
        super().__init__()
        if len(criterion_names) != N_CRITERIA:
            raise ValueError("must have exactly 7 criteria")

        self.criterion_names = list(criterion_names)
        self.criterion_class_counts = dict(criterion_class_counts)
        self.task_feature_dim = int(task_feature_dim)
        self.max_logit_shift = float(max_logit_shift)

        # Store positive indices as buffers for each criterion
        for name in criterion_names:
            indices = list(criterion_positive_indices[name])
            self.register_buffer(
                f"_pos_idx_{name}",
                torch.tensor(indices, dtype=torch.long),
                persistent=True,
            )

        # 8 query tokens: 7 criterion + 1 diagnosis
        self.query_tokens = nn.Parameter(
            torch.randn(N_QUERIES, task_feature_dim) * 0.02
        )
        self.token_proj = nn.Linear(token_dim, task_feature_dim)
        self.query_norm = nn.LayerNorm(task_feature_dim)
        self.token_norm = nn.LayerNorm(task_feature_dim)

        # Cross-attention: queries attend to all token sources
        self.cross_attn = nn.MultiheadAttention(
            task_feature_dim, n_attention_heads, batch_first=True,
        )

        # Task-token self-attention with zero gate
        self.task_attention = TaskTokenAttention(
            task_feature_dim, n_heads=n_attention_heads,
        )

        # Per-criterion multiclass classifiers
        self.classifiers = nn.ModuleDict()
        for name in criterion_names:
            n_classes = criterion_class_counts[name]
            self.classifiers[name] = nn.Sequential(
                nn.Linear(task_feature_dim, task_feature_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(task_feature_dim, n_classes),
            )

        self.residual_gate_raw = nn.Parameter(torch.zeros(N_CRITERIA))

        # Checklist-derived melanoma
        self.checklist_raw_weights = nn.Parameter(torch.ones(N_CRITERIA))
        self.checklist_threshold = nn.Parameter(torch.tensor(3.0))

    def _derive_binary_logit(self, raw_logits: Tensor, criterion_name: str) -> Tensor:
        """Deterministic: sum positive-state softmax probs → clamped logit."""
        probs = torch.softmax(raw_logits, dim=-1)
        pos_idx = getattr(self, f"_pos_idx_{criterion_name}")
        positive_prob = probs[:, pos_idx].sum(dim=-1)
        clamped = positive_prob.clamp(1e-6, 1.0 - 1e-6)
        return _stable_logit(clamped)

    def checklist_melanoma(self, criterion_logits: Tensor) -> Tensor:
        probs = torch.sigmoid(criterion_logits)
        weights = F.softplus(self.checklist_raw_weights)
        return (probs * weights.unsqueeze(0)).sum(dim=-1) - self.checklist_threshold

    def residual_gates(self) -> Tensor:
        """Return detached gates for receipts; forward uses the live parameter."""
        return torch.tanh(self.residual_gate_raw).detach()

    def diagnosis_only_feature(
        self,
        tokens: Tensor,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """Attend the diagnosis query without activating criterion task exchange."""
        batch = tokens.shape[0]
        projected = self.token_norm(self.token_proj(tokens))
        query = self.query_norm(
            self.query_tokens[N_CRITERIA:].unsqueeze(0).expand(batch, -1, -1)
        )
        key_padding_mask = None if token_mask is None else ~token_mask
        attended, _ = self.cross_attn(
            query,
            projected,
            projected,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return (query + attended).squeeze(1)

    def forward(
        self,
        all_tokens: Tensor,
        token_mask: Tensor | None,
        base_criterion_logits: Tensor,
    ) -> MulticlassCriterionOutput:
        """Process criterion queries over all token sources."""
        batch = all_tokens.shape[0]
        M = self.max_logit_shift

        # Project tokens to task feature space
        projected = self.token_norm(self.token_proj(all_tokens))

        # Expand query tokens for batch
        queries = self.query_norm(
            self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        )

        # Cross-attention: queries attend to all projected tokens
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask

        attended, attn_weights = self.cross_attn(
            queries, projected, projected,
            key_padding_mask=key_padding_mask,
        )
        task_features = queries + attended

        # Task-token self-attention (zero-gated)
        refined = self.task_attention(task_features)

        # Split: 7 criterion features + 1 diagnosis feature
        criterion_features = refined[:, :N_CRITERIA]
        diagnosis_features = refined[:, N_CRITERIA:]

        # Per-criterion multiclass classification
        raw_logits = {}
        derived_binary = torch.zeros(batch, N_CRITERIA, device=all_tokens.device)

        for c_idx, name in enumerate(self.criterion_names):
            mc_logits = self.classifiers[name](criterion_features[:, c_idx])
            raw_logits[name] = mc_logits
            derived_binary[:, c_idx] = self._derive_binary_logit(mc_logits, name)

        # Bounded residual: base + M * tanh(derived - base) * tanh(gate)
        residual_raw = derived_binary - base_criterion_logits
        gates = torch.tanh(self.residual_gate_raw)
        bounded_residual = M * torch.tanh(residual_raw / M) * gates.unsqueeze(0)
        final_criteria = base_criterion_logits + bounded_residual

        # Checklist melanoma
        checklist_mel = self.checklist_melanoma(final_criteria)
        checklist_w = F.softplus(self.checklist_raw_weights).detach()

        return MulticlassCriterionOutput(
            raw_logits=raw_logits,
            derived_binary_logits=derived_binary,
            criterion_residual=bounded_residual,
            final_criterion_logits=final_criteria,
            diagnosis_features=diagnosis_features.squeeze(1),
            task_token_features=refined,
            attention_weights=attn_weights,
            residual_gates=gates.detach(),
            checklist_melanoma_logit=checklist_mel,
            checklist_weights=checklist_w,
        )


__all__ = [
    "MulticlassCriterionModule",
    "MulticlassCriterionOutput",
    "TaskTokenAttention",
    "TwoWayLoss",
    "_stable_logit",
]
