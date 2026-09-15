"""Criterion-only clinical-image fusion for Phase 2.4B."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


_N_CRITERIA = 7
_N_POLARITIES = 2
_N_CLINICAL_PATCHES = 196


@dataclass(frozen=True)
class ClinicalCriterionFusionOutput:
    """Observable criterion-level fusion result."""

    base_logits: Tensor
    final_logits: Tensor
    clinical_residual_logits: Tensor
    criterion_gates: Tensor
    attention_maps: Tensor
    temperatures: Tensor
    semantic_contributions: Tensor
    effective_output_scales: Tensor
    maximum_logit_shift: Tensor


@dataclass(frozen=True)
class CrossModalClinicalCriterionFusionOutput:
    """Auditable localized dermoscopy--clinical criterion fusion result."""

    base_logits: Tensor
    final_logits: Tensor
    clinical_residual_logits: Tensor
    criterion_gates: Tensor
    semantic_attention_maps: Tensor
    cross_attention_maps: Tensor
    semantic_contributions: Tensor
    cross_view_contributions: Tensor
    mixture_weights: Tensor
    temperatures: Tensor
    effective_output_scales: Tensor
    maximum_logit_shift: Tensor


class ClinicalCriterionFusion(nn.Module):
    """Fuse final clinical patch tokens into seven criterion logits."""

    def __init__(
        self,
        *,
        clinical_patch_dim: int,
        prototype_embeddings: Tensor,
        attention_dim: int,
        initial_temperature: float = 10.0,
        maximum_temperature: float = 100.0,
        maximum_logit_shift: float = 1.0,
    ) -> None:
        super().__init__()
        if clinical_patch_dim <= 0 or attention_dim <= 0:
            raise ValueError("clinical_patch_dim and attention_dim must be positive")
        if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive and finite")
        if (
            not math.isfinite(maximum_temperature)
            or maximum_temperature < initial_temperature
        ):
            raise ValueError(
                "maximum_temperature must be finite and at least initial_temperature"
            )
        if not math.isfinite(maximum_logit_shift) or maximum_logit_shift <= 0.0:
            raise ValueError("maximum_logit_shift must be positive and finite")
        prototypes = torch.as_tensor(prototype_embeddings).detach().clone().float()
        if prototypes.ndim != 3 or prototypes.shape[:2] != (
            _N_CRITERIA,
            _N_POLARITIES,
        ):
            raise ValueError("prototype_embeddings must have shape [7, 2, prototype_dim]")
        if prototypes.shape[2] <= 0 or not torch.isfinite(prototypes).all():
            raise ValueError("prototype_embeddings must be non-empty and finite")

        self.clinical_patch_dim = int(clinical_patch_dim)
        self.attention_dim = int(attention_dim)
        self.maximum_temperature = float(maximum_temperature)
        self.register_buffer(
            "_maximum_logit_shift",
            torch.tensor(float(maximum_logit_shift), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("prototype_embeddings", prototypes, persistent=True)
        self.prototype_projection = nn.Linear(
            int(prototypes.shape[2]), self.attention_dim, bias=False
        )
        self.clinical_projection = nn.Linear(
            self.clinical_patch_dim, self.attention_dim, bias=False
        )
        self.log_temperature = nn.Parameter(
            torch.full((_N_CRITERIA,), math.log(float(initial_temperature)))
        )
        self.raw_output_scale = nn.Parameter(torch.zeros(_N_CRITERIA))
        # Seven independent gates, minimally conditioned on baseline uncertainty.
        self.gate_bias = nn.Parameter(torch.zeros(_N_CRITERIA))
        self.gate_uncertainty_weight = nn.Parameter(torch.zeros(_N_CRITERIA))

    def temperatures(self) -> Tensor:
        """Return positive, bounded per-criterion inverse temperatures."""
        return self.log_temperature.exp().clamp(max=self.maximum_temperature)

    def effective_output_scales(self) -> Tensor:
        """Return signed scales that guarantee the configured residual bound."""
        return (
            0.5
            * self._maximum_logit_shift
            * torch.tanh(self.raw_output_scale)
        )

    @staticmethod
    def _uncertainty(
        base_logits: Tensor,
        base_probabilities: Tensor | None,
        base_uncertainty: Tensor | None,
    ) -> Tensor:
        if base_probabilities is not None and base_uncertainty is not None:
            raise ValueError(
                "provide at most one of base_probabilities and base_uncertainty"
            )
        if base_uncertainty is not None:
            if base_uncertainty.shape != base_logits.shape:
                raise ValueError("base_uncertainty must have shape [batch, 7]")
            if not torch.isfinite(base_uncertainty).all():
                raise ValueError("base_uncertainty must be finite")
            if torch.any((base_uncertainty < 0.0) | (base_uncertainty > 1.0)):
                raise ValueError("base_uncertainty must lie in [0, 1]")
            return base_uncertainty

        probabilities = base_logits.sigmoid() if base_probabilities is None else base_probabilities
        if probabilities.shape != base_logits.shape:
            raise ValueError("base_probabilities must have shape [batch, 7]")
        if not torch.isfinite(probabilities).all():
            raise ValueError("base_probabilities must be finite")
        if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("base_probabilities must lie in [0, 1]")
        # Maximal at p=.5 and zero at p in {0, 1}; no labels are consulted.
        return 1.0 - (2.0 * probabilities - 1.0).abs()

    def forward(
        self,
        clinical_patch_tokens: Tensor,
        base_criterion_logits: Tensor,
        clinical_present_mask: Tensor,
        *,
        base_probabilities: Tensor | None = None,
        base_uncertainty: Tensor | None = None,
    ) -> ClinicalCriterionFusionOutput:
        batch = clinical_patch_tokens.shape[0] if clinical_patch_tokens.ndim > 0 else 0
        expected_tokens = (batch, _N_CLINICAL_PATCHES, self.clinical_patch_dim)
        if clinical_patch_tokens.shape != expected_tokens:
            raise ValueError(
                "clinical_patch_tokens must have shape "
                f"[batch, {_N_CLINICAL_PATCHES}, {self.clinical_patch_dim}]"
            )
        if base_criterion_logits.shape != (batch, _N_CRITERIA):
            raise ValueError("base_criterion_logits must have shape [batch, 7]")
        if clinical_present_mask.shape != (batch,):
            raise ValueError("clinical_present_mask must have shape [batch]")
        if clinical_present_mask.dtype is not torch.bool:
            raise TypeError("clinical_present_mask must have dtype torch.bool")
        if clinical_present_mask.device != clinical_patch_tokens.device:
            raise ValueError(
                "clinical_present_mask must share the clinical token device"
            )
        if base_criterion_logits.device != clinical_patch_tokens.device:
            raise ValueError(
                "base_criterion_logits must share the clinical token device"
            )
        if base_criterion_logits.dtype != clinical_patch_tokens.dtype:
            raise TypeError(
                "base_criterion_logits must share the clinical token dtype"
            )
        if not clinical_patch_tokens.is_floating_point():
            raise TypeError("clinical_patch_tokens must have a floating dtype")
        if not torch.isfinite(clinical_patch_tokens).all():
            raise ValueError("clinical_patch_tokens must be finite")
        if not torch.isfinite(base_criterion_logits).all():
            raise ValueError("base_criterion_logits must be finite")

        uncertainty = self._uncertainty(
            base_criterion_logits, base_probabilities, base_uncertainty
        )
        patch_mask = clinical_present_mask.view(batch, 1, 1)
        masked_tokens = torch.where(
            patch_mask,
            clinical_patch_tokens,
            torch.zeros_like(clinical_patch_tokens),
        )
        queries = F.normalize(
            self.prototype_projection(self.prototype_embeddings), dim=-1
        )
        keys = F.normalize(self.clinical_projection(masked_tokens), dim=-1)
        cosine = torch.einsum("cpd,bnd->bcpn", queries, keys).clamp(-1.0, 1.0)
        unmasked_attention = torch.softmax(
            cosine * self.temperatures().view(1, _N_CRITERIA, 1, 1),
            dim=-1,
        )
        unmasked_contributions = (unmasked_attention * cosine).sum(dim=-1)
        semantic_contrast = (
            unmasked_contributions[:, :, 0]
            - unmasked_contributions[:, :, 1]
        )
        unmasked_gates = torch.sigmoid(
            self.gate_bias.unsqueeze(0)
            + self.gate_uncertainty_weight.unsqueeze(0) * uncertainty
        )
        unmasked_residual = (
            semantic_contrast
            * self.effective_output_scales().unsqueeze(0)
            * unmasked_gates
        )

        criterion_mask = clinical_present_mask.view(batch, 1)
        residual = torch.where(
            criterion_mask,
            unmasked_residual,
            torch.zeros_like(unmasked_residual),
        )
        gates = torch.where(
            criterion_mask,
            unmasked_gates,
            torch.zeros_like(unmasked_gates),
        )
        attention_mask = clinical_present_mask.view(batch, 1, 1, 1)
        attention = torch.where(
            attention_mask,
            unmasked_attention,
            torch.zeros_like(unmasked_attention),
        )
        contribution_mask = clinical_present_mask.view(batch, 1, 1)
        contributions = torch.where(
            contribution_mask,
            unmasked_contributions,
            torch.zeros_like(unmasked_contributions),
        )

        final_logits = torch.where(
            clinical_present_mask.unsqueeze(1),
            base_criterion_logits + residual,
            base_criterion_logits,
        )
        return ClinicalCriterionFusionOutput(
            base_logits=base_criterion_logits,
            final_logits=final_logits,
            clinical_residual_logits=residual,
            criterion_gates=gates,
            attention_maps=attention,
            temperatures=self.temperatures(),
            semantic_contributions=contributions,
            effective_output_scales=self.effective_output_scales(),
            maximum_logit_shift=self._maximum_logit_shift,
        )


class CrossModalClinicalCriterionFusion(nn.Module):
    """Localized symmetric cross-view fusion for seven criterion logits."""

    def __init__(
        self,
        *,
        derm_patch_dim: int,
        clinical_patch_dim: int,
        prototype_embeddings: Tensor,
        attention_dim: int = 64,
        initial_temperature: float = 10.0,
        maximum_temperature: float = 100.0,
        maximum_logit_shift: float = 1.0,
    ) -> None:
        super().__init__()
        if derm_patch_dim <= 0 or clinical_patch_dim <= 0 or attention_dim <= 0:
            raise ValueError("patch dimensions and attention_dim must be positive")
        if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive and finite")
        if (
            not math.isfinite(maximum_temperature)
            or maximum_temperature < initial_temperature
        ):
            raise ValueError(
                "maximum_temperature must be finite and at least initial_temperature"
            )
        if not math.isfinite(maximum_logit_shift) or maximum_logit_shift <= 0.0:
            raise ValueError("maximum_logit_shift must be positive and finite")
        prototypes = torch.as_tensor(prototype_embeddings).detach().clone().float()
        if prototypes.ndim != 3 or prototypes.shape[:2] != (
            _N_CRITERIA,
            _N_POLARITIES,
        ):
            raise ValueError("prototype_embeddings must have shape [7, 2, prototype_dim]")
        if prototypes.shape[2] <= 0 or not torch.isfinite(prototypes).all():
            raise ValueError("prototype_embeddings must be non-empty and finite")

        self.derm_patch_dim = int(derm_patch_dim)
        self.clinical_patch_dim = int(clinical_patch_dim)
        self.attention_dim = int(attention_dim)
        self.maximum_temperature = float(maximum_temperature)
        self.register_buffer("prototype_embeddings", prototypes, persistent=True)
        self.register_buffer(
            "_maximum_logit_shift",
            torch.tensor(float(maximum_logit_shift), dtype=torch.float32),
            persistent=True,
        )
        self.prototype_projection = nn.Linear(
            int(prototypes.shape[2]), self.attention_dim, bias=False
        )
        self.derm_projection = nn.Linear(
            self.derm_patch_dim, self.attention_dim, bias=False
        )
        self.clinical_projection = nn.Linear(
            self.clinical_patch_dim, self.attention_dim, bias=False
        )
        self.log_temperature = nn.Parameter(
            torch.full((_N_CRITERIA,), math.log(float(initial_temperature)))
        )
        self.raw_mixture_weight = nn.Parameter(torch.zeros(_N_CRITERIA))
        self.raw_output_scale = nn.Parameter(torch.zeros(_N_CRITERIA))
        self.gate_bias = nn.Parameter(torch.zeros(_N_CRITERIA))
        self.gate_uncertainty_weight = nn.Parameter(torch.zeros(_N_CRITERIA))

    def temperatures(self) -> Tensor:
        return self.log_temperature.exp().clamp(max=self.maximum_temperature)

    def mixture_weights(self) -> Tensor:
        """Clinical-semantic weight; the complementary weight is cross-view."""
        return torch.sigmoid(self.raw_mixture_weight)

    def effective_output_scales(self) -> Tensor:
        # Both convex-mixture inputs are polarity contrasts in [-2, 2].
        return 0.5 * self._maximum_logit_shift * torch.tanh(self.raw_output_scale)

    @staticmethod
    def _uncertainty(
        base_logits: Tensor,
        base_probabilities: Tensor | None,
        base_uncertainty: Tensor | None,
    ) -> Tensor:
        return ClinicalCriterionFusion._uncertainty(
            base_logits, base_probabilities, base_uncertainty
        )

    def _validate(
        self,
        derm_patch_tokens: Tensor,
        clinical_patch_tokens: Tensor,
        base_criterion_logits: Tensor,
        clinical_present_mask: Tensor,
    ) -> int:
        batch = derm_patch_tokens.shape[0] if derm_patch_tokens.ndim > 0 else 0
        expected_derm = (batch, _N_CLINICAL_PATCHES, self.derm_patch_dim)
        expected_clinical = (
            batch,
            _N_CLINICAL_PATCHES,
            self.clinical_patch_dim,
        )
        if derm_patch_tokens.shape != expected_derm:
            raise ValueError(
                "derm_patch_tokens must have shape "
                f"[batch, {_N_CLINICAL_PATCHES}, {self.derm_patch_dim}]"
            )
        if clinical_patch_tokens.shape != expected_clinical:
            raise ValueError(
                "clinical_patch_tokens must have shape "
                f"[batch, {_N_CLINICAL_PATCHES}, {self.clinical_patch_dim}]"
            )
        if base_criterion_logits.shape != (batch, _N_CRITERIA):
            raise ValueError("base_criterion_logits must have shape [batch, 7]")
        if clinical_present_mask.shape != (batch,):
            raise ValueError("clinical_present_mask must have shape [batch]")
        if clinical_present_mask.dtype is not torch.bool:
            raise TypeError("clinical_present_mask must have dtype torch.bool")
        tensors = (clinical_patch_tokens, base_criterion_logits, clinical_present_mask)
        if any(value.device != derm_patch_tokens.device for value in tensors):
            raise ValueError("all inputs must share the derm token device")
        if clinical_patch_tokens.dtype != derm_patch_tokens.dtype:
            raise TypeError("derm and clinical tokens must share a dtype")
        if base_criterion_logits.dtype != derm_patch_tokens.dtype:
            raise TypeError("base_criterion_logits must share the token dtype")
        if not derm_patch_tokens.is_floating_point():
            raise TypeError("patch tokens must have a floating dtype")
        for name, value in (
            ("derm_patch_tokens", derm_patch_tokens),
            ("clinical_patch_tokens", clinical_patch_tokens),
            ("base_criterion_logits", base_criterion_logits),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        return batch

    def forward(
        self,
        derm_patch_tokens: Tensor,
        clinical_patch_tokens: Tensor,
        base_criterion_logits: Tensor,
        clinical_present_mask: Tensor,
        *,
        base_probabilities: Tensor | None = None,
        base_uncertainty: Tensor | None = None,
    ) -> CrossModalClinicalCriterionFusionOutput:
        batch = self._validate(
            derm_patch_tokens,
            clinical_patch_tokens,
            base_criterion_logits,
            clinical_present_mask,
        )
        uncertainty = self._uncertainty(
            base_criterion_logits, base_probabilities, base_uncertainty
        )
        token_mask = clinical_present_mask.view(batch, 1, 1)
        masked_clinical = torch.where(
            token_mask,
            clinical_patch_tokens,
            torch.zeros_like(clinical_patch_tokens),
        )
        prototypes = F.normalize(
            self.prototype_projection(self.prototype_embeddings), dim=-1
        )
        derm_keys = F.normalize(self.derm_projection(derm_patch_tokens), dim=-1)
        clinical_keys = F.normalize(
            self.clinical_projection(masked_clinical), dim=-1
        )
        temperature = self.temperatures().view(1, _N_CRITERIA, 1, 1)

        derm_cosine = torch.einsum(
            "cpd,bnd->bcpn", prototypes, derm_keys
        ).clamp(-1.0, 1.0)
        clinical_cosine = torch.einsum(
            "cpd,bnd->bcpn", prototypes, clinical_keys
        ).clamp(-1.0, 1.0)
        derm_attention = torch.softmax(derm_cosine * temperature, dim=-1)
        clinical_attention = torch.softmax(clinical_cosine * temperature, dim=-1)
        derm_summary = F.normalize(
            torch.einsum("bcpn,bnd->bcpd", derm_attention, derm_keys), dim=-1
        )
        clinical_summary = F.normalize(
            torch.einsum("bcpn,bnd->bcpd", clinical_attention, clinical_keys),
            dim=-1,
        )
        clinical_semantic = (clinical_attention * clinical_cosine).sum(dim=-1)

        derm_to_clinical_cosine = torch.einsum(
            "bcpd,bnd->bcpn", derm_summary, clinical_keys
        ).clamp(-1.0, 1.0)
        clinical_to_derm_cosine = torch.einsum(
            "bcpd,bnd->bcpn", clinical_summary, derm_keys
        ).clamp(-1.0, 1.0)
        derm_to_clinical_attention = torch.softmax(
            derm_to_clinical_cosine * temperature, dim=-1
        )
        clinical_to_derm_attention = torch.softmax(
            clinical_to_derm_cosine * temperature, dim=-1
        )
        cross_contributions = 0.5 * (
            (derm_to_clinical_attention * derm_to_clinical_cosine).sum(dim=-1)
            + (clinical_to_derm_attention * clinical_to_derm_cosine).sum(dim=-1)
        )

        semantic_contrast = clinical_semantic[:, :, 0] - clinical_semantic[:, :, 1]
        cross_contrast = (
            cross_contributions[:, :, 0] - cross_contributions[:, :, 1]
        )
        mixture = self.mixture_weights().unsqueeze(0)
        mixed_contrast = mixture * semantic_contrast + (1.0 - mixture) * cross_contrast
        unmasked_gates = torch.sigmoid(
            self.gate_bias.unsqueeze(0)
            + self.gate_uncertainty_weight.unsqueeze(0) * uncertainty
        )
        unmasked_residual = (
            mixed_contrast
            * self.effective_output_scales().unsqueeze(0)
            * unmasked_gates
        ).clamp(
            min=-self._maximum_logit_shift,
            max=self._maximum_logit_shift,
        )

        criterion_mask = clinical_present_mask.view(batch, 1)
        polarity_mask = clinical_present_mask.view(batch, 1, 1)
        attention_mask = clinical_present_mask.view(batch, 1, 1, 1)
        residual = torch.where(
            criterion_mask, unmasked_residual, torch.zeros_like(unmasked_residual)
        )
        gates = torch.where(
            criterion_mask, unmasked_gates, torch.zeros_like(unmasked_gates)
        )
        semantic_attention = torch.stack(
            (
                torch.where(
                    attention_mask,
                    derm_attention,
                    torch.zeros_like(derm_attention),
                ),
                torch.where(
                    attention_mask,
                    clinical_attention,
                    torch.zeros_like(clinical_attention),
                ),
            ),
            dim=1,
        )
        cross_attention = torch.stack(
            (
                torch.where(
                    attention_mask,
                    derm_to_clinical_attention,
                    torch.zeros_like(derm_to_clinical_attention),
                ),
                torch.where(
                    attention_mask,
                    clinical_to_derm_attention,
                    torch.zeros_like(clinical_to_derm_attention),
                ),
            ),
            dim=1,
        )
        semantic_contributions = torch.where(
            polarity_mask,
            clinical_semantic,
            torch.zeros_like(clinical_semantic),
        )
        audited_cross = torch.where(
            polarity_mask,
            cross_contributions,
            torch.zeros_like(cross_contributions),
        )
        final_logits = torch.where(
            criterion_mask,
            base_criterion_logits + residual,
            base_criterion_logits,
        )
        return CrossModalClinicalCriterionFusionOutput(
            base_logits=base_criterion_logits,
            final_logits=final_logits,
            clinical_residual_logits=residual,
            criterion_gates=gates,
            semantic_attention_maps=semantic_attention,
            cross_attention_maps=cross_attention,
            semantic_contributions=semantic_contributions,
            cross_view_contributions=audited_cross,
            mixture_weights=self.mixture_weights(),
            temperatures=self.temperatures(),
            effective_output_scales=self.effective_output_scales(),
            maximum_logit_shift=self._maximum_logit_shift,
        )


__all__ = [
    "ClinicalCriterionFusion",
    "ClinicalCriterionFusionOutput",
    "CrossModalClinicalCriterionFusion",
    "CrossModalClinicalCriterionFusionOutput",
]
