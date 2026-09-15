"""Winner-agnostic paired-photo criterion wrapper for Phase 2.4B."""
from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from dermpact.perception.clinical_criterion_fusion import (
    CrossModalClinicalCriterionFusion,
)
from dermpact.perception.clinical_tail_adapter import ClinicalTailAdapter


_TOKEN_COUNT = 197
_TOKEN_DIM = 768
_CRITERIA = 7


class _FrozenDermTail(nn.Module):
    """Private frozen deep copy of the same retained blocks used clinically."""

    def __init__(self, retained_visual_tail: nn.Module) -> None:
        super().__init__()
        if not isinstance(retained_visual_tail, nn.Module):
            raise TypeError("retained_visual_tail must be a torch.nn.Module")
        if not hasattr(retained_visual_tail, "blocks") or not hasattr(
            retained_visual_tail, "norm"
        ):
            raise TypeError("retained_visual_tail must expose blocks and norm")
        if len(retained_visual_tail.blocks) != 2:
            raise ValueError("retained_visual_tail must contain exactly two blocks")
        self.blocks = nn.ModuleList(copy.deepcopy(list(retained_visual_tail.blocks)))
        self.norm = copy.deepcopy(retained_visual_tail.norm)
        self.shared_rel_pos_bias = copy.deepcopy(
            getattr(retained_visual_tail, "shared_rel_pos_bias", None)
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, prefix_tokens: Tensor) -> Tensor:
        tokens = prefix_tokens
        bias = (
            None
            if self.shared_rel_pos_bias is None
            else self.shared_rel_pos_bias()
        )
        if bias is not None and not isinstance(bias, Tensor):
            raise TypeError("shared_rel_pos_bias must return a tensor")
        for block in self.blocks:
            tokens = block(tokens, rel_pos_bias=bias)
            if not isinstance(tokens, Tensor) or tokens.shape != prefix_tokens.shape:
                raise ValueError("retained visual blocks must preserve token shape")
            if not torch.isfinite(tokens).all():
                raise ValueError("retained visual block output must be finite")
        tokens = self.norm(tokens)
        if not isinstance(tokens, Tensor) or tokens.shape != prefix_tokens.shape:
            raise ValueError("retained visual tail norm must preserve token shape")
        if not torch.isfinite(tokens).all():
            raise ValueError("retained visual tail norm output must be finite")
        return tokens[:, 1:, :]

    def train(self, mode: bool = True):
        super().train(False)
        return self


class Phase24BPairedCriterionModel(nn.Module):
    """Add bounded clinical-photo evidence to any eligible frozen baseline."""

    def __init__(
        self,
        baseline_model: nn.Module,
        retained_visual_tail: nn.Module,
        *,
        prototype_embeddings: Tensor,
        attention_dim: int,
        bottleneck_dim: int = 32,
        initial_temperature: float = 10.0,
        maximum_temperature: float = 100.0,
        maximum_logit_shift: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(baseline_model, nn.Module):
            raise TypeError("baseline_model must be a torch.nn.Module")

        self.baseline_model = baseline_model
        for parameter in self.baseline_model.parameters():
            parameter.requires_grad_(False)
        self.baseline_model.eval()

        self.clinical_tail = ClinicalTailAdapter(
            retained_visual_tail,
            bottleneck_dim=bottleneck_dim,
        )
        self.derm_tail = _FrozenDermTail(retained_visual_tail)
        self.cross_modal_fusion = CrossModalClinicalCriterionFusion(
            derm_patch_dim=_TOKEN_DIM,
            clinical_patch_dim=_TOKEN_DIM,
            prototype_embeddings=prototype_embeddings,
            attention_dim=attention_dim,
            initial_temperature=initial_temperature,
            maximum_temperature=maximum_temperature,
            maximum_logit_shift=maximum_logit_shift,
        )

    @property
    def clinical_fusion(self) -> CrossModalClinicalCriterionFusion:
        """Compatibility alias; production state is named cross_modal_fusion."""
        return self.cross_modal_fusion

    @staticmethod
    def _validate_inputs(
        derm_prefix: Tensor,
        clinical_prefix: Tensor,
        clinical_present_mask: Tensor,
    ) -> int:
        for name, value in (
            ("derm_prefix", derm_prefix),
            ("clinical_prefix", clinical_prefix),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 3 or value.shape[1:] != (_TOKEN_COUNT, _TOKEN_DIM):
                raise ValueError(f"{name} must have shape [batch, 197, 768]")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating dtype")

        if clinical_prefix.shape != derm_prefix.shape:
            raise ValueError("derm_prefix and clinical_prefix must be batch-aligned")
        if clinical_prefix.device != derm_prefix.device:
            raise ValueError("derm_prefix and clinical_prefix must share a device")
        if clinical_prefix.dtype != derm_prefix.dtype:
            raise TypeError("derm_prefix and clinical_prefix must share a dtype")
        if not torch.isfinite(derm_prefix).all():
            raise ValueError("derm_prefix must be finite")

        batch = derm_prefix.shape[0]
        if not isinstance(clinical_present_mask, Tensor):
            raise TypeError("clinical_present_mask must be a torch.Tensor")
        if clinical_present_mask.shape != (batch,):
            raise ValueError("clinical_present_mask must have shape [batch]")
        if clinical_present_mask.dtype is not torch.bool:
            raise TypeError("clinical_present_mask must have dtype torch.bool")
        if clinical_present_mask.device != derm_prefix.device:
            raise ValueError("clinical_present_mask must share the prefix device")
        return batch

    @staticmethod
    def _required_baseline_outputs(
        outputs: Any,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, Any], Tensor, Tensor, Tensor]:
        if not isinstance(outputs, Mapping):
            raise TypeError("baseline_model must return a mapping")
        missing = {"diagnosis", "melanoma", "criteria"}.difference(outputs)
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"baseline_model output is missing: {names}")

        diagnosis = outputs["diagnosis"]
        melanoma = outputs["melanoma"]
        criteria = outputs["criteria"]
        if not all(isinstance(value, Tensor) for value in (diagnosis, melanoma, criteria)):
            raise TypeError("baseline diagnosis, melanoma, and criteria must be tensors")
        if diagnosis.ndim != 2 or diagnosis.shape[0] != batch or diagnosis.shape[1] < 2:
            raise ValueError("baseline diagnosis must have shape [batch, classes>=2]")
        if melanoma.shape != (batch,):
            raise ValueError("baseline melanoma must have shape [batch]")
        if criteria.shape != (batch, _CRITERIA):
            raise ValueError("baseline criteria must have shape [batch, 7]")
        for name, value in (
            ("diagnosis", diagnosis),
            ("melanoma", melanoma),
            ("criteria", criteria),
        ):
            if not value.is_floating_point():
                raise TypeError(f"baseline {name} must have a floating dtype")
            if value.device != device:
                raise ValueError(f"baseline {name} must share the prefix device")
            if not torch.isfinite(value).all():
                raise ValueError(f"baseline {name} must be finite")
        if criteria.dtype != dtype:
            raise TypeError("baseline criteria must share the prefix dtype")
        return dict(outputs), diagnosis, melanoma, criteria

    def forward(
        self,
        derm_prefix: Tensor,
        clinical_prefix: Tensor,
        clinical_present_mask: Tensor,
    ) -> Mapping[str, Any]:
        batch = self._validate_inputs(
            derm_prefix,
            clinical_prefix,
            clinical_present_mask,
        )

        with torch.no_grad():
            raw_baseline = self.baseline_model(derm_prefix)
        baseline, diagnosis, melanoma, base_criteria = self._required_baseline_outputs(
            raw_baseline,
            batch=batch,
            device=derm_prefix.device,
            dtype=derm_prefix.dtype,
        )

        clinical_tail = self.clinical_tail(
            clinical_prefix,
            clinical_present_mask,
        )
        with torch.no_grad():
            derm_patch_tokens = self.derm_tail(derm_prefix)
        fusion = self.cross_modal_fusion(
            derm_patch_tokens,
            clinical_tail.final_patch_tokens,
            base_criteria,
            clinical_present_mask,
        )

        result: dict[str, Any] = dict(baseline)
        result.update(
            {
                "baseline_outputs": baseline,
                "diagnosis": diagnosis,
                "melanoma": melanoma,
                "base_criteria": fusion.base_logits,
                "criteria": fusion.final_logits,
                "final_criteria": fusion.final_logits,
                "clinical_residual": fusion.clinical_residual_logits,
                "clinical_gates": fusion.criterion_gates,
                "clinical_attention": fusion.semantic_attention_maps,
                "cross_modal_attention": fusion.cross_attention_maps,
                "clinical_temperatures": fusion.temperatures,
                "clinical_semantic_contributions": fusion.semantic_contributions,
                "cross_modal_contributions": fusion.cross_view_contributions,
                "cross_modal_mixture_weights": fusion.mixture_weights,
                "clinical_effective_scales": fusion.effective_output_scales,
                "clinical_maximum_logit_shift": fusion.maximum_logit_shift,
                "clinical_adapter_contributions": clinical_tail.adapter_contributions,
            }
        )
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline_model.eval()
        self.clinical_tail.frozen_tail.eval()
        self.derm_tail.eval()
        return self


__all__ = ["Phase24BPairedCriterionModel"]
