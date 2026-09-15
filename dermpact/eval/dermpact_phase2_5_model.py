"""Phase 2.5 unified perception model."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dermpact.perception.multiclass_criterion_module import (
    MulticlassCriterionModule,
)
from dermpact.perception.unified_perception_tail import (
    UnifiedPerceptionTail,
    PerceptionFeatures,
)

N_CRITERIA = 7
TOKEN_DIM = 768


@dataclass
class Phase25Output:
    diagnosis: Tensor
    melanoma: Tensor
    criteria: Tensor
    criterion_residual: Tensor
    raw_logits: dict[str, Tensor]
    checklist_melanoma: Tensor
    diagnosis_delta: Tensor
    melanoma_delta: Tensor
    residual_gates: Tensor
    frozen_diagnosis: Tensor
    frozen_melanoma: Tensor
    frozen_criteria: Tensor


class Phase25Model(nn.Module):
    """Frozen Phase 2.4B + trainable high-res/multiclass route."""

    def __init__(
        self,
        frozen_baseline,
        unified_tail: UnifiedPerceptionTail,
        multiclass_module: MulticlassCriterionModule,
        *,
        n_diagnosis_classes: int = 9,
        max_logit_shift: float = 2.0,
    ) -> None:
        super().__init__()
        self.frozen_baseline = frozen_baseline
        for p in self.frozen_baseline.parameters():
            p.requires_grad = False
        self.frozen_baseline.eval()

        self.unified_tail = unified_tail
        self.multiclass_module = multiclass_module
        self.max_logit_shift = float(max_logit_shift)

        feat_dim = multiclass_module.task_feature_dim

        self.diagnosis_delta_head = nn.Linear(feat_dim, n_diagnosis_classes)
        nn.init.zeros_(self.diagnosis_delta_head.weight)
        nn.init.zeros_(self.diagnosis_delta_head.bias)

        self.melanoma_delta_head = nn.Linear(feat_dim, 1)
        nn.init.zeros_(self.melanoma_delta_head.weight)
        nn.init.zeros_(self.melanoma_delta_head.bias)

        self.diagnosis_gate = nn.Parameter(torch.tensor(0.1))
        self.melanoma_gate = nn.Parameter(torch.tensor(0.1))

    def _bounded_residual(
        self, base: Tensor, delta: Tensor, gate: Tensor
    ) -> Tensor:
        M = self.max_logit_shift
        return base + M * torch.tanh(delta / M) * torch.tanh(gate)

    def forward(
        self,
        global_prefix: Tensor,
        window_prefixes: Tensor,
        clinical_prefix: Tensor,
        patch_valid_mask: Tensor,
        window_positions: Tensor,
        clinical_present: Tensor,
        frozen_outputs: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        batch = global_prefix.shape[0]

        if frozen_outputs is None:
            with torch.no_grad():
                frozen_out = self.frozen_baseline(
                    global_prefix, clinical_prefix, clinical_present
                )
                frozen_diag = frozen_out["diagnosis"].detach()
                frozen_mel = frozen_out["melanoma"].detach()
                frozen_crit = frozen_out["criteria"].detach()
        else:
            frozen_diag = frozen_outputs["diagnosis"].detach()
            frozen_mel = frozen_outputs["melanoma"].detach()
            frozen_crit = frozen_outputs["criteria"].detach()
            if frozen_diag.shape[0] != batch or frozen_mel.shape != (batch,):
                raise ValueError("frozen diagnosis/melanoma batch shape mismatch")
            if frozen_crit.shape != (batch, N_CRITERIA):
                raise ValueError("frozen criteria must have shape [batch, 7]")
            for value in (frozen_diag, frozen_mel, frozen_crit):
                if value.device != global_prefix.device:
                    raise ValueError("precomputed frozen outputs must share the trainable device")

        features = self.unified_tail(
            global_prefix, window_prefixes, clinical_prefix,
            patch_valid_mask, window_positions, clinical_present,
        )

        all_tokens = [features.global_patches]
        n_global = features.global_patches.shape[1]

        # Regional features from the region transformer (cross-window hierarchy)
        n_regional = 0
        if features.regional_features is not None:
            all_tokens.append(features.regional_features)
            n_regional = features.regional_features.shape[1]

        n_window_patches = 0
        if features.window_patches is not None:
            wp = features.window_patches
            all_tokens.append(wp.reshape(batch, -1, TOKEN_DIM))
            n_window_patches = wp.shape[1] * wp.shape[2]

        n_clinical = 0
        if features.clinical_patches is not None:
            all_tokens.append(features.clinical_patches)
            n_clinical = features.clinical_patches.shape[1]

        token_cat = torch.cat(all_tokens, dim=1)
        n_total = n_global + n_regional + n_window_patches + n_clinical

        # Token validity mask
        token_mask = torch.ones(batch, n_total, dtype=torch.bool, device=global_prefix.device)
        # Regional: valid where any window patch is valid
        if n_regional > 0 and patch_valid_mask is not None:
            valid_windows = patch_valid_mask.any(dim=2)
            token_mask[:, n_global:n_global + n_regional] = valid_windows
        # Window patches
        offset = n_global + n_regional
        if patch_valid_mask is not None and n_window_patches > 0:
            token_mask[:, offset:offset + n_window_patches] = patch_valid_mask.reshape(batch, -1)
        # Clinical patches
        if n_clinical > 0:
            token_mask[:, offset + n_window_patches:] = (
                clinical_present.unsqueeze(1).expand(-1, n_clinical)
            )

        # Multiclass criterion module
        mc_out = self.multiclass_module(
            token_cat, token_mask, frozen_crit,
        )

        # Diagnosis and melanoma residuals from diagnosis query features
        diag_delta = self.diagnosis_delta_head(mc_out.diagnosis_features)
        mel_delta = self.melanoma_delta_head(mc_out.diagnosis_features).squeeze(-1)

        final_diag = self._bounded_residual(frozen_diag, diag_delta, self.diagnosis_gate)
        final_mel = self._bounded_residual(frozen_mel, mel_delta, self.melanoma_gate)

        return {
            "diagnosis": final_diag,
            "melanoma": final_mel,
            "criteria": mc_out.final_criterion_logits,
            "criterion_residual": mc_out.criterion_residual,
            "raw_logits": mc_out.raw_logits,
            "checklist_melanoma": mc_out.checklist_melanoma_logit,
            "diagnosis_delta": diag_delta,
            "melanoma_delta": mel_delta,
            "residual_gates": mc_out.residual_gates,
            "diagnosis_gate": torch.tanh(self.diagnosis_gate).detach(),
            "melanoma_gate": torch.tanh(self.melanoma_gate).detach(),
            "frozen_diagnosis": frozen_diag,
            "frozen_melanoma": frozen_mel,
            "frozen_criteria": frozen_crit,
        }

    def forward_diagnosis_only(
        self,
        global_prefix: Tensor,
        frozen_outputs: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Global-only partial-label route for HAM10000 diagnosis training."""
        if global_prefix.ndim != 3 or global_prefix.shape[1:] != (197, TOKEN_DIM):
            raise ValueError("global_prefix must have shape [batch, 197, 768]")
        batch = global_prefix.shape[0]
        if frozen_outputs is None:
            missing_clinical = torch.zeros_like(global_prefix)
            clinical_present = torch.zeros(
                batch, dtype=torch.bool, device=global_prefix.device
            )
            with torch.no_grad():
                frozen = self.frozen_baseline(
                    global_prefix, missing_clinical, clinical_present
                )
                frozen_diagnosis = frozen["diagnosis"].detach()
                frozen_melanoma = frozen["melanoma"].detach()
        else:
            frozen_diagnosis = frozen_outputs["diagnosis"].detach()
            frozen_melanoma = frozen_outputs["melanoma"].detach()
            if frozen_diagnosis.shape[0] != batch or frozen_melanoma.shape != (batch,):
                raise ValueError("precomputed diagnosis-only baseline shape mismatch")
            if frozen_diagnosis.device != global_prefix.device or frozen_melanoma.device != global_prefix.device:
                raise ValueError("precomputed diagnosis-only baseline must share device")

        global_features = self.unified_tail.forward_global_only(global_prefix)
        token_mask = torch.ones(
            global_features.patches.shape[:2],
            dtype=torch.bool,
            device=global_prefix.device,
        )
        diagnosis_feature = self.multiclass_module.diagnosis_only_feature(
            global_features.patches, token_mask
        )
        diagnosis_delta = self.diagnosis_delta_head(diagnosis_feature)
        melanoma_delta = self.melanoma_delta_head(diagnosis_feature).squeeze(-1)
        return {
            "diagnosis": self._bounded_residual(
                frozen_diagnosis, diagnosis_delta, self.diagnosis_gate
            ),
            "melanoma": self._bounded_residual(
                frozen_melanoma, melanoma_delta, self.melanoma_gate
            ),
            "diagnosis_delta": diagnosis_delta,
            "melanoma_delta": melanoma_delta,
            "frozen_diagnosis": frozen_diagnosis,
            "frozen_melanoma": frozen_melanoma,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_baseline.eval()
        return self

    def trainable_parameter_groups(self) -> list[dict[str, Any]]:
        """Return optimizer parameter groups with suggested LRs."""
        tail_backbone = []
        tail_lora = []
        tail_new = []
        for name, p in self.unified_tail.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("blocks.") and "lora_" in name:
                tail_lora.append(p)
            elif any(prefix in name for prefix in ("blocks.", "norm.", "projection.")):
                tail_backbone.append(p)
            else:
                tail_new.append(p)

        mc_params = []
        route_controls = []
        for name, parameter in self.multiclass_module.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                "gate" in name
                or name.startswith("checklist_raw_weights")
                or name.startswith("checklist_threshold")
            ):
                route_controls.append(parameter)
            else:
                mc_params.append(parameter)
        delta_params = (
            list(self.diagnosis_delta_head.parameters())
            + list(self.melanoma_delta_head.parameters())
        )
        route_controls.extend([self.diagnosis_gate, self.melanoma_gate])

        return [
            {"params": tail_backbone, "lr": 1e-5, "name": "backbone_tail"},
            {"params": tail_lora, "lr": 3e-5, "name": "tail_lora"},
            {"params": tail_new, "lr": 1e-4, "name": "new_modules"},
            {"params": mc_params, "lr": 1e-4, "name": "multiclass"},
            {"params": delta_params, "lr": 1e-4, "name": "delta_heads"},
            {
                "params": route_controls,
                "lr": 1e-3,
                "name": "route_controls",
                "weight_decay": 0.0,
            },
        ]


__all__ = ["Phase25Model"]
