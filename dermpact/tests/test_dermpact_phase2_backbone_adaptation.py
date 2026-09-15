from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from dermpact.eval import dermpact_phase2_backbone_adaptation as adaptation
from dermpact.eval import dermpact_phase2_backbone_training as training


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(values)
        return self.proj(qkv[..., : values.shape[-1]])


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _Attention(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(
        self, values: torch.Tensor, rel_pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del rel_pos_bias
        return values + self.attn(values) + self.mlp(values)


class _Visual(nn.Module):
    def __init__(self, dim: int = 8, out: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(12)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out)
        self.rel_pos_bias = None


def test_lora_weight_is_exactly_zero_at_initialization() -> None:
    torch.manual_seed(4)
    parameterization = adaptation.LoRAWeight(6, 5, rank=2, alpha=4.0)
    weight = torch.randn(6, 5)
    assert torch.equal(parameterization(weight), weight)
    assert parameterization.lora_a.requires_grad
    assert parameterization.lora_b.requires_grad


def test_last_block_arm_trains_only_last_block_norm_and_projection(monkeypatch) -> None:
    monkeypatch.setattr(adaptation, "TOKEN_DIM", 8)
    monkeypatch.setattr(adaptation, "FEATURE_DIM", 4)
    tail = adaptation.AdaptedVisualTail(_Visual(), "last_block")
    names = tail.adaptation_parameter_names()
    assert names
    assert all(
        name.startswith("blocks.1")
        or name.startswith("norm")
        or name.startswith("projection")
        for name in names
    )
    assert not any(name.startswith("blocks.0") for name in names)


def test_lora_arm_changes_only_parametrized_attention_weights(monkeypatch) -> None:
    monkeypatch.setattr(adaptation, "TOKEN_DIM", 8)
    monkeypatch.setattr(adaptation, "FEATURE_DIM", 4)
    visual = _Visual()
    tail = adaptation.AdaptedVisualTail(visual, "lora_last_two")
    names = tail.adaptation_parameter_names()
    assert len(names) == 8
    assert all("parametrizations.weight.0.lora_" in name for name in names)
    tokens = torch.randn(3, 5, 8)
    with torch.inference_mode():
        expected = adaptation._tail_features(
            visual.blocks[10:], visual.norm, visual.head, tokens
        )
        observed = tail(tokens)
    assert torch.equal(expected, observed)


def test_adaptation_scale_zero_restores_the_visual_anchor(monkeypatch) -> None:
    monkeypatch.setattr(adaptation, "TOKEN_DIM", 8)
    monkeypatch.setattr(adaptation, "FEATURE_DIM", 4)
    visual = _Visual()
    tokens = torch.randn(2, 5, 8)
    tail = adaptation.AdaptedVisualTail(visual, "last_block")
    expected = tail(tokens).detach().clone()
    with torch.no_grad():
        next(value for value in tail.parameters() if value.requires_grad).add_(0.25)
    tail.capture_adapted_state()
    changed = tail(tokens).detach().clone()
    assert not torch.equal(changed, expected)
    tail.set_adaptation_scale(0.0)
    assert torch.allclose(tail(tokens), expected, atol=1e-7, rtol=0.0)
    tail.set_adaptation_scale(1.0)
    assert torch.allclose(tail(tokens), changed, atol=1e-7, rtol=0.0)


def test_task_head_initialization_is_matched_across_adaptation_arms(monkeypatch) -> None:
    monkeypatch.setattr(adaptation, "TOKEN_DIM", 8)
    monkeypatch.setattr(adaptation, "FEATURE_DIM", 4)
    torch.manual_seed(19)
    left = adaptation.AdaptedHybridModel(_Visual(), "last_block")
    torch.manual_seed(19)
    right = adaptation.AdaptedHybridModel(_Visual(), "lora_last_two")
    for name, value in left.task_head.state_dict().items():
        assert torch.equal(value, right.task_head.state_dict()[name])


def test_cache_loader_rejects_identity_mismatch(tmp_path, monkeypatch) -> None:
    tokens = np.zeros((1, adaptation.TOKEN_COUNT, adaptation.TOKEN_DIM), dtype="float32")
    token_path = tmp_path / "ham_tokens.npy"
    np.save(token_path, tokens)
    manifest_path = tmp_path / "ham_manifest.json"
    manifest_path.write_text(
        '{"record_identity_sha256":"wrong","token_file_sha256":"wrong"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(training, "_cache_paths", lambda dataset: (token_path, manifest_path))
    with pytest.raises(AssertionError, match="identity"):
        training._load_prefix_tokens(
            "ham", [{"case_id": "x", "image_sha256": "a"}]
        )


def test_protocol_declares_every_protected_role_closed() -> None:
    protocol = json.loads(adaptation.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["hard_scope"]["derm7pt_official_test_rows_loaded"] == 0
    assert protocol["hard_scope"]["ham_calibration_rows_loaded"] == 0
    assert protocol["hard_scope"]["bcn_validation_or_final_rows_loaded"] == 0
    assert protocol["hard_scope"]["ph2_rows_loaded"] == 0
    assert "exact WiSE-FT" in protocol["interpolation"]["meaning"]


def test_current_environment_is_the_declared_canonical_cpu_environment() -> None:
    observed = training._canonical_environment()
    assert observed["status"] == "PASS"
    assert observed["observed"] == training.CANONICAL_ENVIRONMENT


def test_colab_notebook_is_valid_and_requires_local_reevaluation() -> None:
    notebook = adaptation.ROOT / "research/notebooks/DermPact_Phase2_3D_Backbone_Adaptation.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "--reevaluate" in sources
    assert "assert not IN_COLAB" in sources
    assert "--finalize" in sources
