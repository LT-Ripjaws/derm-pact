"""Focused tests for the Phase 2.4B fold-training core."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from dermpact.eval import dermpact_phase2_4b_training as training


def _write_bound(path: Path, value: object) -> str:
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def test_real_sharp_fold_and_clinical_cache_are_discovered_read_only() -> None:
    binding = training.resolve_sharp_fold(0)
    cache = training.load_clinical_cache()

    assert binding.training_receipt["arm"] == "sharp"
    assert binding.training_receipt["seed"] == training.SEED
    assert binding.training_receipt["completed_epochs"] == training.EPOCHS
    assert cache.derm_tokens.shape == (616, 197, 768)
    assert cache.clinical_tokens.shape == cache.derm_tokens.shape
    assert cache.clinical_available.sum() == 609
    assert np.count_nonzero(np.asarray(cache.clinical_tokens[~cache.clinical_available])) == 0


def test_pure_protocol_generator_is_deterministic_and_binds_every_fold() -> None:
    first = training.generate_protocol_spec()
    second = training.generate_protocol_spec()

    assert first == second
    assert first["identity"] == {
        "seed": 20260827,
        "folds": [0, 1, 2, 3, 4],
        "baseline_arm": "sharp",
    }
    assert len(first["artifacts"]["sharp_folds"]) == 5
    assert first["training"]["epochs"] == 20
    assert first["training"]["effective_batch_size"] == 64
    assert first["training"]["clinical_modality_dropout"] == 0.5
    assert first["protocol_sha256"] == training._canonical_hash(
        {
            key: value
            for key, value in first.items()
            if key not in {"protocol_sha256", "signature_payload"}
        }
    )
    assert training.validate_protocol_spec(first) == first["protocol_sha256"]
    tampered = dict(first)
    tampered["training"] = {**first["training"], "epochs": 21}
    with pytest.raises(AssertionError, match="digest"):
        training.validate_protocol_spec(tampered)


def test_sharp_resolver_rejects_receipt_bound_to_another_protocol(tmp_path: Path) -> None:
    protocol = {
        "schema_version": training.sharp_factor.PROTOCOL_SCHEMA,
        "status": "FROZEN",
        "identity": {
            "seed": training.SEED,
            "folds": list(training.FOLDS),
            "arms": ["sharp", "multidepth"],
        },
    }
    protocol_path = tmp_path / "attention_factor_protocol.json"
    _write_bound(protocol_path, protocol)
    receipt_dir = tmp_path / "sharp"
    receipt_dir.mkdir()
    checkpoint = tmp_path / "sharp.pt"
    checkpoint_sha = _write_bound(checkpoint, b"not-a-real-checkpoint")
    receipt = {
        "schema_version": training.sharp_factor.TRAINING_SCHEMA,
        "status": "PASS",
        "protocol_sha256": "0" * 64,
        "arm": "sharp",
        "seed": training.SEED,
        "fold": 0,
        "completed_epochs": training.EPOCHS,
        "new_residual_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
        },
    }
    _write_bound(receipt_dir / f"seed_{training.SEED}_fold_0_training.json", receipt)

    with pytest.raises(AssertionError, match="receipt identity mismatch"):
        training.resolve_sharp_fold(0, protocol_path=protocol_path)


class _FakeBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(7), requires_grad=False)


class _FakeFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prototype_projection = nn.Linear(1, 1, bias=False)
        self.derm_projection = nn.Linear(1, 1, bias=False)
        self.clinical_projection = nn.Linear(1, 1, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(7))
        self.raw_mixture_weight = nn.Parameter(torch.zeros(7))
        self.raw_output_scale = nn.Parameter(torch.zeros(7))
        self.gate_bias = nn.Parameter(torch.zeros(7))
        self.gate_uncertainty_weight = nn.Parameter(torch.zeros(7))


class _FakePaired(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.baseline_model = _FakeBaseline()
        self.clinical_tail = nn.Module()
        self.clinical_tail.adapters = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
        self.cross_modal_fusion = _FakeFusion()

    def forward(self, derm: Tensor, clinical: Tensor, present: Tensor):
        derm_value = derm.reshape(len(derm), -1).mean(dim=1, keepdim=True)
        clinical_value = clinical.reshape(len(clinical), -1).mean(dim=1, keepdim=True)
        adapted = clinical_value
        for adapter in self.clinical_tail.adapters:
            adapted = adapter(adapted)
        projected = self.cross_modal_fusion.clinical_projection(adapted)
        projected = projected + self.cross_modal_fusion.derm_projection(derm_value)
        projected = self.cross_modal_fusion.prototype_projection(projected)
        controls = (
            self.cross_modal_fusion.raw_output_scale
            + self.cross_modal_fusion.raw_mixture_weight
            + self.cross_modal_fusion.gate_bias
            + self.cross_modal_fusion.gate_uncertainty_weight
            + 0.01 * self.cross_modal_fusion.log_temperature
        )
        residual = projected * controls.unsqueeze(0)
        base_criteria = derm_value.repeat(1, 7)
        criteria = base_criteria + torch.where(
            present.unsqueeze(1), residual, torch.zeros_like(residual)
        )
        return {"base_criteria": base_criteria, "criteria": criteria}


class _TailBlock(nn.Module):
    def forward(self, tokens: Tensor, *, rel_pos_bias=None) -> Tensor:
        return tokens


class _RetainedTail(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TailBlock(), _TailBlock()])
        self.norm = nn.LayerNorm(768)
        self.shared_rel_pos_bias = None


class _SharpStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.new_residual = nn.Linear(1, 1, bias=False)
        residual = nn.Module()
        residual.register_buffer("prototype_embeddings", torch.randn(7, 2, 3))
        self.confirmed_model = SimpleNamespace(criterion_residual=residual)
        self.base_model = SimpleNamespace(visual_tail=_RetainedTail())


def test_model_construction_strictly_reloads_sharp_and_limits_trainable_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sharp = _SharpStub()
    expected = {"weight": torch.full_like(sharp.new_residual.weight, 0.25)}
    binding = _fake_binding(tmp_path)
    monkeypatch.setattr(training.sharp_factor, "_new_model", lambda *args: sharp)
    monkeypatch.setattr(training.torch, "load", lambda *args, **kwargs: expected)

    model = training.construct_paired_model(binding)

    assert torch.equal(model.baseline_model.new_residual.weight, expected["weight"])
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(
        name.startswith("clinical_tail.adapters.")
        or name.startswith("cross_modal_fusion.")
        for name in trainable
    )


def _fake_binding(tmp_path: Path) -> training.SharpFoldBinding:
    protocol = tmp_path / "protocol.json"
    receipt = tmp_path / "receipt.json"
    checkpoint = tmp_path / "sharp.pt"
    for path in (protocol, receipt, checkpoint):
        path.write_bytes(b"x")
    return training.SharpFoldBinding(
        fold=0,
        protocol={},
        protocol_path=protocol,
        protocol_sha256="1" * 64,
        training_receipt={},
        training_receipt_path=receipt,
        training_receipt_sha256="2" * 64,
        residual_checkpoint_path=checkpoint,
        residual_checkpoint_sha256="3" * 64,
    )


def _fake_cache(tmp_path: Path, rows: int) -> training.ClinicalCache:
    manifest = tmp_path / "cache_manifest.json"
    manifest.write_bytes(b"cache")
    values = np.zeros((rows, 1, 1), dtype="float32")
    return training.ClinicalCache(
        cache_dir=tmp_path,
        manifest={},
        manifest_sha256="4" * 64,
        derm_tokens=values,
        clinical_tokens=values,
        clinical_available=np.ones(rows, dtype=bool),
        bindings={},
    )


def _fake_data() -> training.FoldTrainingData:
    rows = 20
    labels = np.asarray(
        [[float((row + criterion) % 2) for criterion in range(7)] for row in range(rows)],
        dtype="float32",
    )
    return training.FoldTrainingData(
        derm_tokens=np.linspace(-1, 1, rows, dtype="float32").reshape(rows, 1, 1),
        clinical_tokens=np.linspace(1, -1, rows, dtype="float32").reshape(rows, 1, 1),
        clinical_available=np.asarray([row % 5 != 0 for row in range(rows)], dtype=bool),
        criterion_targets=labels,
        folds=np.asarray([row % 5 for row in range(rows)], dtype="int64"),
    )


def test_optimizer_groups_are_disjoint_exhaustive_and_clinical_only() -> None:
    model = _FakePaired()
    optimizer = training.build_optimizer(model)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    assert [group["group_name"] for group in optimizer.param_groups] == list(
        training.OPTIMIZER_GROUPS
    )
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert model.baseline_model.bias.requires_grad is False


def test_mutable_state_helper_is_exact_and_rejects_missing_or_extra_keys() -> None:
    source = _FakePaired()
    target = _FakePaired()
    state = training.mutable_clinical_state(source)

    assert state
    assert all(
        name.startswith("clinical_tail.adapters.")
        or name.startswith("cross_modal_fusion.")
        for name in state
    )
    assert not any(name.startswith("baseline_model.") for name in state)
    training.load_mutable_clinical_state(target, state)
    restored = training.mutable_clinical_state(target)
    assert all(torch.equal(state[name], restored[name]) for name in state)

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(AssertionError, match="missing="):
        training.load_mutable_clinical_state(target, missing)
    extra = {**state, "cross_modal_fusion.unregistered": torch.zeros(1)}
    with pytest.raises(AssertionError, match="extra="):
        training.load_mutable_clinical_state(target, extra)


def test_paired_loss_is_equal_average_and_dropout_mask_is_deterministic() -> None:
    model = _FakePaired()
    derm = torch.tensor([[[0.2]], [[-0.3]], [[0.8]], [[-0.1]]])
    clinical = torch.ones_like(derm)
    available = torch.tensor([True, True, False, True])
    retained = torch.tensor([True, False, True, False])
    targets = torch.tensor([[0, 1, 0, 1, 0, 1, 0]] * 4, dtype=torch.float32)
    weights = torch.ones_like(targets)

    total, parts = training.paired_training_loss(
        model, derm, clinical, available, retained, targets, weights
    )

    assert torch.equal(total, 0.5 * parts["true_pair_bce"] + 0.5 * parts["modality_dropout_bce"])
    assert parts["true_pair_available_fraction"].item() == pytest.approx(0.75)
    assert parts["dropout_available_fraction"].item() == pytest.approx(0.25)


def test_single_forward_dropout_loss_matches_explicit_two_forward_reference() -> None:
    optimized = _FakePaired()
    with torch.no_grad():
        for index, parameter in enumerate(optimized.parameters()):
            if parameter.requires_grad:
                parameter.fill_(0.07 * (index + 1))
    reference = copy.deepcopy(optimized)
    derm = torch.tensor(
        [[[0.2]], [[-0.3]], [[0.8]], [[-0.1]], [[0.4]], [[-0.6]]]
    )
    clinical = torch.tensor(
        [[[0.9]], [[-0.2]], [[0.5]], [[1.1]], [[-0.7]], [[0.3]]]
    )
    available = torch.tensor([True, True, False, True, False, True])
    retained = torch.tensor([True, False, True, False, False, True])
    targets = torch.tensor(
        [
            [float((row + criterion) % 2) for criterion in range(7)]
            for row in range(len(derm))
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [[1.0 + 0.1 * ((row + criterion) % 3) for criterion in range(7)]
         for row in range(len(derm))],
        dtype=torch.float32,
    )

    actual_loss, actual_parts = training.paired_training_loss(
        optimized, derm, clinical, available, retained, targets, weights
    )
    actual_loss.backward()

    dropout_mask = available & retained
    true_output = reference(derm, clinical, available)
    dropout_output = reference(derm, clinical, dropout_mask)
    reference_true = training._criterion_bce(
        true_output["criteria"], targets, weights
    )
    reference_dropout = training._criterion_bce(
        dropout_output["criteria"], targets, weights
    )
    reference_loss = 0.5 * reference_true + 0.5 * reference_dropout
    reference_loss.backward()

    assert torch.equal(actual_loss, reference_loss)
    assert torch.equal(actual_parts["true_pair_bce"], reference_true)
    assert torch.equal(actual_parts["modality_dropout_bce"], reference_dropout)
    assert torch.equal(
        actual_parts["true_pair_available_fraction"], available.float().mean()
    )
    assert torch.equal(
        actual_parts["dropout_available_fraction"], dropout_mask.float().mean()
    )
    actual_gradients = {
        name: parameter.grad
        for name, parameter in optimized.named_parameters()
        if parameter.requires_grad
    }
    reference_gradients = {
        name: parameter.grad
        for name, parameter in reference.named_parameters()
        if parameter.requires_grad
    }
    assert actual_gradients.keys() == reference_gradients.keys()
    for name in actual_gradients:
        assert actual_gradients[name] is not None, name
        assert reference_gradients[name] is not None, name
        torch.testing.assert_close(
            actual_gradients[name],
            reference_gradients[name],
            rtol=1e-6,
            atol=1e-8,
            msg=name,
        )


def test_epoch_resume_reproduces_uninterrupted_clinical_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _fake_data()
    binding = _fake_binding(tmp_path)
    cache = _fake_cache(tmp_path, len(data.folds))
    monkeypatch.setattr(training, "resolve_sharp_fold", lambda *args, **kwargs: binding)
    monkeypatch.setattr(training, "load_clinical_cache", lambda **kwargs: cache)
    monkeypatch.setattr(
        training,
        "validate_protocol_spec",
        lambda protocol, **kwargs: str(protocol["protocol_sha256"]),
    )
    protocol = {"protocol_sha256": "5" * 64, "hard_scope": {"official_rows": 0}}

    continuous = training.train_fold(
        0,
        output_dir=tmp_path / "continuous",
        protocol_spec=protocol,
        training_data=data,
        model_factory=lambda _: _FakePaired(),
    )
    interrupted = training.train_fold(
        0,
        output_dir=tmp_path / "resumed",
        protocol_spec=protocol,
        training_data=data,
        model_factory=lambda _: _FakePaired(),
        stop_after_epoch=7,
    )
    resumed = training.train_fold(
        0,
        output_dir=tmp_path / "resumed",
        protocol_spec=protocol,
        training_data=data,
        model_factory=lambda _: _FakePaired(),
    )

    assert interrupted["status"] == "INCOMPLETE"
    assert interrupted["completed_epochs"] == 7
    resume_payload = torch.load(
        interrupted["resume"]["path"], map_location="cpu", weights_only=False
    )
    assert "model_state" not in resume_payload
    assert resume_payload["mutable_clinical_state"]
    assert all(
        name.startswith("clinical_tail.adapters.")
        or name.startswith("cross_modal_fusion.")
        for name in resume_payload["mutable_clinical_state"]
    )
    assert not any(
        name.startswith("baseline_model.")
        for name in resume_payload["mutable_clinical_state"]
    )
    assert continuous["status"] == resumed["status"] == "PASS"
    first = torch.load(continuous["checkpoint"]["path"], weights_only=True)
    second = torch.load(resumed["checkpoint"]["path"], weights_only=True)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert continuous["history"][-1]["mean_training_loss"] == pytest.approx(
        resumed["history"][-1]["mean_training_loss"], abs=0.0
    )
