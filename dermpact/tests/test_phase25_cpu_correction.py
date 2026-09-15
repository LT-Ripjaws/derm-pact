"""Bounded CPU regressions for the Phase 2.5 raw-CE correction."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dermpact.eval import dermpact_phase2_5_stage2_training as stage2
from dermpact.eval import dermpact_phase2_5_training as stage1


N = 16
DEVICE = torch.device("cpu")


def _targets() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Return targets containing every declared raw-state class."""
    raw_targets = {
        criterion: np.asarray(
            [index % stage1.RAW_N_CLASSES[criterion] for index in range(N)],
            dtype=np.int64,
        )
        for criterion in stage1.CRITERIA
    }
    criterion_targets = np.asarray(
        [[index % 2 for _ in stage1.CRITERIA] for index in range(N)],
        dtype=np.float64,
    )
    derm_data = {
        "derm_global": np.asarray([index % 5 for index in range(N)], dtype=np.int64),
        "derm_melanoma": np.asarray([index % 2 for index in range(N)], dtype=np.float64),
    }
    return raw_targets, criterion_targets, derm_data


def _weights(raw_targets: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        criterion: torch.linspace(
            1.0, 1.0 + 0.25 * (stage1.RAW_N_CLASSES[criterion] - 1),
            stage1.RAW_N_CLASSES[criterion],
            dtype=torch.float64,
        )
        for criterion in raw_targets
    }


def _base_output() -> dict[str, object]:
    generator = torch.Generator().manual_seed(250904)
    return {
        "criteria": torch.randn(N, len(stage1.CRITERIA), generator=generator, dtype=torch.float64),
        "raw_logits": {
            criterion: torch.randn(
                N,
                stage1.RAW_N_CLASSES[criterion],
                generator=generator,
                dtype=torch.float64,
            )
            for criterion in stage1.CRITERIA
        },
        "diagnosis": torch.randn(N, 5, generator=generator, dtype=torch.float64),
        "melanoma": torch.randn(N, generator=generator, dtype=torch.float64),
        "checklist_melanoma": torch.randn(N, generator=generator, dtype=torch.float64),
        "frozen_criteria": torch.randn(
            N, len(stage1.CRITERIA), generator=generator, dtype=torch.float64
        ),
    }


def _leaf_output(base: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in base.items():
        if name == "raw_logits":
            result[name] = {
                criterion: tensor.detach().clone().requires_grad_()
                for criterion, tensor in value.items()  # type: ignore[union-attr]
            }
        else:
            result[name] = value.detach().clone().requires_grad_()  # type: ignore[union-attr]
    return result


def _denominators(
    indices: np.ndarray,
    raw_targets: dict[str, np.ndarray],
    weights: dict[str, torch.Tensor],
) -> dict[str, float]:
    return stage1._class_weight_denominators(indices, raw_targets, weights)


def _loss(
    kind: str,
    output: dict[str, object],
    indices: np.ndarray,
    raw_targets: dict[str, np.ndarray],
    criterion_targets: np.ndarray,
    derm_data: dict[str, np.ndarray],
    weights: dict[str, torch.Tensor],
    denominators: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if kind == "stage1":
        return stage1._multitask_loss(
            output,
            indices,
            raw_targets,
            weights,
            criterion_targets,
            derm_data,
            DEVICE,
            denominators,
        )
    return stage2.derm_stage2_loss(
        output,
        indices=indices,
        raw_targets=raw_targets,
        class_weights=weights,
        paired_data=SimpleNamespace(criterion_targets=criterion_targets),
        derm_data=derm_data,
        device=DEVICE,
        class_weight_denominators=denominators,
    )


def _parts(partition: list[int]) -> list[np.ndarray]:
    boundaries = np.cumsum([0, *partition])
    return [np.arange(start, stop, dtype=np.int64) for start, stop in zip(boundaries, boundaries[1:])]


@pytest.mark.parametrize("kind", ["stage1", "stage2"])
@pytest.mark.parametrize(
    "partition",
    [[16], [8, 8], [4, 4, 4, 4], [2] * 8, [5, 3, 8]],
    ids=["mb16", "mb8", "mb4", "mb2", "unequal-5-3-8"],
)
def test_raw_ce_accumulation_preserves_real_loss_and_gradients(
    kind: str, partition: list[int]
) -> None:
    """Microbatching must match one logical batch for both real loss paths."""
    raw_targets, criterion_targets, derm_data = _targets()
    weights = _weights(raw_targets)
    indices = np.arange(N, dtype=np.int64)
    denominators = _denominators(indices, raw_targets, weights)
    base = _base_output()

    full_output = _leaf_output(base)
    full_loss, _ = _loss(
        kind, full_output, indices, raw_targets, criterion_targets, derm_data,
        weights, denominators,
    )
    full_loss.backward()

    accumulated_output = _leaf_output(base)
    accumulated = torch.zeros((), dtype=torch.float64)
    for selection in _parts(partition):
        piece, parts = _loss(
            kind,
            {
                name: (
                    {
                        criterion: tensor[selection]
                        for criterion, tensor in value.items()  # type: ignore[union-attr]
                    }
                    if name == "raw_logits"
                    else value[selection]  # type: ignore[index]
                )
                for name, value in accumulated_output.items()
            },
            selection,
            raw_targets,
            criterion_targets,
            derm_data,
            weights,
            denominators,
        )
        fraction = len(selection) / N
        accumulated = accumulated + stage1._scale_microbatch_loss(
            piece, parts["raw_ce"], fraction
        ).detach()
        stage1._scale_microbatch_loss(piece, parts["raw_ce"], fraction).backward()

    torch.testing.assert_close(accumulated, full_loss.detach(), rtol=1e-9, atol=1e-9)
    for name, full_value in full_output.items():
        if name == "raw_logits":
            for criterion, full_tensor in full_value.items():  # type: ignore[union-attr]
                observed = accumulated_output[name][criterion].grad  # type: ignore[index,union-attr]
                torch.testing.assert_close(observed, full_tensor.grad, rtol=1e-9, atol=1e-9)
        else:
            torch.testing.assert_close(
                accumulated_output[name].grad,  # type: ignore[union-attr]
                full_value.grad,  # type: ignore[union-attr]
                rtol=1e-9,
                atol=1e-9,
            )


def test_raw_ce_without_logical_denominator_keeps_native_mean_smoke_semantics() -> None:
    raw_targets, criterion_targets, derm_data = _targets()
    weights = _weights(raw_targets)
    output = _leaf_output(_base_output())
    indices = np.arange(N, dtype=np.int64)
    _, parts = stage1._multitask_loss(
        output,
        indices,
        raw_targets,
        weights,
        criterion_targets,
        derm_data,
        DEVICE,
        class_weight_denominators=None,
    )
    expected = torch.stack(
        [
            F.cross_entropy(
                output["raw_logits"][criterion],  # type: ignore[index]
                torch.from_numpy(raw_targets[criterion]),
                weight=weights[criterion],
            )
            for criterion in stage1.CRITERIA
        ]
    ).mean()
    torch.testing.assert_close(parts["raw_ce"], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["stage1", "stage2"])
def test_real_loss_invariance_regression_detects_original_double_scaling(kind: str) -> None:
    """Replay the old accumulator in memory, without reverting any source file."""
    with patch.object(stage1, "_scale_microbatch_loss", side_effect=lambda total, raw, fraction: total * fraction):
        with pytest.raises(AssertionError):
            test_raw_ce_accumulation_preserves_real_loss_and_gradients(kind, [8, 8])


def test_shared_microbatch_scaler_enforces_fraction_contract() -> None:
    total = torch.tensor(5.0, requires_grad=True)
    raw = torch.tensor(2.0, requires_grad=True)
    assert stage1._scale_microbatch_loss(total, raw, 0.25).item() == pytest.approx(2.75)
    for fraction in (0.0, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="fraction"):
            stage1._scale_microbatch_loss(total, raw, fraction)


def test_both_training_loops_use_the_raw_ce_preserving_scaler() -> None:
    for train_fold in (stage1.train_fold, stage2.train_fold):
        source = inspect.getsource(train_fold)
        assert "_scale_microbatch_loss" in source


def test_receipt_gate_accessor_is_safe_for_stage1_numpy_logging() -> None:
    from dermpact.perception.multiclass_criterion_module import MulticlassCriterionModule
    parameter = torch.tensor([0.1, -0.2], requires_grad=True)
    module = SimpleNamespace(residual_gate_raw=parameter)
    snapshot = MulticlassCriterionModule.residual_gates(module)
    assert not snapshot.requires_grad
    np.testing.assert_allclose(snapshot.cpu().numpy(), torch.tanh(parameter).detach().numpy())
    assert parameter.requires_grad


def test_explicit_cpu_backend_and_auto_accelerator_routes() -> None:
    for resolver in (stage1._resolve_device, stage2.resolve_device):
        device, name, resolved = resolver("cpu")
        assert str(device) == "cpu"
        assert name == "CPU"
        assert resolved == "cpu"

        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "get_device_name", return_value="mock-cuda"
        ):
            device, name, resolved = resolver("auto")
            assert str(device) == "cuda:0"
            assert name == "mock-cuda"
            assert resolved == "cuda"

    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
        stage1, "_get_device", return_value=(torch.device("meta"), "mock-directml")
    ):
        assert stage1._resolve_device("auto") == (
            torch.device("meta"), "mock-directml", "directml"
        )
        assert stage2.resolve_device("auto") == (
            torch.device("meta"), "mock-directml", "directml"
        )


def test_phase25_defaults_are_versioned_away_from_original_results() -> None:
    assert stage1.DEFAULT_OUTPUT_DIR != stage1.ROOT / "research/experiments/dermpact_phase2_5/stage1_v3"
    assert stage2.DEFAULT_OUTPUT_DIR != stage2.ROOT / "research/experiments/dermpact_phase2_5/stage2_v2"
    assert stage1.DEFAULT_OUTPUT_DIR.name.startswith("stage1_cefix_")
    assert stage2.DEFAULT_OUTPUT_DIR.name.startswith("stage2_cefix_")


def test_cpu_thread_arguments_reject_invalid_or_non_cpu_use() -> None:
    with pytest.raises(ValueError, match="positive"):
        stage1._configure_execution_threads(backend="cpu", cpu_threads=0)
    with pytest.raises(ValueError, match="positive"):
        stage1._configure_execution_threads(backend="cpu", cpu_threads=-2)
    with pytest.raises(ValueError, match="only with backend='cpu'"):
        stage1._configure_execution_threads(backend="cuda", cpu_threads=2)


def test_corrected_stage1_rejects_old_objective_resume(tmp_path) -> None:
    checkpoint = tmp_path / "old.pt"
    stage1._atomic_torch(checkpoint, {"schema_version": "dermpact-phase2-5-stage1-resume-v2"})
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(AssertionError, match="resume schema mismatch"):
        stage1._load_resume(checkpoint, model=model, optimizer=optimizer, device=DEVICE,
                            fold=0, protocol_sha256="unused", source_bundle_sha256="unused")


def test_corrected_stage2_rejects_uncorrected_stage1_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "fold_0" / "final_stage1.pt"
    stage1._atomic_torch(checkpoint, {"schema_version": "dermpact-phase2-5-stage1-final-v2"})
    with pytest.raises(AssertionError, match="Stage 1 checkpoint schema mismatch"):
        stage2.load_stage1_checkpoint(torch.nn.Linear(2, 1), fold=0, stage1_dir=tmp_path)


def test_portable_training_metadata_binding_is_strict_and_complete() -> None:
    root = stage1.ROOT / "friend-cuda"
    previous = (
        stage1.baselines.DERM7PT_INVENTORY,
        stage1.baselines.HAM_MANIFEST,
        stage1.baselines.HAM_ROLE_LEDGER,
        stage1._base.screen.FOLD_MANIFEST,
    )
    try:
        bindings = stage1._configure_training_metadata_root(root)
        assert bindings is not None
        assert set(bindings) == {
            "derm7pt_inventory", "ham_manifest", "ham_role_ledger", "fold_manifest",
            "clinical_photo_audit", "structured_fact_support",
        }
        assert len(stage1.baselines._derm7pt_records()) == 616
        assert len(stage2.screen._ham_records()) == 7008
        assert all(Path(row["path"]).is_relative_to(root) for row in bindings.values())
    finally:
        (
            stage1.baselines.DERM7PT_INVENTORY,
            stage1.baselines.HAM_MANIFEST,
            stage1.baselines.HAM_ROLE_LEDGER,
            stage1._base.screen.FOLD_MANIFEST,
        ) = previous
