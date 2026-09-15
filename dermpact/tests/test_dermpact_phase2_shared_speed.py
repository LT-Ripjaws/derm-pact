from __future__ import annotations

import numpy as np
import pytest
import torch

from dermpact.eval.dermpact_phase2_shared_speed import (
    CRITERIA,
    SharedPerceptionHead,
    _cyclic_batches,
    _locked_training_rows,
    masked_multitask_loss,
)


def _outputs(batch: int = 4):
    model = SharedPerceptionHead(512, ham_classes=7, derm_classes=5)
    return model(torch.randn(batch, 512))


def test_shared_head_shapes_cover_native_diagnosis_and_seven_criteria():
    outputs = _outputs(3)
    assert outputs["ham_diagnosis"].shape == (3, 7)
    assert outputs["derm_diagnosis"].shape == (3, 5)
    assert outputs["melanoma"].shape == (3,)
    assert outputs["criteria"].shape == (3, len(CRITERIA))


def test_ham_loss_requires_criteria_to_remain_masked():
    outputs = _outputs()
    total, components = masked_multitask_loss(
        outputs,
        dataset="ham",
        diagnosis=torch.tensor([0, 1, 2, 3]),
        melanoma=torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )
    assert torch.isfinite(total)
    assert float(components["criteria"].detach()) == 0.0
    with pytest.raises(ValueError, match="fabricated"):
        masked_multitask_loss(
            outputs,
            dataset="ham",
            diagnosis=torch.tensor([0, 1, 2, 3]),
            melanoma=torch.zeros(4),
            criteria=torch.zeros(4, len(CRITERIA)),
        )


def test_derm_loss_requires_all_criterion_targets():
    outputs = _outputs()
    total, components = masked_multitask_loss(
        outputs,
        dataset="derm7pt",
        diagnosis=torch.tensor([0, 1, 2, 3]),
        melanoma=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        criteria=torch.zeros(4, len(CRITERIA)),
    )
    assert torch.isfinite(total)
    assert float(components["criteria"].detach()) > 0.0
    with pytest.raises(ValueError, match="requires"):
        masked_multitask_loss(
            outputs,
            dataset="derm7pt",
            diagnosis=torch.tensor([0, 1, 2, 3]),
            melanoma=torch.zeros(4),
        )


def test_cyclic_batches_are_full_and_reuse_only_inside_allowed_pool():
    allowed = np.asarray([10, 20, 30])
    batches = list(
        _cyclic_batches(
            allowed, steps=4, batch_size=5, rng=np.random.default_rng(7)
        )
    )
    assert [len(batch) for batch in batches] == [5, 5, 5, 5]
    assert set(np.concatenate(batches)).issubset(set(allowed))
    assert set(allowed).issubset(set(np.concatenate(batches)))


def test_locked_fold_zero_training_rows_exclude_held_out_groups():
    rows = _locked_training_rows()
    assert len(rows["ham_indices"]) < 7008
    assert len(rows["derm_indices"]) == 489
    assert rows["held_out_derm_cases"] == 127
    assert rows["held_out_ham_lesions"] > 0
