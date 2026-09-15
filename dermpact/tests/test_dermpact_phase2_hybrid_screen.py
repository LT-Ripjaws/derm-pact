from __future__ import annotations

import numpy as np
import pytest
import torch

from dermpact.eval.dermpact_phase2_hybrid_screen import (
    CONFIGS,
    CRITERIA,
    DERM_HARMONIZED_PROBABILITY_MAP,
    GLOBAL_CLASSES,
    HybridSharedHead,
    _data,
    _mapped_probabilities,
    derm_raw_to_global,
    multitask_loss,
)


def test_six_prespecified_configs_are_a_three_by_two_cross():
    assert len(CONFIGS) == 6
    assert {config.learning_rate for config in CONFIGS} == {3e-4, 1e-3, 3e-3}
    assert {config.loss_weighting for config in CONFIGS} == {
        "equal_family", "equal_output"
    }


def test_all_locked_derm_diagnoses_have_a_global_mapping():
    data = _data()
    mapped = {
        derm_raw_to_global(row["diagnosis"]["raw"])
        for row in data["derm_rows"]
    }
    assert mapped.issubset(set(GLOBAL_CLASSES))
    with pytest.raises(ValueError, match="unmapped"):
        derm_raw_to_global("not a real diagnosis")


def test_hybrid_shared_head_shapes():
    model = HybridSharedHead()
    outputs = model(torch.randn(4, 512))
    assert outputs["diagnosis"].shape == (4, len(GLOBAL_CLASSES))
    assert outputs["melanoma"].shape == (4,)
    assert outputs["criteria"].shape == (4, len(CRITERIA))


def test_ham_loss_cannot_receive_criterion_targets():
    outputs = HybridSharedHead()(torch.randn(4, 512))
    common = {
        "outputs": outputs,
        "dataset": "ham",
        "diagnosis": torch.tensor([0, 1, 2, 3]),
        "melanoma": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "diagnosis_weight": torch.ones(4),
        "melanoma_weight": torch.ones(4),
        "loss_weighting": "equal_family",
    }
    loss, components = multitask_loss(**common)
    assert torch.isfinite(loss)
    assert float(components["criteria_mean"].detach()) == 0.0
    with pytest.raises(ValueError, match="masked"):
        multitask_loss(
            **common,
            criteria=torch.zeros(4, len(CRITERIA)),
            criterion_weights=torch.ones(4, len(CRITERIA)),
        )


def test_derm_loss_weightings_are_finite_and_distinct():
    outputs = HybridSharedHead()(torch.randn(4, 512))
    common = {
        "outputs": outputs,
        "dataset": "derm7pt",
        "diagnosis": torch.tensor([0, 1, 2, 3]),
        "melanoma": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "diagnosis_weight": torch.ones(4),
        "melanoma_weight": torch.ones(4),
        "criteria": torch.zeros(4, len(CRITERIA)),
        "criterion_weights": torch.ones(4, len(CRITERIA)),
    }
    family, _ = multitask_loss(**common, loss_weighting="equal_family")
    output, _ = multitask_loss(**common, loss_weighting="equal_output")
    assert torch.isfinite(family) and torch.isfinite(output)
    assert float(family.detach()) != float(output.detach())


def test_harmonized_probability_projection_is_complete():
    rng = np.random.default_rng(5)
    universal = rng.random((7, len(GLOBAL_CLASSES)))
    universal /= universal.sum(axis=1, keepdims=True)
    classes, mapped = _mapped_probabilities(
        universal, DERM_HARMONIZED_PROBABILITY_MAP, renormalize=False
    )
    assert classes == sorted(DERM_HARMONIZED_PROBABILITY_MAP)
    np.testing.assert_allclose(mapped.sum(axis=1), 1.0)


def test_locked_folds_are_group_disjoint_and_complete():
    data = _data()
    assert len(data["ham_folds"]) == 7008
    assert len(data["derm_folds"]) == 616
    assert set(data["ham_folds"].tolist()) == set(range(5))
    assert set(data["derm_folds"].tolist()) == set(range(5))
    for fold in range(5):
        train_lesions = {
            data["ham_rows"][index]["cluster_id"]
            for index in np.flatnonzero(data["ham_folds"] != fold)
        }
        valid_lesions = {
            data["ham_rows"][index]["cluster_id"]
            for index in np.flatnonzero(data["ham_folds"] == fold)
        }
        assert train_lesions.isdisjoint(valid_lesions)
