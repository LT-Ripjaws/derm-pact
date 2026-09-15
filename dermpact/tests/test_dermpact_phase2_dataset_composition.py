from __future__ import annotations

import numpy as np
import pytest
import torch

from dermpact.eval import dermpact_phase2_dataset_composition as composition
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


def test_bcn_mapping_covers_every_training_label():
    data = composition._load_bcn_data()
    observed = {str(row["label"]) for row in data["bcn_rows"]}
    assert observed == set(composition.BCN_TO_GLOBAL)
    assert set(data["bcn_global"].tolist()).issubset(
        set(range(len(screen.GLOBAL_CLASSES)))
    )


def test_bcn_folds_are_lesion_disjoint_complete_and_stratified():
    data = composition._load_bcn_data()
    assert set(data["bcn_folds"].tolist()) == set(range(5))
    assert len(data["bcn_fold_assignments"]) == 2143
    for fold in range(5):
        train_lesions = {
            str(row["cluster_id"])
            for row, value in zip(data["bcn_rows"], data["bcn_folds"])
            if int(value) != fold
        }
        valid_lesions = {
            str(row["cluster_id"])
            for row, value in zip(data["bcn_rows"], data["bcn_folds"])
            if int(value) == fold
        }
        assert train_lesions.isdisjoint(valid_lesions)
        valid_labels = {
            str(row["label"])
            for row in data["bcn_fold_assignments"]
            if int(row["fold"]) == fold
        }
        assert valid_labels == set(composition.BCN_TO_GLOBAL)


def test_bcn_loss_masks_all_checklist_targets():
    data = composition._data()
    weights = composition._training_weights(data, fold=0)
    model = screen.HybridSharedHead()
    loss, components = composition._bcn_batch_loss(
        model, np.arange(8), data, weights
    )
    assert torch.isfinite(loss)
    assert float(components["criteria_mean"].detach()) == 0.0


def test_optimizer_budget_matches_completed_reference_formula():
    data = composition._data()
    for fold in range(5):
        expected = max(
            int(np.ceil(np.sum(data["ham_folds"] != fold) / screen.BATCH_SIZE)),
            int(np.ceil(np.sum(data["derm_folds"] != fold) / screen.BATCH_SIZE)),
        )
        assert composition._steps_per_epoch(data, fold) == expected


def test_derm_only_weights_do_not_require_or_touch_bcn_data():
    data = screen._data()
    weights = composition._training_weights(data, fold=0)
    assert "bcn_diagnosis" not in weights
    assert "bcn_melanoma" not in weights


def test_arm_and_split_inputs_fail_closed():
    with pytest.raises(ValueError):
        composition.train_fold("invented_arm", fold=0, seed=20260827, data={})
    with pytest.raises(ValueError):
        composition._point({}, "bcn", "made_up")
