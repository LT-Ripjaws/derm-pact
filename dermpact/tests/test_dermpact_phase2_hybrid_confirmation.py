from __future__ import annotations

import json

import numpy as np
import torch

from dermpact.eval import dermpact_phase2_hybrid_confirmation as confirmation
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


def test_confirmation_uses_exactly_the_two_screen_promotions():
    receipt = json.loads(confirmation.SCREEN_RECEIPT.read_text(encoding="utf-8"))
    assert tuple(receipt["promoted_configurations"]) == confirmation.CONFIG_IDS
    assert confirmation.CONFIG_IDS == (
        "lr0p0003_equal_family",
        "lr0p0003_equal_output",
    )


def test_confirmation_schedule_is_fixed_and_multi_seed():
    assert confirmation.EPOCHS == 40
    assert confirmation.SEEDS == (20260827, 20260828, 20260829)
    assert confirmation.N_BOOT == 2000
    assert len(set(confirmation.SEEDS)) == 3


def test_prediction_with_logits_returns_aligned_raw_and_probability_arrays():
    torch.manual_seed(4)
    model = screen.HybridSharedHead()
    features = np.random.default_rng(8).normal(size=(7, 512)).astype("float32")
    indices = np.array([1, 3, 6], dtype=np.int64)
    result = confirmation._predict_with_logits(model, features, indices)
    assert result["diagnosis_logits"].shape == (3, len(screen.GLOBAL_CLASSES))
    assert result["melanoma_logits"].shape == (3,)
    assert result["criterion_logits"].shape == (3, len(screen.CRITERIA))
    assert np.allclose(result["universal"].sum(axis=1), 1.0, atol=1e-6)
    assert np.all((result["melanoma"] >= 0.0) & (result["melanoma"] <= 1.0))


def test_derm_joint_strata_include_diagnosis_and_all_criteria():
    row = {
        "label": "MEL",
        "criterion_true": {
            criterion: index % 2 == 0
            for index, criterion in enumerate(screen.CRITERIA)
        },
    }
    strata = confirmation._derm_bootstrap_strata([row])
    assert strata == ["MEL|1010101"]


def test_seed_resampled_mean_uses_the_selected_seed_for_each_replicate():
    samples = [
        np.array([1.0, 2.0, 3.0]),
        np.array([10.0, 20.0, 30.0]),
        np.array([100.0, 200.0, 300.0]),
    ]
    draws = np.array([[0, 1, 2], [1, 1, 1], [2, 0, 2]])
    observed = confirmation._seed_resampled_mean(samples, draws)
    assert np.allclose(observed, [37.0, 20.0, 201.0])


def test_scope_contract_keeps_bcn_and_ph2_out_of_confirmation():
    assert "bcn" not in str(confirmation.EXPERIMENT_DIR).lower()
    assert "ph2" not in str(confirmation.EXPERIMENT_DIR).lower()
    data = screen._data()
    assert data["ham_features"].shape[0] == 7008
    assert data["derm_features"].shape[0] == 616
