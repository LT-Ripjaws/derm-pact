from __future__ import annotations

import json

from dermpact.eval import dermpact_phase2_hybrid_confirmation as confirmation
from dermpact.eval import dermpact_phase2_hybrid_duration_repair as repair
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


def test_repair_changes_only_to_twelve_epoch_equal_family_schedule():
    assert repair.CONFIG_ID == "lr0p0003_equal_family"
    assert screen.CONFIG_BY_ID[repair.CONFIG_ID].learning_rate == 3e-4
    assert screen.CONFIG_BY_ID[repair.CONFIG_ID].loss_weighting == "equal_family"
    assert repair.EPOCHS == 12
    assert repair.SEEDS == (20260827, 20260828, 20260829)


def test_repair_is_conditioned_on_failed_confirmation():
    receipt = json.loads(repair.CONFIRMATION_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["selected_configuration"] is None
    comparisons = json.loads(
        repair.CONFIRMATION_COMPARISONS.read_text(encoding="utf-8")
    )
    direct = comparisons["direct_equal_family_minus_equal_output"]["derm7pt"][
        "primary"
    ]
    assert direct["estimate"] > 0.0
    assert direct["ci_low"] > 0.0


def test_runtime_override_is_explicit_and_limited():
    old_epochs = confirmation.EPOCHS
    old_output = confirmation.OUTPUT_DIR
    old_experiment = confirmation.EXPERIMENT_DIR
    try:
        repair._configure_training_runtime()
        assert confirmation.EPOCHS == 12
        assert confirmation.OUTPUT_DIR == repair.OUTPUT_DIR
        assert confirmation.EXPERIMENT_DIR == repair.EXPERIMENT_DIR
    finally:
        confirmation.EPOCHS = old_epochs
        confirmation.OUTPUT_DIR = old_output
        confirmation.EXPERIMENT_DIR = old_experiment


def test_repair_scope_keeps_protected_datasets_out():
    assert "bcn" not in str(repair.EXPERIMENT_DIR).lower()
    assert "ph2" not in str(repair.EXPERIMENT_DIR).lower()
    data = screen._data()
    assert data["ham_features"].shape == (7008, 512)
    assert data["derm_features"].shape == (616, 512)
