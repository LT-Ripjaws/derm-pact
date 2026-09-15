from __future__ import annotations

import json

from dermpact.eval import dermpact_phase2_dermlip_control as control
from dermpact.eval import dermpact_phase2_hybrid_duration_repair as repair
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


def test_control_is_exact_twelve_epoch_equal_family_match():
    assert control.CANDIDATE == "dermlip"
    assert control.MODEL_ID == "redlessone/DermLIP_ViT-B-16"
    assert control.CONFIG_ID == repair.CONFIG_ID == "lr0p0003_equal_family"
    assert control.EPOCHS == 12
    assert control.SEEDS == (20260827, 20260828, 20260829)
    config = screen.CONFIG_BY_ID[control.CONFIG_ID]
    assert config.learning_rate == 3e-4
    assert config.loss_weighting == "equal_family"


def test_control_starts_after_duration_is_closed_at_twelve():
    selected = json.loads(control.HYBRID_RECEIPT.read_text(encoding="utf-8"))
    duration = json.loads(control.EPOCH20_RECEIPT.read_text(encoding="utf-8"))
    assert selected["status"] == "PASS"
    assert selected["selected_configuration"] == control.CONFIG_ID
    assert selected["epochs"] == 12
    assert duration["decision"] == "RETAIN_12"
    assert duration["selected_epochs"] == 12
    assert duration["hard_scope_checks"][
        "further_frozen_head_epoch_testing_authorized"
    ] is False


def test_runtime_override_changes_only_candidate_duration_and_output_paths():
    old = {
        "candidate": screen.CANDIDATE,
        "model_id": screen.MODEL_ID,
        "epochs": repair.EPOCHS,
        "output": repair.OUTPUT_DIR,
        "experiment": repair.EXPERIMENT_DIR,
    }
    try:
        control._configure_runtime()
        assert screen.CANDIDATE == control.CANDIDATE
        assert screen.MODEL_ID == control.MODEL_ID
        assert repair.EPOCHS == 12
        assert repair.OUTPUT_DIR == control.OUTPUT_DIR
        assert repair.EXPERIMENT_DIR == control.EXPERIMENT_DIR
    finally:
        screen.CANDIDATE = old["candidate"]
        screen.MODEL_ID = old["model_id"]
        repair.EPOCHS = old["epochs"]
        repair.OUTPUT_DIR = old["output"]
        repair.EXPERIMENT_DIR = old["experiment"]


def test_dermlip_features_are_complete_and_hash_verified():
    old_candidate = screen.CANDIDATE
    try:
        screen.CANDIDATE = control.CANDIDATE
        data = screen._data()
        assert data["ham_features"].shape == (7008, 512)
        assert data["derm_features"].shape == (616, 512)
        assert data["ham_feature_manifest"]["candidate"] == "dermlip"
        assert data["derm_feature_manifest"]["candidate"] == "dermlip"
    finally:
        screen.CANDIDATE = old_candidate


def test_decision_rule_is_fail_closed_and_does_not_force_superiority():
    no_regression = []
    assert control._decision(
        {"ci_low": -0.01, "ci_high": -0.001}, no_regression
    ) == "HYBRID_COMPLEXITY_JUSTIFIED"
    assert control._decision(
        {"ci_low": 0.001, "ci_high": 0.01}, no_regression
    ) == "REOPEN_BACKBONE_DECISION"
    assert control._decision(
        {"ci_low": -0.01, "ci_high": 0.01}, no_regression
    ) == "HYBRID_ADVANTAGE_UNRESOLVED"
    assert control._decision(
        {"ci_low": -0.01, "ci_high": -0.001}, [{"endpoint": "ham"}]
    ) == "REOPEN_BACKBONE_DECISION"


def test_hybrid_protected_regression_requires_large_certain_dermlip_gain():
    comparison = {
        "derm7pt": {
            "primary": {"estimate": 0.0, "ci_low": -0.01, "ci_high": 0.01},
            "balanced_accuracy": {
                "estimate": 0.019,
                "ci_low": 0.001,
                "ci_high": 0.03,
            },
        },
        "ham": {
            "balanced_accuracy": {
                "estimate": 0.03,
                "ci_low": -0.001,
                "ci_high": 0.05,
            },
            "melanoma_auc": {
                "estimate": 0.021,
                "ci_low": 0.001,
                "ci_high": 0.03,
            },
        },
    }
    violations = control._hybrid_protected_regressions(comparison)
    assert [item["endpoint"] for item in violations] == ["ham_melanoma_auc"]


def test_control_scope_excludes_every_protected_dataset():
    assert "bcn" not in str(control.EXPERIMENT_DIR).lower()
    assert "ph2" not in str(control.EXPERIMENT_DIR).lower()
    assert control.DERMLIP_FROZEN_RESULT.exists()
    assert control.DERMLIP_FROZEN_HAM.exists()
    assert control.DERMLIP_FROZEN_DERM.exists()
