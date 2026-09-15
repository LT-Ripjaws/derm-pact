from __future__ import annotations

import json

from dermpact.eval import dermpact_phase2_hybrid_duration_repair as repair
from dermpact.eval import dermpact_phase2_hybrid_epoch20_probe as probe
from dermpact.eval import dermpact_phase2_hybrid_screen as screen


def test_probe_changes_only_the_fixed_duration_and_schedule_length():
    assert probe.CONFIG_ID == repair.CONFIG_ID == "lr0p0003_equal_family"
    assert screen.CONFIG_BY_ID[probe.CONFIG_ID].learning_rate == 3e-4
    assert screen.CONFIG_BY_ID[probe.CONFIG_ID].loss_weighting == "equal_family"
    assert probe.EPOCHS == 20
    assert repair.EPOCHS == 12
    assert probe.SEEDS == (20260827, 20260828, 20260829)


def test_probe_starts_from_the_selected_twelve_epoch_result():
    receipt = json.loads(probe.DURATION12_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["selected_configuration"] == probe.CONFIG_ID
    assert receipt["epochs"] == 12


def test_runtime_override_is_explicit_and_restorable():
    old_epochs = repair.EPOCHS
    old_output = repair.OUTPUT_DIR
    old_experiment = repair.EXPERIMENT_DIR
    try:
        probe._configure_runtime()
        assert repair.EPOCHS == 20
        assert repair.OUTPUT_DIR == probe.OUTPUT_DIR
        assert repair.EXPERIMENT_DIR == probe.EXPERIMENT_DIR
    finally:
        repair.EPOCHS = old_epochs
        repair.OUTPUT_DIR = old_output
        repair.EXPERIMENT_DIR = old_experiment


def test_probe_has_no_early_stopping_or_protected_dataset_access(monkeypatch):
    assert "bcn" not in str(probe.EXPERIMENT_DIR).lower()
    assert "ph2" not in str(probe.EXPERIMENT_DIR).lower()
    source = probe.__file__
    assert source is not None
    monkeypatch.setattr(probe.screen, "_write_json", lambda path, payload: "0" * 64)
    protocol = probe.build_protocol()
    assert protocol["configuration"]["early_stopping"] is False
    assert protocol["selection_rule"]["no_more_epoch_tests"] is True
    assert "BCN20000" in protocol["forbidden_data"]
    assert "PH2" in protocol["forbidden_data"]
    assert protocol["hard_scope_counts"]["forbidden_rows"] == 0


def test_protected_regression_rule_requires_large_certain_deficit():
    harmless = {
        "derm7pt": {
            "balanced_accuracy": {"estimate": -0.019, "ci_high": -0.001}
        },
        "ham": {
            "balanced_accuracy": {"estimate": -0.03, "ci_high": 0.001},
            "melanoma_auc": {"estimate": 0.0, "ci_high": 0.01},
        },
    }
    harmful = {
        **harmless,
        "ham": {
            **harmless["ham"],
            "balanced_accuracy": {"estimate": -0.021, "ci_high": -0.001},
        },
    }
    assert probe._protected_violations(harmless, reference="test") == []
    violations = probe._protected_violations(harmful, reference="test")
    assert [item["endpoint"] for item in violations] == [
        "ham_balanced_accuracy"
    ]
