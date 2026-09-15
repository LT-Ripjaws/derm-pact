import numpy as np

from dermpact.core.eval.final_metrics import (
    binary_metrics,
    criterion_metrics,
    multiclass_metrics,
    sensitivity_at_specificity,
)


def test_multiclass_metrics_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[y] * 0.9 + 0.1 / 3
    result = multiclass_metrics(y, p, melanoma_index=0)
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["mcc"] == 1.0
    assert result["melanoma_auroc"] == 1.0


def test_binary_metrics_uses_conventional_single_probability_brier():
    result = binary_metrics([0, 1], [0.25, 0.75])
    assert result["brier"] == 0.0625
    assert result["accuracy"] == 1.0


def test_sensitivity_at_specificity_reports_observed_operating_point():
    result = sensitivity_at_specificity(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.90)
    assert result["observed_specificity"] >= 0.90
    assert result["sensitivity"] == 1.0


def test_criterion_metrics_scores_raw_state_and_presence_separately():
    labels = ["absent", "atypical", "typical"]
    truth = ["absent", "atypical", "typical"]
    probabilities = np.eye(3)
    result = criterion_metrics(truth, probabilities, labels, {"atypical"}, [0.1, 0.9, 0.1])
    assert result["raw_state_accuracy"] == 1.0
    assert result["presence_f1"] == 1.0
    assert result["positive_support"] == 1
