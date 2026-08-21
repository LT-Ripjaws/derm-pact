"""The label-free calibrated selective-abstention gate (Fix 4)."""
from medguard.eval.selective_gate import (
    CalibratedAbstentionGate,
    fit_diagnostic_abstention_gate,
)


def test_gate_answers_target_fraction_by_confidence_quantile():
    # Ten confidences 0.10..1.00; fitting at 0.90 answer-coverage puts tau at the
    # 10th percentile, so only the single lowest-confidence lesion abstains.
    confidences = [round(0.1 * i, 2) for i in range(1, 11)]
    gate = fit_diagnostic_abstention_gate(confidences, target_answer_coverage=0.90)
    assert gate.fitted is True
    assert gate.n_calibration == 10
    answered = [c for c in confidences if not gate.should_abstain(c)]
    abstained = [c for c in confidences if gate.should_abstain(c)]
    assert abstained == [0.10]
    assert len(answered) == 9


def test_gate_is_label_free_and_uses_only_confidence():
    # The gate never sees a label; it abstains purely below the fitted tau.
    gate = CalibratedAbstentionGate(diagnostic_confidence_tau=0.5)
    assert gate.should_abstain(0.49) is True
    assert gate.should_abstain(0.50) is False
    assert gate.should_abstain(0.80) is False
    # A missing confidence never abstains (nothing to threshold).
    assert gate.should_abstain(None) is False


def test_empty_calibration_is_inert_not_guessing():
    gate = fit_diagnostic_abstention_gate([None, None], target_answer_coverage=0.9)
    assert gate.fitted is False
    assert gate.n_calibration == 0
    # An inert gate abstains on nothing.
    assert gate.should_abstain(0.01) is False


def test_gate_round_trips_through_json(tmp_path):
    gate = fit_diagnostic_abstention_gate(
        [0.2, 0.4, 0.6, 0.8, 1.0], target_answer_coverage=0.8)
    path = tmp_path / "gate.json"
    gate.save(path)
    restored = CalibratedAbstentionGate.load(path)
    assert restored.diagnostic_confidence_tau == gate.diagnostic_confidence_tau
    assert restored.target_answer_coverage == gate.target_answer_coverage
    assert restored.fitted == gate.fitted
    assert restored.n_calibration == gate.n_calibration
