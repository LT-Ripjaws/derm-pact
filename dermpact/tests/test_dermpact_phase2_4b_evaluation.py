"""Focused tests for pure Phase 2.4B canonical-array evaluation."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from dermpact.eval.dermpact_phase2_4b_evaluation import (
    evaluate_phase24b,
    fixed_bootstrap_weights,
)


def _fixture() -> dict:
    n = 616
    ids = np.asarray([f"case-{index:03d}" for index in range(n)])
    available = np.ones(n, dtype=bool)
    available[-7:] = False
    target = np.tile(np.asarray([0, 1], dtype=np.int8), n // 2)
    criteria = np.repeat(target[:, None], 7, axis=1)
    # Baseline gets overlapping class distributions; paired separates them.
    baseline_criteria = np.stack(
        [0.30 * (2 * criteria[:, index] - 1) + np.sin(np.arange(n) * (0.11 + index * 0.01))
         for index in range(7)],
        axis=1,
    )
    paired_criteria = 3.0 * (2 * criteria - 1)
    diagnosis = np.zeros((n, 5), dtype=float)
    diagnosis[np.arange(n), target] = 2.0
    melanoma = 3.0 * (2 * target - 1)

    truth = {
        "ids": ids,
        "diagnosis_targets": np.where(target == 1, "MEL", "NEV"),
        "melanoma_targets": target,
        "criterion_targets": criteria,
        "clinical_available": available,
    }
    def condition(criteria_logits, residual=0.0):
        residual_values = np.full((n, 7), residual, dtype=float)
        residual_values[~available] = 0.0
        return {
            "ids": ids.copy(),
            "diagnosis_logits": diagnosis.copy(),
            "melanoma_logits": melanoma.copy(),
            "criterion_logits": np.asarray(criteria_logits, dtype=float).copy(),
            "clinical_residual": residual_values,
        }
    baseline = condition(baseline_criteria)
    baseline.pop("clinical_residual")
    payload = {
        "truth": truth,
        "baseline": baseline,
        "true_pair": condition(paired_criteria, 0.25),
        "forced_missing": condition(baseline_criteria, 0.0),
        "wrong_pair": condition(baseline_criteria, 0.10),
        "duplicate_dermoscopy": condition(baseline_criteria, 0.10),
        "bootstrap_weights_all": np.ones((16, n), dtype=np.uint16),
        "bootstrap_weights_available": np.ones((16, 609), dtype=np.uint16),
    }
    return payload


def test_good_true_pair_promotes_and_controls_never_can() -> None:
    result = evaluate_phase24b(**_fixture())

    assert result["validation"]["status"] == "PASS"
    assert result["promotion"]["promoted"] is True
    assert all(result["promotion"]["gates"].values())
    assert result["negative_controls"]["wrong_pair"]["can_promote"] is False
    assert result["negative_controls"]["duplicate_dermoscopy"]["can_promote"] is False
    assert len(result["points"]["true_pair"]["all_cases"]["per_criterion_auroc"]) == 7
    assert len(result["ranking"]) == 5


@pytest.mark.parametrize("control", ("wrong_pair", "duplicate_dermoscopy"))
def test_true_pair_must_beat_nonpaired_controls(control: str) -> None:
    payload = _fixture()
    payload[control]["criterion_logits"] = payload["true_pair"][
        "criterion_logits"
    ].copy()

    result = evaluate_phase24b(**payload)

    assert not result["promotion"]["gates"][f"true_pair_beats_{control}_control"]
    assert not result["promotion"]["promoted"]


def test_available_subset_failure_blocks_an_all_case_gain() -> None:
    payload = _fixture()
    baseline = payload["baseline"]["criterion_logits"]
    candidate = baseline.copy()
    unavailable = ~payload["truth"]["clinical_available"]
    target = payload["truth"]["criterion_targets"]
    candidate[unavailable] = 3.0 * (2 * target[unavailable] - 1)
    payload["true_pair"]["criterion_logits"] = candidate

    result = evaluate_phase24b(**payload)

    assert result["promotion"]["gates"]["all_case_criterion_mean_ci_low_gt_0"]
    assert not result["promotion"]["gates"]["available_609_criterion_mean_ci_low_gt_0"]
    assert not result["promotion"]["promoted"]


def test_forced_missing_fallback_failure_blocks_promotion() -> None:
    payload = _fixture()
    payload["forced_missing"]["criterion_logits"][0, 0] += 2e-6

    result = evaluate_phase24b(**payload)

    assert not result["promotion"]["gates"]["forced_missing_fallback_max_le_1e-6"]
    assert result["promotion"]["decision"] == "RETAIN_BASELINE"


def test_direct_rule_disagreement_failure_blocks_promotion() -> None:
    payload = _fixture()
    payload["true_pair"]["criterion_logits"] *= -1.0

    result = evaluate_phase24b(**payload)

    assert result["paired_comparisons"]["true_pair"]["direct_rule_disagreement_delta"] > 0.02
    assert not result["promotion"]["gates"]["direct_rule_disagreement_delta_le_0p02"]
    assert not result["promotion"]["promoted"]


def test_nonfinite_and_row_alignment_fail_closed() -> None:
    payload = _fixture()
    payload["wrong_pair"]["criterion_logits"][3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_phase24b(**payload)

    payload = _fixture()
    payload["duplicate_dermoscopy"]["ids"] = np.roll(
        payload["duplicate_dermoscopy"]["ids"], 1
    )
    with pytest.raises(ValueError, match="canonical aligned order"):
        evaluate_phase24b(**payload)


def test_exact_output_and_residual_safety_gates_are_enforced() -> None:
    payload = _fixture()
    payload["true_pair"]["melanoma_logits"][0] += 1e-12
    payload["wrong_pair"]["clinical_residual"][0, 0] = 1.000001

    result = evaluate_phase24b(**payload)

    assert not result["promotion"]["gates"]["diagnosis_and_melanoma_exact"]
    assert not result["promotion"]["gates"]["clinical_residual_max_le_1"]
    assert not result["promotion"]["promoted"]


def test_fixed_bootstrap_stratifies_on_diagnosis_and_criterion_pattern() -> None:
    n = 616
    diagnosis = np.where(np.arange(n) % 2, "MEL", "NEV")
    available = np.ones(n, dtype=bool)
    available[-7:] = False
    criteria = np.zeros((n, 7), dtype=np.int8)
    criteria[10, 6] = 1
    criteria[11, 6] = 1

    all_weights, available_weights = fixed_bootstrap_weights(
        diagnosis, criteria, available, n_boot=128, seed=17
    )

    assert np.all(all_weights @ criteria[:, 6] == 2)
    assert np.all(available_weights @ criteria[available, 6] == 2)
    bad = criteria.astype(float)
    bad[0, 0] = 0.5
    with pytest.raises(ValueError, match="finite and binary"):
        fixed_bootstrap_weights(diagnosis, bad, available, n_boot=2)


@pytest.mark.parametrize(
    "condition", ("true_pair", "wrong_pair", "duplicate_dermoscopy")
)
def test_nonzero_missing_path_residual_blocks_promotion(condition: str) -> None:
    payload = _fixture()
    unavailable = np.flatnonzero(~payload["truth"]["clinical_available"])
    payload[condition]["clinical_residual"][unavailable[0], 0] = 1e-12

    result = evaluate_phase24b(**payload)

    invariant = result["validation"]["missing_path_clinical_residual_max_abs"]
    assert invariant[f"{condition}_unavailable_rows"] == pytest.approx(1e-12)
    assert not result["validation"]["missing_path_clinical_residual_exact_zero"]
    assert not result["promotion"]["gates"]["missing_path_clinical_residual_exact_zero"]
    assert not result["promotion"]["promoted"]


def test_forced_missing_residual_must_be_zero_on_every_row() -> None:
    payload = _fixture()
    payload["forced_missing"]["clinical_residual"][0, 0] = 1e-12

    result = evaluate_phase24b(**payload)

    invariant = result["validation"]["missing_path_clinical_residual_max_abs"]
    assert invariant["forced_missing_all_rows"] == pytest.approx(1e-12)
    assert not result["promotion"]["gates"]["missing_path_clinical_residual_exact_zero"]
