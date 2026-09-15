"""Pure, auditable evaluation for the DermPact Phase 2.4B paired path."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from . import dermpact_phase2_architecture_ladder as ladder
from .dermpact_phase2_baselines import CRITERIA
from .dermpact_phase2_vector_bootstrap import (
    binary_auc_samples,
    interval,
    stratified_bootstrap_weights,
)


N_CASES = 616
N_AVAILABLE = 609
N_UNAVAILABLE = 7
N_CRITERIA = 7
CONDITIONS = (
    "baseline",
    "true_pair",
    "forced_missing",
    "wrong_pair",
    "duplicate_dermoscopy",
)
FORCED_MISSING_TOLERANCE = 1e-6
MAXIMUM_RESIDUAL = 1.0
DIRECT_RULE_DISAGREEMENT_DELTA_MAX = 0.02
JOINT_CORRECT_AGREEMENT_DELTA_MIN = -0.02
DEFAULT_BOOTSTRAP_SEED = 20260830


def fixed_bootstrap_weights(
    diagnosis_targets: Sequence[Any],
    criterion_targets: Sequence[Sequence[int]],
    clinical_available: Sequence[bool],
    *,
    n_boot: int = 2000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the prespecified stratified all-case and available-case draws."""
    diagnosis = np.asarray([str(value) for value in diagnosis_targets], dtype=object)
    criteria = np.asarray(criterion_targets)
    available = np.asarray(clinical_available)
    if diagnosis.shape != (N_CASES,):
        raise ValueError(f"diagnosis_targets must have shape ({N_CASES},)")
    if criteria.shape != (N_CASES, N_CRITERIA):
        raise ValueError(
            f"criterion_targets must have shape ({N_CASES}, {N_CRITERIA})"
        )
    if not np.all(np.isfinite(criteria)) or not set(np.unique(criteria)).issubset({0, 1}):
        raise ValueError("criterion_targets must be finite and binary")
    if available.shape != (N_CASES,) or available.dtype.kind != "b":
        raise ValueError(f"clinical_available must be Boolean shape ({N_CASES},)")
    if int(available.sum()) != N_AVAILABLE:
        raise ValueError(f"clinical_available must contain {N_AVAILABLE} true rows")
    labels = np.asarray(
        [
            f"{diagnosis[index]}|{''.join(str(int(bit)) for bit in criteria[index])}"
            for index in range(N_CASES)
        ],
        dtype=object,
    )
    return (
        stratified_bootstrap_weights(labels, n_boot=n_boot, seed=seed),
        stratified_bootstrap_weights(
            labels[available], n_boot=n_boot, seed=seed + 1
        ),
    )


def _array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    finite: bool = True,
) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if finite:
        try:
            is_finite = np.isfinite(result)
        except TypeError as exc:
            raise TypeError(f"{name} must be numeric") from exc
        if not np.all(is_finite):
            raise ValueError(f"{name} must be finite")
    return result


def _validate_weights(weights: Any, *, n_cases: int, name: str) -> np.ndarray:
    matrix = np.asarray(weights)
    if matrix.ndim != 2 or matrix.shape[1] != n_cases or matrix.shape[0] == 0:
        raise ValueError(f"{name} must have shape (replicates, {n_cases})")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{name} must contain finite non-negative weights")
    if np.any(matrix.sum(axis=1) <= 0):
        raise ValueError(f"{name} contains an empty replicate")
    return matrix


def _validate_truth(truth: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "ids",
        "diagnosis_targets",
        "melanoma_targets",
        "criterion_targets",
        "clinical_available",
    }
    missing = required.difference(truth)
    if missing:
        raise KeyError(f"truth is missing: {', '.join(sorted(missing))}")
    ids = _array(truth["ids"], name="truth.ids", shape=(N_CASES,), finite=False)
    if len({str(value) for value in ids}) != N_CASES:
        raise ValueError("truth.ids must contain 616 unique identities")
    diagnosis = _array(
        truth["diagnosis_targets"],
        name="truth.diagnosis_targets",
        shape=(N_CASES,),
        finite=False,
    )
    melanoma = _array(
        truth["melanoma_targets"],
        name="truth.melanoma_targets",
        shape=(N_CASES,),
    ).astype(np.int8)
    criteria = _array(
        truth["criterion_targets"],
        name="truth.criterion_targets",
        shape=(N_CASES, N_CRITERIA),
    ).astype(np.int8)
    available = _array(
        truth["clinical_available"],
        name="truth.clinical_available",
        shape=(N_CASES,),
        finite=False,
    )
    if available.dtype.kind != "b":
        raise TypeError("truth.clinical_available must be Boolean")
    if (int(available.sum()), int((~available).sum())) != (
        N_AVAILABLE,
        N_UNAVAILABLE,
    ):
        raise ValueError("clinical availability must be exactly 609 available / 7 unavailable")
    for name, value in (("melanoma_targets", melanoma), ("criterion_targets", criteria)):
        if not set(np.unique(value)).issubset({0, 1}):
            raise ValueError(f"truth.{name} must be binary")
    if any(np.unique(criteria[:, index]).size != 2 for index in range(N_CRITERIA)):
        raise ValueError("every criterion target must contain both classes")
    if np.unique(melanoma).size != 2:
        raise ValueError("melanoma_targets must contain both classes")

    if "melanocytic_mask" in truth:
        melanocytic = _array(
            truth["melanocytic_mask"],
            name="truth.melanocytic_mask",
            shape=(N_CASES,),
            finite=False,
        )
        if melanocytic.dtype.kind != "b":
            raise TypeError("truth.melanocytic_mask must be Boolean")
    else:
        normalized = np.asarray([str(value).strip().upper() for value in diagnosis])
        recognized = np.isin(normalized, ["MEL", "MELANOMA", "NEV", "NEVUS"])
        melanocytic = recognized if np.any(recognized) else np.ones(N_CASES, dtype=bool)
    return {
        "ids": ids,
        "diagnosis_targets": diagnosis,
        "melanoma_targets": melanoma,
        "criterion_targets": criteria,
        "clinical_available": available,
        "melanocytic_mask": melanocytic,
    }


def _validate_condition(
    name: str,
    raw: Mapping[str, Any],
    *,
    truth_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    required = {"ids", "diagnosis_logits", "melanoma_logits", "criterion_logits"}
    missing = required.difference(raw)
    if missing:
        raise KeyError(f"{name} is missing: {', '.join(sorted(missing))}")
    ids = _array(raw["ids"], name=f"{name}.ids", shape=(N_CASES,), finite=False)
    if [str(value) for value in ids] != [str(value) for value in truth_ids]:
        raise ValueError(f"{name}.ids are not in canonical aligned order")
    diagnosis = np.asarray(raw["diagnosis_logits"])
    if diagnosis.ndim != 2 or diagnosis.shape[0] != N_CASES or diagnosis.shape[1] < 2:
        raise ValueError(f"{name}.diagnosis_logits must have shape (616, classes>=2)")
    if not np.all(np.isfinite(diagnosis)):
        raise ValueError(f"{name}.diagnosis_logits must be finite")
    result = {
        "ids": ids,
        "diagnosis_logits": diagnosis,
        "melanoma_logits": _array(
            raw["melanoma_logits"], name=f"{name}.melanoma_logits", shape=(N_CASES,)
        ),
        "criterion_logits": _array(
            raw["criterion_logits"],
            name=f"{name}.criterion_logits",
            shape=(N_CASES, N_CRITERIA),
        ),
    }
    if name == "baseline":
        residual = np.zeros((N_CASES, N_CRITERIA), dtype=float)
    else:
        if "clinical_residual" not in raw:
            raise KeyError(f"{name} is missing: clinical_residual")
        residual = _array(
            raw["clinical_residual"],
            name=f"{name}.clinical_residual",
            shape=(N_CASES, N_CRITERIA),
        )
    result["clinical_residual"] = residual
    return result


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    # Stable conversion is useful for the existing row-shaped metric adapter.
    positive = logits >= 0
    result = np.empty_like(logits, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _criterion_point(targets: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    per = {
        CRITERIA[index]: float(roc_auc_score(targets[:, index], logits[:, index]))
        for index in range(N_CRITERIA)
    }
    return {"per_criterion_auroc": per, "criterion_mean_auroc": float(np.mean(list(per.values())))}


def _criterion_deltas(
    targets: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    per: dict[str, dict[str, float]] = {}
    samples = []
    points = []
    n_boot = int(weights.shape[0])
    for index, criterion in enumerate(CRITERIA):
        candidate_samples = binary_auc_samples(targets[:, index], candidate[:, index], weights)
        baseline_samples = binary_auc_samples(targets[:, index], baseline[:, index], weights)
        delta_samples = candidate_samples - baseline_samples
        point = float(
            roc_auc_score(targets[:, index], candidate[:, index])
            - roc_auc_score(targets[:, index], baseline[:, index])
        )
        per[criterion] = interval(delta_samples, point, n_boot=n_boot)
        samples.append(delta_samples)
        points.append(point)
    mean_samples = np.mean(np.stack(samples, axis=1), axis=1)
    return {
        "per_criterion": per,
        "criterion_mean": interval(mean_samples, float(np.mean(points)), n_boot=n_boot),
    }


def _rule_rows(truth: Mapping[str, np.ndarray], logits: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    criterion_probabilities = _sigmoid(logits["criterion_logits"])
    melanoma_probabilities = _sigmoid(logits["melanoma_logits"])
    rows = []
    for index in range(N_CASES):
        if truth["melanocytic_mask"][index]:
            label = "MEL" if truth["melanoma_targets"][index] else "NEV"
        else:
            label = "OTHER"
        rows.append(
            {
                "id": str(truth["ids"][index]),
                "label": label,
                "melanoma_probability": float(melanoma_probabilities[index]),
                "criterion_true": {
                    criterion: bool(truth["criterion_targets"][index, column])
                    for column, criterion in enumerate(CRITERIA)
                },
                "criterion_probability": {
                    criterion: float(criterion_probabilities[index, column])
                    for column, criterion in enumerate(CRITERIA)
                },
            }
        )
    return rows


def _rule_metrics(truth: Mapping[str, np.ndarray], condition: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows = _rule_rows(truth, condition)
    eligible, values = ladder._rule_vectors(rows)
    return {
        "eligible_cases": int(eligible.sum()),
        "eligible_mask": eligible,
        "joint_correct_agreement": float(np.mean(values[0, eligible])),
        "direct_rule_disagreement": float(np.mean(values[1, eligible])),
    }


def evaluate_phase24b(
    *,
    truth: Mapping[str, Any],
    baseline: Mapping[str, Any],
    true_pair: Mapping[str, Any],
    forced_missing: Mapping[str, Any],
    wrong_pair: Mapping[str, Any],
    duplicate_dermoscopy: Mapping[str, Any],
    bootstrap_weights_all: Any,
    bootstrap_weights_available: Any,
) -> dict[str, Any]:
    """Validate and compare the five aligned canonical Phase 2.4B conditions."""
    canonical_truth = _validate_truth(truth)
    raw_conditions = {
        "baseline": baseline,
        "true_pair": true_pair,
        "forced_missing": forced_missing,
        "wrong_pair": wrong_pair,
        "duplicate_dermoscopy": duplicate_dermoscopy,
    }
    conditions = {
        name: _validate_condition(name, value, truth_ids=canonical_truth["ids"])
        for name, value in raw_conditions.items()
    }
    weights_all = _validate_weights(
        bootstrap_weights_all, n_cases=N_CASES, name="bootstrap_weights_all"
    )
    weights_available = _validate_weights(
        bootstrap_weights_available,
        n_cases=N_AVAILABLE,
        name="bootstrap_weights_available",
    )
    available = canonical_truth["clinical_available"]
    reference = conditions["baseline"]

    exact = {
        name: {
            "diagnosis_logits_exact": bool(
                np.array_equal(value["diagnosis_logits"], reference["diagnosis_logits"])
            ),
            "melanoma_logits_exact": bool(
                np.array_equal(value["melanoma_logits"], reference["melanoma_logits"])
            ),
        }
        for name, value in conditions.items()
        if name != "baseline"
    }
    forced_difference = float(
        np.max(
            np.abs(
                conditions["forced_missing"]["criterion_logits"]
                - reference["criterion_logits"]
            )
        )
    )
    residual_max = {
        name: float(np.max(np.abs(value["clinical_residual"])))
        for name, value in conditions.items()
        if name != "baseline"
    }
    exact_zero_residual_max = {
        "forced_missing_all_rows": residual_max["forced_missing"],
        **{
            f"{name}_unavailable_rows": float(
                np.max(np.abs(conditions[name]["clinical_residual"][~available]))
            )
            for name in ("true_pair", "wrong_pair", "duplicate_dermoscopy")
        },
    }

    points: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    rule: dict[str, Any] = {}
    available_targets = canonical_truth["criterion_targets"][available]
    baseline_available = reference["criterion_logits"][available]
    for name, value in conditions.items():
        points[name] = {
            "all_cases": _criterion_point(
                canonical_truth["criterion_targets"], value["criterion_logits"]
            ),
            "available_609": _criterion_point(
                available_targets, value["criterion_logits"][available]
            ),
        }
        rule[name] = _rule_metrics(canonical_truth, value)
        if name != "baseline":
            comparisons[name] = {
                "all_cases_candidate_minus_baseline": _criterion_deltas(
                    canonical_truth["criterion_targets"],
                    value["criterion_logits"],
                    reference["criterion_logits"],
                    weights_all,
                ),
                "available_609_candidate_minus_baseline": _criterion_deltas(
                    available_targets,
                    value["criterion_logits"][available],
                    baseline_available,
                    weights_available,
                ),
                "direct_rule_disagreement_delta": float(
                    rule[name]["direct_rule_disagreement"]
                    - rule["baseline"]["direct_rule_disagreement"]
                ),
                "joint_correct_agreement_delta": float(
                    rule[name]["joint_correct_agreement"]
                    - rule["baseline"]["joint_correct_agreement"]
                ),
            }

    pair_comparison = comparisons["true_pair"]
    gates = {
        "row_identity_and_availability": True,
        "diagnosis_and_melanoma_exact": all(
            item["diagnosis_logits_exact"] and item["melanoma_logits_exact"]
            for item in exact.values()
        ),
        "forced_missing_fallback_max_le_1e-6": forced_difference <= FORCED_MISSING_TOLERANCE,
        "clinical_residual_max_le_1": all(value <= MAXIMUM_RESIDUAL for value in residual_max.values()),
        "missing_path_clinical_residual_exact_zero": all(
            value == 0.0 for value in exact_zero_residual_max.values()
        ),
        "all_case_criterion_mean_ci_low_gt_0": (
            pair_comparison["all_cases_candidate_minus_baseline"]["criterion_mean"]["ci_low"] > 0.0
        ),
        "available_609_criterion_mean_ci_low_gt_0": (
            pair_comparison["available_609_candidate_minus_baseline"]["criterion_mean"]["ci_low"] > 0.0
        ),
        "true_pair_beats_wrong_pair_control": (
            points["true_pair"]["all_cases"]["criterion_mean_auroc"]
            > points["wrong_pair"]["all_cases"]["criterion_mean_auroc"]
        ),
        "true_pair_beats_duplicate_dermoscopy_control": (
            points["true_pair"]["all_cases"]["criterion_mean_auroc"]
            > points["duplicate_dermoscopy"]["all_cases"]["criterion_mean_auroc"]
        ),
        "direct_rule_disagreement_delta_le_0p02": (
            pair_comparison["direct_rule_disagreement_delta"]
            <= DIRECT_RULE_DISAGREEMENT_DELTA_MAX
        ),
        "joint_correct_agreement_delta_ge_minus_0p02": (
            pair_comparison["joint_correct_agreement_delta"]
            >= JOINT_CORRECT_AGREEMENT_DELTA_MIN
        ),
    }
    promoted = bool(all(gates.values()))

    controls = {}
    for name in ("wrong_pair", "duplicate_dermoscopy"):
        controls[name] = {
            "criterion_mean_auroc": points[name],
            "candidate_minus_baseline": comparisons[name],
            "true_pair_minus_control_all_case_mean": float(
                points["true_pair"]["all_cases"]["criterion_mean_auroc"]
                - points[name]["all_cases"]["criterion_mean_auroc"]
            ),
            "can_promote": False,
        }

    ranking = sorted(
        (
            {
                "condition": name,
                "all_case_criterion_mean_auroc": values["all_cases"]["criterion_mean_auroc"],
                "available_609_criterion_mean_auroc": values["available_609"]["criterion_mean_auroc"],
                "promotion_eligible": name == "true_pair",
                "promoted": name == "true_pair" and promoted,
            }
            for name, values in points.items()
        ),
        key=lambda row: (
            -row["all_case_criterion_mean_auroc"],
            -row["available_609_criterion_mean_auroc"],
            row["condition"],
        ),
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    return {
        "contract": "dermpact-phase2-4b-pure-evaluation-v1",
        "validation": {
            "status": "PASS",
            "rows": N_CASES,
            "clinical_available": N_AVAILABLE,
            "clinical_unavailable": N_UNAVAILABLE,
            "condition_order": list(CONDITIONS),
            "exact_invariants": exact,
            "forced_missing_max_abs_criterion_difference": forced_difference,
            "clinical_residual_max_abs": residual_max,
            "missing_path_clinical_residual_max_abs": exact_zero_residual_max,
            "missing_path_clinical_residual_exact_zero": all(
                value == 0.0 for value in exact_zero_residual_max.values()
            ),
        },
        "points": points,
        "paired_comparisons": comparisons,
        "rule_metrics": {
            name: {key: value for key, value in metrics.items() if key != "eligible_mask"}
            for name, metrics in rule.items()
        },
        "negative_controls": controls,
        "promotion": {
            "candidate": "true_pair",
            "promoted": promoted,
            "decision": "PROMOTE_TRUE_PAIR" if promoted else "RETAIN_BASELINE",
            "gates": gates,
        },
        "ranking": ranking,
    }


__all__ = [
    "CONDITIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "evaluate_phase24b",
    "fixed_bootstrap_weights",
]
