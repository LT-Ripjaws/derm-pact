"""Vectorized clustered-bootstrap helpers for DermPact Phase 2."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def stratified_bootstrap_weights(
    labels: Sequence[str], *, n_boot: int, seed: int
) -> np.ndarray:
    """Return ``n_boot x n_cases`` integer multiplicities."""
    labels_array = np.asarray([str(label) for label in labels], dtype=object)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    strata = [np.flatnonzero(labels_array == label) for label in sorted(set(labels_array))]
    if any(indices.size == 0 for indices in strata):
        raise AssertionError("empty bootstrap stratum")
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_boot, labels_array.size), dtype=np.uint16)
    for replicate in range(n_boot):
        for indices in strata:
            sampled = rng.choice(indices, size=indices.size, replace=True)
            weights[replicate] += np.bincount(
                sampled, minlength=labels_array.size
            ).astype(np.uint16)
    if not np.all(weights.sum(axis=1) == labels_array.size):
        raise AssertionError("bootstrap sample size changed")
    return weights


def _validate_weights(weights: np.ndarray, n_cases: int) -> np.ndarray:
    matrix = np.asarray(weights)
    if matrix.ndim != 2 or matrix.shape[1] != n_cases:
        raise ValueError("weights must have shape (replicates, cases)")
    if np.any(matrix < 0):
        raise ValueError("bootstrap weights cannot be negative")
    return matrix


def binary_auc_samples(
    truth: Sequence[int], scores: Sequence[float], weights: np.ndarray
) -> np.ndarray:
    """Compute binary AUROC for every weighted bootstrap sample."""
    y = np.asarray(truth, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape:
        raise ValueError("truth and scores must be aligned one-dimensional arrays")
    if not set(np.unique(y)).issubset({0, 1}) or np.unique(y).size != 2:
        raise ValueError("binary AUROC requires both classes")
    matrix = _validate_weights(weights, y.size).astype(np.float64, copy=False)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_truth = y[order]
    sorted_weight = matrix[:, order]
    positive_total = sorted_weight[:, sorted_truth == 1].sum(axis=1)
    negative_total = sorted_weight[:, sorted_truth == 0].sum(axis=1)
    denominator = positive_total * negative_total
    if np.any(denominator == 0):
        raise ValueError("a bootstrap replicate has only one binary class")

    if np.unique(sorted_score).size == sorted_score.size:
        negative_weight = sorted_weight * (sorted_truth == 0)[None, :]
        negative_before = np.cumsum(negative_weight, axis=1) - negative_weight
        positive_weight = sorted_weight * (sorted_truth == 1)[None, :]
        concordance = np.sum(positive_weight * negative_before, axis=1)
        return concordance / denominator

    concordance = np.zeros(matrix.shape[0], dtype=np.float64)
    negative_before = np.zeros(matrix.shape[0], dtype=np.float64)
    boundaries = np.flatnonzero(
        np.r_[True, sorted_score[1:] != sorted_score[:-1], True]
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        block = sorted_weight[:, start:stop]
        block_truth = sorted_truth[start:stop]
        positive = block[:, block_truth == 1].sum(axis=1)
        negative = block[:, block_truth == 0].sum(axis=1)
        concordance += positive * (negative_before + 0.5 * negative)
        negative_before += negative
    return concordance / denominator


def balanced_accuracy_samples(
    truth: Sequence[str], predictions: Sequence[str], weights: np.ndarray
) -> np.ndarray:
    """Compute multiclass balanced accuracy for each bootstrap sample."""
    true = np.asarray([str(value) for value in truth], dtype=object)
    predicted = np.asarray([str(value) for value in predictions], dtype=object)
    if true.ndim != 1 or predicted.shape != true.shape:
        raise ValueError("truth and predictions must be aligned")
    matrix = _validate_weights(weights, true.size).astype(np.float64, copy=False)
    recalls = []
    for label in sorted(set(true)):
        mask = true == label
        denominator = matrix[:, mask].sum(axis=1)
        if np.any(denominator == 0):
            raise ValueError("a bootstrap replicate lost a diagnosis class")
        correct = (predicted[mask] == label).astype(np.float64)
        recalls.append((matrix[:, mask] @ correct) / denominator)
    return np.mean(np.stack(recalls, axis=1), axis=1)


def mean_binary_auc_samples(
    targets: Mapping[str, Sequence[int]],
    scores: Mapping[str, Sequence[float]],
    weights: np.ndarray,
) -> np.ndarray:
    if set(targets) != set(scores) or not targets:
        raise ValueError("targets and scores must contain the same non-empty outputs")
    samples = [binary_auc_samples(targets[name], scores[name], weights) for name in targets]
    return np.mean(np.stack(samples, axis=1), axis=1)


def interval(samples: Sequence[float], estimate: float, *, n_boot: int) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    if values.shape != (n_boot,) or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap samples are missing or non-finite")
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": int(n_boot),
    }
