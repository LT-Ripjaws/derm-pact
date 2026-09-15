from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from dermpact.eval.dermpact_phase2_vector_bootstrap import (
    balanced_accuracy_samples,
    binary_auc_samples,
    interval,
    mean_binary_auc_samples,
    stratified_bootstrap_weights,
)


def _expanded_indices(weight_row: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(weight_row.size), weight_row.astype(int))


def test_stratified_weights_are_deterministic_and_preserve_each_class():
    labels = np.array(["a", "a", "a", "b", "b", "c"], dtype=object)
    first = stratified_bootstrap_weights(labels, n_boot=20, seed=91)
    second = stratified_bootstrap_weights(labels, n_boot=20, seed=91)
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == labels.size)
    for label in sorted(set(labels)):
        mask = labels == label
        assert np.all(first[:, mask].sum(axis=1) == int(mask.sum()))


def test_vector_binary_auc_matches_explicit_resampling_with_unique_scores():
    truth = np.array([0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.2, 0.3, 0.8, 0.7, 0.9])
    weights = stratified_bootstrap_weights(
        [str(value) for value in truth], n_boot=30, seed=12
    )
    observed = binary_auc_samples(truth, scores, weights)
    expected = np.array([
        roc_auc_score(truth[index := _expanded_indices(row)], scores[index])
        for row in weights
    ])
    assert np.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_vector_binary_auc_matches_explicit_resampling_with_ties():
    truth = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.9, 0.5, 0.5])
    weights = stratified_bootstrap_weights(
        [str(value) for value in truth], n_boot=25, seed=31
    )
    observed = binary_auc_samples(truth, scores, weights)
    expected = []
    for row in weights:
        index = _expanded_indices(row)
        expected.append(roc_auc_score(truth[index], scores[index]))
    assert np.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_vector_balanced_accuracy_matches_explicit_resampling():
    truth = np.array(["a", "a", "b", "b", "c", "c", "c"], dtype=object)
    predicted = np.array(["a", "b", "b", "a", "c", "a", "c"], dtype=object)
    weights = stratified_bootstrap_weights(truth, n_boot=40, seed=7)
    observed = balanced_accuracy_samples(truth, predicted, weights)
    expected = []
    for row in weights:
        index = _expanded_indices(row)
        expected.append(balanced_accuracy_score(truth[index], predicted[index]))
    assert np.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_mean_auc_and_interval_contract():
    weights = stratified_bootstrap_weights(
        ["00", "01", "00", "11", "10", "11"], n_boot=10, seed=1
    )
    targets = {
        "one": [0, 0, 0, 1, 1, 1],
        "two": [0, 1, 0, 1, 0, 1],
    }
    scores = {
        "one": [0.1, 0.2, 0.3, 0.8, 0.7, 0.9],
        "two": [0.2, 0.7, 0.1, 0.8, 0.3, 0.9],
    }
    samples = mean_binary_auc_samples(targets, scores, weights)
    summary = interval(samples, float(np.mean(samples)), n_boot=10)
    assert summary["n_boot"] == 10
    assert summary["ci_low"] <= summary["estimate"] <= summary["ci_high"]
