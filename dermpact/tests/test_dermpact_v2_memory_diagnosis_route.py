from __future__ import annotations

import numpy as np

from dermpact.core.eval import memory_diagnosis_route as route


def test_zero_weight_preserves_base_exactly():
    base = np.asarray([[0.7, 0.2, 0.05, 0.03, 0.02], [0.1, 0.6, 0.1, 0.1, 0.1]])
    memory = base[:, ::-1]
    fused, alpha = route._fuse(base, memory, np.asarray([0.9, 0.2]), weight=0.0, min_quality=0.0)
    assert np.array_equal(fused, base)
    assert np.array_equal(alpha, np.zeros(2))


def test_memory_distribution_is_smoothed_and_normalized():
    distances = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
    targets = np.asarray([0, 0, 1])
    prior = np.asarray([0.5, 0.2, 0.1, 0.1, 0.1])
    probability, quality = route._memory_distribution(
        distances, targets, k=2, prior_strength=2.0, class_prior=prior, distance_scale=0.2,
    )
    assert probability.shape == (2, 5)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.all(probability > 0.0)
    assert np.all((quality >= 0.0) & (quality <= 1.0))


def test_collapse_diagnosis_preserves_probability_mass():
    source = np.full((3, len(route.GLOBAL_CLASSES)), 1.0 / len(route.GLOBAL_CLASSES))
    collapsed = route._collapse_diagnosis(source)
    assert collapsed.shape == (3, 5)
    assert np.allclose(collapsed.sum(axis=1), 1.0)
