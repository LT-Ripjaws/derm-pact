import numpy as np

from medguard.eval.derm_encoder_fusion import _concatenate_features, _mean_probabilities


def test_neutral_fusion_is_exact_arithmetic_mean():
    a = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    b = np.asarray([[0.4, 0.6], [0.3, 0.7]])
    observed = _mean_probabilities([a, b])
    np.testing.assert_allclose(observed, [[0.6, 0.4], [0.2, 0.8]])


def test_neutral_fusion_rejects_row_mismatch():
    try:
        _mean_probabilities([np.ones((2, 2)), np.ones((3, 2))])
    except ValueError:
        return
    raise AssertionError("row mismatch was accepted")


def test_concatenation_preserves_unit_norm_for_unit_inputs():
    a = np.asarray([[1.0, 0.0]])
    b = np.asarray([[0.0, 1.0]])
    observed = _concatenate_features([a, b])
    np.testing.assert_allclose(np.linalg.norm(observed, axis=1), [1.0])
