import importlib.util
from unittest import SkipTest

from medguard.pipeline.adaptive_selector import GMMAdaptiveTopKSelector


def _require_sklearn():
    if importlib.util.find_spec("sklearn") is None:
        raise SkipTest("scikit-learn is absent from the bundled runtime")


def test_bimodal_scores_select_high_similarity_component():
    _require_sklearn()
    result = GMMAdaptiveTopKSelector().select([0.95, 0.90, 0.85, 0.20, 0.15, 0.10])
    assert result.n_components >= 2
    assert result.selected_indices == [0, 1, 2]


def test_constant_and_empty_distributions_are_explicit():
    selector = GMMAdaptiveTopKSelector()
    assert selector.select([]).selected_indices == []
    constant = selector.select([0.5, 0.5, 0.5])
    assert constant.selected_indices == [0, 1, 2]
    assert constant.fallback_reason == "constant_score_distribution"


def test_invalid_scores_and_selection_bounds_fail():
    try:
        GMMAdaptiveTopKSelector().select([0.5, float("nan")])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite score accepted")
    try:
        GMMAdaptiveTopKSelector(min_selected=2, max_selected=1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid selection bounds accepted")


def test_selection_is_deterministic():
    _require_sklearn()
    scores = [0.91, 0.88, 0.72, 0.3, 0.2, 0.1]
    selector = GMMAdaptiveTopKSelector(random_state=7)
    assert selector.select(scores) == selector.select(scores)
