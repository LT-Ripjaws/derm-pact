from medguard.eval.metrics import (
    harm_coverage_auc, harm_coverage_curve, matched_harm_coverage_auc,
)
from medguard.eval.statistics import delong_auc_test, holm_adjust, mcnemar_exact


def test_harm_coverage_auc_rewards_safe_ranking():
    safe = harm_coverage_auc([0.9, 0.1], [0.0, 2.0])
    unsafe = harm_coverage_auc([0.1, 0.9], [0.0, 2.0])
    assert safe < unsafe
    assert harm_coverage_curve([], []) == []


def test_matched_hcauc_does_not_reward_abstention_as_zero_harm():
    value = matched_harm_coverage_auc(
        [0.9, 0.99], [1.0, 0.0], [True, False],
        total_cases=2, max_coverage=0.5,
    )
    assert value == 1.0


def test_matched_hcauc_rejects_method_below_common_coverage():
    assert matched_harm_coverage_auc(
        [0.9, 0.8], [0.0, 0.0], [True, False],
        total_cases=2, max_coverage=1.0,
    ) is None


def test_mcnemar_and_holm():
    b, c, p = mcnemar_exact([True, True, True], [False, False, True])
    assert (b, c) == (2, 0) and 0 <= p <= 1
    adjusted = holm_adjust({"a": 0.01, "b": 0.04})
    assert adjusted["a"] <= adjusted["b"]


def test_delong_detects_auc_ordering():
    result = delong_auc_test([True, True, False, False], [0.9, 0.8, 0.2, 0.1],
                             [0.6, 0.4, 0.7, 0.3])
    assert result["auc_a"] > result["auc_b"]
