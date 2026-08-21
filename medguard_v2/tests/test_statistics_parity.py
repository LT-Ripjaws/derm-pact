"""Parity: the ported v2 statistics must be indistinguishable from v1.

Bootstrap procedures are seeded explicitly (``SEED``) and the fixtures are drawn
from a local ``random.Random`` instance, so nothing here depends on global RNG
state or on test execution order.  Replicate *distributions* are compared, not
just the intervals, so any divergence in RNG call order is caught immediately.
"""
from __future__ import annotations

import inspect
import random
import unittest

from medguard.eval import statistics as v1
from medguard_v2.eval import statistics as v2


PORTED = (
    "clustered_bootstrap_distribution", "clustered_bootstrap_ci",
    "bootstrap_two_sided_p", "paired_clustered_bootstrap_difference",
    "mcnemar_exact", "holm_adjust", "binary_auc", "delong_auc_test",
)

SEED = 20260810
N_BOOT = 200  # kept small so the suite stays fast; parity is seed-exact either way

_rng = random.Random(SEED)
ROWS_A = [round(_rng.random(), 6) for _ in range(40)]
ROWS_B = [round(_rng.random(), 6) for _ in range(40)]
CLUSTERS = [f"patient_{i % 8:02d}" for i in range(40)]
LABELS = [bool(_rng.random() < 0.4) for _ in range(40)]
# Signal strengths are deliberately weak enough that the classes OVERLAP: a
# perfectly separating fixture saturates binary_auc at 1.0 and would hide a
# midrank bug in the ported estimator.
SCORES_A = [round(0.35 * label + 0.65 * _rng.random(), 6) for label in LABELS]
SCORES_B = [round(0.15 * label + 0.85 * _rng.random(), 6) for label in LABELS]
A_CORRECT = [bool(_rng.random() < 0.7) for _ in range(40)]
B_CORRECT = [bool(_rng.random() < 0.5) for _ in range(40)]


def mean(rows):
    """Shared metric callable: identical object for both modules, so any
    difference observed must originate in the module under test."""
    values = [float(r) for r in rows]
    return sum(values) / len(values) if values else 0.0


class StatisticsSurfaceParityTest(unittest.TestCase):
    def test_every_ported_function_exists_in_both(self):
        for name in PORTED:
            self.assertTrue(hasattr(v1, name), f"v1 lost {name}")
            self.assertTrue(hasattr(v2, name), f"v2 is missing {name}")

    def test_signatures_are_identical(self):
        for name in PORTED:
            self.assertEqual(
                str(inspect.signature(getattr(v1, name))),
                str(inspect.signature(getattr(v2, name))),
                f"signature drift in {name}",
            )

    def test_default_bootstrap_seed_is_preserved(self):
        """The default seed is part of the frozen protocol, not an implementation
        detail: changing it would silently move every published CI."""
        for name in ("clustered_bootstrap_distribution", "clustered_bootstrap_ci",
                     "paired_clustered_bootstrap_difference"):
            left = inspect.signature(getattr(v1, name)).parameters["seed"].default
            right = inspect.signature(getattr(v2, name)).parameters["seed"].default
            self.assertEqual(left, right, f"seed default drift in {name}")

    def test_v2_statistics_is_not_v1(self):
        self.assertTrue(v2.__name__.startswith("medguard_v2."))
        self.assertIsNot(v1, v2)
        self.assertIsNot(v1.binary_auc, v2.binary_auc)


class BootstrapParityTest(unittest.TestCase):
    def test_clustered_bootstrap_distribution(self):
        left, left_samples = v1.clustered_bootstrap_distribution(
            ROWS_A, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
        right, right_samples = v2.clustered_bootstrap_distribution(
            ROWS_A, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
        self.assertEqual(left.estimate, right.estimate)
        self.assertEqual(left.low, right.low)
        self.assertEqual(left.high, right.high)
        self.assertEqual(left_samples, right_samples)
        self.assertEqual(len(right_samples), N_BOOT)

    def test_clustered_bootstrap_distribution_across_seeds_and_confidence(self):
        for seed in (1, 42, SEED):
            for confidence in (0.90, 0.95, 0.99):
                left, left_samples = v1.clustered_bootstrap_distribution(
                    ROWS_A, CLUSTERS, mean, n_boot=N_BOOT,
                    confidence=confidence, seed=seed)
                right, right_samples = v2.clustered_bootstrap_distribution(
                    ROWS_A, CLUSTERS, mean, n_boot=N_BOOT,
                    confidence=confidence, seed=seed)
                self.assertEqual((left.estimate, left.low, left.high),
                                 (right.estimate, right.low, right.high),
                                 f"CI drift at seed={seed}, confidence={confidence}")
                self.assertEqual(left_samples, right_samples)

    def test_clustered_bootstrap_ci(self):
        left = v1.clustered_bootstrap_ci(ROWS_A, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
        right = v2.clustered_bootstrap_ci(ROWS_A, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
        self.assertEqual((left.estimate, left.low, left.high),
                         (right.estimate, right.low, right.high))

    def test_clustered_bootstrap_ci_matches_its_own_distribution_in_both(self):
        """v1 documents that the CI is byte-identical to the distribution helper;
        the port must keep that guarantee, not merely agree with v1's CI."""
        for module in (v1, v2):
            interval = module.clustered_bootstrap_ci(
                ROWS_A, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
            paired, _ = module.clustered_bootstrap_distribution(
                ROWS_A, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
            self.assertEqual((interval.estimate, interval.low, interval.high),
                             (paired.estimate, paired.low, paired.high))

    def test_clustered_bootstrap_validation_parity(self):
        for module in (v1, v2):
            with self.assertRaises(ValueError):
                module.clustered_bootstrap_ci(ROWS_A, CLUSTERS[:-1], mean, n_boot=N_BOOT)
            with self.assertRaises(ValueError):
                module.clustered_bootstrap_ci([], [], mean, n_boot=N_BOOT)
            with self.assertRaises(ValueError):
                module.clustered_bootstrap_ci(ROWS_A, CLUSTERS, mean, n_boot=0)

    def test_bootstrap_two_sided_p(self):
        _, samples = v1.clustered_bootstrap_distribution(
            ROWS_A, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
        for null_value in (0.0, 0.5, 0.6, 1.0):
            self.assertEqual(
                v1.bootstrap_two_sided_p(samples, null_value),
                v2.bootstrap_two_sided_p(samples, null_value),
                f"bootstrap p drift at null={null_value}",
            )

    def test_bootstrap_two_sided_p_never_returns_zero_in_either_module(self):
        samples = [1.0] * 50
        self.assertEqual(v1.bootstrap_two_sided_p(samples, 0.0),
                         v2.bootstrap_two_sided_p(samples, 0.0))
        self.assertGreater(v2.bootstrap_two_sided_p(samples, 0.0), 0.0)

    def test_bootstrap_two_sided_p_rejects_empty_in_both(self):
        for module in (v1, v2):
            with self.assertRaises(ValueError):
                module.bootstrap_two_sided_p([])

    def test_paired_clustered_bootstrap_difference(self):
        left = v1.paired_clustered_bootstrap_difference(
            ROWS_A, ROWS_B, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
        right = v2.paired_clustered_bootstrap_difference(
            ROWS_A, ROWS_B, CLUSTERS, mean, n_boot=N_BOOT, seed=SEED)
        self.assertEqual((left.estimate, left.low, left.high),
                         (right.estimate, right.low, right.high))

    def test_paired_clustered_bootstrap_validation_parity(self):
        for module in (v1, v2):
            with self.assertRaises(ValueError):
                module.paired_clustered_bootstrap_difference(
                    ROWS_A, ROWS_B[:-1], CLUSTERS, mean, n_boot=N_BOOT)
            with self.assertRaises(ValueError):
                module.paired_clustered_bootstrap_difference(
                    ROWS_A, ROWS_B, CLUSTERS[:-1], mean, n_boot=N_BOOT)


class PairedTestParityTest(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertEqual(v1.mcnemar_exact(A_CORRECT, B_CORRECT),
                         v2.mcnemar_exact(A_CORRECT, B_CORRECT))

    def test_mcnemar_exact_no_discordant_pairs(self):
        same = [True, False, True, False]
        self.assertEqual(v1.mcnemar_exact(same, same), v2.mcnemar_exact(same, same))

    def test_mcnemar_exact_length_mismatch_parity(self):
        for module in (v1, v2):
            with self.assertRaises(ValueError):
                module.mcnemar_exact([True], [True, False])

    def test_holm_adjust(self):
        families = [
            {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.7},
            {"only": 0.03},
            {"tied_a": 0.05, "tied_b": 0.05, "tied_c": 0.05},
        ]
        for family in families:
            self.assertEqual(v1.holm_adjust(family), v2.holm_adjust(family))

    def test_holm_adjust_empty_family(self):
        self.assertEqual(v1.holm_adjust({}), v2.holm_adjust({}))


class AucParityTest(unittest.TestCase):
    def test_binary_auc(self):
        observed = v2.binary_auc(LABELS, SCORES_A)
        self.assertEqual(v1.binary_auc(LABELS, SCORES_A), observed)
        self.assertEqual(v1.binary_auc(LABELS, SCORES_B), v2.binary_auc(LABELS, SCORES_B))
        # Non-vacuity: a saturated AUC of exactly 1.0 or 0.5 is a weak witness.
        self.assertGreater(observed, 0.5)
        self.assertLess(observed, 1.0)

    def test_binary_auc_with_ties_uses_the_same_midranks(self):
        labels = [True, True, False, False]
        tied = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(v1.binary_auc(labels, tied), v2.binary_auc(labels, tied))

    def test_binary_auc_validation_parity(self):
        for module in (v1, v2):
            with self.assertRaises(ValueError):
                module.binary_auc([True, False], [0.1])
            with self.assertRaises(ValueError):
                module.binary_auc([True, True], [0.1, 0.2])

    def test_delong_auc_test(self):
        observed = v2.delong_auc_test(LABELS, SCORES_A, SCORES_B)
        self.assertEqual(v1.delong_auc_test(LABELS, SCORES_A, SCORES_B), observed)
        # Non-vacuity: the fixture must reach the finite-variance branch, not the
        # degenerate one that is asserted separately below.
        self.assertGreater(observed["variance"], 0.0)
        self.assertNotEqual(observed["difference"], 0.0)

    def test_delong_auc_test_is_antisymmetric_identically(self):
        left = v1.delong_auc_test(LABELS, SCORES_B, SCORES_A)
        right = v2.delong_auc_test(LABELS, SCORES_B, SCORES_A)
        self.assertEqual(left, right)

    def test_delong_auc_test_zero_variance_branch(self):
        """Identical score vectors drive variance to 0; both must take the same
        degenerate branch rather than one raising and the other returning."""
        self.assertEqual(v1.delong_auc_test(LABELS, SCORES_A, SCORES_A),
                         v2.delong_auc_test(LABELS, SCORES_A, SCORES_A))

    def test_delong_auc_test_validation_parity(self):
        for module in (v1, v2):
            with self.assertRaises(ValueError):
                module.delong_auc_test(LABELS, SCORES_A, SCORES_B[:-1])
            with self.assertRaises(ValueError):
                module.delong_auc_test([True, True], [0.1, 0.2], [0.3, 0.4])


if __name__ == "__main__":
    unittest.main()
