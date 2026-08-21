"""Regression tests for the DermLIP probe baseline.

The first version of this probe selected its regularisation strength using ``sklearn.metrics.
log_loss``, passing the canonical Derm7pt label order.  ``log_loss`` re-sorts its ``labels``
argument lexicographically **without** reordering the probability columns, so truth was silently
mis-aligned against predictions.  Selection then picked the most-regularised candidate and
diagnosis collapsed to the majority class at balanced accuracy exactly ``0.2000`` -- a number that
looks like a plausible weak result rather than a broken one, which is what made it dangerous.

The bug produced no exception and no NaN.  Only the value gave it away.  These tests exist so it
cannot come back silently: they pin the failure signature itself, not just the current numbers.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "research" / "evidence" / "derm7pt_v1" / "perception_baseline.json"

#: 1 / n_classes. A multiclass probe that lands exactly here is predicting one class for
#: everything, which is the collapse signature.
MAJORITY_COLLAPSE_BALANCED_ACCURACY = 0.2000


@unittest.skipUnless(ARTIFACT.exists(), "perception baseline not built")
class PerceptionBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    # --- the regression itself -------------------------------------------------

    def test_diagnosis_did_not_collapse_to_the_majority_class(self):
        balanced = self.artifact["metrics"]["diagnosis"]["valid"]["balanced_accuracy"]
        self.assertNotAlmostEqual(
            balanced,
            MAJORITY_COLLAPSE_BALANCED_ACCURACY,
            places=4,
            msg=(
                "diagnosis balanced accuracy is exactly 1/n_classes, the signature of a probe "
                "predicting one class for every case. This is what the log_loss label-ordering "
                "bug produced. Check that C selection is not scoring mis-aligned columns."
            ),
        )
        self.assertGreater(balanced, 0.40, "diagnosis probe is barely above collapse")

    def test_the_probe_predicts_more_than_one_class(self):
        matrix = self.artifact["metrics"]["diagnosis"]["valid"]["confusion_matrix"]
        predicted_per_column = [sum(row[i] for row in matrix) for i in range(len(matrix))]
        used = sum(1 for total in predicted_per_column if total > 0)
        self.assertGreater(
            used, 1, f"probe emitted only {used} distinct class(es) across the valid split"
        )

    def test_selection_did_not_pin_the_most_regularised_extreme(self):
        # Not proof of correctness, but the broken run chose exactly this, and a correct run on
        # L2-normalised CLIP features should not: they need weak regularisation.
        grid = self.artifact["probe"]["C_grid"]
        for target in ("diagnosis", "management"):
            selected = self.artifact["probe"]["targets"][target]["selected_C"]
            self.assertNotEqual(
                float(selected),
                float(min(grid)),
                f"{target}: C selection landed on the strongest regularisation in the grid "
                f"({min(grid)}), which is what the mis-aligned log_loss produced",
            )

    def test_temperature_scaling_did_not_make_calibration_worse(self):
        for target in ("diagnosis", "management"):
            cv = self.artifact["probe"]["targets"][target]["cross_validation"]
            self.assertLessEqual(
                cv["oof_nll_after_temperature"],
                cv["oof_nll_before_temperature"] + 1e-9,
                f"{target}: temperature scaling increased out-of-fold NLL",
            )

    # --- leakage and shape -----------------------------------------------------

    def test_probe_was_fit_on_train_only(self):
        self.assertEqual(self.artifact["probe"]["fit_split"], "train")
        self.assertEqual(self.artifact["probe"]["n_fit_cases"], 413)

        fit_case_nums = set(self.artifact["probe"]["fit_case_nums"])
        non_train = [
            row["case_num"]
            for row in self.artifact["cases"]
            if row["split"] != "train" and row["case_num"] in fit_case_nums
        ]
        self.assertEqual(non_train, [], f"non-train cases appear in the fit set: {non_train[:5]}")

    def test_every_case_has_a_prediction(self):
        self.assertEqual(len(self.artifact["cases"]), 1011)

    def test_posteriors_are_distributions(self):
        for row in self.artifact["cases"]:
            for target in ("diagnosis", "management"):
                posterior = row[target]["posterior"]
                total = sum(posterior.values())
                self.assertTrue(
                    math.isclose(total, 1.0, abs_tol=1e-3),
                    f"case {row['case_num']} {target} posterior sums to {total}",
                )
                self.assertTrue(all(v >= 0.0 for v in posterior.values()))

    def test_prediction_is_the_posterior_argmax(self):
        for row in self.artifact["cases"]:
            for target in ("diagnosis", "management"):
                posterior = row[target]["posterior"]
                argmax = max(posterior, key=posterior.get)
                self.assertEqual(
                    row[target]["prediction"],
                    argmax,
                    f"case {row['case_num']} {target}: prediction disagrees with its posterior",
                )

    def test_label_order_is_canonical_not_lexicographic(self):
        # The bug was an ordering mismatch; assert the artifact records the canonical order so a
        # future reader cannot assume sklearn's lexicographic default.
        from medguard_v2.data.derm7pt import DERM7PT_CLASS_ORDER

        labels = self.artifact["metrics"]["diagnosis"]["valid"]["labels"]
        self.assertEqual(tuple(labels), tuple(DERM7PT_CLASS_ORDER))
        self.assertNotEqual(
            tuple(labels), tuple(sorted(labels)), "canonical order should not be lexicographic"
        )


class LogLossOrderingHazardTest(unittest.TestCase):
    """Demonstrate the sklearn behaviour that caused the bug, so the reason is executable.

    If a future sklearn makes ``log_loss`` respect the caller's label order, this test fails and
    the guard comment in ``probe.py`` can be revisited.
    """

    def test_sklearn_log_loss_ignores_a_non_lexicographic_label_order(self):
        from sklearn.metrics import log_loss

        labels_canonical = ["BCC", "NEV", "MEL"]
        self.assertNotEqual(labels_canonical, sorted(labels_canonical))

        # Columns follow labels_canonical: case 1 is confidently BCC, case 2 confidently MEL.
        probabilities = [[0.90, 0.05, 0.05], [0.05, 0.05, 0.90]]
        truth = ["BCC", "MEL"]

        as_given = log_loss(truth, probabilities, labels=labels_canonical)

        # Reordered so the columns match lexicographic order: BCC, MEL, NEV.
        order = [labels_canonical.index(name) for name in sorted(labels_canonical)]
        reordered = [[row[i] for i in order] for row in probabilities]
        as_aligned = log_loss(truth, reordered, labels=sorted(labels_canonical))

        self.assertLess(as_aligned, 0.2, "aligned call should show a confident, low loss")
        self.assertGreater(
            as_given,
            as_aligned,
            "if these now agree, sklearn honours caller label order and probe.py can be revisited",
        )


if __name__ == "__main__":
    unittest.main()
