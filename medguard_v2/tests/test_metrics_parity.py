"""Parity: the ported v2 metrics must be indistinguishable from v1.

Every assertion runs the SAME deterministic fixture through both modules and
demands byte-equal output.  Fixtures are constructed here; no sealed artifact is
read, so this suite is safe to run at any time and cannot perturb a frozen run.
"""
from __future__ import annotations

import inspect
import unittest

from medguard.config import MALIGNANT_LABELS as V1_MALIGNANT
from medguard.eval import metrics as v1
from medguard_v2.config import (
    DIAGNOSIS_LABELS as V2_DIAGNOSIS,
    LABEL_FULL_NAMES as V2_FULL_NAMES,
    MALIGNANT_LABELS as V2_MALIGNANT,
)
from medguard.config import DIAGNOSIS_LABELS as V1_DIAGNOSIS
from medguard.config import LABEL_FULL_NAMES as V1_FULL_NAMES
from medguard_v2.eval import metrics as v2


PORTED = (
    "confusion_matrix", "row_normalize", "classification_summary", "ece",
    "risk_coverage_curve", "aurc", "harm_coverage_curve", "harm_coverage_auc",
    "matched_harm_coverage_auc", "selective_accuracy_at_coverage",
    "over_reliance", "copy_reference_ratio", "completeness", "acuity_ordinal",
    "quadratic_weighted_kappa", "triage_rates", "is_actionable",
    "actionable_confusion", "sensitivity_at_fixed_specificity",
    "grounding_score", "faithfulness",
)

# --- deterministic fixtures ------------------------------------------------
TRUE_LABELS = [
    "melanoma", "melanoma", "melanoma", "bcc", "bcc", "akiec",
    "nevus", "nevus", "nevus", "nevus", "bkl", "df", "vasc", "bkl",
]
PRED_LABELS = [
    "melanoma", "nevus", "melanoma", "bcc", "melanoma", "bkl",
    "nevus", "nevus", "melanoma", "bkl", "bkl", "df", "vasc", "nevus",
]
CORRECT = [t == p for t, p in zip(TRUE_LABELS, PRED_LABELS)]
CONF = [0.97, 0.55, 0.88, 0.91, 0.62, 0.40, 0.95, 0.73, 0.51,
        0.66, 0.84, 0.79, 0.99, 0.31]
HARM = [0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.5]
ANSWERED = [True, True, True, True, True, False, True, True, True,
            True, True, False, True, True]
SCORES = [0.93, 0.44, 0.81, 0.88, 0.71, 0.58, 0.12, 0.22, 0.64,
          0.35, 0.29, 0.18, 0.07, 0.41]
# Threshold chosen so the confident-and-wrong count is strictly positive: at 0.7
# this fixture yields over_reliance == 0.0, which a broken port returning a
# constant zero would pass. See test_over_reliance's non-vacuity assertion.
CONFIDENT = [c >= 0.5 for c in CONF]
ACUITY_TRUE = [0 if t in ("nevus", "bkl", "df", "vasc") else (1 if t == "akiec" else 2)
               for t in TRUE_LABELS]
ACUITY_PRED = [0 if p in ("nevus", "bkl", "df", "vasc") else (1 if p == "akiec" else 2)
               for p in PRED_LABELS]

CHUNKS = [
    {"text": "Melanoma requires urgent excision with wide local margins."},
    {"text": "Basal cell carcinoma is treated with Mohs micrographic surgery."},
]
ANSWERS = [
    "Melanoma requires urgent excision with wide local margins [1].",
    "This lesion appears benign and needs no intervention.",
    "",
]
CHUNKS_LIST = [CHUNKS, CHUNKS, CHUNKS]


class MetricsSurfaceParityTest(unittest.TestCase):
    """The port must expose the same names with the same signatures."""

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

    def test_public_surface_has_not_grown_or_shrunk(self):
        V2_ADDITIONS = {"reliability_diagram"}
        def public(module):
            return {n for n, o in vars(module).items()
                    if not n.startswith("_") and inspect.isfunction(o)
                    and o.__module__ == module.__name__}
        self.assertEqual(public(v1), public(v2) - V2_ADDITIONS)

    def test_class_order_and_label_vocabulary_match(self):
        self.assertEqual(v1.DERM_CLASS_ORDER, v2.DERM_CLASS_ORDER)
        self.assertEqual(V1_MALIGNANT, V2_MALIGNANT)
        self.assertEqual(V1_DIAGNOSIS, V2_DIAGNOSIS)
        self.assertEqual(V1_FULL_NAMES, V2_FULL_NAMES)

    def test_v2_metrics_do_not_import_v1(self):
        """A silent fallback to v1 would make every parity check vacuous."""
        self.assertTrue(v2.__name__.startswith("medguard_v2."))
        self.assertIsNot(v1, v2)
        self.assertIsNot(v1.confusion_matrix, v2.confusion_matrix)


class DiagnosticParityTest(unittest.TestCase):
    def test_confusion_matrix(self):
        self.assertEqual(
            v1.confusion_matrix(TRUE_LABELS, PRED_LABELS),
            v2.confusion_matrix(TRUE_LABELS, PRED_LABELS),
        )

    def test_confusion_matrix_with_custom_label_subset(self):
        labels = ("melanoma", "bcc", "nevus")
        truth = ["melanoma", "melanoma", "bcc", "nevus"]
        pred = ["melanoma", "nevus", "bcc", "nevus"]
        self.assertEqual(
            v1.confusion_matrix(truth, pred, labels),
            v2.confusion_matrix(truth, pred, labels),
        )

    def test_confusion_matrix_rejects_the_same_inputs(self):
        bad_cases = [
            (["melanoma"], ["not_a_label"], v1.DERM_CLASS_ORDER),
            (["melanoma"], [], v1.DERM_CLASS_ORDER),
            (["melanoma"], ["melanoma"], ()),
            (["melanoma"], ["melanoma"], ("melanoma", "melanoma")),
        ]
        for truth, pred, labels in bad_cases:
            with self.assertRaises(ValueError):
                v1.confusion_matrix(truth, pred, labels)
            with self.assertRaises(ValueError):
                v2.confusion_matrix(truth, pred, labels)

    def test_row_normalize(self):
        matrix = v1.confusion_matrix(TRUE_LABELS, PRED_LABELS)
        self.assertEqual(v1.row_normalize(matrix), v2.row_normalize(matrix))

    def test_classification_summary(self):
        self.assertEqual(
            v1.classification_summary(TRUE_LABELS, PRED_LABELS),
            v2.classification_summary(TRUE_LABELS, PRED_LABELS),
        )

    def test_classification_summary_with_explicit_labels(self):
        labels = ("melanoma", "bcc", "akiec", "nevus", "bkl", "df", "vasc")
        self.assertEqual(
            v1.classification_summary(TRUE_LABELS, PRED_LABELS, labels),
            v2.classification_summary(TRUE_LABELS, PRED_LABELS, labels),
        )

    def test_classification_summary_rejects_empty(self):
        with self.assertRaises(ValueError):
            v1.classification_summary([], [])
        with self.assertRaises(ValueError):
            v2.classification_summary([], [])


class SelectivePredictionParityTest(unittest.TestCase):
    def test_ece_across_bin_counts(self):
        for n_bins in (2, 5, 10, 20):
            self.assertEqual(
                v1.ece(CONF, CORRECT, n_bins), v2.ece(CONF, CORRECT, n_bins),
                f"ece drift at n_bins={n_bins}",
            )

    def test_ece_on_empty_input(self):
        self.assertEqual(v1.ece([], []), v2.ece([], []))

    def test_risk_coverage_curve(self):
        self.assertEqual(
            v1.risk_coverage_curve(CONF, CORRECT), v2.risk_coverage_curve(CONF, CORRECT),
        )

    def test_aurc(self):
        self.assertEqual(v1.aurc(CONF, CORRECT), v2.aurc(CONF, CORRECT))
        self.assertEqual(v1.aurc([], []), v2.aurc([], []))

    def test_selective_accuracy_at_coverage(self):
        for coverage in (0.25, 0.5, 0.65, 1.0):
            self.assertEqual(
                v1.selective_accuracy_at_coverage(CONF, CORRECT, coverage),
                v2.selective_accuracy_at_coverage(CONF, CORRECT, coverage),
                f"selective accuracy drift at coverage={coverage}",
            )

    def test_harm_coverage_curve(self):
        self.assertEqual(
            v1.harm_coverage_curve(CONF, HARM), v2.harm_coverage_curve(CONF, HARM),
        )

    def test_harm_coverage_auc(self):
        self.assertEqual(
            v1.harm_coverage_auc(CONF, HARM), v2.harm_coverage_auc(CONF, HARM),
        )
        self.assertEqual(v1.harm_coverage_auc([], []), v2.harm_coverage_auc([], []))

    def test_harm_coverage_rejects_negative_harm_in_both(self):
        with self.assertRaises(ValueError):
            v1.harm_coverage_auc([0.5, 0.5], [1.0, -1.0])
        with self.assertRaises(ValueError):
            v2.harm_coverage_auc([0.5, 0.5], [1.0, -1.0])

    def test_matched_harm_coverage_auc(self):
        observed = []
        for coverage in (0.25, 0.5, 0.75):
            left = v1.matched_harm_coverage_auc(
                CONF, HARM, ANSWERED, total_cases=len(CONF), max_coverage=coverage)
            right = v2.matched_harm_coverage_auc(
                CONF, HARM, ANSWERED, total_cases=len(CONF), max_coverage=coverage)
            self.assertEqual(left, right, f"matched harm AUC drift at coverage={coverage}")
            observed.append(right)
        # Non-vacuity: the low-coverage prefixes are all-zero harm by construction,
        # so at least one operating point must produce a positive AUC.
        self.assertTrue(any(value for value in observed), observed)

    def test_matched_harm_coverage_auc_returns_none_identically(self):
        """Insufficient coverage must yield None, not a coerced number."""
        self.assertIsNone(v1.matched_harm_coverage_auc(
            CONF, HARM, ANSWERED, total_cases=len(CONF), max_coverage=1.0))
        self.assertIsNone(v2.matched_harm_coverage_auc(
            CONF, HARM, ANSWERED, total_cases=len(CONF), max_coverage=1.0))

    def test_matched_harm_coverage_auc_validation_parity(self):
        kwargs = [
            dict(total_cases=len(CONF) - 1, max_coverage=0.5),
            dict(total_cases=len(CONF), max_coverage=0.0),
            dict(total_cases=len(CONF), max_coverage=1.5),
        ]
        for kw in kwargs:
            with self.assertRaises(ValueError):
                v1.matched_harm_coverage_auc(CONF, HARM, ANSWERED, **kw)
            with self.assertRaises(ValueError):
                v2.matched_harm_coverage_auc(CONF, HARM, ANSWERED, **kw)


class EvidenceParityTest(unittest.TestCase):
    def test_grounding_score(self):
        for answer in ANSWERS:
            self.assertEqual(
                v1.grounding_score(answer, CHUNKS), v2.grounding_score(answer, CHUNKS),
            )

    def test_faithfulness(self):
        self.assertEqual(
            v1.faithfulness(ANSWERS, CHUNKS_LIST), v2.faithfulness(ANSWERS, CHUNKS_LIST),
        )

    def test_over_reliance(self):
        observed = v2.over_reliance(CONFIDENT, CORRECT)
        self.assertEqual(v1.over_reliance(CONFIDENT, CORRECT), observed)
        # Non-vacuity: a fixture with no confident-and-wrong case would let a port
        # that always returns 0.0 pass this comparison.
        self.assertGreater(observed, 0.0)
        self.assertEqual(v1.over_reliance([], []), v2.over_reliance([], []))

    def test_copy_reference_ratio(self):
        for ngram in (3, 5, 8):
            for answer in ANSWERS:
                self.assertEqual(
                    v1.copy_reference_ratio(answer, CHUNKS, ngram),
                    v2.copy_reference_ratio(answer, CHUNKS, ngram),
                    f"copy-reference drift at ngram={ngram}",
                )

    def test_completeness(self):
        for answer in ANSWERS:
            self.assertEqual(
                v1.completeness(answer, CHUNKS), v2.completeness(answer, CHUNKS),
            )
        self.assertEqual(v1.completeness("anything", []), v2.completeness("anything", []))


class ClinicalParityTest(unittest.TestCase):
    def test_acuity_ordinal(self):
        for label in list(V1_DIAGNOSIS) + ["", "unknown"]:
            self.assertEqual(v1.acuity_ordinal(label), v2.acuity_ordinal(label))

    def test_quadratic_weighted_kappa(self):
        self.assertEqual(
            v1.quadratic_weighted_kappa(ACUITY_TRUE, ACUITY_PRED),
            v2.quadratic_weighted_kappa(ACUITY_TRUE, ACUITY_PRED),
        )

    def test_quadratic_weighted_kappa_degenerate_cases(self):
        self.assertEqual(v1.quadratic_weighted_kappa([], []),
                         v2.quadratic_weighted_kappa([], []))
        # Perfect agreement on a single class drives the denominator to zero.
        self.assertEqual(v1.quadratic_weighted_kappa([1, 1], [1, 1]),
                         v2.quadratic_weighted_kappa([1, 1], [1, 1]))

    def test_triage_rates(self):
        self.assertEqual(
            v1.triage_rates(PRED_LABELS, TRUE_LABELS),
            v2.triage_rates(PRED_LABELS, TRUE_LABELS),
        )

    def test_triage_rates_with_no_malignant_truth(self):
        benign = ["nevus", "bkl", "df"]
        self.assertEqual(v1.triage_rates(benign, benign), v2.triage_rates(benign, benign))

    def test_is_actionable(self):
        for label in list(V1_DIAGNOSIS) + ["", "scc"]:
            self.assertEqual(v1.is_actionable(label), v2.is_actionable(label))
            extended = sorted(set(V1_MALIGNANT) | {"scc"})
            self.assertEqual(v1.is_actionable(label, extended),
                             v2.is_actionable(label, extended))

    def test_actionable_confusion(self):
        self.assertEqual(
            v1.actionable_confusion(PRED_LABELS, TRUE_LABELS),
            v2.actionable_confusion(PRED_LABELS, TRUE_LABELS),
        )

    def test_actionable_confusion_with_scores_uses_ported_binary_auc(self):
        """Exercises the in-function `from .statistics import binary_auc` hop."""
        left = v1.actionable_confusion(PRED_LABELS, TRUE_LABELS, scores=SCORES)
        right = v2.actionable_confusion(PRED_LABELS, TRUE_LABELS, scores=SCORES)
        self.assertEqual(left, right)
        self.assertIsNotNone(right["auroc"])

    def test_actionable_confusion_reports_undefined_rates_as_none(self):
        benign = ["nevus", "bkl"]
        left = v1.actionable_confusion(benign, benign)
        right = v2.actionable_confusion(benign, benign)
        self.assertEqual(left, right)
        self.assertIsNone(right["sensitivity"])

    def test_actionable_confusion_validation_parity(self):
        with self.assertRaises(ValueError):
            v1.actionable_confusion(["nevus"], ["nevus", "bcc"])
        with self.assertRaises(ValueError):
            v2.actionable_confusion(["nevus"], ["nevus", "bcc"])
        with self.assertRaises(ValueError):
            v1.actionable_confusion(["nevus"], ["nevus"], positive_labels=[])
        with self.assertRaises(ValueError):
            v2.actionable_confusion(["nevus"], ["nevus"], positive_labels=[])

    def test_sensitivity_at_fixed_specificity(self):
        truth = [v1.is_actionable(label) for label in TRUE_LABELS]
        for target in (0.0, 0.5, 0.8, 0.9, 1.0):
            self.assertEqual(
                v1.sensitivity_at_fixed_specificity(truth, SCORES, target),
                v2.sensitivity_at_fixed_specificity(truth, SCORES, target),
                f"operating-point drift at target specificity={target}",
            )

    def test_sensitivity_at_fixed_specificity_single_class_returns_none(self):
        truth = [False, False, False]
        self.assertIsNone(v1.sensitivity_at_fixed_specificity(truth, [0.1, 0.2, 0.3], 0.5))
        self.assertIsNone(v2.sensitivity_at_fixed_specificity(truth, [0.1, 0.2, 0.3], 0.5))

    def test_sensitivity_at_fixed_specificity_validation_parity(self):
        truth = [True, False]
        for bad in ([0.1], [0.1, 0.2, 0.3]):
            with self.assertRaises(ValueError):
                v1.sensitivity_at_fixed_specificity(truth, bad, 0.5)
            with self.assertRaises(ValueError):
                v2.sensitivity_at_fixed_specificity(truth, bad, 0.5)
        for bad_target in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                v1.sensitivity_at_fixed_specificity(truth, [0.1, 0.2], bad_target)
            with self.assertRaises(ValueError):
                v2.sensitivity_at_fixed_specificity(truth, [0.1, 0.2], bad_target)


if __name__ == "__main__":
    unittest.main()
