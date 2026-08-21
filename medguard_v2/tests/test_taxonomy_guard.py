"""The taxonomy guard must make every mis-pairing loud and legible.

Two incompatible label sets live in this package (HAM10000's 7 classes, Derm7pt's
5 groups).  These tests pin (a) that each mis-pairing raises, (b) that the error
names the taxonomy that would have worked, and (c) the two ways the mis-pairing
is *not* self-announcing in ``classification_summary`` -- which is the whole
reason the guard exists.
"""
from __future__ import annotations

import unittest

from medguard_v2.config import DIAGNOSIS_LABELS
from medguard_v2.data.derm7pt import DERM7PT_CLASS_ORDER, DIAGNOSIS_GROUP_ORDER
from medguard_v2.eval.metrics import DERM_CLASS_ORDER, classification_summary
from medguard_v2.eval.taxonomy import (
    HAM10000_CLASS_ORDER,
    TAXONOMIES,
    TaxonomyError,
    assert_labels_match,
    resolve_labels,
)


# A Derm7pt cohort and a HAM10000 cohort, each covering every class of its own
# taxonomy so no mis-pairing can pass by accident of coverage.
DERM7PT_TRUE = ["MEL", "NEV", "BCC", "MISC", "SK", "NEV"]
DERM7PT_PRED = ["MEL", "NEV", "MEL", "MISC", "SK", "NEV"]
HAM_TRUE = ["melanoma", "nevus", "bcc", "akiec", "bkl", "df", "vasc"]
HAM_PRED = ["melanoma", "nevus", "bcc", "bkl", "bkl", "df", "vasc"]


class TaxonomyResolutionTest(unittest.TestCase):
    def test_resolve_labels_returns_the_five_derm7pt_groups_in_canonical_order(self):
        self.assertEqual(resolve_labels("derm7pt"), ("BCC", "NEV", "MEL", "SK", "MISC"))
        self.assertIs(resolve_labels("derm7pt"), DERM7PT_CLASS_ORDER)
        self.assertIs(DIAGNOSIS_GROUP_ORDER, DERM7PT_CLASS_ORDER)

    def test_resolve_labels_ham10000_is_the_metrics_default_order(self):
        """A guard that reordered the columns would silently break v1 comparability."""
        self.assertIs(resolve_labels("ham10000"), DERM_CLASS_ORDER)
        self.assertEqual(HAM10000_CLASS_ORDER, DERM_CLASS_ORDER)

    def test_ham10000_order_is_a_permutation_of_the_v1_vocabulary(self):
        self.assertEqual(set(HAM10000_CLASS_ORDER), set(DIAGNOSIS_LABELS))
        self.assertEqual(len(HAM10000_CLASS_ORDER), len(DIAGNOSIS_LABELS))

    def test_the_two_taxonomies_are_different_sizes_and_share_only_bcc(self):
        """Non-vacuity: the overlap is exactly one token, which is why case 2 below
        is silent rather than loud."""
        ham = {label.lower() for label in HAM10000_CLASS_ORDER}
        derm7pt = {label.lower() for label in DERM7PT_CLASS_ORDER}
        self.assertEqual(len(ham), 7)
        self.assertEqual(len(derm7pt), 5)
        self.assertEqual(ham & derm7pt, {"bcc"})

    def test_resolve_labels_tolerates_case_and_whitespace(self):
        self.assertEqual(resolve_labels("  DERM7PT "), DERM7PT_CLASS_ORDER)
        self.assertEqual(resolve_labels("HAM10000"), HAM10000_CLASS_ORDER)

    def test_unknown_taxonomy_name_raises_and_lists_the_registered_ones(self):
        for name in ("ham", "derm7pt_v2", "", "isic"):
            with self.assertRaises(TaxonomyError) as ctx:
                resolve_labels(name)
            message = str(ctx.exception)
            self.assertIn("unknown taxonomy", message)
            for known in TAXONOMIES:
                self.assertIn(repr(known), message)

    def test_taxonomy_error_is_a_value_error(self):
        """Callers already guarding classification_summary's ValueError keep working."""
        self.assertTrue(issubclass(TaxonomyError, ValueError))


class MisPairingTest(unittest.TestCase):
    def test_ham_labels_with_derm7pt_data_raises(self):
        """The declared taxonomy is HAM10000; the data is Derm7pt."""
        with self.assertRaises(TaxonomyError) as ctx:
            assert_labels_match(DERM7PT_TRUE, "ham10000")
        message = str(ctx.exception)
        self.assertIn("'ham10000'", message)
        self.assertIn("'MEL'", message)
        self.assertIn("taxonomy='derm7pt'", message)

    def test_ham_label_sequence_declared_as_derm7pt_raises(self):
        """Same defect from the other side: the ``labels`` argument itself is wrong."""
        with self.assertRaises(TaxonomyError) as ctx:
            assert_labels_match(DIAGNOSIS_LABELS, "derm7pt")
        self.assertIn("taxonomy='ham10000'", str(ctx.exception))

    def test_derm7pt_labels_with_ham_data_raises(self):
        with self.assertRaises(TaxonomyError) as ctx:
            assert_labels_match(HAM_TRUE, "derm7pt")
        message = str(ctx.exception)
        self.assertIn("'derm7pt'", message)
        self.assertIn("'melanoma'", message)
        self.assertIn("taxonomy='ham10000'", message)

    def test_derm7pt_label_sequence_declared_as_ham_raises(self):
        with self.assertRaises(TaxonomyError) as ctx:
            assert_labels_match(DERM7PT_CLASS_ORDER, "ham10000")
        self.assertIn("taxonomy='derm7pt'", str(ctx.exception))

    def test_an_unknown_label_raises_under_either_taxonomy(self):
        for taxonomy, cohort in (
            ("derm7pt", ["MEL", "NEV", "SCC"]),
            ("ham10000", ["melanoma", "nevus", "scc"]),
        ):
            with self.assertRaises(TaxonomyError) as ctx:
                assert_labels_match(cohort, taxonomy)
            message = str(ctx.exception)
            self.assertIn("no registered taxonomy", message)
            self.assertIn("scc", message.lower())

    def test_a_single_stray_label_in_an_otherwise_valid_cohort_raises(self):
        """One bad value must not be diluted by five good ones."""
        with self.assertRaises(TaxonomyError) as ctx:
            assert_labels_match(DERM7PT_TRUE + ["melanoma"], "derm7pt")
        self.assertIn("'melanoma'", str(ctx.exception))

    def test_matching_labels_pass_and_return_the_canonical_order(self):
        self.assertIs(assert_labels_match(DERM7PT_TRUE, "derm7pt"), DERM7PT_CLASS_ORDER)
        self.assertIs(assert_labels_match(HAM_TRUE, "ham10000"), HAM10000_CLASS_ORDER)

    def test_a_class_subset_is_accepted(self):
        """A split need not contain every class; only foreign values are an error."""
        self.assertIs(assert_labels_match(["MEL", "NEV"], "derm7pt"), DERM7PT_CLASS_ORDER)

    def test_label_case_and_whitespace_do_not_trigger_a_false_alarm(self):
        """metrics.confusion_matrix lower-cases its inputs, so the guard must too."""
        assert_labels_match([" mel ", "Nev", "bcc"], "derm7pt")
        assert_labels_match(["Melanoma", "NEVUS"], "ham10000")

    def test_a_bare_string_is_rejected_rather_than_iterated_per_character(self):
        with self.assertRaises(TypeError):
            assert_labels_match("MEL", "derm7pt")

    def test_an_empty_cohort_raises_instead_of_passing_vacuously(self):
        with self.assertRaises(TaxonomyError):
            assert_labels_match([], "derm7pt")

    def test_unknown_taxonomy_is_reported_before_the_labels_are_inspected(self):
        with self.assertRaises(TaxonomyError) as ctx:
            assert_labels_match(DERM7PT_TRUE, "seven_point")
        self.assertIn("unknown taxonomy", str(ctx.exception))


class WhyTheGuardIsNeededTest(unittest.TestCase):
    """The failure modes the guard replaces, asserted rather than asserted-about."""

    def test_metrics_raises_only_a_cryptic_error_on_the_full_mis_pairing(self):
        with self.assertRaises(ValueError) as ctx:
            classification_summary(DERM7PT_TRUE, DERM7PT_PRED, DIAGNOSIS_LABELS)
        message = str(ctx.exception)
        self.assertIn("unknown diagnostic label", message)
        # It names a label, never the taxonomy -- it reads as bad data, not as a
        # mis-declared label set.
        self.assertNotIn("taxonomy", message)
        self.assertNotIn("derm7pt", message)

    def test_the_shared_bcc_token_makes_the_mis_pairing_completely_silent(self):
        """Same inputs, two taxonomies, two different headline numbers, no error."""
        truth = pred = ["BCC", "BCC", "BCC"]
        under_ham = classification_summary(truth, pred, HAM10000_CLASS_ORDER)
        under_derm7pt = classification_summary(truth, pred, DERM7PT_CLASS_ORDER)
        self.assertAlmostEqual(under_ham["balanced_accuracy"], 1 / 7)
        self.assertAlmostEqual(under_derm7pt["balanced_accuracy"], 1 / 5)
        self.assertNotEqual(
            under_ham["balanced_accuracy"], under_derm7pt["balanced_accuracy"]
        )
        # ... and the guard cannot separate them either, because 'bcc' really is a
        # member of both taxonomies.  This is the residual ambiguity: it is a
        # property of the vocabularies, not a hole in the guard.
        assert_labels_match(truth, "ham10000")
        assert_labels_match(truth, "derm7pt")

    def test_a_label_set_of_the_wrong_arity_silently_rescales_macro_averages(self):
        """Why resolve_labels returns *the* canonical order instead of any order."""
        correct = classification_summary(DERM7PT_TRUE, DERM7PT_PRED, DERM7PT_CLASS_ORDER)
        inflated = classification_summary(
            DERM7PT_TRUE, DERM7PT_PRED, DERM7PT_CLASS_ORDER + ("SCC",)
        )
        self.assertEqual(correct["accuracy"], inflated["accuracy"])
        self.assertGreater(
            correct["balanced_accuracy"], inflated["balanced_accuracy"]
        )
        self.assertGreater(correct["macro_f1"], inflated["macro_f1"])

    def test_the_guarded_call_produces_a_derm7pt_per_class_table(self):
        summary = classification_summary(
            DERM7PT_TRUE, DERM7PT_PRED, assert_labels_match(DERM7PT_TRUE, "derm7pt")
        )
        self.assertEqual(list(summary["per_class"]), list(DERM7PT_CLASS_ORDER))
        self.assertEqual(summary["n"], len(DERM7PT_TRUE))


if __name__ == "__main__":
    unittest.main()
