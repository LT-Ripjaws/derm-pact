"""The test split is a one-shot governed by PREREGISTRATION_V2; the pipeline must refuse it by
default rather than rely on every caller remembering not to ask for it.

``run_cohort(split="test")`` raises before doing anything else -- no feature file, no dataset,
no model fit -- so this test needs none of those fixtures either.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.pipeline.run_cohort import FEATURES, FEATURES_P1, run_cohort

# The safety-preserving promoted path reads the plain dermlip probe (diagnosis base + nothing
# else) and dermlip_tta (criteria base); it no longer needs the dermlip_clinic_tta multi-view that
# the regressing monolith used.
_PROMOTED_CACHES = (
    FEATURES.is_file()
    and (FEATURES_P1 / "dermlip_tta" / "features.npz").is_file()
)


class RunCohortTestSplitGuardTest(unittest.TestCase):
    def test_test_split_is_refused_without_allow_test(self):
        with self.assertRaises(RuntimeError):
            run_cohort(split="test")

    def test_the_refusal_names_the_governing_registration(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_cohort(split="test")
        self.assertIn("PREREGISTRATION_V2", str(ctx.exception))
        self.assertIn("allow_test", str(ctx.exception))

    def test_the_test_split_guard_holds_for_the_promoted_path_too(self):
        with self.assertRaises(RuntimeError):
            run_cohort(split="test", promoted=True)


@unittest.skipUnless(_PROMOTED_CACHES, "promoted feature caches (features/ + features_phase1/) absent")
class RunCohortPromotedPathTest(unittest.TestCase):
    """The promoted perception path fits, predicts, and plumbs per-criterion margins into every
    contract, and marks the report so it is never confused with the frozen pre-promotion one."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_cohort("valid", promoted=True, limit=6)

    def test_report_is_marked_promoted(self):
        self.assertEqual(self.report["protocol"], "medguard-v2-derm7pt-criteria-ensemble-v2-promoted")
        self.assertIn("perception_promotion", self.report)

    def test_every_contract_carries_full_criterion_margins(self):
        contracts = self.report["contracts"]
        self.assertEqual(len(contracts), 6)
        for c in contracts:
            self.assertIn("criterion_margins", c)
            self.assertEqual(set(c["criterion_margins"]), set(CRITERION_NAMES))
            for value in c["criterion_margins"].values():
                self.assertGreaterEqual(value, 0.0)  # margin = top - second posterior

    def test_promoted_contracts_are_safe_to_display(self):
        for c in self.report["contracts"]:
            self.assertEqual(c["verification_failures"], [])
            self.assertTrue(c["safe_to_display"])


if __name__ == "__main__":
    unittest.main()
