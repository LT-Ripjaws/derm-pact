"""The evidence layer must fail closed, and must never invent guidance.

Two failure modes matter more than correctness here, because both would be invisible in a metric
and dangerous in a clinic:

1. **Fabricated management.** For most benign lesions no formal guideline exists. A system that
   fills that silence with generated advice is worse than one that abstains -- it converts an
   honest gap into a citation-shaped assertion. Tests below pin that the uncovered path produces
   an explicit absence and no recommendation text.
2. **A displayed contract that does not match its own evidence.** If the score disagrees with the
   criteria table, or a guideline record fails its hash, the contract must refuse to display
   rather than show a plausible-looking answer.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE
from medguard_v2.evidence.contract import build_contract, contract_to_dict
from medguard_v2.evidence.guideline_binding import (
    CATALOG_PATH,
    GROUP_TO_CONDITION,
    GuidelineCatalog,
    GuidelineIntegrityError,
)
from medguard_v2.evidence.render import render_markdown

CLEAN = {name: "absent" for name in CRITERION_NAMES}


def _melanoma_criteria():
    values = dict(CLEAN)
    values["pigment_network"] = "atypical"   # 2
    values["blue_whitish_veil"] = "present"  # 2 -> 4, above threshold
    return values


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class CatalogIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def test_every_record_verifies_its_own_content_hash(self):
        self.assertEqual(len(self.catalog), 66)

    def test_a_tampered_record_is_refused_at_load(self):
        import json

        records = [json.loads(l) for l in CATALOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        records[0] = dict(records[0])
        records[0]["text"] = records[0]["text"] + " and also do something else"
        with self.assertRaises(GuidelineIntegrityError):
            GuidelineCatalog(records)

    def test_coverage_is_only_bcc_and_melanoma_for_derm7pt(self):
        covered = {g for g, c in GROUP_TO_CONDITION.items() if c is not None}
        self.assertEqual(covered, {"BCC", "MEL"})

    def test_uncovered_group_returns_an_explicit_reason_and_no_text(self):
        result = self.catalog.bind("NEV")
        self.assertFalse(result.covered)
        self.assertEqual(result.recommendations, ())
        self.assertIsNotNone(result.absence_reason)
        self.assertIn("no recommendation", result.absence_reason.lower())

    def test_covered_group_returns_verbatim_hashed_recommendations(self):
        result = self.catalog.bind("MEL")
        self.assertTrue(result.covered)
        self.assertGreater(len(result.recommendations), 0)
        for rec in result.recommendations:
            self.assertTrue(rec.verify())
            self.assertTrue(rec.citation_id)
            self.assertTrue(rec.content_sha256)


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis, criteria, confidence=0.9):
        return build_contract(
            case_id="t1",
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            criteria=criteria,
            catalog=self.catalog,
        )

    def test_agreement_produces_no_conflict(self):
        contract = self._contract("MEL", _melanoma_criteria())
        self.assertFalse(contract.conflict.present)
        self.assertFalse(contract.conflict.escalate)
        self.assertTrue(contract.safe_to_display)

    def test_decidable_agreement_is_explicitly_both_views_agree(self):
        # Pins the exact fallback description text, so a future edit to _build_conflict's
        # branch ordering cannot silently change what a clinician reads for the common case.
        contract = self._contract("NEV", CLEAN)
        self.assertTrue(contract.score.decidable)
        self.assertFalse(contract.conflict.present)
        self.assertFalse(contract.conflict.escalate)
        self.assertEqual(contract.conflict.description, "Both views agree.")

    def test_undecidable_score_yields_no_conflict_but_escalates_for_review(self):
        # An unresolved checklist cannot agree OR disagree with the classifier. Before the
        # undecidable-first restructure, non-MEL undecidable cases fell through to "Both views
        # agree" with escalate=False -- silently hiding a case that actually needs a human to
        # resolve the unscored criteria.
        values = dict(CLEAN)
        values["dots_and_globules"] = "irregular"        # 1
        values["streaks"] = "irregular"                  # 1 -> 2, below threshold on its own
        values["blue_whitish_veil"] = INDETERMINATE      # worth 2 -> range [2, 4] spans 3
        contract = self._contract("NEV", values)

        self.assertFalse(contract.score.decidable)
        self.assertFalse(contract.conflict.present)
        self.assertTrue(contract.conflict.escalate)
        self.assertIn("blue_whitish_veil", contract.conflict.description)
        self.assertIn("has not reached a verdict", contract.conflict.description)
        self.assertEqual(contract.verification_failures(), [])

    def test_checklist_raising_melanoma_against_a_benign_call_is_a_conflict(self):
        contract = self._contract("NEV", _melanoma_criteria())
        self.assertTrue(contract.conflict.present)
        self.assertTrue(contract.conflict.escalate)
        self.assertTrue(contract.conflict.criteria_path_flags_melanoma)

    def test_classifier_raising_melanoma_against_a_clean_checklist_is_also_a_conflict(self):
        contract = self._contract("MEL", CLEAN)
        self.assertTrue(contract.conflict.present)
        self.assertFalse(contract.conflict.criteria_path_flags_melanoma)

    def test_conflict_binds_contingent_melanoma_guidance(self):
        contract = self._contract("NEV", _melanoma_criteria())
        self.assertIsNotNone(contract.contingent_guideline)
        self.assertTrue(contract.contingent_guideline.covered)
        rendered = render_markdown(contract)
        self.assertIn("If the checklist is right", rendered)

    def test_no_contingent_guidance_when_the_views_agree(self):
        contract = self._contract("NEV", CLEAN)
        self.assertIsNone(contract.contingent_guideline)

    def test_undecidable_score_also_binds_contingent_melanoma_guidance(self):
        # An undecidable range spans the threshold by definition, so it is exactly as capable of
        # turning out to be melanoma as an explicit conflict is -- it just has not been asked to
        # declare itself yet. Withholding contingent guidance here would make its presence an
        # artefact of criterion-scoring order rather than of actual risk (spec Part 2b).
        values = dict(CLEAN)
        values["dots_and_globules"] = "irregular"        # 1
        values["streaks"] = "irregular"                  # 1 -> 2, below threshold on its own
        values["blue_whitish_veil"] = INDETERMINATE      # worth 2 -> range [2, 4] spans 3
        contract = self._contract("NEV", values)

        self.assertFalse(contract.score.decidable)
        self.assertFalse(contract.conflict.present)  # undecidable, not an explicit conflict
        self.assertIsNotNone(contract.contingent_guideline)
        self.assertTrue(contract.contingent_guideline.covered)

    def test_uncovered_condition_never_yields_management_text(self):
        contract = self._contract("SK", CLEAN)
        rendered = render_markdown(contract)
        self.assertIn("No recommendation available", rendered)
        # The absence must not be dressed up as advice.
        for phrase in ("reassure", "no treatment", "routine follow", "discharge"):
            self.assertNotIn(phrase, rendered.lower())

    def test_indeterminate_criteria_propagate_to_a_range(self):
        values = dict(CLEAN)
        values["blue_whitish_veil"] = INDETERMINATE
        contract = self._contract("NEV", values)
        self.assertEqual(contract.unresolved_criteria, ("blue_whitish_veil",))
        self.assertFalse(contract.score.determinate)
        self.assertTrue(contract.safe_to_display)

    def test_a_tampered_contingent_guideline_record_fails_verification(self):
        # F10: verification_failures() hash-reverifies the committed guideline's own
        # recommendations, but was not doing the same for contingent_guideline -- the checklist's
        # contingent MEL guidance on a conflict/undecidable non-MEL case -- so a tampered
        # contingent record would have passed safe_to_display=True.
        import dataclasses

        contract = self._contract("NEV", _melanoma_criteria())  # binds contingent MEL guidance
        self.assertIsNotNone(contract.contingent_guideline)
        self.assertTrue(contract.contingent_guideline.covered)

        original = contract.contingent_guideline.recommendations[0]
        tampered_rec = dataclasses.replace(original, text=original.text + " and also do this")
        tampered_contingent = dataclasses.replace(
            contract.contingent_guideline,
            recommendations=(tampered_rec,) + contract.contingent_guideline.recommendations[1:],
        )
        tampered = dataclasses.replace(contract, contingent_guideline=tampered_contingent)

        failures = tampered.verification_failures()
        self.assertTrue(
            any(
                "contingent guideline record" in f and "failed its content hash" in f
                for f in failures
            ),
            f"expected a contingent-guideline hash failure, got {failures}",
        )
        self.assertFalse(tampered.safe_to_display)

    def test_a_contract_whose_score_disagrees_with_its_table_refuses_to_display(self):
        import dataclasses

        from medguard_v2.criteria.aggregate import score_criteria

        contract = self._contract("NEV", CLEAN)
        tampered = dataclasses.replace(contract, score=score_criteria(_melanoma_criteria()))
        failures = tampered.verification_failures()
        self.assertTrue(any("score does not match" in f for f in failures))
        self.assertFalse(tampered.safe_to_display)
        self.assertIn("OUTPUT WITHHELD", render_markdown(tampered))

    def test_serialisation_carries_the_denominators_and_failures(self):
        payload = contract_to_dict(self._contract("NEV", _melanoma_criteria()))
        self.assertIn("guideline", payload)
        self.assertIn("conflict", payload)
        self.assertIn("verification_failures", payload)
        self.assertTrue(payload["safe_to_display"])

    def test_render_puts_the_conflict_before_the_diagnosis(self):
        rendered = render_markdown(self._contract("NEV", _melanoma_criteria()))
        self.assertLess(
            rendered.index("disagree"),
            rendered.index("Image pattern model"),
            "a conflict must be visible before the confident-looking diagnosis line",
        )

    def test_render_shows_every_criterion_with_its_points(self):
        rendered = render_markdown(self._contract("MEL", _melanoma_criteria()))
        for name in CRITERION_NAMES:
            self.assertIn(name.replace("_", " "), rendered)


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class CoverageReportTest(unittest.TestCase):
    def test_coverage_report_publishes_the_denominator(self):
        catalog = GuidelineCatalog.load()
        report = catalog.coverage_report(["MEL", "NEV", "NEV", "BCC", "SK"])
        self.assertEqual(report["n_cases"], 5)
        self.assertEqual(report["n_guideline_eligible"], 2)
        self.assertEqual(report["eligible_fraction"], 0.4)
        self.assertEqual(report["uncovered_groups"], ["MISC", "NEV", "SK"])


if __name__ == "__main__":
    unittest.main()
