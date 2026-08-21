"""SingleViewContract must fail closed, disclose its missing view, and never invent management.

The dual-view DecisionContract's invariants (criteria/score/conflict always present) are what
every evaluated number rests on, so the single-view case for annotation-free imaging sources
(HAM/BCN) is a SEPARATE type. These tests pin the obligations that type carries:

1. **The absent checklist is disclosed, or nothing is shown.** An empty
   ``checklist_unavailable_reason`` is a verification failure, not a default.
2. **Management comes from the binding only.** Covered means the action cites the first bound
   record; uncovered means the action states the absence. No ladder step is ever derived,
   because there is no score to derive it from.
3. **The checks that apply are the dual-view checks.** Guideline content hashes, the
   absence-reason requirement and the similar-cases leak-free/cosine checks are shared code
   (``contract.similar_cases_failures``), so the two contract types cannot drift apart.
"""
from __future__ import annotations

import dataclasses
import unittest

from medguard_v2.evidence.contract import SimilarCase, SimilarCasesContext
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog
from medguard_v2.evidence.single_view import (
    build_single_view_contract,
    single_view_action,
    single_view_contract_to_dict,
)

REASON = "this imaging source carries no dermoscopic-criteria annotations; single-view decision"


def _context(query_case_id="ISIC_999", neighbors=None, nearest=None):
    default = (
        SimilarCase(case_id="HAM_1", diagnosis="MEL", similarity=0.91, same_label=True),
        SimilarCase(case_id="HAM_2", diagnosis="NEV", similarity=0.84, same_label=False),
    )
    nbs = default if neighbors is None else neighbors
    return SimilarCasesContext(
        query_case_id=query_case_id,
        reference_diagnosis="MEL",
        neighbors=nbs,
        nearest_different_label=nearest,
        feature_space="dermlip (512-d)",
        bank_name="toy@dermlip",
        bank_hash="cd" * 32,
        k=len(nbs),
    )


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class SingleViewContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _build(self, diagnosis, confidence=0.8712, reason=REASON, **kwargs):
        return build_single_view_contract(
            case_id="ISIC_0000001",
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            catalog=self.catalog,
            reason=reason,
            **kwargs,
        )

    # -- construction -------------------------------------------------------------------------

    def test_covered_diagnosis_binds_hash_verified_recommendations(self):
        contract = self._build("MEL")
        self.assertTrue(contract.guideline.covered)
        self.assertGreater(len(contract.guideline.recommendations), 0)
        for rec in contract.guideline.recommendations:
            self.assertTrue(rec.verify())
        self.assertEqual(contract.verification_failures(), [])
        self.assertTrue(contract.safe_to_display)

    def test_uncovered_diagnosis_carries_an_explicit_absence(self):
        contract = self._build("NEV")
        self.assertFalse(contract.guideline.covered)
        self.assertEqual(contract.guideline.recommendations, ())
        self.assertIsNotNone(contract.guideline.absence_reason)
        self.assertTrue(contract.safe_to_display)

    def test_confidence_is_rounded_to_four_places(self):
        contract = self._build("MEL", confidence=0.87119999)
        self.assertEqual(contract.classifier_confidence, 0.8712)

    def test_unknown_diagnosis_group_is_rejected_before_a_contract_exists(self):
        with self.assertRaises(KeyError):
            self._build("NOT_A_GROUP")

    def test_receipt_is_copied_defensively(self):
        receipt = {"entry_point": "test"}
        contract = self._build("MEL", receipt=receipt)
        receipt["entry_point"] = "mutated"
        self.assertEqual(contract.receipt["entry_point"], "test")

    # -- the derived action -------------------------------------------------------------------

    def test_covered_action_cites_the_first_bound_record(self):
        contract = self._build("MEL")
        action, reason = single_view_action(contract)
        first = contract.guideline.recommendations[0].citation_id
        self.assertEqual(action, f"follow {first}")
        self.assertIn("melanoma", reason)

    def test_uncovered_action_states_the_absence_not_a_management_step(self):
        contract = self._build("SK")
        action, reason = single_view_action(contract)
        self.assertEqual(action, "no citable management")
        self.assertIn("seborrhoeic keratosis", reason)
        # never a ladder step: there is no checklist score to map onto the ladder.
        for ladder_step in ("excision", "clinical follow up", "no further examination"):
            self.assertNotIn(ladder_step, action)

    # -- verification: fail closed ------------------------------------------------------------

    def test_empty_disclosure_reason_fails_verification(self):
        for empty in ("", "   ", "\n"):
            contract = self._build("MEL", reason=empty)
            failures = contract.verification_failures()
            self.assertTrue(
                any("checklist-unavailable disclosure" in f for f in failures),
                f"reason={empty!r}: {failures}",
            )
            self.assertFalse(contract.safe_to_display)

    def test_a_tampered_guideline_record_fails_its_content_hash(self):
        contract = self._build("MEL")
        original = contract.guideline.recommendations[0]
        tampered_rec = dataclasses.replace(original, text=original.text + " and also do this")
        tampered_guideline = dataclasses.replace(
            contract.guideline,
            recommendations=(tampered_rec,) + contract.guideline.recommendations[1:],
        )
        tampered = dataclasses.replace(contract, guideline=tampered_guideline)
        failures = tampered.verification_failures()
        self.assertTrue(any("failed its content hash" in f for f in failures), failures)
        self.assertFalse(tampered.safe_to_display)

    def test_coverage_claim_without_recommendations_fails(self):
        contract = self._build("MEL")
        hollow = dataclasses.replace(
            contract,
            guideline=dataclasses.replace(contract.guideline, recommendations=()),
        )
        failures = hollow.verification_failures()
        self.assertTrue(any("claims coverage but carries no recommendations" in f for f in failures))

    def test_uncovered_without_a_reason_fails(self):
        contract = self._build("NEV")
        silent = dataclasses.replace(
            contract,
            guideline=dataclasses.replace(contract.guideline, absence_reason=None),
        )
        failures = silent.verification_failures()
        self.assertTrue(any("no reason was recorded" in f for f in failures))

    # -- similar-cases checks are the SHARED dual-view checks ----------------------------------

    def test_similar_cases_self_leak_is_rejected(self):
        leaky = _context(
            query_case_id="HAM_1"  # equals a neighbour id -> leave-one-out violated
        )
        contract = self._build("MEL", similar_cases=leaky)
        failures = contract.verification_failures()
        self.assertTrue(any("leaks the query case" in f for f in failures), failures)
        self.assertFalse(contract.safe_to_display)

    def test_similar_cases_cosine_out_of_range_is_rejected(self):
        bad = _context(
            neighbors=(SimilarCase(case_id="HAM_1", diagnosis="MEL", similarity=1.7, same_label=True),)
        )
        contract = self._build("MEL", similar_cases=bad)
        self.assertTrue(
            any("outside the cosine range" in f for f in contract.verification_failures())
        )

    def test_similar_cases_empty_neighbours_is_rejected(self):
        empty = _context(neighbors=())
        contract = self._build("MEL", similar_cases=empty)
        self.assertTrue(any("no neighbours" in f for f in contract.verification_failures()))

    def test_valid_similar_cases_context_passes(self):
        contract = self._build("MEL", similar_cases=_context())
        self.assertTrue(contract.safe_to_display)

    # -- serialisation ------------------------------------------------------------------------

    def test_serialisation_is_marked_single_view_and_carries_no_checklist_keys(self):
        payload = single_view_contract_to_dict(self._build("MEL"))
        self.assertEqual(payload["view"], "single")
        self.assertEqual(payload["checklist_unavailable_reason"], REASON)
        self.assertTrue(payload["safe_to_display"])
        self.assertEqual(payload["verification_failures"], [])
        for absent_key in ("criteria", "seven_point_score", "conflict", "criteria_implied_action"):
            self.assertNotIn(absent_key, payload)

    def test_serialised_derived_action_matches_the_function(self):
        contract = self._build("MEL")
        action, reason = single_view_action(contract)
        payload = single_view_contract_to_dict(contract)
        self.assertEqual(payload["derived_action"]["action"], action)
        self.assertEqual(payload["derived_action"]["reason"], reason)

    def test_serialisation_omits_similar_cases_when_absent_and_carries_it_when_present(self):
        without = single_view_contract_to_dict(self._build("MEL"))
        self.assertNotIn("similar_cases", without)
        with_ctx = single_view_contract_to_dict(self._build("MEL", similar_cases=_context()))
        self.assertIn("similar_cases", with_ctx)
        self.assertEqual(with_ctx["similar_cases"]["query_case_id"], "ISIC_999")

    def test_serialisation_carries_failures_for_an_unsafe_contract(self):
        payload = single_view_contract_to_dict(self._build("MEL", reason=""))
        self.assertFalse(payload["safe_to_display"])
        self.assertTrue(payload["verification_failures"])


if __name__ == "__main__":
    unittest.main()
