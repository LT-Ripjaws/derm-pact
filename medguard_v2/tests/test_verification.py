"""The deterministic verifier must catch faithfulness failures for free, before any LLM judge.

``deterministic_checks()`` is a zero-cost pass over an agent's answer: it does not decide whether
the answer is *good*, only whether it contradicts the contract it is supposed to be faithful to.
Three things matter here, because all three would be invisible to a fluency-oriented judge:

1. A stated 7-point score outside the contract's ``[score_min, score_max]`` is always wrong -- the
   score is arithmetic, not judgement. The one legitimate exception is an override answer, which
   quotes a real before/after pair that need not lie inside the *current* contract's range.
2. A criterion value that contradicts the contract's own table is a fabrication, not an opinion.
3. A cited guideline record ID that does not resolve in the catalog cannot be checked by a
   clinician either, so it is treated as equivalent to a wrong citation.
"""
from __future__ import annotations

import unittest

from medguard_v2.agent.verification import (
    citation_array_checks,
    deterministic_checks,
    extract_score_claim_numbers,
    override_answer_checks,
    uncited_specifics_check,
)
from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog

CLEAN = {name: "absent" for name in CRITERION_NAMES}


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class DeterministicChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis="NEV", criteria=None, confidence=0.9):
        return build_contract(
            case_id="t1",
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            criteria=criteria or dict(CLEAN),
            catalog=self.catalog,
        )

    def test_flags_a_stated_score_outside_the_contracts_range(self):
        contract = self._contract("NEV", CLEAN)  # score is exactly 0
        issues = deterministic_checks(contract, "The checklist score is 9, well above threshold.")
        self.assertTrue(
            any("outside the contract's range" in i for i in issues),
            f"expected a range violation, got {issues}",
        )

    def test_flags_a_criterion_misstatement_against_the_contracts_vocabulary(self):
        # Contract says vascular_structures is absent; the answer claims arborizing vessels, a
        # real vocabulary value the contract does not hold for this case.
        contract = self._contract("NEV", CLEAN)
        issues = deterministic_checks(
            contract, "On inspection, vascular structures show arborizing vessels."
        )
        self.assertTrue(
            any("vascular_structures" in i for i in issues),
            f"expected a vascular_structures misstatement, got {issues}",
        )

    def test_flags_a_fabricated_citation_id(self):
        contract = self._contract("NEV", CLEAN)
        issues = deterministic_checks(
            contract,
            "Per nice:ng99:9.9.9, excision is recommended.",
            catalog=self.catalog,
        )
        self.assertTrue(
            any("nice:ng99:9.9.9" in i and "not found" in i for i in issues),
            f"expected a fabricated-citation flag, got {issues}",
        )

    def test_a_clean_answer_with_correct_score_and_a_real_citation_passes(self):
        values = dict(CLEAN)
        values["pigment_network"] = "atypical"     # 2
        values["blue_whitish_veil"] = "present"     # 2 -> 4
        contract = self._contract("MEL", values)
        real_id = self.catalog.bind("MEL").recommendations[0].record_id

        text = (
            f"The 7-point score is 4, at or above the threshold. "
            f"Recommendation {real_id} supports this management."
        )
        issues = deterministic_checks(contract, text, catalog=self.catalog)
        self.assertEqual(issues, [])

    def test_flags_a_criterion_misstatement_even_when_the_word_clinician_appears_elsewhere(self):
        # F4: "clinician" appearing ANYWHERE in the answer used to silence this whole check
        # (house style encourages the word), which let a genuine criterion misstatement through
        # uncaught as long as an unrelated sentence happened to mention the clinician.
        contract = self._contract("NEV", CLEAN)
        issues = deterministic_checks(
            contract,
            "On inspection, vascular structures show arborizing vessels. "
            "The clinician should correlate this with the visual examination.",
        )
        self.assertTrue(
            any("vascular_structures" in i for i in issues),
            f"expected a vascular_structures misstatement despite the word 'clinician', got {issues}",
        )


class UncitedSpecificsCheckTest(unittest.TestCase):
    """v5.2 (Change 1): a guideline specific quoted with an empty citations array must be
    caught in-loop, using the same ADVICE_SPECIFICS rubric as eval.agent_eval.score_fabrication
    (moved here so both the host check and the eval scorer read one list, not two that could
    drift). These calls pass NO catalog, exercising the ``catalog=None`` fallback where the check
    degrades to the original array-presence test; the catalog-aware channel reconciliation (inline
    OR structured array) is covered by UncitedSpecificsChannelReconciliationTest below."""

    def test_flags_a_specifics_keyword_with_empty_citations(self):
        issues = uncited_specifics_check(
            "Assessment: cryotherapy is recommended for this lesion.", (),
        )
        self.assertTrue(
            any("cryotherapy" in i and "citations array" in i for i in issues),
            f"expected an uncited-specifics flag naming cryotherapy, got {issues}",
        )

    def test_the_same_specifics_keyword_with_a_citation_present_is_clean(self):
        issues = uncited_specifics_check(
            "Assessment: cryotherapy is recommended for this lesion.",
            ("nice:ng12:1.1.1",),
        )
        self.assertEqual(issues, [])

    def test_no_specifics_and_no_citations_does_not_demand_a_citation(self):
        # The check must not punish a pure Diagnosis-section answer (system-computed facts
        # only, per rule 8) that legitimately has nothing to cite.
        issues = uncited_specifics_check(
            "Diagnosis: the 7-point score is 4, at or above the threshold.", (),
        )
        self.assertEqual(issues, [])

    def test_multiple_keyword_hits_are_all_named_in_the_single_issue(self):
        issues = uncited_specifics_check(
            "Assessment: perform wide local excision with sentinel node evaluation.", [],
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("wide local", issues[0])
        self.assertIn("sentinel", issues[0])


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class UncitedSpecificsChannelReconciliationTest(unittest.TestCase):
    """The safety property is TRACEABILITY, achievable through either citation channel. With a
    catalog to resolve against, a quoted specific backed by a resolvable id -- inline (rule 7) OR
    in the structured array (rule 6) -- is safe; only a specific with no resolvable id anywhere is
    flagged. This closes the empty-array false positive (14 v5.2 / 15 v5.3 answers) without
    weakening the real catch (a fabricated id resolves against nothing and does not count)."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()
        cls.bcc_id = cls.catalog.bind("BCC").recommendations[0].record_id
        cls.mel_id = cls.catalog.bind("MEL").recommendations[0].record_id

    def test_a_real_inline_id_with_an_empty_array_is_traceable_and_clean(self):
        # The derm7pt-14 shape: quotes "4-mm clinical margins" and prefixes it with a REAL inline
        # record id, but leaves the structured citations array empty. Old check flagged this as
        # uncited; reconciled check resolves the inline id and passes.
        text = f"Assessment: {self.bcc_id}: surgical excision with 4-mm clinical margins."
        self.assertEqual(uncited_specifics_check(text, [], self.catalog), [])

    def test_a_fabricated_inline_id_does_not_count_as_traceability(self):
        # Same shape but the inline id is fabricated: it resolves against nothing, so the specific
        # is still uncited and must be flagged (deterministic_checks flags the fake token too).
        text = "Assessment: nice:fake:9.9.9: surgical excision with 4-mm clinical margins."
        issues = uncited_specifics_check(text, [], self.catalog)
        self.assertTrue(
            any("4-mm" in i and "citations array" in i for i in issues),
            f"expected an uncited-specifics flag, got {issues}",
        )

    def test_a_specific_with_no_id_in_either_channel_is_flagged(self):
        issues = uncited_specifics_check(
            "Assessment: cryotherapy is recommended for this lesion.", [], self.catalog,
        )
        self.assertTrue(
            any("cryotherapy" in i for i in issues),
            f"expected a flag when nothing traces the specific, got {issues}",
        )

    def test_a_real_id_in_the_structured_array_is_traceable_and_clean(self):
        # The array channel still works: a resolvable id in the structured array backs the
        # specific even with no inline id in the text.
        self.assertEqual(
            uncited_specifics_check(
                "Assessment: sentinel lymph node biopsy is advised.", [self.mel_id], self.catalog,
            ),
            [],
        )

    def test_a_resolvable_inline_id_saves_an_answer_whose_array_holds_only_junk(self):
        # Adversarial cross-channel case: the structured array carries only a fabricated id (which
        # citation_array_checks flags separately), but a resolvable inline id backs the specific,
        # so the traceability check itself is clean -- it must not double-punish the real inline id.
        text = f"Assessment: {self.mel_id}: sentinel lymph node biopsy is advised."
        self.assertEqual(
            uncited_specifics_check(text, ["nice:fake:9.9.9"], self.catalog), []
        )


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class CitationArrayChecksTest(unittest.TestCase):
    """The structured ``citations`` array must be resolved against the catalog in its own right.

    Before this check, ``deterministic_checks`` only scanned the answer TEXT for citation-shaped
    tokens and ``uncited_specifics_check`` stood down entirely once ``citations`` was non-empty,
    so a fabricated record_id placed ONLY in the structured array (never repeated inline) resolved
    against nothing and raised no issue -- the exact bypass proved in
    research/evidence/derm7pt_v1/redteam_v1.json (deterministic_architectural_probe,
    variant_citations_array_only_caught=false). These tests pin that gap closed while guarding
    against overreach: genuine record IDs pass, and an answer that cites nothing is never forced
    to invent a citation."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _real_ids(self, n=2):
        recs = self.catalog.bind("MEL").recommendations
        ids = [r.record_id for r in recs][:n]
        self.assertEqual(len(ids), n, "catalog did not yield enough real MEL record IDs")
        return ids

    def test_a_fabricated_id_in_the_array_only_is_caught(self):
        # (a) The exact red-team bypass: the id is passed ONLY through the citations array. The
        # function never sees any answer text, so "not inline" is intrinsic -- deterministic_checks
        # (which scans text) could never have caught this; citation_array_checks does.
        issues = citation_array_checks(["nice:fake-poisoned:9.9.9"], self.catalog)
        self.assertTrue(
            any("nice:fake-poisoned:9.9.9" in i and "does not resolve" in i for i in issues),
            f"expected the fabricated array id to be flagged, got {issues}",
        )

    def test_a_real_catalog_record_id_in_the_array_is_clean(self):
        # (b) Guard against overreach: a genuinely valid record_id must pass.
        (real_id,) = self._real_ids(1)
        self.assertTrue(self.catalog.has_record(real_id))
        self.assertEqual(citation_array_checks([real_id], self.catalog), [])

    def test_an_empty_citations_array_is_clean(self):
        # (c) Runs regardless of emptiness: an empty array cites nothing and must raise nothing.
        self.assertEqual(citation_array_checks([], self.catalog), [])
        self.assertEqual(citation_array_checks((), self.catalog), [])

    def test_a_valid_and_a_fabricated_id_flags_only_the_fabricated_one(self):
        # (d) Mixed array: only the non-resolving entry is named; the real one is left alone.
        real_id, other_real = self._real_ids(2)
        issues = citation_array_checks(
            [real_id, "aad:ng-nonexistent:0.0.0", other_real], self.catalog
        )
        self.assertEqual(len(issues), 1, f"expected exactly one flag, got {issues}")
        self.assertIn("aad:ng-nonexistent:0.0.0", issues[0])
        self.assertNotIn(real_id, issues[0])
        self.assertNotIn(other_real, issues[0])

    def test_a_duplicated_fabricated_id_is_reported_once(self):
        # Determinism guard: a repeated bad id yields one issue, not one per occurrence.
        issues = citation_array_checks(
            ["nice:fake:1.1.1", "nice:fake:1.1.1"], self.catalog
        )
        self.assertEqual(len(issues), 1, f"expected a single deduplicated flag, got {issues}")

    def test_no_catalog_available_stands_down(self):
        # Symmetry with deterministic_checks: with no catalog to resolve against, raise nothing
        # rather than flag every id as unresolved.
        self.assertEqual(citation_array_checks(["anything:at:all"], None), [])


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class OverrideAnswerChecksTest(unittest.TestCase):
    """The exact check (spec 1a): an override answer's stated before/after must be the host's
    own recomputation, not the contract's un-overridden range (deterministic_checks stands its
    own score check down for these; this is the real yardstick that replaces it)."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis="NEV", criteria=None, confidence=0.9):
        return build_contract(
            case_id="t1",
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            criteria=criteria or dict(CLEAN),
            catalog=self.catalog,
        )

    def test_no_overrides_means_no_issues(self):
        contract = self._contract("NEV", dict(CLEAN))
        self.assertEqual(override_answer_checks(contract, [], "anything at all, even 999"), [])

    def test_a_fabricated_after_score_is_flagged(self):
        # Real after-score for pigment_network absent -> atypical on a clean base is 2.
        contract = self._contract("NEV", dict(CLEAN))
        issues = override_answer_checks(
            contract,
            [("pigment_network", "atypical")],
            "Based on your correction, the score is now 9.",
        )
        self.assertTrue(
            any("number(s)" in i and "9" in i for i in issues),
            f"expected a fabricated-number flag, got {issues}",
        )

    def test_correct_before_and_after_ranges_pass_clean(self):
        contract = self._contract("NEV", dict(CLEAN))
        issues = override_answer_checks(
            contract,
            [("pigment_network", "atypical")],
            "Before your correction the score was 0. Based on your correction, the score is now 2.",
        )
        self.assertEqual(issues, [])

    def test_a_correct_but_unstated_revised_score_is_flagged(self):
        contract = self._contract("NEV", dict(CLEAN))
        issues = override_answer_checks(
            contract, [("pigment_network", "atypical")], "Applied your correction as requested.",
        )
        self.assertTrue(
            any("does not state the revised score" in i for i in issues),
            f"expected a missing-revised-score flag, got {issues}",
        )

    def test_the_missing_revised_score_message_states_the_host_computed_number(self):
        # v5.2 (Change 2): the bare message left the model to guess what to state; it must now
        # hand the model the exact host-computed number (and its range) to quote back. Absent
        # -> atypical on a clean base is a determinate +2, so after.score_min == after.score_max
        # == 2 here.
        contract = self._contract("NEV", dict(CLEAN))
        issues = override_answer_checks(
            contract, [("pigment_network", "atypical")], "Applied your correction as requested.",
        )
        self.assertTrue(
            any(
                "does not state the revised score" in i
                and "revised score 2" in i
                and "[2,2]" in i
                and "Diagnosis section" in i
                for i in issues
            ),
            f"expected the enriched revised-score message quoting host number 2, got {issues}",
        )

    def test_a_multi_step_override_is_checked_against_the_marginal_before_after_step(self):
        # Two overrides in one conversation: pigment_network first (absent -> atypical, +2),
        # then blue_whitish_veil (absent -> present, +2, crossing the threshold). Mirrors
        # AgentTools.override_criterion's own convention: "before" is every override except the
        # most recently applied one (score 2), "after" is all of them (score 4).
        contract = self._contract("NEV", dict(CLEAN))
        sequence = [("pigment_network", "atypical"), ("blue_whitish_veil", "present")]
        issues = override_answer_checks(
            contract, sequence,
            "Before your correction the score was 2. Based on your correction, the score is now 4.",
        )
        self.assertEqual(issues, [])

    def test_a_reoverridden_criterion_replays_by_true_chronological_order_not_dict_insertion(self):
        # F2 regression: pigment_network is overridden, then blue_whitish_veil, then
        # pigment_network AGAIN (a different value). A flat {criterion: value} dict loses this
        # -- dict iteration order reflects FIRST insertion, so list(dict)[-1] would name
        # blue_whitish_veil as the "last step" even though pigment_network was actually last.
        # That misidentifies the marginal before/after pair and flags a faithful answer.
        # Replaying the ordered sequence gets the true marginal step: before this last step the
        # score was 4 (pigment_network=atypical +2, blue_whitish_veil=present +2); after it,
        # pigment_network reverts to typical (0 points), so the score is now 2.
        contract = self._contract("NEV", dict(CLEAN))
        sequence = [
            ("pigment_network", "atypical"),
            ("blue_whitish_veil", "present"),
            ("pigment_network", "typical"),
        ]
        issues = override_answer_checks(
            contract, sequence,
            "Before your correction the score was 4. Based on your correction, the score is now 2.",
        )
        self.assertEqual(issues, [])

    # -- F1: context-anchored number extraction (adversarial false positives) --------------

    def test_a_confidence_decimal_is_not_read_as_a_score_claim(self):
        # Base criteria already put the true before/after pair at {2, 4} (not {0, ...}), so a
        # bare "0" shed from "0.82" by the OLD raw-digit extraction would be a genuine false
        # positive -- 0 is not in the expected set here.
        values = dict(CLEAN)
        values["blue_whitish_veil"] = "present"     # +2, present before the override below
        contract = self._contract("NEV", values)
        issues = override_answer_checks(
            contract,
            [("pigment_network", "atypical")],
            "Before your correction the score was 2. Based on your correction, the score is "
            "now 4 (model confidence 0.82).",
        )
        self.assertEqual(issues, [])

    def test_a_parenthetical_confidence_decimal_is_not_read_as_two_score_claims(self):
        # "(model confidence 1.0)" must not shed a "0" and a "1" as if they were score claims.
        values = dict(CLEAN)
        values["blue_whitish_veil"] = "present"
        contract = self._contract("NEV", values)
        issues = override_answer_checks(
            contract,
            [("pigment_network", "atypical")],
            "Before your correction the score was 2. Based on your correction, the score is "
            "now 4 (model confidence 1.0).",
        )
        self.assertEqual(issues, [])

    def test_enumerated_list_markers_are_not_read_as_score_claims(self):
        values = dict(CLEAN)
        values["blue_whitish_veil"] = "present"
        contract = self._contract("NEV", values)
        issues = override_answer_checks(
            contract,
            [("pigment_network", "atypical")],
            "Before your correction the score was 2. Based on your correction, the score is "
            "now 4. Next steps: 1. Verify the criterion visually. 2. Document the correction.",
        )
        self.assertEqual(issues, [])


class ExtractScoreClaimNumbersTest(unittest.TestCase):
    """Direct unit tests of the shared F1 extraction helper, independent of any contract."""

    def test_score_statement_variants_are_captured(self):
        for text, expected in [
            ("the score is 4", {4}),
            ("scores 3 on the checklist", {3}),
            ("the score was 2 before the correction", {2}),
            ("score: 5", {5}),
            ("the score is now 9", {9}),
        ]:
            self.assertEqual(extract_score_claim_numbers(text), expected, text)

    def test_bracketed_range_pairs_are_captured(self):
        self.assertEqual(
            extract_score_claim_numbers("the range is [2, 5] given the unresolved criterion"),
            {2, 5},
        )

    def test_was_now_and_from_to_transitions_near_score_are_captured(self):
        self.assertEqual(
            extract_score_claim_numbers("The score was 2 and is now 4."), {2, 4},
        )
        self.assertEqual(
            extract_score_claim_numbers("The score moves from 2 to 4 after the correction."),
            {2, 4},
        )

    def test_was_now_far_from_any_score_word_is_ignored(self):
        # No "score" mentioned anywhere near this was/now pair -- not a score claim.
        self.assertEqual(
            extract_score_claim_numbers("The lesion was 2 cm wide and is now 4 cm wide."), set(),
        )

    def test_decimal_confidence_and_list_markers_are_ignored(self):
        self.assertEqual(
            extract_score_claim_numbers(
                "(model confidence 1.0) Next: 1. Review. 2. Document."
            ),
            set(),
        )

    def test_numbers_above_ten_are_dropped(self):
        self.assertEqual(extract_score_claim_numbers("the score is 99"), set())


if __name__ == "__main__":
    unittest.main()
