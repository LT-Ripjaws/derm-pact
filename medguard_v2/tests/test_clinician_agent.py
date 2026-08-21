"""The agent must stay inside its evidence, and must never do the arithmetic itself.

Two properties matter more than answer quality, because both would be invisible in a fluency
score and dangerous in a clinic:

1. **It cannot invent management.** Where the corpus holds no recommendation, the honest output is
   an explicit gap. A plausible-sounding recommendation is the failure this whole architecture
   exists to prevent.
2. **The host owns the arithmetic.** A clinician override changes a judgement; the score that
   follows is computed deterministically, so the agent can quote it but never derive it. These
   tests pin that the recomputation is exact, including when it flips the verdict.

No network. A scripted fake model drives the loop.
"""
from __future__ import annotations

import json
import unittest

from medguard_v2.agent.clinician_agent import AgentTools, run_agent
from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog

CLEAN = {name: "absent" for name in CRITERION_NAMES}


def _melanoma_criteria():
    values = dict(CLEAN)
    values["pigment_network"] = "atypical"    # 2
    values["blue_whitish_veil"] = "present"   # 2 -> 4
    return values


class ScriptedModel:
    """Returns a fixed sequence of raw model strings, recording what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages):
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else json.dumps(
            {"action": "refuse", "text": "out of scripted replies"}
        )


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class AgentToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis="NEV", criteria=None):
        return build_contract(
            case_id="t1",
            classifier_diagnosis=diagnosis,
            classifier_confidence=0.8,
            criteria=criteria or dict(CLEAN),
            catalog=self.catalog,
        )

    def test_override_recomputes_exactly_and_reports_the_flip(self):
        tools = AgentTools(self._contract("NEV", dict(CLEAN)), self.catalog)
        # absent -> atypical is worth 2; still below the threshold of 3.
        first = tools.override_criterion("pigment_network", "atypical")
        self.assertTrue(first.ok)
        self.assertEqual(first.payload["score_after"]["min"], 2)
        self.assertFalse(first.payload["verdict_changed"])

        # adding the veil takes it to 4 and crosses the threshold.
        second = tools.override_criterion("blue_whitish_veil", "present")
        self.assertEqual(second.payload["score_after"]["min"], 4)
        self.assertTrue(second.payload["verdict_changed"])
        self.assertEqual(second.payload["verdict_after"], "excision")
        self.assertIn("host", second.payload["computed_by"])

    def test_override_sequence_tracks_true_chronological_order_across_a_reoverride(self):
        # F2: AgentTools.override_sequence must record every call in true chronological order,
        # including a repeated criterion -- the flat `overrides` dict cannot recover this once
        # pigment_network is set twice, since dict iteration order reflects first insertion, not
        # most recent update.
        tools = AgentTools(self._contract("NEV", dict(CLEAN)), self.catalog)
        tools.override_criterion("pigment_network", "atypical")
        tools.override_criterion("blue_whitish_veil", "present")
        tools.override_criterion("pigment_network", "typical")

        self.assertEqual(
            tools.override_sequence,
            [
                ("pigment_network", "atypical"),
                ("blue_whitish_veil", "present"),
                ("pigment_network", "typical"),
            ],
        )
        # The flat dict still reflects only the final value per criterion (kept for callers that
        # only need end state, e.g. AgentAnswer.overrides_applied).
        self.assertEqual(
            tools.overrides, {"pigment_network": "typical", "blue_whitish_veil": "present"}
        )

    def test_override_rejects_a_value_outside_the_vocabulary(self):
        tools = AgentTools(self._contract(), self.catalog)
        result = tools.override_criterion("pigment_network", "very atypical")
        self.assertFalse(result.ok)
        self.assertIn("not a permitted value", result.note)

    def test_override_rejects_an_unknown_criterion(self):
        tools = AgentTools(self._contract(), self.catalog)
        self.assertFalse(tools.override_criterion("asymmetry", "present").ok)

    def test_override_accepts_indeterminate_and_produces_a_range(self):
        tools = AgentTools(self._contract("NEV", _melanoma_criteria()), self.catalog)
        result = tools.override_criterion("blue_whitish_veil", INDETERMINATE)
        self.assertTrue(result.ok)
        self.assertLess(result.payload["score_after"]["min"], result.payload["score_after"]["max"])

    def test_search_is_router_gated_for_no_coverage_conditions_but_discloses_absence(self):
        # NEV is NO_COVERAGE, so allowed_retrievals() is empty and every search_guidelines call
        # is gated -- but the committed (uncovered) condition's absence_reason must still come
        # through: absence disclosure is the honest answer and must not be blocked by gating.
        tools = AgentTools(self._contract(), self.catalog)
        result = tools.search_guidelines("NEV")
        self.assertTrue(result.ok)  # gated calls report ok=True; they are not a tool failure
        self.assertTrue(result.payload["router_gated"])
        self.assertEqual(result.payload["router_state"], "NO_COVERAGE")
        self.assertFalse(result.payload["served"])
        self.assertEqual(result.payload["condition_coverage"], "not_covered")
        self.assertIsNotNone(result.payload["absence_reason"])
        self.assertIn("no recommendation", result.payload["absence_reason"].lower())
        self.assertEqual(tools.retrieval_served, 0)
        self.assertEqual(tools.retrieval_skipped, 1)
        self.assertGreater(tools.retrieval_counterfactual_chars_skipped, 0)

    def test_search_serves_hash_pinned_records_when_in_the_routers_plan(self):
        # MEL/ROUTINE: the committed condition's own management guidance is exactly what
        # ROUTINE's plan authorises, so the call is served in full.
        tools = AgentTools(self._contract("MEL", _melanoma_criteria()), self.catalog)
        result = tools.search_guidelines("MEL", "initial_local_management")
        self.assertTrue(result.ok)
        self.assertNotIn("router_gated", result.payload)
        for rec in result.payload["recommendations"]:
            self.assertTrue(rec["content_sha256"])
            self.assertTrue(rec["citation_id"])
        self.assertEqual(tools.retrieval_served, 1)
        self.assertEqual(tools.retrieval_skipped, 0)

    def test_search_serves_the_non_committed_side_of_a_conflict(self):
        # NEV committed, but the checklist raises melanoma: CONFLICT's plan explicitly includes
        # MEL on the theory the clinician may need guidance for either reading.
        tools = AgentTools(self._contract("NEV", _melanoma_criteria()), self.catalog)
        self.assertEqual(tools.router_state.value, "CONFLICT")
        result = tools.search_guidelines("MEL", "initial_local_management")
        self.assertTrue(result.ok)
        self.assertNotIn("router_gated", result.payload)
        self.assertGreater(len(result.payload["recommendations"]), 0)
        self.assertEqual(tools.retrieval_served, 1)

    def test_gated_retrieval_false_reproduces_the_ungated_v4_behaviour(self):
        # The ablation arm: search_guidelines always serves, exactly like clinician-agent-v4.
        tools = AgentTools(self._contract(), self.catalog, gated_retrieval=False)
        result = tools.search_guidelines("NEV")
        self.assertFalse(result.ok)
        self.assertIn("no recommendation", result.note.lower())
        self.assertEqual(result.payload["recommendations"], [])
        self.assertEqual(tools.retrieval_served, 1)
        self.assertEqual(tools.retrieval_skipped, 0)

    def test_the_agent_cannot_see_the_image(self):
        tools = AgentTools(self._contract(), self.catalog)
        payload = tools.get_decision().payload
        blob = json.dumps(payload).lower()
        for leak in ("image_path", ".jpg", "pixel", "clinic_path", "derm_path"):
            self.assertNotIn(leak, blob)


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class AgentLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis="NEV", criteria=None):
        return build_contract(
            case_id="t1",
            classifier_diagnosis=diagnosis,
            classifier_confidence=0.8,
            criteria=criteria or dict(CLEAN),
            catalog=self.catalog,
        )

    def _run(self, replies, question="why?", contract=None, **run_kwargs):
        model = ScriptedModel(replies)
        answer = run_agent(
            contract=contract or self._contract(),
            catalog=self.catalog,
            question=question,
            chat=model,
            **run_kwargs,
        )
        return answer, model

    def test_it_can_call_a_tool_then_answer(self):
        answer, _ = self._run([
            json.dumps({"action": "tool", "tool": "get_decision", "args": {}}),
            json.dumps({"action": "answer", "text": "The score is 0.", "citations": []}),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.turns, 2)
        self.assertEqual(answer.tool_calls[0]["tool"], "get_decision")

    def test_it_can_refuse(self):
        answer, _ = self._run([
            json.dumps({"action": "refuse", "text": "The corpus holds nothing for naevi."})
        ])
        self.assertEqual(answer.kind, "refusal")

    def test_unparseable_output_becomes_a_refusal_not_a_displayed_answer(self):
        answer, _ = self._run(["I think it's probably fine, honestly"])
        self.assertEqual(answer.kind, "refusal")
        self.assertIn("could not be parsed", answer.text)
        self.assertIn("JSONDecodeError", answer.raw_error or "")

    def test_a_valid_but_non_dict_json_reply_becomes_a_refusal(self):
        # `json.loads` happily parses a bare string, number, or list. None of those has an
        # "action" key to dispatch on, so a naive `.get("action")` would crash with an
        # AttributeError instead of failing safely -- this must be caught and refused instead.
        answer, _ = self._run(['"just a string"'])
        self.assertEqual(answer.kind, "refusal")
        self.assertIn("could not be parsed", answer.text)
        self.assertIn("not an object", answer.raw_error or "")

    def test_an_unknown_tool_is_reported_back_rather_than_crashing(self):
        answer, model = self._run([
            json.dumps({"action": "tool", "tool": "diagnose", "args": {}}),
            json.dumps({"action": "answer", "text": "ok", "citations": []}),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertIn("unknown tool", model.seen[-1])

    def test_a_looping_agent_terminates_and_refers_to_the_clinician(self):
        replies = [json.dumps({"action": "tool", "tool": "get_decision", "args": {}})] * 10
        answer, _ = self._run(replies)
        self.assertEqual(answer.kind, "refusal")
        self.assertIn("referred back to the clinician", answer.text)

    def test_override_through_the_loop_is_recorded_on_the_answer(self):
        # Text states the revised score (pigment_network absent -> atypical is +2 on a clean
        # base) and names it a correction, so the verification gate passes clean on the first
        # draft and this exercises exactly one turn per scripted reply, not a forced revision.
        answer, _ = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "override_criterion",
                    "args": {"criterion": "pigment_network", "value": "atypical"},
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Recomputed after your correction: the score is now 2.",
                    "citations": [],
                }),
            ],
            question="criterion 1 is atypical, not absent",
        )
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.overrides_applied, {"pigment_network": "atypical"})
        self.assertFalse(answer.verification["revised"])

    def test_the_observation_carries_the_host_computed_consequence(self):
        # Default contract (NEV+CLEAN) is NO_COVERAGE, so the retrieval-completeness check (1c)
        # does not apply here; the answer states the revised score so the exact override check
        # (1a) passes clean too -- neither check should force a second chat() call, which would
        # otherwise shift model.seen[-1] away from the override tool's own observation.
        _, model = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "override_criterion",
                    "args": {"criterion": "blue_whitish_veil", "value": "present"},
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Based on your correction, the score is now 2.",
                    "citations": [],
                }),
            ],
        )
        observation = json.loads(model.seen[-1])["observation"]
        self.assertTrue(observation["ok"])
        self.assertIn("host", observation["payload"]["computed_by"])

    # -- verification loop (spec 1a/1b/1c) -----------------------------------

    def test_verification_gate_forces_one_revision_then_accepts_the_correction(self):
        # Default contract is NEV+CLEAN (NO_COVERAGE, score 0): a first draft that states a
        # score outside [0, 0] must be rejected by the deterministic gate (1a) and get exactly
        # one revision turn; a corrected second draft is accepted as-is.
        answer, model = self._run([
            json.dumps({"action": "answer", "text": "The 7-point score is 9.", "citations": []}),
            json.dumps({"action": "answer", "text": "The 7-point score is 0.", "citations": []}),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.text, "The 7-point score is 0.")
        self.assertEqual(answer.turns, 2)
        self.assertTrue(answer.verification["revised"])
        self.assertTrue(answer.verification["issues_first_draft"])
        self.assertEqual(answer.verification["issues_final"], [])
        # The host's rejection must reach the model but never surface in the displayed answer.
        self.assertIn("HOST VERIFICATION FAILED", model.seen[-1])
        self.assertNotIn("HOST VERIFICATION FAILED", answer.text)

    def test_a_clean_first_draft_is_never_revised(self):
        answer, model = self._run([
            json.dumps({"action": "answer", "text": "The 7-point score is 0.", "citations": []}),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.turns, 1)
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])
        self.assertEqual(answer.verification["issues_final"], [])
        self.assertEqual(len(model.seen), 1)   # llm_verifier is off by default: no second call

    def test_a_router_plan_alone_no_longer_forces_a_search_bounce(self):
        # F3 design amendment (supersedes spec 1c): the OLD blanket check guaranteed a bounce
        # any time router_plan was non-empty and no search_guidelines call had been made, full
        # stop -- measured to fire on 100% of why-questions and 41.5% of override questions on
        # plan-armed cases. NEV + melanoma-scoring criteria is CONFLICT (router_plan non-empty),
        # but this answer has no Assessment section, so under the new hard structural check (a
        # plan existing is no longer sufficient on its own) it must NOT bounce.
        contract = self._contract("NEV", _melanoma_criteria())
        answer, _ = self._run(
            [json.dumps({"action": "answer", "text": "The checklist score is 4.", "citations": []})],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])

    def test_retrieval_completeness_check_also_fires_on_pure_override_arithmetic(self):
        # F3 design amendment: the OLD blanket check bounced this every time, even though it is
        # pure override arithmetic with no Assessment section -- a rule-9-compliant override
        # answer (rule 9: override answers use only Diagnosis) was GUARANTEED a bounce it could
        # never satisfy, since revising the answer TEXT cannot add a search call. The NEW hard
        # structural check only fires when an Assessment section is present without a search, so
        # this override answer -- no Assessment section -- must NOT bounce.
        contract = self._contract("NEV", _melanoma_criteria())  # CONFLICT: router_plan non-empty
        answer_text = "Based on your correction, the score is now 5."
        answer, _ = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "override_criterion",
                    "args": {"criterion": "streaks", "value": "irregular"},
                }),
                json.dumps({"action": "answer", "text": answer_text, "citations": []}),
            ],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])

    # -- F3a: hard structural check (Assessment section implies a real search) --------------

    def test_assessment_section_without_any_search_call_is_a_hard_bounce(self):
        # Rule 7 requires an Assessment section to quote search_guidelines results verbatim.
        # Where quotable content exists (router plan non-empty), an Assessment with ZERO
        # search_guidelines calls is self-contradictory and bounces.
        contract = self._contract("MEL", _melanoma_criteria())  # ROUTINE: plan non-empty
        answer, _ = self._run(
            [
                json.dumps({
                    "action": "answer",
                    "text": "Diagnosis: score is 4.\nAssessment: excise within 4 weeks.",
                    "citations": [],
                }),
                json.dumps({"action": "answer", "text": "Diagnosis: score is 4.",
                            "citations": []}),
            ],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertTrue(answer.verification["revised"])
        self.assertTrue(
            any("Assessment section" in i for i in answer.verification["issues_first_draft"])
        )
        self.assertEqual(answer.verification["issues_final"], [])

    def test_absence_statement_on_a_no_coverage_case_is_not_bounced(self):
        # On NO_COVERAGE cases there is nothing to quote; an Assessment stating the absence of
        # guidance (grounded in get_decision's guideline block) is the DESIRED answer. The v5.0
        # blanket form of this check bounced 254 such drafts and the model deleted the section,
        # collapsing absence disclosure 0.9912 -> 0.177. The check must stand down here.
        answer, _ = self._run([
            json.dumps({
                "action": "answer",
                "text": ("Diagnosis: score is 0.\nAssessment: The licensed guideline corpus "
                         "holds no recommendation for naevi; this is a coverage boundary."),
                "citations": [],
            }),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])

    def test_assessment_section_after_a_real_search_call_does_not_trigger_the_hard_check(self):
        # Same Assessment-bearing shape, but search_guidelines WAS called first this
        # conversation, so rule 7's precondition is genuinely satisfied and the hard check must
        # not fire.
        contract = self._contract("MEL", _melanoma_criteria())  # ROUTINE: MEL is covered
        answer, _ = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "search_guidelines",
                    "args": {"condition": "MEL", "question_type": "initial_local_management"},
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Diagnosis: score is 4.\nAssessment: quoted guideline text.",
                    "citations": [],
                }),
            ],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])

    # -- F3b: per-case retrieval nudge in the system prompt ----------------------------------

    def test_system_prompt_gains_a_retrieval_nudge_when_the_router_plan_is_non_empty(self):
        from medguard_v2.agent.clinician_agent import _build_system_prompt

        with_plan = _build_system_prompt(None, retrieval_plan_nonempty=True)
        without_plan = _build_system_prompt(None, retrieval_plan_nonempty=False)
        self.assertIn("guidance exists", with_plan.lower())
        self.assertNotIn("guidance exists", without_plan.lower())

    # -- F6: gated_retrieval controls the prompt's own claim about itself -------------------

    def test_gated_retrieval_flag_controls_the_router_governed_sentence_in_the_prompt(self):
        from medguard_v2.agent.clinician_agent import _build_system_prompt

        gated_on = _build_system_prompt(None, gated_retrieval=True)
        gated_off = _build_system_prompt(None, gated_retrieval=False)
        self.assertIn("router-governed", gated_on.lower())
        self.assertNotIn("router-governed", gated_off.lower())

    # -- F5: verification must survive onto every terminal return, not just the answer path -

    def test_a_refusal_after_a_failed_first_draft_still_records_the_first_draft_failure(self):
        answer, _ = self._run([
            json.dumps({"action": "answer", "text": "The 7-point score is 9.", "citations": []}),
            json.dumps({"action": "refuse", "text": "Unable to resolve this satisfactorily."}),
        ])
        self.assertEqual(answer.kind, "refusal")
        self.assertTrue(answer.verification["issues_first_draft"])
        self.assertTrue(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_final"], answer.verification["issues_first_draft"])

    def test_a_parse_failure_after_a_failed_first_draft_still_records_the_first_draft_failure(self):
        answer, _ = self._run([
            json.dumps({"action": "answer", "text": "The 7-point score is 9.", "citations": []}),
            "not json at all",
        ])
        self.assertEqual(answer.kind, "refusal")
        self.assertTrue(answer.verification["issues_first_draft"])

    def test_turn_exhaustion_after_a_failed_first_draft_still_records_the_first_draft_failure(self):
        replies = [
            json.dumps({"action": "answer", "text": "The 7-point score is 9.", "citations": []}),
        ] + [json.dumps({"action": "tool", "tool": "get_decision", "args": {}})] * 10
        answer, _ = self._run(replies)
        self.assertEqual(answer.kind, "refusal")
        self.assertIn("referred back to the clinician", answer.text)
        self.assertTrue(answer.verification["issues_first_draft"])

    def test_a_clean_refusal_with_no_prior_draft_has_empty_verification(self):
        # No answer was ever drafted before the refusal, so there is nothing to report --
        # verification stays the empty-dict default, not a fabricated snapshot.
        answer, _ = self._run([
            json.dumps({"action": "refuse", "text": "The corpus holds nothing for naevi."})
        ])
        self.assertEqual(answer.kind, "refusal")
        self.assertEqual(answer.verification, {})

    def test_llm_verifier_flag_triggers_one_revision_on_a_failed_verdict(self):
        # llm_verifier defaults to False and is only consulted once the deterministic bucket is
        # clean; it shares the single revision budget with 1a/1c and deterministic issues would
        # take precedence, but here there are none to compete with.
        answer, model = self._run(
            [
                json.dumps({"action": "answer", "text": "The 7-point score is 0.", "citations": []}),
                json.dumps({"passed": False, "issues": ["misquotes a criterion"], "confidence": "high"}),
                json.dumps({"action": "answer", "text": "The 7-point score is 0.", "citations": []}),
                json.dumps({"passed": True, "issues": [], "confidence": "high"}),
            ],
            llm_verifier=True,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertTrue(answer.verification["revised"])
        self.assertIn("misquotes a criterion", answer.verification["issues_first_draft"])
        self.assertEqual(answer.verification["issues_final"], [])
        self.assertEqual(len(model.seen), 4)

    def test_llm_verifier_is_skipped_entirely_for_override_answers(self):
        # research/evidence/derm7pt_v1/llm_verifier_study_v1.json measured 0.278 precision /
        # 0.980 recall for the LLM verifier on override answers: it only ever sees the
        # pre-override contract, so it systematically flags legitimate clinician corrections.
        # override_answer_checks already owns this surface exactly, so verify_answer must not
        # be called at all once an override has been applied -- not even a scripted reply for
        # it is queued here, so a spurious call would surface as "out of scripted replies"
        # (a refusal) rather than the clean answer this asserts.
        answer, model = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "override_criterion",
                    "args": {"criterion": "pigment_network", "value": "atypical"},
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Based on your correction, the score is now 2.",
                    "citations": [],
                }),
            ],
            llm_verifier=True,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_final"], [])
        self.assertEqual(len(model.seen), 2)   # override tool call + answer; no 3rd (verifier) call

    # -- v5.2 Change 1: uncited-specifics check (shares ADVICE_SPECIFICS with score_fabrication) -

    def test_uncited_specifics_after_a_real_search_still_bounces_without_citations(self):
        # The v5.1 eval found 26 answers that had genuinely called search_guidelines this
        # conversation, and so do NOT trip the Assessment-without-search check (1c), but still
        # quoted a guideline specific with an EMPTY citations array. This reproduces that shape:
        # search happens for real, the draft names a specific ("cryotherapy") with no citation,
        # and only the new check catches it.
        contract = self._contract("MEL", _melanoma_criteria())  # ROUTINE: MEL is covered
        real_id = self.catalog.bind("MEL").recommendations[0].record_id
        text = "Diagnosis: score is 4.\nAssessment: cryotherapy is advised for this lesion."
        answer, _ = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "search_guidelines",
                    "args": {"condition": "MEL", "question_type": "initial_local_management"},
                }),
                json.dumps({"action": "answer", "text": text, "citations": []}),
                json.dumps({"action": "answer", "text": text, "citations": [real_id]}),
            ],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertTrue(answer.verification["revised"])
        self.assertTrue(
            any(
                "citations array" in i and "cryotherapy" in i
                for i in answer.verification["issues_first_draft"]
            ),
            f"expected an uncited-specifics flag, got {answer.verification['issues_first_draft']}",
        )
        self.assertEqual(answer.verification["issues_final"], [])
        self.assertEqual(answer.citations, (real_id,))

    def test_a_specifics_keyword_with_a_citation_present_is_never_bounced(self):
        contract = self._contract("MEL", _melanoma_criteria())
        real_id = self.catalog.bind("MEL").recommendations[0].record_id
        answer, _ = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "search_guidelines",
                    "args": {"condition": "MEL", "question_type": "initial_local_management"},
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Diagnosis: score is 4.\nAssessment: cryotherapy is advised for this lesion.",
                    "citations": [real_id],
                }),
            ],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])

    def test_a_pure_diagnosis_answer_with_no_specifics_is_not_forced_to_cite(self):
        # (b) Companion to the two tests above: no ADVICE_SPECIFICS keyword appears, so the
        # empty citations array must not be penalised -- a system-computed-only answer
        # legitimately has nothing to cite. Same case already covered by
        # test_a_clean_first_draft_is_never_revised; asserted explicitly here for Change 1.
        answer, _ = self._run([
            json.dumps({"action": "answer", "text": "The 7-point score is 0.", "citations": []}),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])

    # -- citation-array channel: the structured citations array must resolve in-loop ---------

    def test_a_fabricated_citation_in_the_array_only_is_bounced_in_loop(self):
        # The red-team bypass, exercised through the whole loop: the fabricated id is placed ONLY
        # in the structured citations array, never repeated inline in the answer text, so
        # deterministic_checks (which scans text) sees nothing. Before citation_array_checks was
        # wired into _draft_issues this drew zero issues and was displayed as-is; now it bounces.
        answer, model = self._run([
            json.dumps({
                "action": "answer", "text": "Diagnosis: the 7-point score is 0.",
                "citations": ["nice:fake-poisoned:9.9.9"],
            }),
            json.dumps({
                "action": "answer", "text": "Diagnosis: the 7-point score is 0.",
                "citations": [],
            }),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertTrue(answer.verification["revised"])
        self.assertTrue(
            any(
                "nice:fake-poisoned:9.9.9" in i and "does not resolve" in i
                for i in answer.verification["issues_first_draft"]
            ),
            f"expected a citation-array flag naming the fabricated id, got "
            f"{answer.verification['issues_first_draft']}",
        )
        self.assertEqual(answer.verification["issues_final"], [])
        self.assertEqual(answer.citations, ())
        # The host's rejection reaches the model but never surfaces in the displayed answer.
        self.assertIn("HOST VERIFICATION FAILED", model.seen[-1])
        self.assertNotIn("does not resolve", answer.text)

    def test_a_resolvable_citation_in_the_array_is_never_bounced_by_the_array_check(self):
        # Guard against overreach at the loop level: a genuinely valid record_id in the array must
        # pass on the first draft with no revision.
        real_id = self.catalog.bind("MEL").recommendations[0].record_id
        answer, _ = self._run([
            json.dumps({
                "action": "answer", "text": "Diagnosis: the 7-point score is 0.",
                "citations": [real_id],
            }),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_first_draft"], [])
        self.assertEqual(answer.citations, (real_id,))

    def test_a_real_inline_id_with_an_empty_array_is_not_bounced_as_uncited(self):
        # The derm7pt-14 false positive, through the whole loop: the answer quotes "4-mm clinical
        # margins" and prefixes it with a REAL inline record id, but leaves the structured citations
        # array empty. Before channel reconciliation this bounced on empty-array and (v5.3) failed
        # to converge over both revisions; now the host resolves the inline id and passes it clean.
        contract = self._contract("BCC", dict(CLEAN))  # BCC is covered -> router plan non-empty
        real_id = self.catalog.bind("BCC").recommendations[0].record_id
        text = (
            f"Diagnosis: the 7-point score is 0.\n"
            f"Assessment: {real_id}: surgical excision with 4-mm clinical margins is recommended."
        )
        answer, _ = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "search_guidelines",
                    "args": {"condition": "BCC", "question_type": "initial_local_management"},
                }),
                json.dumps({"action": "answer", "text": text, "citations": []}),
            ],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(
            answer.verification["revised"],
            f"real inline-cited answer should not bounce, got {answer.verification}",
        )
        self.assertEqual(answer.verification["issues_first_draft"], [])
        self.assertEqual(answer.citations, ())

    # -- v5.2 Change 2: enriched missing-revised-score message ------------------------------

    def test_missing_revised_score_bounce_quotes_the_host_number_then_the_correction_passes(self):
        # A first draft that applies the override but never states the resulting score must be
        # bounced with a message that quotes the host's own computed number (not a bare
        # "you forgot" message); a second draft that quotes that exact number back is accepted.
        answer, model = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "override_criterion",
                    "args": {"criterion": "pigment_network", "value": "atypical"},
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Applied your correction as requested.",
                    "citations": [],
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Based on your correction, the score is now 2.",
                    "citations": [],
                }),
            ],
            question="criterion 1 is atypical, not absent",
        )
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.text, "Based on your correction, the score is now 2.")
        self.assertTrue(answer.verification["revised"])
        self.assertEqual(answer.verification["revisions"], 1)
        self.assertEqual(answer.verification["issues_final"], [])
        bounce = model.seen[-1]
        self.assertIn("HOST VERIFICATION FAILED", bounce)
        self.assertIn("does not state the revised score", bounce)
        self.assertIn("revised score 2", bounce)
        self.assertIn("(range [2,2])", bounce)

    # -- v5.3: a conditional SECOND revision for exclusively mechanically-fixable issues ------

    def test_second_revision_fires_when_the_second_draft_still_misses_the_revised_score(self):
        # (a) A second draft that STILL fails only the mechanically-fixable missing-revised-
        # score check earns a second bounce (revisions capped at 2); a third draft that finally
        # states the host's own number is accepted.
        answer, model = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "override_criterion",
                    "args": {"criterion": "pigment_network", "value": "atypical"},
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Applied your correction as requested.",
                    "citations": [],
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Your correction has been applied to the record.",
                    "citations": [],
                }),
                json.dumps({
                    "action": "answer",
                    "text": "Based on your correction, the score is now 2.",
                    "citations": [],
                }),
            ],
            question="criterion 1 is atypical, not absent",
        )
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.text, "Based on your correction, the score is now 2.")
        self.assertEqual(answer.turns, 4)
        # (c) revisions is recorded as an explicit count, not just the "revised" bool.
        self.assertEqual(answer.verification["revisions"], 2)
        self.assertTrue(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_final"], [])
        bounces = [m for m in model.seen if "HOST VERIFICATION FAILED" in m]
        self.assertEqual(len(bounces), 2)

    def test_a_new_general_issue_on_the_second_draft_does_not_earn_a_second_revision(self):
        # (b) The second bounce is reserved for exclusively host-mechanically-fixable issues. A
        # second draft that still asserts a (different) out-of-range score is a general content
        # error, not a mechanical omission, so it is returned as-is rather than bounced again.
        answer, model = self._run([
            json.dumps({"action": "answer", "text": "The 7-point score is 9.", "citations": []}),
            json.dumps({"action": "answer", "text": "The 7-point score is 8.", "citations": []}),
        ])
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.text, "The 7-point score is 8.")
        self.assertEqual(answer.turns, 2)
        self.assertEqual(answer.verification["revisions"], 1)
        self.assertTrue(answer.verification["revised"])
        self.assertTrue(answer.verification["issues_final"])
        self.assertIn("score(s) [8]", answer.verification["issues_final"][0])
        bounces = [m for m in model.seen if "HOST VERIFICATION FAILED" in m]
        self.assertEqual(len(bounces), 1)

    def test_second_revision_covers_the_missing_score_and_uncited_specifics_together(self):
        # The two mechanically-fixable markers are an OR, not an exclusive choice: a second
        # draft carrying BOTH the missing-revised-score issue and the uncited-specifics issue at
        # once is still eligible, since every issue present matches one marker or the other.
        contract = self._contract("MEL", _melanoma_criteria())  # ROUTINE: MEL is covered
        real_id = self.catalog.bind("MEL").recommendations[0].record_id
        unfixed_text = "Applied your correction. Cryotherapy might be considered."
        answer, model = self._run(
            [
                json.dumps({
                    "action": "tool", "tool": "search_guidelines",
                    "args": {"condition": "MEL", "question_type": "initial_local_management"},
                }),
                json.dumps({
                    "action": "tool", "tool": "override_criterion",
                    "args": {"criterion": "pigment_network", "value": "typical"},
                }),
                json.dumps({"action": "answer", "text": unfixed_text, "citations": []}),
                json.dumps({"action": "answer", "text": unfixed_text, "citations": []}),
                json.dumps({
                    "action": "answer",
                    "text": "Based on your correction, the score is now 2.",
                    "citations": [real_id],
                }),
            ],
            contract=contract,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.turns, 5)
        self.assertEqual(answer.verification["revisions"], 2)
        self.assertTrue(answer.verification["revised"])
        self.assertEqual(answer.verification["issues_final"], [])
        self.assertEqual(len(answer.verification["issues_first_draft"]), 2)
        bounces = [m for m in model.seen if "HOST VERIFICATION FAILED" in m]
        self.assertEqual(len(bounces), 2)

    def test_llm_verifier_issue_on_the_second_draft_does_not_earn_a_second_revision(self):
        # General issues never earn the second bounce, including an llm_verifier issue -- free
        # text against no fixed template, so it can never be host-mechanically-fixable.
        answer, model = self._run(
            [
                json.dumps({"action": "answer", "text": "The 7-point score is 9.", "citations": []}),
                json.dumps({"action": "answer", "text": "The 7-point score is 0.", "citations": []}),
                json.dumps({"passed": False, "issues": ["misquotes a criterion"], "confidence": "high"}),
            ],
            llm_verifier=True,
        )
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.text, "The 7-point score is 0.")
        self.assertEqual(answer.verification["revisions"], 1)
        self.assertEqual(answer.verification["issues_final"], ["misquotes a criterion"])
        bounces = [m for m in model.seen if "HOST VERIFICATION FAILED" in m]
        self.assertEqual(len(bounces), 1)

    def test_the_system_prompt_forbids_self_computed_scores_and_invented_advice(self):
        from medguard_v2.agent.clinician_agent import SYSTEM_PROMPT

        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("never state a 7-point score", lowered)
        self.assertIn("never invent management advice", lowered)
        self.assertIn("you do not see the image", lowered)

    def test_prompt_version_is_v5_and_the_system_prompt_states_the_gating_rule(self):
        from medguard_v2.agent.clinician_agent import PROMPT_VERSION, SYSTEM_PROMPT

        self.assertEqual(PROMPT_VERSION, "clinician-agent-v5.4")
        self.assertIn("router-governed", SYSTEM_PROMPT.lower())

    def test_vision_swaps_rule_5_instead_of_stacking_a_contradiction(self):
        # A vision-enabled agent that still read "you do not see the image" would be told two
        # contradictory things by its own system prompt. _build_system_prompt must swap rule 5
        # for the vision variant rather than leaving both in, and the text-only prompt must be
        # untouched by the vision feature's existence.
        from medguard_v2.agent.clinician_agent import _build_system_prompt

        text_only = _build_system_prompt(None, vision=False)
        vision_on = _build_system_prompt(None, vision=True)

        self.assertIn("You do not see the image", text_only)
        self.assertNotIn("descriptive only", text_only)

        self.assertIn("descriptive only", vision_on)
        self.assertNotIn("You do not see the image", vision_on)


if __name__ == "__main__":
    unittest.main()
