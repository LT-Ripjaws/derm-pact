"""generate_case_brief must never return unverified generated prose.

Two properties matter here, mirroring clinician_agent's own host-verification discipline at a
much smaller scale:

1. **A draft is checked against the contract, not trusted.** The same free deterministic checks
   the primary agent's drafts pass through catch a wrong score, a wrong action, or a fabricated
   management specific quoted with no citation.
2. **Verification failure never surfaces generated text.** A draft that still fails after its
   one revision turn is discarded entirely; what the caller gets back is a deterministic sentence
   built from the contract, not the model's second, still-unverified attempt.

No network. A scripted fake chat drives the loop.
"""
from __future__ import annotations

import unittest

from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.guideline_binding import CATALOG_PATH, GuidelineCatalog
from medguard_v2.evidence.render import render_headline, render_markdown
from medguard_v2.evidence.summary import CaseBrief, generate_case_brief

CLEAN = {name: "absent" for name in CRITERION_NAMES}


def _melanoma_criteria():
    values = dict(CLEAN)
    values["pigment_network"] = "atypical"    # 2
    values["blue_whitish_veil"] = "present"   # 2 -> 4, decidable, meets threshold (excision)
    return values


class ScriptedChat:
    """Returns a fixed sequence of raw model strings, recording what it was asked each turn."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages):
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else "out of scripted replies"


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class GenerateCaseBriefTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self, diagnosis="MEL", criteria=None, confidence=0.85, case_id="brief-1"):
        return build_contract(
            case_id=case_id,
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            criteria=criteria if criteria is not None else _melanoma_criteria(),
            catalog=self.catalog,
        )

    def test_verified_first_try(self):
        contract = self._contract()
        draft = (
            "The 7-point checklist score is 4, at or above the excision threshold, so excision "
            "is the checklist-implied action for this case."
        )
        chat = ScriptedChat([draft])

        brief = generate_case_brief(contract, chat)

        self.assertIsInstance(brief, CaseBrief)
        self.assertTrue(brief.verified)
        self.assertFalse(brief.fallback_used)
        self.assertEqual(brief.attempts, 1)
        self.assertEqual(brief.issues, ())
        self.assertEqual(brief.text, draft)
        self.assertEqual(len(chat.seen), 1)

    def test_bounce_then_verified(self):
        contract = self._contract()
        # Names neither the action nor the score -- fails both summary-specific checks.
        bad_draft = "This lesion shows concerning dermoscopic features and merits attention."
        good_draft = (
            "The checklist score is 4, meeting the excision threshold, so excision applies here."
        )
        chat = ScriptedChat([bad_draft, good_draft])

        brief = generate_case_brief(contract, chat)

        self.assertTrue(brief.verified)
        self.assertFalse(brief.fallback_used)
        self.assertEqual(brief.attempts, 2)
        self.assertEqual(brief.text, good_draft)
        self.assertEqual(len(chat.seen), 2)
        # Turn 2's prompt is the host's revision message, quoting the issues found in turn 1 --
        # never the bad draft's own text (see _revision_message's docstring).
        self.assertIn("HOST VERIFICATION FAILED", chat.seen[1])
        self.assertNotIn(bad_draft, chat.seen[1])

    def test_double_failure_returns_deterministic_fallback(self):
        contract = self._contract()
        bad_draft = "This lesion shows concerning dermoscopic features and merits attention."
        chat = ScriptedChat([bad_draft, bad_draft])

        brief = generate_case_brief(contract, chat)

        self.assertFalse(brief.verified)
        self.assertTrue(brief.fallback_used)
        self.assertEqual(brief.attempts, 2)
        self.assertGreater(len(brief.issues), 0)
        # Never the model's unverified text -- the fully deterministic headline instead.
        self.assertEqual(brief.text, render_headline(contract))
        self.assertNotEqual(brief.text, bad_draft)

    def test_fabricated_margin_advice_is_bounced_by_the_uncited_specifics_check(self):
        contract = self._contract()
        # States the action and score correctly (isolating the failure to ONE cause) but adds a
        # management specific ("4-mm margin") the contract never authorised and cites nothing.
        fabricated = (
            "Excision is recommended given a score of 4. Apply a 4-mm margin during excision."
        )
        clean = "The checklist score is 4, meeting the excision threshold, so excision applies."
        chat = ScriptedChat([fabricated, clean])

        brief = generate_case_brief(contract, chat)

        self.assertTrue(brief.verified)
        self.assertEqual(brief.attempts, 2)
        # The bounce (turn 2's prompt) must name the uncited-specifics failure specifically --
        # proving THIS check, not the action/score checks, is what caught the fabrication.
        revision_message = chat.seen[1]
        self.assertIn("4-mm", revision_message)
        self.assertIn("empty citations array", revision_message)

    def test_undecidable_contract_requires_unresolved_wording_not_a_score_number(self):
        values = dict(CLEAN)
        values["dots_and_globules"] = "irregular"      # 1
        values["streaks"] = "irregular"                 # 1 -> 2, below threshold on its own
        values["blue_whitish_veil"] = INDETERMINATE     # worth 2 -> range [2, 4] spans 3
        contract = self._contract("NEV", values)
        self.assertFalse(contract.score.decidable)

        stating_a_bound = "The checklist score is 2, and clinical follow up applies."
        saying_unresolved = (
            "The verdict is undetermined pending further assessment; clinical follow up applies "
            "for now."
        )

        bounced = generate_case_brief(contract, ScriptedChat([stating_a_bound, stating_a_bound]))
        self.assertFalse(bounced.verified)
        self.assertTrue(bounced.fallback_used)

        accepted = generate_case_brief(contract, ScriptedChat([saying_unresolved]))
        self.assertTrue(accepted.verified)
        self.assertEqual(accepted.text, saying_unresolved)

    def test_max_attempts_one_gives_no_revision_turn(self):
        contract = self._contract()
        bad_draft = "This lesion shows concerning dermoscopic features and merits attention."
        chat = ScriptedChat([bad_draft])

        brief = generate_case_brief(contract, chat, max_attempts=1)

        self.assertTrue(brief.fallback_used)
        self.assertEqual(brief.attempts, 1)
        self.assertEqual(len(chat.seen), 1)

    def test_max_attempts_below_one_is_rejected(self):
        contract = self._contract()
        with self.assertRaises(ValueError):
            generate_case_brief(contract, ScriptedChat([]), max_attempts=0)


@unittest.skipUnless(CATALOG_PATH.exists(), "guideline catalog not present")
class RenderMarkdownBriefIntegrationTest(unittest.TestCase):
    """render_markdown's optional `brief` parameter (Part 2's last bullet)."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = GuidelineCatalog.load()

    def _contract(self):
        return build_contract(
            case_id="brief-render-1",
            classifier_diagnosis="MEL",
            classifier_confidence=0.85,
            criteria=_melanoma_criteria(),
            catalog=self.catalog,
        )

    def test_default_none_is_byte_stable_with_no_brief_argument_at_all(self):
        contract = self._contract()
        self.assertEqual(render_markdown(contract), render_markdown(contract, brief=None))

    def test_verified_brief_renders_under_the_headline(self):
        contract = self._contract()
        draft = "The checklist score is 4, meeting the excision threshold, so excision applies."
        brief = generate_case_brief(contract, ScriptedChat([draft]))
        self.assertTrue(brief.verified)

        rendered = render_markdown(contract, brief=brief)
        expected_line = f"_AI summary (verified against the decision record):_ {draft}"
        self.assertIn(expected_line, rendered)
        # Directly under the headline: headline, blank, summary line.
        lines = [l for l in rendered.splitlines() if l.strip()]
        self.assertEqual(lines[1], render_headline(contract))
        self.assertEqual(lines[2], expected_line)
        # brief=None must still produce the plain form -- adding the parameter is additive.
        self.assertNotIn("AI summary", render_markdown(contract))

    def test_fallback_brief_renders_nothing_extra(self):
        contract = self._contract()
        bad_draft = "This lesion shows concerning dermoscopic features and merits attention."
        brief = generate_case_brief(contract, ScriptedChat([bad_draft, bad_draft]))
        self.assertTrue(brief.fallback_used)

        rendered_with_fallback = render_markdown(contract, brief=brief)
        rendered_without = render_markdown(contract)
        self.assertEqual(rendered_with_fallback, rendered_without)
        self.assertNotIn("AI summary", rendered_with_fallback)


if __name__ == "__main__":
    unittest.main()
