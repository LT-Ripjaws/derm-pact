"""Report-aggregation logic in agent_eval.py.

``_build_report`` takes an already-built ``rows`` list (see ``run()``) and computes every
endpoint, the verification breakdown, and the retrieval-gating summary. It is tested directly
here, against synthetic rows, rather than through ``run()`` end to end -- ``run()`` needs the
Derm7pt data files and a live (or scripted) model, neither of which belongs in a fast, offline
unit test.

Two things matter:
  - F12: an endpoint's rate must read "not measured" (None), not a misleading 0.0, when its own
    denominator is empty -- e.g. a dev slice with no override questions, or no unanswerable ones.
  - F3(d)/F1: first-draft issues must be attributable to a category, and score_counterfactual's
    number extraction must agree with verification.override_answer_checks (both call the same
    shared helper), so a fabricated score is caught the same way on both paths.
"""
from __future__ import annotations

import unittest

from medguard_v2.eval.agent_eval import _build_report, _categorize_issue


def _row(**overrides) -> dict:
    base = {
        "case_id": "t1",
        "kind": "guideline",
        "question": "q",
        "answer_kind": "answer",
        "answer": "a",
        "turns": 1,
        "tool_calls": [],
        "citations": {
            "cited": 0, "resolved": 0, "intact": 0, "resolvability": None, "unresolvable_ids": [],
        },
        "refusal": {
            "answerable": True, "expects_absence": False, "refused": False,
            "correct": True, "states_absence": None,
        },
        "counterfactual": None,
        "fabrication": {
            "guideline_specifics": [], "searched": False, "cited_anything": False,
            "fabricated_management": False,
        },
        "verification": {},
        "retrieval": {},
    }
    base.update(overrides)
    return base


def _cf_block(**overrides) -> dict:
    base = {
        "override_applied": True,
        "host_score_before": 0,
        "host_score_after": 2,
        "host_verdict_before": "clinical follow up",
        "host_verdict_after": "clinical follow up",
        "verdict_changed": False,
        "states_correct_score": True,
        "asserts_a_wrong_score": False,
        "states_correct_verdict": True,
        "correct": True,
    }
    base.update(overrides)
    return base


class BuildReportEmptyDenominatorTest(unittest.TestCase):
    """F12: endpoints must read None ('not measured'), not 0.0, when their denominator is empty."""

    def test_counterfactual_endpoints_are_none_when_no_override_questions_exist(self):
        rows = [_row(), _row(kind="why")]  # no "counterfactual" rows at all
        report = _build_report(
            rows, n_cases=1, model="m", cohort="dev", llm_verifier=False, gated_retrieval=True,
        )
        cf = report["endpoints"]["counterfactual_accuracy"]
        self.assertEqual(cf["n"], 0)
        for key in ("override_applied", "states_correct_score", "asserts_a_wrong_score", "correct"):
            self.assertIsNone(cf[key], f"{key} should read 'not measured', got {cf[key]}")

    def test_refusal_on_unanswerable_is_none_when_every_question_is_answerable(self):
        rows = [_row()]
        report = _build_report(
            rows, n_cases=1, model="m", cohort="dev", llm_verifier=False, gated_retrieval=True,
        )
        self.assertIsNone(report["endpoints"]["refusal_correctness"]["on_unanswerable"])
        self.assertIsNotNone(report["endpoints"]["refusal_correctness"]["on_answerable"])

    def test_refusal_on_answerable_is_none_when_every_question_is_unanswerable(self):
        rows = [_row(refusal={
            "answerable": False, "expects_absence": False, "refused": True,
            "correct": True, "states_absence": None,
        })]
        report = _build_report(
            rows, n_cases=1, model="m", cohort="dev", llm_verifier=False, gated_retrieval=True,
        )
        self.assertIsNone(report["endpoints"]["refusal_correctness"]["on_answerable"])
        self.assertIsNotNone(report["endpoints"]["refusal_correctness"]["on_unanswerable"])

    def test_absence_disclosure_is_none_when_nothing_expects_absence(self):
        rows = [_row()]
        report = _build_report(
            rows, n_cases=1, model="m", cohort="dev", llm_verifier=False, gated_retrieval=True,
        )
        self.assertIsNone(report["endpoints"]["absence_disclosure"]["states_absence"])

    def test_rates_still_compute_normally_when_the_denominator_is_nonempty(self):
        rows = [
            _row(counterfactual=_cf_block()),
            _row(counterfactual=_cf_block(correct=False, states_correct_score=False)),
        ]
        report = _build_report(
            rows, n_cases=1, model="m", cohort="dev", llm_verifier=False, gated_retrieval=True,
        )
        cf = report["endpoints"]["counterfactual_accuracy"]
        self.assertEqual(cf["n"], 2)
        self.assertEqual(cf["correct"], 0.5)
        self.assertEqual(cf["override_applied"], 1.0)


class CategorizeIssueTest(unittest.TestCase):
    """F3(d): first-draft issue strings must be attributable to a category."""

    def test_categorizes_the_known_deterministic_issue_shapes(self):
        cases = [
            ("Answer states score(s) [9] outside the contract's range [0, 0]", "score_mismatch"),
            (
                "Override answer states number(s) [9] that match none of the expected "
                "before/after scores or the threshold [0, 3, 7, 10]",
                "score_mismatch",
            ),
            ("Override answer does not state the revised score", "score_mismatch"),
            (
                "Answer may misstate vascular_structures: contract says 'absent' but text "
                "mentions 'arborizing'",
                "criterion_misstatement",
            ),
            (
                "Cited record ID 'nice:ng99:9.9.9' not found in the guideline catalog",
                "citation_unresolved",
            ),
            (
                "The answer contains an Assessment section, which under rule 7 must quote "
                "search_guidelines results verbatim, but no search_guidelines call was made "
                "this conversation. Either consult search_guidelines first, or remove the "
                "Assessment section.",
                "assessment_without_search",
            ),
            (
                "Answer quotes guideline specifics (mm clinical margin, 4-mm) with an empty "
                "citations array: list the record IDs of the quoted recommendations in the "
                "citations array; if the specifics were not quoted from search results, "
                "remove them.",
                "uncited_specifics",
            ),
            ("Score misquotes the contract's stated confidence somehow", "llm_verifier"),
        ]
        for issue, expected in cases:
            self.assertEqual(_categorize_issue(issue), expected, issue)

    def test_revision_reasons_breakdown_counts_first_draft_issues_by_category(self):
        rows = [
            _row(verification={
                "issues_first_draft": [
                    "Answer states score(s) [9] outside the contract's range [0, 0]"
                ],
                "revised": True,
                "issues_final": [],
            }),
            _row(verification={
                "issues_first_draft": [
                    "The answer contains an Assessment section, which under rule 7 must quote "
                    "search_guidelines results verbatim, but no search_guidelines call was made "
                    "this conversation. Either consult search_guidelines first, or remove the "
                    "Assessment section."
                ],
                "revised": True,
                "issues_final": [],
            }),
            _row(verification={"issues_first_draft": [], "revised": False, "issues_final": []}),
            _row(answer_kind="refusal", verification={}),  # not an answer row: excluded
        ]
        report = _build_report(
            rows, n_cases=1, model="m", cohort="dev", llm_verifier=False, gated_retrieval=True,
        )
        self.assertEqual(
            report["verification"]["revision_reasons"],
            {"score_mismatch": 1, "assessment_without_search": 1},
        )
        self.assertEqual(report["verification"]["n_answers_checked"], 3)


class ScoreCounterfactualMirrorsOverrideAnswerChecksTest(unittest.TestCase):
    """F1: score_counterfactual's number extraction must be the SAME extraction
    override_answer_checks uses, not an independently-drifting copy."""

    def test_score_counterfactual_imports_the_shared_extraction_helper(self):
        import inspect

        from medguard_v2.agent.verification import extract_score_claim_numbers
        from medguard_v2.eval import agent_eval

        source = inspect.getsource(agent_eval.score_counterfactual)
        self.assertIn("extract_score_claim_numbers", source)
        self.assertIs(agent_eval.extract_score_claim_numbers, extract_score_claim_numbers)


if __name__ == "__main__":
    unittest.main()
