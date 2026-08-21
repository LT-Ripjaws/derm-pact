"""The renderer must answer part two of the clinical question, or say why it cannot."""
from __future__ import annotations

import pytest

from medguard.eval.matched_full_h2h_run import (
    ASSESSMENT_SECTION_LABEL,
    NO_GUIDANCE_SECTION_LABEL,
    NO_GUIDANCE_STATEMENT,
    _safe_answer,
)
from medguard.eval.ragas_judge import _extract_scored_span


class _Claim:
    def __init__(self, claim_type, text, evidence_ids=()):
        self.claim_type = claim_type
        self.text = text
        self.evidence_ids = tuple(evidence_ids)


class _Evidence:
    def __init__(self, evidence_id):
        self.evidence_id = evidence_id


class _Contract:
    def __init__(self, claims, evidence=(), diagnosis="bkl"):
        self.claims = list(claims)
        self.evidence = list(evidence)
        self.committed_diagnosis = diagnosis
        self.unresolved_uncertainties = ()


def _visual(text):
    return _Claim("diagnosis", text)


def _source_bound(text, evidence_ids=("e1",)):
    from medguard.eval.matched_full_h2h_run import _SOURCE_BOUND_CLAIM_TYPES
    return _Claim(next(iter(_SOURCE_BOUND_CLAIM_TYPES)), text, evidence_ids)


def test_a_lesion_with_no_guideline_says_so_instead_of_going_silent():
    """The defect: part two of the question was dropped without explanation."""
    answer = _safe_answer(_Contract([_visual("Findings consistent with bkl")]))
    assert NO_GUIDANCE_SECTION_LABEL in answer
    assert NO_GUIDANCE_STATEMENT in answer
    assert ASSESSMENT_SECTION_LABEL not in answer


def test_a_lesion_with_a_guideline_keeps_its_assessment_and_adds_no_disclosure():
    contract = _Contract(
        [_visual("Findings consistent with melanoma"),
         _source_bound("Excise with a 1 cm margin")],
        evidence=[_Evidence("e1")], diagnosis="melanoma")
    answer = _safe_answer(contract)
    assert ASSESSMENT_SECTION_LABEL in answer
    assert NO_GUIDANCE_SECTION_LABEL not in answer


def test_the_two_sections_are_mutually_exclusive():
    """If both could appear, the disclosure would be swept into the graded span."""
    with_guideline = _safe_answer(_Contract(
        [_source_bound("Excise with a 1 cm margin")], evidence=[_Evidence("e1")]))
    without = _safe_answer(_Contract([_visual("Findings consistent with bkl")]))
    assert (ASSESSMENT_SECTION_LABEL in with_guideline) != (
        NO_GUIDANCE_SECTION_LABEL in with_guideline)
    assert (ASSESSMENT_SECTION_LABEL in without) != (
        NO_GUIDANCE_SECTION_LABEL in without)


def test_the_disclosure_does_not_change_the_faithfulness_scoring_scope():
    """The whole point of not calling it 'Assessment:'.

    If the disclosure were graded as a literature claim, a text judge would try to
    ground 'the corpus contains no recommendation' in retrieved guideline text, fail,
    and score it zero -- turning an honest disclosure into a faithfulness penalty.
    """
    answer = _safe_answer(_Contract([_visual("Findings consistent with bkl")]))
    _, scope = _extract_scored_span(answer)
    assert scope == "whole", "disclosure must not create an assessment-scoped span"


def test_the_disclosure_carries_no_citation_marker():
    """It is a host statement about corpus coverage, not a cited claim."""
    answer = _safe_answer(_Contract([_visual("Findings consistent with bkl")]))
    assert "[" not in answer.split(NO_GUIDANCE_SECTION_LABEL)[1]


def test_the_diagnosis_line_still_comes_last_and_is_the_frozen_label():
    answer = _safe_answer(_Contract([_visual("x")], diagnosis="vasc"))
    assert answer.strip().splitlines()[-1] == "Diagnosis: vasc"


def test_an_answer_with_neither_visual_nor_source_claims_still_discloses():
    answer = _safe_answer(_Contract([]))
    assert NO_GUIDANCE_SECTION_LABEL in answer
    assert answer.strip().splitlines()[-1].startswith("Diagnosis:")


def test_the_disclosure_is_stated_as_corpus_coverage_not_clinical_advice():
    """It must not read as 'no treatment is indicated' -- that would be a claim."""
    lowered = NO_GUIDANCE_STATEMENT.lower()
    assert "licensed guideline corpus" in lowered
    assert "no management guidance is asserted" in lowered
    for forbidden in ("no treatment", "not required", "benign", "reassure", "no action"):
        assert forbidden not in lowered
