"""Citation-shaped text versus citation that resolves to a source.

Every retrieval baseline in this study writes ``[1]``-style markers into its answers.
None of those markers resolves to anything durable: they are positional indices into
that run's own ephemeral context list, carrying no document identity, version, page, or
content hash.  Nobody reading the answer later can tell what ``[3]`` was, and nobody can
detect that the underlying text has since changed.

MedGuard's citations are identifiers into a hash-pinned registry, so the same marker
resolves to an exact document, version, section and content hash months later, and a
tampered source fails the check.

This module measures the gap.  It is **descriptive, not inferential**: the outcome is
settled by architecture, and dressing it up as a hypothesis test would be dishonest.  It
is reported as counts with explicit denominators.

The comparison is deliberately unflattering to MedGuard in one respect and that is the
point: the baselines emit roughly **2.7x more** citation-shaped markers than MedGuard
does.  MedGuard writes fewer citations because it will only write ones it can resolve.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence

#: Bracketed numeric markers -- the citation shape every arm in this study emits.
CITATION_MARKER = re.compile(r"\[\d+\]")


def count_markers(answer: str) -> int:
    """How many citation-shaped markers this answer contains."""
    return len(CITATION_MARKER.findall(answer or ""))


def summarize_arm(
    answers: Sequence[str],
    *,
    resolvable_citations: int,
    cases_with_resolvable_citations: int,
    intact_citations: int,
    n_cases: int,
    guideline_eligible_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """One arm's citation-shape vs citation-resolvability profile.

    ``guideline_eligible_cases`` is the host-determined count of lesions whose committed
    condition has any recommendation in the licensed corpus.  It is fixed by the corpus
    and the frozen probe *before* any answer is read, so conditioning on it is a
    denominator choice, not a post-hoc filter -- but the unconditional figure is always
    reported beside it.
    """
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    answered = [a for a in answers if (a or "").strip()]
    marked = sum(1 for a in answered if CITATION_MARKER.search(a))
    total_markers = sum(count_markers(a) for a in answered)
    return {
        "answers_scored": len(answered),
        "answers_with_citation_markers": marked,
        "total_citation_markers": total_markers,
        "resolvable_citations": resolvable_citations,
        "intact_citations": intact_citations,
        # The headline: of the citation-shaped text this arm produced, how much of it
        # resolves to a versioned, content-hashed source?
        "citation_resolvability": (
            round(min(resolvable_citations, total_markers) / total_markers, 4)
            if total_markers else None),
        "integrity_of_resolvable": (
            round(intact_citations / resolvable_citations, 4)
            if resolvable_citations else None),
        "verifiable_citation_coverage_unconditional": round(
            cases_with_resolvable_citations / n_cases, 4),
        "verifiable_citation_coverage_on_eligible": (
            round(cases_with_resolvable_citations / guideline_eligible_cases, 4)
            if guideline_eligible_cases else None),
        "guideline_eligible_cases": guideline_eligible_cases,
        "n_cases": n_cases,
    }


def build_report(
    per_arm: Mapping[str, Mapping[str, Any]], *, n_cases: int,
    guideline_eligible_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the cross-arm citation-resolvability table."""
    return {
        "endpoint": "citation_resolvability",
        "role": "architectural_descriptive_not_inferential",
        "definition": (
            "Of the citation-shaped markers an arm writes into its answers, the fraction "
            "that resolve to a versioned, content-hashed source record. A marker that "
            "indexes only the run's own retrieval list does not resolve."
        ),
        "why_not_a_hypothesis_test": (
            "The outcome is settled by architecture: an arm without a source-statement "
            "registry cannot score above zero no matter how well it retrieves or writes. "
            "Reporting it as a paired test would manufacture significance out of a design "
            "choice. It is reported as counts with denominators instead."
        ),
        "denominator_note": (
            "verifiable_citation_coverage is reported twice: unconditionally over every "
            "lesion, and over the lesions whose committed condition has any recommendation "
            "in the licensed corpus. The licensed corpus covers only malignant/pre-malignant "
            "conditions because formal dermatology guidelines do -- the benign classes have "
            "no recommendations to cite, which is a property of the guideline literature, "
            "not a retrieval failure."
        ),
        "n_cases": n_cases,
        "guideline_eligible_cases": guideline_eligible_cases,
        "arms": dict(per_arm),
    }
