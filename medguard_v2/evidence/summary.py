"""Generate a short clinician-facing prose summary, verified against the decision contract.

Same discipline as ``clinician_agent.run_agent``'s host-verification loop, at a much smaller
scale: the contract is the ONLY source of facts the model is given, the draft is checked against
that contract with the same free deterministic checks the primary agent's own drafts pass
through (imported, not reimplemented -- see ``medguard_v2/agent/verification.py``), and a draft
that fails gets exactly one revision turn, with the issues injected back the same way the agent
injects them, before falling back to a purely deterministic sentence built from the contract
alone. There is never a code path where unverified generated text reaches the clinician.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from medguard_v2.agent.verification import (
    deterministic_checks,
    extract_score_claim_numbers,
    uncited_specifics_check,
)
from medguard_v2.evidence.contract import DecisionContract, contract_to_dict
from medguard_v2.evidence.render import render_headline


@dataclass(frozen=True)
class CaseBrief:
    """The result of one summary-generation attempt, honest about whether it can be trusted.

    ``verified`` and ``fallback_used`` are never both true, and exactly one of ``verified`` /
    ``fallback_used`` is true on every value this module returns: a brief either passed
    verification, or it is the deterministic fallback -- there is no third, unverified-but-shown
    state.
    """

    text: str
    verified: bool
    issues: Tuple[str, ...]
    attempts: int
    fallback_used: bool


_SYSTEM_PROMPT = """You are drafting a short clinical summary of ONE dermatology decision record \
for the clinician who will read it.

You are given the decision record as JSON -- the diagnostic pipeline's own structured facts. \
That JSON is the ONLY source of truth you may use. Do not use outside medical knowledge, do not \
speculate about the patient, and do not invent anything the record does not state.

Write 2-3 sentences of plain clinical English. The summary MUST:
- state the checklist's implied action (the "criteria_implied_action" field) using that exact \
phrase somewhere in the text
- state the 7-point score: the numeric "seven_point_score.min" value if \
"seven_point_score.decidable" is true, or say plainly that the verdict is unresolved/undetermined \
if it is false
- give no management advice beyond that implied action and, if present, the guideline \
recommendation's citation id -- never a margin width, a follow-up interval, a procedure name, or \
any other clinical specific that is not quoted verbatim from a guideline recommendation in the \
record, with its citation id given
- make no claim about the patient beyond what the record states

Reply with the summary text only. No preamble, no headers, no JSON, no markdown formatting.
"""


def _summary_specific_issues(contract: DecisionContract, text: str) -> List[str]:
    """Checks specific to a case brief, on top of the general-purpose answer checks below.

    (a) the checklist-implied action must be named verbatim -- the one clinical fact every reader
    needs regardless of anything else the summary says.
    (b) the score must appear, using the same score-claim extraction the agent's own override
    checks use (``extract_score_claim_numbers``) so a classifier confidence or a citation-id digit
    is not mistaken for a score claim. When the score is undecidable there is no single number to
    require -- the check instead requires the summary to actually say so, rather than silently
    accepting one bound stated as if it were final.
    """
    issues: List[str] = []
    action = contract.criteria_implied_action
    if action.lower() not in text.lower():
        issues.append(f"Summary does not state the checklist-implied action '{action}' verbatim.")

    if contract.score.decidable:
        claimed = extract_score_claim_numbers(text)
        if contract.score.score_min not in claimed:
            issues.append(
                f"Summary does not state the 7-point score {contract.score.score_min} in a "
                "recognisable 'score of N' or bracketed-range form."
            )
    else:
        lowered = text.lower()
        if "unresolved" not in lowered and "undetermined" not in lowered:
            issues.append(
                "Summary does not state that the verdict is unresolved/undetermined, despite "
                "the contract's score being undecidable."
            )
    return issues


def _verify_brief(contract: DecisionContract, text: str) -> List[str]:
    """The full check set for one drafted brief: the same free, zero-cost checks a primary
    agent's own drafts pass through -- contract/citation fidelity (``catalog=None``: a case brief
    never claims a guideline record ID, so there is nothing to resolve against a catalog, the same
    reason ``clinician_agent`` skips that sub-check when it has no catalog to hand in) and the
    uncited-guideline-specifics check (``citations=()`` always: this module never asks the model
    for a structured citations list, so any guideline specific in the text is by construction
    uncited and must be caught) -- plus the two summary-specific checks above.
    """
    issues = deterministic_checks(contract, text, catalog=None)
    issues += uncited_specifics_check(text, citations=())
    issues += _summary_specific_issues(contract, text)
    return issues


def _revision_message(issues: Sequence[str]) -> str:
    """Same host-revision pattern as ``clinician_agent._revision_message``: number the issues,
    never quote the draft back (so a fabricated claim cannot be echoed into the revision), and
    forbid mentioning the check in the revised text."""
    numbered = " ".join(f"{i + 1}. {issue}" for i, issue in enumerate(issues))
    return (
        "HOST VERIFICATION FAILED — your draft summary did not pass verification against the "
        f"decision record: {numbered} Revise the summary. Same instructions: 2-3 sentences, "
        "plain clinical English, no invented specifics. Do not apologise, do not mention this "
        "check in the summary text."
    )


def generate_case_brief(
    contract: DecisionContract,
    chat: Callable[[List[Dict[str, Any]]], str],
    max_attempts: int = 2,
) -> CaseBrief:
    """Draft a short prose summary of ``contract`` and verify it before returning it.

    ``chat`` has the same signature as the primary agent's -- a list of ``{"role", "content"}``
    messages in, a raw string out -- so the fake-chat test doubles used elsewhere in this project
    work here unmodified.

    A failing draft gets exactly ``max_attempts - 1`` revision turns (one, at the default of 2)
    with the issues injected back as a host message, mirroring
    ``clinician_agent._draft_issues`` / ``_revision_message``. If the draft is still unverified
    after ``max_attempts``, the return carries ``fallback_used=True`` and ``text`` is the fully
    deterministic headline sentence (``render_headline``) -- never unverified generated prose.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    facts = contract_to_dict(contract)
    facts.pop("receipt", None)  # internal provenance, not a clinical fact to summarise

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(facts, indent=2, default=str)},
    ]

    issues: List[str] = []
    for attempt in range(1, max_attempts + 1):
        raw = chat(messages)
        text = raw.strip()
        issues = _verify_brief(contract, text)
        if not issues:
            return CaseBrief(
                text=text, verified=True, issues=(), attempts=attempt, fallback_used=False
            )
        if attempt < max_attempts:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": _revision_message(issues)})

    return CaseBrief(
        text=render_headline(contract),
        verified=False,
        issues=tuple(issues),
        attempts=max_attempts,
        fallback_used=True,
    )
