"""Gated retrieval: decide what evidence this case needs *before* assembling the answer.

Until now retrieval was unconditional -- `catalog.bind(committed_diagnosis)` ran on every case and
always asked for management guidance. That is retrieval *conditioned* on perception, not retrieval
*gated* by it, and it wastes work on the 72% of cases where the catalog holds nothing while asking
the wrong question on the cases that need something else.

The router turns the perception state into a typed retrieval action. Four states, each firing on a
trigger measured in advance rather than assumed:

| state | trigger | measured rate | what it retrieves |
|---|---|---|---|
| `NO_COVERAGE` | catalog holds nothing for the committed condition | ~72% | nothing; reports the absence |
| `CONFLICT` | classifier and criteria path disagree on melanoma | ~15% | both candidate conditions, differential + management |
| `UNDECIDABLE` | score range spans the threshold | ~21% of criteria-abstaining cases | visual-assessment guidance for the unresolved criteria |
| `ROUTINE` | otherwise | remainder | management guidance for the committed condition |

**Deterministic on purpose.** v1 built the learned version twice: a depth-two controller that lost
to plain confidence (regret 0.20373 vs 0.10780), and a five-action remediation loop that fired on
0 of 160 clean cases. A transparent table that fires on measured triggers is both more auditable
and, on this project's evidence, more effective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple

from medguard_v2.criteria.aggregate import SevenPointScore
from medguard_v2.criteria.schema import MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.evidence.guideline_binding import (
    GROUP_TO_CONDITION,
    MANAGEMENT_QUESTION_TYPES,
    BindingResult,
    GuidelineCatalog,
)


class RetrievalState(str, Enum):
    NO_COVERAGE = "NO_COVERAGE"
    CONFLICT = "CONFLICT"
    UNDECIDABLE = "UNDECIDABLE"
    ROUTINE = "ROUTINE"


#: Which guideline question types each state actually needs. A conflict case needs to know how the
#: two candidate conditions are told apart; an undecidable case needs to know what the unresolved
#: feature looks like; a routine case needs to know what to do -- which is exactly what
#: MANAGEMENT_QUESTION_TYPES (guideline_binding.py) already means, so ROUTINE mirrors it directly
#: rather than hand-listing a subset that can silently drift out of sync with it. Before this fix
#: ROUTINE's own tuple omitted follow_up_surveillance even though search_guidelines' tool doc
#: lists it as a valid question type: a ROUTINE-state agent asking for it by name got gated.
QUESTION_TYPES_FOR_STATE: Dict[RetrievalState, Tuple[str, ...]] = {
    RetrievalState.ROUTINE: MANAGEMENT_QUESTION_TYPES,
    RetrievalState.CONFLICT: ("differential_diagnosis", "initial_local_management", "referral_triage"),
    RetrievalState.UNDECIDABLE: ("clinical_visual_assessment", "differential_diagnosis"),
    RetrievalState.NO_COVERAGE: (),
}

#: Below this the classifier is not committing to anything, and single-condition management
#: guidance would be answering a question the system has not actually asked.
LOW_CONFIDENCE = 0.40


@dataclass(frozen=True)
class RetrievalDecision:
    """A typed, receipted retrieval action."""

    state: RetrievalState
    reason: str
    question_types: Tuple[str, ...]
    conditions_requested: Tuple[str, ...]
    results: Dict[str, BindingResult] = field(default_factory=dict)
    retrieval_calls: int = 0
    skipped: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "question_types": list(self.question_types),
            "conditions_requested": list(self.conditions_requested),
            "retrieval_calls": self.retrieval_calls,
            "skipped": self.skipped,
            "results": {k: v.as_dict() for k, v in self.results.items()},
        }


def classify_state(
    *,
    classifier_diagnosis: str,
    classifier_confidence: float,
    score: SevenPointScore,
) -> Tuple[RetrievalState, str]:
    """Decide what kind of case this is. Order matters: safety-relevant states win."""
    criteria_flags = score.score_min >= MELANOMA_SUSPICION_THRESHOLD
    conflict = (criteria_flags and classifier_diagnosis != "MEL") or (
        classifier_diagnosis == "MEL" and score.decidable and not criteria_flags
    )

    if conflict:
        return (
            RetrievalState.CONFLICT,
            f"classifier says {classifier_diagnosis}; checklist score range "
            f"[{score.score_min},{score.score_max}] against threshold "
            f"{MELANOMA_SUSPICION_THRESHOLD}",
        )

    if not score.decidable:
        return (
            RetrievalState.UNDECIDABLE,
            f"unresolved criteria {list(score.indeterminate)} leave the score range "
            f"[{score.score_min},{score.score_max}] spanning the threshold",
        )

    if GROUP_TO_CONDITION.get(classifier_diagnosis) is None:
        return (
            RetrievalState.NO_COVERAGE,
            f"the licensed corpus holds no recommendation for '{classifier_diagnosis}'",
        )

    if classifier_confidence < LOW_CONFIDENCE:
        return (
            RetrievalState.CONFLICT,
            f"classifier confidence {classifier_confidence:.2f} is below "
            f"{LOW_CONFIDENCE:.2f}; treat as an unresolved differential",
        )

    return (RetrievalState.ROUTINE, "views agree, score determinate, condition covered")


def route(
    *,
    classifier_diagnosis: str,
    classifier_confidence: float,
    score: SevenPointScore,
    catalog: GuidelineCatalog,
) -> RetrievalDecision:
    """Gate retrieval on the perception state, then execute only what that state needs."""
    state, reason = classify_state(
        classifier_diagnosis=classifier_diagnosis,
        classifier_confidence=classifier_confidence,
        score=score,
    )
    question_types = QUESTION_TYPES_FOR_STATE[state]

    if state is RetrievalState.NO_COVERAGE:
        # The only honest action is to retrieve nothing and say so. Issuing a lookup we know
        # returns empty would inflate the retrieval count without informing anything.
        return RetrievalDecision(
            state=state,
            reason=reason,
            question_types=(),
            conditions_requested=(),
            results={},
            retrieval_calls=0,
            skipped=True,
        )

    conditions: List[str] = [classifier_diagnosis]
    if state is RetrievalState.CONFLICT and classifier_diagnosis != "MEL":
        # Both readings, so the clinician sees guidance for the condition each view is proposing.
        conditions.append("MEL")

    results: Dict[str, BindingResult] = {}
    calls = 0
    for condition in conditions:
        if GROUP_TO_CONDITION.get(condition) is None:
            continue
        results[condition] = catalog.bind(condition, question_types=question_types)
        calls += 1

    return RetrievalDecision(
        state=state,
        reason=reason,
        question_types=question_types,
        conditions_requested=tuple(conditions),
        results=results,
        retrieval_calls=calls,
        skipped=False,
    )


def allowed_retrievals(state: RetrievalState, committed_group: str) -> Set[Tuple[str, str]]:
    """The (diagnosis_group, question_type) pairs a gated ``search_guidelines`` may serve.

    Where ``route()`` executes retrieval itself, this is the same policy expressed as a lookup
    table so a tool layer can check an agent-issued call against it before serving anything --
    the host still executes nothing on the agent's behalf, it only says yes or no.

    ``"*"`` in the question-type slot means any question type is in scope for that condition; a
    literal question type means only that one is. ROUTINE has no wildcard because that state
    knows exactly what it needs (management guidance); CONFLICT and UNDECIDABLE do not, because
    an unresolved differential or an unresolved criterion could reasonably need any guideline
    angle on either candidate condition.
    """
    if state is RetrievalState.NO_COVERAGE:
        # The only honest action is absence; nothing is indicated, so nothing is in-plan.
        return set()

    if state is RetrievalState.ROUTINE:
        return {
            (committed_group, qt)
            for qt in QUESTION_TYPES_FOR_STATE[RetrievalState.ROUTINE]
        }

    # CONFLICT and UNDECIDABLE: both sides of the differential, any question type. A set
    # naturally collapses this to one entry when the committed group is already MEL.
    return {(committed_group, "*"), ("MEL", "*")}
