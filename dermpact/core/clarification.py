"""WP-A2: decision-relevant clarification analysis."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dermpact.guidelines.rules import ApplicabilityState, RuleStatus

from dermpact.core import versions as V
from dermpact.core.record_v2 import (
    CLINICIAN_ASSERTION_FACTS,
    NEXT_STEP_PRIORITY,
    AmendmentDeltaV2,
    StructuredCaseRecordV2,
)

CLARIFICATION_VERSION = V.CLARIFICATION_VERSION
ANALYSIS_CAVEAT = "not an expected-utility calculation; counts branches only"

_NEVER_DECISIVE_REASON: Mapping[str, str] = {
    "patient.total_nevus_count": "counts are not consumed as a decisive fact in v2; assert patient.high_total_nevus_count directly",
    "patient.personal_melanoma_history": "not required by any executable catalogue rule",
    "patient.genetic_predisposition": "not required by any executable catalogue rule",
}

_QUESTION_TEXT: Mapping[str, str] = {
    "clinician.melanoma_suspected": "Is melanoma clinically suspected for this lesion (yes/no)?",
    "clinician.equivocal": "Is this lesion clinically or dermatoscopically equivocal (yes/no)?",
    "lesion.under_assessment": "Is a pigmented or non-pigmented skin lesion being assessed (yes/no)?",
    "patient.high_risk_status": "Is this patient in a known melanoma high-risk group (genetic predisposition, personal melanoma history, high total nevus count) (yes/no)?",
    "patient.high_total_nevus_count": "Does the patient have a high total nevus count (about 100 nevi or more) (yes/no)?",
    "site.rcm_available": "Is reflectance confocal microscopy available at this site (yes/no)?",
    "patient.total_nevus_count": "What is the patient's total nevus count?",
    "patient.personal_melanoma_history": "Has the patient had a personal history of melanoma (yes/no)?",
    "patient.genetic_predisposition": "Does the patient have a known genetic predisposition to melanoma (yes/no)?",
}


@dataclass(frozen=True)
class ClarificationQuestion:
    fact: str
    question_text: str
    permitted_values: Tuple[Any, ...]
    affected_rules: Tuple[str, ...]
    applicability_changes: Dict[str, Dict[str, str]]
    next_step_changes: Dict[str, Dict[str, Any]]
    changes_next_step: bool
    decision_relevant: bool
    reason: str
    priority_rank: int
    current_record_version: int
    record_sha256: str
    already_marked_unavailable: bool
    analysis_version: str = CLARIFICATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact, "question_text": self.question_text, "permitted_values": list(self.permitted_values),
            "affected_rules": list(self.affected_rules), "applicability_changes": self.applicability_changes,
            "next_step_changes": self.next_step_changes, "changes_next_step": self.changes_next_step,
            "decision_relevant": self.decision_relevant, "reason": self.reason, "priority_rank": self.priority_rank,
            "current_record_version": self.current_record_version, "record_sha256": self.record_sha256,
            "already_marked_unavailable": self.already_marked_unavailable, "analysis_version": self.analysis_version,
        }


@dataclass(frozen=True)
class ClarificationAnalysis:
    case_id: str
    record_version: int
    record_sha256: str
    questions: Tuple[ClarificationQuestion, ...]
    analysis_version: str = CLARIFICATION_VERSION
    caveat: str = ANALYSIS_CAVEAT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id, "record_version": self.record_version, "record_sha256": self.record_sha256,
            "questions": [q.to_dict() for q in self.questions], "analysis_version": self.analysis_version,
            "caveat": self.caveat,
        }


def _next_step_summary(record: StructuredCaseRecordV2) -> Dict[str, Any]:
    return {"status": record.next_step.status, "rule_id": record.next_step.rule_id, "action_code": record.next_step.action_code}


def _unavailable_facts_from_episode_events(episode_events: Sequence[Any]) -> frozenset:
    """Accepts either raw event-like mappings/objects (``event_type``/ ``payload`` attributes or keys) as produced by :mod:`dermpact.core.episode`, or an already-derived :class:`~dermpact.core.episode.EpisodeState` exposing ``unavailable_facts``."""
    if hasattr(episode_events, "unavailable_facts"):
        return frozenset(episode_events.unavailable_facts)
    unavailable: set = set()
    for ev in episode_events:
        etype = ev.get("event_type") if isinstance(ev, Mapping) else getattr(ev, "event_type", None)
        payload = ev.get("payload") if isinstance(ev, Mapping) else getattr(ev, "payload", {})
        if etype == "awaiting_information" and payload:
            fact = payload.get("fact")
            if payload.get("unavailable"):
                unavailable.add(fact)
            else:
                unavailable.discard(fact)
        elif etype == "information_received" and payload:
            fact = payload.get("fact")
            if fact is not None:
                unavailable.discard(fact)
    return frozenset(unavailable)


def _rules_requiring(catalogue, fact: str) -> Tuple[Any, ...]:
    return tuple(
        r for r in catalogue.records
        if r.status is RuleStatus.EXECUTABLE and fact in r.required_facts
    )


def _rule_state(record: StructuredCaseRecordV2, rule_id: str) -> Optional[str]:
    for row in record.applicability.get("results", []):
        if row["rule_id"] == rule_id:
            return row["state"]
    return None


def _priority_index(rule_id: str) -> int:
    try:
        return NEXT_STEP_PRIORITY.index(rule_id)
    except ValueError:
        return len(NEXT_STEP_PRIORITY)


_NOT_RELEVANT_RANK = (len(NEXT_STEP_PRIORITY) + 1) * 100


def decision_relevant_questions(
    record: StructuredCaseRecordV2, *, engine, catalogue, calibrators: Optional[Mapping[str, Any]] = None,
    bundle: Optional[Any] = None, episode_events: Sequence[Any] = (),
) -> ClarificationAnalysis:
    """Enumerate every currently-unknown clinician-assertion fact and report which executable rules, and whether the displayed next step, could change depending on the answer."""
    before_sha = record.record_sha256
    already_unavailable = _unavailable_facts_from_episode_events(episode_events)

    candidates: List[Tuple[str, Any]] = []  # (fact, current_value)
    for fact in CLINICIAN_ASSERTION_FACTS:
        current = record.clinician_assertions[fact]["value"]
        if current is not None:
            continue
        if fact in already_unavailable:
            continue
        candidates.append((fact, current))

    questions: List[ClarificationQuestion] = []
    for fact, current in candidates:
        if fact in _NEVER_DECISIVE_REASON:
            permitted: Tuple[Any, ...] = ("integer",) if fact == "patient.total_nevus_count" else (True, False)
            questions.append(ClarificationQuestion(
                fact=fact, question_text=_QUESTION_TEXT[fact], permitted_values=permitted, affected_rules=(),
                applicability_changes={}, next_step_changes={}, changes_next_step=False, decision_relevant=False,
                reason=_NEVER_DECISIVE_REASON[fact], priority_rank=_NOT_RELEVANT_RANK, current_record_version=record.identity.record_version,
                record_sha256=record.record_sha256, already_marked_unavailable=False,
            ))
            continue

        candidate_rules = _rules_requiring(catalogue, fact)
        if not candidate_rules:
            questions.append(ClarificationQuestion(
                fact=fact, question_text=_QUESTION_TEXT.get(fact, f"Please provide: {fact}"), permitted_values=(True, False),
                affected_rules=(), applicability_changes={}, next_step_changes={}, changes_next_step=False,
                decision_relevant=False, reason="not required by any executable catalogue rule", priority_rank=_NOT_RELEVANT_RANK,
                current_record_version=record.identity.record_version, record_sha256=record.record_sha256,
                already_marked_unavailable=False,
            ))
            continue

        hyp_true = record.with_amendment(
            delta=AmendmentDeltaV2(kind="clinician_fact", field=fact, before=current, after=True,
                                   actor="clarification_analysis", session_id="analysis"),
            catalogue=catalogue, engine=engine, calibrators=calibrators, created_at=record.identity.created_at, bundle=bundle,
        )
        hyp_false = record.with_amendment(
            delta=AmendmentDeltaV2(kind="clinician_fact", field=fact, before=current, after=False,
                                   actor="clarification_analysis", session_id="analysis"),
            catalogue=catalogue, engine=engine, calibrators=calibrators, created_at=record.identity.created_at, bundle=bundle,
        )

        affected: List[str] = []
        true_states: Dict[str, str] = {}
        false_states: Dict[str, str] = {}
        best_priority = len(NEXT_STEP_PRIORITY)
        best_secondary = 0
        for rule in candidate_rules:
            rid = rule.rule_id
            st_true = _rule_state(hyp_true, rid)
            st_false = _rule_state(hyp_false, rid)
            if st_true != st_false:
                affected.append(rid)
                true_states[rid] = st_true
                false_states[rid] = st_false
                idx = _priority_index(rid)
                if idx < best_priority:
                    best_priority = idx
                    try:
                        best_secondary = list(rule.required_facts).index(fact)
                    except ValueError:
                        best_secondary = 0

        ns_true = _next_step_summary(hyp_true)
        ns_false = _next_step_summary(hyp_false)
        changes_next_step = ns_true != ns_false
        decision_relevant = bool(affected)

        if decision_relevant:
            reason = (
                f"answering changes applicability of {', '.join(sorted(affected))}"
                + (" and the displayed next step" if changes_next_step else "; the displayed next step is already decided by a higher-priority rule")
            )
        else:
            reason = "every candidate rule's applicability is already decided independently of this fact's answer"

        priority_rank = (best_priority, best_secondary) if decision_relevant else (len(NEXT_STEP_PRIORITY) + 1, 0)

        questions.append(ClarificationQuestion(
            fact=fact, question_text=_QUESTION_TEXT.get(fact, f"Please provide: {fact}"), permitted_values=(True, False),
            affected_rules=tuple(sorted(affected)),
            applicability_changes=({"true": true_states, "false": false_states} if affected else {}),
            next_step_changes=({"true": ns_true, "false": ns_false} if changes_next_step else {}),
            changes_next_step=changes_next_step, decision_relevant=decision_relevant, reason=reason,
            priority_rank=priority_rank[0] * 100 + priority_rank[1],
            current_record_version=record.identity.record_version, record_sha256=record.record_sha256,
            already_marked_unavailable=False,
        ))

    questions.sort(key=lambda q: (q.priority_rank, q.fact))
    questions = [dataclasses.replace(q, priority_rank=i) for i, q in enumerate(questions)]

    assert record.record_sha256 == before_sha, "decision_relevant_questions must never mutate the record"
    return ClarificationAnalysis(
        case_id=record.identity.case_id, record_version=record.identity.record_version,
        record_sha256=record.record_sha256, questions=tuple(questions),
    )


__all__ = [
    "ANALYSIS_CAVEAT",
    "CLARIFICATION_VERSION",
    "ClarificationAnalysis",
    "ClarificationQuestion",
    "decision_relevant_questions",
]
