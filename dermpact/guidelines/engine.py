"""Deterministic guideline applicability engine (Phase 4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dermpact.guidelines.facts import FACT_SCHEMA, CaseFacts, FactProvenance, TV
from dermpact.guidelines.rules import (
    ApplicabilityState,
    PredicateOutcome,
    RuleCatalogue,
    RuleRecord,
    RuleStatus,
)


class EntityMismatchError(ValueError):
    """The facts payload is bound to a different case than the one requested."""


#: Human-readable one-question prompts per fact (the rule engine's targeted question).
QUESTION_TEXT: Mapping[str, str] = {
    "clinician.melanoma_suspected": "Is melanoma clinically suspected for this lesion (yes/no)?",
    "lesion.melanoma_suspected": "Is melanoma clinically suspected for this lesion (yes/no)? The checklist range is undecidable.",
    "site.rcm_available": "Is reflectance confocal microscopy available at this site (yes/no)?",
    "patient.high_risk_status": "Is this patient in a known melanoma high-risk group (genetic predisposition, personal melanoma history, high total nevus count) (yes/no)?",
    "patient.high_total_nevus_count": "Does the patient have a high total nevus count (about 100 nevi or more) (yes/no)?",
    "patient.total_nevus_count": "What is the patient's total nevus count?",
    "image.iqa_state": "Is the dermoscopic image of interpretable quality?",
    "image.dermoscopic_present": "Is a dermoscopic image available for this lesion?",
    "checklist.decidable": "One or more checklist criteria are indeterminate; can they be re-read?",
}


@dataclass(frozen=True)
class ApplicabilityResult:
    rule_id: str
    state: ApplicabilityState
    reason_code: str
    action_code: Optional[str]
    action_text: Optional[str]
    conditional_on: Tuple[str, ...]
    decisive_missing_fact: Optional[str]
    question: Optional[str]
    rule_path: Tuple[PredicateOutcome, ...]
    facts_used: Mapping[str, Any]
    checks: Mapping[str, bool]
    source: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "state": self.state.value,
            "reason_code": self.reason_code,
            "action_code": self.action_code,
            "action_text": self.action_text,
            "conditional_on": list(self.conditional_on),
            "decisive_missing_fact": self.decisive_missing_fact,
            "question": self.question,
            "rule_path": [
                {"predicate": p.predicate, "outcome": p.outcome.value, "facts_used": list(p.facts_used)}
                for p in self.rule_path
            ],
            "facts_used": dict(self.facts_used),
            "checks": dict(self.checks),
            "source": dict(self.source),
        }

    @property
    def rule_path_signature(self) -> str:
        return "|".join(f"{p.predicate}={p.outcome.value}" for p in self.rule_path)


@dataclass(frozen=True)
class CaseApplicability:
    case_id: str
    evaluation_date: str
    jurisdiction: str
    results: Tuple[ApplicabilityResult, ...]
    supported_next_steps: Tuple[Tuple[str, str], ...]     # (rule_id, action_code)
    conditional_next_steps: Tuple[Tuple[str, str, str], ...]  # (rule_id, action_code, decisive fact)
    targeted_question: Optional[Tuple[str, str, str]]     # (rule_id, fact, question)
    coverage: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "evaluation_date": self.evaluation_date,
            "jurisdiction": self.jurisdiction,
            "results": [r.to_dict() for r in self.results],
            "supported_next_steps": [list(x) for x in self.supported_next_steps],
            "conditional_next_steps": [list(x) for x in self.conditional_next_steps],
            "targeted_question": list(self.targeted_question) if self.targeted_question else None,
            "coverage": dict(self.coverage),
        }

    def result_for(self, rule_id: str) -> ApplicabilityResult:
        for r in self.results:
            if r.rule_id == rule_id:
                return r
        raise KeyError(rule_id)


class ApplicabilityEngine:
    """Runtime evaluator.  Holds a verified catalogue; evaluates facts; never mutates them."""

    #: Rule priority for the single targeted question: melanoma-pathway first.
    QUESTION_PRIORITY: Tuple[str, ...] = (
        "eado-mel-p1-2024:R3",
        "eado-mel-p1-2024:R4",
        "eado-mel-p1-2024:R7",
        "eado-mel-p1-2024:R6",
        "eado-mel-p1-2024:R5",
    )

    def __init__(self, catalogue: RuleCatalogue, *, covered_jurisdictions: Sequence[str] = ("EU",)):
        self.catalogue = catalogue
        self.covered_jurisdictions = tuple(covered_jurisdictions)

    # -- source/version/instrument checks ---------------------------------

    def _source_current(self, rule: RuleRecord, on: date) -> bool:
        sup = rule.supersession
        if sup.status != "current":
            return False
        try:
            start = date.fromisoformat(sup.valid_from)
            end = date.fromisoformat(sup.valid_until)
        except ValueError:
            return False
        return start <= on <= end

    def _instrument_ok(self, rule: RuleRecord, facts: CaseFacts) -> bool:
        if rule.instrument_id is None:
            return True
        current = facts.scalar("checklist.instrument_id")
        return current == rule.instrument_id

    def _jurisdiction_ok(self, facts: CaseFacts) -> bool:
        return facts.jurisdiction in self.covered_jurisdictions

    # -- per-rule evaluation ----------------------------------------------

    def evaluate_rule(self, rule: RuleRecord, facts: CaseFacts) -> ApplicabilityResult:
        checks = {
            "integrity_ok": rule.verify(),
            "source_current": self._source_current(rule, facts.evaluation_date),
            "jurisdiction_ok": self._jurisdiction_ok(facts),
            "instrument_ok": self._instrument_ok(rule, facts),
            "executable": rule.status is RuleStatus.EXECUTABLE,
        }
        source = rule.source.as_dict()
        base = dict(
            rule_id=rule.rule_id,
            action_code=None,
            action_text=None,
            conditional_on=(),
            decisive_missing_fact=None,
            question=None,
            rule_path=(),
            facts_used={},
            checks=checks,
            source=source,
        )
        if not checks["integrity_ok"]:
            return ApplicabilityResult(state=ApplicabilityState.NO_COVERAGE, reason_code="integrity_failure", **base)
        if not checks["executable"]:
            return ApplicabilityResult(state=ApplicabilityState.NO_COVERAGE, reason_code=f"rule_status_{rule.status.value}", **base)
        if not checks["source_current"]:
            return ApplicabilityResult(state=ApplicabilityState.NO_COVERAGE, reason_code="source_stale_or_superseded", **base)
        if not checks["jurisdiction_ok"]:
            return ApplicabilityResult(state=ApplicabilityState.NO_COVERAGE, reason_code="jurisdiction_not_covered", **base)

        working = facts
        if not checks["instrument_ok"]:
            working = facts.copy()
            for name in (
                "checklist.score_min", "checklist.score_max", "checklist.decidable",
                "checklist.meets_threshold", "lesion.melanoma_suspected", "lesion.equivocal",
                "lesion.direct_checklist_disagreement",
            ):
                if name not in working.values:
                    continue
                # An explicit clinician assertion is not checklist-derived and survives.
                prov = working.provenance.get(name)
                if prov is not None and prov.source == "asserted":
                    continue
                working.values[name] = None
                working.provenance[name] = FactProvenance("engine", "blanked: instrument mismatch")

        trace: List[PredicateOutcome] = []
        pre = rule.prerequisites.evaluate(working, trace) if rule.prerequisites else TV.TRUE
        exc = rule.exclusions.evaluate(working, trace) if rule.exclusions else TV.FALSE
        facts_used = {
            name: (working.values.get(name).value if isinstance(working.values.get(name), TV) else working.values.get(name))
            for name in rule.required_facts
        }
        base["rule_path"] = tuple(trace)
        base["facts_used"] = facts_used

        if exc is TV.TRUE:
            return ApplicabilityResult(state=ApplicabilityState.NOT_APPLICABLE, reason_code="exclusion_true", **base)
        if pre is TV.FALSE:
            return ApplicabilityResult(state=ApplicabilityState.NOT_APPLICABLE, reason_code="prerequisite_false", **base)
        if pre is TV.TRUE and exc is TV.FALSE:
            assert rule.action is not None
            base["action_code"] = rule.action.code
            base["action_text"] = rule.action.text
            base["conditional_on"] = rule.action.conditional_on
            return ApplicabilityResult(state=ApplicabilityState.APPLICABLE, reason_code="prerequisites_satisfied", **base)

        # Some decisive fact is UNKNOWN. Find the single most decisive one.
        decisive = self._decisive_missing_fact(rule, working)
        base["decisive_missing_fact"] = decisive
        base["question"] = QUESTION_TEXT.get(decisive, f"Please provide: {decisive}") if decisive else None
        if rule.action is not None:
            base["action_code"] = rule.action.code
            base["action_text"] = rule.action.text
            base["conditional_on"] = tuple([*rule.action.conditional_on, *( [decisive] if decisive else [] )])
        reason = "instrument_mismatch_blanked_facts" if not checks["instrument_ok"] else "decisive_fact_unknown"
        return ApplicabilityResult(state=ApplicabilityState.UNKNOWN, reason_code=reason, **base)

    def _decisive_missing_fact(self, rule: RuleRecord, facts: CaseFacts) -> Optional[str]:
        """The first UNKNOWN required fact whose resolution could change the state."""
        for name in rule.required_facts:
            if facts.is_known(name):
                continue
            outcomes = set()
            for hypothetical in self._hypotheticals(name):
                probe = facts.copy()
                probe.values[name] = hypothetical
                probe.provenance[name] = FactProvenance("engine", "hypothetical")
                pre = rule.prerequisites.evaluate(probe) if rule.prerequisites else TV.TRUE
                exc = rule.exclusions.evaluate(probe) if rule.exclusions else TV.FALSE
                outcomes.add((pre, exc))
            if len(outcomes) > 1:
                return name
        for name in rule.required_facts:
            if not facts.is_known(name):
                return name
        return None

    @staticmethod
    def _hypotheticals(name: str) -> Tuple[Any, ...]:
        kind = FACT_SCHEMA[name]["kind"]
        if kind == "tv":
            return (TV.TRUE, TV.FALSE)
        if kind == "number":
            return (0, 10 ** 6)
        if kind == "enum":
            return tuple(FACT_SCHEMA[name]["values"])
        return (("x",), ())

    # -- case-level -------------------------------------------------------

    def evaluate_case(self, facts: CaseFacts, *, expected_case_id: Optional[str] = None,
                      rule_ids: Optional[Sequence[str]] = None) -> CaseApplicability:
        if expected_case_id is not None and facts.case_id != expected_case_id:
            raise EntityMismatchError(
                f"facts are bound to {facts.case_id!r}, evaluation requested for {expected_case_id!r}"
            )
        rules = self.catalogue.records if rule_ids is None else tuple(self.catalogue.get(r) for r in rule_ids)
        results = tuple(self.evaluate_rule(rule, facts) for rule in rules)

        supported = tuple((r.rule_id, r.action_code) for r in results
                          if r.state is ApplicabilityState.APPLICABLE and r.action_code)
        conditional = tuple((r.rule_id, r.action_code, r.decisive_missing_fact) for r in results
                            if r.state is ApplicabilityState.UNKNOWN and r.action_code and r.decisive_missing_fact)
        question: Optional[Tuple[str, str, str]] = None
        for rid in self.QUESTION_PRIORITY:
            for r in results:
                if r.rule_id == rid and r.state is ApplicabilityState.UNKNOWN and r.decisive_missing_fact:
                    question = (r.rule_id, r.decisive_missing_fact, r.question or "")
                    break
            if question:
                break
        if question is None:
            for r in results:
                if r.state is ApplicabilityState.UNKNOWN and r.decisive_missing_fact:
                    question = (r.rule_id, r.decisive_missing_fact, r.question or "")
                    break

        n_exec = sum(1 for r in rules if r.status is RuleStatus.EXECUTABLE)
        coverage = {
            "rules_evaluated": len(results),
            "executable_rules": n_exec,
            "applicable": sum(1 for r in results if r.state is ApplicabilityState.APPLICABLE),
            "not_applicable": sum(1 for r in results if r.state is ApplicabilityState.NOT_APPLICABLE),
            "applicability_unknown": sum(1 for r in results if r.state is ApplicabilityState.UNKNOWN),
            "no_catalogue_coverage": sum(1 for r in results if r.state is ApplicabilityState.NO_COVERAGE),
            "any_supported_next_step": bool(supported),
        }
        return CaseApplicability(
            case_id=facts.case_id,
            evaluation_date=facts.evaluation_date.isoformat(),
            jurisdiction=facts.jurisdiction,
            results=results,
            supported_next_steps=supported,
            conditional_next_steps=conditional,
            targeted_question=question,
            coverage=coverage,
        )


__all__ = [
    "ApplicabilityEngine",
    "ApplicabilityResult",
    "CaseApplicability",
    "EntityMismatchError",
    "QUESTION_TEXT",
]
