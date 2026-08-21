"""Deterministic failure-to-action hierarchy with explicit exhaustion behavior."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..evidence.verification import FailureType, VerificationReport
from .actions import Action


@dataclass(frozen=True)
class ActionBudget:
    max_total_repairs: int = 3
    per_action: Mapping[Action, int] = field(default_factory=lambda: {
        Action.EXPAND: 1,
        Action.REFORMULATE: 1,
        Action.INDEPENDENT_SOURCE: 1,
    })


class DeterministicRemediationPolicy:
    """Map typed failures to one auditable action; never learn from sealed outcomes."""

    def __init__(self, budget: ActionBudget = ActionBudget()):
        self.budget = budget

    def decide(self, report: VerificationReport, counts: Mapping[Action, int]) -> Action:
        if report.ok:
            return Action.PASS
        failures = set(report.failures)
        if failures.intersection({
            FailureType.INVALID_CONTRACT,
            FailureType.IMAGE_EVIDENCE_CONFLICT,
            FailureType.UNVERIFIED_RELATION,
            FailureType.UNRESOLVED_UNCERTAINTY,
        }):
            return Action.ABSTAIN
        repairs_used = sum(int(counts.get(action, 0)) for action in self.budget.per_action)
        if repairs_used >= self.budget.max_total_repairs:
            return Action.ABSTAIN
        priority = (
            (FailureType.SOURCE_CONTRADICTION, Action.INDEPENDENT_SOURCE),
            (FailureType.MISSING_EVIDENCE, Action.EXPAND),
            (FailureType.MISALIGNED_EVIDENCE, Action.REFORMULATE),
        )
        for failure, action in priority:
            if failure in failures and int(counts.get(action, 0)) < int(self.budget.per_action.get(action, 0)):
                return action
        return Action.ABSTAIN
