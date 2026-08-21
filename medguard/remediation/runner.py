"""Transport-independent bounded agentic loop over evidence contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol

from ..evidence.contract import EvidenceContract, VerificationState
from ..evidence.verification import FailureType, verify_contract
from .actions import Action
from .policy import DeterministicRemediationPolicy


class RemediationTools(Protocol):
    def expand(self, contract: EvidenceContract) -> EvidenceContract: ...
    def reformulate(self, contract: EvidenceContract) -> EvidenceContract: ...
    def independent_source(self, contract: EvidenceContract) -> EvidenceContract: ...


@dataclass(frozen=True)
class RemediationStep:
    step: int
    failures: tuple[FailureType, ...]
    action: Action


@dataclass
class RemediationResult:
    contract: EvidenceContract
    final_action: Action
    trajectory: List[RemediationStep] = field(default_factory=list)


def run_remediation(
    contract: EvidenceContract, policy: DeterministicRemediationPolicy,
    tools: RemediationTools, *, max_steps: int = 4,
) -> RemediationResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    counts: Dict[Action, int] = {}
    trajectory: List[RemediationStep] = []
    current = contract
    for step in range(1, max_steps + 1):
        report = verify_contract(current)
        action = policy.decide(report, counts)
        trajectory.append(RemediationStep(step, report.failures, action))
        if action is Action.PASS:
            current.verification_state = VerificationState.SATISFIED
            current.proposed_action = Action.PASS.value
            current.termination_reason = "evidence contract satisfied"
            current.validate()
            return RemediationResult(current, action, trajectory)
        if action is Action.ABSTAIN:
            current.verification_state = VerificationState.FAILED_CLOSED
            current.proposed_action = Action.ABSTAIN.value
            current.termination_reason = "; ".join(report.issues) or "repair budget exhausted"
            current.validate()
            return RemediationResult(current, action, trajectory)
        counts[action] = counts.get(action, 0) + 1
        if action is Action.EXPAND:
            current = tools.expand(current)
        elif action is Action.REFORMULATE:
            current = tools.reformulate(current)
        else:
            current = tools.independent_source(current)
        current.verification_state = VerificationState.NEEDS_REMEDIATION
    current.verification_state = VerificationState.FAILED_CLOSED
    current.proposed_action = Action.ABSTAIN.value
    current.termination_reason = "step budget exhausted"
    current.validate()
    return RemediationResult(current, Action.ABSTAIN, trajectory)
