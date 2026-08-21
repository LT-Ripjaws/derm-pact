"""Trajectory-level CATIA/GRiP-aligned measurement over evidence contracts.

Dimensions remain separate primary outputs. The composite is secondary and uses
equal weights only as a compact diagnostic; NIST/CATIA/GRiP are interpretation
frameworks, not algorithmic novelty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..evidence import (
    EntailmentState, EvidenceContract, VerificationState,
    claim_evidence_sufficient, score_contract,
)
from ..remediation import Action


AUTONOMY_LEVEL = "L1-advisory"


@dataclass(frozen=True)
class ProcessTrustProfile:
    dimensions: Dict[str, Optional[float]]
    composite: float
    autonomy: str = AUTONOMY_LEVEL

    def as_dict(self) -> Dict[str, Any]:
        return {"dimensions": self.dimensions, "composite": self.composite,
                "autonomy": self.autonomy}


def _mean_applicable(values: Dict[str, Optional[float]]) -> float:
    applicable = [float(value) for value in values.values() if value is not None]
    return sum(applicable) / len(applicable) if applicable else 0.0


def score_process_trust(
    contract: EvidenceContract, trajectory: Optional[Sequence[Any]] = None,
) -> ProcessTrustProfile:
    """Score observable process properties; never infer clinical correctness."""
    contract.validate()
    trajectory = list(trajectory or [])
    claims = contract.claims
    evidence = contract.evidence
    contract_score = score_contract(contract)
    d: Dict[str, Optional[float]] = {}

    d["P1_image_claim_consistency"] = contract_score.image_support_rate
    # P2 was mis-specified: it demanded EVERY claim carry a SUPPORTED text-entailment,
    # which the image-bound diagnosis claim (discharged by the frozen image, not by
    # retrieved text) can never satisfy -> P2 was 0.0 for every arm. Redefine it as the
    # rate of claims meeting the evidence obligation appropriate to their own type:
    # image-bound claims by image support, source-bound claims by hash-bound SUPPORTED
    # relation + evidence_ids. This reuses the PASS gate's per-claim rule so it cannot
    # drift from it.
    d["P2_evidence_sufficiency"] = (
        round(sum(claim_evidence_sufficient(contract, claim) for claim in claims)
              / len(claims), 4)
        if claims else 0.0
    )
    d["P3_provenance_completeness"] = (
        1.0 if evidence and all(
            item.source_id and item.source_version and item.authority and item.retrieval_path
            and len(item.content_sha256) == 64 for item in evidence) else 0.0
    )
    known = {item.evidence_id for item in evidence}
    d["P4_claim_evidence_binding"] = (
        1.0 if claims and all(set(claim.evidence_ids).issubset(known) for claim in claims) else 0.0
    )
    contradictions = any(claim.contradiction for claim in claims)
    used_independent = any(getattr(step, "action", None) is Action.INDEPENDENT_SOURCE
                           for step in trajectory)
    d["P5_contradiction_handling"] = (
        1.0 if not contradictions else (1.0 if used_independent or
                                        contract.proposed_action == Action.ABSTAIN.value else 0.0)
    )
    d["P6_uncertainty_handling"] = (
        1.0 if not contract.unresolved_uncertainties else
        (1.0 if contract.proposed_action == Action.ABSTAIN.value else 0.0)
    )
    d["P7_bounded_action_appropriateness"] = (
        1.0 if contract.proposed_action in {Action.PASS.value, Action.ABSTAIN.value}
        and len(trajectory) <= 4 else 0.0
    )
    d["P8_receipt_completeness"] = (
        1.0 if trajectory and contract.termination_reason and
        contract.verification_state in {VerificationState.SATISFIED, VerificationState.FAILED_CLOSED}
        else 0.0
    )
    return ProcessTrustProfile(d, _mean_applicable(d))


# Compatibility alias for historical callers that survived outside the active suite.
score_process_trust_v2 = score_process_trust
