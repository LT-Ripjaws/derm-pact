"""Structured verification over an evidence contract, without free-text safety rules."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .contract import (
    IMAGE_BOUND_CLAIM_TYPES,
    EntailmentState,
    EvidenceContract,
    RelationOrigin,
)
from .scoring import derive_image_support


class FailureType(str, Enum):
    MISSING_EVIDENCE = "missing_evidence"
    MISALIGNED_EVIDENCE = "misaligned_evidence"
    SOURCE_CONTRADICTION = "source_contradiction"
    IMAGE_EVIDENCE_CONFLICT = "image_evidence_conflict"
    UNVERIFIED_RELATION = "unverified_relation"
    UNRESOLVED_UNCERTAINTY = "unresolved_uncertainty"
    INVALID_CONTRACT = "invalid_contract"


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    failures: Tuple[FailureType, ...]
    issues: Tuple[str, ...]


def verify_contract(contract: EvidenceContract) -> VerificationReport:
    failures: list[FailureType] = []
    issues: list[str] = []
    try:
        contract.validate()
    except ValueError as exc:
        return VerificationReport(False, (FailureType.INVALID_CONTRACT,), (str(exc),))

    if not contract.evidence or not contract.claims:
        failures.append(FailureType.MISSING_EVIDENCE)
        issues.append("contract has no evidence or claims")
    for claim in contract.claims:
        # An image-bound claim cites host findings, not retrieved passages, and is
        # discharged by the patient image.  Judging it on evidence_ids and
        # entailment reports a real image-bound diagnosis as both missing and
        # misaligned, which is what previously failed every contract closed.  Its
        # image support is judged by the derived checks further down.
        if claim.claim_type in IMAGE_BOUND_CLAIM_TYPES:
            if not claim.image_finding_ids:
                if FailureType.MISSING_EVIDENCE not in failures:
                    failures.append(FailureType.MISSING_EVIDENCE)
                issues.append(f"claim {claim.claim_id} has no image finding")
        else:
            if not claim.evidence_ids:
                if FailureType.MISSING_EVIDENCE not in failures:
                    failures.append(FailureType.MISSING_EVIDENCE)
                issues.append(f"claim {claim.claim_id} has no evidence")
            if claim.entailment in (
                EntailmentState.INSUFFICIENT, EntailmentState.NOT_ASSESSED,
            ):
                if FailureType.MISALIGNED_EVIDENCE not in failures:
                    failures.append(FailureType.MISALIGNED_EVIDENCE)
                issues.append(
                    f"claim {claim.claim_id} is not supported by the retrieved evidence")
        referenced = [item for item in contract.evidence if item.evidence_id in claim.evidence_ids]
        scoped = [item for item in referenced if item.eligible_claim_types]
        if (claim.entailment is EntailmentState.SUPPORTED and scoped
                and not any(claim.claim_type.value in item.eligible_claim_types for item in scoped)):
            if FailureType.MISALIGNED_EVIDENCE not in failures:
                failures.append(FailureType.MISALIGNED_EVIDENCE)
            issues.append(
                f"claim {claim.claim_id} type {claim.claim_type.value} is outside its cited source scope")
        if claim.contradiction or claim.entailment is EntailmentState.CONTRADICTED:
            if FailureType.SOURCE_CONTRADICTION not in failures:
                failures.append(FailureType.SOURCE_CONTRADICTION)
            issues.append(f"claim {claim.claim_id} is contradicted by external evidence")
        if (claim.entailment is not EntailmentState.NOT_ASSESSED
                and claim.evidence_relation_origin is RelationOrigin.GENERATOR_DECLARED):
            if FailureType.UNVERIFIED_RELATION not in failures:
                failures.append(FailureType.UNVERIFIED_RELATION)
            issues.append(
                f"claim {claim.claim_id} evidence relation is generator-declared only")
        derived_image_support = derive_image_support(contract, claim)
        if derived_image_support.value == "contradicts":
            if FailureType.IMAGE_EVIDENCE_CONFLICT not in failures:
                failures.append(FailureType.IMAGE_EVIDENCE_CONFLICT)
            issues.append(
                f"claim {claim.claim_id} conflicts with host-owned patient-image findings")
        elif derived_image_support.value == "uncertain":
            if FailureType.UNRESOLVED_UNCERTAINTY not in failures:
                failures.append(FailureType.UNRESOLVED_UNCERTAINTY)
            issues.append(f"claim {claim.claim_id} has unresolved image support")
    if contract.unresolved_uncertainties:
        if FailureType.UNRESOLVED_UNCERTAINTY not in failures:
            failures.append(FailureType.UNRESOLVED_UNCERTAINTY)
        issues.append("contract retains unresolved uncertainty")
    return VerificationReport(not failures, tuple(failures), tuple(issues))
