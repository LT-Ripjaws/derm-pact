"""Deterministic C1 scoring over typed evidence contracts.

This scorer measures structural completeness and derives patient-image support
from host-owned findings.  It never promotes a generator's declared
``image_support`` or ``entailment`` label into independent evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Tuple

from .contract import (
    IMAGE_BOUND_CLAIM_TYPES,
    ClaimType,
    EntailmentState,
    EvidenceContract,
    FindingState,
    FindingType,
    ImageSupport,
    OutputObligation,
    RelationOrigin,
)


_INDEPENDENT_RELATION_ORIGINS = {
    RelationOrigin.HOST_DETERMINISTIC,
    RelationOrigin.HUMAN_REVIEWED,
}
_IMAGE_RELEVANT = {ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL, ClaimType.VISUAL_FINDING}


@dataclass(frozen=True)
class ContractScore:
    valid: bool
    structural_completion: float
    contract_complete: bool
    structured_pass_eligible: bool
    free_text_semantics_assessed: bool
    image_claims_applicable: int
    image_claims_supported: int
    image_claims_contradicted: int
    image_claims_uncertain: int
    image_support_rate: Optional[float]
    declared_image_relation_matches: int
    declared_image_relation_mismatches: int
    independent_evidence_relation_rate: Optional[float]
    output_obligation_results: Mapping[str, bool]
    issues: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_image_support(contract: EvidenceContract, claim: Any) -> ImageSupport:
    """Derive one claim relation solely from host-owned typed findings."""
    if claim.claim_type not in _IMAGE_RELEVANT:
        return ImageSupport.NOT_APPLICABLE
    findings = {
        finding.finding_id: finding for finding in contract.perception.findings
        if finding.finding_id in claim.image_finding_ids
    }
    if claim.claim_type in {ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL}:
        commitment = next(
            (finding for finding in findings.values()
             if finding.finding_type is FindingType.DIAGNOSTIC_COMMITMENT), None)
        if commitment is None or claim.target_state is FindingState.UNCERTAIN:
            return ImageSupport.UNCERTAIN
        if claim.target_concept == commitment.concept_code:
            return (ImageSupport.SUPPORTS if claim.target_state is FindingState.PRESENT
                    else ImageSupport.CONTRADICTS)
        if claim.claim_type is ClaimType.DIFFERENTIAL:
            return ImageSupport.UNCERTAIN
        return (ImageSupport.CONTRADICTS if claim.target_state is FindingState.PRESENT
                else ImageSupport.UNCERTAIN)

    matching = [
        finding for finding in findings.values()
        if finding.finding_type is FindingType.VISUAL_OBSERVATION
        and finding.concept_code == claim.target_concept
    ]
    if not matching:
        return ImageSupport.UNCERTAIN
    if any(finding.state is FindingState.UNCERTAIN for finding in matching):
        return ImageSupport.UNCERTAIN
    if any(finding.state is claim.target_state for finding in matching):
        return ImageSupport.SUPPORTS
    return ImageSupport.CONTRADICTS


def _claim_obligation_satisfied(claim: Any, image_support: ImageSupport) -> bool:
    """Does this claim meet the obligation appropriate to its own type?

    Holding every claim to external entailment is what made all 160 development
    contracts fail closed: an image-bound diagnosis has no external evidence that
    could ever entail it.  The obligation therefore follows the claim:

    * image-bound (diagnosis, differential) -> the frozen perception commitment
      must support it;
    * everything else (image exemplars, source-bound claims) -> an independent,
      hash-bound relation must support it.  A generator's own declaration never
      counts.
    """
    if claim.contradiction or claim.entailment is EntailmentState.CONTRADICTED:
        return False
    if claim.claim_type in IMAGE_BOUND_CLAIM_TYPES:
        return image_support is ImageSupport.SUPPORTS
    return (
        claim.entailment is EntailmentState.SUPPORTED
        and claim.evidence_relation_origin in _INDEPENDENT_RELATION_ORIGINS
    )


def claim_evidence_sufficient(contract: EvidenceContract, claim: Any) -> bool:
    """Does this claim meet the evidence obligation appropriate to its own type?

    Image-bound claims (diagnosis/differential) are discharged by the frozen
    perception commitment; every other claim (image exemplars, source-bound
    guideline claims) by an independent, hash-bound SUPPORTED relation.  This is the
    SAME per-claim rule the PASS gate applies, exposed so the trajectory-trust
    P2 dimension measures type-appropriate sufficiency instead of demanding uniform
    text-entailment (which no image-bound diagnosis can ever meet).
    """
    return _claim_obligation_satisfied(claim, derive_image_support(contract, claim))


def _invalid_score(issue: str) -> ContractScore:
    return ContractScore(
        valid=False, structural_completion=0.0, contract_complete=False,
        structured_pass_eligible=False, free_text_semantics_assessed=False,
        image_claims_applicable=0, image_claims_supported=0,
        image_claims_contradicted=0, image_claims_uncertain=0,
        image_support_rate=None, declared_image_relation_matches=0,
        declared_image_relation_mismatches=0,
        independent_evidence_relation_rate=None,
        output_obligation_results={}, issues=(issue,),
    )


def score_contract(contract: EvidenceContract) -> ContractScore:
    """Score one contract without reading generator prose or mutable external state."""
    try:
        contract.validate()
    except (TypeError, ValueError) as exc:
        return _invalid_score(str(exc))

    claims = list(contract.claims)
    evidence = list(contract.evidence)
    known_evidence = {item.evidence_id for item in evidence}
    structural_checks = {
        "identity": bool(contract.case_id and contract.modality and contract.clinical_question),
        "perception_binding": bool(contract.perception.findings),
        "claims_present": bool(claims),
        "evidence_present": bool(evidence),
        "claim_evidence_binding": bool(claims) and all(
            claim.entailment not in {EntailmentState.SUPPORTED, EntailmentState.CONTRADICTED}
            or bool(claim.evidence_ids) and set(claim.evidence_ids).issubset(known_evidence)
            for claim in claims
        ),
        "lineage_complete": bool(evidence) and all(
            item.source_id and item.source_version and item.authority and item.retrieval_path
            and len(item.content_sha256) == 64 for item in evidence
        ),
        "output_obligations_present": len(contract.output_obligations) == len(OutputObligation),
    }
    structural_completion = sum(structural_checks.values()) / len(structural_checks)

    relations = [(claim, derive_image_support(contract, claim)) for claim in claims]
    applicable = [(claim, relation) for claim, relation in relations
                  if relation is not ImageSupport.NOT_APPLICABLE]
    supported = sum(relation is ImageSupport.SUPPORTS for _, relation in applicable)
    contradicted = sum(relation is ImageSupport.CONTRADICTS for _, relation in applicable)
    uncertain = sum(relation is ImageSupport.UNCERTAIN for _, relation in applicable)
    declared_matches = sum(claim.image_support is relation for claim, relation in applicable)
    declared_mismatches = len(applicable) - declared_matches

    assessed = [claim for claim in claims if claim.entailment is not EntailmentState.NOT_ASSESSED]
    independent = sum(
        claim.evidence_relation_origin in _INDEPENDENT_RELATION_ORIGINS for claim in assessed
    )
    independent_rate = independent / len(assessed) if assessed else None

    externally_contradicted = any(
        claim.contradiction or claim.entailment is EntailmentState.CONTRADICTED
        for claim in claims
    )
    unresolved_conflict = bool(contradicted or externally_contradicted)
    obligations = {
        OutputObligation.CITE_SUPPORTED_CLAIMS.value: all(
            claim.entailment not in {EntailmentState.SUPPORTED, EntailmentState.CONTRADICTED}
            or bool(claim.evidence_ids) and set(claim.evidence_ids).issubset(known_evidence)
            for claim in claims
        ),
        OutputObligation.DISCLOSE_UNRESOLVED_UNCERTAINTY.value: (
            not contract.unresolved_uncertainties
            or contract.proposed_action == "ABSTAIN" and bool(contract.termination_reason.strip())
        ),
        OutputObligation.ABSTAIN_ON_UNRESOLVED_CONFLICT.value: (
            not unresolved_conflict or contract.proposed_action == "ABSTAIN"
        ),
    }
    issues = []
    if declared_mismatches:
        issues.append(
            f"{declared_mismatches} generator image-relation declaration(s) disagree with host findings")
    if uncertain:
        issues.append(f"{uncertain} image-bound claim(s) remain uncertain")
    if contradicted:
        issues.append(f"{contradicted} image-bound claim(s) contradict frozen perception")
    if assessed and independent < len(assessed):
        issues.append("one or more evidence relations are generator-declared, not independently assessed")
    if contract.unresolved_uncertainties:
        issues.append("contract retains unresolved uncertainty")
    for name, satisfied in obligations.items():
        if not satisfied:
            issues.append(f"output obligation not satisfied: {name}")

    contract_complete = (
        structural_completion == 1.0
        and uncertain == 0
        # None means the contract asserted no external claim at all, which is a
        # valid minimal contract (image-bound diagnosis only), not an incomplete
        # one.  Any external claim it does make must be independently assessed.
        and (independent_rate is None or independent_rate == 1.0)
        and all(obligations.values())
    )
    all_obligations_met = bool(claims) and all(
        _claim_obligation_satisfied(claim, relation) for claim, relation in relations
    )
    structured_pass_eligible = (
        contract_complete
        and all_obligations_met
        and contradicted == 0
        and not externally_contradicted
        and not contract.unresolved_uncertainties
    )
    return ContractScore(
        valid=True, structural_completion=structural_completion,
        contract_complete=contract_complete,
        structured_pass_eligible=structured_pass_eligible,
        free_text_semantics_assessed=False,
        image_claims_applicable=len(applicable), image_claims_supported=supported,
        image_claims_contradicted=contradicted, image_claims_uncertain=uncertain,
        image_support_rate=(supported / len(applicable) if applicable else None),
        declared_image_relation_matches=declared_matches,
        declared_image_relation_mismatches=declared_mismatches,
        independent_evidence_relation_rate=independent_rate,
        output_obligation_results=obligations, issues=tuple(issues),
    )


def score_contract_mapping(value: Mapping[str, Any]) -> ContractScore:
    """Fail closed on missing, unknown, malformed, or non-enum serialized fields."""
    try:
        contract = EvidenceContract.from_mapping(value)
    except (KeyError, TypeError, ValueError) as exc:
        return _invalid_score(str(exc))
    return score_contract(contract)


def contract_sha256(contract: EvidenceContract) -> str:
    """Stable content binding for a validated contract; this is not a truth score."""
    payload = json.dumps(
        contract.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
