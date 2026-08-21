"""Small deterministic fixtures for the living C1 contract tests."""
from __future__ import annotations

import hashlib

from medguard.evidence import (
    ClinicalClaim,
    ClaimType,
    EntailmentState,
    EvidenceContract,
    EvidenceItem,
    FindingOrigin,
    FindingState,
    FindingType,
    HAM7_CLASS_ORDER,
    ImageFinding,
    ImageSupport,
    PerceptionBinding,
    RelationOrigin,
    UncertaintyType,
    UnresolvedUncertainty,
)


RECORD_SHA = "1" * 64


def perception(*, diagnosis: str = "melanoma", visual_findings=()) -> PerceptionBinding:
    probabilities = [0.02, 0.02, 0.02, 0.01, 0.90, 0.02, 0.01]
    if diagnosis != "melanoma":
        top = HAM7_CLASS_ORDER.index(diagnosis)
        probabilities[4], probabilities[top] = probabilities[top], probabilities[4]
    confidence = probabilities[HAM7_CLASS_ORDER.index(diagnosis)]
    findings = [ImageFinding(
        finding_id="f_commitment", finding_type=FindingType.DIAGNOSTIC_COMMITMENT,
        concept_code=diagnosis, state=FindingState.PRESENT, confidence=confidence,
        origin=FindingOrigin.FROZEN_PERCEPTION, source_record_sha256=RECORD_SHA,
    )]
    findings.extend(visual_findings)
    return PerceptionBinding(
        artifact_id="development-dermlip-predictions",
        artifact_sha256="2" * 64,
        record_sha256=RECORD_SHA,
        model_id="redlessone/DermLIP_ViT-B-16",
        class_order=HAM7_CLASS_ORDER,
        diagnosis_probabilities=tuple(probabilities),
        view_image_sha256s=("3" * 64,),
        findings=tuple(findings),
    )


def evidence_item(text: str = "Melanoma requires prompt specialist assessment.") -> EvidenceItem:
    return EvidenceItem(
        evidence_id="e1", text=text, source_id="NICE-NG12", source_version="2025",
        authority="professional_guideline", retrieval_path=("text_dense", "guideline"),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        provenance_uri="https://www.nice.org.uk/guidance/ng12",
        eligible_claim_types=("diagnosis", "assessment", "referral"),
    )


def diagnosis_claim(
    *, target: str = "melanoma", declared: ImageSupport = ImageSupport.SUPPORTS,
    entailment: EntailmentState = EntailmentState.SUPPORTED,
    origin: RelationOrigin = RelationOrigin.HOST_DETERMINISTIC,
    contradiction: bool = False, with_evidence: bool = True,
) -> ClinicalClaim:
    return ClinicalClaim(
        claim_id="c1", claim_type=ClaimType.DIAGNOSIS,
        text=f"The lesion is most consistent with {target}.",
        target_concept=target, target_state=FindingState.PRESENT,
        image_finding_ids=("f_commitment",), image_support=declared,
        evidence_ids=("e1",) if with_evidence else (), entailment=entailment,
        contradiction=contradiction, evidence_relation_origin=origin,
        evidence_relation_assessor_id=(
            "generator_declared" if origin is RelationOrigin.GENERATOR_DECLARED
            else "c1-deterministic-fixture" if origin is RelationOrigin.HOST_DETERMINISTIC
            else ""),
        evidence_relation_receipt_sha256=(
            "4" * 64 if origin is RelationOrigin.HOST_DETERMINISTIC else ""),
    )


def contract(
    *, claim: ClinicalClaim | None = None, evidence=None,
    perception_binding: PerceptionBinding | None = None,
    uncertainty: bool = False,
) -> EvidenceContract:
    claim = claim or diagnosis_claim()
    uncertainties = ([UnresolvedUncertainty(
        uncertainty_id="u1", uncertainty_type=UncertaintyType.IMAGE_SUPPORT,
        description="Image support remains unresolved.", related_claim_ids=(claim.claim_id,),
    )] if uncertainty else [])
    return EvidenceContract(
        case_id="HAM_1", modality="dermoscopy", clinical_question="What is the diagnosis?",
        perception=perception_binding or perception(), claims=[claim],
        evidence=list([evidence_item()] if evidence is None else evidence),
        unresolved_uncertainties=uncertainties,
    )
