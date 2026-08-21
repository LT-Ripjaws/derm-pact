"""Shared construction of image-bound claims for a MedGuard evidence contract.

Both the C1+C2 development preflight and the matched multi-arm validator build
contracts from the same retrieved evidence, and they must build them the same
way.  Keeping two copies is what previously let the two paths disagree, so the
rule lives here once.

The rule, in one sentence: the patient's diagnosis is bound to the frozen
perception commitment, and retrieved evidence may corroborate it but is never
asked to prove it.

Concretely:

* one image-bound DIAGNOSIS claim, citing the perception finding and no external
  passage.  A retrieved passage about melanoma in general cannot establish that
  this lesion is melanoma, so asking it to is the conflation the contract exists
  to prevent -- and asking it per passage is what failed every contract closed.
* one IMAGE_EXEMPLAR_SUPPORT claim per retrieved train exemplar that actually
  carries the committed diagnosis.  These are hash-bound and independently
  assessed, so they can be supported.
* a differently labelled neighbour is another lesion, not counterevidence: it
  stays in the contract as retrieved evidence but the contract asserts no claim
  on it, so it creates no obligation it cannot discharge.
* text passages stay as lineage-bound context and assert nothing.

If no retrieved exemplar corroborates the commitment, the caller is told so
(``supporting_exemplars == 0``) and must record an unresolved uncertainty.  That
matters for more than tidiness: without it, an arm that retrieves *nothing*
would assert nothing, fail nothing, and score a free PASS -- so removing evidence
would look like an improvement.  Corroboration has to be earned.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .atomic_claims import AtomicClaimCatalog
from .contract import (
    ClaimType,
    ClinicalClaim,
    EntailmentState,
    FindingState,
    ImageSupport,
    RelationOrigin,
)
from .guideline_claims import GUIDELINE_CHANNEL, build_supported_guideline_claim
from .relation_assessor import (
    CuratedRelationRegistry,
    ImageExemplarMetadata,
    apply_relation_assessment,
    assess_evidence_relation,
)


DIAGNOSIS_CLAIM_ID = "c_diagnosis"
NO_EXEMPLAR_SUPPORT_UNCERTAINTY_ID = "u_image_evidence"
NO_EXEMPLAR_SUPPORT_DESCRIPTION = (
    "No retrieved train exemplar carries the committed diagnosis, so the visual "
    "neighbourhood does not corroborate the commitment."
)
NO_EXEMPLAR_SUPPORT_TERMINATION = (
    "no retrieved exemplar supports the committed diagnosis"
)


def build_image_bound_claims(
    *,
    perception: Any,
    evidence: Sequence[Any],
    record_by_evidence: Mapping[str, Mapping[str, Any]],
    guideline_catalog: Optional[AtomicClaimCatalog] = None,
    guideline_registry: Optional[CuratedRelationRegistry] = None,
) -> Tuple[List[ClinicalClaim], List[Dict[str, str]], int]:
    """Build the claim set for one case.

    Returns ``(claims, relations, supporting_exemplars)``.  ``relations`` is the
    per-assessment audit trail; ``supporting_exemplars`` is how many retrieved
    exemplars independently corroborated the committed diagnosis.

    When ``guideline_catalog``/``guideline_registry`` are supplied, each
    authoritative-guideline evidence item becomes a SUPPORTED source-bound claim by
    source-statement integrity (the claim restates the cited guideline verbatim).
    These are conditional-management / method provenance and never assert the
    patient's diagnosis; retrieval is condition-scoped, so a benign commitment
    reaches no guideline evidence and the contract owes no external claim.
    """
    diagnosis = perception.commitment.concept_code
    claims: List[ClinicalClaim] = [ClinicalClaim(
        claim_id=DIAGNOSIS_CLAIM_ID,
        claim_type=ClaimType.DIAGNOSIS,
        # Clinical phrasing, not system internals: this sentence is the delivered
        # answer's opening line, and "image-committed" read as plumbing rather than a
        # finding (it scored ~0.27 answer-relevancy against a prose baseline's 0.93).
        # The claim binding, target concept and provenance are unchanged.
        text=f"The dermoscopic image findings are most consistent with {diagnosis}.",
        target_concept=diagnosis,
        target_state=FindingState.PRESENT,
        image_finding_ids=("f_commitment",),
        image_support=ImageSupport.SUPPORTS,
        evidence_ids=(),
        entailment=EntailmentState.NOT_ASSESSED,
        evidence_relation_origin=RelationOrigin.NOT_ASSESSED,
    )]

    relations: List[Dict[str, str]] = []
    supporting_exemplars = 0
    for index, item in enumerate(evidence, start=1):
        if item.retrieval_channel == "image_exemplar":
            raw_record = record_by_evidence[item.evidence_id]
            claim = ClinicalClaim(
                claim_id=f"c{index}",
                claim_type=ClaimType.IMAGE_EXEMPLAR_SUPPORT,
                text=(f"A labelled {diagnosis} exemplar from the frozen reference atlas "
                      f"corroborates this appearance."),
                target_concept=diagnosis,
                target_state=FindingState.PRESENT,
                image_finding_ids=(),
                image_support=ImageSupport.NOT_APPLICABLE,
                evidence_ids=(item.evidence_id,),
                entailment=EntailmentState.NOT_ASSESSED,
                evidence_relation_origin=RelationOrigin.NOT_ASSESSED,
            )
            exemplar = ImageExemplarMetadata(
                evidence_id=item.evidence_id,
                evidence_content_sha256=item.content_sha256,
                evidence_record_sha256=item.record_sha256,
                canonical_label=str(raw_record.get("label") or ""),
            )
            assessment = assess_evidence_relation(claim, item, image_exemplar=exemplar)
            assessed = apply_relation_assessment(claim, assessment)
            relations.append({
                "claim_id": assessed.claim_id,
                "claim_type": assessed.claim_type.value,
                "evidence_id": item.evidence_id,
                "relation": assessment.relation.value,
                "reason": assessment.reason,
                "assessment_sha256": assessment.assessment_sha256,
            })
            if assessed.entailment is EntailmentState.SUPPORTED:
                claims.append(assessed)
                supporting_exemplars += 1
        elif (item.retrieval_channel == GUIDELINE_CHANNEL
              and guideline_catalog is not None and guideline_registry is not None):
            atomic = guideline_catalog.lookup(f"guideline.{item.document_id}")
            if atomic is None:
                # A record eligible only for image-bound types backs no source claim;
                # it remains lineage-bound context and asserts nothing.
                continue
            claim = build_supported_guideline_claim(
                claim_id=f"g{index}", evidence=item,
                atomic_claim=atomic, registry=guideline_registry,
            )
            claims.append(claim)
            relations.append({
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type.value,
                "evidence_id": item.evidence_id,
                "relation": EntailmentState.SUPPORTED.value,
                "reason": "host_owned_source_statement_integrity",
                "assessment_sha256": claim.evidence_relation_receipt_sha256,
            })
    return claims, relations, supporting_exemplars
