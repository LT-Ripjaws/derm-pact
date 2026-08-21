"""Source-statement-integrity claims derived from the frozen guideline catalog.

Each verbatim guideline recommendation becomes a host-owned ``AtomicSourceClaim``
-- a source-scoped MANAGEMENT/TREATMENT/REFERRAL/ASSESSMENT statement, never a
patient-diagnosis claim -- plus a ``CuratedRelationRegistry`` ENTAILS entry that
asserts only that the claim faithfully restates its cited source record, hash for
hash (the claim text IS the record text). This is *source-statement integrity*,
NOT independently curated semantic relation gold, and it must never be read as
proof that the patient has the committed diagnosis; that stays image-bound. See
research/experiments/medguard_h2h_v1/guideline_catalog_feasibility_v1.json (F4):
the catalog's honest roles are method licensure and conditional management.

diagnosis / visual_finding / differential are image-bound claim types and are
never satisfiable by source text, so a record is bound only through the strongest
management-first source-bound type it is eligible for.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .atomic_claims import AtomicClaimCatalog, AtomicSourceClaim, SourcePassage
from .contract import ClaimType, ClinicalClaim, EvidenceItem
from .relation_assessor import (
    AtomicClaimRelationEntry,
    CuratedRelationRegistry,
    EvidenceRelation,
    apply_relation_assessment,
    assess_evidence_relation,
)


GUIDELINE_CHANNEL = "authoritative_guideline_catalog"

# Management-first: the strongest actionable claim a guideline can back for an
# already-committed condition. Ordered by clinical specificity.
SOURCE_BOUND_PREFERENCE: Tuple[str, ...] = (
    "management", "treatment", "referral", "assessment",
)


def source_bound_claim_type(eligible_claim_types: Sequence[str]) -> Optional[ClaimType]:
    """The strongest source-bound ClaimType a record is eligible for, or None."""
    eligible = {str(value) for value in eligible_claim_types}
    for name in SOURCE_BOUND_PREFERENCE:
        if name in eligible:
            return ClaimType(name)
    return None


def atomic_claim_for_record(record: Mapping[str, Any]) -> Optional[AtomicSourceClaim]:
    """A host-owned atomic source claim for one guideline record, or None."""
    claim_type = source_bound_claim_type(record.get("eligible_claim_types", ()))
    if claim_type is None:
        return None
    claim = AtomicSourceClaim(
        claim_key=f"guideline.{record['document_id']}",
        text=str(record["text"]),
        claim_type=claim_type,
        source_passage=SourcePassage(
            source_id=str(record["source_id"]),
            source_version=str(record["source_version"]),
            document_id=str(record["document_id"]),
            citation_id=str(record["citation_id"]),
            content_sha256=str(record["content_sha256"]),
            record_sha256=str(record["record_sha256"]),
        ),
    )
    claim.validate()
    return claim


def build_guideline_claim_assets(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[AtomicClaimCatalog, CuratedRelationRegistry]:
    """Host-owned atomic catalog + source-statement-integrity ENTAILS registry.

    The ENTAILS relation asserts only that each claim faithfully restates its cited
    source (verified because the claim text hashes to the record's content_sha256).
    It is not semantic gold and must not be described as proving a patient diagnosis.
    """
    claims: List[AtomicSourceClaim] = []
    for record in records:
        claim = atomic_claim_for_record(record)
        if claim is not None:
            claims.append(claim)
    catalog = AtomicClaimCatalog(claims)
    registry = CuratedRelationRegistry(atomic_entries=tuple(
        AtomicClaimRelationEntry.from_claim(claim, EvidenceRelation.ENTAILS)
        for claim in claims
    ))
    return catalog, registry


def guideline_evidence_item(record: Mapping[str, Any], *, evidence_id: str) -> EvidenceItem:
    """The canonical EvidenceItem for a guideline record (runtime + tests share it)."""
    item = EvidenceItem(
        evidence_id=str(evidence_id),
        text=str(record["text"]),
        source_id=str(record["source_id"]),
        source_version=str(record["source_version"]),
        authority=str(record.get("authority_class") or record.get("authority")),
        retrieval_path=(GUIDELINE_CHANNEL,),
        content_sha256=str(record["content_sha256"]),
        document_id=str(record["document_id"]),
        citation_id=str(record["citation_id"]),
        eligible_claim_types=tuple(str(value) for value in record["eligible_claim_types"]),
        retrieval_channel=GUIDELINE_CHANNEL,
        record_sha256=str(record["record_sha256"]),
    )
    item.validate()
    return item


# The shared clinical question is diagnosis/assessment-oriented, so guideline
# citations should LEAD with assessment-sanctioning records and add management
# context, not be dominated by whichever source_id sorts first alphabetically.
# Cross-issuer ingestion exposed this: AAD records (document_id "aad:*", all
# management-first) sort before EADO/NICE and, under a plain id-sorted top-k,
# displaced every assessment record so the citation mix went 100% management.
GUIDELINE_SELECTION_PRIORITY: Tuple[str, ...] = (
    "assessment", "management", "treatment", "referral",
)


def select_guideline_evidence(
    retriever: Any,
    committed_diagnosis: str,
    k: int,
    *,
    priority: Tuple[str, ...] = GUIDELINE_SELECTION_PRIORITY,
) -> List[Any]:
    """Claim-type-stratified, deterministic top-k over the guideline catalog.

    Round-robins the priority strata (assessment first) so the selected citations
    reflect the clinical question rather than document_id alphabetical order. Within
    a stratum records keep the retriever's stable document_id order; no relevance
    scorer is introduced -- the only signal used is the structured, source-declared
    eligible_claim_types. Every returned record is source-bound eligible (the four
    priority types), so none is wasted on an image-bound-only record.
    """
    if k <= 0:
        return []
    strata = [list(retriever.search(committed_diagnosis, claim_type)) for claim_type in priority]
    cursors = [0] * len(strata)
    selected: List[Any] = []
    seen: set = set()
    while len(selected) < k:
        progressed = False
        for index, stratum in enumerate(strata):
            while cursors[index] < len(stratum):
                evidence = stratum[cursors[index]]
                cursors[index] += 1
                if evidence.document_id not in seen:
                    seen.add(evidence.document_id)
                    selected.append(evidence)
                    progressed = True
                    break
            if len(selected) >= k:
                break
        if not progressed:
            break
    return selected[:k]


def build_supported_guideline_claim(
    *,
    claim_id: str,
    evidence: EvidenceItem,
    atomic_claim: AtomicSourceClaim,
    registry: CuratedRelationRegistry,
) -> ClinicalClaim:
    """Bind + assess one guideline citation into a SUPPORTED source-bound claim.

    Support here is host-deterministic source-statement integrity, discharged by
    the registry ENTAILS entry; it never asserts the patient's diagnosis.
    """
    bound = atomic_claim.bind(claim_id=claim_id, evidence=evidence)
    assessment = assess_evidence_relation(
        bound, evidence, registry=registry, atomic_claim=atomic_claim,
    )
    if assessment.relation is not EvidenceRelation.ENTAILS:
        raise ValueError(
            f"guideline claim {claim_id!r} did not resolve to ENTAILS: {assessment.reason}")
    return apply_relation_assessment(bound, assessment)
