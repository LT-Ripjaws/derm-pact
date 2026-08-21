"""Bridge from an evidence contract to scorable guideline citations.

Citation integrity is scored over what an arm *cites*, and a contract does not
carry a ready-made list of citations -- it carries typed claims that reference
evidence items.  This module turns the second into the first for the one channel
whose integrity is checkable: the authoritative guideline catalog.

Why only that channel.  Only the guideline catalog is a frozen, hash-verified,
condition-scoped index that a ``FrozenGuidelineRetriever`` can resolve, offer, and
type-check.  The PubMed-abstract channel and the image-exemplar channel are not in
that catalog; scoring an id from either against the guideline retriever would
report every one of them as ``hallucinated_id`` and manufacture a failure that is
an artifact of the wrong denominator.  So a claim's PubMed or exemplar citations
are deliberately dropped here, not scored as intact and not scored as broken.

The provenance fields are read from the arm's own copy of the evidence, because
that is exactly what citation integrity checks: if an arm carried a tampered
source id / version / citation id / content hash under a real document id, the
provenance guard must see the tampered value, not a value re-read from the frozen
record.  When the arm's evidence is faithful (the normal case) these equal the
frozen record and the guard passes.
"""
from __future__ import annotations

from typing import List

from medguard.evidence.contract import EvidenceContract
from medguard.evidence.guideline_claims import GUIDELINE_CHANNEL
from medguard.eval.citation_integrity import (
    AssertedCitation,
    CaseCitationScore,
    CitationResolver,
    score_case,
)


def guideline_citations_from_contract(
    contract: EvidenceContract,
) -> List[AssertedCitation]:
    """Every guideline-channel citation a contract's claims assert.

    One ``AssertedCitation`` is emitted per (claim, cited guideline evidence)
    pair.  A claim that cites three guideline records makes three citations; a
    claim that cites only PubMed or exemplar evidence makes none.  The order is
    claim order then the claim's own evidence order, so the output is
    deterministic for a deterministic contract.
    """
    evidence_by_id = {item.evidence_id: item for item in contract.evidence}
    citations: List[AssertedCitation] = []
    for claim in contract.claims:
        for evidence_id in claim.evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is None:
                # A validated contract has no dangling references; guard anyway so
                # a partially built contract cannot raise here.
                continue
            if item.retrieval_channel != GUIDELINE_CHANNEL:
                continue
            citations.append(AssertedCitation(
                cited_id=item.document_id,
                claim_type=claim.claim_type.value,
                claim_id=claim.claim_id,
                source_id=item.source_id,
                source_version=item.source_version,
                citation_id=item.citation_id,
                content_sha256=item.content_sha256,
            ))
    return citations


def score_contract_citations(
    contract: EvidenceContract,
    committed_diagnosis: str,
    resolver: CitationResolver,
) -> CaseCitationScore:
    """Extract this contract's guideline citations and score them for one lesion.

    A contract that cites no guideline record scores ``all_intact=None`` (no
    citation apparatus), never a free perfect score -- the same anti-gaming rule
    the citation-integrity scorer applies at the case level.
    """
    return score_case(
        committed_diagnosis,
        guideline_citations_from_contract(contract),
        resolver,
    )
