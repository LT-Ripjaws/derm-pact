"""Detectable source contradiction and the independent-source remediation (D3).

The INDEPENDENT_SOURCE action was inert: nothing ever set ``contradiction`` on a
claim, so the deterministic policy never routed to it, and even if it had, the
tool returned the contract unchanged.  Both halves are fixed here.

*Detection.*  A source contradiction is a curated, host-owned fact, never a
lexical or model guess.  ``SourceContradictionRegistry`` flags a specific
``(condition, document_id)`` -- read as "this text-index record's claim conflicts
with the authoritative guideline for this condition".  A flagged record is bound
into the contract as a CONTRADICTED source claim through the existing conservative
relation assessor (origin ``host_deterministic``), so ``verify_contract`` raises
``SOURCE_CONTRADICTION`` and *not* ``UNVERIFIED_RELATION`` -- the distinction that
lets the policy route to INDEPENDENT_SOURCE instead of abstaining outright.  A
flagged record is the one and only way a PubMed-abstract record ever asserts a
claim; unflagged text stays lineage-bound context.

*Routing to a genuinely separate index.*  The remediation tool built here consults
the ``FrozenGuidelineRetriever`` -- the authoritative guideline catalog, a
structurally separate index from the PubMed-abstract text index the contradicted
record came from (and now itself multi-issuer).  It drops the contradicted claim
and its text record and binds an independent guideline record for the same
committed condition, so the evidence set genuinely changes and the contradiction
is resolved by a different source rather than papered over.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .atomic_claims import AtomicSourceClaim, SourcePassage
from .contract import ClaimType, ClinicalClaim, EvidenceItem
from .guideline_claims import source_bound_claim_type
from .relation_assessor import (
    AtomicClaimRelationEntry,
    CuratedRelationRegistry,
    EvidenceRelation,
    apply_relation_assessment,
    assess_evidence_relation,
)


@dataclass(frozen=True)
class ContradictionFlag:
    """One curated conflict between a text-index record and the guideline."""

    condition: str
    document_id: str
    reason: str = "conflicts_with_authoritative_guideline"

    def __post_init__(self) -> None:
        if not str(self.condition).strip() or not str(self.document_id).strip():
            raise ValueError("contradiction flag requires condition and document_id")


class SourceContradictionRegistry:
    """Immutable set of curated source-contradiction flags, keyed by condition+id."""

    def __init__(self, flags: Iterable[ContradictionFlag] = ()) -> None:
        by_key: dict[Tuple[str, str], ContradictionFlag] = {}
        for flag in flags:
            key = (str(flag.condition), str(flag.document_id))
            if key in by_key:
                raise ValueError(f"duplicate contradiction flag: {key}")
            by_key[key] = flag
        self._by_key = by_key

    def flag(self, condition: str, document_id: str) -> Optional[ContradictionFlag]:
        return self._by_key.get((str(condition), str(document_id)))

    def is_flagged(self, condition: str, document_id: str) -> bool:
        return (str(condition), str(document_id)) in self._by_key

    def __len__(self) -> int:
        return len(self._by_key)


def build_contradicted_source_claim(
    *, claim_id: str, evidence: EvidenceItem,
) -> ClinicalClaim:
    """Bind a flagged text record as a host-deterministic CONTRADICTED claim.

    The claim restates the record verbatim and is assessed CONTRADICTS through the
    conservative assessor, so its relation origin is ``host_deterministic`` (never
    generator-declared).  That is what makes ``verify_contract`` surface a clean
    ``SOURCE_CONTRADICTION`` the policy can route on, rather than an
    ``UNVERIFIED_RELATION`` that would force an immediate abstain.
    """
    claim_type = source_bound_claim_type(evidence.eligible_claim_types)
    if claim_type is None:
        raise ValueError(
            f"record {evidence.document_id!r} has no source-bound claim type to contradict"
        )
    atomic = AtomicSourceClaim(
        claim_key=f"contradiction.{evidence.document_id}",
        text=evidence.text,
        claim_type=claim_type,
        source_passage=SourcePassage(
            source_id=evidence.source_id,
            source_version=evidence.source_version,
            document_id=evidence.document_id,
            citation_id=evidence.citation_id,
            content_sha256=evidence.content_sha256,
            record_sha256=evidence.record_sha256,
        ),
    )
    bound = atomic.bind(claim_id=claim_id, evidence=evidence)
    registry = CuratedRelationRegistry(atomic_entries=(
        AtomicClaimRelationEntry.from_claim(atomic, EvidenceRelation.CONTRADICTS),
    ))
    assessment = assess_evidence_relation(
        bound, evidence, registry=registry, atomic_claim=atomic,
    )
    if assessment.relation is not EvidenceRelation.CONTRADICTS:
        raise ValueError(
            f"contradiction claim {claim_id!r} did not resolve to CONTRADICTS: "
            f"{assessment.reason}"
        )
    return apply_relation_assessment(bound, assessment)


def flagged_text_records(
    records: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    registry: SourceContradictionRegistry,
) -> List[Mapping[str, Any]]:
    """The retrieved text records the registry flags as contradicting the guideline."""
    flagged: List[Mapping[str, Any]] = []
    for record in records:
        document_id = str(record.get("document_id") or record.get("id") or "")
        if document_id and registry.is_flagged(condition, document_id):
            flagged.append(record)
    return flagged
