from __future__ import annotations

import hashlib

import pytest

from medguard.evidence import (
    AtomicClaimCatalog,
    AtomicClaimRelationEntry,
    AtomicSourceClaim,
    ClaimType,
    CuratedRelationRegistry,
    EvidenceItem,
    EvidenceRelation,
    SourcePassage,
    assess_evidence_relation,
)


def _evidence(*, evidence_id: str = "e1", path=("initial", "dense"),
              text: str = "Dermoscopy improves diagnostic accuracy for BCC."):
    return EvidenceItem(
        evidence_id=evidence_id,
        text=text,
        source_id="PMID:26450622",
        source_version="pubmed-snapshot-2026-07-17",
        authority="peer_reviewed_biomedical_literature",
        retrieval_path=path,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        provenance_uri="https://pubmed.ncbi.nlm.nih.gov/26450622/",
        document_id="pubmed:26450622:abstract",
        citation_id="PMID:26450622",
        eligible_claim_types=("assessment",),
        retrieval_channel="bm25+dense",
        record_sha256="a" * 64,
    )


def _atomic() -> AtomicSourceClaim:
    evidence = _evidence()
    return AtomicSourceClaim(
        claim_key="bcc.dermoscopy_diagnostic_accuracy",
        text="Dermoscopy improves clinical diagnostic accuracy for BCC.",
        claim_type=ClaimType.ASSESSMENT,
        source_passage=SourcePassage(
            source_id=evidence.source_id,
            source_version=evidence.source_version,
            document_id=evidence.document_id,
            citation_id=evidence.citation_id,
            content_sha256=evidence.content_sha256,
            record_sha256=evidence.record_sha256,
        ),
    )


def test_atomic_claim_is_source_and_rank_bound_but_never_self_validating():
    evidence = _evidence()
    atomic = _atomic()
    claim = atomic.bind(claim_id="c1", evidence=evidence)

    result = assess_evidence_relation(
        claim, evidence, registry=CuratedRelationRegistry(), atomic_claim=atomic,
    )
    assert result.relation is EvidenceRelation.UNKNOWN
    assert result.reason == "no_independent_relation_basis"


def test_atomic_relation_entry_survives_evidence_rank_and_path_changes():
    original = _evidence()
    replayed = _evidence(evidence_id="e99", path=("remediation", "reformulate"))
    atomic = _atomic()
    registry = CuratedRelationRegistry(
        atomic_entries=(AtomicClaimRelationEntry.from_claim(
            atomic, EvidenceRelation.ENTAILS,
        ),),
    )

    original_result = assess_evidence_relation(
        atomic.bind(claim_id="c1", evidence=original), original,
        registry=registry, atomic_claim=atomic,
    )
    replay_result = assess_evidence_relation(
        atomic.bind(claim_id="c2", evidence=replayed), replayed,
        registry=registry, atomic_claim=atomic,
    )
    assert original_result.relation is EvidenceRelation.ENTAILS
    assert replay_result.relation is EvidenceRelation.ENTAILS
    assert original_result.reason == replay_result.reason == "exact_atomic_registry_match"


def test_atomic_claim_or_source_mismatch_fails_closed():
    evidence = _evidence()
    atomic = _atomic()
    registry = CuratedRelationRegistry(
        atomic_entries=(AtomicClaimRelationEntry.from_claim(
            atomic, EvidenceRelation.ENTAILS,
        ),),
    )
    changed_source = _evidence(text="The source passage has changed.")
    with pytest.raises(ValueError, match="does not match"):
        atomic.bind(claim_id="c1", evidence=changed_source)

    different_claim = AtomicSourceClaim(
        claim_key=atomic.claim_key,
        text="A different source claim.",
        claim_type=ClaimType.ASSESSMENT,
        source_passage=atomic.source_passage,
    )
    assert registry.lookup_atomic(different_claim) is None
    result = assess_evidence_relation(
        atomic.bind(claim_id="c1", evidence=evidence), evidence,
        registry=registry, atomic_claim=different_claim,
    )
    assert result.relation is EvidenceRelation.UNKNOWN
    assert result.reason == "atomic_source_claim_mismatch"


def test_catalog_and_registry_round_trip_with_no_implicit_relation_labels():
    atomic = _atomic()
    catalog = AtomicClaimCatalog((atomic,))
    assert AtomicClaimCatalog.from_mapping(catalog.to_dict()).catalog_sha256 == catalog.catalog_sha256

    registry = CuratedRelationRegistry(
        atomic_entries=(AtomicClaimRelationEntry.from_claim(
            atomic, EvidenceRelation.CONTRADICTS,
        ),),
    )
    restored = CuratedRelationRegistry.from_mapping(registry.to_dict())
    assert restored.registry_sha256 == registry.registry_sha256
    assert restored.lookup_atomic(atomic) is EvidenceRelation.CONTRADICTS

    with pytest.raises(ValueError, match="duplicate atomic claim key"):
        AtomicClaimCatalog((atomic, atomic))
