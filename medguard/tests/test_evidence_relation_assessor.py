from __future__ import annotations

import hashlib

from medguard.evidence import (
    ClinicalClaim,
    ClaimType,
    CuratedRelationEntry,
    CuratedRelationRegistry,
    EntailmentState,
    EvidenceItem,
    EvidenceRelation,
    FindingState,
    ImageExemplarMetadata,
    ImageSupport,
    RelationOrigin,
    apply_relation_assessment,
    assess_evidence_relation,
    score_contract,
)
from medguard.tests.c1_fixtures import contract


def _claim(
    *,
    claim_type: ClaimType = ClaimType.MANAGEMENT,
    text: str = "Prompt specialist assessment is indicated.",
    target: str = "",
    generator_declared: bool = False,
) -> ClinicalClaim:
    return ClinicalClaim(
        claim_id="c1",
        text=text,
        image_support=ImageSupport.NOT_APPLICABLE,
        evidence_ids=("e1",),
        entailment=(
            EntailmentState.SUPPORTED
            if generator_declared else EntailmentState.NOT_ASSESSED
        ),
        claim_type=claim_type,
        target_concept=target,
        target_state=(
            FindingState.PRESENT if target else FindingState.UNCERTAIN
        ),
        evidence_relation_origin=(
            RelationOrigin.GENERATOR_DECLARED
            if generator_declared else RelationOrigin.NOT_ASSESSED
        ),
        evidence_relation_assessor_id=(
            "generator_declared" if generator_declared else ""
        ),
    )


def _evidence(
    *,
    text: str = "Prompt specialist assessment is indicated.",
    eligible=("management",),
    content_sha256: str | None = None,
    channel: str = "guideline",
    record_sha256: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id="e1",
        text=text,
        source_id="PMID:99999999",
        source_version="2026",
        authority="highest_possible_authority",
        retrieval_path=("knowledge_graph", "verified_source"),
        content_sha256=(
            content_sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
        ),
        provenance_uri="https://spoofed.example/guideline",
        document_id="doc-1",
        citation_id="citation-1",
        eligible_claim_types=eligible,
        retrieval_channel=channel,
        record_sha256=record_sha256,
    )


def test_spoofed_provenance_and_authority_never_establish_truth():
    claim = _claim()
    evidence = _evidence(text="Unrelated management prose.")
    result = assess_evidence_relation(claim, evidence)
    assert result.relation is EvidenceRelation.UNKNOWN
    assert result.reason == "no_independent_relation_basis"
    assert len(result.claim_sha256) == len(result.evidence_sha256) == 64


def test_lexical_overlap_never_establishes_truth():
    claim = _claim()
    result = assess_evidence_relation(claim, _evidence(text=claim.text))
    assert result.relation is EvidenceRelation.UNKNOWN
    assert result.evidence_content_sha256 == hashlib.sha256(claim.text.encode()).hexdigest()


def test_content_hash_mismatch_and_wrong_scope_fail_closed():
    claim = _claim()
    mismatched = _evidence(content_sha256="0" * 64)
    assert assess_evidence_relation(
        claim, mismatched,
    ).relation is EvidenceRelation.UNKNOWN

    wrong_scope = _evidence(eligible=("diagnosis",))
    registry = CuratedRelationRegistry((CuratedRelationEntry.from_pair(
        claim, wrong_scope, EvidenceRelation.ENTAILS,
    ),))
    result = assess_evidence_relation(claim, wrong_scope, registry=registry)
    assert result.relation is EvidenceRelation.INSUFFICIENT
    assert result.reason == "evidence_outside_claim_scope"


def test_generator_declaration_is_replaced_not_promoted():
    claim = _claim(generator_declared=True)
    result = assess_evidence_relation(claim, _evidence())
    assert result.relation is EvidenceRelation.UNKNOWN
    assessed = apply_relation_assessment(claim, result)
    assert assessed.entailment is EntailmentState.INSUFFICIENT
    assert assessed.evidence_relation_origin is RelationOrigin.HOST_DETERMINISTIC
    assert assessed.evidence_relation_assessor_id.endswith("@1.0.0")
    assert len(assessed.evidence_relation_receipt_sha256) == 64
    assert not score_contract(contract(claim=assessed, evidence=[_evidence()])).structured_pass_eligible


def test_exact_curated_contradiction_is_hash_bound():
    claim = _claim()
    evidence = _evidence()
    registry = CuratedRelationRegistry((CuratedRelationEntry.from_pair(
        claim, evidence, EvidenceRelation.CONTRADICTS,
    ),))
    result = assess_evidence_relation(claim, evidence, registry=registry)
    assert result.relation is EvidenceRelation.CONTRADICTS
    assert result.curated_registry_sha256 == registry.registry_sha256
    assessed = apply_relation_assessment(claim, result)
    assert assessed.entailment is EntailmentState.CONTRADICTED
    assert assessed.contradiction is True

    changed = _evidence(text="The source now says something else.")
    assert assess_evidence_relation(
        claim, changed, registry=registry,
    ).relation is EvidenceRelation.UNKNOWN


def test_image_exemplar_requires_explicit_canonical_hash_bound_label():
    claim = _claim(
        claim_type=ClaimType.IMAGE_EXEMPLAR_SUPPORT,
        text="The frozen exemplar supports melanoma.",
        target="melanoma",
    )
    record_hash = "a" * 64
    evidence = _evidence(
        text="Opaque exemplar record.",
        eligible=("image_exemplar_support",),
        channel="image_exemplar",
        record_sha256=record_hash,
    )
    assert assess_evidence_relation(
        claim, evidence,
    ).relation is EvidenceRelation.UNKNOWN

    metadata = ImageExemplarMetadata(
        evidence_id=evidence.evidence_id,
        evidence_content_sha256=evidence.content_sha256,
        evidence_record_sha256=record_hash,
        canonical_label="melanoma",
    )
    entails = assess_evidence_relation(claim, evidence, image_exemplar=metadata)
    assert entails.relation is EvidenceRelation.ENTAILS
    assert entails.structured_metadata_sha256 == metadata.metadata_sha256

    non_supporting = assess_evidence_relation(
        claim,
        evidence,
        image_exemplar=ImageExemplarMetadata(
            evidence_id=evidence.evidence_id,
            evidence_content_sha256=evidence.content_sha256,
            evidence_record_sha256=record_hash,
            canonical_label="bcc",
        ),
    )
    assert non_supporting.relation is EvidenceRelation.INSUFFICIENT
    assert non_supporting.reason == "exemplar_label_does_not_support_target"
    assessed = apply_relation_assessment(claim, non_supporting)
    assert assessed.entailment is EntailmentState.INSUFFICIENT
    assert assessed.contradiction is False
    assert not score_contract(
        contract(claim=assessed, evidence=[evidence])
    ).structured_pass_eligible


def test_exemplar_metadata_identity_or_hash_mismatch_is_unknown():
    claim = _claim(
        claim_type=ClaimType.IMAGE_EXEMPLAR_SUPPORT,
        target="melanoma",
    )
    evidence = _evidence(
        eligible=("image_exemplar_support",),
        channel="image_exemplar",
        record_sha256="b" * 64,
    )
    metadata = ImageExemplarMetadata(
        evidence_id=evidence.evidence_id,
        evidence_content_sha256=evidence.content_sha256,
        evidence_record_sha256="c" * 64,
        canonical_label="melanoma",
    )
    result = assess_evidence_relation(claim, evidence, image_exemplar=metadata)
    assert result.relation is EvidenceRelation.UNKNOWN
    assert result.reason == "exemplar_hash_or_identity_mismatch"
