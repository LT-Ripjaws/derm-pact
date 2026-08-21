from __future__ import annotations

import hashlib

from medguard.evidence import (
    ClinicalClaim,
    ClaimType,
    EntailmentState,
    EvidenceItem,
    FailureType,
    FindingState,
    ImageSupport,
    RelationOrigin,
    VerificationState,
    evidence_from_chunk,
    evidence_receipt_sha256,
    verify_contract,
)
from medguard.tests.c1_fixtures import contract, diagnosis_claim, evidence_item


def test_supported_cross_modal_contract_passes_verification():
    report = verify_contract(contract())
    assert report.ok and not report.failures


def test_provenance_hash_does_not_make_contradicted_content_pass():
    claim = diagnosis_claim(
        entailment=EntailmentState.CONTRADICTED,
        contradiction=True,
    )
    report = verify_contract(contract(claim=claim))
    assert FailureType.SOURCE_CONTRADICTION in report.failures


def test_image_evidence_conflict_is_derived_not_generator_asserted():
    # The model says "supports", but BCC conflicts with the frozen melanoma commitment.
    claim = diagnosis_claim(target="bcc", declared=ImageSupport.SUPPORTS)
    report = verify_contract(contract(claim=claim))
    assert FailureType.IMAGE_EVIDENCE_CONFLICT in report.failures


def test_content_tampering_fails_contract_validation():
    good = evidence_item()
    bad = EvidenceItem(**{**good.__dict__, "content_sha256": "0" * 64})
    try:
        contract(evidence=[bad]).validate()
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered evidence hash was accepted")


def test_chunk_normalization_preserves_retrieval_path_and_receipt():
    item = evidence_from_chunk(
        {"text": "  Prompt   assessment. ", "source": "NICE", "version": "1",
         "authority": "guideline"},
        evidence_id="e1", retrieval_path=("hybrid", "guideline"),
    )
    assert item.text == "Prompt assessment."
    assert item.retrieval_path == ("hybrid", "guideline")
    assert len(evidence_receipt_sha256([item])) == 64


def test_pass_requires_satisfied_contract():
    value = contract()
    value.proposed_action = "PASS"
    value.verification_state = VerificationState.PENDING
    try:
        value.validate()
    except ValueError as exc:
        assert "PASS requires" in str(exc)
    else:
        raise AssertionError("PASS was accepted without a satisfied contract")


def _management_claim() -> ClinicalClaim:
    return ClinicalClaim(
        claim_id="c1", claim_type=ClaimType.MANAGEMENT,
        text="Specialist assessment is indicated.",
        target_concept="", target_state=FindingState.UNCERTAIN,
        image_finding_ids=(), image_support=ImageSupport.NOT_APPLICABLE,
        evidence_ids=("e1",), entailment=EntailmentState.SUPPORTED,
        contradiction=False, evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
        evidence_relation_assessor_id="c1-deterministic-fixture",
        evidence_relation_receipt_sha256="4" * 64,
    )


def test_claim_scope_blocks_management_from_diagnosis_only_literature():
    text = "A PubMed study describes visual diagnostic features."
    literature = EvidenceItem(
        evidence_id="e1", text=text, source_id="PMID:1", source_version="v1",
        authority="peer_reviewed_biomedical_literature", retrieval_path=("hybrid",),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        eligible_claim_types=("assessment", "diagnosis", "visual_finding"),
    )
    report = verify_contract(contract(claim=_management_claim(), evidence=[literature]))
    assert FailureType.MISALIGNED_EVIDENCE in report.failures
    assert any("outside its cited source scope" in issue for issue in report.issues)


def test_claim_scope_allows_management_from_management_eligible_guideline():
    text = "The guideline recommends specialist assessment."
    guideline = EvidenceItem(
        evidence_id="e1", text=text, source_id="PMID:2", source_version="v1",
        authority="clinical_practice_guideline", retrieval_path=("hybrid",),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        eligible_claim_types=("assessment", "management", "referral"),
    )
    assert verify_contract(contract(claim=_management_claim(), evidence=[guideline])).ok
