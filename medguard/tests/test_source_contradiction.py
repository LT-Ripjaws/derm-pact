"""D3: source contradiction is detectable and routes to INDEPENDENT_SOURCE.

The "action fires and changes the evidence set" half is exercised end-to-end
through the runner's remediation tools in test_matched_full_h2h_run.py; here we
prove the detection primitive and the policy routing it enables.
"""
import hashlib

import pytest

from medguard.evidence.contract import (
    ClaimType,
    ClinicalClaim,
    EntailmentState,
    EvidenceContract,
    EvidenceItem,
    FindingState,
    ImageSupport,
    RelationOrigin,
    VerificationState,
)
from medguard.evidence.source_contradiction import (
    ContradictionFlag,
    SourceContradictionRegistry,
    build_contradicted_source_claim,
    flagged_text_records,
)
from medguard.evidence.verification import FailureType, verify_contract
from medguard.remediation.actions import Action
from medguard.remediation.policy import DeterministicRemediationPolicy
from medguard.tests.c1_fixtures import perception


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pubmed_evidence(*, eligible=("assessment",)) -> EvidenceItem:
    text = "This abstract advises against biopsy for suspicious pigmented lesions."
    item = EvidenceItem(
        evidence_id="e1", text=text, source_id="pmid:99", source_version="2018",
        authority="peer_reviewed_secondary", retrieval_path=("dense", "bm25"),
        content_sha256=_sha256(text), document_id="pubmed:99:abstract",
        citation_id="PMID:99", eligible_claim_types=tuple(eligible),
        retrieval_channel="dense", record_sha256="c" * 64,
    )
    item.validate()
    return item


def _diagnosis_claim() -> ClinicalClaim:
    claim = ClinicalClaim(
        claim_id="c_diagnosis", claim_type=ClaimType.DIAGNOSIS,
        text="The image-committed diagnosis is melanoma.", target_concept="melanoma",
        target_state=FindingState.PRESENT, image_finding_ids=("f_commitment",),
        image_support=ImageSupport.SUPPORTS,
    )
    claim.validate()
    return claim


def test_registry_detects_only_flagged_condition_document_pairs():
    registry = SourceContradictionRegistry([
        ContradictionFlag(condition="melanoma", document_id="pubmed:99:abstract"),
    ])
    assert registry.is_flagged("melanoma", "pubmed:99:abstract")
    assert not registry.is_flagged("bcc", "pubmed:99:abstract")
    assert not registry.is_flagged("melanoma", "pubmed:1:abstract")
    records = [
        {"document_id": "pubmed:99:abstract"},
        {"document_id": "pubmed:1:abstract"},
        {"id": "isic:exemplar"},
    ]
    flagged = flagged_text_records(records, condition="melanoma", registry=registry)
    assert [r["document_id"] for r in flagged] == ["pubmed:99:abstract"]


def test_duplicate_flag_is_rejected():
    with pytest.raises(ValueError):
        SourceContradictionRegistry([
            ContradictionFlag("melanoma", "pubmed:99:abstract"),
            ContradictionFlag("melanoma", "pubmed:99:abstract"),
        ])


def test_contradicted_claim_is_host_deterministic_and_verifies_as_contradiction():
    evidence = _pubmed_evidence()
    claim = build_contradicted_source_claim(claim_id="x1", evidence=evidence)
    assert claim.entailment is EntailmentState.CONTRADICTED
    assert claim.contradiction is True
    # host-owned, not a generator declaration -- this is the load-bearing property.
    assert claim.evidence_relation_origin is RelationOrigin.HOST_DETERMINISTIC
    assert len(claim.evidence_relation_receipt_sha256) == 64

    contract = EvidenceContract(
        case_id="lesion:1", modality="dermoscopy", clinical_question="q",
        perception=perception(diagnosis="melanoma"),
        claims=[_diagnosis_claim(), claim], evidence=[evidence],
        verification_state=VerificationState.PENDING,
    ).validate()
    report = verify_contract(contract)
    assert FailureType.SOURCE_CONTRADICTION in report.failures
    # It must NOT read as a generator-declared relation, or the policy would abstain
    # immediately instead of routing to INDEPENDENT_SOURCE.
    assert FailureType.UNVERIFIED_RELATION not in report.failures


def test_policy_routes_a_pure_source_contradiction_to_independent_source():
    evidence = _pubmed_evidence()
    claim = build_contradicted_source_claim(claim_id="x1", evidence=evidence)
    contract = EvidenceContract(
        case_id="lesion:1", modality="dermoscopy", clinical_question="q",
        perception=perception(diagnosis="melanoma"),
        claims=[_diagnosis_claim(), claim], evidence=[evidence],
        verification_state=VerificationState.PENDING,
    ).validate()
    report = verify_contract(contract)
    action = DeterministicRemediationPolicy().decide(report, counts={})
    assert action is Action.INDEPENDENT_SOURCE


def test_record_without_source_bound_type_cannot_be_contradicted():
    # A diagnosis/visual-only record is image-bound; there is no source claim to
    # contradict, so the builder refuses rather than fabricate one.
    evidence = _pubmed_evidence(eligible=("diagnosis",))
    with pytest.raises(ValueError):
        build_contradicted_source_claim(claim_id="x1", evidence=evidence)
