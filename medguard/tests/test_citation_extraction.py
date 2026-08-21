"""The contract -> guideline-citation bridge extracts and scopes correctly."""
import hashlib

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
from medguard.evidence.guideline_claims import GUIDELINE_CHANNEL
from medguard.eval.citation_extraction import (
    guideline_citations_from_contract,
    score_contract_citations,
)
from medguard.tests.c1_fixtures import perception


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _guideline_evidence(evidence_id: str, document_id: str, text: str,
                        eligible=("management",)) -> EvidenceItem:
    item = EvidenceItem(
        evidence_id=evidence_id, text=text, source_id="aad", source_version="2019",
        authority="clinical_practice_guideline", retrieval_path=(GUIDELINE_CHANNEL,),
        content_sha256=_sha256(text), provenance_uri="https://doi.org/x",
        document_id=document_id, citation_id="AAD:BCC-1",
        eligible_claim_types=tuple(eligible), retrieval_channel=GUIDELINE_CHANNEL,
        record_sha256="a" * 64,
    )
    item.validate()
    return item


def _pubmed_evidence(evidence_id: str, text: str) -> EvidenceItem:
    item = EvidenceItem(
        evidence_id=evidence_id, text=text, source_id="pmid:1", source_version="2020",
        authority="peer_reviewed_secondary", retrieval_path=("dense", "bm25"),
        content_sha256=_sha256(text), document_id="pm:1", citation_id="PMID:1",
        retrieval_channel="dense",
    )
    item.validate()
    return item


def _management_claim(claim_id: str, evidence_id: str,
                      claim_type: ClaimType = ClaimType.MANAGEMENT) -> ClinicalClaim:
    claim = ClinicalClaim(
        claim_id=claim_id, claim_type=claim_type,
        text="The guideline directs complete excision with a clear margin.",
        image_support=ImageSupport.NOT_APPLICABLE, evidence_ids=(evidence_id,),
        entailment=EntailmentState.NOT_ASSESSED,
        evidence_relation_origin=RelationOrigin.NOT_ASSESSED,
    )
    claim.validate()
    return claim


def _diagnosis_claim() -> ClinicalClaim:
    claim = ClinicalClaim(
        claim_id="c_diagnosis", claim_type=ClaimType.DIAGNOSIS,
        text="The image-committed diagnosis is bcc.", target_concept="bcc",
        target_state=FindingState.PRESENT, image_finding_ids=("f_commitment",),
        image_support=ImageSupport.SUPPORTS,
    )
    claim.validate()
    return claim


def _contract(claims, evidence) -> EvidenceContract:
    return EvidenceContract(
        case_id="lesion:X", modality="dermoscopy",
        clinical_question="q", perception=perception(diagnosis="bcc"),
        claims=list(claims), evidence=list(evidence),
        verification_state=VerificationState.PENDING,
    ).validate()


class _FakeResolver:
    """Minimal CitationResolver over an in-memory frozen catalog."""

    def __init__(self, records):
        self.records = dict(records)

    def resolve(self, document_id):
        return self.records.get(str(document_id))

    def offered_ids(self, committed_diagnosis, claim_type=None):
        return set(self.records)


def _record(document_id, eligible, *, source_id="aad", source_version="2019",
            citation_id="AAD:BCC-1", content_sha256=None):
    return {
        "document_id": document_id,
        "eligible_claim_types": tuple(eligible),
        "source_id": source_id,
        "source_version": source_version,
        "citation_id": citation_id,
        "content_sha256": content_sha256 or ("b" * 64),
    }


def test_only_guideline_channel_citations_are_extracted():
    guideline = _guideline_evidence("eg1", "aad:bcc:1", "Excise with a clear margin.")
    pubmed = _pubmed_evidence("ep1", "Dermatoscopy improves diagnostic accuracy.")
    contract = _contract(
        claims=[
            _diagnosis_claim(),
            _management_claim("g1", "eg1"),
            # A management-style claim citing the PubMed record must NOT be scored
            # against the guideline catalog: it is not in that index.
            _management_claim("p1", "ep1"),
        ],
        evidence=[guideline, pubmed],
    )
    citations = guideline_citations_from_contract(contract)
    assert [c.cited_id for c in citations] == ["aad:bcc:1"]
    only = citations[0]
    assert only.claim_type == "management"
    assert only.claim_id == "g1"
    # Provenance fields come from the arm's own evidence copy.
    assert only.source_id == "aad"
    assert only.source_version == "2019"
    assert only.citation_id == "AAD:BCC-1"
    assert only.content_sha256 == guideline.content_sha256


def test_intact_when_eligible_and_offered():
    text = "Excise with a clear margin."
    guideline = _guideline_evidence("eg1", "aad:bcc:1", text)
    contract = _contract(
        claims=[_diagnosis_claim(), _management_claim("g1", "eg1")],
        evidence=[guideline],
    )
    resolver = _FakeResolver({
        "aad:bcc:1": _record("aad:bcc:1", ("management",),
                             content_sha256=guideline.content_sha256),
    })
    score = score_contract_citations(contract, "bcc", resolver)
    assert score.n_citations == 1
    assert score.all_intact is True
    assert score.checks[0].integrity_ok


def test_claim_type_mismatch_is_caught():
    text = "Excise with a clear margin."
    guideline = _guideline_evidence("eg1", "aad:bcc:1", text)
    # An assessment claim citing a management-only record: real, offered id,
    # wrong claim type.
    contract = _contract(
        claims=[_diagnosis_claim(),
                _management_claim("g1", "eg1", claim_type=ClaimType.ASSESSMENT)],
        evidence=[guideline],
    )
    resolver = _FakeResolver({
        "aad:bcc:1": _record("aad:bcc:1", ("management",),
                             content_sha256=guideline.content_sha256),
    })
    score = score_contract_citations(contract, "bcc", resolver)
    assert score.all_intact is False
    assert score.checks[0].failure_reason == "claim_type_mismatch"


def test_no_guideline_citations_scores_none_not_perfect():
    pubmed = _pubmed_evidence("ep1", "Dermatoscopy improves diagnostic accuracy.")
    contract = _contract(
        claims=[_diagnosis_claim(), _management_claim("p1", "ep1")],
        evidence=[pubmed],
    )
    resolver = _FakeResolver({})
    score = score_contract_citations(contract, "bcc", resolver)
    assert score.n_citations == 0
    assert score.all_intact is None
