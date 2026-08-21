"""Guideline citations become SUPPORTED source-bound claims via source integrity.

Exercised against the real 66-record frozen guideline corpus so the atomic catalog
and registry are the ones the runtime will use.
"""
import json
from pathlib import Path

from medguard.evidence.contract import (
    ClaimType, EntailmentState, ImageSupport, RelationOrigin,
)
from medguard.evidence.guideline_claims import (
    GUIDELINE_CHANNEL, atomic_claim_for_record, build_guideline_claim_assets,
    build_supported_guideline_claim, guideline_evidence_item, source_bound_claim_type,
)
from medguard.evidence.relation_assessor import (
    CuratedRelationRegistry, EvidenceRelation, assess_evidence_relation,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "medguard/data/corpus/derm_guideline_recommendations_v1.jsonl"

_SOURCE_BOUND = {ClaimType.MANAGEMENT, ClaimType.TREATMENT, ClaimType.REFERRAL, ClaimType.ASSESSMENT}


def _records():
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_source_bound_type_is_management_first_and_skips_image_bound():
    assert source_bound_claim_type(["assessment", "management", "diagnosis"]) is ClaimType.MANAGEMENT
    assert source_bound_claim_type(["treatment", "assessment"]) is ClaimType.TREATMENT
    assert source_bound_claim_type(["assessment", "visual_finding"]) is ClaimType.ASSESSMENT
    # A record eligible only for image-bound types backs no source claim.
    assert source_bound_claim_type(["diagnosis", "visual_finding", "differential"]) is None


def test_all_records_become_atomic_claims_with_matching_registry():
    catalog, registry = build_guideline_claim_assets(_records())
    assert len(catalog.claims) == 66 and len(registry.atomic_entries) == 66
    # keys are canonical and unique
    keys = [claim.claim_key for claim in catalog.claims]
    assert len(set(keys)) == 66 and all(key.startswith("guideline.") for key in keys)


def test_guideline_citation_resolves_to_supported_via_source_integrity():
    records = _records()
    _catalog, registry = build_guideline_claim_assets(records)
    record = records[0]
    atomic = atomic_claim_for_record(record)
    evidence = guideline_evidence_item(record, evidence_id="g1")
    claim = build_supported_guideline_claim(
        claim_id="cg1", evidence=evidence, atomic_claim=atomic, registry=registry)
    assert claim.entailment is EntailmentState.SUPPORTED
    assert claim.evidence_relation_origin is RelationOrigin.HOST_DETERMINISTIC
    # Source-bound, never a patient-diagnosis claim.
    assert claim.claim_type in _SOURCE_BOUND
    assert claim.image_support is ImageSupport.NOT_APPLICABLE
    assert not claim.target_concept and not claim.image_finding_ids
    assert claim.evidence_ids == ("g1",)
    assert evidence.retrieval_channel == GUIDELINE_CHANNEL
    # The claim text is the verbatim source (integrity, not paraphrase).
    assert claim.text == record["text"]


def test_stratified_selection_leads_with_assessment_and_is_deterministic():
    from medguard.evidence.guideline_claims import select_guideline_evidence
    from medguard.pipeline.guideline_retriever import FrozenGuidelineRetriever
    retriever = FrozenGuidelineRetriever.load(CORPUS)
    picked = select_guideline_evidence(retriever, "melanoma", 3)
    assert len(picked) == 3
    # deterministic
    assert [e.document_id for e in picked] == [
        e.document_id for e in select_guideline_evidence(retriever, "melanoma", 3)]
    # at least one assessment-eligible record leads the set (the question's claim
    # type), rather than three alphabetically-first management records
    assert any("assessment" in e.eligible_claim_types for e in picked)
    # every pick is source-bound eligible (never wasted on an image-bound-only record)
    source_bound = {"assessment", "management", "treatment", "referral"}
    assert all(source_bound & set(e.eligible_claim_types) for e in picked)
    # a benign commitment owes nothing
    assert select_guideline_evidence(retriever, "nevus", 3) == []


def test_source_integrity_requires_the_host_registry_entry():
    # An empty registry means the host has not asserted integrity for this claim,
    # so it fails closed rather than earning free support.
    records = _records()
    record = records[0]
    atomic = atomic_claim_for_record(record)
    evidence = guideline_evidence_item(record, evidence_id="g1")
    bound = atomic.bind(claim_id="cg1", evidence=evidence)
    result = assess_evidence_relation(
        bound, evidence, registry=CuratedRelationRegistry(), atomic_claim=atomic)
    assert result.relation is EvidenceRelation.UNKNOWN
