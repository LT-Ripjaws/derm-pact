from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from dermpact.pipeline.guideline_retriever import (
    COMMITMENT_TO_CONDITION,
    FrozenGuidelineRetriever,
    GuidelineCatalogError,
    VERBATIM_CONTENT_FORM,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "dermpact/data/corpus/derm_guideline_recommendations_v1.jsonl"
CATALOG_SHA256 = "98ccd42e3a85648d3d5da975a454eea86ccef13e37dc67360a62f612f9923a16"


def _record(document_id="eado:pmid:37604067:box-1", condition="basal_cell_carcinoma",
            text="Superficial and nodular BCCs can be diagnosed clinically.",
            claim_types=("assessment", "diagnosis", "visual_finding", "differential"),
            **overrides):
    record = {
        "authority_class": "clinical_practice_guideline",
        "citation_id": "DOI:10.1016/j.ejca.2023.113254",
        "condition": condition,
        "content_form": VERBATIM_CONTENT_FORM,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "document_id": document_id,
        "eligible_claim_types": list(claim_types),
        "record_sha256": "0" * 64,
        "section": "6 Diagnosis",
        "article_page": 11,
        "source_id": "DOI:10.1016/j.ejca.2023.113254",
        "source_pdf_sha256": "1" * 64,
        "source_version": "update-2023",
        "text": text,
    }
    record.update(overrides)
    if "text" in overrides and "content_sha256" not in overrides:
        record["content_sha256"] = hashlib.sha256(
            overrides["text"].encode("utf-8")).hexdigest()
    return record


def test_scoping_is_keyed_on_the_commitment_and_never_falls_back():
    retriever = FrozenGuidelineRetriever([
        _record(document_id="bcc-1"),
        _record(document_id="mel-1", condition="melanoma"),
    ])
    assert [row.document_id for row in retriever.search("bcc")] == ["bcc-1"]
    assert [row.document_id for row in retriever.search("melanoma")] == ["mel-1"]


def test_an_unlicensed_commitment_reaches_nothing_rather_than_the_nearest_record():
    """A vascular lesion must not surface basal-cell surgical margins."""
    retriever = FrozenGuidelineRetriever([_record(), _record(document_id="mel-1", condition="melanoma")])
    # licensed condition. These four remain genuinely uncovered.
    for diagnosis in ("bkl", "df", "nevus", "vasc"):
        assert retriever.search(diagnosis) == []
        assert retriever.offered_ids(diagnosis) == set()
        assert diagnosis not in COMMITMENT_TO_CONDITION
    assert COMMITMENT_TO_CONDITION["akiec"] == "actinic_keratosis"
    assert retriever.search("akiec") == []


def test_claim_type_filter_admits_only_records_declaring_the_type():
    retriever = FrozenGuidelineRetriever([
        _record(document_id="dx", claim_types=("diagnosis",)),
        _record(document_id="mgmt", claim_types=("management", "treatment")),
    ])
    assert [r.document_id for r in retriever.search("bcc", "diagnosis")] == ["dx"]
    assert [r.document_id for r in retriever.search("bcc", "management")] == ["mgmt"]
    assert retriever.search("bcc", "follow_up") == []


def test_tampered_text_is_rejected_against_its_content_hash():
    """The whole point of a source-exact catalog is that the text is exact."""
    record = _record()
    record["text"] = record["text"] + " And also use laser."
    with pytest.raises(GuidelineCatalogError, match="content_sha256"):
        FrozenGuidelineRetriever([record])


def test_a_non_verbatim_record_cannot_enter_the_catalog():
    """A guideline abstract describes a guideline; it does not state a recommendation."""
    with pytest.raises(GuidelineCatalogError, match="verbatim"):
        FrozenGuidelineRetriever([_record(content_form="pubmed_abstract_excerpt")])


def test_a_non_guideline_authority_cannot_enter_the_catalog():
    with pytest.raises(GuidelineCatalogError, match="non-guideline authority"):
        FrozenGuidelineRetriever([
            _record(authority_class="peer_reviewed_biomedical_literature")])


def test_missing_lineage_fields_fail_closed():
    record = _record()
    del record["source_version"]
    with pytest.raises(GuidelineCatalogError, match="missing lineage fields"):
        FrozenGuidelineRetriever([record])


def test_duplicate_document_ids_are_rejected():
    with pytest.raises(GuidelineCatalogError, match="duplicate"):
        FrozenGuidelineRetriever([_record(), _record()])


def test_empty_catalog_is_rejected():
    with pytest.raises(GuidelineCatalogError, match="empty"):
        FrozenGuidelineRetriever([])


def test_load_enforces_the_receipt_hash_when_supplied():
    record = _record()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.jsonl"
        payload = json.dumps(record, ensure_ascii=False) + "\n"
        path.write_text(payload, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = FrozenGuidelineRetriever.load(path, expected_sha256=digest)
        assert len(loaded.records) == 1
        with pytest.raises(GuidelineCatalogError, match="hash mismatch"):
            FrozenGuidelineRetriever.load(path, expected_sha256="0" * 64)


def test_evidence_record_carries_the_full_lineage_for_the_contract():
    retriever = FrozenGuidelineRetriever([_record()])
    evidence = retriever.search("bcc")[0].to_evidence_record()
    for field in ("source_id", "source_version", "citation_id", "content_sha256",
                  "record_sha256", "source_pdf_sha256", "authority_class",
                  "retrieval_path", "eligible_claim_types"):
        assert evidence[field], f"{field} missing from the bound evidence record"
    assert evidence["retrieval_path"] == "authoritative_guideline_catalog"


def test_resolve_returns_none_for_a_hallucinated_identifier():
    """This is how a fabricated citation is caught."""
    retriever = FrozenGuidelineRetriever([_record(document_id="real")])
    assert retriever.resolve("real") is not None
    assert retriever.resolve("eado:pmid:99999999:box-42") is None


def test_repository_catalog_loads_under_its_receipt_hash_and_scopes_as_measured():
    retriever = FrozenGuidelineRetriever.load(CATALOG, expected_sha256=CATALOG_SHA256)
    assert len(retriever.records) == 66
    assert retriever.conditions() == {"basal_cell_carcinoma": 27, "melanoma": 26, "actinic_keratosis": 13}
    assert len(retriever.search("bcc")) == 27
    assert len(retriever.search("melanoma")) == 26
    assert retriever.search("nevus") == []
    assert len(retriever.search("melanoma", "visual_finding")) == 7
    assert len(retriever.search("melanoma", "assessment")) == 7
    assert len(retriever.search("bcc", "visual_finding")) == 3
    # 54 of 66 are management-eligible.
    management = sum(len(retriever.search(dx, "management"))
                     for dx in ("bcc", "melanoma", "akiec"))
    assert management == 54


def test_the_three_aad_guideline_abstracts_cannot_enter_this_channel():
    """The hollow management path, closed at the channel boundary."""
    abstract = _record(
        document_id="pubmed:29331385:abstract",
        content_form="pubmed_abstract_excerpt",
        text="Guidelines of care for the management of basal cell carcinoma. "
             "In these evidence-based guidelines of care, we provide recommendations.",
    )
    with pytest.raises(GuidelineCatalogError, match="not a verbatim projection"):
        FrozenGuidelineRetriever([abstract])


def test_repository_catalog_ordering_is_deterministic():
    a = FrozenGuidelineRetriever.load(CATALOG, expected_sha256=CATALOG_SHA256)
    b = FrozenGuidelineRetriever.load(CATALOG, expected_sha256=CATALOG_SHA256)
    assert [r.document_id for r in a.search("bcc")] == [r.document_id for r in b.search("bcc")]
    assert [r.document_id for r in a.search("bcc", k=5)] == [
        r.document_id for r in a.search("bcc")][:5]
