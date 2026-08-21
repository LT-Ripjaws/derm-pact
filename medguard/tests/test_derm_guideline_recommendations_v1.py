import hashlib
import json

import pytest

from medguard.data.freeze_derm_guideline_recommendations_v1 import (
    OUTPUT_PATH,
    RECEIPT_PATH,
    _extract_block,
    build_records,
    freeze_recommendation_subset,
    verify_recommendation_subset,
)


def test_frozen_recommendation_subset_has_expected_sources_and_records():
    records, manifest = build_records()
    assert len(manifest) == 12
    assert len(records) == 66
    assert {row["source_document_id"] for row in records} == {
        # EADO (original four)
        "eado:pmid:37604067",
        "eado:pmid:39709737",
        "eado:pmid:39700658",
        "eado:pmid:38451047",
        # NICE (Crown/(c) NICE, no DOI -> citation_id like "NICE:NG14")
        "nice:ng14",
        "nice:ng12",
        "nice:qs130",
        "nice:csg8",
        # AAD (JAAD/Elsevier) + BAD (BJD/Wiley), internal non-commercial only
        "aad:pmid:30392755",
        "aad:pmid:29331385",
        "aad:pmid:33820677",
        "bad:pmid:28098380",
    }


def test_nice_records_use_a_citation_id_while_doi_sources_use_the_doi():
    """NICE guidance carries no DOI, so an explicit citation_id stands in.

    A NICE record that silently fell back to ``DOI:`` (empty DOI) would mint a
    malformed, unresolvable citation identifier; DOI-bearing sources must keep
    the DOI form so the two families stay distinguishable and resolvable.
    """
    records, _manifest = build_records()
    by_source = {}
    for row in records:
        by_source.setdefault(row["source_document_id"], row)
    for nice_id in ("nice:ng14", "nice:ng12", "nice:qs130", "nice:csg8"):
        row = by_source[nice_id]
        assert row["citation_id"] == f"NICE:{nice_id.split(':', 1)[1].upper()}"
        assert row["source_id"] == row["citation_id"]
        assert row["doi"] == ""
    for doi_id in ("aad:pmid:29331385", "bad:pmid:28098380", "eado:pmid:37604067"):
        row = by_source[doi_id]
        assert row["citation_id"].startswith("DOI:") and row["doi"]
        assert row["source_id"] == row["citation_id"]


def test_subset_excludes_advanced_systemic_recommendations():
    records, _manifest = build_records()
    # recommendation_id is namespaced by document, so scope the exclusion by source.
    ids_by_source = {}
    for row in records:
        ids_by_source.setdefault(row["source_document_id"], set()).add(row["recommendation_id"])
    # BCC: advanced/adjuvant boxes excluded.
    assert not {"box-18", "box-19", "box-20", "box-21"}.intersection(ids_by_source["eado:pmid:37604067"])
    # Melanoma Part 2 (Treatment): advanced systemic recommendations 7+ excluded.
    assert not {f"recommendation-{n}" for n in range(7, 32)}.intersection(
        ids_by_source["eado:pmid:39709737"])
    # Melanoma Part 1 (Diagnostics): only the diagnostic block 3-7; staging (1,2),
    # molecular (8) and follow-up (9,10) are intentionally not in the subset.
    assert ids_by_source["eado:pmid:39700658"] == {
        "recommendation-3", "recommendation-4", "recommendation-5",
        "recommendation-6", "recommendation-7"}


def test_every_record_has_recommendation_lineage_and_clean_text():
    records, _manifest = build_records()
    for row in records:
        assert row["section"]
        assert row["pdf_page"] >= 1
        assert row["article_page"] >= 1
        assert row["question_types"]
        assert row["eligible_claim_types"]
        assert "�" not in row["text"]
        assert hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == row["content_sha256"]


def test_distribution_restrictions_are_preserved_per_source():
    records, _manifest = build_records()
    by_source = {}
    for row in records:
        by_source.setdefault(row["source_document_id"], set()).add(row["distribution_status"])
    assert by_source["eado:pmid:37604067"] == {"internal_noncommercial_no_derivatives"}
    assert by_source["eado:pmid:39709737"] == {"redistributable_with_attribution"}
    assert by_source["eado:pmid:39700658"] == {"redistributable_with_attribution"}


def test_missing_extraction_marker_fails_closed():
    with pytest.raises(ValueError, match="missing recommendation end marker"):
        _extract_block("Box 1 useful recommendation", "Box 1", "Box 2")


def test_recommendation_receipt_binds_raw_bytes():
    freeze_recommendation_subset()
    receipt = verify_recommendation_subset()
    assert receipt["recommendation_record_count"] == 66
    assert len(OUTPUT_PATH.read_text(encoding="utf-8").splitlines()) == 66
    raw_digest = hashlib.sha256(RECEIPT_PATH.read_bytes()).hexdigest()
    assert RECEIPT_PATH.with_name(RECEIPT_PATH.name + ".sha256").read_text(encoding="ascii").strip() == raw_digest
    assert json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))["claim_guard"].startswith("This artifact does not")
