"""Hand-checkable tests for the citation-integrity provenance endpoint.

Exercised against the real FrozenGuidelineRetriever (so the resolve/offered_ids
primitives are the ones the runtime uses) and against a dict-based fake resolver
(so the Protocol is shown to accept plain mappings).
"""
import hashlib

from medguard.eval.citation_integrity import (
    AssertedCitation, CaseCitationScore, check_citation, score_case, summarize,
    HALLUCINATED_ID, UNOFFERED_ID, PROVENANCE_MISMATCH, CLAIM_TYPE_MISMATCH,
)
from medguard.pipeline.guideline_retriever import FrozenGuidelineRetriever


def _record(document_id, condition, eligible, text):
    return {
        "schema_version": "medguard-guideline-recommendation-v1",
        "authority_class": "clinical_practice_guideline",
        "citation_id": f"cit:{document_id}",
        "condition": condition,
        "content_form": "verbatim_recommendation_text_projection",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "document_id": document_id,
        "eligible_claim_types": list(eligible),
        "record_sha256": "0" * 64,
        "source_id": f"eado:{condition}",
        "source_pdf_sha256": "1" * 64,
        "source_version": "2023",
        "text": text,
    }


def _retriever():
    return FrozenGuidelineRetriever([
        _record("bcc-1", "basal_cell_carcinoma", ["management", "assessment"], "BCC management text"),
        _record("mel-1", "melanoma", ["management"], "Melanoma management text"),
        _record("mel-2", "melanoma", ["assessment", "visual_finding"], "Melanoma assessment text"),
    ])


def test_clean_citation_is_intact():
    check = check_citation(
        AssertedCitation(cited_id="mel-1", claim_type="management"), "melanoma", _retriever())
    assert check.integrity_ok
    assert check.resolvable and check.offered and check.claim_type_eligible
    assert check.failure_reason == "" and check.failed_checks == ()


def test_hallucinated_id_is_caught():
    check = check_citation(
        AssertedCitation(cited_id="ghost-9", claim_type="management"), "melanoma", _retriever())
    assert not check.integrity_ok and not check.resolvable
    assert check.failure_reason == HALLUCINATED_ID
    # Nothing downstream is knowable, so the other guards report False, not True.
    assert not check.offered and not check.claim_type_eligible


def test_real_id_from_wrong_condition_is_unoffered():
    # bcc-1 exists and is management-eligible, but it was never offered for a
    # melanoma commitment.
    check = check_citation(
        AssertedCitation(cited_id="bcc-1", claim_type="management"), "melanoma", _retriever())
    assert check.resolvable and not check.offered and not check.integrity_ok
    assert check.failure_reason == UNOFFERED_ID


def test_real_offered_id_on_wrong_claim_type():
    # mel-1 is management-only; citing it to back a diagnosis claim is a misuse.
    check = check_citation(
        AssertedCitation(cited_id="mel-1", claim_type="diagnosis"), "melanoma", _retriever())
    assert check.resolvable and check.offered and not check.claim_type_eligible
    assert check.failure_reason == CLAIM_TYPE_MISMATCH


def test_provenance_tamper_under_a_real_id():
    check = check_citation(
        AssertedCitation(cited_id="mel-1", claim_type="management", content_sha256="dead" * 16),
        "melanoma", _retriever())
    assert check.resolvable and check.offered and not check.provenance_intact
    assert check.failure_reason == PROVENANCE_MISMATCH


def test_asserted_provenance_matching_frozen_record_stays_intact():
    ret = _retriever()
    frozen = ret.resolve("mel-1")
    check = check_citation(
        AssertedCitation(
            cited_id="mel-1", claim_type="management",
            source_id=frozen.source_id, source_version=frozen.source_version,
            citation_id=frozen.citation_id, content_sha256=frozen.content_sha256),
        "melanoma", ret)
    assert check.integrity_ok


def test_failure_precedence_unoffered_beats_claim_type():
    # bcc-1 for a melanoma commitment on a diagnosis claim fails BOTH guards;
    # the primary reason is the more serious unoffered-id.
    check = check_citation(
        AssertedCitation(cited_id="bcc-1", claim_type="diagnosis"), "melanoma", _retriever())
    assert set(check.failed_checks) == {UNOFFERED_ID, CLAIM_TYPE_MISMATCH}
    assert check.failure_reason == UNOFFERED_ID


def test_benign_commitment_owes_no_citation():
    # nevus has no licensed condition; citing anything is unoffered, and citing
    # nothing yields all_intact=None (not a free perfect score).
    ret = _retriever()
    empty = score_case("nevus", [], ret)
    assert empty.n_citations == 0 and empty.all_intact is None
    cited = check_citation(AssertedCitation(cited_id="mel-1", claim_type="management"), "nevus", ret)
    assert not cited.offered and cited.failure_reason == UNOFFERED_ID


def test_missing_claim_type_is_rejected_as_malformed():
    for bad in ("", "   "):
        try:
            check_citation(AssertedCitation(cited_id="mel-1", claim_type=bad), "melanoma", _retriever())
        except ValueError:
            pass
        else:
            raise AssertionError("a citation without a claim_type must be rejected")


def test_summarize_separates_uncited_lesions_and_counts_failures():
    ret = _retriever()
    case_a = score_case("melanoma", [
        AssertedCitation(cited_id="mel-1", claim_type="management"),   # intact
        AssertedCitation(cited_id="ghost-9", claim_type="management"),  # hallucinated
    ], ret)
    case_b = score_case("melanoma", [
        AssertedCitation(cited_id="mel-2", claim_type="assessment"),   # intact
    ], ret)
    case_c = score_case("nevus", [], ret)                              # owes nothing
    out = summarize([case_a, case_b, case_c])
    assert out["total_citations"] == 3 and out["intact_citations"] == 2
    assert out["integrity_rate"] == 0.6667
    assert out["cases_scored"] == 3 and out["cases_with_citations"] == 2
    assert out["cases_without_citations"] == 1
    assert out["cases_all_intact"] == 1 and out["case_all_intact_rate"] == 0.5
    assert out["failure_breakdown"] == {HALLUCINATED_ID: 1}


def test_summarize_all_uncited_has_no_rate():
    out = summarize([score_case("nevus", [], _retriever()), score_case("df", [], _retriever())])
    assert out["total_citations"] == 0 and out["integrity_rate"] is None
    assert out["case_all_intact_rate"] is None and out["cases_without_citations"] == 2


def test_protocol_accepts_a_dict_based_fake_resolver():
    class FakeResolver:
        def __init__(self):
            self._records = {
                "r-1": {"eligible_claim_types": ["management"], "content_sha256": "abc",
                        "source_id": "s", "source_version": "v", "citation_id": "c"},
            }

        def resolve(self, document_id):
            return self._records.get(str(document_id))

        def offered_ids(self, committed_diagnosis, claim_type=None):
            return {"r-1"} if committed_diagnosis == "melanoma" else set()

    resolver = FakeResolver()
    ok = check_citation(AssertedCitation(cited_id="r-1", claim_type="management"), "melanoma", resolver)
    assert ok.integrity_ok
    bad = check_citation(
        AssertedCitation(cited_id="r-1", claim_type="management", content_sha256="zzz"),
        "melanoma", resolver)
    assert not bad.provenance_intact and bad.failure_reason == PROVENANCE_MISMATCH
