from medguard.evidence import (
    ClinicalClaim, ClaimType, EntailmentState, EvidenceContract, FindingState,
    ImageSupport, RelationOrigin,
)
from medguard.pipeline.ops import EvidencePipelineOps, reciprocal_rank_fusion
from medguard.tests.c1_fixtures import perception


class Retriever:
    def __init__(self, source):
        self.source = source

    def search(self, query, k):
        chunks = [{"text": f"{query} evidence", "source": self.source,
                   "version": "1", "authority": "guideline",
                   "retrieval_channel": "dense"} for _ in range(k)]
        return chunks, [1.0 - i / 100 for i in range(k)]


class Generator:
    def __init__(self):
        self.calls = []

    def generate_contract(self, **kwargs):
        self.calls.append(kwargs)
        chunks = kwargs["chunks"]
        from medguard.evidence import evidence_from_chunk
        evidence = [evidence_from_chunk(
            chunk, evidence_id=str(i), retrieval_path=chunk["retrieval_path"])
            for i, chunk in enumerate(chunks, start=1)]
        return EvidenceContract(
            case_id=kwargs["case_id"], modality=kwargs["modality"],
            clinical_question=kwargs["question"],
            perception=kwargs["perception"], evidence=evidence,
            claims=[ClinicalClaim(
                claim_id="c1", claim_type=ClaimType.DIAGNOSIS,
                text="supported", target_concept="melanoma",
                target_state=FindingState.PRESENT,
                image_finding_ids=("f_commitment",), image_support=ImageSupport.SUPPORTS,
                evidence_ids=("1",), entailment=EntailmentState.SUPPORTED,
                evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
                evidence_relation_assessor_id="c1-deterministic-fixture",
                evidence_relation_receipt_sha256="4" * 64)],
        )


def _case():
    return {"case_id": "HAM_1", "modality": "dermoscopy",
            "question": "melanoma assessment", "perception": perception()}


class ImageRetriever:
    def search_records_by_image(self, image_path, k):
        assert image_path == "case.jpg"
        return [{
            "id": f"exemplar:{index}",
            "document_id": f"exemplar:{index}",
            "citation_id": f"HAM10000:{index}",
            "source": f"HAM10000:{index}",
            "source_version": "dermlip:frozen",
            "authority_class": "image_exemplar",
            "eligible_claim_types": ["image_exemplar_support"],
            "retrieval_channel": "image_exemplar",
            "text": "Frozen training exemplar diagnosis label: melanoma.",
            "score": 1.0 - index / 100,
        } for index in range(k)]

    def search_records_by_images(self, image_paths, k):
        assert image_paths == ["case-a.jpg", "case-b.jpg"]
        rows = self.search_records_by_image("case.jpg", k)
        for row in rows:
            row["query_view_count"] = len(image_paths)
        return rows


def test_initial_and_independent_paths_are_distinct():
    generator = Generator()
    ops = EvidencePipelineOps(
        retriever=Retriever("primary"), independent_retriever=Retriever("independent"),
        generator=generator, top_k=2, expand_k=3)
    initial = ops.initial_contract(_case())
    independent = ops.independent_source(initial)
    assert initial.evidence[0].retrieval_path == ("initial", "dense")
    assert independent.evidence[0].source_id == "independent"
    assert independent.evidence[0].retrieval_path == ("remediation", "independent_source")


def test_missing_independent_retriever_does_not_fake_independence():
    ops = EvidencePipelineOps(retriever=Retriever("primary"), generator=Generator(),
                              top_k=1, expand_k=2)
    initial = ops.initial_contract(_case())
    assert ops.independent_source(initial) is initial


def test_rrf_is_deterministic():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]]) == ["a", "b"]


def test_multimodal_initial_retrieval_preserves_lineage_under_one_top_k_ceiling():
    case = dict(_case(), image_path="case.jpg")
    generator = Generator()
    ops = EvidencePipelineOps(
        retriever=Retriever("primary"), image_retriever=ImageRetriever(),
        generator=generator, top_k=4, expand_k=5)
    contract = ops.initial_contract(case)
    assert len(contract.evidence) == 4
    channels = {item.retrieval_channel for item in contract.evidence}
    assert channels == {"dense", "image_exemplar"}
    exemplar = next(item for item in contract.evidence if item.retrieval_channel == "image_exemplar")
    assert exemplar.document_id.startswith("exemplar:")
    assert exemplar.citation_id.startswith("HAM10000:")
    assert exemplar.eligible_claim_types == ("image_exemplar_support",)
    assert exemplar.retrieval_path == ("initial", "image_exemplar")


def test_configured_multimodal_arm_fails_closed_without_an_image():
    ops = EvidencePipelineOps(
        retriever=Retriever("primary"), image_retriever=ImageRetriever(),
        generator=Generator(), top_k=2, expand_k=3)
    try:
        ops.initial_contract(_case())
    except ValueError as error:
        assert "no image path" in str(error)
    else:
        raise AssertionError("multimodal arm silently degraded to text-only")


def test_multiview_case_uses_every_ordered_view_for_retrieval_and_generation():
    case = dict(_case(), image_paths=["case-a.jpg", "case-b.jpg"])
    generator = Generator()
    ops = EvidencePipelineOps(
        retriever=Retriever("primary"), image_retriever=ImageRetriever(),
        generator=generator, top_k=4, expand_k=5)
    contract = ops.initial_contract(case)
    assert generator.calls[0]["image_path"] == ["case-a.jpg", "case-b.jpg"]
    assert any(item.retrieval_channel == "image_exemplar" for item in contract.evidence)
