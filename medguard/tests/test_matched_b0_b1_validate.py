from __future__ import annotations

import hashlib

import pytest

from medguard.eval.matched_b0_b1_validate import (
    BoundaryViolationError,
    load_matched_development_inputs,
    prepare_matched_b0_b1_validate_only,
)


class FakeDenseRetriever:
    def __init__(self, *, channel: str = "dense") -> None:
        self.channel = channel
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5):
        self.calls.append((query, k))
        records = []
        scores = []
        for index in range(k):
            text = f"frozen dense document {index} for {query}"
            records.append({
                "document_id": f"doc:{hashlib.sha256(query.encode()).hexdigest()}:{index}",
                "source_id": f"source:{index}",
                "citation_id": f"CITATION:{index}",
                "source_version": "focused-test-v1",
                "authority_class": "focused_test_authority",
                "provenance_uri": f"https://example.invalid/evidence/{index}",
                "text": text,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "record_sha256": hashlib.sha256(
                    f"record:{query}:{index}".encode("utf-8")
                ).hexdigest(),
                "retrieval_channel": self.channel,
            })
            scores.append(1.0 - index / 100.0)
        return records, scores


def _identity():
    return {
        "kind": "focused_test_dense_retriever",
        "bundle_verified": True,
        "index_sha256": "0" * 64,
        "corpus_sha256": "1" * 64,
    }


def test_matched_inputs_are_exact_all_view_truth_redacted_development_inputs():
    cases, summary = load_matched_development_inputs()

    assert len(cases) == summary["lesions"] == 160
    assert sum(len(case.ordered_view_case_ids) for case in cases) == summary["views"] == 244
    assert all(case.case_id == f"lesion:{case.cluster_id}" for case in cases)
    assert all(len(case.ordered_view_case_ids) == len(case.ordered_view_sha256s) for case in cases)
    assert len({case.input_lineage_sha256 for case in cases}) == 160


def test_b0_b1_validate_only_uses_identical_inputs_and_true_dense_top_five():
    retriever = FakeDenseRetriever()
    report = prepare_matched_b0_b1_validate_only(
        dense_retriever=retriever,
        dense_runtime_identity=_identity(),
    )

    assert report["aggregate"] == {
        "lesions": 160,
        "views": 244,
        "arm_rows": 320,
        "b0_rows": 160,
        "b1_rows": 160,
        "b1_final_dense_evidence": 800,
        "generation_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "boundary_violation_count": 0,
    }
    assert report["boundary_violations"] == []
    assert len(retriever.calls) == 160
    assert {k for _query, k in retriever.calls} == {20}

    by_case = {}
    for row in report["cases"]:
        by_case.setdefault(row["input"]["case_id"], {})[row["arm_id"]] = row
        assert row["call_accounting"]["actual_attempts"] == 0
        assert row["call_accounting"]["logical_calls"] == 0
        assert row["score_state"]["status"] == "not_executed_validate_only"
        assert row["score_state"]["missing_score_fields"]
        assert row["components"]["evidence_contract"] is False
        assert row["components"]["verification_gate"] is False
        assert row["components"]["remediation"] is False
    assert len(by_case) == 160
    for arms in by_case.values():
        assert set(arms) == {"B0", "B1"}
        assert arms["B0"]["input"] == arms["B1"]["input"]
        assert arms["B0"]["retrieval"]["evidence"] == []
        assert arms["B0"]["components"]["retrieval"] is False
        evidence = arms["B1"]["retrieval"]["evidence"]
        assert [item["rank"] for item in evidence] == [1, 2, 3, 4, 5]
        assert {item["retrieval_channel"] for item in evidence} == {"dense"}
        assert all(item["citation_id"] and item["source_version"] for item in evidence)
        assert all(item["authority"] and item["provenance_uri"] for item in evidence)
        assert [item["dense_score"] for item in evidence] == sorted(
            [item["dense_score"] for item in evidence], reverse=True,
        )

    def visit(value):
        if isinstance(value, dict):
            assert not ({"label", "truth", "target", "ground_truth"} & set(value))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(report)


def test_b1_non_dense_channel_is_reported_and_strict_mode_fails_closed():
    report = prepare_matched_b0_b1_validate_only(
        dense_retriever=FakeDenseRetriever(channel="bm25"),
        dense_runtime_identity=_identity(),
        strict=False,
    )
    assert report["aggregate"]["boundary_violation_count"] == 160 * 5
    assert all("non_dense_channel" in item for item in report["boundary_violations"])

    with pytest.raises(BoundaryViolationError, match="non_dense_channel"):
        prepare_matched_b0_b1_validate_only(
            dense_retriever=FakeDenseRetriever(channel="image_exemplar"),
            dense_runtime_identity=_identity(),
            strict=True,
        )
