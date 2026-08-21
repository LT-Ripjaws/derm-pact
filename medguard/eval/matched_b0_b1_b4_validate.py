"""One offline matched-input validation for B0, B1, and the B4 component.

This runner proves arm wiring and comparison boundaries only.  It never calls a
generator and therefore produces no diagnostic, RAGAS, or paper-performance
result.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from medguard.evidence import load_frozen_evidence_runtime
from medguard.eval.h2h_b4_adapter import (
    RAW_COSINE_SEMANTICS,
    DenseTextCandidate,
    DynaRAGGMMATKSComponentAdapter,
    ImageExemplarCandidate,
    OrderedViewCandidates,
)
from medguard.eval.h2h_protocol import (
    ArmId,
    ArmProtocol,
    CaseLineage,
    EvidenceIdentity,
    UsageAccounting,
    build_request_key,
    enforce_usage,
    reject_truth_keys,
    summarize_scores,
    verify_identical_lineage,
)
from medguard.eval.matched_b0_b1_validate import (
    DENSE_CANDIDATE_K,
    QUESTION,
    QUERY_TERMS,
    load_matched_development_inputs,
    prepare_matched_b0_b1_validate_only,
)
from medguard.eval.medguard_development_preflight import FrozenDermLIPFeatureEncoder


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/manifests/medguard_h2h_development_v1.json"
PERCEPTION_RECEIPT = ROOT / (
    "research/experiments/medguard_h2h_v1/perception/dermlip_v1/receipt.json"
)
EVIDENCE_RECEIPT = ROOT / "research/evidence/dermatology_corpus_v1/receipt.json"
OUTPUT = ROOT / (
    "research/experiments/medguard_h2h_v1/matched_b0_b1_b4_validate_only.json"
)
SCHEMA_VERSION = "medguard-matched-b0-b1-b4-validate-only-v1"
FINAL_K = 5
# This runner compares exactly these three arms.  Pin them explicitly rather
# than iterating ArmId: the enum carries the whole frozen roster, and this
# report must not silently acquire an arm it never constructed.
ARMS = (ArmId.B0, ArmId.B1, ArmId.B4)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _content_sha256(record: Mapping[str, Any]) -> str:
    text = " ".join(str(record.get("text") or "").split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _QueryCache:
    """Return copies of one real dense result for byte-identical repeated queries."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self._cache: dict[tuple[str, int], tuple[list[dict[str, Any]], list[float]]] = {}

    def search(self, query: str, k: int = 5):
        key = (str(query), int(k))
        if key not in self._cache:
            records, scores = self.delegate.search(query, k)
            self._cache[key] = (
                [dict(record) for record in records],
                [float(score) for score in scores],
            )
        records, scores = self._cache[key]
        return [dict(record) for record in records], list(scores)

    @property
    def distinct_queries(self) -> int:
        return len(self._cache)


def _safe_record(record: Mapping[str, Any]) -> dict[str, Any]:
    document_id = str(record.get("document_id") or record.get("id") or "")
    source_id = str(record.get("source_id") or record.get("source") or "")
    citation_id = str(record.get("citation_id") or "")
    source_version = str(record.get("source_version") or "")
    authority = str(record.get("authority_class") or record.get("authority") or "")
    provenance_uri = str(record.get("provenance_uri") or "")
    content_sha256 = _content_sha256(record)
    if not all((document_id, source_id, citation_id, source_version, authority, provenance_uri)):
        raise ValueError(f"retrieved evidence lacks provenance: {document_id or '<missing>'}")
    declared = str(record.get("content_sha256") or "")
    if declared and declared != content_sha256:
        raise ValueError(f"retrieved evidence content hash mismatch: {document_id}")
    record_sha256 = str(record.get("record_sha256") or "")
    if len(record_sha256) != 64:
        # Frozen exemplar metadata has no stored canonical-record digest. Bind
        # its identity, image bytes, source version, and train label without
        # exposing the label in the report.
        record_sha256 = _canonical_sha256({
            "schema": "medguard-h2h-exemplar-record-v1",
            "document_id": document_id,
            "citation_id": citation_id,
            "source_id": source_id,
            "source_version": source_version,
            "image_sha256": record.get("image_sha256"),
            "canonical_label": record.get("label"),
            "split": record.get("split"),
        })
    return {
        "document_id": document_id,
        "source_id": source_id,
        "citation_id": citation_id,
        "source_version": source_version,
        "authority": authority,
        "provenance_uri": provenance_uri,
        "content_sha256": content_sha256,
        "record_sha256": record_sha256,
    }


def _load_ordered_views() -> dict[str, list[dict[str, str]]]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        payload.get("metadata", {}).get("all_views_retained") is not True
        or int(payload.get("metadata", {}).get("test_records_loaded", -1)) != 0
    ):
        raise ValueError("development manifest violates the all-view/test boundary")
    grouped: dict[str, list[dict[str, str]]] = {}
    for raw in payload.get("records", []):
        # Explicit allowlist: patient truth and clinical metadata never enter.
        row = {
            "case_id": str(raw["case_id"]),
            "cluster_id": str(raw["cluster_id"]),
            "image_path": str(raw["image_path"]),
            "image_sha256": str(raw["image_sha256"]),
            "split": str(raw["split"]),
        }
        if row["split"] != "calibration":
            raise ValueError("matched runner encountered a non-calibration view")
        grouped.setdefault(row["cluster_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["case_id"])
    if len(grouped) != 160 or sum(map(len, grouped.values())) != 244:
        raise ValueError("matched runner requires exactly 160 lesions / 244 views")
    return grouped


def _query(diagnosis: str) -> str:
    term = QUERY_TERMS.get(diagnosis)
    if term is None:
        raise ValueError(f"commitment outside the H2H vocabulary: {diagnosis}")
    return f"dermoscopy {term} diagnosis differential clinical evidence"


def _code_snapshot() -> str:
    paths = [
        Path(__file__).resolve(),
        ROOT / "medguard/eval/matched_b0_b1_validate.py",
        ROOT / "medguard/eval/h2h_b4_adapter.py",
        ROOT / "medguard/eval/h2h_protocol.py",
        ROOT / "medguard/pipeline/adaptive_selector.py",
    ]
    return _canonical_sha256({str(path.relative_to(ROOT)): _file_sha256(path) for path in paths})


def _protocols(perception_artifact_sha256: str) -> dict[ArmId, ArmProtocol]:
    common = {
        "protocol_id": "medguard-derm-evidence-contract-h2h-v1",
        "model_placeholder": "NO_GENERATION_VALIDATE_ONLY",
        "prompt_hash_placeholder": hashlib.sha256(
            b"NO_PROMPT_VALIDATE_ONLY"
        ).hexdigest(),
        "decoding": {"status": "not_applicable_validate_only"},
        "retry_policy": {"maximum_retries": 0},
        "corpus_receipt": _file_sha256(EVIDENCE_RECEIPT),
        "perception_receipt": _file_sha256(PERCEPTION_RECEIPT),
        "perception_artifact": perception_artifact_sha256,
        "code_snapshot": _code_snapshot(),
    }
    configs = {
        ArmId.B0: {"retrieval": "none", "candidate_k": 0, "final_k": 0},
        ArmId.B1: {
            "retrieval": "receipt_verified_text_dense",
            "candidate_k": 20,
            "final_k": 5,
        },
        ArmId.B4: {
            "retrieval": "dense_plus_raw_cosine_image_exemplar",
            "candidate_k": 20,
            "final_k": 5,
            "selector": {
                "kind": "GMM_ATKS_component_proxy",
                "gamma": 0.5,
                "max_components": 4,
                "min_selected": 1,
                "max_selected": 5,
                "covariance_floor": 1e-6,
                "n_init": 5,
                "random_state": 0,
            },
            "view_aggregation": "max_raw_cosine_with_full_view_vector_audit",
            "merge": "balanced_round_robin_image_exemplar_first_shared_top_5",
        },
    }
    if tuple(configs) != ARMS:
        raise ValueError("matched B0/B1/B4 protocol set drifted from its pinned arms")
    return {
        arm: ArmProtocol(arm_id=arm, arm_config=config, **common)
        for arm, config in configs.items()
    }


def run_validate_only(output_path: str | Path = OUTPUT) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cases, input_summary = load_matched_development_inputs()
    views_by_cluster = _load_ordered_views()
    feature_encoder = FrozenDermLIPFeatureEncoder()
    runtime = load_frozen_evidence_runtime(image_encoder=feature_encoder)
    dense = _QueryCache(runtime.text_retriever.dense)
    dense_identity = {
        "kind": "receipt_verified_frozen_dense_index",
        "embedding_model": runtime.text_retriever.dense.model,
        "index_sha256": _file_sha256(
            ROOT / "research/evidence/dermatology_corpus_v1/text_dense.faiss"
        ),
        "corpus_sha256": _file_sha256(
            ROOT / "research/evidence/dermatology_corpus_v1/corpus.jsonl"
        ),
        "receipt_sha256": _file_sha256(EVIDENCE_RECEIPT),
        "bundle_verified": True,
    }
    base = prepare_matched_b0_b1_validate_only(
        dense_retriever=dense, dense_runtime_identity=dense_identity, strict=True,
    )
    base_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in base["cases"]:
        base_rows.setdefault(row["input"]["case_id"], {})[row["arm_id"]] = row
    protocols = _protocols(str(input_summary["perception_artifact_sha256"]))
    lineages = tuple(CaseLineage(
        case_id=case.case_id,
        ordered_image_sha256s=case.ordered_view_sha256s,
        frozen_perception={
            "committed_diagnosis": case.committed_diagnosis,
            "commitment_confidence": case.commitment_confidence,
            "artifact_sha256": case.perception_artifact_sha256,
            "record_sha256": case.perception_record_sha256,
            "distribution_sha256": case.perception_distribution_sha256,
        },
    ) for case in cases)
    verify_identical_lineage({arm: lineages for arm in ARMS}, expected=ARMS)
    b4_adapter = DynaRAGGMMATKSComponentAdapter()
    usage = UsageAccounting()
    for protocol in protocols.values():
        enforce_usage(protocol, usage)

    rows = []
    component_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    for case, lineage in zip(cases, lineages):
        cluster_id = case.cluster_id
        ordered_rows = views_by_cluster.get(cluster_id)
        if not ordered_rows:
            raise ValueError(f"case has no ordered image views: {case.case_id}")
        if tuple(row["image_sha256"] for row in ordered_rows) != tuple(
            case.ordered_view_sha256s
        ):
            raise ValueError(f"ordered view lineage mismatch: {case.case_id}")
        query = _query(case.committed_diagnosis)
        dense_records, dense_scores = dense.search(query, DENSE_CANDIDATE_K)
        if len(dense_records) != DENSE_CANDIDATE_K:
            raise ValueError(f"B4 dense candidates incomplete: {case.case_id}")
        dense_candidates = tuple(DenseTextCandidate(
            evidence_id=_safe_record(record)["document_id"],
            dense_score=score,
            payload=_safe_record(record),
        ) for record, score in zip(dense_records, dense_scores))

        ordered_views = []
        for view in ordered_rows:
            retrieved = runtime.image_retriever.search_records_by_image(
                view["image_path"], DENSE_CANDIDATE_K,
            )
            if len(retrieved) != DENSE_CANDIDATE_K:
                raise ValueError(f"B4 exemplar candidates incomplete: {case.case_id}")
            candidates = []
            for record in retrieved:
                if record.get("split") != "train":
                    raise ValueError("non-train exemplar reached B4")
                score = float(record.get("score"))
                if not math.isfinite(score):
                    raise ValueError("non-finite raw exemplar cosine")
                safe = _safe_record(record)
                candidates.append(ImageExemplarCandidate(
                    exemplar_id=safe["document_id"],
                    cosine_similarity=score,
                    payload=safe,
                ))
            ordered_views.append(OrderedViewCandidates(
                view_id=view["case_id"], candidates=tuple(candidates),
                score_semantics=RAW_COSINE_SEMANTICS,
            ))
        b4 = b4_adapter.run(tuple(ordered_views), dense_candidates)
        if len(b4.merged_evidence) != FINAL_K:
            raise ValueError(f"B4 did not fill the shared top-five: {case.case_id}")
        b4_evidence = []
        for rank, item in enumerate(b4.merged_evidence, start=1):
            safe = dict(item.payload)
            safe.update({
                "rank": rank,
                "modality": item.modality.value,
                "modality_rank": item.modality_rank,
                "selection_score": item.score,
            })
            b4_evidence.append(safe)
            modality_counts[item.modality.value] += 1

        arms = base_rows[case.case_id]
        b0_evidence: Sequence[Mapping[str, Any]] = ()
        b1_evidence = arms["B1"]["retrieval"]["evidence"]
        if [row["document_id"] for row in b1_evidence] != [
            _safe_record(record)["document_id"] for record in dense_records[:5]
        ]:
            raise ValueError(f"B1/B4 dense candidates diverged: {case.case_id}")
        evidence_by_arm = {
            ArmId.B0: b0_evidence,
            ArmId.B1: b1_evidence,
            ArmId.B4: b4_evidence,
        }
        request_keys = {
            arm: build_request_key(
                protocols[arm], lineage, question=QUESTION,
                retrieved_evidence=tuple(EvidenceIdentity(
                    evidence_id=str(record["document_id"]),
                    content_sha256=str(record["content_sha256"]),
                ) for record in evidence_by_arm[arm]),
            ) for arm in ARMS
        }
        if len(set(request_keys.values())) != 3:
            raise ValueError(f"cross-arm request-key collision: {case.case_id}")
        reason = b4.audit.selector_fallback_reason or "none"
        fallback_counts[reason] += 1
        component_counts[str(b4.audit.selector_components)] += 1
        selected_counts[str(b4.audit.selected_image_count)] += 1
        rows.append({
            "case_id": case.case_id,
            "input": case.public_input(),
            "arms": {
                "B0": {
                    "retrieval": arms["B0"]["retrieval"],
                    "cache_namespace": protocols[ArmId.B0].cache_namespace,
                    "request_key": request_keys[ArmId.B0],
                    "score_state": arms["B0"]["score_state"],
                },
                "B1": {
                    "retrieval": arms["B1"]["retrieval"],
                    "cache_namespace": protocols[ArmId.B1].cache_namespace,
                    "request_key": request_keys[ArmId.B1],
                    "score_state": arms["B1"]["score_state"],
                },
                "B4": {
                    "retrieval": {
                        "strategy": "GMM_ATKS_component_proxy",
                        "candidate_k_per_modality": 20,
                        "final_k": 5,
                        "evidence": b4_evidence,
                        "audit": {
                            "ordered_view_ids": list(b4.audit.ordered_view_ids),
                            "selector_input_semantics": b4.audit.selector_input_semantics,
                            "selector_input_count": len(b4.audit.selector_input_ids),
                            "selector_input_sha256": _canonical_sha256({
                                "ids": b4.audit.selector_input_ids,
                                "raw_cosines": b4.audit.selector_input_raw_cosines,
                                "per_view": b4.audit.selector_input_per_view_cosines,
                            }),
                            "selector_components": b4.audit.selector_components,
                            "selector_bic": (
                                b4.audit.selector_bic
                                if math.isfinite(b4.audit.selector_bic) else None
                            ),
                            "selector_fallback_reason": b4.audit.selector_fallback_reason,
                            "fallback_applied": b4.audit.fallback_applied,
                            "selected_image_count": b4.audit.selected_image_count,
                            "dense_candidate_count": b4.audit.dense_candidate_count,
                            "merge_rule": b4.audit.merge_rule,
                        },
                    },
                    "cache_namespace": protocols[ArmId.B4].cache_namespace,
                    "request_key": request_keys[ArmId.B4],
                    "score_state": {
                        "status": "not_executed_validate_only",
                        "values": {
                            "diagnosis_correct": None,
                            "actionable_correct": None,
                            "answer_relevancy": None,
                            "faithfulness": None,
                            "citation_precision": None,
                            "citation_recall": None,
                        },
                        "missing_score_fields": [
                            "diagnosis_correct", "actionable_correct", "answer_relevancy",
                            "faithfulness", "citation_precision", "citation_recall",
                        ],
                        "not_applicable_score_fields": ["evidence_contract_completion"],
                    },
                },
            },
            "usage": asdict(usage),
        })

    missing = {
        arm.value: asdict(summarize_scores({case.case_id: None for case in cases}))
        for arm in ARMS
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "medguard-derm-evidence-contract-h2h-v1",
        "status": "validate_only_not_a_performance_result",
        "boundaries": {
            "development_only": True,
            "sealed_test_access": False,
            "test_records_loaded": 0,
            "truth_fields_consumed": False,
            "generation_calls": 0,
            "logical_generation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "paid_calls": 0,
        },
        "inputs": input_summary,
        "runtime": {
            "evidence_bundle_verified": True,
            "dense_runtime": dense_identity,
            "distinct_dense_queries": dense.distinct_queries,
            "query_feature_model": feature_encoder.model_id,
            "query_feature_sha256": feature_encoder.feature_sha256,
        },
        "protocol": {
            "cache_namespaces": {
                arm.value: protocols[arm].cache_namespace for arm in ARMS
            },
            "arm_configs": {
                arm.value: _plain(protocols[arm].arm_config) for arm in ARMS
            },
            "code_snapshot_sha256": protocols[ArmId.B0].code_snapshot,
            "missing_scores": missing,
        },
        "aggregate": {
            "cases": len(cases),
            "views": sum(len(case.ordered_view_sha256s) for case in cases),
            "arm_rows": len(cases) * 3,
            "b0_final_evidence": 0,
            "b1_final_dense_evidence": sum(
                len(row["arms"]["B1"]["retrieval"]["evidence"]) for row in rows
            ),
            "b4_final_evidence": sum(
                len(row["arms"]["B4"]["retrieval"]["evidence"]) for row in rows
            ),
            "b4_modality_counts": dict(sorted(modality_counts.items())),
            "b4_gmm_component_counts": dict(sorted(component_counts.items())),
            "b4_fallback_counts": dict(sorted(fallback_counts.items())),
            "b4_selected_image_counts": dict(sorted(selected_counts.items())),
            "generation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "boundary_violation_count": 0,
        },
        "interpretation": {
            "verified": (
                "B0, B1, and the B4 GMM-ATKS component receive identical frozen "
                "development input lineage under separate cache/request identities."
            ),
            "not_claimed": (
                "No generation or outcome scoring occurred; this report contains no "
                "accuracy, RAGAS, safety, or rival-beating result."
            ),
        },
        "cases": rows,
    }
    reject_truth_keys(report, path="report")
    if (
        report["aggregate"]["cases"] != 160
        or report["aggregate"]["views"] != 244
        or report["aggregate"]["arm_rows"] != 480
        or report["aggregate"]["b1_final_dense_evidence"] != 800
        or report["aggregate"]["b4_final_evidence"] != 800
        or len(set(report["protocol"]["cache_namespaces"].values())) != 3
    ):
        raise ValueError("matched B0/B1/B4 report failed its final invariant audit")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ) + "\n").encode("utf-8"))
    return report


def main() -> int:
    report = run_validate_only()
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
