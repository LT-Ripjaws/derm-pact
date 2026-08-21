"""Validate-only construction of the matched B0 and B1 H2H arms.

This module deliberately stops before prompting, generation, or outcome scoring.
B0 receives the frozen all-view/perception input and no evidence. B1 receives the
same input plus the true top five from one dense search over the frozen text
index. No BM25, image retrieval, EvidenceContract, verifier, or remediation path
is imported into either arm.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

from medguard.evidence.perception import load_development_perception_bindings
from medguard.eval.h2h_generation import H2H_CLINICAL_QUESTION


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_CARDS = ROOT / "research/implementation_cards/H2H_IMPLEMENTATION_CARDS_V1.json"
PROTOCOL_ID = "medguard-derm-evidence-contract-h2h-v1"
SCHEMA_VERSION = "medguard-matched-b0-b1-validate-only-v1"
QUESTION = H2H_CLINICAL_QUESTION
QUERY_TERMS = {
    "akiec": "actinic keratosis intraepidermal carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesion",
    "df": "dermatofibroma",
    "melanoma": "cutaneous melanoma",
    "nevus": "melanocytic nevus",
    "vasc": "vascular skin lesion",
}
EXPECTED_LESIONS = 160
EXPECTED_VIEWS = 244
DENSE_CANDIDATE_K = 20
DENSE_FINAL_K = 5
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_OUTPUT_KEYS = frozenset({"label", "truth", "target", "ground_truth"})
_MISSING_CASE_SCORE_FIELDS = (
    "diagnosis_correct",
    "actionable_correct",
    "answer_relevancy",
    "faithfulness",
    "citation_precision",
    "citation_recall",
)


class DenseRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> Tuple[Sequence[Mapping[str, Any]], Sequence[float]]:
        ...


class BoundaryViolationError(ValueError):
    """Raised when a constructed arm no longer satisfies its matched card."""


@dataclass(frozen=True)
class MatchedDevelopmentInput:
    case_id: str
    cluster_id: str
    ordered_view_case_ids: Tuple[str, ...]
    ordered_view_sha256s: Tuple[str, ...]
    committed_diagnosis: str
    commitment_confidence: float
    perception_artifact_sha256: str
    perception_record_sha256: str
    perception_distribution_sha256: str
    input_lineage_sha256: str

    def public_input(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "cluster_id": self.cluster_id,
            "ordered_view_case_ids": list(self.ordered_view_case_ids),
            "ordered_view_sha256s": list(self.ordered_view_sha256s),
            "view_count": len(self.ordered_view_case_ids),
            "committed_diagnosis": self.committed_diagnosis,
            "commitment_confidence": self.commitment_confidence,
            "perception_artifact_sha256": self.perception_artifact_sha256,
            "perception_record_sha256": self.perception_record_sha256,
            "perception_distribution_sha256": self.perception_distribution_sha256,
            "input_lineage_sha256": self.input_lineage_sha256,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cards() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(IMPLEMENTATION_CARDS.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "medguard-h2h-implementation-cards-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
    ):
        raise BoundaryViolationError("unsupported H2H implementation-card specification")
    cards = {str(card.get("id")): card for card in payload.get("cards", [])}
    if "B0" not in cards or "B1" not in cards:
        raise BoundaryViolationError("H2H cards do not define both B0 and B1")
    b0, b1 = cards["B0"], cards["B1"]
    if (
        b0.get("retrieval", {}).get("strategy") != "none"
        or b0.get("inputs", {}).get("evidence_channels") != []
        or int(b0.get("retrieval", {}).get("final_k", -1)) != 0
    ):
        raise BoundaryViolationError("B0 card is no longer a no-retrieval comparator")
    b1_retrieval = b1.get("retrieval", {})
    if (
        b1.get("inputs", {}).get("evidence_channels") != ["text_dense"]
        or int(b1_retrieval.get("candidate_k", -1)) != DENSE_CANDIDATE_K
        or int(b1_retrieval.get("final_k", -1)) != DENSE_FINAL_K
        or b1_retrieval.get("graph") is not False
        or b1_retrieval.get("adaptive") is not False
    ):
        raise BoundaryViolationError("B1 card is no longer the pinned dense-only top-five arm")
    return payload, b0, b1


def load_matched_development_inputs() -> tuple[list[MatchedDevelopmentInput], Dict[str, Any]]:
    """Load and verify the exact all-view, truth-redacted 160-lesion input set."""
    payload, _b0, _b1 = _load_cards()
    specification = payload["common_conditions"]
    manifest_spec = specification["dataset"]["development_manifest"]
    perception_spec = specification["frozen_perception"]["artifact"]
    manifest_path = (ROOT / str(manifest_spec["path"])).resolve()
    perception_path = (ROOT / str(perception_spec["path"])).resolve()
    if _sha256(manifest_path) != str(manifest_spec["sha256"]):
        raise BoundaryViolationError("development manifest differs from the H2H card")
    if _sha256(perception_path) != str(perception_spec["sha256"]):
        raise BoundaryViolationError("frozen perception artifact differs from the H2H card")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    if metadata.get("all_views_retained") is not True or int(metadata.get("test_records_loaded", -1)) != 0:
        raise BoundaryViolationError("development manifest violates the all-view/test boundary")
    grouped: Dict[str, list[Dict[str, str]]] = {}
    for raw in manifest.get("records", []):
        # Whitelist only model inputs. In particular, do not read the manifest's
        # diagnosis or clinical metadata fields into this runner.
        projected = {
            "case_id": str(raw["case_id"]),
            "cluster_id": str(raw["cluster_id"]),
            "image_sha256": str(raw["image_sha256"]),
            "modality": str(raw["modality"]),
            "split": str(raw["split"]),
        }
        if projected["split"] != "calibration" or projected["modality"] != "dermoscopy":
            raise BoundaryViolationError("matched input contains a non-calibration dermoscopy view")
        grouped.setdefault(projected["cluster_id"], []).append(projected)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["case_id"])
    if len(grouped) != EXPECTED_LESIONS or sum(map(len, grouped.values())) != EXPECTED_VIEWS:
        raise BoundaryViolationError("matched input must contain exactly 160 lesions / 244 views")

    bindings = load_development_perception_bindings(
        predictions_path=perception_path, manifest_path=manifest_path,
    )
    expected_binding_ids = {f"lesion:{cluster_id}" for cluster_id in grouped}
    if set(bindings) != expected_binding_ids:
        raise BoundaryViolationError("perception and manifest case sets differ")

    inputs: list[MatchedDevelopmentInput] = []
    for cluster_id in sorted(grouped):
        rows = grouped[cluster_id]
        case_id = f"lesion:{cluster_id}"
        binding = bindings[case_id]
        view_ids = tuple(row["case_id"] for row in rows)
        view_hashes = tuple(row["image_sha256"] for row in rows)
        if view_hashes != tuple(binding.view_image_sha256s):
            raise BoundaryViolationError(f"ordered image lineage differs for {case_id}")
        commitment = binding.commitment
        probabilities = tuple(float(value) for value in binding.diagnosis_probabilities)
        distribution_hash = _canonical_sha256(probabilities)
        lineage_payload = {
            "schema": "medguard-matched-input-lineage-v1",
            "case_id": case_id,
            "ordered_view_case_ids": view_ids,
            "ordered_view_sha256s": view_hashes,
            "perception_artifact_sha256": binding.artifact_sha256,
            "perception_record_sha256": binding.record_sha256,
            "perception_model_id": binding.model_id,
            "perception_class_order": binding.class_order,
            "perception_distribution_sha256": distribution_hash,
            "committed_diagnosis": commitment.concept_code,
            "commitment_confidence": commitment.confidence,
            "question_sha256": hashlib.sha256(QUESTION.encode("utf-8")).hexdigest(),
        }
        inputs.append(MatchedDevelopmentInput(
            case_id=case_id,
            cluster_id=cluster_id,
            ordered_view_case_ids=view_ids,
            ordered_view_sha256s=view_hashes,
            committed_diagnosis=commitment.concept_code,
            commitment_confidence=commitment.confidence,
            perception_artifact_sha256=binding.artifact_sha256,
            perception_record_sha256=binding.record_sha256,
            perception_distribution_sha256=distribution_hash,
            input_lineage_sha256=_canonical_sha256(lineage_payload),
        ))
    return inputs, {
        "implementation_cards_sha256": _sha256(IMPLEMENTATION_CARDS),
        "development_manifest_sha256": _sha256(manifest_path),
        "perception_artifact_sha256": _sha256(perception_path),
        "lesions": len(inputs),
        "views": sum(len(item.ordered_view_case_ids) for item in inputs),
    }


def load_frozen_dense_retriever() -> tuple[DenseRetriever, Dict[str, Any]]:
    """Load only the receipt-verified dense index; do not construct hybrid/image paths."""
    from medguard import config
    from medguard.evidence.bundle import verify_evidence_bundle
    from medguard.pipeline.retriever import FaissTextRetriever

    base = Path(config.EVIDENCE_BUNDLE_DIR).resolve()
    receipt = verify_evidence_bundle(base)
    retriever = FaissTextRetriever.load(
        index_path=str(base / "text_dense.faiss"),
        meta_path=str(base / "corpus.jsonl"),
    )
    return retriever, {
        "kind": "receipt_verified_frozen_dense_index",
        "embedding_model": retriever.model,
        "index_sha256": _sha256(base / "text_dense.faiss"),
        "corpus_sha256": _sha256(base / "corpus.jsonl"),
        "receipt_sha256": _sha256(base / "receipt.json"),
        "bundle_verified": True,
        "runtime_lock_required_before_paid_h2h": bool(
            receipt.get("_verification", {}).get("runtime_lock_required_before_paid_h2h")
        ),
    }


def _dense_query(case: MatchedDevelopmentInput) -> str:
    term = QUERY_TERMS.get(case.committed_diagnosis)
    if term is None:
        raise BoundaryViolationError(
            f"frozen commitment is outside the H2H diagnosis vocabulary: {case.case_id}"
        )
    return f"dermoscopy {term} diagnosis differential clinical evidence"


def _validate_dense_candidates(
    *, case_id: str, records: Sequence[Mapping[str, Any]], scores: Sequence[float],
) -> tuple[list[Dict[str, Any]], list[str]]:
    violations: list[str] = []
    if len(records) != DENSE_CANDIDATE_K or len(scores) != DENSE_CANDIDATE_K:
        violations.append(
            f"{case_id}:B1:dense_search_must_return_exactly_{DENSE_CANDIDATE_K}_candidates"
        )
    if len(records) != len(scores):
        violations.append(f"{case_id}:B1:record_score_length_mismatch")
    finite_scores = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in finite_scores):
        violations.append(f"{case_id}:B1:non_finite_dense_score")
    if any(left < right for left, right in zip(finite_scores, finite_scores[1:])):
        violations.append(f"{case_id}:B1:dense_scores_not_ranked_descending")

    evidence: list[Dict[str, Any]] = []
    identities: set[str] = set()
    for rank, (raw, score) in enumerate(zip(records[:DENSE_FINAL_K], finite_scores[:DENSE_FINAL_K]), start=1):
        record = dict(raw)
        channel = str(record.get("retrieval_channel") or "")
        document_id = str(record.get("document_id") or record.get("id") or "")
        source_id = str(record.get("source_id") or record.get("source") or "")
        citation_id = str(record.get("citation_id") or "")
        source_version = str(record.get("source_version") or "")
        authority = str(record.get("authority_class") or record.get("authority") or "")
        provenance_uri = str(record.get("provenance_uri") or "")
        content_sha = str(record.get("content_sha256") or "")
        record_sha = str(record.get("record_sha256") or "")
        text = str(record.get("text") or "")
        if channel != "dense":
            violations.append(f"{case_id}:B1:non_dense_channel_at_rank_{rank}")
        if not document_id or document_id in identities:
            violations.append(f"{case_id}:B1:missing_or_duplicate_document_at_rank_{rank}")
        identities.add(document_id)
        if not source_id:
            violations.append(f"{case_id}:B1:missing_source_id_at_rank_{rank}")
        if not citation_id:
            violations.append(f"{case_id}:B1:missing_citation_id_at_rank_{rank}")
        if not source_version:
            violations.append(f"{case_id}:B1:missing_source_version_at_rank_{rank}")
        if not authority:
            violations.append(f"{case_id}:B1:missing_authority_at_rank_{rank}")
        if not provenance_uri:
            violations.append(f"{case_id}:B1:missing_provenance_uri_at_rank_{rank}")
        if not _SHA256_RE.fullmatch(content_sha) or hashlib.sha256(text.encode("utf-8")).hexdigest() != content_sha:
            violations.append(f"{case_id}:B1:content_hash_mismatch_at_rank_{rank}")
        if not _SHA256_RE.fullmatch(record_sha):
            violations.append(f"{case_id}:B1:missing_record_hash_at_rank_{rank}")
        evidence.append({
            "rank": rank,
            "document_id": document_id,
            "source_id": source_id,
            "citation_id": citation_id,
            "source_version": source_version,
            "authority": authority,
            "provenance_uri": provenance_uri,
            "content_sha256": content_sha,
            "record_sha256": record_sha,
            "retrieval_channel": channel,
            "dense_score": score,
        })
    if len(evidence) != DENSE_FINAL_K:
        violations.append(f"{case_id}:B1:final_evidence_must_equal_{DENSE_FINAL_K}")
    return evidence, violations


def _empty_call_accounting() -> Dict[str, Any]:
    return {
        "actual_attempts": 0,
        "logical_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_seconds": 0.0,
        "cache_hit": False,
        "generation_error": None,
        "answered": False,
        "abstained": False,
        "status": "not_executed_validate_only",
    }


def _score_state(arm_id: str) -> Dict[str, Any]:
    not_applicable = ["evidence_contract_completion"]
    if arm_id == "B0":
        not_applicable.extend(("retrieval_score", "citation_precision", "citation_recall"))
    return {
        "status": "not_executed_validate_only",
        "values": {field: None for field in _MISSING_CASE_SCORE_FIELDS},
        "missing_score_fields": list(_MISSING_CASE_SCORE_FIELDS),
        "not_applicable_score_fields": not_applicable,
    }


def _has_forbidden_output_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(str(key) in _FORBIDDEN_OUTPUT_KEYS for key in value):
            return True
        return any(_has_forbidden_output_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_output_key(item) for item in value)
    return False


def prepare_matched_b0_b1_validate_only(
    *,
    dense_retriever: DenseRetriever | None = None,
    dense_runtime_identity: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Construct both arms without generation and return an in-memory audit report.

    A retriever may be injected for focused tests. Production validation leaves it
    unset so this function loads the receipt-verified frozen FAISS dense index.
    """
    inputs, input_summary = load_matched_development_inputs()
    if dense_retriever is None:
        dense_retriever, loaded_identity = load_frozen_dense_retriever()
        dense_identity = loaded_identity
    else:
        dense_identity = dict(dense_runtime_identity or {
            "kind": "dependency_injected_dense_retriever",
            "bundle_verified": False,
        })

    rows: list[Dict[str, Any]] = []
    global_violations: list[str] = []
    for case in inputs:
        common_input = case.public_input()
        b0 = {
            "arm_id": "B0",
            "input": common_input,
            "retrieval": {
                "strategy": "none",
                "candidate_k": 0,
                "final_k": 0,
                "query_sha256": None,
                "evidence": [],
            },
            "components": {
                "retrieval": False,
                "bm25": False,
                "image_exemplar": False,
                "evidence_contract": False,
                "verification_gate": False,
                "remediation": False,
            },
            "call_accounting": _empty_call_accounting(),
            "score_state": _score_state("B0"),
            "boundary_violations": [],
        }
        rows.append(b0)

        query = _dense_query(case)
        records, scores = dense_retriever.search(query, DENSE_CANDIDATE_K)
        evidence, violations = _validate_dense_candidates(
            case_id=case.case_id, records=list(records), scores=list(scores),
        )
        b1 = {
            "arm_id": "B1",
            "input": common_input,
            "retrieval": {
                "strategy": "single_pass_frozen_text_dense",
                "candidate_k": DENSE_CANDIDATE_K,
                "final_k": DENSE_FINAL_K,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "evidence": evidence,
            },
            "components": {
                "retrieval": True,
                "bm25": False,
                "image_exemplar": False,
                "evidence_contract": False,
                "verification_gate": False,
                "remediation": False,
            },
            "call_accounting": _empty_call_accounting(),
            "score_state": _score_state("B1"),
            "boundary_violations": violations,
        }
        rows.append(b1)
        global_violations.extend(violations)

    by_case: Dict[str, Dict[str, str]] = {}
    for row in rows:
        case_id = str(row["input"]["case_id"])
        by_case.setdefault(case_id, {})[str(row["arm_id"])] = str(
            row["input"]["input_lineage_sha256"]
        )
    for case_id, lineage in sorted(by_case.items()):
        if set(lineage) != {"B0", "B1"} or lineage["B0"] != lineage["B1"]:
            global_violations.append(f"{case_id}:cross_arm_input_lineage_mismatch")

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "validate_only_not_a_result",
        "boundaries": {
            "development_only": True,
            "sealed_test_access": False,
            "test_records_loaded": 0,
            "truth_fields_consumed": False,
            "generation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "paid_calls": 0,
        },
        "inputs": input_summary,
        "dense_runtime": dense_identity,
        "aggregate": {
            "lesions": len(inputs),
            "views": sum(len(case.ordered_view_case_ids) for case in inputs),
            "arm_rows": len(rows),
            "b0_rows": sum(row["arm_id"] == "B0" for row in rows),
            "b1_rows": sum(row["arm_id"] == "B1" for row in rows),
            "b1_final_dense_evidence": sum(
                len(row["retrieval"]["evidence"]) for row in rows if row["arm_id"] == "B1"
            ),
            "generation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "boundary_violation_count": len(global_violations),
        },
        "boundary_violations": global_violations,
        "cases": rows,
    }
    if _has_forbidden_output_key(report):
        global_violations.append("report_contains_forbidden_outcome_field")
        report["aggregate"]["boundary_violation_count"] = len(global_violations)
    if strict and global_violations:
        preview = "; ".join(global_violations[:5])
        raise BoundaryViolationError(
            f"matched B0/B1 construction failed with {len(global_violations)} violation(s): {preview}"
        )
    return report


__all__ = [
    "BoundaryViolationError",
    "DenseRetriever",
    "MatchedDevelopmentInput",
    "load_frozen_dense_retriever",
    "load_matched_development_inputs",
    "prepare_matched_b0_b1_validate_only",
]
