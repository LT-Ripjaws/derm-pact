"""One offline matched-input validation for the full H2H arm set.

This runner proves arm wiring and comparison boundaries only.  It never calls a
generator, so it produces no diagnostic, RAGAS, safety, or rival-beating result.
Every number it emits is an engineering observation about wiring.

Arms: B0, B1, B4 (reusing the existing matched adapters byte-for-byte), the
MedGraphRAG-inspired graph comparator B5, the Agentic-Graph-RAG-inspired bounded
refinement comparator B6, the proposed method M, and the paired
leave-one-component-out ablations A0-A4.

Two construction choices are called out because they are load-bearing:

* ``A0`` is a declared alias of ``M``.  It reuses M's protocol object, so its
  execution payload, cache namespace, and request key are byte-identical by
  construction. The emitted row adds only ``alias_of: M`` as audit metadata.
* The host-built contract leaves ``unresolved_uncertainties`` empty at
  construction and lets structured verification type the failure.  The C1+C2
  preflight pre-declares uncertainties instead, which makes every case abstain
  before the deterministic hierarchy can choose an action.  Uncertainty is a
  termination-time record, not a retrieval-time input, and pre-declaring it here
  would collapse A3 onto M and hide the remediation contrast this study exists
  to measure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from medguard.evidence import (
    EvidenceContract,
    UncertaintyType,
    UnresolvedUncertainty,
    VerificationState,
    contract_sha256,
    evidence_from_chunk,
    load_development_perception_bindings,
    load_frozen_evidence_runtime,
    score_contract,
    verify_contract,
)
from medguard.evidence.claim_builder import (
    DIAGNOSIS_CLAIM_ID,
    NO_EXEMPLAR_SUPPORT_DESCRIPTION,
    NO_EXEMPLAR_SUPPORT_UNCERTAINTY_ID,
    build_image_bound_claims,
)
from medguard.evidence.verification import FailureType
from medguard.eval.h2h_b4_adapter import (
    RAW_COSINE_SEMANTICS,
    DenseTextCandidate,
    DynaRAGGMMATKSComponentAdapter,
    ImageExemplarCandidate,
    OrderedViewCandidates,
)
from medguard.eval.h2h_b6_adapter import (
    MAX_ATTEMPTS,
    MAX_REFINEMENTS,
    METRIC_THRESHOLD,
    SUFFICIENCY_THRESHOLD,
    WEIGHT_COVERAGE,
    WEIGHT_DENSITY,
    WEIGHT_SEMANTIC_RELEVANCE,
    LoopDecision,
    RefinementBounds,
    SearchStrategy,
    assert_disjoint_from_medguard_failures,
    run_bounded_refinement,
)
from medguard.eval.h2h_evidence_graph import (
    CONCEPT_TYPE,
    DELTA_R,
    GRAPH_SCHEMA_VERSION,
    EdgeType,
    EvidenceLineageGraph,
    GraphRetrieval,
    NodeType,
    aggregate_context_text,
    build_evidence_lineage_graph,
    concept_content,
    link_concepts_by_embedding,
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
# Import the frozen projections rather than restating them: a second copy of the
# truth-redacting evidence projection is a divergence risk, not reuse.
from medguard.eval.matched_b0_b1_b4_validate import (
    _canonical_sha256,
    _file_sha256,
    _load_ordered_views,
    _plain,
    _QueryCache,
    _safe_record,
)
from medguard.eval.medguard_development_preflight import (
    FrozenDermLIPFeatureEncoder,
    _exemplar_record_sha256,
)
from medguard.pipeline.hybrid_retriever import reciprocal_rank_fusion_records
from medguard.remediation.actions import Action
from medguard.remediation.policy import ActionBudget, DeterministicRemediationPolicy
from medguard.remediation.runner import run_remediation


ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_RECEIPT = ROOT / (
    "research/experiments/medguard_h2h_v1/perception/dermlip_v1/receipt.json"
)
EVIDENCE_BUNDLE = ROOT / "research/evidence/dermatology_corpus_v1"
EVIDENCE_RECEIPT = EVIDENCE_BUNDLE / "receipt.json"
OUTPUT = ROOT / (
    "research/experiments/medguard_h2h_v1/matched_full_arm_validate_only.json"
)
SCHEMA_VERSION = "medguard-matched-full-arm-validate-only-v1"
PROTOCOL_ID = "medguard-derm-evidence-contract-h2h-v1"

TEXT_CANDIDATE_K = 20
IMAGE_CANDIDATE_K = 20
FINAL_K = 5
# Pinned explicitly rather than inherited from class defaults; the card lists the
# unresolved expand_k runtime mismatch as a pre-run lock blocker.
EXPAND_K = 20
MAX_REMEDIATION_STEPS = 4

ARMS = (
    ArmId.B0, ArmId.B1, ArmId.B4, ArmId.B5, ArmId.B6,
    ArmId.M, ArmId.A0, ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4,
)
MEDGUARD_ARMS = (ArmId.M, ArmId.A0, ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4)

# Frozen deterministic query templates.  Both are study parameters pinned before
# any H2H output was viewed; neither is tuned on outcomes.
QUERY_TEMPLATE = "dermoscopy {term} diagnosis differential clinical evidence"
REFORMULATED_QUERY_TEMPLATE = (
    "{term} dermoscopic diagnostic criteria and supporting clinical evidence"
)

# A4 answers without a verification gate, so it cannot use the contract's PASS
# action (PASS requires a satisfied contract).  These labels keep the ungated
# path explicit and keep malformed structured output a failure.
ACTION_ANSWER_UNGATED = "ANSWER_UNGATED"
ACTION_STRUCTURED_OUTPUT_ERROR = "STRUCTURED_OUTPUT_ERROR"


@dataclass(frozen=True)
class MedGuardArmConfig:
    """One MedGuard configuration; each ablation flips exactly one flag."""

    arm_id: ArmId
    image_exemplar_retrieval: bool = True
    graph_lineage: bool = True
    bounded_remediation: bool = True
    verification_gate: bool = True

    @property
    def component_flags(self) -> Dict[str, bool]:
        return {
            "image_exemplar_retrieval": self.image_exemplar_retrieval,
            "graph_lineage": self.graph_lineage,
            "bounded_remediation": self.bounded_remediation,
            "verification_gate": self.verification_gate,
        }

    def components(self) -> Dict[str, Any]:
        return {
            "authoritative_text_retrieval": True,
            "evidence_contract": True,
            "flat_provenance": True,
            **self.component_flags,
        }


MEDGUARD_CONFIGS: Dict[ArmId, MedGuardArmConfig] = {
    ArmId.M: MedGuardArmConfig(ArmId.M),
    ArmId.A1: MedGuardArmConfig(ArmId.A1, image_exemplar_retrieval=False),
    ArmId.A2: MedGuardArmConfig(ArmId.A2, graph_lineage=False),
    ArmId.A3: MedGuardArmConfig(ArmId.A3, bounded_remediation=False),
    ArmId.A4: MedGuardArmConfig(ArmId.A4, verification_gate=False),
}


def single_component_deltas() -> Dict[str, List[str]]:
    """Machine-check that every ablation differs from M in exactly one component."""
    reference = MEDGUARD_CONFIGS[ArmId.M].component_flags
    if not all(reference.values()):
        raise ValueError("M must enable every MedGuard component")
    deltas: Dict[str, List[str]] = {}
    for arm in (ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4):
        flags = MEDGUARD_CONFIGS[arm].component_flags
        changed = sorted(
            name for name, value in flags.items() if reference[name] != value
        )
        if len(changed) != 1:
            raise ValueError(
                f"{arm.value} must change exactly one component, changed={changed}"
            )
        deltas[arm.value] = changed
    if len({tuple(value) for value in deltas.values()}) != len(deltas):
        raise ValueError("two ablations disable the same component")
    return deltas


def _a0_row_differs_only_by_alias(
    m_row: Mapping[str, Any], a0_row: Mapping[str, Any],
) -> bool:
    """Verify the emitted alias row adds no difference beyond its annotation."""
    projected = dict(a0_row)
    if projected.pop("alias_of", None) != ArmId.M.value:
        return False
    return json.dumps(
        projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ) == json.dumps(
        m_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _query(diagnosis: str) -> str:
    term = QUERY_TERMS.get(diagnosis)
    if term is None:
        raise ValueError(f"commitment outside the H2H vocabulary: {diagnosis}")
    return QUERY_TEMPLATE.format(term=term)


def _reformulated_query(diagnosis: str) -> str:
    term = QUERY_TERMS.get(diagnosis)
    if term is None:
        raise ValueError(f"commitment outside the H2H vocabulary: {diagnosis}")
    return REFORMULATED_QUERY_TEMPLATE.format(term=term)


def _code_snapshot() -> str:
    paths = [
        Path(__file__).resolve(),
        ROOT / "medguard/eval/matched_b0_b1_validate.py",
        ROOT / "medguard/eval/matched_b0_b1_b4_validate.py",
        ROOT / "medguard/eval/h2h_b4_adapter.py",
        ROOT / "medguard/eval/h2h_b6_adapter.py",
        ROOT / "medguard/eval/h2h_evidence_graph.py",
        ROOT / "medguard/eval/h2h_protocol.py",
        ROOT / "medguard/eval/medguard_development_preflight.py",
        ROOT / "medguard/evidence/contract.py",
        ROOT / "medguard/evidence/relation_assessor.py",
        ROOT / "medguard/evidence/scoring.py",
        ROOT / "medguard/evidence/verification.py",
        ROOT / "medguard/pipeline/adaptive_selector.py",
        ROOT / "medguard/pipeline/hybrid_retriever.py",
        ROOT / "medguard/remediation/policy.py",
        ROOT / "medguard/remediation/runner.py",
    ]
    return _canonical_sha256(
        {str(path.relative_to(ROOT)): _file_sha256(path) for path in paths}
    )


def _graph_arm_config(depth: int, node_limit: int) -> Dict[str, Any]:
    return {
        "schema": GRAPH_SCHEMA_VERSION,
        "delta_r": DELTA_R,
        "concept_type": CONCEPT_TYPE,
        "seed": "frozen_committed_diagnosis_to_controlled_concept",
        "depth": depth,
        "node_limit": node_limit,
    }


def _protocols(perception_artifact_sha256: str, graph_sha256: str) -> Dict[ArmId, ArmProtocol]:
    common = {
        "protocol_id": PROTOCOL_ID,
        "model_placeholder": "NO_GENERATION_VALIDATE_ONLY",
        "prompt_hash_placeholder": hashlib.sha256(b"NO_PROMPT_VALIDATE_ONLY").hexdigest(),
        "decoding": {"status": "not_applicable_validate_only"},
        "retry_policy": {"maximum_retries": 0},
        "corpus_receipt": _file_sha256(EVIDENCE_RECEIPT),
        "perception_receipt": _file_sha256(PERCEPTION_RECEIPT),
        "perception_artifact": perception_artifact_sha256,
        "code_snapshot": _code_snapshot(),
    }
    medguard_common = {
        "retrieval": "receipt_verified_dense_bm25_and_all_view_exemplar_rrf",
        "candidate_k": TEXT_CANDIDATE_K,
        "final_k": FINAL_K,
        "expand_k": EXPAND_K,
        "evidence_contract": True,
        "remediation_priority": [
            "SOURCE_CONTRADICTION->INDEPENDENT_SOURCE",
            "MISSING_EVIDENCE->EXPAND",
            "MISALIGNED_EVIDENCE->REFORMULATE",
        ],
        "graph_sha256": graph_sha256,
    }
    configs: Dict[ArmId, Dict[str, Any]] = {
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
        ArmId.B5: {
            "retrieval": "bounded_evidence_lineage_graph_traversal",
            "candidate_k": TEXT_CANDIDATE_K,
            "final_k": FINAL_K,
            "graph": _graph_arm_config(depth=1, node_limit=TEXT_CANDIDATE_K),
            "graph_sha256": graph_sha256,
        },
        ArmId.B6: {
            "retrieval": "bounded_retrieve_evaluate_refine_over_b5_graph",
            "candidate_k": TEXT_CANDIDATE_K,
            "final_k": FINAL_K,
            "graph": _graph_arm_config(
                depth=RefinementBounds().start_depth,
                node_limit=RefinementBounds().start_node_limit,
            ),
            "graph_sha256": graph_sha256,
            "sufficiency_weights": {
                "semantic_relevance": WEIGHT_SEMANTIC_RELEVANCE,
                "coverage": WEIGHT_COVERAGE,
                "density": WEIGHT_DENSITY,
            },
            "sufficiency_basis": {
                "semantic_relevance": (
                    "cosine(frozen_text_embedding(query), "
                    "frozen_text_embedding(rendered_subgraph_context))"
                ),
                "coverage": "retrieved_evidence_nodes / shared_candidate_k",
                "density": "mean_populated_attributes_per_subgraph_node / 8",
                "measured_over": "bounded_retrieved_subgraph_not_returned_prefix",
            },
            "sufficiency_threshold": SUFFICIENCY_THRESHOLD,
            "metric_threshold": METRIC_THRESHOLD,
            "maximum_refinements": MAX_REFINEMENTS,
            "maximum_attempts": MAX_ATTEMPTS,
            "refinement_bounds": asdict(RefinementBounds()),
            "unresolved_insufficiency": "ABSTAIN",
        },
        ArmId.M: {**medguard_common, **MEDGUARD_CONFIGS[ArmId.M].components()},
    }
    for arm in (ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4):
        configs[arm] = {**medguard_common, **MEDGUARD_CONFIGS[arm].components()}
    protocols = {
        arm: ArmProtocol(arm_id=arm, arm_config=config, **common)
        for arm, config in configs.items()
    }
    # A0 is a declared alias of M: it reuses M's protocol identity rather than
    # constructing a look-alike, so byte-equivalence cannot drift.
    protocols[ArmId.A0] = protocols[ArmId.M]
    if set(protocols) != set(ARMS):
        raise ValueError("full-arm protocol set drifted from its pinned roster")
    return protocols


def _projected_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Truth-redacted evidence projection with a bound record hash for both modalities."""
    row = dict(record)
    if str(row.get("retrieval_channel") or "") == "image_exemplar":
        row["record_sha256"] = _exemplar_record_sha256(row)
    safe = _safe_record(row)
    safe["retrieval_channel"] = str(row.get("retrieval_channel") or "")
    return safe


def _graph_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        safe = _projected_record(record)
        rows.append({
            "document_id": safe["document_id"],
            "citation_id": safe["citation_id"],
            "source_id": safe["source_id"],
            "source_version": safe["source_version"],
            "authority_class": safe["authority"],
            "provenance_uri": safe["provenance_uri"],
            "content_sha256": safe["content_sha256"],
            "record_sha256": safe["record_sha256"],
            "title": safe["document_id"],
        })
    return rows


def _build_contract(
    *, case_id: str, perception: Any, records: Sequence[Mapping[str, Any]], question: str,
) -> Tuple[EvidenceContract, List[Dict[str, str]]]:
    """Build one host-owned contract with no generator and no pre-declared uncertainty."""
    evidence = []
    record_by_evidence: Dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(records, start=1):
        record = dict(raw_record)
        channel = str(record.get("retrieval_channel") or "")
        if channel == "image_exemplar":
            record["record_sha256"] = _exemplar_record_sha256(record)
            path = ("initial", "all_view_image_rrf", "cross_modal_rrf")
        else:
            path = ("initial", "dense_bm25_rrf", "cross_modal_rrf")
        item = evidence_from_chunk(record, evidence_id=f"e{index}", retrieval_path=path)
        evidence.append(item)
        record_by_evidence[item.evidence_id] = record

    claims: List[ClinicalClaim] = []
    relations: List[Dict[str, str]] = []
    claims, relations, supporting_exemplars = build_image_bound_claims(
        perception=perception, evidence=evidence, record_by_evidence=record_by_evidence,
    )

    # An arm that retrieves no corroborating exemplar asserts nothing about the
    # visual neighbourhood, and must not be rewarded for it.  Without this an
    # ablation that removes image-exemplar retrieval entirely (A1) would make no
    # claims, fail nothing, and PASS every case -- so deleting evidence would score
    # as an improvement.  Corroboration has to be earned, so its absence is
    # recorded as unresolved uncertainty and the loop resolves it.
    unresolved = [] if supporting_exemplars else [UnresolvedUncertainty(
        uncertainty_id=NO_EXEMPLAR_SUPPORT_UNCERTAINTY_ID,
        uncertainty_type=UncertaintyType.EVIDENCE_SUPPORT,
        description=NO_EXEMPLAR_SUPPORT_DESCRIPTION,
        related_claim_ids=(DIAGNOSIS_CLAIM_ID,),
    )]

    contract = EvidenceContract(
        case_id=case_id,
        modality="dermoscopy",
        clinical_question=question,
        perception=perception,
        claims=claims,
        evidence=evidence,
        unresolved_uncertainties=unresolved,
        verification_state=VerificationState.PENDING,
        proposed_action="",
        termination_reason="",
    ).validate()
    return contract, relations


class _OfflineRemediationTools:
    """Zero-generation remediation tools: re-retrieve and rebuild the contract.

    ``independent_source`` is intentionally inert.  A faithful INDEPENDENT_SOURCE
    action requires a separately configured source/index, which this study does
    not yet have; returning the contract unchanged fails closed instead of
    claiming source independence the wiring cannot deliver.
    """

    def __init__(
        self,
        *,
        case_id: str,
        diagnosis: str,
        perception: Any,
        retrieve: Callable[[str, int], List[Dict[str, Any]]],
    ) -> None:
        self.case_id = case_id
        self.diagnosis = diagnosis
        self.perception = perception
        self._retrieve = retrieve
        self.applied: List[str] = []

    def _rebuild(self, question: str, records: Sequence[Mapping[str, Any]]) -> EvidenceContract:
        contract, _relations = _build_contract(
            case_id=self.case_id, perception=self.perception,
            records=records, question=question,
        )
        return contract

    def expand(self, contract: EvidenceContract) -> EvidenceContract:
        self.applied.append("EXPAND")
        records = self._retrieve(_query(self.diagnosis), EXPAND_K)
        return self._rebuild(contract.clinical_question, records)

    def reformulate(self, contract: EvidenceContract) -> EvidenceContract:
        self.applied.append("REFORMULATE")
        query = _reformulated_query(self.diagnosis)
        records = self._retrieve(query, FINAL_K)
        return self._rebuild(contract.clinical_question, records)

    def independent_source(self, contract: EvidenceContract) -> EvidenceContract:
        self.applied.append("INDEPENDENT_SOURCE_UNAVAILABLE")
        return contract


def _evidence_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for rank, record in enumerate(records, start=1):
        safe = _projected_record(record)
        safe["rank"] = rank
        rows.append(safe)
    return rows


def _graph_evidence_rows(
    graph: EvidenceLineageGraph, retrieval: GraphRetrieval,
) -> List[Dict[str, Any]]:
    rows = []
    paths = retrieval.paths_by_evidence
    for rank, evidence_id in enumerate(retrieval.returned_evidence_ids, start=1):
        node = graph.nodes[evidence_id]
        attributes = dict(node.attributes)
        rows.append({
            "rank": rank,
            "document_id": attributes["document_id"],
            "source_id": attributes["source_id"],
            "citation_id": attributes["citation_id"],
            "source_version": attributes["source_version"],
            "authority": attributes["authority"],
            "provenance_uri": attributes["provenance_uri"],
            "content_sha256": attributes["content_sha256"],
            "record_sha256": attributes["record_sha256"],
            "graph_path": paths[evidence_id].to_dict(),
        })
    return rows


def _score_state(*, contract_applicable: bool) -> Dict[str, Any]:
    fields = (
        "diagnosis_correct", "actionable_correct", "answer_relevancy",
        "faithfulness", "citation_precision", "citation_recall",
    )
    return {
        "status": "not_executed_validate_only",
        "values": {field: None for field in fields},
        "missing_score_fields": list(fields),
        "not_applicable_score_fields": (
            [] if contract_applicable else ["evidence_contract_completion"]
        ),
    }


def _empty_call_accounting(*, reserved_logical_calls: int) -> Dict[str, Any]:
    return {
        "actual_attempts": 0,
        "logical_calls": 0,
        "reserved_logical_calls": reserved_logical_calls,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_seconds": 0.0,
        "cache_hit": False,
        "generation_error": None,
        "answered": False,
        "abstained": False,
        "status": "not_executed_validate_only",
    }


def run_validate_only(output_path: str | Path = OUTPUT) -> Dict[str, Any]:
    """Run the full matched validate-only integration and write one report."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np

    assert_disjoint_from_medguard_failures()
    component_deltas = single_component_deltas()

    cases, input_summary = load_matched_development_inputs()
    views_by_cluster = _load_ordered_views()
    feature_encoder = FrozenDermLIPFeatureEncoder()
    runtime = load_frozen_evidence_runtime(image_encoder=feature_encoder)
    dense_backend = runtime.text_retriever.dense
    dense = _QueryCache(dense_backend)
    dense_identity = {
        "kind": "receipt_verified_frozen_dense_index",
        "embedding_model": dense_backend.model,
        "index_sha256": _file_sha256(EVIDENCE_BUNDLE / "text_dense.faiss"),
        "corpus_sha256": _file_sha256(EVIDENCE_BUNDLE / "corpus.jsonl"),
        "receipt_sha256": _file_sha256(EVIDENCE_RECEIPT),
        "bundle_verified": True,
    }

    # ---- frozen evidence-lineage graph over the shared text corpus ----
    corpus_records = [dict(record) for record in dense_backend._meta]
    corpus_vectors = np.asarray(
        dense_backend._index.reconstruct_n(0, dense_backend._index.ntotal),
        dtype="float32",
    )
    if corpus_vectors.shape[0] != len(corpus_records):
        raise ValueError("frozen dense index and corpus metadata disagree")
    concept_vectors = dense_backend.embed(
        [concept_content(code, term) for code, term in QUERY_TERMS.items()]
    )
    concept_links = link_concepts_by_embedding(
        records=corpus_records, concept_terms=QUERY_TERMS,
        concept_vectors=concept_vectors, evidence_vectors=corpus_vectors,
        delta_r=DELTA_R,
    )
    graph = build_evidence_lineage_graph(
        records=corpus_records, concept_terms=QUERY_TERMS, concept_links=concept_links,
    )
    graph_sha256 = graph.graph_sha256
    corpus_index = {
        f"evidence:{record['document_id']}": index
        for index, record in enumerate(corpus_records)
    }

    base = prepare_matched_b0_b1_validate_only(
        dense_retriever=dense, dense_runtime_identity=dense_identity, strict=True,
    )
    base_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in base["cases"]:
        base_rows.setdefault(row["input"]["case_id"], {})[row["arm_id"]] = row

    protocols = _protocols(str(input_summary["perception_artifact_sha256"]), graph_sha256)
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

    perception_bindings = load_development_perception_bindings()
    if set(perception_bindings) != {case.case_id for case in cases}:
        raise ValueError("perception bindings and matched inputs describe different cases")

    text_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _admitted_text(query: str, k: int) -> List[Dict[str, Any]]:
        """Policy-admitted dense+BM25 fusion over the frozen text index."""
        key = f"{query}::{k}"
        if key not in text_cache:
            retrieved, _scores = runtime.text_retriever.search(query, k)
            admitted = []
            for record in retrieved:
                decision = runtime.source_policy.decide(record, "diagnosis")
                if decision.admitted:
                    admitted.append(dict(record))
            text_cache[key] = admitted
        return [dict(record) for record in text_cache[key]]

    query_vector_cache: Dict[str, Any] = {}

    def _query_vector(query: str):
        if query not in query_vector_cache:
            query_vector_cache[query] = np.asarray(
                dense_backend.embed([query])[0], dtype="float32",
            )
        return query_vector_cache[query]

    context_vector_cache: Dict[str, Any] = {}

    def _context_vector(context: str):
        """Embed one rendered subgraph context with the frozen text model."""
        key = hashlib.sha256(context.encode("utf-8")).hexdigest()
        if key not in context_vector_cache:
            vector = np.asarray(dense_backend.embed([context])[0], dtype="float32")
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ValueError("frozen text model returned a degenerate context vector")
            context_vector_cache[key] = vector / norm
        return context_vector_cache[key]

    similarity_cache: Dict[str, Dict[str, float]] = {}

    def _graph_similarity(query: str) -> Dict[str, float]:
        if query not in similarity_cache:
            scores = corpus_vectors @ _query_vector(query)
            similarity_cache[query] = {
                evidence_id: float(scores[index])
                for evidence_id, index in corpus_index.items()
            }
        return similarity_cache[query]

    rows: List[Dict[str, Any]] = []
    modality_counts: Counter[str] = Counter()
    b4_modality_counts: Counter[str] = Counter()
    action_counts: Dict[str, Counter[str]] = {
        arm.value: Counter() for arm in MEDGUARD_ARMS
    }
    repair_counts: Dict[str, Counter[str]] = {
        arm.value: Counter() for arm in MEDGUARD_ARMS
    }
    failure_counts: Dict[str, Counter[str]] = {
        arm.value: Counter() for arm in MEDGUARD_ARMS
    }
    b6_decisions: Counter[str] = Counter()
    b6_attempts_hist: Counter[str] = Counter()
    b6_refinements_hist: Counter[str] = Counter()
    b6_rules: Counter[str] = Counter()
    b5_seed_reached: List[int] = []
    graph_lineage_complete = 0
    graph_lineage_expected = 0
    a0_alias_mismatches = 0

    for ordinal, (case, lineage) in enumerate(zip(cases, lineages), start=1):
        cluster_id = case.cluster_id
        ordered_rows = views_by_cluster.get(cluster_id)
        if not ordered_rows:
            raise ValueError(f"case has no ordered image views: {case.case_id}")
        if tuple(row["image_sha256"] for row in ordered_rows) != tuple(
            case.ordered_view_sha256s
        ):
            raise ValueError(f"ordered view lineage mismatch: {case.case_id}")
        diagnosis = case.committed_diagnosis
        query = _query(diagnosis)
        perception = perception_bindings[case.case_id]
        if perception.commitment.concept_code != diagnosis:
            raise ValueError(f"perception commitment drift: {case.case_id}")

        arm_rows: Dict[str, Dict[str, Any]] = {}
        evidence_by_arm: Dict[ArmId, Sequence[Mapping[str, Any]]] = {}

        # ---- B0 / B1 (reused byte-for-byte from the matched B0/B1 builder) ----
        existing = base_rows[case.case_id]
        arm_rows["B0"] = {
            "retrieval": existing["B0"]["retrieval"],
            "score_state": existing["B0"]["score_state"],
            "call_accounting": _empty_call_accounting(reserved_logical_calls=1),
        }
        arm_rows["B1"] = {
            "retrieval": existing["B1"]["retrieval"],
            "score_state": existing["B1"]["score_state"],
            "call_accounting": _empty_call_accounting(reserved_logical_calls=1),
        }
        evidence_by_arm[ArmId.B0] = ()
        evidence_by_arm[ArmId.B1] = existing["B1"]["retrieval"]["evidence"]

        # ---- B4 (DynaRAG GMM-ATKS component comparator) ----
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
                candidates.append(ImageExemplarCandidate(
                    exemplar_id=_safe_record(record)["document_id"],
                    cosine_similarity=score,
                    payload=_safe_record(record),
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
            b4_modality_counts[item.modality.value] += 1
        arm_rows["B4"] = {
            "retrieval": {
                "strategy": "GMM_ATKS_component_proxy",
                "candidate_k_per_modality": DENSE_CANDIDATE_K,
                "final_k": FINAL_K,
                "evidence": b4_evidence,
                "audit": {
                    "selector_components": b4.audit.selector_components,
                    "selector_fallback_reason": b4.audit.selector_fallback_reason,
                    "selected_image_count": b4.audit.selected_image_count,
                    "merge_rule": b4.audit.merge_rule,
                },
            },
            "score_state": _score_state(contract_applicable=False),
            "call_accounting": _empty_call_accounting(reserved_logical_calls=1),
        }
        evidence_by_arm[ArmId.B4] = b4_evidence

        # ---- B5 (MedGraphRAG-inspired bounded graph traversal) ----
        similarity = _graph_similarity(query)
        seed_concept = f"concept:{diagnosis}"
        b5_retrieval = graph.traverse(
            seed_concept_ids=[seed_concept], similarity=similarity,
            depth=1, node_limit=TEXT_CANDIDATE_K, final_k=FINAL_K,
        )
        if len(b5_retrieval.returned_evidence_ids) != FINAL_K:
            raise ValueError(f"B5 graph traversal did not fill the top-five: {case.case_id}")
        b5_evidence = _graph_evidence_rows(graph, b5_retrieval)
        b5_seed_reached.append(b5_retrieval.reached_evidence_count)
        arm_rows["B5"] = {
            "retrieval": {
                "strategy": "bounded_evidence_lineage_graph_traversal",
                "candidate_k": TEXT_CANDIDATE_K,
                "final_k": FINAL_K,
                "evidence": b5_evidence,
                "audit": {
                    "seed_concept_ids": list(b5_retrieval.seed_concept_ids),
                    "depth": b5_retrieval.depth,
                    "node_limit": b5_retrieval.node_limit,
                    "reached_evidence_count": b5_retrieval.reached_evidence_count,
                    "frontier_sizes": list(b5_retrieval.frontier_sizes),
                    "candidate_count": len(b5_retrieval.candidate_evidence_ids),
                    "subgraph_node_count": b5_retrieval.subgraph_node_count,
                    "subgraph_edge_count": len(b5_retrieval.subgraph_edge_ids),
                    "graph_sha256": graph_sha256,
                },
            },
            "score_state": _score_state(contract_applicable=False),
            "call_accounting": _empty_call_accounting(reserved_logical_calls=1),
        }
        evidence_by_arm[ArmId.B5] = [
            {"document_id": row["document_id"], "content_sha256": row["content_sha256"]}
            for row in b5_evidence
        ]

        # ---- B6 (bounded retrieve-evaluate-refine over the same graph) ----
        query_vector = _query_vector(query)

        def _b6_retrieve(strategy: SearchStrategy) -> GraphRetrieval:
            return graph.traverse(
                seed_concept_ids=[seed_concept], similarity=similarity,
                depth=strategy.depth, node_limit=strategy.node_limit,
                final_k=min(FINAL_K, strategy.node_limit),
            )

        def _b6_relevance(retrieval: GraphRetrieval) -> float:
            # Paper 104: cosine between the query and the aggregated context of
            # the retrieved subgraph, where the context integrator renders the
            # subgraph's nodes, relations, and paths as structured text.
            context = aggregate_context_text(graph, retrieval)
            return float(_context_vector(context) @ query_vector)

        b6 = run_bounded_refinement(
            retrieve=_b6_retrieve, semantic_relevance=_b6_relevance,
            bounds=RefinementBounds(), candidate_ceiling=TEXT_CANDIDATE_K,
            attribute_ceiling=8,
        )
        if b6.attempts_used > MAX_ATTEMPTS or b6.refinements_used > MAX_REFINEMENTS:
            raise ValueError(f"B6 exceeded its bounded loop budget: {case.case_id}")
        b6_evidence = _graph_evidence_rows(graph, b6.retrieval)
        b6_decisions[b6.final_decision.value] += 1
        b6_attempts_hist[str(b6.attempts_used)] += 1
        b6_refinements_hist[str(b6.refinements_used)] += 1
        for attempt in b6.attempts:
            if attempt.optimizer_rule is not None:
                b6_rules[attempt.optimizer_rule.value] += 1
        arm_rows["B6"] = {
            "retrieval": {
                "strategy": "bounded_retrieve_evaluate_refine_over_b5_graph",
                "candidate_k": TEXT_CANDIDATE_K,
                "final_k": FINAL_K,
                "evidence": b6_evidence,
                "audit": {
                    "attempts": [attempt.to_dict() for attempt in b6.attempts],
                    "attempts_used": b6.attempts_used,
                    "refinements_used": b6.refinements_used,
                    "maximum_attempts": MAX_ATTEMPTS,
                    "maximum_refinements": MAX_REFINEMENTS,
                    "final_decision": b6.final_decision.value,
                    "graph_sha256": graph_sha256,
                },
            },
            "score_state": _score_state(contract_applicable=False),
            "call_accounting": {
                **_empty_call_accounting(reserved_logical_calls=MAX_ATTEMPTS),
                "abstained": b6.abstained,
            },
        }
        evidence_by_arm[ArmId.B6] = [
            {"document_id": row["document_id"], "content_sha256": row["content_sha256"]}
            for row in b6_evidence
        ]

        # ---- M and the paired ablations ----
        image_records_cache: Dict[str, List[Dict[str, Any]]] = {}

        def _medguard_retrieve(
            config: MedGuardArmConfig,
        ) -> Callable[[str, int], List[Dict[str, Any]]]:
            def retrieve(retrieval_query: str, k: int) -> List[Dict[str, Any]]:
                rankings = [_admitted_text(retrieval_query, TEXT_CANDIDATE_K)]
                if config.image_exemplar_retrieval:
                    if "records" not in image_records_cache:
                        image_records_cache["records"] = (
                            runtime.image_retriever.search_records_by_images(
                                [row["image_path"] for row in ordered_rows],
                                IMAGE_CANDIDATE_K,
                            )
                        )
                        if any(
                            int(record.get("query_view_count", -1)) != len(ordered_rows)
                            for record in image_records_cache["records"]
                        ):
                            raise ValueError(
                                f"all-view image retrieval failed for {cluster_id}"
                            )
                    rankings.append(
                        [dict(record) for record in image_records_cache["records"]]
                    )
                # One fusion path for every configuration: A1 simply supplies one
                # ranking instead of two.  It receives no compensating retrieval.
                fused = reciprocal_rank_fusion_records(rankings)[:k]
                if len(fused) != k:
                    raise ValueError(
                        f"cross-modal retrieval returned fewer than {k}: {cluster_id}"
                    )
                return fused
            return retrieve

        medguard_payloads: Dict[ArmId, Dict[str, Any]] = {}
        for arm in (ArmId.M, ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4):
            config = MEDGUARD_CONFIGS[arm]
            retrieve = _medguard_retrieve(config)
            records = retrieve(query, FINAL_K)
            contract, relations = _build_contract(
                case_id=case.case_id, perception=perception,
                records=records, question=QUESTION,
            )
            observed = verify_contract(contract)
            tools = _OfflineRemediationTools(
                case_id=case.case_id, diagnosis=diagnosis, perception=perception,
                retrieve=retrieve,
            )
            if not config.verification_gate:
                # A4: verification executes and is recorded, but cannot control
                # repair or abstention.  Malformed structured output still fails
                # and can never be coerced to PASS.
                malformed = FailureType.INVALID_CONTRACT in observed.failures
                final_action = (
                    ACTION_STRUCTURED_OUTPUT_ERROR if malformed else ACTION_ANSWER_UNGATED
                )
                trajectory: List[Dict[str, Any]] = []
                final_contract = contract
            else:
                budget = (
                    ActionBudget()
                    if config.bounded_remediation
                    else ActionBudget(max_total_repairs=0, per_action={})
                )
                result = run_remediation(
                    contract, DeterministicRemediationPolicy(budget), tools,
                    max_steps=MAX_REMEDIATION_STEPS,
                )
                final_contract = result.contract
                final_action = result.final_action.value
                trajectory = [{
                    "step": step.step,
                    "failures": [failure.value for failure in step.failures],
                    "action": step.action.value,
                } for step in result.trajectory]
            final_report = verify_contract(final_contract)
            score = score_contract(final_contract)
            for failure in observed.failures:
                failure_counts[arm.value][failure.value] += 1
            action_counts[arm.value][final_action] += 1
            for applied in tools.applied:
                repair_counts[arm.value][applied] += 1
            if not tools.applied:
                repair_counts[arm.value]["none"] += 1

            final_records = list(final_contract.evidence)
            payload: Dict[str, Any] = {
                "retrieval": {
                    "strategy": (
                        "receipt_verified_dense_bm25_and_all_view_exemplar_rrf"
                        if config.image_exemplar_retrieval
                        else "receipt_verified_dense_bm25_text_only_rrf"
                    ),
                    # This is the first-attempt retrieval, which is what the
                    # request key binds.  A repair re-retrieves, so the final
                    # contract's evidence is recorded separately below rather
                    # than silently overwriting this block.
                    "stage": "initial_attempt",
                    "candidate_k": TEXT_CANDIDATE_K,
                    "final_k": FINAL_K,
                    "evidence": _evidence_rows(records),
                },
                "components": config.components(),
                "contract": {
                    "contract_sha256": contract_sha256(final_contract),
                    "stage": "post_remediation_final",
                    "evidence_document_ids": [
                        item.document_id for item in final_records
                    ],
                    "evidence_count": len(final_records),
                    "claim_count": len(final_contract.claims),
                    "verification_state": final_contract.verification_state.value,
                    "structural_completion": score.structural_completion,
                    "contract_complete": score.contract_complete,
                    "structured_pass_eligible": score.structured_pass_eligible,
                    "independent_evidence_relation_rate": (
                        score.independent_evidence_relation_rate
                    ),
                    "image_support_rate": score.image_support_rate,
                },
                "verification": {
                    "executed": True,
                    "gate_active": config.verification_gate,
                    "ok": observed.ok,
                    "failures": [failure.value for failure in observed.failures],
                    "final_ok": final_report.ok,
                    "final_failures": [failure.value for failure in final_report.failures],
                },
                "remediation": {
                    "enabled": config.bounded_remediation,
                    "trajectory": trajectory,
                    "applied_actions": list(tools.applied),
                    "repairs_used": len([
                        item for item in tools.applied
                        if item in {"EXPAND", "REFORMULATE"}
                    ]),
                },
                "final_action": final_action,
                "score_state": _score_state(contract_applicable=True),
                "call_accounting": {
                    **_empty_call_accounting(reserved_logical_calls=MAX_ATTEMPTS),
                    "abstained": final_action == Action.ABSTAIN.value,
                    "answered": final_action in {
                        Action.PASS.value, ACTION_ANSWER_UNGATED,
                    },
                },
            }

            if config.graph_lineage:
                graph_rows = _graph_records(records)
                case_graph = build_evidence_lineage_graph(
                    records=graph_rows, concept_terms=QUERY_TERMS,
                    concept_links={
                        row["document_id"]: concept_links.get(row["document_id"], {})
                        for row in graph_rows
                    },
                )
                paths = [
                    case_graph.lineage_path(f"evidence:{row['document_id']}", seed_concept)
                    for row in graph_rows
                ]
                for path, evidence_row in zip(paths, payload["retrieval"]["evidence"]):
                    evidence_row["graph_path"] = path.to_dict()
                    graph_lineage_expected += 1
                    if (
                        path.source_id and path.source_version
                        and len(path.content_sha256) == 64
                        and len(path.record_sha256) == 64
                        and path.hops >= 2
                    ):
                        graph_lineage_complete += 1
                payload["graph_lineage"] = {
                    "enabled": True,
                    "graph_sha256": case_graph.graph_sha256,
                    "frozen_corpus_graph_sha256": graph_sha256,
                    "counts": case_graph.counts(),
                    "paths_bound": len(paths),
                }
            else:
                # A2 keeps every flat provenance field and every citation check;
                # only the graph object, its traversal, and its lineage edges go.
                payload["graph_lineage"] = {
                    "enabled": False,
                    "flat_provenance_retained": True,
                    "flat_fields": [
                        "source_id", "source_version", "authority",
                        "retrieval_path", "content_sha256", "record_sha256",
                    ],
                }
            medguard_payloads[arm] = payload
            evidence_by_arm[arm] = [
                {
                    "document_id": row["document_id"],
                    "content_sha256": row["content_sha256"],
                }
                for row in payload["retrieval"]["evidence"]
            ]
            for row in payload["retrieval"]["evidence"]:
                if arm is ArmId.M:
                    modality_counts[row.get("retrieval_channel") or "unknown"] += 1

        # A0's execution payload is M byte-for-byte. Copy through canonical JSON
        # so the assertion tests serialized bytes, not object identity. The
        # report-only alias annotation is added after request identity checks.
        m_bytes = json.dumps(
            medguard_payloads[ArmId.M], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        )
        medguard_payloads[ArmId.A0] = json.loads(m_bytes)
        if json.dumps(
            medguard_payloads[ArmId.A0], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ) != m_bytes:
            a0_alias_mismatches += 1
        evidence_by_arm[ArmId.A0] = evidence_by_arm[ArmId.M]
        for arm in MEDGUARD_ARMS:
            arm_rows[arm.value] = medguard_payloads[arm]

        # ---- protocol identities ----
        request_keys = {
            arm: build_request_key(
                protocols[arm], lineage, question=QUESTION,
                retrieved_evidence=tuple(EvidenceIdentity(
                    evidence_id=str(record["document_id"]),
                    content_sha256=str(record["content_sha256"]),
                ) for record in evidence_by_arm[arm]),
            ) for arm in ARMS
        }
        if request_keys[ArmId.A0] != request_keys[ArmId.M]:
            a0_alias_mismatches += 1
        # Every arm except the declared M/A0 alias must be separately keyed.
        distinct_expected = len(ARMS) - 1
        if len(set(request_keys.values())) != distinct_expected:
            raise ValueError(f"cross-arm request-key collision: {case.case_id}")

        for arm in ARMS:
            arm_rows[arm.value]["cache_namespace"] = protocols[arm].cache_namespace
            arm_rows[arm.value]["request_key"] = request_keys[arm]
        arm_rows[ArmId.A0.value]["alias_of"] = ArmId.M.value

        rows.append({
            "case_id": case.case_id,
            "input": case.public_input(),
            "arms": {arm.value: arm_rows[arm.value] for arm in ARMS},
            "usage": asdict(usage),
        })
        if ordinal % 40 == 0:
            print(f"matched full-arm validate {ordinal}/{len(cases)}")

    # A0 is not a separate execution: its rows are M's rows.  Mirror M's
    # counters rather than leave the alias half-populated, which would contradict
    # the per-case payloads it copies.
    for counters in (action_counts, repair_counts, failure_counts):
        counters[ArmId.A0.value] = Counter(counters[ArmId.M.value])

    # ---- report ----
    missing = {
        arm.value: asdict(summarize_scores({case.case_id: None for case in cases}))
        for arm in ARMS
    }
    namespaces = {arm.value: protocols[arm].cache_namespace for arm in ARMS}
    b5_concept_heads = sum(
        1
        for row in rows
        for evidence in row["arms"]["B5"]["retrieval"]["evidence"]
        if evidence["graph_path"]["hops"] >= 3
        and any(
            item.startswith(f"{EdgeType.THE_DEFINITION_OF.value}:")
            for item in evidence["graph_path"]["edge_ids"]
        )
    )
    b5_returned = sum(len(row["arms"]["B5"]["retrieval"]["evidence"]) for row in rows)
    graph_counts = graph.counts()

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
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
            "distinct_text_queries": len(text_cache),
            "query_feature_model": feature_encoder.model_id,
            "query_feature_sha256": feature_encoder.feature_sha256,
            "graph": {
                "schema": GRAPH_SCHEMA_VERSION,
                "graph_sha256": graph_sha256,
                "delta_r": DELTA_R,
                "concept_type": CONCEPT_TYPE,
                "concept_content_format": "paper133_Ce_name_type_context",
                "counts": graph_counts,
                "corpus_records": len(corpus_records),
                "records_linked_to_at_least_one_concept": sum(
                    1 for links in concept_links.values() if links
                ),
                "records_linked_to_two_or_more_concepts": sum(
                    1 for links in concept_links.values() if len(links) >= 2
                ),
            },
        },
        "protocol": {
            "arms": [arm.value for arm in ARMS],
            "cache_namespaces": namespaces,
            "arm_configs": {
                arm.value: _plain(protocols[arm].arm_config) for arm in ARMS
            },
            "arm_alias": {"A0": "M"},
            "code_snapshot_sha256": protocols[ArmId.B0].code_snapshot,
            "missing_scores": missing,
        },
        "aggregate": {
            "cases": len(cases),
            "views": sum(len(case.ordered_view_sha256s) for case in cases),
            "arms": len(ARMS),
            "arm_rows": len(cases) * len(ARMS),
            "b1_final_dense_evidence": sum(
                len(row["arms"]["B1"]["retrieval"]["evidence"]) for row in rows
            ),
            "b4_final_evidence": sum(
                len(row["arms"]["B4"]["retrieval"]["evidence"]) for row in rows
            ),
            "b4_modality_counts": dict(sorted(b4_modality_counts.items())),
            "b5_final_evidence": b5_returned,
            "b5_mean_reached_evidence": (
                sum(b5_seed_reached) / len(b5_seed_reached) if b5_seed_reached else None
            ),
            "b6_final_decisions": dict(sorted(b6_decisions.items())),
            "b6_attempts_used": dict(sorted(b6_attempts_hist.items())),
            "b6_refinements_used": dict(sorted(b6_refinements_hist.items())),
            "b6_optimizer_rules": dict(sorted(b6_rules.items())),
            "m_final_evidence_channel_counts": dict(sorted(modality_counts.items())),
            "medguard_final_actions": {
                arm: dict(sorted(counter.items())) for arm, counter in action_counts.items()
            },
            "medguard_applied_actions": {
                arm: dict(sorted(counter.items())) for arm, counter in repair_counts.items()
            },
            "medguard_first_pass_failures": {
                arm: dict(sorted(counter.items())) for arm, counter in failure_counts.items()
            },
            "generation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "boundary_violation_count": 0,
        },
        "verification": {
            "identical_ordered_images_and_frozen_perception": True,
            "truth_fields_in_arm_visible_inputs": 0,
            "distinct_cache_namespaces": len(set(namespaces.values())),
            "expected_distinct_cache_namespaces": len(ARMS) - 1,
            "distinct_request_keys_per_case": len(ARMS) - 1,
            "graph_lineage_paths_expected": graph_lineage_expected,
            "graph_lineage_paths_complete": graph_lineage_complete,
            "missing_score_accounting": "explicit_none_never_zero",
            "call_ceiling_per_arm": {
                arm.value: protocols[arm].generation_call_ceiling for arm in ARMS
            },
            "token_ceiling_per_arm": {
                arm.value: protocols[arm].token_ceiling for arm in ARMS
            },
            "zero_generation": True,
            "b5_is_not_flat_provenance": {
                "node_types_present": sorted({
                    node_type.value for node_type in NodeType
                    if graph_counts[f"nodes_{node_type.value}"] > 0
                }),
                "concept_edges": graph_counts[f"edges_{EdgeType.THE_DEFINITION_OF.value}"],
                "returned_paths_with_concept_head": b5_concept_heads,
                "returned_paths_total": b5_returned,
                "graph_constrains_candidate_pool": True,
                "corpus_records": len(corpus_records),
            },
            "b6_loop_bounded": {
                "maximum_attempts": MAX_ATTEMPTS,
                "maximum_refinements": MAX_REFINEMENTS,
                "observed_maximum_attempts": max(
                    int(key) for key in b6_attempts_hist
                ) if b6_attempts_hist else 0,
                "observed_maximum_refinements": max(
                    int(key) for key in b6_refinements_hist
                ) if b6_refinements_hist else 0,
                "reuses_medguard_failure_labels": False,
                "refinement_branch_exercised_by_development_data": bool(b6_rules),
                "note": (
                    "Boundedness here is partly vacuous: under the pre-registered "
                    "sufficiency threshold every development case is declared "
                    "sufficient on the first attempt, so the refinement and "
                    "abstention branches are exercised by focused tests rather "
                    "than by this data. The threshold must not be retuned now "
                    "that these outputs have been seen."
                ),
            },
            "a0_execution_payload_equivalent_to_m": a0_alias_mismatches == 0,
            "a0_final_report_row_differs_only_by_alias_annotation": all(
                _a0_row_differs_only_by_alias(
                    row["arms"][ArmId.M.value], row["arms"][ArmId.A0.value],
                )
                for row in rows
            ),
            "ablations_change_exactly_one_component": component_deltas,
        },
        "fidelity": {
            "B5": {
                "classification": "compatible_adaptation",
                "faithful_reproduction": False,
                "reporting_label": "MedGraphRAG-inspired evidence graph comparator",
            },
            "B6": {
                "classification": "compatible_adaptation",
                "faithful_reproduction": False,
                "reporting_label": (
                    "Agentic-Graph-RAG-inspired bounded refinement comparator"
                ),
            },
        },
        "interpretation": {
            "verified": (
                "Every arm receives identical frozen development input lineage under "
                "separate cache/request identities; B5 traverses a real four-tier "
                "evidence graph, B6's loop stays inside its declared bound, A0's "
                "execution payload and request identity match M before its report-only "
                "alias annotation, and A1-A4 each change exactly one component."
            ),
            "not_claimed": (
                "No generation or outcome scoring occurred. This report contains no "
                "accuracy, RAGAS, safety improvement, or rival-beating result, and "
                "graph membership, provenance, authority, and retrieval similarity "
                "establish no clinical truth."
            ),
            "boundary_conditions": [
                "Every MedGuard arm fails closed on misaligned_evidence because no "
                "curated claim-evidence relation gold exists yet. The arm-to-arm "
                "differences recorded here are wiring behaviour, not evidence "
                "quality, and no ablation effect may be read from them.",
                "M and A2 are identical in decision by construction: the graph is "
                "lineage over already-fused evidence and does not select it, so in "
                "this validate-only run the graph ablation changes only the recorded "
                "lineage structure. Any decision effect would appear only once the "
                "graph-path prompt adapter reaches a generator.",
                "A1 fills the shared five-item ceiling with text because its modality "
                "was removed under a fixed ceiling. Its candidate pool stays at the "
                "shared candidate_k, so it receives no extra retrieval, call, or "
                "token budget.",
            ],
        },
        "cases": rows,
    }
    reject_truth_keys(report, path="report")

    aggregate = report["aggregate"]
    checks = report["verification"]
    if (
        aggregate["cases"] != 160
        or aggregate["views"] != 244
        or aggregate["arm_rows"] != 160 * len(ARMS)
        or aggregate["b5_final_evidence"] != 160 * FINAL_K
        or checks["distinct_cache_namespaces"] != len(ARMS) - 1
        or checks["graph_lineage_paths_complete"] != checks["graph_lineage_paths_expected"]
        or checks["graph_lineage_paths_expected"] == 0
        or not checks["a0_execution_payload_equivalent_to_m"]
        or not checks["a0_final_report_row_differs_only_by_alias_annotation"]
        or checks["b6_loop_bounded"]["observed_maximum_attempts"] > MAX_ATTEMPTS
        or checks["b6_loop_bounded"]["observed_maximum_refinements"] > MAX_REFINEMENTS
        or checks["b5_is_not_flat_provenance"]["returned_paths_with_concept_head"]
        != checks["b5_is_not_flat_provenance"]["returned_paths_total"]
        or len(checks["b5_is_not_flat_provenance"]["node_types_present"]) != 4
    ):
        raise ValueError("matched full-arm report failed its final invariant audit")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ) + "\n").encode("utf-8"))
    return report


def main() -> int:
    report = run_validate_only()
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    print(json.dumps(report["verification"], indent=2, sort_keys=True, default=str))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
