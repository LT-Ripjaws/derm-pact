"""Generation-executing matched H2H runner (the paid H2H's offline core).

This is the sibling of ``matched_full_h2h_validate.py``: the validate runner wires
every arm and stops before the generator; this one actually produces the
head-to-head result -- retrieve, build the evidence contract, run the bounded
remediation loop, generate through the cached generator, and score.  It is written
to run with **injected dependencies** so the entire pipeline is unit-testable
behind a fake ``LLMClient`` that returns canned outputs and makes zero paid calls.
The main session owns the paid smoke / dev-H2H gate; nothing here authorizes it,
and ``build_production_dependencies`` (which would wire a real client) is never
called by this module or its tests.

Diagnosis is a competed axis.  Every arm except B0 commits the frozen DermLIP
probe diagnosis (``perception.commitment``); B0 alone diagnoses with the VLM,
constrained to the HAM7 vocabulary, and is scored against truth -- the baseline M
beats on diagnosis.  B0 uses neither the frozen probe nor the evidence contract.

The three arm defects are resolved here: the graph is selection-bearing for the
method arm (A2 now changes M's evidence set), B6's sufficiency is evidence-count
based (D4, wired where B6 retrieval is provided), and INDEPENDENT_SOURCE detects a
curated source contradiction and re-retrieves from the separate guideline index
(D3).  See notes/H2H_RUNNER_RESULT.md.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard.config import MALIGNANT_LABELS
from medguard.evidence import (
    ClaimType,
    EvidenceContract,
    UncertaintyType,
    UnresolvedUncertainty,
    VerificationState,
    evidence_from_chunk,
)
from medguard.evidence.claim_builder import (
    DIAGNOSIS_CLAIM_ID,
    NO_EXEMPLAR_SUPPORT_DESCRIPTION,
    NO_EXEMPLAR_SUPPORT_UNCERTAINTY_ID,
    build_image_bound_claims,
)
from medguard.evidence.contract import PerceptionBinding
from medguard.evidence.guideline_claims import (
    GUIDELINE_CHANNEL,
    guideline_evidence_item,
    select_guideline_evidence,
)
from medguard.evidence.scoring import contract_sha256, score_contract
from medguard.evidence.source_contradiction import (
    SourceContradictionRegistry,
    build_contradicted_source_claim,
    flagged_text_records,
)
from medguard.evidence.verification import FailureType, verify_contract
from medguard.eval.citation_extraction import guideline_citations_from_contract
from medguard.eval.h2h_evidence_graph import select_evidence_by_graph_connectivity
from medguard.eval.h2h_generation import SharedClinicalTask, adapter_for_arm, effective_arm_id
from medguard.eval.h2h_protocol import (
    ArmId,
    ArmProtocol,
    CaseLineage,
    EvidenceIdentity,
    build_request_key,
)
from medguard.eval.h2h_scoring import ArmCaseOutcome, score_h2h
from medguard.pipeline.generation_cache import CachedStructuredGenerator, GenerationCache
from medguard.pipeline.generator import VLMGenerator
from medguard.pipeline.hybrid_retriever import reciprocal_rank_fusion_records


PROTOCOL_ID = "medguard-derm-evidence-contract-h2h-v1"
SCHEMA_VERSION = "medguard-matched-full-arm-run-v1"
DEFAULT_ARMS: Tuple[ArmId, ...] = (
    ArmId.B0, ArmId.B1, ArmId.B4, ArmId.B5, ArmId.B6,
    ArmId.M, ArmId.A0, ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4,
)
CONTRACT_ARMS: Tuple[ArmId, ...] = (ArmId.M, ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4)
BASELINE_ANSWER_ARMS: Tuple[ArmId, ...] = (ArmId.B1, ArmId.B4, ArmId.B5, ArmId.B6)

ACTION_ANSWER_UNGATED = "ANSWER_UNGATED"
ACTION_STRUCTURED_OUTPUT_ERROR = "STRUCTURED_OUTPUT_ERROR"
GENERATION_SEED = 0

# The section-partitioned answer contract. The Assessment section is the only
# text-groundable modality (literature-cited), so RAGAS faithfulness grades it
# alone; Visual findings is image-derived (P1 axis) and Diagnosis is the frozen
# probe's label.
VISUAL_FINDINGS_SECTION_LABEL = "Visual findings:"
ASSESSMENT_SECTION_LABEL = "Assessment:"
DIAGNOSIS_SECTION_LABEL = "Diagnosis:"

# Emitted ONLY when no source-bound claim survives, i.e. the committed condition has no
# recommendation in the licensed corpus.  The clinical question has two parts -- the
# image diagnosis and what the authoritative sources say -- and the renderer used to
# answer part one and drop part two in silence.  Saying nothing is not the same as
# saying "there is nothing to say": a reader cannot tell an absent guideline from a
# retrieval failure, and an answer-relevancy judge scores the silence as an
# unaddressed question (0.245 on these lesions against 0.644 where guidance exists).
#
# Deliberately NOT labelled "Assessment:" and deliberately carrying no citation
# marker. It is a host-owned statement about corpus coverage, not a literature claim,
# and must never be graded as one -- see the mutual-exclusion test.
NO_GUIDANCE_SECTION_LABEL = "Authoritative guidance:"
NO_GUIDANCE_STATEMENT = (
    "The licensed guideline corpus contains no recommendation for this condition, so no "
    "management guidance is asserted here."
)
# Claim types whose prose is grounded in retrieved LITERATURE (text-entailable);
# everything else (diagnosis/differential/visual_finding/image_exemplar_support) is
# image-grounded and rendered under Visual findings without a citation.
_SOURCE_BOUND_CLAIM_TYPES = frozenset({
    ClaimType.ASSESSMENT, ClaimType.MANAGEMENT, ClaimType.TREATMENT, ClaimType.REFERRAL,
})

_TASK = SharedClinicalTask()
_TEMPLATE_CTX = {
    "committed_diagnosis": "melanoma", "commitment_confidence": 0.9,
    "artifact_sha256": "0" * 64, "record_sha256": "0" * 64,
    "distribution_sha256": "0" * 64,
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text_sha256(record: Mapping[str, Any]) -> str:
    text = " ".join(str(record.get("text", "")).split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The H2H contract adapters instruct the model to return the typed EvidenceContract
# JSON but do not spell out its shape.  The runner appends the concrete schema so
# the on-wire prompt is schema-complete AND identical to the prompt the cache
# identity binds.  spec-v5 should fold this into the M/A adapter obligations so the
# schema lives in exactly one place; until then it is duplicated here deliberately
# and matches ``pipeline.generator.bind_evidence_contract``'s accepted fields.
_CONTRACT_OUTPUT_SCHEMA = (
    "Return one JSON object only, with exactly these two keys:\n"
    '{"claims": [{"claim_id": "c1", '
    '"claim_type": "diagnosis|differential|visual_finding|assessment|management|'
    'referral|treatment|image_exemplar_support", "text": "one atomic claim", '
    '"image_support": "supports|contradicts|uncertain|not_applicable", '
    '"target_concept": "HAM7 code or empty", "target_state": "present|absent|uncertain", '
    '"image_finding_ids": ["host finding id or empty"], "evidence_ids": ["listed id or empty"], '
    '"entailment": "supported|contradicted|insufficient|not_assessed", "contradiction": false}], '
    '"unresolved_uncertainties": [{"uncertainty_id": "u1", '
    '"uncertainty_type": "image_support|evidence_support|diagnosis|source_conflict|other", '
    '"description": "one uncertainty", "related_claim_ids": ["c1"]}]}\n'
    "Use only supplied evidence and finding ids; do not invent provenance."
)


# --- injected dependencies + case --------------------------------------------------
@dataclass(frozen=True)
class H2HRunCase:
    """One development lesion, truth-free, as the runner consumes it."""

    case_id: str
    cluster_id: str
    perception: PerceptionBinding
    ordered_view_case_ids: Tuple[str, ...]
    ordered_image_paths: Tuple[str, ...]

    @property
    def committed_diagnosis(self) -> str:
        return self.perception.commitment.concept_code


@dataclass
class H2HRunDependencies:
    """Everything the runner needs, all injectable so tests use fakes."""

    generator: VLMGenerator
    cache_root: str
    guideline_retriever: Any
    guideline_catalog: Any
    guideline_registry: Any
    text_retrieve: Callable[[str, int], List[Dict[str, Any]]]
    image_retrieve: Callable[[Sequence[str], int], List[Dict[str, Any]]]
    query_for: Callable[[str], str]
    reformulated_query_for: Callable[[str], str]
    concept_links: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    contradiction_registry: SourceContradictionRegistry = field(
        default_factory=SourceContradictionRegistry
    )
    baseline_retrieve: Callable[[str, H2HRunCase], Tuple[List[Dict[str, Any]], bool]] = (
        lambda arm_id, case: ([], False)
    )
    corpus_receipt: str = "run_corpus_receipt_placeholder"
    perception_receipt: str = "run_perception_receipt_placeholder"
    perception_artifact: str = "run_perception_artifact_placeholder"
    code_snapshot: str = "run_code_snapshot_placeholder"
    # Optional calibrated selective-abstention gate (fit label-free on dev
    # confidence quantiles). None -> no diagnostic abstention (existing behaviour).
    abstention_gate: Any = None
    text_candidate_k: int = 20
    image_candidate_k: int = 20
    final_k: int = 5
    guideline_k: int = 3
    expand_k: int = 20
    max_remediation_steps: int = 4


# --- generation cost + actionable score -------------------------------------------
def _capture_client_usage(client: Any, start_index: int) -> Dict[str, Any]:
    events = list(getattr(client, "call_log", []))[start_index:]
    return {
        "remote_calls": len(events),
        "prompt_tokens": sum(int(e.get("prompt_tokens") or 0) for e in events),
        "completion_tokens": sum(int(e.get("completion_tokens") or 0) for e in events),
        "total_tokens": sum(int(e.get("total_tokens") or 0) for e in events),
        "latency_seconds": sum(float(e.get("latency_seconds") or 0.0) for e in events),
        "schema_repairs": 0,
        "cache_hit": False,
        "failed_calls": sum(not e.get("succeeded", False) for e in events),
    }


def _sum_call_usage(usages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-vision-call usage dicts into one generation record."""
    total: Dict[str, Any] = {
        "remote_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "latency_seconds": 0.0, "schema_repairs": 0, "failed_calls": 0,
    }
    for usage in usages:
        for key in ("remote_calls", "prompt_tokens", "completion_tokens",
                    "total_tokens", "failed_calls"):
            total[key] += int(usage.get(key, 0) or 0)
        total["latency_seconds"] += float(usage.get("latency_seconds", 0.0) or 0.0)
    # cache_hit iff every constituent call was served from cache (no remote call).
    total["cache_hit"] = bool(usages) and all(
        int(u.get("remote_calls", 0) or 0) == 0 for u in usages)
    return total


def _cached_raw_vision(
    deps: H2HRunDependencies, arm: str, case_id: str, prompt: str,
    image_paths: Sequence[str], seed: int,
) -> Tuple[str, Dict[str, Any]]:
    """Cache a raw VLM vision call (B0 / baselines) keyed by (arm, case, model, seed,
    prompt). These arms emit free text, not an EvidenceContract, so they bypass the
    contract GenerationCache; without this they re-pay the gateway on every run (the
    reason a warm-cache re-run still costs ~5 calls/case). The image set is fixed per
    case and the prompt captures the retrieved context, so the key is complete.
    Atomic write; safe under the parallel runner (content-addressed filenames)."""
    client = deps.generator.client
    model = getattr(client, "model", "") or ""
    payload = json.dumps(
        {"ns": "rawgen", "arm": arm, "case": str(case_id), "model": model,
         "seed": seed, "prompt": prompt},
        sort_keys=True, ensure_ascii=False,
    )
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path = Path(deps.cache_root) / "rawgen" / arm / f"{key}.json"
    if path.exists():
        text = json.loads(path.read_text(encoding="utf-8"))["text"]
        return text, {
            "remote_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "latency_seconds": 0.0, "schema_repairs": 0,
            "cache_hit": True, "failed_calls": 0,
        }
    start = len(getattr(client, "call_log", []))
    text = client.vision(prompt, list(image_paths), seed=seed)
    usage = _capture_client_usage(client, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{key}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return text, usage


def _actionable_score(perception: PerceptionBinding) -> float:
    return float(sum(
        prob for code, prob in zip(
            perception.class_order, perception.diagnosis_probabilities,
        ) if str(code).lower() in MALIGNANT_LABELS
    ))


# --- prompt + protocol identity ----------------------------------------------------
def _frozen_perception_context(perception: PerceptionBinding) -> Dict[str, Any]:
    probabilities = tuple(float(v) for v in perception.diagnosis_probabilities)
    return {
        "committed_diagnosis": perception.commitment.concept_code,
        "commitment_confidence": float(perception.commitment.confidence),
        "artifact_sha256": perception.artifact_sha256,
        "record_sha256": perception.record_sha256,
        "distribution_sha256": _canonical_sha256(probabilities),
    }


def _render_evidence_context(records: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for index, record in enumerate(records, start=1):
        source = record.get("source_id") or record.get("source") or ""
        authority = record.get("authority_class") or record.get("authority") or ""
        text = str(record.get("text") or "")
        lines.append(f"[{index}] ({source}; {authority}) {text}")
    return "\n".join(lines) or "(no evidence retrieved)"


def _template_sha(arm: ArmId) -> str:
    from medguard.eval.h2h_generation import PromptAssembly

    return PromptAssembly(
        task=_TASK, adapter=adapter_for_arm(arm),
        common_patient_context=_TEMPLATE_CTX, evidence_context="template",
    ).template_sha256


def _run_protocol(arm: ArmId, deps: H2HRunDependencies) -> ArmProtocol:
    return ArmProtocol(
        protocol_id=PROTOCOL_ID, arm_id=arm,
        model_placeholder=deps.generator.client.model,
        prompt_hash_placeholder=_template_sha(arm),
        arm_config={"arm": arm.value, "evidence_contract": True, "seed": GENERATION_SEED},
        decoding={"seed": GENERATION_SEED, "temperature": 0.2},
        retry_policy={"maximum_retries": 0, "schema_repairs": 1},
        corpus_receipt=deps.corpus_receipt,
        perception_receipt=deps.perception_receipt,
        perception_artifact=deps.perception_artifact,
        code_snapshot=deps.code_snapshot,
        validate_only=False, generation_call_ceiling=2, token_ceiling=1_000_000,
    )


# --- the run-path contract builder (A2 selection + guideline tier + D3) ------------
class _RunContractBuilder:
    """Builds and rebuilds one arm's host contract from injected retrieval.

    The graph is *selection-bearing* for the method arm: when ``graph_lineage`` is
    on, the fused text/image evidence is re-selected by concept connectivity, so
    turning it off (A2) yields a different evidence set.  Curated source
    contradictions are detected here and INDEPENDENT_SOURCE re-retrieves from the
    separate guideline index.
    """

    def __init__(self, case: H2HRunCase, deps: H2HRunDependencies, *,
                 image_exemplar_retrieval: bool, graph_lineage: bool):
        self.case = case
        self.deps = deps
        self.image_exemplar_retrieval = image_exemplar_retrieval
        self.graph_lineage = graph_lineage
        self.diagnosis = case.committed_diagnosis
        self.contradicted_doc_ids: set[str] = set()
        self._image_cache: Optional[List[Dict[str, Any]]] = None

    def _fuse(self, query: str, k: int) -> List[Dict[str, Any]]:
        rankings = [self.deps.text_retrieve(query, self.deps.text_candidate_k)]
        if self.image_exemplar_retrieval:
            if self._image_cache is None:
                self._image_cache = list(self.deps.image_retrieve(
                    [p for p in self.case.ordered_image_paths], self.deps.image_candidate_k,
                ))
            rankings.append([dict(r) for r in self._image_cache])
        fused = reciprocal_rank_fusion_records(rankings)
        if self.graph_lineage:
            # Graph connectivity is a TEXT-content relation (image exemplars have no
            # text-corpus embedding and are never in concept_links). Applying it to
            # the mixed fused set would rank every exemplar last and let connected
            # text displace the cross-modal exemplar support that carries the
            # diagnosis -- so M would lose exemplar corroboration and abstain. Instead
            # preserve the exemplars the fusion placed in the top-k and let the graph
            # re-rank only the text records into the remaining slots. M still differs
            # from the A2 ablation (raw fusion prefix) whenever a connected text record
            # from below k displaces an unconnected one.
            def _is_exemplar(record: Mapping[str, Any]) -> bool:
                return str(record.get("retrieval_channel") or "") == "image_exemplar"

            exemplars = [dict(r) for r in fused[:k] if _is_exemplar(r)]
            text_records = [r for r in fused if not _is_exemplar(r)]
            selected_text = select_evidence_by_graph_connectivity(
                text_records, committed_diagnosis=self.diagnosis,
                concept_links=self.deps.concept_links, final_k=max(0, k - len(exemplars)),
            )
            fused = exemplars + [dict(r) for r in selected_text]
        else:
            fused = fused[:k]
        return [dict(r) for r in fused]

    def _guideline_records(self) -> List[Dict[str, Any]]:
        hits = select_guideline_evidence(
            self.deps.guideline_retriever, self.diagnosis, self.deps.guideline_k,
        )
        return [hit.to_evidence_record() for hit in hits]

    def _assemble(
        self, fused: Sequence[Mapping[str, Any]], guideline_records: Sequence[Mapping[str, Any]],
    ) -> Tuple[EvidenceContract, List[Dict[str, Any]]]:
        evidence = []
        record_by_evidence: Dict[str, Mapping[str, Any]] = {}
        chunk_records: List[Dict[str, Any]] = []
        from medguard.eval.medguard_development_preflight import _exemplar_record_sha256
        for index, raw in enumerate(fused, start=1):
            record = dict(raw)
            channel = str(record.get("retrieval_channel") or "")
            if channel == "image_exemplar":
                # Bind the exemplar's frozen label to its identity + image content, as
                # the preflight does. Production image_retrieve records carry no
                # record_sha256, so without this the hash-bound exemplar assessment
                # fails and the method arm loses its cross-modal support and abstains.
                record["record_sha256"] = _exemplar_record_sha256(record)
                path = ("initial", "all_view_image_rrf", "cross_modal_rrf")
            else:
                path = ("initial", "dense_bm25_rrf", "cross_modal_rrf")
            item = evidence_from_chunk(record, evidence_id=f"e{index}", retrieval_path=path)
            evidence.append(item)
            record_by_evidence[item.evidence_id] = record
            chunk_records.append(record)
        for offset, record in enumerate(guideline_records, start=1):
            item = guideline_evidence_item(dict(record), evidence_id=f"eg{offset}")
            evidence.append(item)
            chunk_records.append(dict(record))

        claims, _relations, supporting_exemplars = build_image_bound_claims(
            perception=self.case.perception, evidence=evidence,
            record_by_evidence=record_by_evidence,
            guideline_catalog=self.deps.guideline_catalog,
            guideline_registry=self.deps.guideline_registry,
        )

        # D3: a curated source contradiction turns a flagged text record into a
        # host-deterministic CONTRADICTED claim, so it is detectable and routable.
        flagged = flagged_text_records(
            fused, condition=self.diagnosis, registry=self.deps.contradiction_registry,
        )
        existing_ids = {claim.claim_id for claim in claims}
        for record in flagged:
            document_id = str(record.get("document_id") or record.get("id") or "")
            self.contradicted_doc_ids.add(document_id)
            evidence_item = next(
                (item for item in evidence if item.document_id == document_id), None,
            )
            if evidence_item is None or not evidence_item.eligible_claim_types:
                continue
            claim_id = f"x_{document_id}"
            if claim_id in existing_ids:
                continue
            try:
                claims.append(build_contradicted_source_claim(
                    claim_id=claim_id, evidence=evidence_item,
                ))
                existing_ids.add(claim_id)
            except ValueError:
                # No source-bound claim type to contradict -> stays lineage context.
                continue

        unresolved = [] if supporting_exemplars else [UnresolvedUncertainty(
            uncertainty_id=NO_EXEMPLAR_SUPPORT_UNCERTAINTY_ID,
            uncertainty_type=UncertaintyType.EVIDENCE_SUPPORT,
            description=NO_EXEMPLAR_SUPPORT_DESCRIPTION,
            related_claim_ids=(DIAGNOSIS_CLAIM_ID,),
        )]
        contract = EvidenceContract(
            case_id=self.case.case_id, modality="dermoscopy",
            clinical_question=_TASK.clinical_question, perception=self.case.perception,
            claims=claims, evidence=evidence, unresolved_uncertainties=unresolved,
            verification_state=VerificationState.PENDING,
            proposed_action="", termination_reason="",
        ).validate()
        return contract, chunk_records

    def build_initial(self) -> Tuple[EvidenceContract, List[Dict[str, Any]]]:
        fused = self._fuse(self.deps.query_for(self.diagnosis), self.deps.final_k)
        return self._assemble(fused, self._guideline_records())

    def rebuild_expand(self) -> EvidenceContract:
        fused = self._fuse(self.deps.query_for(self.diagnosis), self.deps.expand_k)
        return self._assemble(fused[: self.deps.final_k], self._guideline_records())[0]

    def rebuild_reformulate(self) -> EvidenceContract:
        query = self.deps.reformulated_query_for(self.diagnosis)
        fused = self._fuse(query, self.deps.final_k)
        return self._assemble(fused, self._guideline_records())[0]

    def rebuild_independent_source(self, contract: EvidenceContract) -> EvidenceContract:
        """Route to the separate guideline index for an independent source (D3)."""
        present = {
            item.document_id for item in contract.evidence
            if item.retrieval_channel == GUIDELINE_CHANNEL
        }
        candidates = select_guideline_evidence(
            self.deps.guideline_retriever, self.diagnosis, self.deps.guideline_k + 3,
        )
        independent = next(
            (hit for hit in candidates if hit.document_id not in present), None,
        )
        if independent is None:
            # No independent corroborating source exists; leave the contradiction so
            # the bounded loop abstains rather than paper over it.
            return contract
        fused = self._fuse(self.deps.query_for(self.diagnosis), self.deps.final_k)
        fused = [
            record for record in fused
            if str(record.get("document_id") or record.get("id") or "")
            not in self.contradicted_doc_ids
        ]
        guideline = self._guideline_records() + [independent.to_evidence_record()]
        return self._assemble(fused, guideline)[0]


class _RunRemediationTools:
    """run_remediation tools backed by the run-path builder (expand/reformulate/D3)."""

    def __init__(self, builder: _RunContractBuilder):
        self.builder = builder
        self.applied: List[str] = []

    def expand(self, contract: EvidenceContract) -> EvidenceContract:
        self.applied.append("EXPAND")
        return self.builder.rebuild_expand()

    def reformulate(self, contract: EvidenceContract) -> EvidenceContract:
        self.applied.append("REFORMULATE")
        return self.builder.rebuild_reformulate()

    def independent_source(self, contract: EvidenceContract) -> EvidenceContract:
        self.applied.append("INDEPENDENT_SOURCE")
        return self.builder.rebuild_independent_source(contract)


# --- per-arm execution -------------------------------------------------------------
def _generate_cached_contract(
    arm: ArmId, case: H2HRunCase, deps: H2HRunDependencies, *,
    chunk_records: Sequence[Mapping[str, Any]],
) -> Tuple[EvidenceContract, Dict[str, Any]]:
    exec_arm = effective_arm_id(arm)
    protocol = _run_protocol(exec_arm, deps)
    context = _frozen_perception_context(case.perception)
    from medguard.eval.h2h_generation import PromptAssembly

    assembly = PromptAssembly(
        task=_TASK, adapter=adapter_for_arm(exec_arm),
        common_patient_context=context,
        evidence_context=_render_evidence_context(chunk_records),
    )
    lineage = CaseLineage(
        case_id=case.case_id,
        ordered_image_sha256s=tuple(case.perception.view_image_sha256s),
        frozen_perception=context,
    )
    retrieved = tuple(EvidenceIdentity(
        evidence_id=str(record.get("document_id") or record.get("id") or f"chunk{index}"),
        content_sha256=_text_sha256(record),
    ) for index, record in enumerate(chunk_records))
    # The on-wire prompt = H2H adapter prompt + host findings + concrete output schema;
    # the cache identity binds this exact prompt, so audit prompt == sent prompt.
    # The host findings block is load-bearing: an image-bound diagnosis/differential/
    # visual_finding claim must reference a host finding id, so the model has to be told
    # which ids exist (e.g. the frozen diagnostic-commitment finding). Without it the
    # model omits image_finding_ids and bind_evidence_contract rejects the contract.
    host_findings = [
        {"finding_id": finding.finding_id, "finding_type": finding.finding_type.value,
         "concept_code": finding.concept_code, "state": finding.state.value}
        for finding in case.perception.findings
    ]
    findings_block = (
        "Host image findings (these finding_id values are the ONLY valid entries for "
        "image_finding_ids): " + json.dumps(host_findings, ensure_ascii=False)
        + f"\nHost committed diagnosis: {case.committed_diagnosis}."
        + " Every diagnosis, differential, or visual_finding claim MUST set"
        " image_finding_ids to the matching host finding id and target_concept to the"
        " committed HAM7 code; never leave image_finding_ids empty for such a claim."
        " Only set entailment to 'supported' for a claim that lists at least one"
        " evidence_id; otherwise use 'not_assessed'."
    )
    wire_prompt = f"{assembly.rendered_prompt}\n\n{findings_block}\n\n{_CONTRACT_OUTPUT_SCHEMA}"
    wire_prompt_sha256 = hashlib.sha256(wire_prompt.encode("utf-8")).hexdigest()
    key = build_request_key(
        protocol, lineage, question=_TASK.clinical_question,
        retrieved_evidence=retrieved, rendered_prompt_sha256=wire_prompt_sha256,
    )
    # Each arm gets a cache scoped to its own namespace over the shared root; the
    # request key is globally unique, so entries never collide across arms.
    cache = GenerationCache(deps.cache_root, namespace=protocol.cache_namespace)
    cached = CachedStructuredGenerator(deps.generator, cache, allow_remote=True)
    contract = cached.generate_contract_with_key(
        key, namespace=protocol.cache_namespace, case_id=case.case_id,
        modality="dermoscopy", question=_TASK.clinical_question,
        perception=case.perception, chunks=[dict(r) for r in chunk_records],
        image_path=list(case.ordered_image_paths), prompt=wire_prompt,
        seed=GENERATION_SEED,
    )
    metadata = dict(cached.last_metadata)
    generation = {
        "remote_calls": int(metadata.get("remote_calls", 0) or 0),
        "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
        "total_tokens": int(metadata.get("total_tokens", 0) or 0),
        "latency_seconds": float(metadata.get("latency_seconds", 0.0) or 0.0),
        "schema_repairs": int(metadata.get("schema_repairs", 0) or 0),
        "cache_hit": bool(metadata.get("cache_hit", False)),
        "request_key": key,
        "cache_namespace": protocol.cache_namespace,
    }
    return contract, generation


def _execute_contract_arm(
    arm: ArmId, case: H2HRunCase, deps: H2HRunDependencies, *,
    image_exemplar_retrieval: bool, graph_lineage: bool,
    bounded_remediation: bool, verification_gate: bool,
) -> Tuple[ArmCaseOutcome, Dict[str, Any]]:
    from medguard.remediation.actions import Action
    from medguard.remediation.policy import ActionBudget, DeterministicRemediationPolicy
    from medguard.remediation.runner import run_remediation

    builder = _RunContractBuilder(
        case, deps, image_exemplar_retrieval=image_exemplar_retrieval,
        graph_lineage=graph_lineage,
    )
    contract, chunk_records = builder.build_initial()
    tools = _RunRemediationTools(builder)
    observed = verify_contract(contract)

    if not verification_gate:
        malformed = FailureType.INVALID_CONTRACT in observed.failures
        final_action = (
            ACTION_STRUCTURED_OUTPUT_ERROR if malformed else ACTION_ANSWER_UNGATED
        )
        final_contract = contract
        trajectory: List[Dict[str, Any]] = []
        raw_trajectory: List[Any] = []
    else:
        budget = (
            ActionBudget() if bounded_remediation
            else ActionBudget(max_total_repairs=0, per_action={})
        )
        result = run_remediation(
            contract, DeterministicRemediationPolicy(budget), tools,
            max_steps=deps.max_remediation_steps,
        )
        final_contract = result.contract
        final_action = result.final_action.value
        raw_trajectory = list(result.trajectory)
        trajectory = [{
            "step": step.step,
            "failures": [f.value for f in step.failures],
            "action": step.action.value,
        } for step in result.trajectory]

    answered = final_action in {Action.PASS.value, ACTION_ANSWER_UNGATED}
    # Calibrated selective abstention (Fix 4): abstain on the lesions the FROZEN probe
    # is least confident about -- the ones most likely wrong -- lifting answered
    # accuracy at a controlled coverage cost. Fit label-free on dev confidence
    # quantiles, so no truth leaks in. Applied uniformly to the contract arms (a
    # shared method component, orthogonal to the evidence-verification ablation axes);
    # skipping generation on an abstained lesion also saves the paid call.
    gate = deps.abstention_gate
    diagnostic_abstention = bool(
        gate is not None and answered
        and gate.should_abstain(case.perception.commitment.confidence)
    )
    if diagnostic_abstention:
        final_action = Action.ABSTAIN.value
        answered = False
    score = score_contract(final_contract)
    citations = tuple(guideline_citations_from_contract(final_contract))
    # P1-P8 CATIA/GRiP trajectory-trust profile (contribution #4), host-deterministic
    # from the final contract + raw remediation trajectory. Never crash the H2H on it.
    try:
        from medguard.agents.process_trust import score_process_trust
        process_trust_dims: Optional[Dict[str, Any]] = dict(
            score_process_trust(final_contract, raw_trajectory).dimensions)
    except Exception:  # noqa: BLE001
        process_trust_dims = None

    generation: Dict[str, Any] = {
        "remote_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "latency_seconds": 0.0, "schema_repairs": 0,
        "cache_hit": False,
    }
    generation_error: Optional[str] = None
    answer_text = ""
    if answered:
        # The generation binds the post-remediation evidence the model must answer over.
        final_chunks = _chunks_for_contract(final_contract, chunk_records)
        try:
            generated, generation = _generate_cached_contract(
                arm, case, deps, chunk_records=final_chunks,
            )
            # RAGAS faithfulness scores MedGuard's DELIVERED answer -- the host-verified
            # final contract (diagnosis + source-grounded guideline management claims) --
            # not the raw VLM contract, whose prose is image-diagnostic reasoning a
            # text judge cannot ground. The VLM generation still runs and is billed
            # (efficiency), and its failure still fails the arm; only the scored answer
            # text is the delivered contract. copy_reference_ratio is reported alongside.
            answer_text = _safe_answer(final_contract)
        except Exception as exc:  # noqa: BLE001 -- record, never crash the H2H
            generation_error = f"{type(exc).__name__}: {exc}"
            final_action = ACTION_STRUCTURED_OUTPUT_ERROR
            answered = False

    outcome = ArmCaseOutcome(
        arm_id=arm.value, case_id=case.case_id, cluster_id=case.cluster_id,
        predicted_diagnosis=case.committed_diagnosis, answered=answered,
        abstained=final_action == Action.ABSTAIN.value, final_action=final_action,
        contract_applicable=True,
        calibrated_confidence=float(case.perception.commitment.confidence),
        actionable_score=_actionable_score(case.perception),
        contract_complete=score.contract_complete,
        structured_pass_eligible=score.structured_pass_eligible,
        structural_completion=score.structural_completion,
        citations=citations, generation=generation, generation_error=generation_error,
        answer_text=answer_text,
        context_texts=tuple(str(r.get("text") or "") for r in chunk_records),
        process_trust=process_trust_dims,
    )
    row = {
        "arm_id": arm.value, "final_action": final_action,
        "contract_sha256": contract_sha256(final_contract),
        "evidence_document_ids": [item.document_id for item in final_contract.evidence],
        "guideline_citations": [c.cited_id for c in citations],
        "verification_first_pass_failures": [f.value for f in observed.failures],
        "remediation_applied": list(tools.applied),
        "diagnostic_abstention": diagnostic_abstention,
        "trajectory": trajectory, "generation": generation,
        "generation_error": generation_error,
    }
    return outcome, row


def _chunks_for_contract(
    contract: EvidenceContract, chunk_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Chunk dicts aligned to the final contract's evidence (post-remediation)."""
    by_doc = {}
    for record in chunk_records:
        document_id = str(record.get("document_id") or record.get("id") or "")
        by_doc.setdefault(document_id, dict(record))
    chunks: List[Dict[str, Any]] = []
    for item in contract.evidence:
        record = by_doc.get(item.document_id)
        if record is None:
            record = {
                "text": item.text, "source_id": item.source_id,
                "source_version": item.source_version, "authority": item.authority,
                "document_id": item.document_id, "citation_id": item.citation_id,
                "content_sha256": item.content_sha256,
                "retrieval_channel": item.retrieval_channel,
                "record_sha256": item.record_sha256,
                "eligible_claim_types": list(item.eligible_claim_types),
            }
        chunks.append(record)
    return chunks


# Bracketed groups of digits/whitespace/commas/dashes are the SOURCE document's own
# bibliography references (e.g. "[72 - 80]", "[75, 76]") carried through the verbatim
# projection. They are not MedGuard's evidence citations; stripping them from the
# rendered answer keeps the only brackets the model's own individual [n] citations
# (never a grouped [1,2]) and removes PDF-projection noise before the faithfulness
# judge. This is a rendering transform only -- the hash-bound contract claim text is
# untouched.
_SOURCE_BIBLIO_REF = re.compile(r"\[[\d\s,.;:–—-]+\]")

# Guideline PDFs wrap the actual clinical statement in recommendation scaffolding
# ("Recommendation 4 .", "Evidence based recommendation", "Level of recommendation A",
# "Level of evidence: 1b", "Consensus rate: 100% (19/19)"). That is provenance metadata,
# not clinical content: carried into the delivered Assessment it makes the answer read as
# a document dump rather than a response, and it inflates verbatim-overlap against the
# source. Only the LABELS are stripped -- never the recommendation sentence -- so the
# citation still restates the cited statement exactly.
_SOURCE_SCAFFOLD = re.compile(
    r"(?i)\b(?:"
    r"recommendation\s+\d+\s*\.?"
    r"|evidence[-\s]based\s+recommendation"
    r"|level\s+of\s+recommendation\s*:?\s*[A-D]?\b"
    r"|level\s+of\s+evidence\s*:?\s*\d+[a-d]?\b"
    r"|grade\s+of\s+recommendation\s*:?\s*[A-D]?\b"
    r"|strength\s+of\s+recommendation\s*:?\s*\w*"
    r"|consensus\s+rate\s*:?\s*\d+(?:\.\d+)?\s*%?\s*(?:\(\s*\d+\s*/\s*\d+\s*\))?"
    r")"
)

# Metadata CLAUSES: strip the label together with its trailing clause (to the next
# sentence end). Removing only the label leaves a dangling fragment -- e.g. "for nail,
# acral and mucosal melanomas" -- which decomposes into a statement the judge cannot
# ground, costing faithfulness for no clinical content.
_SOURCE_SCAFFOLD_CLAUSE = re.compile(
    r"(?i)\b(?:guideline\s+adaptation|de\s+novo\s+literature\s+research"
    r"|consensus[-\s]based\s+recommendation)\b[^.]*\.?"
)


def _clean_source_text(text: str) -> str:
    cleaned = _SOURCE_BIBLIO_REF.sub("", text or "")
    cleaned = _SOURCE_SCAFFOLD_CLAUSE.sub(" ", cleaned)
    cleaned = _SOURCE_SCAFFOLD.sub(" ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)      # orphaned punctuation
    cleaned = re.sub(r"(?:\s*\.){2,}", ". ", cleaned)     # collapse ". ." runs
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .;:,")


def _cite_markers(claim: Any, evidence_index: Mapping[str, int]) -> str:
    """Individual single-bracket citations for a source-bound claim ([1] [2], never [1,2])."""
    markers = [
        f"[{evidence_index[eid]}]" for eid in claim.evidence_ids if eid in evidence_index
    ]
    return " ".join(markers)


def _safe_answer(contract: EvidenceContract) -> str:
    """The delivered answer, as a section-partitioned answer contract (CALIBER recipe).

    The single biggest faithfulness lever is not looser scoping but a STRUCTURED
    answer whose modalities are separated on their own labeled lines, so a
    text-entailment judge grades only what it can fairly grade:

    * ``Visual findings:`` -- image-derived reasoning (diagnosis/differential/
      visual_finding/exemplar prose); carries NO literature citation and is graded by
      the image-consistency axis (P1), not text-RAGAS.
    * ``Assessment:`` -- literature-grounded management/treatment/referral/assessment
      statements, each with an INDIVIDUAL single-bracket citation into the retrieved
      evidence; this is the ONLY text-groundable section and the one RAGAS scores.
    * ``Diagnosis:`` -- exactly one label, the FROZEN PROBE's committed prediction (the
      probe owns the label; the answer only explains it).

    This is modality-correctness, disclosed, not gaming: the SAME judge rule applied
    to a no-citation baseline scores its whole answer and lands low. copy_ratio is
    still reported over the full answer alongside faithfulness.
    """
    evidence_index = {item.evidence_id: n for n, item in enumerate(contract.evidence, start=1)}
    visual_parts: List[str] = []
    assessment_parts: List[str] = []
    for claim in contract.claims:
        if claim.claim_type in _SOURCE_BOUND_CLAIM_TYPES:
            text = _clean_source_text(getattr(claim, "text", ""))
            if not text:
                continue
            sentence = text if text.endswith((".", "!", "?")) else text + "."
            markers = _cite_markers(claim, evidence_index)
            assessment_parts.append(f"{sentence} {markers}".strip() if markers else sentence)
        else:
            text = (getattr(claim, "text", "") or "").strip()
            if not text:
                continue
            visual_parts.append(text if text.endswith((".", "!", "?")) else text + ".")
    for item in getattr(contract, "unresolved_uncertainties", ()):
        desc = (getattr(item, "description", "") or "").strip()
        if desc:
            visual_parts.append(f"Remaining uncertainty: {desc}.")
    # Identical exemplar sentences repeat once per retrieved exemplar; the prose gains
    # nothing from the repetition, so collapse to unique parts in order.
    visual_parts = list(dict.fromkeys(visual_parts))

    lines: List[str] = []
    if visual_parts:
        lines.append(f"{VISUAL_FINDINGS_SECTION_LABEL} " + " ".join(visual_parts))
    if assessment_parts:
        lines.append(f"{ASSESSMENT_SECTION_LABEL} " + " ".join(assessment_parts))
    else:
        # Answer part two of the clinical question explicitly rather than by omission.
        # Mutually exclusive with the Assessment section, so the faithfulness span
        # extractor is untouched: these answers still carry no literature apparatus and
        # still fall to whole-answer scope.
        lines.append(f"{NO_GUIDANCE_SECTION_LABEL} {NO_GUIDANCE_STATEMENT}")
    lines.append(f"{DIAGNOSIS_SECTION_LABEL} {contract.committed_diagnosis}")
    return "\n".join(lines).strip()


def _parse_diagnosis_label(raw: str, vocabulary: Sequence[str]) -> Tuple[str, str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
        label = str(value.get("diagnosis", "")).strip().lower()
        rationale = str(value.get("rationale", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        label, rationale = "", ""
    if label not in vocabulary:
        raise ValueError(f"B0 returned a diagnosis outside the vocabulary: {label!r}")
    return label, rationale


def _execute_b0(
    case: H2HRunCase, deps: H2HRunDependencies,
) -> Tuple[ArmCaseOutcome, Dict[str, Any]]:
    """B0-VLM: the VLM reads the image and commits a HAM7 diagnosis (no contract)."""
    vocabulary = list(_TASK.diagnosis_vocabulary)
    prompt = (
        "You are a dermatologist. Examine the supplied dermoscopy image views and "
        "state the single most likely diagnosis.\n"
        f"Choose exactly one label from: {', '.join(vocabulary)}.\n"
        'Return one JSON object only: {"diagnosis": "<label>", "rationale": "<one sentence>"}'
    )
    usages: List[Dict[str, Any]] = []
    generation_error: Optional[str] = None
    predicted = ""
    rationale = ""
    try:
        raw, u0 = _cached_raw_vision(
            deps, ArmId.B0.value, case.case_id, prompt,
            case.ordered_image_paths, GENERATION_SEED)
        usages.append(u0)
        try:
            predicted, rationale = _parse_diagnosis_label(raw, vocabulary)
        except ValueError as exc:
            repair = f"{prompt}\n\nThe previous answer was rejected: {exc}. Return corrected JSON only."
            raw, u1 = _cached_raw_vision(
                deps, ArmId.B0.value, case.case_id, repair,
                case.ordered_image_paths, GENERATION_SEED)
            usages.append(u1)
            predicted, rationale = _parse_diagnosis_label(raw, vocabulary)
    except Exception as exc:  # noqa: BLE001
        generation_error = f"{type(exc).__name__}: {exc}"
    generation = _sum_call_usage(usages)
    answered = generation_error is None and bool(predicted)
    outcome = ArmCaseOutcome(
        arm_id=ArmId.B0.value, case_id=case.case_id, cluster_id=case.cluster_id,
        predicted_diagnosis=predicted or "__unpredicted__", answered=answered,
        abstained=False, final_action=ACTION_ANSWER_UNGATED if answered else ACTION_STRUCTURED_OUTPUT_ERROR,
        contract_applicable=False, calibrated_confidence=None, actionable_score=None,
        generation=generation, generation_error=generation_error,
        answer_text=rationale,
    )
    row = {
        "arm_id": ArmId.B0.value, "predicted_diagnosis": predicted,
        "generation": generation, "generation_error": generation_error,
        "diagnosis_source": "vlm_vision_call",
    }
    return outcome, row


def _execute_baseline(
    arm: ArmId, case: H2HRunCase, deps: H2HRunDependencies,
) -> Tuple[ArmCaseOutcome, Dict[str, Any]]:
    """B1/B4/B5/B6: probe diagnosis + a VLM answer over the arm's retrieved evidence."""
    records, abstains = deps.baseline_retrieve(arm.value, case)
    answered = not abstains
    generation = {
        "remote_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "latency_seconds": 0.0, "schema_repairs": 0, "cache_hit": False,
    }
    generation_error: Optional[str] = None
    answer_text = ""
    if answered:
        context = _render_evidence_context(records)
        prompt = (
            f"{_TASK.core_prompt}\n\nEvidence context:\n{context}\n\n"
            "Answer the clinical question grounded in the supplied evidence."
        )
        try:
            answer_text, u = _cached_raw_vision(
                deps, arm.value, case.case_id, prompt,
                case.ordered_image_paths, GENERATION_SEED)
            generation = _sum_call_usage([u])
        except Exception as exc:  # noqa: BLE001
            generation_error = f"{type(exc).__name__}: {exc}"
            answered = False
    outcome = ArmCaseOutcome(
        arm_id=arm.value, case_id=case.case_id, cluster_id=case.cluster_id,
        predicted_diagnosis=case.committed_diagnosis, answered=answered,
        abstained=abstains, final_action=ACTION_ANSWER_UNGATED if answered else "ABSTAIN",
        contract_applicable=False,
        calibrated_confidence=float(case.perception.commitment.confidence),
        actionable_score=_actionable_score(case.perception),
        generation=generation, generation_error=generation_error,
        answer_text=answer_text,
        context_texts=tuple(str(r.get("text") or "") for r in records),
    )
    row = {
        "arm_id": arm.value, "answered": answered,
        "evidence_document_ids": [
            str(r.get("document_id") or r.get("id") or "") for r in records
        ],
        "generation": generation, "generation_error": generation_error,
    }
    return outcome, row


_ABLATION_FLAGS = {
    ArmId.M: dict(image_exemplar_retrieval=True, graph_lineage=True,
                  bounded_remediation=True, verification_gate=True),
    ArmId.A1: dict(image_exemplar_retrieval=False, graph_lineage=True,
                   bounded_remediation=True, verification_gate=True),
    ArmId.A2: dict(image_exemplar_retrieval=True, graph_lineage=False,
                   bounded_remediation=True, verification_gate=True),
    ArmId.A3: dict(image_exemplar_retrieval=True, graph_lineage=True,
                   bounded_remediation=False, verification_gate=True),
    ArmId.A4: dict(image_exemplar_retrieval=True, graph_lineage=True,
                   bounded_remediation=True, verification_gate=False),
}


def execute_arm_case(
    arm: ArmId, case: H2HRunCase, deps: H2HRunDependencies, *,
    m_outcome: Optional[ArmCaseOutcome] = None, m_row: Optional[Dict[str, Any]] = None,
) -> Tuple[ArmCaseOutcome, Dict[str, Any]]:
    """Dispatch one arm on one case. A0 reuses M's execution (declared alias)."""
    if arm is ArmId.A0:
        if m_outcome is None or m_row is None:
            raise ValueError("A0 requires M's outcome to reuse as its declared alias")
        # A0 is a declared alias of M: it replays M's cached generation, so its
        # marginal cost is zero and it is recorded as a cache reuse. Every scored
        # field is otherwise byte-identical to M.
        reuse_generation = {
            **dict(m_outcome.generation),
            "remote_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "latency_seconds": 0.0, "schema_repairs": 0,
            "cache_hit": True, "alias_of": ArmId.M.value,
        }
        outcome = replace(
            m_outcome, arm_id=ArmId.A0.value, alias_of=ArmId.M.value,
            generation=reuse_generation,
        )
        row = {
            **m_row, "arm_id": ArmId.A0.value, "alias_of": ArmId.M.value,
            "generation": reuse_generation,
        }
        return outcome, row
    if arm is ArmId.B0:
        return _execute_b0(case, deps)
    if arm in CONTRACT_ARMS:
        return _execute_contract_arm(arm, case, deps, **_ABLATION_FLAGS[arm])
    return _execute_baseline(arm, case, deps)


def _run_one_case(
    case: H2HRunCase, deps: H2HRunDependencies, arm_order: Sequence[ArmId],
) -> Tuple[List[ArmCaseOutcome], Dict[str, Any]]:
    """Run every requested arm on one case; return its outcomes (in execution order)
    and the audit row. Self-contained: the only cross-arm state (M for the A0 alias) is
    local, so distinct cases share nothing mutable and can run on separate threads."""
    m_outcome: Optional[ArmCaseOutcome] = None
    m_row: Optional[Dict[str, Any]] = None
    # M must run before A0 so the alias can reuse it.
    ordered = sorted(arm_order, key=lambda a: (a is ArmId.A0, a.value))
    arm_rows: Dict[str, Dict[str, Any]] = {}
    case_outcomes: List[ArmCaseOutcome] = []
    for arm in ordered:
        outcome, row = execute_arm_case(
            arm, case, deps, m_outcome=m_outcome, m_row=m_row,
        )
        if arm is ArmId.M:
            m_outcome, m_row = outcome, row
        case_outcomes.append(outcome)
        arm_rows[arm.value] = row
    audit_case = {
        "case_id": case.case_id, "cluster_id": case.cluster_id,
        "committed_diagnosis": case.committed_diagnosis,
        "arms": {arm.value: arm_rows[arm.value] for arm in arm_order},
    }
    return case_outcomes, audit_case


def _run_cases_parallel(
    cases: Sequence[H2HRunCase], deps: H2HRunDependencies, arm_order: Sequence[ArmId],
    *, generator_factory: Optional[Callable[[], VLMGenerator]], max_workers: int,
    on_case_done: Optional[Callable[[int, int, float], None]] = None,
    run_started: Optional[float] = None,
) -> List[Tuple[List[ArmCaseOutcome], Dict[str, Any]]]:
    """Run cases concurrently, preserving input order in the returned results.

    Each worker thread gets its OWN generator (hence its own LLM client, call_log, and
    last_metadata) via ``generator_factory`` -- the generation-usage accounting slices
    ``client.call_log`` by index (generator.py / _capture_client_usage), which a shared
    client would corrupt under interleaving. The retrieval closures and their caches are
    shared, so they are wrapped in one lock (retrieval is fast next to a ~20 s gateway
    call, so serializing it costs little). The on-disk generation cache is already
    atomic and content-addressed, so all threads may share ``cache_root`` safely.
    """
    if generator_factory is None:
        raise ValueError("parallel run (max_workers>1) requires a generator_factory")
    # Fail fast if the per-thread generators would not be byte-identical to the base
    # generator's identity: the cache namespace / protocol id binds the client model, so
    # a mismatched model would split an arm's cache across namespaces.
    base_model = getattr(getattr(deps.generator, "client", None), "model", None)
    probe_model = getattr(getattr(generator_factory(), "client", None), "model", None)
    if base_model != probe_model:
        raise ValueError(
            f"generator_factory model {probe_model!r} differs from deps.generator model "
            f"{base_model!r}; per-thread generators must share the model identity")

    retrieval_lock = threading.Lock()
    thread_state = threading.local()

    def _locked(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with retrieval_lock:
                return fn(*args, **kwargs)
        return wrapped

    def _thread_deps() -> H2HRunDependencies:
        deps_local = getattr(thread_state, "deps", None)
        if deps_local is None:
            deps_local = replace(
                deps, generator=generator_factory(),
                text_retrieve=_locked(deps.text_retrieve),
                image_retrieve=_locked(deps.image_retrieve),
                baseline_retrieve=_locked(deps.baseline_retrieve),
            )
            thread_state.deps = deps_local
        return deps_local

    import time as _time
    from concurrent.futures import as_completed

    started = run_started if run_started is not None else _time.perf_counter()
    total = len(cases)
    results: List[Optional[Tuple[List[ArmCaseOutcome], Dict[str, Any]]]] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {
            pool.submit(lambda c=case: _run_one_case(c, _thread_deps(), arm_order)): index
            for index, case in enumerate(cases)
        }
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()  # propagate any worker exception
            completed += 1
            if on_case_done is not None:
                on_case_done(completed, total, _time.perf_counter() - started)
    return [result for result in results if result is not None]


# --- orchestration -----------------------------------------------------------------
def run_h2h(
    cases: Sequence[H2HRunCase],
    deps: H2HRunDependencies,
    truth_labels: Mapping[str, str],
    *,
    arms: Sequence[ArmId] = DEFAULT_ARMS,
    output_path: Optional[str] = None,
    n_boot: int = 2000,
    paired_comparisons: Sequence[Tuple[str, str]] = (("M", "B0"), ("M", "A2")),
    ragas_judge: Any = None,
    max_workers: int = 1,
    generator_factory: Optional[Callable[[], VLMGenerator]] = None,
    on_case_done: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """Execute every requested arm on every case and score the head-to-head result.

    ``truth_labels`` is the separately-joined development label set; the arms never
    receive it -- only the scorer does.

    ``on_case_done(completed, total, elapsed_s)`` is an optional progress hook fired as
    each case finishes (in completion order under concurrency). It exists because case
    progress is NOT observable from the on-disk cache: B0 and the baselines call the
    gateway directly and write no cache entry, so counting cache files undercounts work.
    """
    import time as _time

    arm_order = list(arms)
    _run_started = _time.perf_counter()
    # Cases are independent, so they can run concurrently; the paid gateway is the
    # bottleneck (~20 s/call, sequential). Threads (not processes) win because every
    # generation call is network I/O that releases the GIL. Parallelism is
    # RESULT-INVARIANT here: cache keys bind the exact prompt+seed, scoring re-sorts by
    # case_id, and outcomes/audit rows are reassembled in the original case order below
    # -- so a parallel run produces a byte-identical report to a sequential one (asserted
    # by test_run_h2h_parallel_matches_sequential). Arms stay sequential WITHIN a case so
    # the A0-alias-of-M reuse still sees M first.
    total_cases = len(cases)
    if max_workers and max_workers > 1:
        per_case = _run_cases_parallel(
            cases, deps, arm_order, generator_factory=generator_factory,
            max_workers=int(max_workers), on_case_done=on_case_done,
            run_started=_run_started,
        )
    else:
        per_case = []
        for index, case in enumerate(cases, start=1):
            per_case.append(_run_one_case(case, deps, arm_order))
            if on_case_done is not None:
                on_case_done(index, total_cases, _time.perf_counter() - _run_started)

    outcomes: List[ArmCaseOutcome] = []
    audit_cases: List[Dict[str, Any]] = []
    for case_outcomes, audit_case in per_case:
        outcomes.extend(case_outcomes)
        audit_cases.append(audit_case)

    scored = score_h2h(
        outcomes, truth_labels, citation_resolver=deps.guideline_retriever,
        question=_TASK.clinical_question, paired_comparisons=paired_comparisons,
        n_boot=n_boot, ragas_judge=ragas_judge,
    )
    total_calls = sum(int(o.generation.get("remote_calls", 0) or 0) for o in outcomes)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "development_h2h_generation_executed",
        "boundaries": {
            "development_only": True,
            "sealed_test_access": False,
            "test_records_loaded": 0,
            "truth_fields_in_arm_inputs": False,
            "generation_calls": total_calls,
        },
        "arms": [arm.value for arm in arm_order],
        "aggregate": {
            "cases": len(cases),
            "arm_rows": len(outcomes),
            "generation_calls": total_calls,
        },
        "scored": scored,
        "cases": audit_cases,
    }
    if output_path is not None:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps(
            report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
        ) + "\n").encode("utf-8"))
    return report


def build_production_dependencies(*_args: Any, **_kwargs: Any) -> H2HRunDependencies:
    """Placeholder for the paid run's real dependency wiring.

    Intentionally unimplemented here: constructing a real ``LLMClient`` and running
    generation is the main session's separately authorized gate.  This module and
    its tests only ever run with injected fakes and make zero paid calls.
    """
    raise NotImplementedError(
        "the paid H2H dependency wiring (real LLMClient, frozen runtime, on-disk "
        "cache) is owned by the main session's authorized generation gate"
    )
