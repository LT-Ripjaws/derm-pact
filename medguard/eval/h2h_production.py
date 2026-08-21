"""Production dependency assembly for the H2H runner + development case/truth loaders.

This fills the runner's ``build_production_dependencies`` stub by reusing the same
frozen runtime, retrievers, concept-link graph, and guideline catalog the validate
scaffold already builds -- so the paid runner consumes exactly the frozen artifacts
the offline pipeline was verified against.

Nothing here makes a paid call at import or assembly time. The real ``LLMClient``
inside the returned ``VLMGenerator`` calls the gateway only when the runner asks it
to generate; the API key is read from the environment at call time (never from a
file by this module). Loading the frozen encoders/indexes is heavy but offline
(HF_HUB_OFFLINE).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard.evidence import load_development_perception_bindings, load_frozen_evidence_runtime
from medguard.evidence.guideline_claims import build_guideline_claim_assets
from medguard.pipeline.generator import VLMGenerator
from medguard.pipeline.guideline_retriever import FrozenGuidelineRetriever
from medguard.pipeline.hybrid_retriever import reciprocal_rank_fusion_records
from medguard.pipeline.llm import LLMClient

from medguard.eval.h2h_evidence_graph import DELTA_R, concept_content, link_concepts_by_embedding
from medguard.eval.matched_b0_b1_validate import QUERY_TERMS
from medguard.eval.matched_full_h2h_run import H2HRunCase, H2HRunDependencies
from medguard.eval.medguard_development_preflight import (
    FrozenDermLIPFeatureEncoder,
    GUIDELINE_CORPUS,
    IMAGE_CANDIDATE_K,
    MANIFEST,
    TEXT_CANDIDATE_K,
    FINAL_K,
    _load_cases,
)


ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_PREDICTIONS = ROOT / "research/experiments/medguard_h2h_v1/perception/dermlip_v1/predictions.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_for(diagnosis: str) -> str:
    return f"dermoscopy {QUERY_TERMS[diagnosis]} diagnosis differential clinical evidence"


def reformulated_query_for(diagnosis: str) -> str:
    return f"{QUERY_TERMS[diagnosis]} management treatment guideline recommendation dermatology"


def load_development_run_cases() -> List[H2HRunCase]:
    """The 160 frozen development lesions as truth-free run cases.

    Perception comes from the frozen truth-free bindings; ordered view ids and local
    image paths come from the manifest (label/benign_malignant are not read here).
    """
    grouped, _manifest_sha = _load_cases()
    bindings = load_development_perception_bindings()
    cases: List[H2HRunCase] = []
    for cluster_id in sorted(grouped):
        rows = grouped[cluster_id]
        binding = bindings[f"lesion:{cluster_id}"]
        cases.append(H2HRunCase(
            case_id=f"lesion:{cluster_id}",
            cluster_id=cluster_id,
            perception=binding,
            ordered_view_case_ids=tuple(row["case_id"] for row in rows),
            ordered_image_paths=tuple(row["image_path"] for row in rows),
        ))
    return cases


def load_development_truth() -> Dict[str, str]:
    """Development labels keyed by cluster id, joined out-of-band by the scorer only."""
    import json
    truth: Dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line) if line.lstrip().startswith("{") else None
        if record is not None and "records" in record:
            for row in record["records"]:
                truth[str(row["cluster_id"])] = str(row["label"])
            break
    if not truth:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for row in data["records"]:
            truth[str(row["cluster_id"])] = str(row["label"])
    return truth


def build_production_dependencies(
    cache_root: str | Path,
    *,
    generator: Optional[VLMGenerator] = None,
    sealed_feature_manifest: Optional[str | Path] = None,
) -> H2HRunDependencies:
    """Assemble the real, frozen-artifact-backed dependencies for the paid runner.

    Pass a fake-client ``generator`` to exercise the whole assembly with zero paid
    calls; omit it and a real ``VLMGenerator(LLMClient())`` is built (the gateway is
    called only when the runner generates).
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np

    feature_encoder = FrozenDermLIPFeatureEncoder()
    if sealed_feature_manifest is not None:
        # Opt-in, and only the SEALED runner passes it. The frozen lookup covers
        # train+calibration, so without this the exemplar channel cannot encode a sealed
        # patient image at all. The exemplar atlas itself stays train-only.
        added = feature_encoder.attach_sealed_features(Path(sealed_feature_manifest))
        print(f"[features] attached {added} sealed query embeddings "
              f"(exemplar atlas remains train-only)")
    runtime = load_frozen_evidence_runtime(image_encoder=feature_encoder)
    dense_backend = runtime.text_retriever.dense

    # Concept-link graph over the frozen text corpus (the A2 selection signal).
    corpus_records = [dict(record) for record in dense_backend._meta]
    corpus_vectors = np.asarray(
        dense_backend._index.reconstruct_n(0, dense_backend._index.ntotal), dtype="float32")
    concept_vectors = dense_backend.embed(
        [concept_content(code, term) for code, term in QUERY_TERMS.items()])
    concept_links = link_concepts_by_embedding(
        records=corpus_records, concept_terms=QUERY_TERMS,
        concept_vectors=concept_vectors, evidence_vectors=corpus_vectors, delta_r=DELTA_R)

    guideline_retriever = FrozenGuidelineRetriever.load(GUIDELINE_CORPUS)
    guideline_catalog, guideline_registry = build_guideline_claim_assets(guideline_retriever.records)

    text_cache: Dict[str, List[Dict[str, Any]]] = {}

    def text_retrieve(query: str, k: int) -> List[Dict[str, Any]]:
        key = f"{query}::{k}"
        if key not in text_cache:
            retrieved, _scores = runtime.text_retriever.search(query, k)
            text_cache[key] = [
                dict(record) for record in retrieved
                if runtime.source_policy.decide(record, "diagnosis").admitted
            ]
        return [dict(record) for record in text_cache[key]]

    def image_retrieve(image_paths: Sequence[str], k: int) -> List[Dict[str, Any]]:
        return runtime.image_retriever.search_records_by_images(list(image_paths), k)

    def baseline_retrieve(arm_id: str, case: H2HRunCase) -> Tuple[List[Dict[str, Any]], bool]:
        """Flat cross-modal fusion baseline (B1/B4/B5/B6 share the same candidate pool).

        A faithful GMM-ATKS / graph / bounded-loop retrieval can be substituted here
        without touching the runner; the fused prefix is the honest matched baseline.
        """
        diagnosis = case.committed_diagnosis
        fused = reciprocal_rank_fusion_records([
            text_retrieve(query_for(diagnosis), TEXT_CANDIDATE_K),
            image_retrieve(case.ordered_image_paths, IMAGE_CANDIDATE_K),
        ])[:FINAL_K]
        return fused, False

    # The typed EvidenceContract JSON is long (diagnosis + exemplar + guideline claims
    # + uncertainties); the 768-token default truncates it mid-object. 1536 fits the
    # contract with headroom while staying bounded for cost.
    gen = generator if generator is not None else VLMGenerator(LLMClient(), max_tokens=1536)

    return H2HRunDependencies(
        generator=gen,
        cache_root=str(cache_root),
        guideline_retriever=guideline_retriever,
        guideline_catalog=guideline_catalog,
        guideline_registry=guideline_registry,
        text_retrieve=text_retrieve,
        image_retrieve=image_retrieve,
        query_for=query_for,
        reformulated_query_for=reformulated_query_for,
        concept_links=concept_links,
        baseline_retrieve=baseline_retrieve,
        corpus_receipt=_sha256(GUIDELINE_CORPUS),
        perception_receipt=_sha256(PERCEPTION_PREDICTIONS),
        perception_artifact=next(iter(load_development_perception_bindings().values())).artifact_sha256,
        code_snapshot=_sha256(Path(__file__)),
        text_candidate_k=TEXT_CANDIDATE_K,
        image_candidate_k=IMAGE_CANDIDATE_K,
        final_k=FINAL_K,
    )
