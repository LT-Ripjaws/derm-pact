"""Run the frozen dermoscopy retrieval and poisoning tournament.

No text generator is loaded.  The only neural text models are the existing
all-mpnet dense encoder and the fixed MedCPT relevance cross-encoder.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from medguard.pipeline.hybrid_retriever import (
    BM25Retriever,
    MedCPTReranker,
    TrustRegistry,
    canonical_document_hash,
    document_id,
    reciprocal_rank_fusion_records,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "research" / "experiments" / "retrieval" / "derm_benchmark_v1"
DEFAULT_CORPUS = ROOT / "medguard" / "data" / "index" / "text_meta.jsonl"
DEFAULT_INDEX = ROOT / "medguard" / "data" / "index" / "text.faiss"
DEFAULT_OUTPUT = ROOT / "research" / "experiments" / "retrieval" / "derm_tournament_v1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
TOP_K = 5
CANDIDATE_K = 20
RRF_CONSTANT = 60
MARGIN_CANDIDATES = [0.05, 0.10, 0.20, 0.30, 0.40]
METHODS = [
    "dense_top1",
    "bm25_top1",
    "hybrid_top1",
    "hybrid_medcpt_top1",
    "adaptive_hybrid_medcpt",
    "trust_adaptive_hybrid_medcpt",
]


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CachedDenseRetriever:
    def __init__(self, index: Any, records: Sequence[Mapping[str, Any]], embedder: Any):
        if int(index.ntotal) != len(records):
            raise ValueError("dense index and metadata length differ")
        self.index = index
        self.records = [dict(record) for record in records]
        self.embedder = embedder
        self.query_vectors: Dict[str, np.ndarray] = {}

    def prime(self, queries: Sequence[str], batch_size: int = 64) -> None:
        unique = sorted({query for query in queries if query and query not in self.query_vectors})
        if not unique:
            return
        vectors = self.embedder.encode(
            unique,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        for query, vector in zip(unique, vectors):
            self.query_vectors[query] = vector.reshape(1, -1)

    def search(self, query: str, k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        if k <= 0 or not query.strip():
            return [], []
        if query not in self.query_vectors:
            self.prime([query])
        width = min(k, len(self.records))
        scores, indices = self.index.search(self.query_vectors[query], width)
        rows: List[Dict[str, Any]] = []
        values: List[float] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            row = dict(self.records[int(index)])
            row["score"] = float(score)
            row["retrieval_channel"] = "dense"
            rows.append(row)
            values.append(float(score))
        return rows, values


class CachedMedCPTReranker:
    def __init__(self, base: MedCPTReranker):
        self.base = base
        self.cache: Dict[Tuple[str, str, str], float] = {}
        self.model_pair_evaluations = 0

    @staticmethod
    def _key(query: str, row: Mapping[str, Any]) -> Tuple[str, str, str]:
        return query, document_id(row), canonical_document_hash(row)

    def rerank(self, query: str, candidates: Sequence[Mapping[str, Any]], k: int) -> List[Dict[str, Any]]:
        missing = [row for row in candidates if self._key(query, row) not in self.cache]
        if missing:
            scored = self.base.rerank(query, missing, len(missing))
            self.model_pair_evaluations += len(missing)
            for row in scored:
                self.cache[self._key(query, row)] = float(row["medcpt_score"])
        output: List[Dict[str, Any]] = []
        for candidate in candidates:
            row = dict(candidate)
            row["pre_rerank_score"] = float(row.get("score", 0.0))
            row["score"] = self.cache[self._key(query, row)]
            row["medcpt_score"] = row["score"]
            output.append(row)
        output.sort(key=lambda row: (-float(row["score"]), document_id(row)))
        return output[: min(k, len(output))]


class RetrievalEnvironment:
    def __init__(
        self,
        dense: CachedDenseRetriever,
        sparse: BM25Retriever,
        reranker: CachedMedCPTReranker,
        trust_registry: TrustRegistry,
    ):
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.trust_registry = trust_registry

    def hybrid_candidates(self, query: str, *, trusted: bool = False) -> tuple[List[Dict[str, Any]], int]:
        dense, _ = self.dense.search(query, CANDIDATE_K)
        sparse, _ = self.sparse.search(query, CANDIDATE_K)
        candidates = reciprocal_rank_fusion_records(
            [dense, sparse], rank_constant=RRF_CONSTANT
        )
        quarantined = 0
        if trusted:
            candidates, rejected = self.trust_registry.partition(candidates)
            quarantined = len(rejected)
        return candidates[:CANDIDATE_K], quarantined

    def rank_single(self, query: str, method: str, *, output_k: int = TOP_K) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        audit = {"retrieval_calls": 1, "reranker_pairs": 0, "quarantined": 0}
        if method == "dense_top1":
            rows, _ = self.dense.search(query, output_k)
            return rows, audit
        if method == "bm25_top1":
            rows, _ = self.sparse.search(query, output_k)
            return rows, audit
        trusted = method == "trust_adaptive_hybrid_medcpt"
        candidates, quarantined = self.hybrid_candidates(query, trusted=trusted)
        audit["quarantined"] = quarantined
        if method == "hybrid_top1":
            return candidates[:output_k], audit
        audit["reranker_pairs"] = len(candidates)
        return self.reranker.rerank(query, candidates, output_k), audit


def _adaptive_rank(
    environment: RetrievalEnvironment,
    query: Mapping[str, Any],
    method: str,
    margin_threshold: float,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    adaptive = method in {"adaptive_hybrid_medcpt", "trust_adaptive_hybrid_medcpt"}
    width = CANDIDATE_K if adaptive else TOP_K
    top1, audit = environment.rank_single(str(query["query_top1"]), method, output_k=width)
    if not adaptive or float(query["probability_margin"]) >= margin_threshold:
        return top1[:TOP_K], audit
    top2, audit2 = environment.rank_single(str(query["query_top2"]), method, output_k=width)
    for key in audit:
        audit[key] += audit2[key]
    fused = reciprocal_rank_fusion_records([top1, top2], rank_constant=RRF_CONSTANT)
    return fused[:TOP_K], audit


def _case_metrics(
    query: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    poison_ids: set[str],
    audit: Mapping[str, int],
    elapsed_ms: float,
) -> Dict[str, Any]:
    ids = [document_id(row) for row in rows]
    concept = set(map(str, query["relevant_concept_document_ids"]))
    management = set(map(str, query["relevant_management_document_ids"]))
    first_relevant = next((index for index, value in enumerate(ids, start=1) if value in concept), None)
    poison_count = sum(value in poison_ids for value in ids)
    return {
        "case_id": str(query["case_id"]),
        "cluster_id": str(query["cluster_id"]),
        "label": str(query["ground_truth_label"]),
        "concept_recall_at_5": float(any(value in concept for value in ids)),
        "management_recall_at_5": float(any(value in management for value in ids)),
        "concept_precision_at_5": sum(value in concept for value in ids) / max(len(ids), 1),
        "mrr": 0.0 if first_relevant is None else 1.0 / first_relevant,
        "poison_retrieval_at_5": float(poison_count > 0),
        "poison_fraction_at_5": poison_count / max(len(ids), 1),
        "returned_count": len(ids),
        "retrieval_calls": int(audit["retrieval_calls"]),
        "reranker_pairs": int(audit["reranker_pairs"]),
        "quarantined": int(audit["quarantined"]),
        "elapsed_ms_shared_cache": elapsed_ms,
        "retrieved_ids": ids,
    }


def _bootstrap_mean(values: Sequence[float], *, seed: int, iterations: int = 2000) -> List[float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(iterations, len(array)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def aggregate(rows: Sequence[Mapping[str, Any]], *, seed: int = 1701) -> Dict[str, Any]:
    metric_names = [
        "concept_recall_at_5", "management_recall_at_5", "concept_precision_at_5", "mrr",
        "poison_retrieval_at_5", "poison_fraction_at_5", "retrieval_calls", "reranker_pairs",
        "quarantined", "elapsed_ms_shared_cache",
    ]
    result: Dict[str, Any] = {"n_cases": len(rows)}
    for offset, name in enumerate(metric_names):
        values = [float(row[name]) for row in rows]
        result[name] = float(np.mean(values)) if values else float("nan")
        result[name + "_ci95"] = _bootstrap_mean(values, seed=seed + offset)
    result["full_evidence_budget_rate"] = float(
        np.mean([int(row["returned_count"]) == TOP_K for row in rows])
    ) if rows else float("nan")
    return result


def paired_bootstrap_difference(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    seed: int = 991,
    iterations: int = 5000,
) -> Dict[str, Any]:
    left_by_id = {str(row["cluster_id"]): float(row[metric]) for row in left}
    right_by_id = {str(row["cluster_id"]): float(row[metric]) for row in right}
    ids = sorted(set(left_by_id) & set(right_by_id))
    differences = np.asarray([left_by_id[key] - right_by_id[key] for key in ids], dtype=float)
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), size=(iterations, len(differences)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "n_pairs": len(ids),
        "estimate": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "left_minus_right": metric,
    }


def select_margin_threshold(
    queries: Sequence[Mapping[str, Any]],
    environment: RetrievalEnvironment,
) -> tuple[float, List[Dict[str, Any]]]:
    trials: List[Dict[str, Any]] = []
    cached: Dict[str, tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for query in queries:
        top1, _ = environment.rank_single(str(query["query_top1"]), "hybrid_top1", output_k=CANDIDATE_K)
        top2, _ = environment.rank_single(str(query["query_top2"]), "hybrid_top1", output_k=CANDIDATE_K)
        cached[str(query["case_id"])] = (top1, top2)
    for threshold in MARGIN_CANDIDATES:
        rows: List[Dict[str, Any]] = []
        extra = 0
        for query in queries:
            top1, top2 = cached[str(query["case_id"])]
            use_second = float(query["probability_margin"]) < threshold
            ranking = reciprocal_rank_fusion_records([top1, top2], rank_constant=RRF_CONSTANT) if use_second else top1
            extra += int(use_second)
            rows.append(_case_metrics(query, ranking[:TOP_K], set(), {
                "retrieval_calls": 1 + int(use_second), "reranker_pairs": 0, "quarantined": 0,
            }, 0.0))
        metrics = aggregate(rows, seed=207 + int(threshold * 100))
        extra_rate = extra / max(len(queries), 1)
        trials.append({
            "threshold": threshold,
            "concept_recall_at_5": metrics["concept_recall_at_5"],
            "management_recall_at_5": metrics["management_recall_at_5"],
            "extra_query_rate": extra_rate,
            "objective": metrics["concept_recall_at_5"] - 0.01 * extra_rate,
        })
    best = max(trials, key=lambda row: (row["objective"], row["management_recall_at_5"], -row["threshold"]))
    return float(best["threshold"]), trials


def _build_environments(
    corpus: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    index_path: Path,
    all_queries: Sequence[str],
) -> tuple[Dict[str, RetrievalEnvironment], Dict[str, Any]]:
    import faiss
    from sentence_transformers import SentenceTransformer

    clean_index = faiss.read_index(str(index_path))
    if int(clean_index.ntotal) != len(corpus):
        raise ValueError("frozen FAISS index does not align with clean corpus")
    embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    clean_vectors = clean_index.reconstruct_n(0, int(clean_index.ntotal)).astype("float32")
    attack_vectors = embedder.encode(
        [str(row.get("text") or "") for row in attacks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    attack_index = faiss.IndexFlatIP(clean_vectors.shape[1])
    attack_index.add(np.vstack([clean_vectors, attack_vectors]))

    dense_clean = CachedDenseRetriever(clean_index, corpus, embedder)
    dense_attack = CachedDenseRetriever(attack_index, [*corpus, *attacks], embedder)
    dense_clean.prime(all_queries)
    dense_attack.query_vectors = dense_clean.query_vectors
    reranker = CachedMedCPTReranker(
        MedCPTReranker(RERANKER_MODEL, batch_size=16, offline=True, max_length=512)
    )
    registry = TrustRegistry.from_records(corpus)
    environments = {
        "clean": RetrievalEnvironment(dense_clean, BM25Retriever(corpus), reranker, registry),
        "attack": RetrievalEnvironment(dense_attack, BM25Retriever([*corpus, *attacks]), reranker, registry),
    }
    metadata = {
        "clean_index_vectors": int(clean_index.ntotal),
        "attack_index_vectors": int(attack_index.ntotal),
        "embedding_dim": int(clean_vectors.shape[1]),
        "attack_vectors_encoded": len(attacks),
    }
    return environments, metadata


def run_tournament(
    benchmark_dir: Path,
    corpus_path: Path,
    index_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = benchmark_dir / "receipt.json"
    queries_path = benchmark_dir / "queries.jsonl"
    attacks_path = benchmark_dir / "attack_documents.jsonl"
    attack_labels_path = benchmark_dir / "attack_labels.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("test_records_loaded") != 0:
        raise ValueError("retrieval benchmark receipt is not development-only")
    expected = receipt["artifacts"]
    actual = {
        "queries_sha256": _sha256(queries_path),
        "attack_documents_sha256": _sha256(attacks_path),
        "attack_labels_sha256": _sha256(attack_labels_path),
    }
    for key, digest in actual.items():
        if expected.get(key) != digest:
            raise ValueError(f"frozen benchmark hash mismatch for {key}")

    queries = _jsonl(queries_path)
    attacks = _jsonl(attacks_path)
    attack_labels = _jsonl(attack_labels_path)
    corpus = _jsonl(corpus_path)
    all_query_text = [str(row[name]) for row in queries for name in ("query_top1", "query_top2")]

    protocol = {
        "schema_version": "medguard-derm-retrieval-tournament-v1",
        "created_before_scoring": True,
        "benchmark_receipt_sha256": _sha256(receipt_path),
        "corpus_sha256": _sha256(corpus_path),
        "index_sha256": _sha256(index_path),
        "methods": METHODS,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_rank_constant": RRF_CONSTANT,
        "margin_candidates": MARGIN_CANDIDATES,
        "reranker_model": RERANKER_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "generator_loaded": False,
        "test_records_loaded": 0,
    }
    protocol_path = output_dir / "protocol_manifest.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol_sha = _sha256(protocol_path)

    start = time.perf_counter()
    environments, build_metadata = _build_environments(corpus, attacks, index_path, all_query_text)
    build_seconds = time.perf_counter() - start
    tune = [row for row in queries if row["split"] == "retrieval_tune"]
    heldout = [row for row in queries if row["split"] == "retrieval_heldout"]
    margin_threshold, margin_trials = select_margin_threshold(tune, environments["clean"])
    poison_ids = {str(row["id"]) for row in attack_labels}

    per_case: List[Dict[str, Any]] = []
    grouped: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for environment_name, environment in environments.items():
        active_poison_ids = poison_ids if environment_name == "attack" else set()
        for method in METHODS:
            for query in heldout:
                case_start = time.perf_counter()
                ranking, audit = _adaptive_rank(environment, query, method, margin_threshold)
                row = _case_metrics(
                    query, ranking, active_poison_ids, audit,
                    (time.perf_counter() - case_start) * 1000.0,
                )
                row["environment"] = environment_name
                row["method"] = method
                grouped[(environment_name, method)].append(row)
                per_case.append(row)

    metrics = {
        environment: {
            method: aggregate(grouped[(environment, method)], seed=3000 + index)
            for index, method in enumerate(METHODS)
        }
        for environment in ("clean", "attack")
    }
    trust_method = "trust_adaptive_hybrid_medcpt"
    nontrust_method = "adaptive_hybrid_medcpt"
    comparisons = {
        "trust_vs_nontrust_attack_poison_rate": paired_bootstrap_difference(
            grouped[("attack", trust_method)], grouped[("attack", nontrust_method)],
            "poison_retrieval_at_5", seed=8801,
        ),
        "trust_vs_nontrust_attack_concept_recall": paired_bootstrap_difference(
            grouped[("attack", trust_method)], grouped[("attack", nontrust_method)],
            "concept_recall_at_5", seed=8802,
        ),
        "adaptive_vs_top1_clean_concept_recall": paired_bootstrap_difference(
            grouped[("clean", "adaptive_hybrid_medcpt")],
            grouped[("clean", "hybrid_medcpt_top1")],
            "concept_recall_at_5", seed=8803,
        ),
        "medcpt_vs_hybrid_clean_concept_recall": paired_bootstrap_difference(
            grouped[("clean", "hybrid_medcpt_top1")],
            grouped[("clean", "hybrid_top1")],
            "concept_recall_at_5", seed=8804,
        ),
    }
    trust_clean_delta = (
        metrics["clean"][trust_method]["concept_recall_at_5"]
        - metrics["clean"][nontrust_method]["concept_recall_at_5"]
    )
    poison_delta = comparisons["trust_vs_nontrust_attack_poison_rate"]
    keep_trust = trust_clean_delta >= -0.03 and poison_delta["estimate"] < 0 and poison_delta["ci95"][1] < 0
    result = {
        "schema_version": "medguard-derm-retrieval-tournament-result-v1",
        "evidence_level": "development/calibration screening; not final-test evidence",
        "protocol_manifest_sha256": protocol_sha,
        "test_records_loaded": 0,
        "n_tune": len(tune),
        "n_heldout": len(heldout),
        "selected_adaptive_margin_threshold": margin_threshold,
        "margin_tuning_trials": margin_trials,
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "build_metadata": build_metadata,
        "build_seconds": build_seconds,
        "medcpt_unique_pair_evaluations": environments["clean"].reranker.model_pair_evaluations,
        "decision": "keep_trust_adaptive_retrieval" if keep_trust else "do_not_keep_full_trust_variant",
        "decision_rule": (
            "Keep only if clean concept recall is within 3 points of non-trust adaptive retrieval and "
            "the paired poison-retrieval reduction is strictly below zero at the 95% interval."
        ),
        "limitations": [
            "Automatic concept-string relevance may favor lexical retrieval and is not clinician adjudication.",
            "Attacks are deterministic query-copy insertion, metadata-spoof, and false-consensus stresses, not LLM-generated final attacks.",
            "Observed wall time uses a shared MedCPT cache and is not a fair cold-latency comparison; calls and reranked pairs are the compute audit.",
            "This experiment evaluates retrieval and poisoning containment, not end-to-end answer harm.",
            "The corpus is dermoscopy-only; no CXR retrieval claim is made.",
        ],
    }
    per_case_path = output_dir / "per_case.jsonl"
    result_path = output_dir / "result.json"
    _write_jsonl(per_case_path, per_case)
    result["artifacts"] = {"per_case_sha256": _sha256(per_case_path)}
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_tournament(args.benchmark, args.corpus, args.index, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
