"""Offline BM25 sanity benchmark for the licensed guideline subset.

This is not the final paper benchmark: its engineering-authored relevance sets
cover only the eight condition-question cells currently backed by indexed
recommendation text. It exists to make corpus/index wiring testable while the
remaining guideline permissions and independent gold review are pending.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from medguard.pipeline.hybrid_retriever import BM25Retriever


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "medguard" / "data" / "corpus" / "derm_guideline_recommendations_v1.jsonl"
OUTPUT_DIR = ROOT / "research" / "experiments" / "retrieval" / "guideline_subset_benchmark_v1"

QUERIES: Sequence[Dict[str, Any]] = (
    {
        "query_id": "bcc-clinical-001",
        "condition": "basal_cell_carcinoma",
        "question_type": "clinical_visual_assessment",
        "query": "How should suspected basal cell carcinoma be clinically assessed and which non-invasive diagnostic methods improve accuracy?",
        "gold_ids": ["eado:pmid:37604067:box-1", "eado:pmid:37604067:box-2", "eado:pmid:37604067:box-3"],
    },
    {
        "query_id": "bcc-differential-001",
        "condition": "basal_cell_carcinoma",
        "question_type": "differential_diagnosis",
        "query": "Which non-invasive techniques help distinguish a difficult-to-recognize basal cell carcinoma?",
        "gold_ids": ["eado:pmid:37604067:box-2", "eado:pmid:37604067:box-3"],
    },
    {
        "query_id": "bcc-biopsy-001",
        "condition": "basal_cell_carcinoma",
        "question_type": "biopsy_histopathology",
        "query": "When is histopathological confirmation mandatory for basal cell carcinoma?",
        "gold_ids": ["eado:pmid:37604067:box-4"],
    },
    {
        "query_id": "bcc-management-001",
        "condition": "basal_cell_carcinoma",
        "question_type": "initial_local_management",
        "query": "What is the standard initial local treatment for basal cell carcinoma?",
        "gold_ids": ["eado:pmid:37604067:box-5"],
    },
    {
        "query_id": "bcc-followup-001",
        "condition": "basal_cell_carcinoma",
        "question_type": "follow_up_surveillance",
        "query": "At what interval should a patient with basal cell carcinoma be followed according to risk?",
        "gold_ids": ["eado:pmid:37604067:box-22"],
    },
    {
        "query_id": "mel-biopsy-001",
        "condition": "melanoma",
        "question_type": "biopsy_histopathology",
        "query": "How should a lesion suspected to be melanoma be excised or biopsied for histological diagnosis?",
        "gold_ids": ["eado:pmid:39709737:recommendation-1"],
    },
    {
        "query_id": "mel-referral-001",
        "condition": "melanoma",
        "question_type": "referral_triage",
        "query": "When should sentinel lymph node biopsy be offered for melanoma staging and treatment decisions?",
        "gold_ids": ["eado:pmid:39709737:recommendation-6"],
    },
    {
        "query_id": "mel-management-001",
        "condition": "melanoma",
        "question_type": "initial_local_management",
        "query": "What initial excision and local surgical management is recommended for primary melanoma, including special anatomic sites?",
        "gold_ids": [
            "eado:pmid:39709737:recommendation-1",
            "eado:pmid:39709737:recommendation-2",
            "eado:pmid:39709737:recommendation-4",
            "eado:pmid:39709737:recommendation-5",
        ],
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(dict(row)) for row in rows)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dcg(relevance: Sequence[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def score_query(gold_ids: Sequence[str], retrieved_ids: Sequence[str], k: int) -> Dict[str, float]:
    if not gold_ids or len(gold_ids) != len(set(gold_ids)):
        raise ValueError("gold_ids must be non-empty and unique")
    gold = set(gold_ids)
    ranked = list(retrieved_ids[:k])
    relevance = [int(item in gold) for item in ranked]
    relevant_count = sum(relevance)
    ideal = [1] * min(len(gold), k)
    first_rank = next((index + 1 for index, item in enumerate(ranked) if item in gold), None)
    return {
        f"precision@{k}": relevant_count / k,
        f"recall@{k}": relevant_count / len(gold),
        f"ndcg@{k}": _dcg(relevance) / _dcg(ideal) if ideal else 0.0,
        f"mrr@{k}": 1.0 / first_rank if first_rank else 0.0,
    }


def _write(path: Path, payload: bytes) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_name(path.name + ".sha256").write_bytes((digest + "\n").encode("ascii"))
    return {"path": _relative(path), "sha256": digest, "bytes": len(payload)}


def run_benchmark(output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    corpus = _load_jsonl(CORPUS_PATH)
    corpus_ids = {str(row["document_id"]) for row in corpus}
    query_ids = [str(row["query_id"]) for row in QUERIES]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate benchmark query id")
    for query in QUERIES:
        missing = set(query["gold_ids"]) - corpus_ids
        if missing:
            raise ValueError(f"query {query['query_id']} has missing gold records: {sorted(missing)}")
    retriever = BM25Retriever(corpus)
    per_query: List[Dict[str, Any]] = []
    for query in QUERIES:
        rows, scores = retriever.search(str(query["query"]), k=5)
        retrieved_ids = [str(row["document_id"]) for row in rows]
        metrics: Dict[str, float] = {}
        for k in (1, 3, 5):
            metrics.update(score_query(query["gold_ids"], retrieved_ids, k))
        top_source = str(rows[0]["source_document_id"]) if rows else ""
        gold_sources = {item.rsplit(":", 1)[0] for item in query["gold_ids"]}
        metrics["source_version_accuracy@1"] = float(bool(rows) and top_source in gold_sources and bool(rows[0].get("source_version")))
        per_query.append({
            **query,
            "retrieved_ids": retrieved_ids,
            "scores": scores,
            "metrics": metrics,
        })

    metric_names = sorted(per_query[0]["metrics"])
    aggregate = {name: mean(row["metrics"][name] for row in per_query) for name in metric_names}
    strata: Dict[str, Dict[str, Dict[str, float]]] = {}
    for field in ("condition", "question_type"):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in per_query:
            grouped[str(row[field])].append(row)
        strata[field] = {
            value: {name: mean(row["metrics"][name] for row in rows) for name in metric_names}
            for value, rows in sorted(grouped.items())
        }
    result = {
        "schema_version": "medguard-guideline-subset-retrieval-result-v1",
        "protocol": "deterministic-bm25-engineering-sanity",
        "query_count": len(QUERIES),
        "covered_condition_question_cells": len({(row["condition"], row["question_type"]) for row in QUERIES}),
        "total_scoped_condition_question_cells": 29,
        "aggregate": aggregate,
        "strata": strata,
        "claim_guard": "Engineering-authored subset benchmark; not independent clinical gold and not comprehensive retrieval evidence.",
        "execution_guards": {"model_generation": False, "network_access": False, "sealed_test_access": False},
    }
    query_artifact = _write(output_dir / "queries.jsonl", _jsonl_bytes(QUERIES))
    cases_artifact = _write(output_dir / "per_query.jsonl", _jsonl_bytes(per_query))
    result_artifact = _write(output_dir / "result.json", _json_bytes(result))
    receipt = {
        "schema_version": "medguard-guideline-subset-retrieval-receipt-v1",
        "corpus": {"path": _relative(CORPUS_PATH), "sha256": _sha256(CORPUS_PATH)},
        "builder": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "artifacts": [query_artifact, cases_artifact, result_artifact],
        "claim_guard": result["claim_guard"],
        "execution_guards": result["execution_guards"],
    }
    _write(output_dir / "receipt.json", _json_bytes(receipt))
    return result


def verify_benchmark(output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    receipt_path = output_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for row in [receipt["corpus"], receipt["builder"], *receipt["artifacts"]]:
        if _sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"guideline retrieval benchmark hash mismatch: {row['path']}")
    if _sha256(receipt_path) != receipt_path.with_name("receipt.json.sha256").read_text(encoding="ascii").strip():
        raise ValueError("guideline retrieval benchmark receipt sidecar mismatch")
    return receipt


def main() -> None:
    result = run_benchmark()
    verify_benchmark()
    print(json.dumps({
        "query_count": result["query_count"],
        "covered_cells": result["covered_condition_question_cells"],
        "recall@5": result["aggregate"]["recall@5"],
        "mrr@5": result["aggregate"]["mrr@5"],
        "claim_guard": result["claim_guard"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
