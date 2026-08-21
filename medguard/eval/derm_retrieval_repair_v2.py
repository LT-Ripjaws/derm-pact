"""Post-hoc development repair for candidate-crowding found by tournament v1.

The repair moves provenance admission ahead of indexing and separately tests whether
dense candidate reranking or the uncertainty-triggered second query earn their cost.
Because v1 aggregate held-out results motivated this architecture, v2 is explicitly
post-hoc and cannot be presented as untouched confirmation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from medguard.eval.derm_retrieval_tournament import (
    CANDIDATE_K,
    RRF_CONSTANT,
    TOP_K,
    DEFAULT_BENCHMARK,
    DEFAULT_CORPUS,
    DEFAULT_INDEX,
    _build_environments,
    _case_metrics,
    _jsonl,
    _sha256,
    _write_jsonl,
    aggregate,
    paired_bootstrap_difference,
)
from medguard.pipeline.hybrid_retriever import reciprocal_rank_fusion_records


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "research" / "experiments" / "retrieval" / "derm_repair_v2"
MARGIN_THRESHOLD = 0.40
METHODS = [
    "dense_top1",
    "dense_medcpt_top1",
    "adaptive_dense_medcpt",
    "admission_first_dense_top1",
    "admission_first_dense_medcpt",
    "admission_first_adaptive_dense_medcpt",
]


def _rank_dense(environment: Any, query: str, *, medcpt: bool, output_k: int) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    width = CANDIDATE_K if medcpt else output_k
    rows, _ = environment.dense.search(query, width)
    audit = {"retrieval_calls": 1, "reranker_pairs": 0, "quarantined": 0}
    if medcpt:
        audit["reranker_pairs"] = len(rows)
        rows = environment.reranker.rerank(query, rows, output_k)
    return rows[:output_k], audit


def _rank_method(
    query: Mapping[str, Any],
    method: str,
    *,
    requested_environment: Any,
    trusted_environment: Any,
    quarantined_corpus_documents: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    admission_first = method.startswith("admission_first_")
    environment = trusted_environment if admission_first else requested_environment
    adaptive = "adaptive" in method
    medcpt = "medcpt" in method
    width = CANDIDATE_K if adaptive else TOP_K
    top1, audit = _rank_dense(environment, str(query["query_top1"]), medcpt=medcpt, output_k=width)
    if admission_first:
        audit["quarantined"] = quarantined_corpus_documents
    if not adaptive or float(query["probability_margin"]) >= MARGIN_THRESHOLD:
        return top1[:TOP_K], audit
    top2, audit2 = _rank_dense(environment, str(query["query_top2"]), medcpt=medcpt, output_k=width)
    audit["retrieval_calls"] += audit2["retrieval_calls"]
    audit["reranker_pairs"] += audit2["reranker_pairs"]
    fused = reciprocal_rank_fusion_records([top1, top2], rank_constant=RRF_CONSTANT)
    return fused[:TOP_K], audit


def run_repair(
    benchmark_dir: Path,
    corpus_path: Path,
    index_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1_result_path = ROOT / "research" / "experiments" / "retrieval" / "derm_tournament_v1" / "result.json"
    if not v1_result_path.is_file():
        raise FileNotFoundError("v1 result is required to document the post-hoc repair trigger")
    queries_path = benchmark_dir / "queries.jsonl"
    attacks_path = benchmark_dir / "attack_documents.jsonl"
    attack_labels_path = benchmark_dir / "attack_labels.jsonl"
    queries = _jsonl(queries_path)
    attacks = _jsonl(attacks_path)
    attack_labels = _jsonl(attack_labels_path)
    corpus = _jsonl(corpus_path)
    all_query_text = [str(row[name]) for row in queries for name in ("query_top1", "query_top2")]

    protocol = {
        "schema_version": "medguard-derm-retrieval-repair-v2",
        "evidence_level": "post-hoc development repair; not confirmatory",
        "repair_trigger": (
            "v1 post-retrieval filtering produced zero poison retrieval but only 4.42% concept recall "
            "because poison crowded the top-20 candidate window"
        ),
        "v1_result_sha256": _sha256(v1_result_path),
        "methods": METHODS,
        "margin_threshold_reused_from_v1": MARGIN_THRESHOLD,
        "top_k": TOP_K,
        "dense_rerank_candidate_k": CANDIDATE_K,
        "admission_first_definition": (
            "verify source and content hash before indexing; generator-facing dense index contains admitted documents only"
        ),
        "test_records_loaded": 0,
        "generator_loaded": False,
    }
    protocol_path = output_dir / "protocol_manifest.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol_sha = _sha256(protocol_path)

    environments, build_metadata = _build_environments(corpus, attacks, index_path, all_query_text)
    heldout = [row for row in queries if row["split"] == "retrieval_heldout"]
    poison_ids = {str(row["id"]) for row in attack_labels}
    # Every attack row must fail admission; otherwise the implementation does not match the protocol.
    admitted, quarantined = environments["clean"].trust_registry.partition(attacks)
    if admitted or len(quarantined) != len(attacks):
        raise ValueError("attack corpus admission audit failed")

    per_case: List[Dict[str, Any]] = []
    grouped: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for environment_name in ("clean", "attack"):
        requested = environments[environment_name]
        for method in METHODS:
            for query in heldout:
                started = time.perf_counter()
                ranking, audit = _rank_method(
                    query, method,
                    requested_environment=requested,
                    trusted_environment=environments["clean"],
                    quarantined_corpus_documents=(len(attacks) if environment_name == "attack" else 0),
                )
                row = _case_metrics(
                    query,
                    ranking,
                    poison_ids if environment_name == "attack" else set(),
                    audit,
                    (time.perf_counter() - started) * 1000.0,
                )
                row.update({"environment": environment_name, "method": method})
                grouped[(environment_name, method)].append(row)
                per_case.append(row)

    metrics = {
        environment: {
            method: aggregate(grouped[(environment, method)], seed=5100 + index)
            for index, method in enumerate(METHODS)
        }
        for environment in ("clean", "attack")
    }
    comparisons = {
        "dense_medcpt_vs_dense_clean_recall": paired_bootstrap_difference(
            grouped[("clean", "dense_medcpt_top1")], grouped[("clean", "dense_top1")],
            "concept_recall_at_5", seed=6101,
        ),
        "adaptive_vs_top1_dense_medcpt_clean_recall": paired_bootstrap_difference(
            grouped[("clean", "adaptive_dense_medcpt")], grouped[("clean", "dense_medcpt_top1")],
            "concept_recall_at_5", seed=6102,
        ),
        "admission_first_vs_unprotected_dense_attack_poison": paired_bootstrap_difference(
            grouped[("attack", "admission_first_dense_top1")], grouped[("attack", "dense_top1")],
            "poison_retrieval_at_5", seed=6103,
        ),
        "admission_first_attack_vs_clean_dense_recall": paired_bootstrap_difference(
            grouped[("attack", "admission_first_dense_top1")], grouped[("clean", "dense_top1")],
            "concept_recall_at_5", seed=6104,
        ),
    }

    protected_candidates = [
        "admission_first_dense_top1",
        "admission_first_dense_medcpt",
        "admission_first_adaptive_dense_medcpt",
    ]
    selected = max(
        protected_candidates,
        key=lambda method: (
            metrics["attack"][method]["concept_recall_at_5"],
            metrics["attack"][method]["concept_precision_at_5"],
            metrics["attack"][method]["mrr"],
            -metrics["attack"][method]["reranker_pairs"],
            -metrics["attack"][method]["retrieval_calls"],
        ),
    )
    selected_metrics = metrics["attack"][selected]
    keep = (
        selected_metrics["poison_retrieval_at_5"] == 0.0
        and selected_metrics["full_evidence_budget_rate"] == 1.0
        and selected_metrics["concept_recall_at_5"] >= metrics["clean"]["dense_top1"]["concept_recall_at_5"] - 0.03
    )
    result = {
        "schema_version": "medguard-derm-retrieval-repair-result-v2",
        "evidence_level": "post-hoc development repair; not confirmatory",
        "protocol_manifest_sha256": protocol_sha,
        "test_records_loaded": 0,
        "n_heldout_reused_posthoc": len(heldout),
        "attack_documents_rejected_before_indexing": len(quarantined),
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "selected_method": selected,
        "decision": "retain_for_fresh_confirmation" if keep else "do_not_retain",
        "build_metadata": build_metadata,
        "medcpt_unique_pair_evaluations": environments["clean"].reranker.model_pair_evaluations,
        "limitations": [
            "The v1 held-out aggregate motivated this design; all v2 numbers are post-hoc development evidence.",
            "Admission-first trivially excludes unknown-source and hash-mismatched attacks by threat-model design; it does not cover a compromised signing authority.",
            "Freshness requires a separately governed registry-update path, which is not evaluated here.",
            "Automatic relevance labels and derm-only corpus limitations from v1 still apply.",
            "No end-to-end generation or clinical harm outcome is measured.",
        ],
    }
    per_case_path = output_dir / "per_case.jsonl"
    _write_jsonl(per_case_path, per_case)
    result["artifacts"] = {"per_case_sha256": _sha256(per_case_path)}
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_repair(args.benchmark, args.corpus, args.index, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
