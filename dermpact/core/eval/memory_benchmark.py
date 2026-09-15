"""Development-only leave-one-out benchmark for Verified Case Memory v2."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dermpact.core import memory
from dermpact.core.eval.amendment_benchmark_v2 import case_clustered_bootstrap
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase6/memory_benchmark_v1"


def run(bank_dir: Path = memory.DEFAULT_BANK_DIR) -> Mapping[str, Any]:
    bank = memory.MemoryBank.load(bank_dir)
    policy = memory.ComparisonPolicy.from_manifest(bank_dir)
    effective = tuple(bank.effective())
    rows = []
    for target in effective:
        query = memory._episode_to_query(target)
        started = time.perf_counter()
        result = memory.compare(query, bank, policy)
        latency_ms = (time.perf_counter() - started) * 1000.0
        target_label = memory.harmonise_diagnosis(target.outcome.get("diagnosis"))
        candidate_relevant = [
            episode for episode in effective
            if episode.episode_id != target.episode_id
            and episode.image_sha256 != target.image_sha256
            and memory.harmonise_diagnosis(episode.outcome.get("diagnosis")) == target_label
        ]
        matched_relevant = sum(
            memory.harmonise_diagnosis(match["outcome"].get("diagnosis")) == target_label
            for match in result.matches
        )
        precision = matched_relevant / len(result.matches) if result.matches else None
        recall = matched_relevant / len(candidate_relevant) if candidate_relevant else None
        rows.append({
            "case_id": target.case_id, "status": result.status, "n_matches": len(result.matches),
            "diagnosis_label_precision_at_k": precision, "diagnosis_label_recall_at_k": recall,
            "counterexample_status": result.counterexample_status,
            "identity_exclusion": result.receipt.get("identity_exclusion"), "latency_ms": latency_ms,
        })

    def metric_ci(key: str, seed: int):
        values = {row["case_id"]: [row[key]] for row in rows if row[key] is not None}
        return case_clustered_bootstrap(values, n_boot=10000, seed=seed)

    return {
        "schema_version": "dermpact-phase6-memory-benchmark-v1",
        "n_bank_episodes": len(bank), "n_queries": len(rows), "k": policy.k,
        "min_similarity": policy.min_similarity,
        "counterexample_min_similarity": policy.counterexample_min_similarity,
        "matches_per_query": {
            "mean": float(np.mean([row["n_matches"] for row in rows])),
            "median": float(np.median([row["n_matches"] for row in rows])),
            "definition": "returned match count; status_ok_rate is nonempty match coverage",
        },
        "status_ok_rate": case_clustered_bootstrap(
            {row["case_id"]: [int(row["status"] == memory.STATUS_OK)] for row in rows},
            n_boot=10000, seed=2026090702,
        ),
        "diagnosis_label_precision_at_k": metric_ci("diagnosis_label_precision_at_k", 2026090703),
        "diagnosis_label_recall_at_k": metric_ci("diagnosis_label_recall_at_k", 2026090704),
        "counterexample_available_rate": case_clustered_bootstrap(
            {row["case_id"]: [int(row["counterexample_status"] == memory.COUNTEREXAMPLE_AVAILABLE)] for row in rows},
            n_boot=10000, seed=2026090705,
        ),
        "retrieval_latency_ms": {"mean": float(np.mean([row["latency_ms"] for row in rows])),
                                 "median": float(np.median([row["latency_ms"] for row in rows])),
                                 "p95": float(np.percentile([row["latency_ms"] for row in rows], 95))},
        "rows": rows,
        "interpretation": ("Diagnosis-label precision/recall are descriptive dataset-label proxies only. "
                           "They do not measure clinical relevance, diagnosis uplift or clinician usefulness."),
        "correction_pattern_retrieval": "not_evaluable: current bank has no confirmed amendment patterns",
        "official_or_final_rows_loaded": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, default=memory.DEFAULT_BANK_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run(args.bank_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "results.json", {k: v for k, v in result.items() if k != "rows"})
    _atomic_write(args.out_dir / "rows.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["rows"]))
    print(json.dumps({"status": "PASS", "n_queries": result["n_queries"],
                      "status_ok_rate": result["status_ok_rate"]["estimate"],
                      "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
