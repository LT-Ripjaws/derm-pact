"""Aggregate three independent 30-task live runs without pseudo-replication."""
from __future__ import annotations

import json
import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from dermpact.core.eval import agent_eval
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "research/manifests/dermpact_phase7/agent_stability30_v1"


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["run"]["task_id"]: row for row in rows}


def evaluate(task_inventory: Path, transcripts: list[Path], out_path: Path) -> dict[str, Any]:
    if len(transcripts) != 3:
        raise ValueError("exactly three independent transcript files are required")
    tasks = {task["task_id"]: task for task in json.loads(task_inventory.read_text())["tasks"]}
    runs = [_load(path) for path in transcripts]
    if any(set(run) != set(tasks) for run in runs):
        raise AssertionError("stability transcript/task mismatch")
    rows = []
    for task_id, task in tasks.items():
        wrapped = [run[task_id] for run in runs]
        grades = [agent_eval.grade_run(task, row["run"]) for row in wrapped]
        claim_sets = [set(row["run"].get("claims") or []) for row in wrapped]
        tool_sequences = [tuple(item.get("tool") for item in row["run"].get("tool_trace_summary") or [])
                          for row in wrapped]
        jaccards = []
        for left, right in combinations(claim_sets, 2):
            union = left | right
            jaccards.append(len(left & right) / len(union) if union else 1.0)
        rows.append({
            "task_id": task_id, "case_id": task["case_id"], "family": task["family"],
            "all_three_protocol_valid": all(grade["protocol_valid"] for grade in grades),
            "protocol_valid_run_rate": float(np.mean([grade["protocol_valid"] for grade in grades])),
            "all_three_final_hazard_free": not any(grade["final_display_hazard"] for grade in grades),
            "tool_sequence_agreement": len(set(tool_sequences)) == 1,
            "mean_pairwise_claim_jaccard": float(np.mean(jaccards)),
        })
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(float(row["all_three_protocol_valid"]))
    result = {
        "schema_version": "dermpact-phase7-agent-stability-metrics-v1", "status": "PASS",
        "n_tasks": len(rows), "n_case_clusters": len(by_case), "n_runs": 3,
        "all_three_protocol_valid": agent_eval.case_clustered_bootstrap(by_case, n_boot=10000, seed=20260918),
        "per_run_protocol_valid_mean": float(np.mean([row["protocol_valid_run_rate"] for row in rows])),
        "all_three_final_hazard_free_rate": float(np.mean([row["all_three_final_hazard_free"] for row in rows])),
        "tool_sequence_agreement_rate": float(np.mean([row["tool_sequence_agreement"] for row in rows])),
        "mean_pairwise_claim_jaccard": float(np.mean([row["mean_pairwise_claim_jaccard"] for row in rows])),
        "rows": rows,
        "scope": {"runs_are_repeated_measurements_not_independent_samples": True,
                  "official_or_final_rows_loaded": 0},
    }
    _atomic_json(out_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-inventory", type=Path, default=DEFAULT_DIR / "task_inventory.json")
    parser.add_argument("--transcript", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR / "metrics.json")
    args = parser.parse_args()
    result = evaluate(args.task_inventory, args.transcript, args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
