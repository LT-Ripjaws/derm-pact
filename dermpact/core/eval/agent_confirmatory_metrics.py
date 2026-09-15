"""Deterministic metrics for the 300-task confirmatory agent evaluation."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dermpact.core.eval import agent_eval
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASKS = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v1/task_inventory_300.json"
DEFAULT_TRANSCRIPTS = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v1/live_qwen3vl/transcripts.jsonl"
DEFAULT_OUTPUT = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v1/metrics.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _required_paths(family: str, paths: Sequence[str]) -> set[str]:
    if family == "evidence_report":
        exact_suffixes = {
            "diagnosis_head.provisional_class", "melanoma_head.probability",
            "reliability.disposition", "reliability.iqa_state", "reliability.ood_state",
            "reliability.view_stability_state", "reliability.selective_abstention",
            "checklist.score_min", "checklist.score_max", "checklist.decidable",
            "agreement.present", "next_step.status", "next_step.action_code",
        }
        return {path for path in paths if any(path.endswith(suffix) for suffix in exact_suffixes)
                or ".criterion_probabilities." in path}
    if family == "explain_deferral":
        suffixes = {"iqa_state", "ood_state", "view_stability_state", "selective_abstention", "disposition"}
        return {path for path in paths if any(path.endswith(suffix) for suffix in suffixes)
                or (".warnings[" in path and path.endswith((".code", ".reason", ".recovery_action")))}
    if family == "case_comparison":
        return {path for path in paths if path == "memory.status"
                or path.endswith("descriptive_reference_label_counts.n_retrieved")
                or ".descriptive_reference_label_counts.counts." in path
                or path.endswith("descriptive_reference_label_counts.diagnostic_effect")
                or ("memory.matches[" in path and path.endswith((".similarity", ".outcome.diagnosis")))
                or path in {"memory.counterexample_status", "memory.reason",
                            "memory.counterexample.similarity", "memory.counterexample.outcome.diagnosis"}}
    return set()


def _fact_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    propositions: dict[str, str] = {}
    for raw in run.get("grounding_contexts") or []:
        try:
            context = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        for proposition in context.get("propositions") or []:
            propositions[str(proposition.get("id"))] = str(proposition.get("path_or_ref"))
    claims = {str(claim) for claim in run.get("claims") or []}
    checker_history = run.get("checker_history") or []
    checker_matched = set(checker_history[-1].get("matched_claims") or []) if checker_history else set()
    valid_claims = claims.intersection(checker_matched or set(propositions))
    required_paths = _required_paths(str(run.get("family")), list(propositions.values()))
    required_ids = {prop_id for prop_id, path in propositions.items() if path in required_paths}
    precision = len(valid_claims) / len(claims) if claims else 0.0
    recall = len(valid_claims.intersection(required_ids)) / len(required_ids) if required_ids else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_submitted_claim_ids": len(claims), "n_valid_claim_ids": len(valid_claims),
        "n_required_facts": len(required_ids),
        "fact_precision": precision, "required_fact_traceability": recall, "fact_f1": f1,
    }


def _clustered(rows: Sequence[Mapping[str, Any]], key: str, seed: int) -> dict[str, Any]:
    by_case: dict[str, list[float]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(float(row[key]))
    return agent_eval.case_clustered_bootstrap(by_case, n_boot=10000, seed=seed)


def evaluate(task_inventory: Path, transcripts: Path, out_path: Path) -> dict[str, Any]:
    task_payload = json.loads(task_inventory.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in task_payload["tasks"]}
    transcript_rows = _load_jsonl(transcripts)
    rows = []
    for wrapped in transcript_rows:
        run = wrapped["run"]
        task = tasks[run["task_id"]]
        grade = agent_eval.grade_run(task, run)
        fact = _fact_metrics(run)
        query_accuracy = float(
            grade["protocol_valid"] and grade["tool_selection_correct"]
            and fact["required_fact_traceability"] == 1.0
        )
        rows.append({
            "task_id": run["task_id"], "case_id": run["case_id"], "family": run["family"],
            "workflow_task_success": float(grade["protocol_valid"]),
            "query_accuracy": query_accuracy,
            "tool_selection_correct": float(grade["tool_selection_correct"]),
            "tool_argument_correct": float(grade["tool_argument_correct"]),
            "tool_correct": float(grade["tool_correct"]),
            "checker_pass": float(grade["checker_ok"]),
            "final_hazard": float(grade["final_display_hazard"]),
            "initial_hazard": float(grade["initial_candidate_hazard"]),
            **fact,
        })
    if len(rows) != len(tasks):
        raise AssertionError(f"transcript/task mismatch: {len(rows)} vs {len(tasks)}")
    metrics = {}
    for offset, key in enumerate(("workflow_task_success", "query_accuracy",
                                  "tool_selection_correct", "tool_argument_correct",
                                  "tool_correct", "checker_pass", "final_hazard",
                                  "initial_hazard", "fact_precision", "required_fact_traceability", "fact_f1")):
        metrics[key] = _clustered(rows, key, 20260908 + offset)
    result = {
        "schema_version": "dermpact-phase7-agent-confirmatory-metrics-v2",
        "status": "PASS", "n_tasks": len(rows), "n_case_clusters": len({row["case_id"] for row in rows}),
        "metrics": metrics,
        "latency_and_process": agent_eval.summarise_arm(tasks, [wrapped["run"] for wrapped in transcript_rows], seed=20260908),
        "rows": rows,
        "scope": {
            "query_accuracy_definition": "protocol valid, correct tool selected, and every prespecified required fact traceable",
            "fact_f1_definition": "checker-matched submitted proposition ids versus prespecified required tool-fact paths",
            "v1_correction": "v1 incorrectly treated locked-record claim ids as invalid when they were absent from the final tool receipt",
            "traceability_definition": "required facts carrying valid proposition ids; not clinician-rated explanation quality",
            "official_or_final_rows_loaded": 0,
        },
    }
    _atomic_json(out_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-inventory", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.task_inventory, args.transcripts, args.out)
    print(json.dumps({"status": result["status"], "n_tasks": result["n_tasks"],
                      "n_case_clusters": result["n_case_clusters"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
