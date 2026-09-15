"""100-case matched live ablation for comparison memory and counterexamples."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from dermpact.core.eval import agent_eval
from dermpact.core.eval.agent_live_eval import PRIMARY_LIVE_MODEL, live_preflight, load_frozen_tasks
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v2_grounded_reads/task_inventory_300.json"
CASES = ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1"
FULL = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v2_grounded_reads/live_qwen3vl/transcripts.jsonl"
OUT = ROOT / "research/manifests/dermpact_phase7/agent_memory_ablation100_v1"


def _mutate(arm: str):
    if arm == "without_comparison_memory":
        def mutate(host):
            host.memory_bank = None
            host.memory_policy = None
        return mutate
    if arm == "without_memory_counterexample":
        def mutate(host):
            host.memory_policy = dataclasses.replace(host.memory_policy, counterexample_min_similarity=1.0)
        return mutate
    raise ValueError(arm)


def _load_full() -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in FULL.read_text().splitlines() if line.strip()]
    return {row["run"]["task_id"]: row for row in rows}


def _memory_data(row: dict[str, Any]) -> dict[str, Any]:
    for raw in row["run"].get("grounding_contexts") or []:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if payload.get("tool") == "retrieve_case_comparisons":
            return dict(payload.get("data") or {})
    return {}


def run(out_dir: Path = OUT, model: str = PRIMARY_LIVE_MODEL) -> dict[str, Any]:
    ready, reason = live_preflight()
    if not ready:
        raise RuntimeError(reason)
    tasks = [task for task in load_frozen_tasks(TASKS) if task["family"] == "case_comparison"]
    if len(tasks) != 100 or len({task["case_id"] for task in tasks}) != 100:
        raise AssertionError("expected one comparison task per case")
    full = _load_full()
    rows = [{**full[task["task_id"]], "run": {**full[task["task_id"]]["run"], "arm": "full_bounded_agent"}}
            for task in tasks]
    out_dir = Path(out_dir)
    for arm in ("without_comparison_memory", "without_memory_counterexample"):
        for index, task in enumerate(tasks, start=1):
            key = hashlib.sha256(json.dumps({"arm": arm, "task": task, "model": model}, sort_keys=True).encode()).hexdigest()
            path = out_dir / "task_cache" / arm / f"{task['task_id']}.json"
            if path.is_file():
                cached = json.loads(path.read_text())
                if cached.get("cache_key") == key:
                    rows.append(cached["row"])
                    continue
            print(f"[{arm} {index}/100] {task['task_id']}", flush=True)
            result = agent_eval._run_task_through_graph(
                task, CASES, arm_name=arm, request=task["message"], host_mode="live",
                live_model=model, host_mutator=_mutate(arm),
            )
            row = {"run": result, "grading": agent_eval.grade_run(task, result)}
            _atomic_json(path, {"cache_key": key, "row": row})
            rows.append(row)
    summaries = {}
    task_map = {task["task_id"]: task for task in tasks}
    for arm in ("full_bounded_agent", "without_comparison_memory", "without_memory_counterexample"):
        arm_rows = [row for row in rows if row["run"]["arm"] == arm]
        summary = agent_eval.summarise_arm(task_map, [row["run"] for row in arm_rows], seed=20261500)
        memory_data = [_memory_data(row) for row in arm_rows]
        summary["memory_service_available_rate"] = sum(
            "comparison memory is unavailable" not in row["run"].get("text", "").lower() for row in arm_rows
        ) / len(arm_rows)
        summary["comparison_available_rate"] = sum(data.get("status") == "ok" for data in memory_data) / len(arm_rows)
        summary["counterexample_candidate_available_rate"] = sum(data.get("counterexample") is not None for data in memory_data) / len(arm_rows)
        summary["counterexample_available_rate"] = sum(
            data.get("status") == "ok" and data.get("counterexample") is not None for data in memory_data
        ) / len(arm_rows)
        summary["counterexample_display_rate"] = sum(
            "discordant reference case" in row["run"].get("text", "").lower() for row in arm_rows
        ) / len(arm_rows)
        summaries[arm] = summary
    transcript = out_dir / "transcripts.jsonl"
    _atomic_write(transcript, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    result = {
        "schema_version": "dermpact-phase7-memory-ablation100-v1", "status": "PASS",
        "n_cases": 100, "summaries": summaries,
        "sources": {"tasks": _sha256_file(TASKS), "full": _sha256_file(FULL)},
        "transcripts_sha256": _sha256_file(transcript),
        "scope": {"official_or_final_rows_loaded": 0, "memory_never_rewrites_prediction": True},
    }
    _atomic_json(out_dir / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
