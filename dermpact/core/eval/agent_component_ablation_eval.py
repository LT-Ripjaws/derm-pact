"""Matched live component knockouts on only the tasks that require each component."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dermpact.core.eval import agent_eval
from dermpact.core.eval.agent_live_eval import PRIMARY_LIVE_MODEL, TASK_INVENTORY, live_preflight, load_frozen_tasks
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file
from dermpact.core.prototype.host import HostServices
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/manifests/dermpact_phase7/agent_component_ablations_v1"


def _mutator(arm: str) -> Callable[[HostServices], None] | None:
    if arm == "full_bounded_agent":
        return None
    if arm == "without_comparison_memory":
        def mutate(host: HostServices) -> None:
            host.memory_bank = None
            host.memory_policy = None
        return mutate
    if arm == "without_memory_counterexample":
        def mutate(host: HostServices) -> None:
            if host.memory_policy is not None:
                host.memory_policy = dataclasses.replace(host.memory_policy, counterexample_min_similarity=1.0)
        return mutate
    if arm == "without_measured_xai":
        return lambda host: setattr(host, "jobs", None)
    if arm == "without_persistent_restart_state":
        return None
    raise ValueError(arm)


def _selected(tasks: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    comparison = [task for task in tasks if task["family"] == "case_comparison"]
    xai = [task for task in tasks if task["family"] in {"region_sensitivity", "clinical_photo"}]
    restart = [task for task in tasks if task["family"] == "resume_after_restart"]
    if not (len(comparison) == 7 and len(xai) == 5 and len(restart) == 2):
        raise AssertionError("frozen relevant-task subsets changed")
    return {
        "without_comparison_memory": comparison,
        "without_memory_counterexample": comparison,
        "without_measured_xai": xai,
        "without_persistent_restart_state": restart,
    }


def _run_one(task: Mapping[str, Any], arm: str, model: str) -> dict[str, Any]:
    result = agent_eval._run_task_through_graph(
        task, inv.DEFAULT_INVENTORY_DIR, arm_name=arm, request=task["message"],
        host_mode="live", live_model=model, host_mutator=_mutator(arm),
        disable_persistence_on_restart=(arm == "without_persistent_restart_state"),
    )
    return {"run": result, "grading": agent_eval.grade_run(task, result)}


def run(out_dir: Path = OUT, model: str = PRIMARY_LIVE_MODEL) -> dict[str, Any]:
    ready, reason = live_preflight()
    if not ready:
        raise RuntimeError(reason)
    tasks = load_frozen_tasks(TASK_INVENTORY)
    subsets = _selected(tasks)
    full_tasks = {task["task_id"]: task for group in subsets.values() for task in group}
    plan = {"full_bounded_agent": list(full_tasks.values()), **subsets}
    out_dir = Path(out_dir)
    rows = []
    source_sha = _sha256_file(Path(agent_eval.__file__))
    for arm, arm_tasks in plan.items():
        for index, task in enumerate(arm_tasks, start=1):
            key = hashlib.sha256(json.dumps({"arm": arm, "task": task, "model": model,
                                             "agent_eval_sha256": source_sha}, sort_keys=True).encode()).hexdigest()
            path = out_dir / "task_cache" / arm / f"{task['task_id']}.json"
            if path.is_file():
                cached = json.loads(path.read_text())
                if cached.get("cache_key") == key:
                    rows.append(cached["row"])
                    continue
            print(f"[{arm} {index}/{len(arm_tasks)}] {task['task_id']}", flush=True)
            row = _run_one(task, arm, model)
            _atomic_json(path, {"cache_key": key, "row": row})
            rows.append(row)
    summaries = {}
    for arm, arm_tasks in plan.items():
        ids = {task["task_id"] for task in arm_tasks}
        arm_rows = [row for row in rows if row["run"]["arm"] == arm and row["run"]["task_id"] in ids]
        task_map = {task["task_id"]: task for task in arm_tasks}
        summaries[arm] = agent_eval.summarise_arm(task_map, [row["run"] for row in arm_rows], seed=20261300)
        comparison_rows = [row for row in arm_rows if row["run"]["family"] == "case_comparison"]
        summaries[arm]["counterexample_display_rate"] = (
            sum("discordant reference case" in row["run"].get("text", "").lower() for row in comparison_rows)
            / len(comparison_rows) if comparison_rows else None
        )
    transcript = out_dir / "transcripts.jsonl"
    _atomic_write(transcript, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    result = {
        "schema_version": "dermpact-phase7-component-ablations-v1", "status": "PASS",
        "model": model, "summaries": summaries,
        "task_inventory_sha256": _sha256_file(TASK_INVENTORY),
        "agent_eval_sha256": source_sha, "transcripts_sha256": _sha256_file(transcript),
        "hard_scope": {"official_or_final_rows_loaded": 0, "unsafe_checker_ablation_run_live": False},
    }
    _atomic_json(out_dir / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
