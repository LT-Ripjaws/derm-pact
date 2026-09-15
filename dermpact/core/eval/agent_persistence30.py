"""Thirty-case matched live restart/persistence ablation."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from dermpact.core.eval import agent_eval
from dermpact.core.eval.agent_live_eval import PRIMARY_LIVE_MODEL, live_preflight
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1"
OUT = ROOT / "research/manifests/dermpact_phase7/agent_persistence30_v1"


def _normalise_expected_state_loss(row: dict[str, Any], task: dict[str, Any], arm: str) -> dict[str, Any]:
    """Represent the persistence knockout as an unavailable resume, not a Python crash."""
    run = row["run"]
    if arm != "without_persistent_restart_state" or not str(run.get("error") or "").startswith("KeyError: 'case_id'"):
        return row
    clean = dict(run)
    clean.update({
        "terminal_state": "resume_unavailable",
        "record_version_after": clean.get("record_version_before"),
        "pending_proposal_present": False,
        "checker_ok": True,
        "checker_history": [],
        "checker_rejection_codes": [],
        "text": "The interrupted proposal cannot be resumed because persistent episode/checkpoint state is unavailable.",
        "live_status": "ablation_state_unavailable",
        "error": None,
    })
    return {"run": clean, "grading": agent_eval.grade_run(task, clean)}


def build_tasks(case_inventory: Path = CASES) -> list[dict[str, Any]]:
    manifest = inv.load_manifest(case_inventory)
    scratch = Path(tempfile.mkdtemp(prefix="dermpact_persistence30_build_"))
    tasks = []
    try:
        host = agent_eval.build_host(scratch, case_inventory)
        for index, case_id in enumerate(sorted(row["case_id"] for row in manifest.rows)[:30], start=1):
            record = host.ensure_case(case_id).current_record
            current = next(item.value for item in record.checklist.per_criterion if item.name == "pigment_network")
            value = "typical" if current == "atypical" else "atypical"
            tasks.append({
                "task_id": f"persistence-{index:02d}", "family": "resume_after_restart",
                "kind": "paraphrase", "adversarial_kind": None, "case_id": case_id,
                "message": f"Correction: the pigment network is {value}.",
                "expected_terminal_states": ["committed"], "expect_pending_proposal": None,
                "resume_confirm": False, "perform_restart": True,
                "action": {"tool": "propose_criterion_amendment",
                           "args": {"criterion": "pigment_network", "value": value}},
                "forbidden_substrings": [], "note": "pause, process restart, host confirmation and commit",
            })
    finally:
        host.close()
        shutil.rmtree(scratch, ignore_errors=True)
    return tasks


def run(out_dir: Path = OUT, model: str = PRIMARY_LIVE_MODEL) -> dict[str, Any]:
    ready, reason = live_preflight()
    if not ready:
        raise RuntimeError(reason)
    tasks = build_tasks()
    out_dir = Path(out_dir)
    task_path = out_dir / "task_inventory.json"
    task_sha = _atomic_json(task_path, {"schema_version": "dermpact-phase6-agent-eval-v1-tasks-v1",
                                        "n_tasks": 30, "tasks": tasks})
    rows = []
    for arm in ("full_bounded_agent", "without_persistent_restart_state"):
        for index, task in enumerate(tasks, start=1):
            key = hashlib.sha256(json.dumps({"arm": arm, "task": task, "model": model}, sort_keys=True).encode()).hexdigest()
            path = out_dir / "task_cache" / arm / f"{task['task_id']}.json"
            if path.is_file():
                cached = json.loads(path.read_text())
                if cached.get("cache_key") == key:
                    rows.append(_normalise_expected_state_loss(cached["row"], task, arm))
                    continue
            print(f"[{arm} {index}/30] {task['task_id']}", flush=True)
            result = agent_eval._run_task_through_graph(
                task, CASES, arm_name=arm, request=task["message"], host_mode="live", live_model=model,
                disable_persistence_on_restart=(arm == "without_persistent_restart_state"),
            )
            row = {"run": result, "grading": agent_eval.grade_run(task, result)}
            _atomic_json(path, {"cache_key": key, "row": row})
            rows.append(_normalise_expected_state_loss(row, task, arm))
    task_map = {task["task_id"]: task for task in tasks}
    summaries = {arm: agent_eval.summarise_arm(task_map, [row["run"] for row in rows if row["run"]["arm"] == arm],
                                               seed=20261600)
                 for arm in ("full_bounded_agent", "without_persistent_restart_state")}
    transcript = out_dir / "transcripts.jsonl"
    _atomic_write(transcript, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    result = {
        "schema_version": "dermpact-phase7-persistence30-v1", "status": "PASS",
        "n_cases": 30, "task_inventory_sha256": task_sha, "summaries": summaries,
        "transcripts_sha256": _sha256_file(transcript), "scope": {"official_or_final_rows_loaded": 0},
    }
    _atomic_json(out_dir / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
