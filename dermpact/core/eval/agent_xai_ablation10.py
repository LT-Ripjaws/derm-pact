"""Matched live measured-XAI capability ablation on the frozen relevant tasks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dermpact.core.eval import agent_eval
from dermpact.core.eval.agent_live_eval import PRIMARY_LIVE_MODEL, TASK_INVENTORY, live_preflight, load_frozen_tasks
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/manifests/dermpact_phase7/agent_xai_ablation10_v1"
ARMS = ("full_bounded_agent", "without_measured_xai")


def _run(task: dict[str, Any], arm: str, model: str) -> dict[str, Any]:
    def disable_xai(host: Any) -> None:
        host.jobs = None

    run = agent_eval._run_task_through_graph(
        task,
        inv.DEFAULT_INVENTORY_DIR,
        arm_name=arm,
        request=task["message"],
        host_mode="live",
        live_model=model,
        host_mutator=(disable_xai if arm == "without_measured_xai" else None),
    )
    return {"run": run, "grading": agent_eval.grade_run(task, run)}


def execute(out_dir: Path = OUT, model: str = PRIMARY_LIVE_MODEL) -> dict[str, Any]:
    ready, reason = live_preflight()
    if not ready:
        raise RuntimeError(reason)
    tasks = [dict(task) for task in load_frozen_tasks(TASK_INVENTORY)
             if task["family"] in {"region_sensitivity", "clinical_photo"}]
    if len(tasks) != 5:
        raise AssertionError("frozen measured-XAI subset changed")
    out_dir = Path(out_dir)
    source_sha = _sha256_file(Path(agent_eval.__file__))
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for index, task in enumerate(tasks, start=1):
            key = hashlib.sha256(json.dumps({"arm": arm, "task": task, "model": model,
                                             "agent_eval_sha256": source_sha}, sort_keys=True).encode()).hexdigest()
            path = out_dir / "task_cache" / arm / f"{task['task_id']}.json"
            if path.is_file():
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == key:
                    rows.append(cached["row"])
                    continue
            print(f"[{arm} {index}/5] {task['task_id']}", flush=True)
            row = _run(task, arm, model)
            _atomic_json(path, {"cache_key": key, "row": row})
            rows.append(row)

    summaries: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["run"]["arm"] == arm]
        receipt_available = []
        unavailable_resolved = []
        for row in selected:
            trace = row["run"].get("tool_trace_summary") or []
            receipt_available.append(any(
                item.get("ok") and item.get("tool") in {"request_measured_xai", "read_measured_xai"}
                for item in trace
            ))
            text = str(row["run"].get("text") or "").lower()
            unavailable_resolved.append(
                row["run"].get("error") is None
                and not row["grading"].get("final_display_hazard")
                and any(token in text for token in ("unavailable", "not available", "no measured"))
            )
        summaries[arm] = {
            "n_tasks": len(selected),
            "n_unique_cases": len({row["run"]["case_id"] for row in selected}),
            "frozen_protocol_success_rate": sum(bool(row["grading"]["protocol_valid"]) for row in selected) / len(selected),
            "measured_receipt_available_rate": sum(receipt_available) / len(receipt_available),
            "safe_unavailability_resolution_rate": sum(unavailable_resolved) / len(unavailable_resolved),
            "final_hazard_rate": sum(bool(row["grading"]["final_display_hazard"]) for row in selected) / len(selected),
            "mean_end_to_end_seconds": sum(float(row["run"].get("end_to_end_latency_s") or 0.0) for row in selected) / len(selected),
        }
    transcript = out_dir / "transcripts.jsonl"
    _atomic_write(transcript, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    result = {
        "schema_version": "dermpact-phase7-agent-xai-ablation10-v1",
        "status": "PASS",
        "model": model,
        "summaries": summaries,
        "task_inventory_sha256": _sha256_file(TASK_INVENTORY),
        "transcripts_sha256": _sha256_file(transcript),
        "scope": {
            "official_or_final_rows_loaded": 0,
            "independent_clinician_benefit_measured": False,
            "interpretation": "matched clinician-facing capability test on the frozen XAI task subset",
        },
    }
    _atomic_json(out_dir / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
