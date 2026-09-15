"""Live bounded-agent qualification over the frozen Phase 6 task inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dermpact.core.agent.prompts import PROMPT_SHA256
from dermpact.core.eval import agent_eval as offline
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file
from dermpact.core.prototype import inventory as inv
from dermpact import config as dermpact_config

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase6/agent_live_eval_v1"
SCHEMA_VERSION = "dermpact-phase6-agent-live-eval-v1"
TASK_INVENTORY = ROOT / "research/manifests/dermpact_phase6/agent_eval_v1/task_inventory.json"
PRIMARY_LIVE_MODEL = offline.V.PRIMARY_LIVE_AGENT_MODEL
SMOKE5_TASK_IDS = (
    "evidence_report-p1",
    "clarification-p1",
    "region_sensitivity-p1",
    "case_comparison-p1",
    "out_of_scope-adv-treatment-1",
)


def live_preflight() -> tuple[bool, str]:
    dermpact_config.load_env()
    if os.environ.get("DERMPACT_ALLOW_LIVE_LLM") != "1":
        return False, "DERMPACT_ALLOW_LIVE_LLM=1 is required"
    if not (os.environ.get("DERMPACT_API_KEY") or os.environ.get("KILOCODE_API_KEY")):
        return False, "DERMPACT_API_KEY or KILOCODE_API_KEY is required"
    return True, "ready"


def load_frozen_tasks(path: Path = TASK_INVENTORY) -> list[Mapping[str, Any]]:
    sidecar = Path(str(path) + ".sha256")
    expected = sidecar.read_text(encoding="ascii").strip()
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"frozen task inventory hash mismatch: expected {expected}, observed {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dermpact-phase6-agent-eval-v1-tasks-v1":
        raise ValueError("unknown frozen task inventory schema")
    tasks = list(payload.get("tasks") or [])
    if len(tasks) != int(payload.get("n_tasks", -1)) or len(tasks) < 1:
        raise ValueError("frozen task inventory count is invalid")
    return tasks


def select_tasks(
    tasks: Sequence[Mapping[str, Any]], *, profile: str, limit: int | None,
    task_ids: Sequence[str] = (),
) -> list[Mapping[str, Any]]:
    if task_ids:
        if profile != "full" or limit is not None:
            raise ValueError("--task-id cannot be combined with --profile smoke5 or --limit")
        by_id = {task["task_id"]: task for task in tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown frozen task ids: {missing}")
        return [by_id[task_id] for task_id in dict.fromkeys(task_ids)]
    if profile == "smoke5":
        if limit is not None:
            raise ValueError("--profile smoke5 and --limit cannot be combined")
        by_id = {task["task_id"]: task for task in tasks}
        missing = [task_id for task_id in SMOKE5_TASK_IDS if task_id not in by_id]
        if missing:
            raise ValueError(f"frozen smoke tasks missing: {missing}")
        return [by_id[task_id] for task_id in SMOKE5_TASK_IDS]
    if profile != "full":
        raise ValueError(f"unknown task profile {profile!r}")
    return list(tasks[:limit] if limit else tasks)


def run_live_task(task: Mapping[str, Any], inventory_dir: Path, *, model: str | None) -> Mapping[str, Any]:
    result = offline._run_task_through_graph(
        task, inventory_dir, arm_name="bounded_agent_live", request=task["message"],
        host_mode="live", live_model=model,
    )
    if result.get("error"):
        return result
    # Clarify-ask tasks pause deterministically before a provider is needed.
    provider_required = task.get("kind") != "clarify_ask" and task.get("family") != "out_of_scope"
    if provider_required and result.get("live_status") != "live_provider_used":
        result = dict(result)
        result["error"] = f"live_provider_not_used:{result.get('live_status')}"
        result["terminal_state"] = "error"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--inventory-dir", type=Path, default=inv.DEFAULT_INVENTORY_DIR)
    parser.add_argument("--task-inventory", type=Path, default=TASK_INVENTORY)
    parser.add_argument("--limit", type=int, default=None, help="cost-control prefix of the frozen task inventory")
    parser.add_argument("--profile", choices=["full", "smoke5"], default="full",
                        help="smoke5 selects five prespecified, cross-family tasks; full uses the complete inventory")
    parser.add_argument("--task-id", action="append", default=[],
                        help="run one or more exact frozen task IDs for targeted diagnostics")
    parser.add_argument("--model", default=PRIMARY_LIVE_MODEL,
                        help="gateway model locked to the previous-paper generator by default; overrides are separate sensitivity runs")
    args = parser.parse_args(argv)

    ready, reason = live_preflight()
    if not ready:
        print(json.dumps({"status": "REFUSED", "reason": reason}, indent=2))
        return 2
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    task_inventory = Path(args.task_inventory)
    tasks = load_frozen_tasks(task_inventory)
    try:
        selected = select_tasks(tasks, profile=args.profile, limit=args.limit, task_ids=args.task_id)
    except ValueError as exc:
        parser.error(str(exc))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_manifest = Path(args.inventory_dir) / "inventory_manifest.json"
    inventory_manifest_sha = _sha256_file(inventory_manifest)
    protocol = {
        "schema_version": SCHEMA_VERSION, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": PROMPT_SHA256, "n_tasks_total": len(tasks), "n_tasks_selected": len(selected),
        "task_inventory_path": (
            task_inventory.resolve().relative_to(ROOT.resolve()).as_posix()
            if task_inventory.resolve().is_relative_to(ROOT.resolve()) else str(task_inventory.resolve())
        ),
        "task_inventory_sha256": _sha256_file(task_inventory),
        "selection": ("explicit frozen task ids" if args.task_id else
                      ("prespecified cross-family smoke5" if args.profile == "smoke5" else
                       ("deterministic task-inventory prefix" if args.limit else "full frozen task inventory"))),
        "selected_task_ids": [task["task_id"] for task in selected],
        "task_profile": args.profile,
        "model_override": args.model, "temperature": 0,
        "case_inventory_manifest_sha256": inventory_manifest_sha,
        "resume_policy": "per-task atomic content-addressed cache",
        "fallback_policy": "provider fallback is an evaluation error, never a live success",
        "xai_task_input": "labelled protocol fixture; not scientific attribution evidence",
        "official_or_final_test_loader_imported": False,
    }
    protocol_path = out_dir / "protocol.json"
    _atomic_json(protocol_path, protocol)

    rows = []
    cache_hits = 0
    cache_dir = out_dir / "task_cache"
    for index, task in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {task['task_id']}", flush=True)
        cache_key = hashlib.sha256(json.dumps({
            "task": task, "model": args.model, "prompt_sha256": PROMPT_SHA256,
            "case_inventory_manifest_sha256": inventory_manifest_sha,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{task['task_id']}.json"
        cached = None
        if cache_path.is_file():
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if candidate.get("cache_key") == cache_key:
                cached = candidate.get("row")
        if cached is not None:
            row = cached
            cache_hits += 1
        else:
            run = run_live_task(task, args.inventory_dir, model=args.model)
            row = {"run": run, "grading": offline.grade_run(task, run)}
            _atomic_json(cache_path, {"cache_key": cache_key, "row": row})
        rows.append(row)
    transcript_path = out_dir / "transcripts.jsonl"
    _atomic_write(transcript_path, "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows))

    tasks_by_id = {task["task_id"]: task for task in selected}
    summary = offline.summarise_arm(tasks_by_id, [row["run"] for row in rows], seed=offline.BOOT_SEED + 100)
    usage: dict[str, float | int] = {}
    for row in rows:
        for key, value in (row["run"].get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    n_protocol_invalid = sum(1 for row in rows if not row["grading"]["protocol_valid"])
    results = {
        "schema_version": SCHEMA_VERSION, "arm": "bounded_agent_live", "summary": summary, "usage": usage,
        "n_live_provider_failures": sum(1 for row in rows if row["run"].get("error")),
        "n_protocol_invalid": n_protocol_invalid,
        "task_cache_hits": cache_hits,
        "protocol_sha256": _sha256_file(protocol_path), "transcripts_sha256": _sha256_file(transcript_path),
        "comparison_controls": {
            "offline_results": "research/manifests/dermpact_phase6/agent_eval_v1/results.json",
            "applicability_live_controls_command": (
                ".\\venv_p2\\Scripts\\python.exe -m dermpact.core.eval.applicability_benchmark_v2 "
                "--llm-mode live --limit 5"
            ),
        },
    }
    results["status"] = "PASS" if results["n_live_provider_failures"] == 0 and n_protocol_invalid == 0 else "FAIL"
    _atomic_json(out_dir / "results.json", results)
    print(json.dumps({"status": results["status"], "n_tasks": len(selected), "usage": usage,
                      "n_live_provider_failures": results["n_live_provider_failures"],
                      "n_protocol_invalid": results["n_protocol_invalid"],
                      "out_dir": str(out_dir)}, indent=2))
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
