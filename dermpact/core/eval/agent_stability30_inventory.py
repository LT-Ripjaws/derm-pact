"""Freeze a 30-task repeated-live subset from the 300-task inventory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _sha256_file

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v1/task_inventory_300.json"
OUT_DIR = ROOT / "research/manifests/dermpact_phase7/agent_stability30_v1"


def build(source: Path = SOURCE, out_dir: Path = OUT_DIR) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    case_ids = sorted({task["case_id"] for task in tasks})[:10]
    subset = [task for task in tasks if task["case_id"] in case_ids]
    if len(subset) != 30:
        raise AssertionError("expected three tasks for each of ten cases")
    out_dir = Path(out_dir)
    inventory = out_dir / "task_inventory.json"
    inventory_sha = _atomic_json(inventory, {
        "schema_version": "dermpact-phase6-agent-eval-v1-tasks-v1",
        "n_tasks": 30, "tasks": subset,
    })
    protocol = {
        "schema_version": "dermpact-phase7-agent-stability30-v1",
        "status": "PRESPECIFIED_BEFORE_REPEATED_LIVE_RUNS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_tasks": 30, "n_case_clusters": 10, "n_independent_runs": 3,
        "selection": "all three tasks for first ten sorted case ids in confirmatory300",
        "task_inventory_sha256": inventory_sha, "source_inventory_sha256": _sha256_file(source),
        "hard_scope": {"official_or_final_rows_loaded": 0},
    }
    _atomic_json(out_dir / "protocol.json", protocol)
    return protocol


def main() -> int:
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
