"""Merge successful targeted retries into a frozen live run with full provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dermpact.core.eval import agent_eval
from dermpact.core.eval.agent_live_eval import load_frozen_tasks
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge(base: Path, retry: Path, task_inventory: Path, out_dir: Path) -> dict[str, Any]:
    base_rows = _load(base)
    retry_rows = _load(retry)
    failed_ids = {row["run"]["task_id"] for row in base_rows if not row["grading"]["protocol_valid"]}
    retry_by_id = {row["run"]["task_id"]: row for row in retry_rows}
    if set(retry_by_id) != failed_ids:
        raise AssertionError(f"retry ids differ from failed ids: retry={set(retry_by_id)} failed={failed_ids}")
    if any(not row["grading"]["protocol_valid"] for row in retry_rows):
        raise AssertionError("a retry remains protocol-invalid")
    merged = [retry_by_id.get(row["run"]["task_id"], row) for row in base_rows]
    if len(merged) != 300 or len({row["run"]["task_id"] for row in merged}) != 300:
        raise AssertionError("merged run is not the complete 300-task inventory")
    tasks = {task["task_id"]: task for task in load_frozen_tasks(task_inventory)}
    unchanged = [row for row in base_rows if row["run"]["task_id"] not in failed_ids]
    merged_unchanged = [row for row in merged if row["run"]["task_id"] not in failed_ids]
    if unchanged != merged_unchanged:
        raise AssertionError("a successful base row changed during merge")
    out_dir = Path(out_dir)
    transcripts = out_dir / "transcripts.jsonl"
    transcripts_sha = _atomic_write(
        transcripts, "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in merged)
    )
    summary = agent_eval.summarise_arm(tasks, [row["run"] for row in merged], seed=20261108)
    result = {
        "schema_version": "dermpact-phase7-agent-retry-merge-v1", "status": "PASS",
        "n_tasks": len(merged), "n_case_clusters": len({row["run"]["case_id"] for row in merged}),
        "n_replaced": len(failed_ids), "replaced_task_ids": sorted(failed_ids),
        "n_preserved_byte_equivalent_rows": len(unchanged), "summary": summary,
        "sources": {"base_transcripts_sha256": _sha256_file(base),
                    "retry_transcripts_sha256": _sha256_file(retry),
                    "task_inventory_sha256": _sha256_file(task_inventory)},
        "transcripts_sha256": transcripts_sha,
        "scope": {"official_or_final_rows_loaded": 0,
                  "merge_rule": "only protocol-invalid base rows may be replaced by protocol-valid exact-task retries"},
    }
    _atomic_json(out_dir / "results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--task-inventory", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.retry, args.task_inventory, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
