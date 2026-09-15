"""Security/utility metrics for the frozen 120-task adversarial run."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

from dermpact.core.eval import agent_eval
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "research/manifests/dermpact_phase7/agent_adversarial120_v1"


def evaluate(task_inventory: Path, transcripts: Path, out_path: Path) -> dict[str, Any]:
    tasks = {task["task_id"]: task for task in json.loads(task_inventory.read_text())["tasks"]}
    wrapped = [json.loads(line) for line in transcripts.read_text().splitlines() if line.strip()]
    if len(wrapped) != len(tasks):
        raise AssertionError("adversarial transcript/task mismatch")
    rows = []
    for item in wrapped:
        run = item["run"]
        task = tasks[run["task_id"]]
        grade = agent_eval.grade_run(task, run)
        rows.append({**grade, "family": task["family"], "adversarial_kind": task["adversarial_kind"]})

    def summarize(group: list[dict[str, Any]], seed: int) -> dict[str, Any]:
        return {
            "n": len(group), "n_case_clusters": len({row["case_id"] for row in group}),
            "protocol_valid": agent_eval._rate_with_ci(group, "protocol_valid", seed=seed),
            "attack_success_final_hazard": agent_eval._rate_with_ci(group, "final_display_hazard", seed=seed + 1),
            "tool_selection_correct": agent_eval._rate_with_ci(group, "tool_selection_correct", seed=seed + 2),
            "tool_argument_correct": agent_eval._rate_with_ci(group, "tool_argument_correct", seed=seed + 3),
            "false_action_claim_count": sum(row["false_action_claim"] for row in group),
            "wrong_case_scope_count": sum(not row["case_scope_ok"] for row in group),
            "forbidden_output_count": sum(row["forbidden_hit"] for row in group),
        }

    by_family = {}
    for index, attack in enumerate(sorted({row["adversarial_kind"] for row in rows})):
        by_family[attack] = summarize([row for row in rows if row["adversarial_kind"] == attack], 20261000 + 10 * index)
    result = {
        "schema_version": "dermpact-phase7-agent-adversarial-metrics-v1", "status": "PASS",
        "overall": summarize(rows, 20260980), "per_attack_family": by_family, "rows": rows,
        "scope": {"security_and_utility_reported_together": True, "adaptive_attacks": False,
                  "official_or_final_rows_loaded": 0},
    }
    _atomic_json(out_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-inventory", type=Path, default=DEFAULT_DIR / "task_inventory.json")
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_DIR / "live_qwen3vl/transcripts.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR / "metrics.json")
    args = parser.parse_args()
    result = evaluate(args.task_inventory, args.transcripts, args.out)
    print(json.dumps({"status": result["status"], "overall": result["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
