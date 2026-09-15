"""Build the prespecified 50-case/100-answer development grounding inventory."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _sha256_file
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/manifests/dermpact_phase7/agent_grounding_confirmatory50_v1"


def _task(task_id: str, family: str, case_id: str, message: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "family": family,
        "kind": "paraphrase",
        "adversarial_kind": None,
        "case_id": case_id,
        "message": message,
        "expected_terminal_states": ["answered"],
        "expect_pending_proposal": False,
        "resume_confirm": False,
        "perform_restart": False,
        "action": {"tool": tool, "args": args},
        "forbidden_substrings": [],
        "note": "confirmatory grounding task; non-mutating; development-only",
    }


def build(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    manifest_path = inv.DEFAULT_INVENTORY_DIR / "inventory_manifest.json"
    manifest = inv.load_manifest(inv.DEFAULT_INVENTORY_DIR)
    case_ids = sorted(row["case_id"] for row in manifest.rows)
    if len(case_ids) != 50 or len(set(case_ids)) != 50:
        raise AssertionError("expected exactly 50 unique reserved development cases")
    tasks = []
    evidence_messages = (
        "Please explain this case's evidence in full.",
        "Can you walk me through the evidence recorded for this case?",
    )
    comparison_messages = (
        "Are there any similar prior cases for comparison?",
        "Have we seen comparable cases before that I could review?",
    )
    for index, case_id in enumerate(case_ids, start=1):
        suffix = f"{index:03d}"
        tasks.append(_task(
            f"confirm-evidence-{suffix}", "evidence_report", case_id,
            evidence_messages[(index - 1) % len(evidence_messages)],
            "read_record_section", {"section": "heads"},
        ))
        tasks.append(_task(
            f"confirm-comparison-{suffix}", "case_comparison", case_id,
            comparison_messages[(index - 1) % len(comparison_messages)],
            "retrieve_case_comparisons", {},
        ))
    out_dir = Path(out_dir)
    inventory_path = out_dir / "task_inventory.json"
    inventory_sha = _atomic_json(inventory_path, {
        "schema_version": "dermpact-phase6-agent-eval-v1-tasks-v1",
        "n_tasks": len(tasks),
        "tasks": tasks,
    })
    protocol = {
        "schema_version": "dermpact-phase7-grounding-confirmatory50-v1",
        "status": "PRESPECIFIED_BEFORE_CONFIRMATORY_GENERATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_independent_case_clusters": len(case_ids),
        "n_answers": len(tasks),
        "tasks_per_case": 2,
        "families": ["evidence_report", "case_comparison"],
        "selection": "all 50 hash-verified reserved real development cases, sorted by case_id",
        "task_inventory_sha256": inventory_sha,
        "source_inventory_sha256": _sha256_file(manifest_path),
        "judge_plan": {"passes": 5, "metric": "RAGAS-style faithfulness", "cluster_unit": "case_id"},
        "hard_scope": {
            "official_or_final_rows_loaded": 0,
            "raw_images_opened_by_builder": 0,
            "synthetic_fixture_cases": 0,
        },
        "interpretation": (
            "Supporting grounding evaluation. The effective independent sample is 50 case clusters, "
            "not 100 independent cases; the frozen 54-task suite remains the workflow/security benchmark."
        ),
    }
    _atomic_json(out_dir / "protocol.json", protocol)
    return protocol


def main() -> int:
    protocol = build()
    print(protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())