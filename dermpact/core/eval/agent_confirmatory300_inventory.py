"""Freeze 300 agent tasks and a 100-answer grounding subset over 100 dev cases."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _sha256_file
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
CASE_INVENTORY = ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1"
OUT_DIR = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v2_grounded_reads"


def _task(task_id: str, family: str, case_id: str, message: str,
          tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id, "family": family, "kind": "paraphrase",
        "adversarial_kind": None, "case_id": case_id, "message": message,
        "expected_terminal_states": ["answered"], "expect_pending_proposal": False,
        "resume_confirm": False, "perform_restart": False,
        "action": {"tool": tool, "args": args}, "forbidden_substrings": [],
        "note": "confirmatory non-mutating development task",
    }


def build(case_inventory: Path = CASE_INVENTORY, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    manifest = inv.load_manifest(case_inventory)
    case_ids = sorted(row["case_id"] for row in manifest.rows)
    if len(case_ids) != 100 or len(set(case_ids)) != 100:
        raise AssertionError("expected 100 unique real development cases")
    tasks: list[dict[str, Any]] = []
    ragas_tasks: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        suffix = f"{index:03d}"
        evidence = _task(
            f"confirm-evidence-{suffix}", "evidence_report", case_id,
            ("Please explain this case's evidence in full." if index % 2 else
             "Can you walk me through the evidence recorded for this case?"),
            "read_record_section", {"section": "overview"},
        )
        reliability = _task(
            f"confirm-reliability-{suffix}", "explain_deferral", case_id,
            ("Explain this case's reliability status and any recovery action." if index % 2 else
             "What reliability flags apply here, and what should be reviewed or reacquired?"),
            "read_reliability", {},
        )
        comparison = _task(
            f"confirm-comparison-{suffix}", "case_comparison", case_id,
            ("Are there any similar prior cases for comparison?" if index % 2 else
             "Have we seen comparable cases before that I could review?"),
            "retrieve_case_comparisons", {},
        )
        tasks.extend((evidence, reliability, comparison))
        ragas_tasks.append(evidence if index % 2 else comparison)

    out_dir = Path(out_dir)
    full_path = out_dir / "task_inventory_300.json"
    ragas_path = out_dir / "task_inventory_ragas100.json"
    full_sha = _atomic_json(full_path, {
        "schema_version": "dermpact-phase6-agent-eval-v1-tasks-v1",
        "n_tasks": len(tasks), "tasks": tasks,
    })
    ragas_sha = _atomic_json(ragas_path, {
        "schema_version": "dermpact-phase6-agent-eval-v1-tasks-v1",
        "n_tasks": len(ragas_tasks), "tasks": ragas_tasks,
    })
    protocol = {
        "schema_version": "dermpact-phase7-agent-confirmatory300-v2-grounded-reads",
        "status": "PRESPECIFIED_BEFORE_LIVE_GENERATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_independent_case_clusters": 100,
        "agent_tasks": {"n": 300, "per_case": 3,
                        "families": ["evidence_report", "explain_deferral", "case_comparison"],
                        "sha256": full_sha},
        "ragas_subset": {"n": 100, "per_case": 1,
                         "selection": "odd case rank evidence; even case rank comparison",
                         "judge_passes": 5, "sha256": ragas_sha},
        "case_inventory_sha256": _sha256_file(case_inventory / "inventory_manifest.json"),
        "hard_scope": {"official_or_final_rows_loaded": 0, "synthetic_fixture_cases": 0},
        "inference_unit": "case_id; prompts within a case are clustered, never counted independent",
        "complementary_benchmark": (
            "The frozen 54-task suite retains amendment, restart, refusal and adversarial coverage; "
            "this 300-task run estimates ordinary grounded workflow performance."
        ),
    }
    _atomic_json(out_dir / "protocol.json", protocol)
    return protocol


def main() -> int:
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
