"""Freeze 20 source-grounding tasks for the guideline-library knockout."""
from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dermpact.core.eval import agent_eval
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _sha256_file
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1"
OUT = ROOT / "research/manifests/dermpact_phase7/agent_guideline20_v1"


def build(case_inventory: Path = CASES, out_dir: Path = OUT) -> dict:
    manifest = inv.load_manifest(case_inventory)
    scratch = Path(tempfile.mkdtemp(prefix="dermpact_guideline20_"))
    tasks = []
    try:
        host = agent_eval.build_host(scratch / "state", case_inventory)
        for case_id in sorted(row["case_id"] for row in manifest.rows):
            record = host.ensure_case(case_id).current_record
            rule_id = record.next_step.rule_id
            if not rule_id:
                continue
            index = len(tasks) + 1
            tasks.append({
                "task_id": f"guideline-source-{index:02d}", "family": "source_quote",
                "kind": "paraphrase", "adversarial_kind": None, "case_id": case_id,
                "message": "Show the exact guideline evidence supporting the recorded next step.",
                "expected_terminal_states": ["answered"], "expect_pending_proposal": False,
                "resume_confirm": False, "perform_restart": False,
                "action": {"tool": "retrieve_guideline_evidence", "args": {"rule_id": rule_id}},
                "forbidden_substrings": [], "note": "guideline-library structural dependency probe",
            })
            if len(tasks) == 20:
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if len(tasks) != 20:
        raise AssertionError("fewer than 20 cases have an executable next-step rule")
    out_dir = Path(out_dir)
    path = out_dir / "task_inventory.json"
    sha = _atomic_json(path, {"schema_version": "dermpact-phase6-agent-eval-v1-tasks-v1",
                              "n_tasks": 20, "tasks": tasks})
    protocol = {
        "schema_version": "dermpact-phase7-guideline20-v1",
        "status": "PRESPECIFIED_BEFORE_LIVE_GENERATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "n_tasks": 20,
        "n_case_clusters": 20, "task_inventory_sha256": sha,
        "case_inventory_sha256": _sha256_file(case_inventory / "inventory_manifest.json"),
        "knockout_definition": "without guideline library, exact source retrieval capability is absent (0/20)",
        "hard_scope": {"official_or_final_rows_loaded": 0},
    }
    _atomic_json(out_dir / "protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(build())
