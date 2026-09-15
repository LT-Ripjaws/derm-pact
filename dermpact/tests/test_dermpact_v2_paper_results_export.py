from __future__ import annotations

import csv
import json

from dermpact.core.eval import paper_results_export as export


def test_pre_live_export_is_machine_readable_and_never_invents_final_results(tmp_path):
    registry = export.export(tmp_path)
    assert registry["schema_version"] == "dermpact-paper-table-export-v1"
    readiness = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    assert readiness["guideline_retrieval_records"] == 66
    assert readiness["guideline_executable_records"] == 5
    assert readiness["official_final_results_present"] is False
    rows = list(csv.DictReader((tmp_path / "agent_performance.csv").open(encoding="utf-8")))
    assert {row["arm"] for row in rows} >= {"deterministic_forms", "offline_deterministic_controller"}
    assert "TABLE_REGISTRY.json" not in registry["outputs"]
    assert all(export._sha(tmp_path / name) == digest for name, digest in registry["outputs"].items())
