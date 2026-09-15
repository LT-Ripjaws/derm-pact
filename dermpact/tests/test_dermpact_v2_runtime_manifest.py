"""Runtime manifest: pins, checkpoint re-hash, drift detection."""
from __future__ import annotations

import json
from pathlib import Path

from dermpact.core.eval import runtime_manifest as rm


def test_build_manifest_pins_sources_anchors_and_policy():
    manifest = rm.build_manifest(tests_summary="unit test")
    assert manifest["schema_version"] == rm.SCHEMA_VERSION
    assert manifest["scope_attestation"]["live_llm_calls_made_in_this_pass"] == 0
    assert manifest["scope_attestation"]["official_or_final_test_rows_loaded_in_this_pass"] == 0
    assert "not_access_log_telemetry" in manifest["scope_attestation"]["kind"]
    assert "dermpact/core/record_v2.py" in manifest["sources"]
    assert "dermpact/core/eval/runtime_manifest.py" in manifest["sources"]
    assert rm.V.SELECTION_MANIFEST in manifest["anchors"]
    assert manifest["checkpoints"]["n_checkpoints"] == 5
    assert manifest["checkpoints"]["all_match_selection_manifest"] is True
    assert all(len(v) == 64 for v in manifest["sources"].values())
    assert manifest["tests_summary"] == "unit test"


def test_write_then_verify_passes_and_detects_drift(tmp_path: Path):
    out = tmp_path / "runtime_manifest_test.json"
    rm.write_manifest(out, tests_summary="t")
    assert Path(str(out) + ".sha256").read_text().strip() == rm.sha256_file(out)
    assert rm.verify_manifest(out)["status"] == "PASS"
    # Tamper with one pinned hash and one missing file entry.
    payload = json.loads(out.read_text(encoding="utf-8"))
    first = next(iter(payload["sources"]))
    payload["sources"][first] = "0" * 64
    payload["artifacts"]["research/experiments/does_not_exist.json"] = "0" * 64
    out.write_text(json.dumps(payload), encoding="utf-8")
    result = rm.verify_manifest(out)
    assert result["status"] == "DRIFT"
    reasons = {(d["path"], d["reason"]) for d in result["drift"]}
    assert (first, "sha256 changed") in reasons
    assert ("research/experiments/does_not_exist.json", "missing") in reasons


def test_cli_verify_exit_codes(tmp_path: Path):
    out = tmp_path / "m.json"
    assert rm.main(["--output", str(out), "--tests-summary", "x"]) == 0
    assert rm.main(["--output", str(out), "--verify"]) == 0
