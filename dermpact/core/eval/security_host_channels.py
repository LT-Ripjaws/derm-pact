"""Execute and summarize deterministic host-security mutation channels."""
from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/manifests/dermpact_phase7/security_host_channels_v1"

CHANNELS = {
    "xai_receipt_tampering": {
        "attack": [
            "dermpact/tests/test_dermpact_v2_xai.py::CacheKeyAndReceiptTest::test_sealed_complete_receipt_rejects_content_tamper",
            "dermpact/tests/test_dermpact_v2_xai.py::CacheKeyAndReceiptTest::test_verify_rejects_wrong_image_sha",
            "dermpact/tests/test_dermpact_v2_xai.py::CacheKeyAndReceiptTest::test_verify_rejects_wrong_case_id",
            "dermpact/tests/test_dermpact_v2_xai.py::CacheKeyAndReceiptTest::test_verify_rejects_wrong_checkpoint_set",
            "dermpact/tests/test_dermpact_v2_xai.py::CacheKeyAndReceiptTest::test_verify_rejects_wrong_calibration_sha",
        ],
        "clean": [
            "dermpact/tests/test_dermpact_v2_xai.py::CacheKeyAndReceiptTest::test_verify_accepts_matching_record",
            "dermpact/tests/test_dermpact_v2_xai.py::CacheKeyAndReceiptTest::test_verify_accepts_amended_record_same_image_and_reports_provenance",
        ],
    },
    "episode_and_record_tampering": {
        "attack": [
            "dermpact/tests/test_dermpact_v2_episode.py::IdempotencyTest::test_event_payload_tamper_is_rejected",
            "dermpact/tests/test_dermpact_v2_episode.py::IdempotencyTest::test_reordered_hash_chain_is_rejected",
            "dermpact/tests/test_dermpact_v2_store.py::ChainAndHeadVerificationTest::test_tampered_record_file_fails_chain_verification",
            "dermpact/tests/test_dermpact_v2_store.py::LedgerAndEpisodeEventPersistenceTest::test_ledger_event_wrong_case_rejected",
            "dermpact/tests/test_dermpact_v2_store.py::LedgerAndEpisodeEventPersistenceTest::test_episode_event_wrong_case_rejected",
        ],
        "clean": [
            "dermpact/tests/test_dermpact_v2_episode.py::IdempotencyTest::test_duplicate_idempotency_key_returns_same_event_no_growth",
            "dermpact/tests/test_dermpact_v2_store.py::AppendRecordTest::test_append_record_extends_chain",
        ],
    },
    "memory_poisoning_revocation_supersession": {
        "attack": [
            "dermpact/tests/test_dermpact_v2_memory.py::MemoryEpisodeTest::test_tampered_hash_rejected",
            "dermpact/tests/test_dermpact_v2_memory.py::MemoryBankTest::test_write_load_rejects_corrupted_line",
            "dermpact/tests/test_dermpact_v2_memory.py::MemoryBankTest::test_effective_drops_superseded_episode",
            "dermpact/tests/test_dermpact_v2_memory.py::CompareTest::test_superseded_episode_never_returned",
            "dermpact/tests/test_dermpact_v2_memory.py::RefreshTest::test_refresh_drops_revoked_match",
        ],
        "clean": [
            "dermpact/tests/test_dermpact_v2_memory.py::CompareTest::test_counterexample_available",
            "dermpact/tests/test_dermpact_v2_memory.py::MemoryEpisodeTest::test_create_round_trip_and_hash",
        ],
    },
    "concurrency_and_restart": {
        "attack": [
            "dermpact/tests/test_dermpact_v2_store.py::LedgerAndEpisodeEventPersistenceTest::test_concurrent_event_appends_do_not_lose_updates",
            "dermpact/tests/test_dermpact_v2_prototype_e2e.py::test_resume_after_restart_stale_record_version",
        ],
        "clean": [
            "dermpact/tests/test_dermpact_v2_agent_eval.py::test_resume_after_restart_task_survives_a_real_host_restart",
        ],
    },
    "source_hash_and_entity_binding": {
        "attack": [
            "dermpact/tests/test_guideline_retriever.py::test_load_enforces_the_receipt_hash_when_supplied",
            "dermpact/tests/test_citation_integrity.py::test_provenance_tamper_under_a_real_id",
        ],
        "clean": [
            "dermpact/tests/test_guideline_retriever.py::test_repository_catalog_loads_under_its_receipt_hash_and_scopes_as_measured",
        ],
    },
}


def _run_nodes(nodes: list[str], xml_path: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *nodes, "-q", f"--junitxml={xml_path}"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if not xml_path.is_file():
        raise RuntimeError(f"pytest produced no JUnit report: {completed.stderr[-500:]}")
    root = ET.parse(xml_path).getroot()
    cases = root.findall(".//testcase")
    failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    return {"n": len(cases), "failures": failed, "rate": failed / len(cases) if cases else None,
            "pytest_exit_code": completed.returncode}


def run(out_dir: Path = OUT) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for channel, groups in CHANNELS.items():
        attack = _run_nodes(groups["attack"], out_dir / f"{channel}_attack.xml")
        clean = _run_nodes(groups["clean"], out_dir / f"{channel}_clean.xml")
        results[channel] = {
            "attack_denominator": attack["n"], "security_failures": attack["failures"],
            "security_failure_rate": attack["rate"],
            "clean_denominator": clean["n"], "clean_false_blocks": clean["failures"],
            "clean_false_block_rate": clean["rate"],
            "attack_scenarios": groups["attack"], "clean_scenarios": groups["clean"],
        }
    result = {
        "schema_version": "dermpact-phase7-security-host-channels-v1",
        "status": "PASS" if all(row["security_failures"] == 0 and row["clean_false_blocks"] == 0
                                  for row in results.values()) else "FAIL",
        "channels": results,
        "scope": {"deterministic_software_mutation_tests": True, "adaptive_attacks": False,
                  "official_or_final_rows_loaded": 0,
                  "confidence_note": "scenario enumerations are not independent clinical samples"},
    }
    _atomic_json(out_dir / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
