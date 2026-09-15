"""Pre-final runtime freeze for the complete DermPact selected configuration."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _sha256_file
from dermpact.core.eval.runtime_manifest import _checkpoint_check

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "research/manifests/dermpact_phase7/runtime_manifest_v2_pre_final.json"
SOURCE_ROOTS = ("dermpact/core", "dermpact/reliability")
ANCHORS = (
    "research/manifests/dermpact_phase2_final/selection_2026-09-06.json",
    "research/manifests/dermpact_phase3/reliability_bundle_v3.json",
    "research/manifests/dermpact_phase4/rule_catalog_v2/catalog_manifest.json",
    "dermpact/data/corpus/eado_melanoma_part1_rules_v2.jsonl",
    "research/manifests/dermpact_phase6/guideline_library_v1/manifest.json",
    "research/experiments/dermpact_phase6/memory_bank_dev_v1/manifest.json",
    "research/experiments/dermpact_phase6/memory_bank_dev_v1/bank.jsonl",
    "research/manifests/dermpact_final_v1/protocol.json",
    "research/manifests/dermpact_final_v1/dev_smoke.json",
)
DEVELOPMENT_RESULTS = (
    "research/manifests/dermpact_phase7/canonical_development_ledger_v1/headline.json",
    "research/manifests/dermpact_phase7/paper_ablation_tables_v1/results.json",
    "research/manifests/dermpact_phase7/memory_controls100_v1/results.json",
    "research/manifests/dermpact_phase7/agent_confirmatory300_v2_grounded_reads/metrics.json",
    "research/manifests/dermpact_phase7/agent_ragas_matched100_v2/results.json",
    "research/manifests/dermpact_phase7/agent_ragas_matched100_v2/paired_statistics.json",
    "research/manifests/dermpact_phase7/agent_adversarial120_v2_host_binding/metrics.json",
    "research/manifests/dermpact_phase7/applicability_live100_v1/results.json",
    "research/manifests/dermpact_phase7/security_host_channels_v1/results.json",
)


def _sources() -> list[Path]:
    output=[]
    for root in SOURCE_ROOTS:
        output.extend(path for path in (ROOT/root).rglob("*.py") if "__pycache__" not in path.parts)
    output.extend((ROOT/"dermpact/eval/dermpact_phase2_hybrid_screen.py",
                   ROOT/"dermpact/eval/dermpact_phase2_baselines.py"))
    return sorted(set(output))


def build(path: Path = OUTPUT, tests_summary: str = "698 passed, 1 slow deselected, 957 subtests; 2026-09-09") -> dict[str,Any]:
    required=[ROOT/item for item in (*ANCHORS,*DEVELOPMENT_RESULTS)]
    missing=[str(item) for item in required if not item.is_file()]
    if missing:raise FileNotFoundError(missing)
    checkpoints=_checkpoint_check()
    if not checkpoints["all_match_selection_manifest"]:raise AssertionError("selected checkpoints drift")
    result={"schema_version":"dermpact-phase7-runtime-manifest-v2-pre-final","status":"FROZEN_AFTER_MECHANICAL_PREPREDICTION_AMENDMENT",
        "generated_at":datetime.now(timezone.utc).isoformat(),"tests_summary":tests_summary,
        "policy":{"perception_training":"closed","post_final_tuning":False,"live_model":"qwen/qwen3-vl-32b-instruct",
                  "presentation":"Python package, compiled LangGraph workflow and CLI; no deployed app claim"},
        "checkpoints":checkpoints,
        "anchors":{item.relative_to(ROOT).as_posix():_sha256_file(item) for item in required},
        "sources":{item.relative_to(ROOT).as_posix():_sha256_file(item) for item in _sources()},
        "scope":{"official_or_final_rows_loaded_in_freeze_pass":0,"final_access_log":ACCESS_LOG_REL,
                 "amendment":"initial final session stopped at BCN role-count assertion before predictions; resolver now uses frozen Phase-1 roles",
                 "historical_exposure":"Derm7pt/HAM/BCN old-configuration exposure disclosed; PH2 prediction-unopened"}}
    _atomic_json(Path(path),result);return result


ACCESS_LOG_REL="research/DERMPACT_SELECTED_FINAL_ACCESS_LOG.jsonl"


def verify(path: Path = OUTPUT) -> dict[str,Any]:
    payload=json.loads(Path(path).read_text(encoding="utf-8"));drift=[]
    for section in ("anchors","sources"):
        for relative,expected in payload[section].items():
            target=ROOT/relative
            if not target.is_file():drift.append({"path":relative,"reason":"missing"})
            elif _sha256_file(target)!=expected:drift.append({"path":relative,"reason":"sha256 changed"})
    current={item.relative_to(ROOT).as_posix() for item in _sources()}
    for relative in sorted(current-set(payload["sources"])):drift.append({"path":relative,"reason":"new source not pinned"})
    checkpoints=_checkpoint_check()
    return {"status":"PASS" if not drift and checkpoints["all_match_selection_manifest"] else "DRIFT",
            "n_sources":len(payload["sources"]),"n_anchors":len(payload["anchors"]),"checkpoints_match":checkpoints["all_match_selection_manifest"],"drift":drift}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true");args=parser.parse_args()
    print(json.dumps(verify() if args.verify else build(),indent=2))
