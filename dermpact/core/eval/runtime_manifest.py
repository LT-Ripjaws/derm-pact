"""Runtime manifest for the connected DermPact prototype (Phases 5-6)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dermpact.core import versions as V

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "research/manifests/dermpact_phase6/runtime_manifest_v1.json"
SCHEMA_VERSION = "dermpact-phase6-runtime-manifest-v1"

#: Source trees pinned file-by-file (repository-relative glob roots).
SOURCE_GLOBS = (
    ("dermpact/core", "**/*.py"),
    ("dermpact/tests", "test_dermpact_v2_*.py"),
    ("dermpact/tests", "_dermpact_v2_fixtures.py"),
    ("dermpact/tests", "conftest.py"),
)
#: Release anchors and frozen sources the runtime loads (hash-verified by their own loaders too).
ANCHORS = (
    V.SELECTION_MANIFEST, V.RELIABILITY_BUNDLE, V.CATALOGUE_MANIFEST, V.CATALOGUE_PATH,
    "research/manifests/dermpact_phase3/verification_v3.json",
    "research/manifests/dermpact_phase4/rule_catalog_v2/verification.json",
    "dermpact/guidelines/reviewed_v2.py", "dermpact/eval/dermpact_phase4_reference_v2.py",
    "dermpact/config.py", "dermpact/eval/dermpact_phase4_arms.py", ".env.example",
    "dermpact/data/corpus/derm_guideline_recommendations_v1.jsonl",
    "dermpact/data/corpus/derm_guideline_recommendations_v1.jsonl.sha256",
    "dermpact/reliability/selected_perception.py", "dermpact/reliability/inference.py", "dermpact/reliability/bundle.py",
    "requirements-p2.lock", "requirements-p2-prototype-additions.txt", "requirements-p2-prototype.in",
    "Planning discussions/PHASE7_EVALUATION_AND_PAPER_UPDATE_SPEC_2026-09-07.md",
    "Planning discussions/PAPER_REWRITE_PLAN_2026-09-07.md",
    "Planning discussions/AUTORESEARCH_AGENT_ADAPTATION_2026-09-07.md",
    "Planning discussions/PRELIVE_SYSTEM_HANDOFF_2026-09-07.md",
    "Planning discussions/LIVE_MODEL_SELECTION_2026-09-07.md",
    "Planning discussions/LIVE_SMOKE_RESULT_2026-09-07.md",
    "Planning discussions/LIVE_FULL_RESULT_2026-09-08.md",
    "Planning discussions/phase6_audits/memory_diagnosis_route_result_2026-09-08.md",
    "Planning discussions/phase2_audits/zero_shot_vlm_ablation_result_2026-09-08.md",
)
#: Result/data artifacts produced by this delegation.
ARTIFACTS = (
    "research/experiments/dermpact_phase6/memory_bank_dev_v1/manifest.json",
    "research/experiments/dermpact_phase6/memory_bank_dev_v1/bank.jsonl",
    "research/experiments/dermpact_phase6/demo_inventory_v1/inventory_manifest.json",
    "research/experiments/dermpact_phase6/xai_feasibility_v1/feasibility.json",
    "research/experiments/dermpact_phase6/xai_feasibility_v1/region_receipt.json",
    "research/experiments/dermpact_phase6/prototype_runs_v2/SUMMARY.md",
    "research/experiments/dermpact_phase6/prototype_runs_v2/CLI_EXAMPLES.md",
    "research/manifests/dermpact_phase6/applicability_benchmark_v2/protocol.json",
    "research/manifests/dermpact_phase6/applicability_benchmark_v2/results.json",
    "research/manifests/dermpact_phase6/amendment_benchmark_v2/protocol.json",
    "research/manifests/dermpact_phase6/amendment_benchmark_v2/curves.json",
    "research/manifests/dermpact_phase6/agent_eval_v1/protocol.json",
    "research/manifests/dermpact_phase6/agent_eval_v1/task_inventory.json",
    "research/manifests/dermpact_phase6/agent_eval_v1/results.json",
    "research/manifests/dermpact_phase6/guideline_library_v1/manifest.json",
    "research/manifests/dermpact_phase6/guideline_library_v1/public_metadata_catalog.jsonl",
    "research/manifests/dermpact_phase6/paper_tables_pre_live/TABLE_REGISTRY.json",
    "research/manifests/dermpact_phase6/paper_tables_pre_live/readiness.json",
    "research/manifests/dermpact_phase6/memory_benchmark_v1/results.json",
    "research/manifests/dermpact_phase6/memory_benchmark_v1/rows.jsonl",
    "research/manifests/dermpact_phase6/agent_live_eval_smoke5_qwen3vl_finalguard_v1/protocol.json",
    "research/manifests/dermpact_phase6/agent_live_eval_smoke5_qwen3vl_finalguard_v1/transcripts.jsonl",
    "research/manifests/dermpact_phase6/agent_live_eval_smoke5_qwen3vl_finalguard_v1/results.json",
    "research/manifests/dermpact_phase6/agent_live_eval_full54_qwen3vl_v3_memorycounts/protocol.json",
    "research/manifests/dermpact_phase6/agent_live_eval_full54_qwen3vl_v3_memorycounts/transcripts.jsonl",
    "research/manifests/dermpact_phase6/agent_live_eval_full54_qwen3vl_v3_memorycounts/results.json",
    "research/manifests/dermpact_phase6/memory_diagnosis_route_v1/results.json",
    "research/manifests/dermpact_phase6/memory_diagnosis_route_v1/oof_predictions.npz",
    "research/manifests/dermpact_phase6/zero_shot_vlm_baseline_v1/results.json",
    "research/experiments/dermpact_phase7/agent_autoresearch_v1/PROGRAM.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _pin(paths: List[Path]) -> Dict[str, str]:
    return {_rel(p): sha256_file(p) for p in sorted(paths, key=lambda q: _rel(q))}


def _source_files() -> List[Path]:
    files: List[Path] = []
    for base, pattern in SOURCE_GLOBS:
        files.extend(p for p in (ROOT / base).glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    return files


def _checkpoint_check() -> Dict[str, Any]:
    selection = json.loads((ROOT / V.SELECTION_MANIFEST).read_text(encoding="utf-8"))
    rows = []
    for binding in selection["artifacts"]["stage2"]["fold_checkpoints"]:
        path = ROOT / binding["path"]
        actual = sha256_file(path) if path.exists() else None
        rows.append({"fold": binding["fold"], "path": binding["path"], "expected_sha256": binding["sha256"],
                     "actual_sha256": actual, "matches": actual == binding["sha256"]})
    return {"primary_model_id": selection.get("primary_model_id"), "n_checkpoints": len(rows),
            "all_match_selection_manifest": all(r["matches"] for r in rows), "checkpoints": rows}


def build_manifest(*, tests_summary: Optional[str] = None) -> Dict[str, Any]:
    existing = [ROOT / rel for rel in ANCHORS if (ROOT / rel).exists()]
    missing_anchors = [rel for rel in ANCHORS if not (ROOT / rel).exists()]
    artifacts = [ROOT / rel for rel in ARTIFACTS if (ROOT / rel).exists()]
    missing_artifacts = [rel for rel in ARTIFACTS if not (ROOT / rel).exists()]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": {name: getattr(V, name) for name in sorted(V.__all__)},
        "policy": {
            "live_llm_status": "deferred by user; adapter gated by DERMPACT_ALLOW_LIVE_LLM=1 and allow_live=True",
            "perception_training": "closed; weights unchanged (see checkpoints)",
            "presentation": "Python package + LangGraph workflow + CLI; no app, server or frontend (scope correction 2026-09-06)",
        },
        "scope_attestation": {
            "kind": "author_and_execution_scope_declaration_not_access_log_telemetry",
            "live_llm_calls_made_in_this_pass": 0,
            "official_or_final_test_rows_loaded_in_this_pass": 0,
            "limitations": ("The manifest verifies bytes and live-call gating; it cannot prove absence of data access "
                            "by unrelated external processes."),
        },
        "checkpoints": _checkpoint_check(),
        "anchors": _pin(existing),
        "missing_anchors": missing_anchors,
        "sources": _pin(_source_files()),
        "artifacts": _pin(artifacts),
        "missing_artifacts": missing_artifacts,
        "tests_summary": tests_summary or "not recorded in this manifest",
        "verification_limit": "Pins bytes only. Does not load datasets or images, run inference, or establish clinical performance.",
    }


def write_manifest(path: Path = OUTPUT, *, tests_summary: Optional[str] = None) -> Dict[str, Any]:
    manifest = build_manifest(tests_summary=tests_summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    tmp = Path(str(path) + ".partial")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
    Path(str(path) + ".sha256").write_text(sha256_file(path) + "\n", encoding="ascii")
    return manifest


def verify_manifest(path: Path = OUTPUT) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    drift: List[Dict[str, str]] = []
    for section in ("anchors", "sources", "artifacts"):
        for rel, expected in manifest[section].items():
            target = ROOT / rel
            if not target.exists():
                drift.append({"path": rel, "section": section, "reason": "missing"})
            elif sha256_file(target) != expected:
                drift.append({"path": rel, "section": section, "reason": "sha256 changed"})
    current_sources = {_rel(p) for p in _source_files()}
    for rel in sorted(current_sources - set(manifest["sources"])):
        drift.append({"path": rel, "section": "sources", "reason": "new source not pinned"})
    checkpoints = _checkpoint_check()
    return {"status": "PASS" if not drift and checkpoints["all_match_selection_manifest"] else "DRIFT",
            "n_pinned": sum(len(manifest[s]) for s in ("anchors", "sources", "artifacts")),
            "checkpoints_match": checkpoints["all_match_selection_manifest"], "drift": drift}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="re-hash everything pinned and report drift")
    parser.add_argument("--tests-summary", default=None, help="free-text test summary to record (e.g. pytest tail line)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_manifest(args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    manifest = write_manifest(args.output, tests_summary=args.tests_summary)
    print(json.dumps({"written": str(args.output), "n_sources": len(manifest["sources"]),
                      "n_anchors": len(manifest["anchors"]), "n_artifacts": len(manifest["artifacts"]),
                      "checkpoints_match": manifest["checkpoints"]["all_match_selection_manifest"],
                      "missing_anchors": manifest["missing_anchors"], "missing_artifacts": manifest["missing_artifacts"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
