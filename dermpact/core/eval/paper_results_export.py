"""Export Phase 6/7 machine-readable paper tables from frozen result files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase6/paper_tables_pre_live"

SOURCES = {
    "applicability": ROOT / "research/manifests/dermpact_phase6/applicability_benchmark_v2/results.json",
    "amendment": ROOT / "research/manifests/dermpact_phase6/amendment_benchmark_v2/curves.json",
    "agent_offline": ROOT / "research/manifests/dermpact_phase6/agent_eval_v1/results.json",
    "guidelines": ROOT / "research/manifests/dermpact_phase6/guideline_library_v1/manifest.json",
    "memory": ROOT / "research/experiments/dermpact_phase6/memory_bank_dev_v1/manifest.json",
    "memory_benchmark": ROOT / "research/manifests/dermpact_phase6/memory_benchmark_v1/results.json",
    "memory_diagnosis": ROOT / "research/manifests/dermpact_phase6/memory_diagnosis_route_v1/results.json",
    "zero_shot_vlm": ROOT / "research/manifests/dermpact_phase6/zero_shot_vlm_baseline_v1/results.json",
}
LIVE_SOURCE = ROOT / "research/manifests/dermpact_phase6/agent_live_eval_full54_qwen3vl_v3_memorycounts/results.json"
LIVE_SMOKE_SOURCE = ROOT / "research/manifests/dermpact_phase6/agent_live_eval_smoke5_qwen3vl_finalguard_v1/results.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ci_fields(metric: Mapping[str, Any]) -> Mapping[str, Any]:
    boundary = metric.get("boundary_case_bound_one_sided_95") or {}
    return {
        "estimate": metric.get("estimate"), "ci_low": metric.get("ci_low"), "ci_high": metric.get("ci_high"),
        "n_cases": metric.get("n_cases"), "boundary_kind": boundary.get("kind"), "boundary_value": boundary.get("value"),
    }


def export(out_dir: Path = OUTPUT_DIR) -> Mapping[str, Any]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"paper-table sources missing: {missing}")
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    applicability_rows = []
    for arm, values in data["applicability"]["arms"].items():
        if "primary_ci" in values:
            applicability_rows.append({"arm": arm, "n_runs": values.get("n_runs"), **_ci_fields(values["primary_ci"])})
    _write_csv(out_dir / "applicability.csv", applicability_rows)

    agent_rows = []
    for arm, values in data["agent_offline"]["arms"].items():
        latency = values.get("latency") or {}
        agent_rows.append({
            "arm": arm, "n_tasks": values["n_tasks"], **_ci_fields(values["completion_rate"]),
            "false_action_claims": values["false_action_claim_count"],
            "initial_candidate_ohr": values["initial_candidate_ohr"]["estimate"],
            "final_display_ohr": values["final_display_ohr"]["estimate"],
            "unsupported_claim_rejections": values["unsupported_claim_rejection_count"],
            "wrong_case_scope": values["wrong_case_scope_count"],
            "mean_tool_calls": values["tool_calls"]["mean"], "mean_turns": values["turns"]["mean"],
            "provider_latency_mean_s": (latency.get("provider_s") or {}).get("mean"),
            "end_to_end_latency_mean_s": (latency.get("end_to_end_s") or {}).get("mean"),
            "end_to_end_latency_p95_s": (latency.get("end_to_end_s") or {}).get("p95"),
        })
    if LIVE_SMOKE_SOURCE.is_file():
        smoke = json.loads(LIVE_SMOKE_SOURCE.read_text(encoding="utf-8"))
        values = smoke["summary"]
        latency = values.get("latency") or {}
        agent_rows.append({
            "arm": "bounded_agent_live_smoke5_not_full_eval", "n_tasks": values["n_tasks"],
            **_ci_fields(values["completion_rate"]),
            "false_action_claims": values["false_action_claim_count"],
            "initial_candidate_ohr": values["initial_candidate_ohr"]["estimate"],
            "final_display_ohr": values["final_display_ohr"]["estimate"],
            "unsupported_claim_rejections": values["unsupported_claim_rejection_count"],
            "wrong_case_scope": values["wrong_case_scope_count"],
            "mean_tool_calls": values["tool_calls"]["mean"], "mean_turns": values["turns"]["mean"],
            "provider_latency_mean_s": (latency.get("provider_s") or {}).get("mean"),
            "end_to_end_latency_mean_s": (latency.get("end_to_end_s") or {}).get("mean"),
            "end_to_end_latency_p95_s": (latency.get("end_to_end_s") or {}).get("p95"),
        })
    if LIVE_SOURCE.is_file():
        live = json.loads(LIVE_SOURCE.read_text(encoding="utf-8"))
        values = live["summary"]
        latency = values.get("latency") or {}
        agent_rows.append({
            "arm": "bounded_agent_live", "n_tasks": values["n_tasks"], **_ci_fields(values["completion_rate"]),
            "false_action_claims": values["false_action_claim_count"],
            "initial_candidate_ohr": values["initial_candidate_ohr"]["estimate"],
            "final_display_ohr": values["final_display_ohr"]["estimate"],
            "unsupported_claim_rejections": values["unsupported_claim_rejection_count"],
            "wrong_case_scope": values["wrong_case_scope_count"],
            "mean_tool_calls": values["tool_calls"]["mean"], "mean_turns": values["turns"]["mean"],
            "provider_latency_mean_s": (latency.get("provider_s") or {}).get("mean"),
            "end_to_end_latency_mean_s": (latency.get("end_to_end_s") or {}).get("mean"),
            "end_to_end_latency_p95_s": (latency.get("end_to_end_s") or {}).get("p95"),
        })
    _write_csv(out_dir / "agent_performance.csv", agent_rows)

    amendment = data["amendment"]
    amendment_rows = [
        {"endpoint": "assertion_truth_aligned_matches_gold", **_ci_fields(amendment["assertion_route"]["truth_aligned_matches_gold"])},
        {"endpoint": "assertion_truth_negated_matches_gold", **_ci_fields(amendment["assertion_route"]["truth_negated_matches_gold"])},
        {"endpoint": "preview_head_unchanged", **_ci_fields(amendment["preview_not_commit"]["head_unchanged_rate"])},
    ]
    amendment_rows.extend({"endpoint": f"structural_{name}", **_ci_fields(metric)}
                          for name, metric in amendment["structural_blocks"].items())
    _write_csv(out_dir / "amendment_integrity.csv", amendment_rows)

    guidelines = data["guidelines"]
    memory = data["memory"]
    memory_benchmark = data["memory_benchmark"]
    _write_csv(out_dir / "memory_performance.csv", [{
        "n_queries": memory_benchmark["n_queries"], "k": memory_benchmark["k"],
        "status_ok_rate": memory_benchmark["status_ok_rate"]["estimate"],
        "diagnosis_label_precision_at_k": memory_benchmark["diagnosis_label_precision_at_k"]["estimate"],
        "diagnosis_label_recall_at_k": memory_benchmark["diagnosis_label_recall_at_k"]["estimate"],
        "counterexample_available_rate": memory_benchmark["counterexample_available_rate"]["estimate"],
        "latency_mean_ms": memory_benchmark["retrieval_latency_ms"]["mean"],
        "latency_p95_ms": memory_benchmark["retrieval_latency_ms"]["p95"],
        "interpretation": memory_benchmark["interpretation"],
    }])
    memory_diagnosis = data["memory_diagnosis"]
    _write_csv(out_dir / "memory_diagnosis_route.csv", [{
        "decision": memory_diagnosis["decision"],
        "promoted": memory_diagnosis["promoted"],
        "base_macro_ovr_auroc": memory_diagnosis["base_calibrated"]["macro_ovr_auroc"],
        "fused_macro_ovr_auroc": memory_diagnosis["memory_fused_calibrated"]["macro_ovr_auroc"],
        "delta": memory_diagnosis["macro_ovr_auroc_delta"]["estimate"],
        "delta_ci_low": memory_diagnosis["macro_ovr_auroc_delta"]["ci_low"],
        "delta_ci_high": memory_diagnosis["macro_ovr_auroc_delta"]["ci_high"],
        "base_accuracy": memory_diagnosis["base_calibrated"]["accuracy"],
        "fused_accuracy": memory_diagnosis["memory_fused_calibrated"]["accuracy"],
        "base_balanced_accuracy": memory_diagnosis["base_calibrated"]["balanced_accuracy"],
        "fused_balanced_accuracy": memory_diagnosis["memory_fused_calibrated"]["balanced_accuracy"],
        "memory_nonzero_alpha_rate": memory_diagnosis["activation"]["nonzero_alpha_rate"],
    }])
    vlm = data["zero_shot_vlm"]
    perception_rows = []
    for row_name in ("selected_phase2_5", "dermlip_zero_shot_vlm"):
        values = vlm["rows"][row_name]
        intervals = vlm["confidence_intervals"][row_name]
        perception_rows.append({
            "arm": row_name, "n_cases": vlm["n_cases_scored"],
            "accuracy": values["accuracy"],
            "accuracy_ci_low": intervals["accuracy"]["ci_low"],
            "accuracy_ci_high": intervals["accuracy"]["ci_high"],
            "balanced_accuracy": values["balanced_accuracy"],
            "balanced_accuracy_ci_low": intervals["balanced_accuracy"]["ci_low"],
            "balanced_accuracy_ci_high": intervals["balanced_accuracy"]["ci_high"],
            "macro_f1": values["macro_f1"], "macro_ovr_auroc": values["macro_ovr_auroc"],
            "macro_ovr_auroc_ci_low": intervals["macro_ovr_auroc"]["ci_low"],
            "macro_ovr_auroc_ci_high": intervals["macro_ovr_auroc"]["ci_high"],
            "melanoma_auroc": values["melanoma_auroc"], "nll": values["nll"],
            "multiclass_brier": values["multiclass_brier"], "ece_10bin": values["ece_10bin"],
        })
    _write_csv(out_dir / "perception_ablation.csv", perception_rows)
    readiness = {
        "guideline_retrieval_records": guidelines["layers"]["retrieval_library"]["n_records"],
        "guideline_reviewed_records": guidelines["layers"]["reviewed_v2_next_step_catalogue"]["n_records"],
        "guideline_executable_records": guidelines["layers"]["reviewed_v2_next_step_catalogue"]["status_counts"].get("executable", 0),
        "memory_development_cases": memory["n_development_cases"], "memory_embedded_cases": memory["n_embedded"],
        "memory_min_similarity": memory["threshold_derivation"]["min_similarity"],
        "memory_counterexample_min_similarity": memory["threshold_derivation"]["counterexample_min_similarity"],
        "memory_match_coverage": memory_benchmark["status_ok_rate"]["estimate"],
        "memory_counterexample_available_rate": memory_benchmark["counterexample_available_rate"]["estimate"],
        "memory_diagnosis_decision": memory_diagnosis["decision"],
        "memory_diagnosis_promoted": memory_diagnosis["promoted"],
        "live_results_present": LIVE_SOURCE.is_file(),
        "live_smoke_status": (json.loads(LIVE_SMOKE_SOURCE.read_text(encoding="utf-8"))["status"]
                              if LIVE_SMOKE_SOURCE.is_file() else "not_run"),
        "official_final_results_present": False,
    }
    (out_dir / "readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Development paper tables", "", "Generated from hash-bound result artifacts; no official/final-test result is included.", "",
             "## Agent controls", "",
             "| arm | n | completion | 95% bootstrap CI | one-sided boundary | initial OHR | final OHR | false actions | mean tools | mean turns |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in agent_rows:
        boundary = (f"{row['boundary_kind']} {row['boundary_value']:.4f}" if row["boundary_value"] is not None else "n/a")
        lines.append(f"| {row['arm']} | {row['n_tasks']} | {row['estimate']:.4f} | "
                     f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {boundary} | "
                     f"{row['initial_candidate_ohr']:.4f} | {row['final_display_ohr']:.4f} | "
                     f"{row['false_action_claims']} | {row['mean_tool_calls']:.2f} | {row['mean_turns']:.2f} |")
    lines += ["", "## Readiness", ""] + [f"- {key}: `{value}`" for key, value in readiness.items()]
    (out_dir / "TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    source_paths = dict(SOURCES)
    if LIVE_SOURCE.is_file():
        source_paths["agent_live"] = LIVE_SOURCE
    if LIVE_SMOKE_SOURCE.is_file():
        source_paths["agent_live_smoke5"] = LIVE_SMOKE_SOURCE
    registry = {
        "schema_version": "dermpact-paper-table-export-v1",
        "sources": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}
                    for name, path in source_paths.items()},
        "outputs": {path.name: _sha(path) for path in sorted(out_dir.iterdir())
                    if path.is_file() and path.name != "TABLE_REGISTRY.json"},
    }
    (out_dir / "TABLE_REGISTRY.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    registry = export(args.out_dir)
    print(json.dumps({"status": "PASS", "n_sources": len(registry["sources"]),
                      "n_outputs": len(registry["outputs"]), "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
