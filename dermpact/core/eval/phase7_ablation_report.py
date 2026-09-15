"""Generate manuscript-ready Phase-7 ablation and control tables from frozen artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/manifests/dermpact_phase7"
OUT = BASE / "paper_ablation_tables_v1"
PATHS = {
    "full300": BASE / "agent_confirmatory300_v2_grounded_reads/live_qwen3vl/results.json",
    "controls100": BASE / "agent_ragas_controls100_v1/results.json",
    "ragas100": BASE / "agent_ragas_matched100_v2/results.json",
    "ragas_stats": BASE / "agent_ragas_matched100_v2/paired_statistics.json",
    "memory100": BASE / "agent_memory_ablation100_v1/results.json",
    "persistence30": BASE / "agent_persistence30_v4_final_matched/results.json",
    "clarification100": BASE / "clarification_ablation100_v1/results.json",
    "xai10": BASE / "agent_xai_ablation10_v1/results.json",
    "applicability100": BASE / "applicability_live100_v1/results.json",
    "final54": BASE / "agent_final54_qualification_v1/results.json",
}


def _load(name: str) -> dict[str, Any]:
    path = PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(f"required frozen artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(out_dir: Path = OUT) -> dict[str, Any]:
    full300 = _load("full300")
    controls = _load("controls100")["live_controls"]
    ragas = _load("ragas100")
    ragas_stats = _load("ragas_stats")
    memory = _load("memory100")["summaries"]
    persistence = _load("persistence30")["summaries"]
    clarification = _load("clarification100")["arms"]
    xai = _load("xai10")["summaries"]
    applicability = _load("applicability100")["summaries"]
    final54 = _load("final54")

    f54 = final54["summary"]
    initial_hazard = float(f54["initial_candidate_ohr"]["estimate"])
    full54_success = float(f54["completion_rate"]["estimate"])
    full54_hazard = float(f54["final_display_ohr"]["estimate"])
    mem_full = memory["full_bounded_agent"]
    mem_none = memory["without_comparison_memory"]
    mem_counter = memory["without_memory_counterexample"]
    pers_full = persistence["full_bounded_agent"]
    pers_none = persistence["without_persistent_restart_state"]
    xai_full = xai["full_bounded_agent"]
    xai_none = xai["without_measured_xai"]

    ablations = [
        {
            "component_removed": "checker",
            "matched_subset": "frozen broad agent inventory",
            "n_cases": f54["completion_rate"]["n_cases"],
            "n_tasks": f54["n_tasks"],
            "full_task_success": full54_success,
            "ablated_task_success": 1.0 - initial_hazard,
            "delta_success": -initial_hazard,
            "full_hazard_rate": full54_hazard,
            "ablated_hazard_rate": initial_hazard,
            "full_mean_latency_s": f54["latency"]["end_to_end_s"]["mean"],
            "ablated_mean_latency_s": None,
            "evidence_type": "exact pre-check candidate counterfactual; unsafe arm not displayed live",
        },
        {
            "component_removed": "one repair pass (zero repair)",
            "matched_subset": "frozen broad agent inventory",
            "n_cases": f54["completion_rate"]["n_cases"],
            "n_tasks": f54["n_tasks"],
            "full_task_success": full54_success,
            "ablated_task_success": 1.0 - initial_hazard,
            "delta_success": -initial_hazard,
            "full_hazard_rate": full54_hazard,
            "ablated_hazard_rate": 0.0,
            "full_mean_latency_s": f54["latency"]["end_to_end_s"]["mean"],
            "ablated_mean_latency_s": None,
            "evidence_type": "exact checker counterfactual; rejected candidates fall back safely",
        },
        {
            "component_removed": "comparison memory",
            "matched_subset": "comparison requests",
            "n_cases": 100,
            "n_tasks": 100,
            "full_task_success": mem_full["completion_rate"]["estimate"],
            "ablated_task_success": mem_none["completion_rate"]["estimate"],
            "delta_success": mem_none["completion_rate"]["estimate"] - mem_full["completion_rate"]["estimate"],
            "full_hazard_rate": mem_full["final_display_ohr"]["estimate"],
            "ablated_hazard_rate": mem_none["final_display_ohr"]["estimate"],
            "full_mean_latency_s": mem_full["latency"]["end_to_end_s"]["mean"],
            "ablated_mean_latency_s": mem_none["latency"]["end_to_end_s"]["mean"],
            "evidence_type": "matched live component knockout",
        },
        {
            "component_removed": "discordant memory counterexample",
            "matched_subset": "comparison requests",
            "n_cases": 100,
            "n_tasks": 100,
            "full_task_success": mem_full["completion_rate"]["estimate"],
            "ablated_task_success": mem_counter["completion_rate"]["estimate"],
            "delta_success": mem_counter["completion_rate"]["estimate"] - mem_full["completion_rate"]["estimate"],
            "full_hazard_rate": mem_full["final_display_ohr"]["estimate"],
            "ablated_hazard_rate": mem_counter["final_display_ohr"]["estimate"],
            "full_mean_latency_s": mem_full["latency"]["end_to_end_s"]["mean"],
            "ablated_mean_latency_s": mem_counter["latency"]["end_to_end_s"]["mean"],
            "evidence_type": "matched live endpoint knockout",
        },
        {
            "component_removed": "measured XAI",
            "matched_subset": "region/clinical-contribution requests",
            "n_cases": xai_full["n_unique_cases"],
            "n_tasks": xai_full["n_tasks"],
            "full_task_success": xai_full["frozen_protocol_success_rate"],
            "ablated_task_success": xai_none["frozen_protocol_success_rate"],
            "delta_success": xai_none["frozen_protocol_success_rate"] - xai_full["frozen_protocol_success_rate"],
            "full_hazard_rate": xai_full["final_hazard_rate"],
            "ablated_hazard_rate": xai_none["final_hazard_rate"],
            "full_mean_latency_s": xai_full["mean_end_to_end_seconds"],
            "ablated_mean_latency_s": xai_none["mean_end_to_end_seconds"],
            "evidence_type": "matched live capability knockout; feasibility sample reported separately",
        },
        {
            "component_removed": "persistent episode/checkpoint state",
            "matched_subset": "restart-resume amendment workflows",
            "n_cases": 30,
            "n_tasks": 30,
            "full_task_success": pers_full["completion_rate"]["estimate"],
            "ablated_task_success": pers_none["completion_rate"]["estimate"],
            "delta_success": pers_none["completion_rate"]["estimate"] - pers_full["completion_rate"]["estimate"],
            "full_hazard_rate": pers_full["final_display_ohr"]["estimate"],
            "ablated_hazard_rate": pers_none["final_display_ohr"]["estimate"],
            "full_mean_latency_s": pers_full["latency"]["end_to_end_s"]["mean"],
            "ablated_mean_latency_s": pers_none["latency"]["end_to_end_s"]["mean"],
            "evidence_type": "matched restart-state knockout",
        },
        {
            "component_removed": "clarification targeting",
            "matched_subset": "records with unknown clinician facts",
            "n_cases": 100,
            "n_tasks": 100,
            "full_task_success": clarification["full_targeted_clarification"]["decisive_route_coverage"],
            "ablated_task_success": clarification["without_clarification_targeting_ask_all"]["decisive_route_coverage"],
            "delta_success": 0.0,
            "full_hazard_rate": None,
            "ablated_hazard_rate": None,
            "full_mean_latency_s": None,
            "ablated_mean_latency_s": None,
            "evidence_type": "deterministic matched burden/relevance ablation",
        },
    ]

    capabilities = [
        {"component": "comparison memory", "clinician_facing_endpoint": "eligible historical comparison available",
         "n_cases": 100, "full": mem_full["comparison_available_rate"], "ablated": mem_none["comparison_available_rate"],
         "claim_boundary": "descriptive precedent only; does not alter diagnosis"},
        {"component": "discordant counterexample", "clinician_facing_endpoint": "discordant precedent displayed",
         "n_cases": 100, "full": mem_full["counterexample_display_rate"], "ablated": mem_counter["counterexample_display_rate"],
         "claim_boundary": "availability, not clinician outcome benefit"},
        {"component": "measured XAI", "clinician_facing_endpoint": "host-verified sensitivity receipt available",
         "n_cases": xai_full["n_unique_cases"], "full": xai_full["measured_receipt_available_rate"],
         "ablated": xai_none["measured_receipt_available_rate"], "claim_boundary": "intervention sensitivity, not causal attribution"},
        {"component": "persistent state", "clinician_facing_endpoint": "restart-resume-confirm-commit continuity",
         "n_cases": 30, "full": pers_full["completion_rate"]["estimate"], "ablated": pers_none["completion_rate"]["estimate"],
         "claim_boundary": "workflow continuity, not diagnostic accuracy"},
        {"component": "clarification targeting", "clinician_facing_endpoint": "mean questions presented per eligible case",
         "n_cases": 100, "full": clarification["full_targeted_clarification"]["mean_questions_presented"],
         "ablated": clarification["without_clarification_targeting_ask_all"]["mean_questions_presented"],
         "claim_boundary": "burden proxy; no reader study"},
    ]

    control_rows = []
    ragas_agg = ragas["aggregate"]
    full_summary = full300["summary"]
    for name, label in (("full_bounded_agent", "DermPact"),
                        ("same_backbone_no_tools", "Bare Qwen (same backbone, no tools)"),
                        ("naive_static_rag", "Naive guideline RAG")):
        if name == "full_bounded_agent":
            success = full_summary["completion_rate"]["estimate"]
            hazard = full_summary["final_display_ohr"]["estimate"]
            latency = full_summary["latency"]["end_to_end_s"]["mean"]
        else:
            success = controls[name]["workflow_task_success"]["rate"]
            hazard = controls[name]["final_output_hazard"]["rate"]
            latency = controls[name]["provider_latency_s"]["mean"]
        control_rows.append({
            "system": label, "n_cases": 100, "workflow_success": success,
            "final_hazard_rate": hazard, "ragas_grounded_quality": ragas_agg[name]["mean_across_passes"],
            "mean_latency_s": latency,
        })

    applicability_rows = [
        {"system": name, **summary}
        for name, summary in applicability.items()
    ]
    coverage = [
        {"planned_arm": "deterministic forms/template", "status": "complete", "evidence": "Phase-6 compiled-graph deterministic-forms control"},
        {"planned_arm": "bare Qwen", "status": "complete", "evidence": "100-case matched live control"},
        {"planned_arm": "naive guideline RAG", "status": "complete", "evidence": "100-case matched live control"},
        {"planned_arm": "LLM-guided rule traversal", "status": "complete", "evidence": "100-case matched applicability control"},
        {"planned_arm": "bounded agent without memory", "status": "complete", "evidence": "100-case matched live knockout"},
        {"planned_arm": "full bounded agent", "status": "complete", "evidence": "300 tasks over 100 cases plus 54-task breadth qualification"},
        {"planned_arm": "without checker", "status": "complete", "evidence": "exact unsafe pre-check counterfactual; no live display"},
        {"planned_arm": "without clarification targeting", "status": "complete", "evidence": "100-case deterministic matched burden ablation"},
        {"planned_arm": "without persistent episode state", "status": "complete", "evidence": "30-case matched restart knockout"},
        {"planned_arm": "without counterexamples", "status": "complete", "evidence": "100-case matched live endpoint knockout"},
        {"planned_arm": "without measured XAI", "status": "complete", "evidence": "five-task matched live capability knockout; two unique cases"},
        {"planned_arm": "zero versus one repair pass", "status": "complete", "evidence": "exact checker/fallback counterfactual"},
    ]
    out_dir = Path(out_dir)
    _write_csv(out_dir / "main_component_ablation.csv", ablations)
    _write_csv(out_dir / "clinician_facing_capabilities.csv", capabilities)
    _write_csv(out_dir / "matched_agent_controls.csv", control_rows)
    _atomic_json(out_dir / "applicability_controls.json", applicability_rows)
    result = {
        "schema_version": "dermpact-phase7-paper-ablation-tables-v1",
        "status": "PASS",
        "tables": {
            "main_component_ablation": ablations,
            "clinician_facing_capabilities": capabilities,
            "matched_agent_controls": control_rows,
            "applicability_controls": applicability_rows,
            "planned_ablation_coverage": coverage,
        },
        "paired_ragas_statistics": ragas_stats,
        "notes": [
            "Unsafe checker removal is an exact pre-check counterfactual and was not exposed live.",
            "Component-specific rows use the frozen subset on which the component can activate; n is explicit.",
            "No row is interpreted as independent clinician outcome benefit without a reader study.",
        ],
    }
    _atomic_json(out_dir / "results.json", result)
    lines = [
        "# Phase 7 agent ablations — frozen development results",
        "",
        "## Matched agent controls",
        "",
        "| System | Cases | Workflow success | Final hazard | RAGAS | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in control_rows:
        lines.append(
            f"| {row['system']} | {row['n_cases']} | {row['workflow_success']:.3f} | "
            f"{row['final_hazard_rate']:.3f} | {row['ragas_grounded_quality']:.3f} | {row['mean_latency_s']:.3f} |"
        )
    lines += [
        "",
        "## Component knockouts",
        "",
        "| Removed component | Matched subset | Cases | Full success | Ablated success | Full hazard | Ablated hazard |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ablations:
        full_h = "—" if row["full_hazard_rate"] is None else f"{row['full_hazard_rate']:.3f}"
        abl_h = "—" if row["ablated_hazard_rate"] is None else f"{row['ablated_hazard_rate']:.3f}"
        lines.append(
            f"| {row['component_removed']} | {row['matched_subset']} | {row['n_cases']} | "
            f"{row['full_task_success']:.3f} | {row['ablated_task_success']:.3f} | {full_h} | {abl_h} |"
        )
    lines += [
        "",
        "## Clinician-facing component endpoints",
        "",
        "| Component | Endpoint | Cases | Full | Ablated | Claim boundary |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in capabilities:
        lines.append(
            f"| {row['component']} | {row['clinician_facing_endpoint']} | {row['n_cases']} | "
            f"{row['full']:.3f} | {row['ablated']:.3f} | {row['claim_boundary']} |"
        )
    lines += [
        "",
        "The 54-task breadth run complements rather than replaces the 300-task/100-case primary qualification. "
        "Unsafe checker removal is evaluated as an exact pre-display counterfactual. XAI evidence is a capability/feasibility result, not a clinician reader study.",
        "",
    ]
    _atomic_write(out_dir / "RESULT.md", "\n".join(lines))
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
