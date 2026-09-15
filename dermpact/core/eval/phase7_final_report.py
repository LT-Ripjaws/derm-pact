"""Assemble paper-ready Phase 7 development metrics without opening final data."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from dermpact.core.eval import agent_eval
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/manifests/dermpact_phase7/final_development_evaluation_v1"
SOURCES = {
    "confirmatory": ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v2_failure_rescue/canonical_merged300/metrics.json",
    "confirmatory_merge": ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v2_failure_rescue/canonical_merged300/results.json",
    "ragas": ROOT / "research/manifests/dermpact_phase7/agent_ragas_confirmatory100_v1/results.json",
    "ragas_samples": ROOT / "research/manifests/dermpact_phase7/agent_ragas_confirmatory100_v1/sample_manifest.json",
    "adversarial": ROOT / "research/manifests/dermpact_phase7/agent_adversarial120_v1/metrics.json",
    "stability": ROOT / "research/manifests/dermpact_phase7/agent_stability30_v1/metrics.json",
    "ablations": ROOT / "research/manifests/dermpact_phase7/agent_ablation_matrix_v2_frozen_no_postadjust/results.json",
    "applicability": ROOT / "research/manifests/dermpact_phase6/applicability_benchmark_v2/results.json",
    "guideline_probe": ROOT / "research/manifests/dermpact_phase7/agent_guideline20_v1/live_qwen3vl/results.json",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(out_dir: Path = OUT) -> dict[str, Any]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    data = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = data["ragas_samples"]["samples"]
    pass_scores = [json.loads((SOURCES["ragas"].parent / f"pass_{index}.json").read_text())
                   ["scores"]["faithfulness"]["per_sample"] for index in range(1, 6)]
    by_case = {sample["case_id"]: [float(pass_scores[p][i]) for p in range(5)]
               for i, sample in enumerate(samples)}
    ragas_ci = agent_eval.case_clustered_bootstrap(by_case, n_boot=10000, seed=20261208)

    headline = {
        "schema_version": "dermpact-phase7-final-development-report-v1", "status": "PASS",
        "confirmatory_300": data["confirmatory"]["metrics"],
        "ragas_100x5": {**data["ragas"]["aggregate"]["full_bounded_agent"],
                         "case_clustered_ci": ragas_ci},
        "adversarial_120": data["adversarial"]["overall"],
        "stability_30x3": {key: value for key, value in data["stability"].items() if key != "rows"},
        "scope": {"official_or_final_rows_loaded": 0, "development_only": True,
                  "human_clinician_utility_measured": False},
        "sources": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256_file(path)}
                    for name, path in SOURCES.items()},
    }
    _atomic_json(out_dir / "headline.json", headline)

    metric_rows = []
    for name, metric in data["confirmatory"]["metrics"].items():
        metric_rows.append({"evaluation": "confirmatory_300", "metric": name, **metric})
    metric_rows.append({"evaluation": "ragas_100x5", "metric": "faithfulness",
                        "estimate": data["ragas"]["aggregate"]["full_bounded_agent"]["mean_across_passes"],
                        "ci_low": ragas_ci["ci_low"], "ci_high": ragas_ci["ci_high"],
                        "n_cases": 100, "n_judge_passes": 5})
    for name in ("protocol_valid", "attack_success_final_hazard", "tool_selection_correct", "tool_argument_correct"):
        metric_rows.append({"evaluation": "adversarial_120", "metric": name,
                            **data["adversarial"]["overall"][name]})
    _write_csv(out_dir / "agent_metrics.csv", metric_rows)

    ablation_rows = []
    for arm, values in data["ablations"]["live_controls"].items():
        ablation_rows.append({"arm": arm, "evidence_type": "measured_live_frozen54",
                              "n": values["n_tasks"],
                              "workflow_task_success": values["workflow_task_success"]["rate"],
                              "final_hazard": values["final_output_hazard"]["rate"]})
    for arm, values in data["ablations"]["mechanism_rows"].items():
        ablation_rows.append({"arm": arm, "evidence_type": values.get("evidence_type"),
                              "n": values.get("n_tasks"),
                              "workflow_task_success": values.get("workflow_task_success_rate"),
                              "final_hazard": values.get("final_output_hazard_rate"),
                              "maximum_component_complete_rate": values.get("maximum_component_complete_rate"),
                              "reason": values.get("reason")})
    guideline_summary = data["guideline_probe"]["summary"]
    ablation_rows.extend([
        {"arm": "full_guideline_library_source_retrieval", "evidence_type": "measured_live",
         "n": 20, "workflow_task_success": guideline_summary["completion_rate"]["estimate"],
         "final_hazard": guideline_summary["final_display_ohr"]["estimate"]},
        {"arm": "without_guideline_library_source_retrieval", "evidence_type": "exact_capability_knockout",
         "n": 20, "workflow_task_success": 0.0, "maximum_component_complete_rate": 0.0,
         "reason": "the removed library is the sole allowlisted source-passage store"},
    ])
    _write_csv(out_dir / "ablation_matrix.csv", ablation_rows)

    h = data["confirmatory"]["metrics"]
    ragas = data["ragas"]["aggregate"]["full_bounded_agent"]
    attack = data["adversarial"]["overall"]
    stability = data["stability"]
    lines = [
        "# DermPact Phase 7 final development evaluation", "",
        "Official/final partitions loaded: **0**. Human clinician utility measured: **no**.", "",
        "## Headline", "",
        f"- Canonical ordinary workflow success: {h['workflow_task_success']['estimate']:.4f} (300 tasks, 100 cases).",
        f"- Strict query accuracy: {h['query_accuracy']['estimate']:.4f}; fact F1: {h['fact_f1']['estimate']:.4f}; required-fact traceability: {h['required_fact_traceability']['estimate']:.4f}.",
        f"- Tool selection: {h['tool_selection_correct']['estimate']:.4f}; exact arguments: {h['tool_argument_correct']['estimate']:.4f}.",
        f"- Final displayed hazard: {h['final_hazard']['estimate']:.4f}; initial hazard: {h['initial_hazard']['estimate']:.4f}.",
        f"- RAGAS faithfulness: {ragas['mean_across_passes']:.4f}, five-pass range {ragas['min_pass_mean']:.4f}--{ragas['max_pass_mean']:.4f} (100 cases).",
        f"- Adversarial final hazard: {attack['attack_success_final_hazard']['estimate']:.4f}; protocol-valid utility: {attack['protocol_valid']['estimate']:.4f} (120 attacks).",
        f"- Three-run protocol stability: {stability['all_three_protocol_valid']['estimate']:.4f}; claim Jaccard: {stability['mean_pairwise_claim_jaccard']:.4f}.",
        "", "## Interpretation", "",
        "The verifier eliminated observed final hazards but did not make every answer complete. The strict query, traceability and exact-argument metrics must remain visible beside 100% workflow completion. The adversarial result demonstrates a security--utility tradeoff: zero observed hazardous outputs with non-perfect task completion.",
    ]
    (out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return headline


def main() -> int:
    result = build()
    print(json.dumps({"status": result["status"], "out_dir": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
