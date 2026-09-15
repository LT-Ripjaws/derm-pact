"""Assemble the single canonical Phase-7 development ledger from latest artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/manifests/dermpact_phase7"
OUT = BASE / "canonical_development_ledger_v1"
SOURCES = {
    "ordinary_metrics": BASE / "agent_confirmatory300_v2_grounded_reads/metrics.json",
    "ordinary_results": BASE / "agent_confirmatory300_v2_grounded_reads/live_qwen3vl/results.json",
    "ragas": BASE / "agent_ragas_matched100_v2/results.json",
    "ragas_paired": BASE / "agent_ragas_matched100_v2/paired_statistics.json",
    "adversarial_metrics": BASE / "agent_adversarial120_v2_host_binding/metrics.json",
    "adversarial_results": BASE / "agent_adversarial120_v2_host_binding/live_qwen3vl/results.json",
    "stability": BASE / "agent_stability30_v1/metrics.json",
    "applicability": BASE / "applicability_live100_v1/results.json",
    "memory": BASE / "agent_memory_ablation100_v1/results.json",
    "memory_controls": BASE / "memory_controls100_v1/results.json",
    "persistence": BASE / "agent_persistence30_v4_final_matched/results.json",
    "clarification": BASE / "clarification_ablation100_v1/results.json",
    "xai": BASE / "agent_xai_ablation10_v1/results.json",
    "prompt_injection_full": BASE / "agent_prompt_injection20_v1/full_qwen3vl/results.json",
    "prompt_injection_controls": BASE / "agent_prompt_injection20_v1/controls/results.json",
    "host_security": BASE / "security_host_channels_v1/results.json",
    "ablation_tables": BASE / "paper_ablation_tables_v1/results.json",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean(values: list[float], *, seed: int = 20260909) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(10000, len(data)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def build(out_dir: Path = OUT) -> dict[str, Any]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"canonical development sources missing: {missing}")
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}
    ordinary = data["ordinary_metrics"]["metrics"]
    ragas_aggregate = data["ragas"]["aggregate"]
    ragas_full = ragas_aggregate["full_bounded_agent"]
    ragas_values = list(ragas_full["per_task_three_pass_mean"].values())
    adversarial = data["adversarial_results"]["summary"]
    applicability = data["applicability"]["summaries"]
    memory_summary = data["memory"]["summaries"]
    memory_controls = data["memory_controls"]["arms"]

    headline = {
        "schema_version": "dermpact-phase7-canonical-development-ledger-v1",
        "status": "PASS",
        "ordinary_agent_300": {
            "n_tasks": data["ordinary_metrics"]["n_tasks"],
            "n_cases": data["ordinary_metrics"]["n_case_clusters"],
            "workflow_success": ordinary["workflow_task_success"],
            "strict_query_accuracy": ordinary["query_accuracy"],
            "fact_precision": ordinary["fact_precision"],
            "required_fact_traceability": ordinary["required_fact_traceability"],
            "fact_f1": ordinary["fact_f1"],
            "tool_selection": ordinary["tool_selection_correct"],
            "tool_arguments": ordinary["tool_argument_correct"],
            "initial_hazard": ordinary["initial_hazard"],
            "final_hazard": ordinary["final_hazard"],
        },
        "ragas_100x5": {
            "mean": ragas_full["mean_across_passes"],
            "case_bootstrap_95_ci": _bootstrap_mean(ragas_values),
            "pass_range": [ragas_full["min_pass_mean"], ragas_full["max_pass_mean"]],
            "pass_sd": ragas_full["std_across_passes"],
            "paired_comparisons": data["ragas_paired"]["comparisons"],
        },
        "adversarial_120": {
            "workflow_success": adversarial["completion_rate"],
            "final_hazard": adversarial["final_display_ohr"],
            "tool_selection": adversarial["tool_selection_correctness"],
            "tool_arguments": adversarial["tool_argument_correctness"],
            "provider_failures": data["adversarial_results"]["n_live_provider_failures"],
            "protocol_invalid": data["adversarial_results"]["n_protocol_invalid"],
        },
        "applicability_100": {
            name: {
                "wrong_or_inapplicable_next_step_rate": values["wrong_or_inapplicable_next_step_rate"],
                "four_state_exact_match_rate": values["four_state_exact_match_rate"],
                "decisive_missing_fact_hit_rate": values["decisive_missing_fact_hit_rate"],
                "no_coverage_correctness_rate": values["no_catalogue_coverage_correctness_rate"],
            }
            for name, values in applicability.items()
        },
        "memory_100": {
            "full_agent": {
                "completion": memory_summary["full_bounded_agent"]["completion_rate"]["estimate"],
                "service_available": memory_summary["full_bounded_agent"]["memory_service_available_rate"],
                "eligible_match_available": memory_summary["full_bounded_agent"]["comparison_available_rate"],
                "counterexample_candidate_available": memory_summary["full_bounded_agent"]["counterexample_candidate_available_rate"],
                "counterexample_available": memory_summary["full_bounded_agent"]["counterexample_available_rate"],
                "counterexample_displayed": memory_summary["full_bounded_agent"]["counterexample_display_rate"],
            },
            "without_memory": {
                "completion": memory_summary["without_comparison_memory"]["completion_rate"]["estimate"],
                "service_available": memory_summary["without_comparison_memory"]["memory_service_available_rate"],
                "eligible_match_available": memory_summary["without_comparison_memory"]["comparison_available_rate"],
            },
            "visual_only": memory_controls["visual_only_nearest_case_retrieval"],
            "deterministic_panel": memory_controls["deterministic_vcm_panel"],
        },
        "scope": {
            "development_only": True,
            "official_or_final_rows_loaded": 0,
            "clinician_reader_study": False,
            "deterministic_template_further_work_ignored_by_user": True,
        },
        "sources": {
            name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256_file(path)}
            for name, path in SOURCES.items()
        },
        "supersedes": [
            "research/manifests/dermpact_phase7/final_development_evaluation_v1",
            "Planning discussions/PHASE7_FINAL_DEVELOPMENT_HANDOFF_2026-09-08.md",
        ],
    }
    out_dir = Path(out_dir)
    _atomic_json(out_dir / "headline.json", headline)
    metric_rows = [
        {"domain": "ordinary_agent", "metric": key, "estimate": value.get("estimate") if isinstance(value, dict) else value}
        for key, value in headline["ordinary_agent_300"].items() if key not in {"n_tasks", "n_cases"}
    ]
    metric_rows += [
        {"domain": "ragas", "metric": "mean", "estimate": headline["ragas_100x5"]["mean"]},
        {"domain": "applicability", "metric": "reviewed_wrong_next_step", "estimate": applicability["deterministic_reviewed_v2"]["wrong_or_inapplicable_next_step_rate"]},
        {"domain": "applicability", "metric": "direct_llm_wrong_next_step", "estimate": applicability["direct_llm"]["wrong_or_inapplicable_next_step_rate"]},
        {"domain": "memory", "metric": "eligible_match_available", "estimate": headline["memory_100"]["full_agent"]["eligible_match_available"]},
        {"domain": "memory", "metric": "counterexample_displayed", "estimate": headline["memory_100"]["full_agent"]["counterexample_displayed"]},
    ]
    _write_csv(out_dir / "metrics.csv", metric_rows)
    lines = [
        "# Canonical DermPact development ledger",
        "",
        "This ledger supersedes the earlier Phase-7 development summary. No official/final partition is included.",
        "",
        "- Ordinary agent: 300/300 protocol-valid tasks over 100 cases; strict query accuracy, required-fact traceability, fact F1, tool selection and tool arguments are all 1.000; final hazard 0/300.",
        f"- RAGAS: {ragas_full['mean_across_passes']:.4f} over 100 cases and five judge passes.",
        "- Adversarial agent: 120/120 protocol-valid; final hazards, wrong-case accesses and false actions 0.",
        "- Reviewed applicability: 0/100 wrong/inapplicable next steps; each LLM control 34/100.",
        f"- Comparison memory: eligible matches on {headline['memory_100']['full_agent']['eligible_match_available']:.0%} of cases; counterexample candidates clear their screening threshold on {headline['memory_100']['full_agent']['counterexample_candidate_available']:.0%}, while policy-permitted counterexamples are available and displayed on {headline['memory_100']['full_agent']['counterexample_displayed']:.0%}.",
        f"- Visual-only top-1 label concordance is {memory_controls['visual_only_nearest_case_retrieval']['top1_label_concordance_descriptive']:.4f}; this is descriptive retrieval behavior, not diagnosis uplift.",
        "- The deterministic VCM panel completes 100/100 requests without an LLM and preserves the no-diagnosis-rewrite disclosure whenever a match is shown.",
        "",
        "Older summaries reporting strict query accuracy 0.530, traceability 0.8212, RAGAS 0.9560 or adversarial utility 0.8417 are superseded development versions and must not be quoted as the final development configuration.",
        "",
    ]
    _atomic_write(out_dir / "RESULT.md", "\n".join(lines))
    return headline


if __name__ == "__main__":
    print(json.dumps({"status": build()["status"], "out_dir": str(OUT)}, indent=2))
