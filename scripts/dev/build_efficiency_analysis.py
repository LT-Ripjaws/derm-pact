"""E8: Cost and efficiency analysis for the paper.

Assembles known system statistics into a single artifact. Every agentic RAG
paper reports efficiency; this script computes our numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "research" / "evidence"


def main():
    agent_report = EVIDENCE / "derm7pt_v1" / "agent_eval_report_v5_4.json"

    # Router savings are DERIVED from the promoted contract distribution, not hardcoded: the
    # promoted system is the one the paper describes, and its router numbers live in
    # promoted_contracts_routing.json (recomputed + cross-checked from
    # cdss_contracts_valid_promoted.json). Build that artifact first.
    routing_path = EVIDENCE / "derm7pt_v1" / "promoted_contracts_routing.json"
    if not routing_path.exists():
        raise SystemExit(
            f"missing {routing_path}; run scripts/dev/build_promoted_contracts_routing.py first"
        )
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    router = routing["router"]
    n_valid = routing["n_cases"]
    states_pct = ", ".join(
        f"{s} {router['state_distribution'][s]['rate'] * 100:.1f}%"
        for s in ("NO_COVERAGE", "CONFLICT", "ROUTINE", "UNDECIDABLE")
    )

    efficiency = {
        "router_savings": {
            "system": "PROMOTED perception (run_cohort(promoted=True), valid n=203)",
            "retrieval_rate": router["case_level_retrieval_rate"],
            "retrieval_avoided": router["case_level_retrieval_avoided"],
            "call_level_hypothetical_savings": router["call_level_hypothetical_savings"],
            "calls_a_gated_deployment_would_make": router["calls_a_gated_deployment_would_make"],
            "note": f"Router classification on the promoted valid contracts (n={n_valid}): "
                    f"{router['cases_needing_retrieval']} of {n_valid} cases would need "
                    "retrieval in a gated deployment. In this pipeline binding is a free "
                    "local catalog lookup and ran for every case, so the saving is "
                    "hypothetical: what gating would avoid where retrieval is remote or "
                    f"paid. Router states: {states_pct}. CONFLICT cases can request guidance "
                    "for both candidate conditions, so gated CALLS "
                    f"({router['calls_a_gated_deployment_would_make']}) exceed retrieving cases; "
                    "the call-level saving is reported alongside the case-level one.",
            "base_arm_reference": {
                "retrieval_rate": 0.30,
                "retrieval_avoided": 0.70,
                "note": "PRE-promotion pipeline on valid (n=203): router states NO_COVERAGE "
                        "55.7%, UNDECIDABLE 20.7%, ROUTINE 14.3%, CONFLICT 9.4%. Kept for "
                        "comparison only; the promoted checklist disagrees with the classifier "
                        "more often, so the promoted router retrieves more.",
            },
        },
        "criterion_abstention": {
            "system": "PROMOTED perception: argmax on the balanced 526-d heads, NO abstention gating",
            "abstention_rate": 0.0,
            "undecidable_rate": 0.0,
            "by_construction": True,
            "note": "The promoted path reads every criterion by argmax (the "
                    "criterion_abstention_v1 cutoffs were fit on the pre-promotion 512-d head "
                    "and are not valid for the promoted heads), so abstention and undecidable "
                    "rates are 0.0 BY CONSTRUCTION, not by measurement. Per-criterion "
                    "top-minus-second margins are plumbed into every promoted contract so a "
                    "future abstention policy can be re-fit.",
            "base_arm_reference": {
                "abstention_rate": 0.158,
                "cases_with_unknown": "149/203",
                "undecidable_from_unknowns": "42/149",
                "decidable_despite_unknowns": "107/149",
                "note": "PRE-promotion margin-based policy (criterion_abstention_v1): "
                        "score-range design kept 107 cases decidable even with unknowns.",
            },
        },
        "pipeline_components": {
            "feature_extraction": {
                "model": "DermLIP (ViT-B-16, frozen, 512-d)",
                "compute": "Local GPU, <1s per image",
                "cost": 0.0,
            },
            "diagnosis_classifier": {
                "model": "LogisticRegression (5-class)",
                "compute": "CPU, <10ms per case",
                "cost": 0.0,
            },
            "criterion_heads": {
                "model": "7x LogisticRegression (per criterion)",
                "compute": "CPU, <10ms per case (all 7)",
                "cost": 0.0,
            },
            "host_aggregation": {
                "model": "Argenziano 7-point rule (deterministic)",
                "compute": "CPU, <1ms per case",
                "cost": 0.0,
            },
            "guideline_binding": {
                "model": "Hash-verified lookup (no embedding search)",
                "compute": "CPU, <1ms per case",
                "cost": 0.0,
                "note": "In a gated deployment, triggered only for the cases the router "
                        "classifies as needing retrieval (promoted: ~51% of valid cases; "
                        "base arm: ~30%)",
            },
            "contract_verification": {
                "model": "Deterministic self-checks",
                "compute": "CPU, <1ms per case",
                "cost": 0.0,
            },
            "clinician_agent": {
                "model": "qwen/qwen3-vl-32b-instruct (text-only)",
                "compute": "API call, ~3s per question",
                "cost_per_question": "fractions of a cent (token-based; see total_pipeline_cost note)",
                "note": "Only invoked when clinician asks a question, not per case",
            },
        },
    }

    calls_per_question = None
    if agent_report.exists():
        report = json.loads(agent_report.read_text(encoding="utf-8"))
        usage = report.get("usage", {})
        n_cases = report.get("n_cases", 0)
        n_questions = report.get("n_questions", 0)
        if n_questions:
            calls_per_question = round(
                (usage.get("calls", 0) + usage.get("cache_hits", 0)) / n_questions, 2
            )

        efficiency["agent_evaluation_cost"] = {
            "n_cases": n_cases,
            "n_questions": n_questions,
            "paid_calls": usage.get("calls", 0),
            "cache_hits": usage.get("cache_hits", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": usage.get("latency_s", 0),
            "calls_per_case": round(usage.get("calls", 0) / n_cases, 2) if n_cases else None,
            "tokens_per_case": (round((usage.get("prompt_tokens", 0) +
                                       usage.get("completion_tokens", 0)) / n_cases)
                                if n_cases else None),
            "estimated_cost_usd": round(
                usage.get("prompt_tokens", 0) * 0.10 / 1e6 +
                usage.get("completion_tokens", 0) * 0.40 / 1e6, 4),
        }

    efficiency["dermagent_comparison"] = {
        "dermagent_modules_per_case": 7,
        "dermagent_tools": ["PanDerm Classifier", "MAKE Annotator", "DermoGPT",
                            "Qwen3-VL", "Case RAG (413k)", "Guideline RAG (3199 chunks)",
                            "Ontology Tool"],
        "dermagent_retry_mechanism": "Critic with confidence thresholds (90%/80%), up to k_max retries",
        "our_modules_per_case": "2 (classifier + criterion heads) + conditional retrieval",
        "our_llm_calls_per_case": 0,
        # Derived from the agent eval report (paid + cached messages per question, so the
        # figure includes tool turns and verification-loop revisions), not hardcoded.
        "our_llm_calls_per_question": calls_per_question,
        "note": "Our pipeline makes 0 LLM calls per case for the decision. "
                "The LLM is invoked only when a clinician asks a question. "
                "DermAgent invokes the LLM controller for EVERY case.",
    }

    # Call counts are the primary figures: the eval runner records no token counts, and the
    # old ~$0.01/call heuristic overstates measured token-based pricing ~40x ($0.10/1M prompt
    # + $0.40/1M completion; a 270-call verifier study cost $0.068 total).
    efficiency["total_pipeline_cost"] = {
        "perception": "$0 (all local)",
        "contracts": "$0 (deterministic)",
        "note": "<$2 total token-based estimate for all paid agent+judge calls. "
                "No frontier model spend. "
                "The only prior frontier spend was G1c's one-off model sweep ($5.79).",
    }

    out_path = EVIDENCE / "derm7pt_v1" / "efficiency_analysis.json"
    out_path.write_text(json.dumps(efficiency, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    print(f"\nRouter savings: {efficiency['router_savings']['retrieval_avoided']*100:.0f}%")
    print(f"LLM calls per case (decision): {efficiency['dermagent_comparison']['our_llm_calls_per_case']}")
    print(f"DermAgent modules per case: {efficiency['dermagent_comparison']['dermagent_modules_per_case']}")
    if "agent_evaluation_cost" in efficiency:
        ac = efficiency["agent_evaluation_cost"]
        print(f"Agent eval: {ac['paid_calls']} calls, ~${ac.get('estimated_cost_usd', '?')}")


if __name__ == "__main__":
    main()
