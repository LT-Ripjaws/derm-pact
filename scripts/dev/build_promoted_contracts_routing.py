"""Contract + routing distribution for the PROMOTED system, from the promoted VALID contracts.

The contracts/routing block the paper used to cite (conflict 0.1544 on test / 0.1724 on valid,
undecidable 0.0, citation 0.2785, router NO_COVERAGE 55.7% / retrieval 0.30-0.70, criterion
abstention 0.158) describes the PRE-promotion pipeline. The promoted system's cascade base is
``cdss_contracts_valid_promoted.json`` (run_cohort(promoted=True) on valid, n=203), so every
distributional claim is recomputed HERE from that artifact's own contracts:

* conflict / escalation / undecidable / withheld rates -- from the contract fields, with the
  7-point score independently re-aggregated from each contract's criterion values as a check;
* router state distribution + gated retrieval calls -- by re-running ``classify_state`` /
  ``route`` over every promoted contract (local catalog lookup, no network, no paid calls);
* citation + contingent-guidance rates and the coverage denominator;
* abstention: the promoted path reads criteria by ARGMAX with NO abstention gating (the
  criterion_abstention_v1 cutoffs were fit on the pre-promotion 512-d head and are not valid for
  the promoted 526-d balanced heads), so undecidable and criterion-abstention rates are 0.0 BY
  CONSTRUCTION for the promoted path. That is stated plainly rather than reusing the base
  numbers; the base-arm figures are kept as an explicitly labeled reference.

Every recomputed rate is cross-checked against the summary blocks run_cohort itself wrote into
the artifact; the script refuses to emit anything on a mismatch.

Output: research/evidence/derm7pt_v1/promoted_contracts_routing.json

Run::
    venv_win/Scripts/python.exe scripts/dev/build_promoted_contracts_routing.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import score_criteria  # noqa: E402
from medguard_v2.criteria.schema import CRITERION_NAMES  # noqa: E402
from medguard_v2.evidence.guideline_binding import GuidelineCatalog  # noqa: E402
from medguard_v2.evidence.router import LOW_CONFIDENCE, classify_state, route  # noqa: E402

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
CONTRACTS = EVIDENCE / "cdss_contracts_valid_promoted.json"
OUT = EVIDENCE / "promoted_contracts_routing.json"


def _round4(x):
    return round(float(x), 4)


def main() -> None:
    t0 = time.perf_counter()
    artifact = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    if artifact["protocol"] != "medguard-v2-derm7pt-criteria-ensemble-v2-promoted":
        raise SystemExit(f"unexpected protocol in {CONTRACTS.name}: {artifact['protocol']}")
    contracts = artifact["contracts"]
    n = len(contracts)
    if n != artifact["n_cases"]:
        raise SystemExit(f"contract count {n} != n_cases {artifact['n_cases']}")

    catalog = GuidelineCatalog.load()

    conflicts = escalations = undecidable = cited = contingent = withheld = 0
    determinate_all = 0
    state_counts: Counter = Counter()
    low_conf_conflicts = 0
    gated_calls = 0
    score_mismatches = []

    for c in contracts:
        values = {entry["name"]: entry["value"] for entry in c["criteria"]}
        if set(values) != set(CRITERION_NAMES):
            raise SystemExit(f"{c['case_id']}: criteria table incomplete")
        scored = score_criteria(values)

        # Independent re-aggregation must match the stored seven_point_score exactly.
        stored = c["seven_point_score"]
        if (scored.score_min != stored["min"] or scored.score_max != stored["max"]
                or scored.decidable != stored["decidable"]):
            score_mismatches.append(c["case_id"])
            continue

        if scored.determinate:
            determinate_all += 1
        if not scored.decidable:
            undecidable += 1
        if c["conflict"]["present"]:
            conflicts += 1
        if c["conflict"]["escalate"]:
            escalations += 1
        if c["guideline"]["covered"]:
            cited += 1
        if c.get("contingent_guideline") is not None:
            contingent += 1
        if not c["safe_to_display"]:
            withheld += 1

        diagnosis = c["classifier"]["diagnosis"]
        confidence = float(c["classifier"]["confidence"])
        state, _reason = classify_state(
            classifier_diagnosis=diagnosis, classifier_confidence=confidence, score=scored,
        )
        state_counts[state.value] += 1
        if state.value == "CONFLICT" and not c["conflict"]["present"]:
            low_conf_conflicts += 1  # low-confidence differential, not a checklist conflict
        decision = route(
            classifier_diagnosis=diagnosis, classifier_confidence=confidence,
            score=scored, catalog=catalog,
        )
        gated_calls += decision.retrieval_calls

    if score_mismatches:
        raise SystemExit(
            f"re-aggregated 7-point score disagrees with stored contracts on {score_mismatches}"
        )

    # ---- cross-checks against the artifact's own summary blocks -----------------------------
    db = artifact["decision_behaviour"]
    rt = artifact["router"]
    gc = artifact["guideline_citation"]
    checks = {
        "conflict_rate": (_round4(conflicts / n), db["conflict_rate"]),
        "escalation_rate": (_round4(escalations / n), db["escalation_rate"]),
        "undecidable_rate": (_round4(undecidable / n), db["undecidable_rate"]),
        "safe_to_display": (n - withheld, db["safe_to_display"]),
        "state_distribution": (dict(state_counts), rt["state_distribution"]),
        "gated_calls": (gated_calls, rt["calls_a_gated_deployment_would_make"]),
        "citation_rate": (_round4(cited / n), gc["citation_rate"]),
        "contingent_cases": (contingent, gc["cases_with_contingent_melanoma_guidance"]),
    }
    failed = {k: v for k, v in checks.items() if v[0] != v[1]}
    if failed:
        raise SystemExit(
            "recomputed distribution disagrees with the artifact's own summary blocks: "
            + json.dumps(failed, indent=2)
        )

    retrieving_states = ("CONFLICT", "UNDECIDABLE", "ROUTINE")
    cases_needing_retrieval = sum(state_counts.get(s, 0) for s in retrieving_states)

    report = {
        "protocol": "derm7pt-promoted-contracts-routing-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(CONTRACTS.relative_to(REPO)).replace("\\", "/"),
        "split": "valid",
        "n_cases": n,
        "perception": artifact["perception_promotion"]["diagnosis"],
        "recomputation_note": (
            "Every rate below is recomputed from the individual promoted contracts (criterion "
            "values re-aggregated, router state re-derived via classify_state/route against the "
            "hash-pinned catalog) and cross-checked against the summary blocks run_cohort wrote "
            "into the artifact. Local lookups only; zero paid calls."
        ),
        "decision_behaviour": {
            "safe_to_display": n - withheld,
            "safe_to_display_rate": _round4((n - withheld) / n),
            "withheld": withheld,
            "conflict_cases": conflicts,
            "conflict_rate": _round4(conflicts / n),
            "escalation_cases": escalations,
            "escalation_rate": _round4(escalations / n),
            "undecidable_cases": undecidable,
            "undecidable_rate": _round4(undecidable / n),
        },
        "abstention": {
            "criterion_abstention_rate": 0.0,
            "undecidable_rate": 0.0,
            "by_construction": True,
            "note": (
                "The promoted path reads the balanced 526-d criterion heads by ARGMAX with no "
                "abstention gating: every criterion is determinate, so score_min == score_max and "
                "every case is decidable -- undecidable and abstention rates are 0.0 BY "
                "CONSTRUCTION, not by measurement. The criterion_abstention_v1 cutoffs were fit "
                "on the pre-promotion 512-d dermlip head and would mis-abstain on these heads; "
                "per-criterion top-minus-second margins are plumbed into each contract "
                "(criterion_margins) so a future abstention policy can be re-fit for the "
                "promoted heads."
            ),
            "determinate_cases": determinate_all,
        },
        "router": {
            "state_distribution": {
                s: {"cases": state_counts.get(s, 0), "rate": _round4(state_counts.get(s, 0) / n)}
                for s in ("NO_COVERAGE", "CONFLICT", "UNDECIDABLE", "ROUTINE")
            },
            "conflict_state_vs_contract_conflict_note": (
                f"Router CONFLICT ({state_counts.get('CONFLICT', 0)}) exceeds contract-level "
                f"conflicts ({conflicts}) by {low_conf_conflicts}: classifier confidence below "
                f"{LOW_CONFIDENCE:.2f} is routed as an unresolved differential even when the "
                "checklist agrees."
            ),
            "cases_needing_retrieval": cases_needing_retrieval,
            "case_level_retrieval_rate": _round4(cases_needing_retrieval / n),
            "case_level_retrieval_avoided": _round4(1.0 - cases_needing_retrieval / n),
            "calls_a_gated_deployment_would_make": gated_calls,
            "calls_this_pipeline_made": n,
            "call_level_hypothetical_savings": _round4(1.0 - gated_calls / n),
            "call_vs_case_note": (
                "CONFLICT cases can request guidance for both candidate conditions, so gated "
                "CALLS can exceed retrieving CASES; both denominators are reported. The saving "
                "is hypothetical: binding is a free local lookup that ran for every case here."
            ),
        },
        "guideline_citation": {
            "cases_with_a_citable_recommendation": cited,
            "citation_rate": _round4(cited / n),
            "cases_with_contingent_melanoma_guidance": contingent,
            "contingent_rate": _round4(contingent / n),
            "coverage_denominator": artifact["guideline_citation"]["coverage_denominator"],
            "note": artifact["guideline_citation"]["note"],
        },
        "base_arm_reference": {
            "label": "PRE-PROMOTION pipeline (dermlip 512-d readers + criterion_abstention_v1)",
            "valid_n203": {
                "conflict_rate": 0.1724,
                "escalation_rate": 0.1724,
                "undecidable_rate_contracts_artifact": 0.0,
                "source": "cdss_contracts_valid.json (frozen, protocol v1)",
            },
            "test_n395": {
                "conflict_rate": 0.1544,
                "escalation_rate": 0.1544,
                "undecidable_rate": 0.0,
                "citation_rate": 0.2785,
                "source": "ablation_table.json contract_layer_context (confirmatory run)",
            },
            "router_and_abstention_valid_n203": {
                "retrieval_rate": 0.30,
                "retrieval_avoided": 0.70,
                "router_states": "NO_COVERAGE 55.7%, UNDECIDABLE 20.7%, ROUTINE 14.3%, CONFLICT 9.4%",
                "criterion_abstention_rate": 0.158,
                "source": "efficiency_analysis.json (pre-promotion seed values)",
            },
            "note": (
                "Base-arm numbers are kept for comparison only; they describe the pre-promotion "
                "readers and the margin-based abstention policy that the promoted path does not "
                "use. The split differs across base references (valid n=203 vs test n=395) -- "
                "the promoted block above is valid n=203."
            ),
        },
        "cross_checks": {
            k: {"recomputed": v[0], "artifact": v[1], "match": True} for k, v in checks.items()
        },
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  n={n} promoted valid contracts")
    print(f"  conflict rate     : {report['decision_behaviour']['conflict_rate']} "
          f"(base valid 0.1724, base test 0.1544)")
    print(f"  escalation rate   : {report['decision_behaviour']['escalation_rate']}")
    print(f"  undecidable       : {report['decision_behaviour']['undecidable_rate']} (by construction)")
    print(f"  citation rate     : {report['guideline_citation']['citation_rate']} (base test 0.2785)")
    print("  router states     : " + ", ".join(
        f"{s} {report['router']['state_distribution'][s]['cases']} "
        f"({report['router']['state_distribution'][s]['rate']:.1%})"
        for s in ("NO_COVERAGE", "CONFLICT", "UNDECIDABLE", "ROUTINE")))
    print(f"  case-level retrieval rate/avoided: "
          f"{report['router']['case_level_retrieval_rate']} / "
          f"{report['router']['case_level_retrieval_avoided']} (base 0.30 / 0.70)")
    print(f"  call-level savings: {report['router']['call_level_hypothetical_savings']}")
    print(f"  all cross-checks against the artifact's own blocks: PASS")


if __name__ == "__main__":
    main()
