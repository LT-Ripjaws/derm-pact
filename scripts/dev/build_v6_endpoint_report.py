"""Citation-resolvability table + a spec-v6 family FEASIBILITY check.

Two outputs, and the distinction between them matters:

* the citation-resolvability table is a **result** -- descriptive, architectural,
  reported as counts;
* the v6 family numbers are a **feasibility check for a registration that has not
  happened yet**. They are computed on development data, which is what development data
  is for, and they are NOT the development verdict. The development verdict is the v5
  one: the registered condition is not met against B1.

Offline; zero paid calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medguard.eval.citation_resolvability import build_report, summarize_arm  # noqa: E402
from medguard.eval.paired_endpoints import (  # noqa: E402
    GUIDELINE_ELIGIBLE_LABELS,
    V6_ALIGNMENT_FAMILY,
    V6_EVIDENCE_USE_FAMILY,
    V6_INFERENTIAL_FAMILY,
    evidence_family_result,
    load_case_series,
    paired_endpoint_difference,
    predicted_diagnosis,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comparator", default="B1")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--truth", choices=["development", "sealed"], default="development",
                    help=("which label set to join. 'sealed' reads the confirmatory "
                          "labels THROUGH the access guard under sealed_scoring_resume, "
                          "so offline analysis of sealed results is on the record like "
                          "every other sealed read."))
    args = ap.parse_args()

    from medguard.eval.h2h_production import load_development_truth

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    scored = json.loads(Path(args.scored).read_text(encoding="utf-8"))
    if args.truth == "sealed":
        from medguard.eval.h2h_sealed import load_sealed_truth
        truth = load_sealed_truth(
            detail="offline re-scoring of the completed confirmatory run",
            purpose="sealed_scoring_resume")
    else:
        truth = load_development_truth()
    cases = raw["cases"]
    n_cases = len(cases)

    # Guideline eligibility is decided by the frozen probe's committed condition and the
    # licensed corpus -- both fixed before any answer exists.
    eligible = sum(
        1 for c in cases
        if predicted_diagnosis(c, "M") in GUIDELINE_ELIGIBLE_LABELS
    )

    per_arm = {}
    for arm, block in scored["per_arm"].items():
        rows = (block.get("ragas_supporting") or {}).get("sample_rows") or []
        prov = block.get("provenance_citation_integrity") or {}
        per_arm[arm] = summarize_arm(
            [str(r.get("response", "")) for r in rows],
            resolvable_citations=int(prov.get("total_citations") or 0),
            cases_with_resolvable_citations=int(prov.get("cases_with_citations") or 0),
            intact_citations=int(prov.get("intact_citations") or 0),
            n_cases=n_cases, guideline_eligible_cases=eligible,
        )
    citation_report = build_report(
        per_arm, n_cases=n_cases, guideline_eligible_cases=eligible)

    # --- v6 family feasibility (NOT a verdict) -------------------------------------
    series_m = load_case_series(raw, scored, "M", truth)
    series_c = load_case_series(raw, scored, args.comparator, truth)

    def run_family(members):
        """Holm runs WITHIN a family, so each is corrected separately."""
        comparisons = []
        for endpoint in members:
            if endpoint not in series_m or endpoint not in series_c:
                continue
            result = paired_endpoint_difference(
                series_m[endpoint], series_c[endpoint], n_boot=args.nboot)
            if result is not None:
                comparisons.append(result)
        return evidence_family_result(comparisons)

    families = {
        "evidence_use": run_family(V6_EVIDENCE_USE_FAMILY),
        "alignment": run_family(V6_ALIGNMENT_FAMILY),
    }
    all_comparisons = [c for f in families.values() for c in f["comparisons"]]
    passing = [
        c["metric"] for c in all_comparisons
        if c.get("significant_after_holm") and c.get("improvement_for_arm_a", 0.0) > 0
    ]
    regressions = [
        c["metric"] for c in all_comparisons
        if c.get("significant_after_holm") and c.get("improvement_for_arm_a", 0.0) < 0
    ]
    feasibility = {"comparisons": all_comparisons,
                   "holm_m_by_family": {k: v["holm_m"] for k, v in families.items()}}

    payload = {
        "citation_resolvability": citation_report,
        "v6_family_feasibility": {
            "status": "FEASIBILITY_CHECK_NOT_A_VERDICT",
            "warning": (
                "These numbers describe what a spec-v6 family WOULD measure. v6 is not yet "
                "registered, and an endpoint family may never be chosen after seeing the "
                "numbers it produces on the same data it will be judged on. The binding "
                "development verdict remains the v5 one (condition not met vs B1). v6 "
                "applies to the SEALED run only, and must be registered before it."
            ),
            "comparator": args.comparator,
            "family_rule": (
                "defined for every arm AND outcome not architecturally determined; "
                "applied before looking at who wins, which is why endpoints MedGuard "
                "loses are members"
            ),
            "evidence_use_family": list(V6_EVIDENCE_USE_FAMILY),
            "alignment_family": list(V6_ALIGNMENT_FAMILY),
            "endpoints_that_would_pass": passing,
            "n_would_pass": len(passing),
            "endpoints_that_would_regress": regressions,
            "n_would_regress": len(regressions),
            "comparisons": all_comparisons,
            "holm_m_by_family": feasibility["holm_m_by_family"],
        },
        "guideline_eligible_cases": eligible,
        "n_cases": n_cases,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"guideline-eligible lesions (committed condition in corpus): {eligible}/{n_cases}")
    print(f"\n{'arm':>4} {'answers':>8} {'marked':>7} {'markers':>8} {'resolvable':>11} "
          f"{'resolvability':>13} {'cov uncond':>11} {'cov eligible':>13}")
    for arm, s in per_arm.items():
        print(f"{arm:>4} {s['answers_scored']:>8} {s['answers_with_citation_markers']:>7} "
              f"{s['total_citation_markers']:>8} {s['resolvable_citations']:>11} "
              f"{str(s['citation_resolvability']):>13} "
              f"{str(s['verifiable_citation_coverage_unconditional']):>11} "
              f"{str(s['verifiable_citation_coverage_on_eligible']):>13}")
    print(f"\n[v6 feasibility vs {args.comparator}] {len(passing)} improve, "
          f"{len(regressions)} regress, of {len(V6_INFERENTIAL_FAMILY)} members")
    for c in all_comparisons:
        print(f"  {c['metric']:>38} {c['estimate']:+.4f} "
              f"[{c['low']:+.4f}, {c['high']:+.4f}] holm={c.get('p_value_holm'):.4f} "
              f"-> {c['favours']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
