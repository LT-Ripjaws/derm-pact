"""Compute the registered evidence-family paired CIs from a completed H2H run.

Offline and free: it re-joins per-sample scores the run already stored, so it issues
zero generation and zero judge calls.

Usage:
  python build_paired_evidence_ci.py \
      --raw    research/evidence/dev_h2h_full/dev_h2h_full_160x11_v4.json \
      --scored research/evidence/dev_h2h_full/dev_h2h_full_160x11_v4.scored.json \
      --out    research/evidence/dev_h2h_full/paired_evidence_v4.json \
      --md     notes/PAIRED_EVIDENCE_V4.md
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

from medguard.eval.paired_endpoints import (  # noqa: E402
    REGISTERED_EVIDENCE_FAMILY,
    SUPPORTING_ENDPOINTS,
    architectural_endpoint,
    evidence_family_result,
    load_case_series,
    paired_endpoint_difference,
    sealed_scale_projection,
)

#: M against each no-contract comparator.  B1 is the registered practical
#: no-contract comparator; B0 is the perception/generation floor.
DEFAULT_COMPARISONS = (("M", "B1"), ("M", "B5"), ("M", "B6"), ("M", "B4"), ("M", "B0"))

PAIRABLE = ("ragas_faithfulness", "ragas_context_recall") + SUPPORTING_ENDPOINTS
ARCHITECTURAL = ("citation_integrity", "contract_completion")


def _architectural_values(scored, arm):
    block = (scored.get("per_arm") or {}).get(arm) or {}
    citation = (block.get("provenance_citation_integrity") or {}).get("integrity_rate")
    process = (block.get("process_contract_completion") or {})
    completion = process.get("contract_complete_rate") if process.get("applicable") else None
    return {"citation_integrity": citation, "contract_completion": completion}


def _fmt(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _markdown(payload) -> str:
    lines = [
        "# Paired evidence-family differences (lesion-clustered bootstrap)",
        "",
        f"Source run: `{payload['source_raw']}`",
        f"Replicates {payload['bootstrap']['replicates']} · seed "
        f"{payload['bootstrap']['seed']} · clustered by lesion · two-sided 95%.",
        "",
        "> Registered acceptance condition: paired improvement with a 95% "
        "clustered-bootstrap interval excluding zero on **at least two** of "
        "faithfulness, context recall, citation precision, or contract completion.",
        "",
    ]
    for pair in payload["pairs"]:
        result = pair["evidence_family"]
        verdict = "MET" if result["condition_met"] else "NOT MET"
        lines += [
            f"## {pair['arm_a']} vs {pair['arm_b']}",
            "",
            f"Registered condition: **{verdict}** "
            f"({result['n_endpoints_passing']} of the family pass; Holm m="
            f"{result['holm_m']}).",
            "",
            "| endpoint | n | A | B | diff (A−B) | 95% CI | p (Holm) | favours |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in result["comparisons"] + pair["supporting"]:
            if not c.get("testable", True):
                lines.append(
                    f"| {c['metric']} | — | {_fmt(c['value_a'])} | "
                    f"{_fmt(c['value_b'])} | architectural | not testable | — | "
                    f"{c['arm_a']} (by construction) |"
                )
                continue
            ci = f"[{c['low']:.4f}, {c['high']:.4f}]"
            holm = c.get("p_value_holm", c.get("p_value"))
            flag = "" if c["higher_is_better"] else " ↓"
            if c.get("post_hoc"):
                flag += " *post-hoc*"
            lines.append(
                f"| {c['metric']}{flag} | "
                f"{c['n_paired']} | {_fmt(c['mean_a_paired'])} | "
                f"{_fmt(c['mean_b_paired'])} | {c['estimate']:+.4f} | {ci} | "
                f"{holm:.4f} | {c['favours']} |"
            )
        lines += ["", "Unconditional means (all of each arm's own defined cases), so a "
                  "coverage-conditioned difference is visible:", ""]
        for c in result["comparisons"] + pair["supporting"]:
            if not c.get("testable", True):
                continue
            lines.append(
                f"- `{c['metric']}`: {c['arm_a']} {_fmt(c['mean_a_unconditional'])} "
                f"(n={c['n_defined_a']}) · {c['arm_b']} "
                f"{_fmt(c['mean_b_unconditional'])} (n={c['n_defined_b']})"
                + ("  ← paired subset is coverage-conditioned"
                   if c.get("coverage_conditioned") else "")
            )
        undecided = [
            p for p in pair["sealed_scale_projection"]
            if not p["already_excludes_zero"]
        ]
        if undecided:
            lines += ["", "**Sealed-scale projection** (a projection, not a result — it "
                      "assumes the effect size and defined-fraction hold):", ""]
            for p in undecided:
                verdict = ("would clear zero" if p["would_exclude_zero_at_sealed_scale"]
                           else "would still contain zero")
                lines.append(
                    f"- `{p['metric']}`: {p['estimate']:+.4f} ±{p['observed_half_width']:.4f} "
                    f"→ ±{p['projected_half_width']:.4f} at n≈{p['projected_n_paired']} "
                    f"paired — **{verdict}**"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--truth", choices=["development", "sealed"], default="development",
                    help=("which label set to join. 'sealed' reads the confirmatory "
                          "labels THROUGH the access guard under sealed_scoring_resume, "
                          "so offline analysis of sealed results is on the record like "
                          "every other sealed read."))
    ap.add_argument("--pairs", default=None,
                    help="comma-separated 'A:B' pairs; default = M vs each comparator")
    ap.add_argument("--dev-lesions", type=int, default=160)
    ap.add_argument("--sealed-lesions", type=int, default=1495)
    args = ap.parse_args()

    from medguard.eval.h2h_production import load_development_truth
    from medguard.eval.paired_endpoints import REGISTERED_BOOTSTRAP_SEED

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    scored = json.loads(Path(args.scored).read_text(encoding="utf-8"))
    if args.truth == "sealed":
        from medguard.eval.h2h_sealed import load_sealed_truth
        truth = load_sealed_truth(
            detail="offline re-scoring of the completed confirmatory run",
            purpose="sealed_scoring_resume")
    else:
        truth = load_development_truth()

    if args.pairs:
        pairs = tuple(tuple(p.split(":", 1)) for p in args.pairs.split(",") if ":" in p)
    else:
        available = set((scored.get("per_arm") or {}))
        pairs = tuple(p for p in DEFAULT_COMPARISONS if p[0] in available and p[1] in available)

    series_cache = {}

    def series_for(arm):
        if arm not in series_cache:
            series_cache[arm] = load_case_series(raw, scored, arm, truth)
        return series_cache[arm]

    payload = {
        "source_raw": args.raw,
        "source_scored": args.scored,
        "bootstrap": {"replicates": args.nboot, "seed": REGISTERED_BOOTSTRAP_SEED,
                      "clustered_by": "lesion_cluster_id", "confidence": 0.95},
        "registered_family": list(REGISTERED_EVIDENCE_FAMILY),
        "pairs": [],
    }

    for arm_a, arm_b in pairs:
        sa, sb = series_for(arm_a), series_for(arm_b)
        family, supporting = [], []
        for endpoint in PAIRABLE:
            if endpoint not in sa or endpoint not in sb:
                continue
            result = paired_endpoint_difference(
                sa[endpoint], sb[endpoint], n_boot=args.nboot)
            if result is None:
                continue
            (family if endpoint in REGISTERED_EVIDENCE_FAMILY else supporting).append(result)
        values_a, values_b = _architectural_values(scored, arm_a), _architectural_values(scored, arm_b)
        for endpoint in ARCHITECTURAL:
            family.append(architectural_endpoint(
                endpoint, arm_a, arm_b, values_a[endpoint], values_b[endpoint]))
        family_result = evidence_family_result(family)
        projections = [
            p for p in (
                sealed_scale_projection(
                    c, dev_lesions=args.dev_lesions, sealed_lesions=args.sealed_lesions)
                for c in family_result["comparisons"] + supporting
            ) if p is not None
        ]
        payload["pairs"].append({
            "arm_a": arm_a, "arm_b": arm_b,
            "evidence_family": family_result,
            "supporting": [dict(s) for s in supporting],
            "sealed_scale_projection": projections,
        })
        print(f"[{arm_a} vs {arm_b}] condition_met="
              f"{payload['pairs'][-1]['evidence_family']['condition_met']}")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.md:
        Path(args.md).write_text(_markdown(payload), encoding="utf-8")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
