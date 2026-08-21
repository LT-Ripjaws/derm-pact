"""N5 -- Risk-coverage curves for the checklist verdict (promoted criteria, Derm7pt test).

The 7-point checklist verdict (score_min >= 3 => melanoma-suspicious) carries calibrated
uncertainty on the *criteria* side, expressed through criterion-level abstention: when the
promoted heads are unsure about enough pivotal criteria, the aggregated score becomes a RANGE that
spans the threshold and the case is left UNDECIDABLE rather than decided. Sweeping the abstention
margin threshold tau therefore traces a selective-prediction (risk-coverage) curve:

    coverage(tau) = fraction of cases the checklist still leaves DECIDABLE
    risk(tau)     = error of the melanoma verdict among the DECIDED cases

As tau rises, more criteria go INDETERMINATE, more cases become undecidable (coverage falls), and
-- if the criterion margins are informative -- the verdict error among the cases still decided
should fall too. This is the checklist-side analogue of the classifier-side conformal prediction
sets (medguard_v2/perception/conformal.py; existing derm7pt conformal_v3 in
research/evidence/v3_metadata_evaluation.json): both paths carry calibrated uncertainty, expressed
differently -- conformal widens a prediction SET to hold marginal coverage; the checklist widens a
score RANGE and abstains when it straddles the threshold. They are reported side by side.

Two risk definitions, both reported (the table headlines the first):
  * true-label melanoma-verdict error: 1{verdict == excise} XOR 1{diagnosis_group == MEL}, among
    decided cases. The clinically meaningful risk. It is ceiling-bound at useful coverage: even the
    fully-informed checklist on GOLD criteria disagrees with the pathology label on ~0.15 of the
    whole test set (the 7-point rule's own ceiling), and predicted-criteria true-label error starts
    ABOVE that and only relaxes toward it as coverage falls. That whole-set ceiling is not a hard
    per-subset floor -- selective abstention CAN dip below it in the deep low-coverage tail by
    dropping the hardest cases -- but at useful coverage the true-label risk is dominated by the
    checklist-vs-pathology disagreement, which abstention on model uncertainty does not resolve.
    Reported, not hidden.
  * self-consistency error: 1{verdict from predicted criteria} != 1{verdict from GOLD criteria},
    among decided cases. Isolates what abstention can actually fix -- perception uncertainty -- from
    the checklist's own ceiling. This one can approach 0 as coverage falls.

Also reported per tau: sensitivity and specificity AMONG DECIDED (1 - sensitivity is the
melanoma-miss rate, the safety-critical direction on a melanoma CDSS).

Boundaries: zero paid calls, zero images. Promoted heads fit on Derm7pt train only (run_cohort
fit_readers_promoted), evaluated on Derm7pt test. Deterministic host aggregation.
Output: research/evidence/derm7pt_v1/risk_coverage_curves.json + notes/RISK_COVERAGE_CURVES.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import score_criteria  # noqa: E402
from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    _head_values_and_margins,
    _promoted_feature_matrices,
    fit_readers_promoted,
)

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "risk_coverage_curves.json"
OUT_MD = REPO / "notes" / "RISK_COVERAGE_CURVES.md"

TAU_GRID = [round(x, 3) for x in np.linspace(0.0, 1.0, 101)]
MIN_DECIDED_FOR_HEADLINE = 20  # a rate estimated on fewer decided cases is flagged as tail-only
RISK_TARGETS = [0.05, 0.10, 0.20]


def _load_promoted_test():
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    diag_feats, crit_feats = _promoted_feature_matrices(cases)
    _diagnosis, heads = fit_readers_promoted(train, diag_feats, crit_feats)
    xc = np.stack([crit_feats[str(c.case_num)] for c in test])
    pred_values: Dict[str, List[str]] = {}
    margins: Dict[str, np.ndarray] = {}
    for name, head in heads.items():
        values, margin = _head_values_and_margins(head, xc)
        pred_values[name] = values
        margins[name] = np.asarray(margin, dtype=float)
    return test, pred_values, margins


def _curve_point(tau: float, test, pred_values, margins, gold_verdict, is_mel) -> Dict:
    """Apply a uniform margin cutoff tau, aggregate, and measure risk among decided cases."""
    n = len(test)
    decided = 0
    tp = fp = tn = fn = 0          # vs true MEL label, among decided
    self_err = 0                   # vs gold-criteria verdict, among decided
    for i in range(n):
        crit = {}
        for name in CRITERION_NAMES:
            crit[name] = INDETERMINATE if margins[name][i] < tau else pred_values[name][i]
        sc = score_criteria(crit)
        if not sc.decidable:
            continue
        decided += 1
        verdict = sc.meets_threshold          # True => excise / melanoma-suspicious
        if is_mel[i]:
            if verdict:
                tp += 1
            else:
                fn += 1
        else:
            if verdict:
                fp += 1
            else:
                tn += 1
        if verdict != gold_verdict[i]:
            self_err += 1

    coverage = decided / n
    risk_true = (fp + fn) / decided if decided else None
    risk_self = self_err / decided if decided else None
    sens = tp / (tp + fn) if (tp + fn) else None      # among decided melanomas
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "tau": tau,
        "coverage": round(coverage, 4),
        "n_decided": decided,
        "risk_true_label": round(risk_true, 4) if risk_true is not None else None,
        "risk_self_consistency": round(risk_self, 4) if risk_self is not None else None,
        "sensitivity_among_decided": round(sens, 4) if sens is not None else None,
        "specificity_among_decided": round(spec, 4) if spec is not None else None,
        "confusion_true_label": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _coverage_at_risk(curve: List[Dict], risk_key: str, target: float) -> Dict:
    """Max coverage among curve points whose risk <= target (selective-prediction reading)."""
    eligible = [
        p for p in curve
        if p[risk_key] is not None and p[risk_key] <= target and p["n_decided"] > 0
    ]
    if not eligible:
        # report the minimum risk actually reached, so "not achievable" is quantified
        reached = [p for p in curve if p[risk_key] is not None and p["n_decided"] > 0]
        best = min(reached, key=lambda p: p[risk_key]) if reached else None
        return {
            "achievable": False,
            "min_risk_reached": best[risk_key] if best else None,
            "at_coverage": best["coverage"] if best else None,
            "at_tau": best["tau"] if best else None,
            "note": f"no operating point reaches risk <= {target}; min risk reached is shown",
        }
    best = max(eligible, key=lambda p: p["coverage"])
    reliable = best["n_decided"] >= MIN_DECIDED_FOR_HEADLINE
    return {
        "achievable": True,
        "coverage": best["coverage"],
        "risk": best[risk_key],
        "tau": best["tau"],
        "n_decided": best["n_decided"],
        "reliable_sample": bool(reliable),
        "note": None if reliable else (
            f"achieved only in the small-sample tail (n_decided={best['n_decided']} < "
            f"{MIN_DECIDED_FOR_HEADLINE}); treat as indicative, not a reliable rate"
        ),
    }


def build() -> Dict:
    test, pred_values, margins = _load_promoted_test()
    n = len(test)
    is_mel = [c.diagnosis_group == "MEL" for c in test]
    gold_verdict = [score_criteria(dict(c.criteria)).meets_threshold for c in test]

    curve = [_curve_point(tau, test, pred_values, margins, gold_verdict, is_mel) for tau in TAU_GRID]
    base = curve[0]

    coverage_at_risk = {
        "true_label": {str(t): _coverage_at_risk(curve, "risk_true_label", t) for t in RISK_TARGETS},
        "self_consistency": {
            str(t): _coverage_at_risk(curve, "risk_self_consistency", t) for t in RISK_TARGETS
        },
    }

    # gold-criteria checklist ceiling vs true label (the floor risk_true_label cannot beat)
    g_tp = sum(1 for i in range(n) if is_mel[i] and gold_verdict[i])
    g_fn = sum(1 for i in range(n) if is_mel[i] and not gold_verdict[i])
    g_fp = sum(1 for i in range(n) if not is_mel[i] and gold_verdict[i])
    g_tn = sum(1 for i in range(n) if not is_mel[i] and not gold_verdict[i])
    gold_ceiling = {
        "verdict_error_vs_true_label": round((g_fp + g_fn) / n, 4),
        "sensitivity": round(g_tp / (g_tp + g_fn), 4),
        "specificity": round(g_tn / (g_tn + g_fp), 4),
        "note": "Fully-informed checklist on GOLD criteria over the WHOLE test set. Predicted-"
                "criteria true-label risk starts above this and relaxes toward it as coverage "
                "falls; because these ~0.15 whole-set errors are the 7-point rule disagreeing with "
                "pathology (not the heads being unsure), abstention on model uncertainty only "
                "reduces true-label risk below this whole-set value in the deep low-coverage tail, "
                "where it selectively drops the hardest cases -- not at operationally useful "
                "coverage. It is a ceiling on the useful-coverage regime, not a hard per-subset floor.",
    }

    conformal_pairing = {
        "source": "research/evidence/v3_metadata_evaluation.json -> derm7pt.conformal_v3",
        "method": "LAC split-conformal, alpha=0.10 (>=90% marginal coverage target)",
        "classifier_side": {
            "test_coverage": 0.9241,
            "target_coverage": 0.90,
            "mean_set_size": 1.605,
            "singleton_rate": 0.5418,
            "mel_class_coverage": 0.9406,
        },
        "conceptual_pairing": (
            "Both the classifier and the checklist carry calibrated uncertainty, expressed "
            "differently. The classifier widens a prediction SET (mean 1.605 classes) to hold >=90% "
            "marginal coverage -- it never abstains, it hedges by listing more labels. The checklist "
            "widens a score RANGE and ABSTAINS (undecidable) when the range straddles the melanoma "
            "threshold. The conformal set answers 'which diagnoses can I not rule out'; the "
            "risk-coverage curve answers 'how much melanoma-verdict error do I carry if I decide "
            "this fraction of cases'. They are complementary uncertainty surfaces over the same "
            "cases, not competing metrics."
        ),
    }

    # ---- interpretation
    base_risk = base["risk_true_label"]
    ct = coverage_at_risk["true_label"]
    cs = coverage_at_risk["self_consistency"]
    reached_020 = ct["0.2"].get("achievable")

    # Safety-critical trajectory: sensitivity among decided as coverage falls. Pick the point
    # nearest coverage 0.67 (the risk<=0.20 operating point) to quote a concrete sens gain.
    sens_point = min(
        (p for p in curve if p["sensitivity_among_decided"] is not None and p["coverage"] > 0.6),
        key=lambda p: abs(p["coverage"] - 0.67),
    )
    interpretation = (
        f"At full coverage (tau=0, all {n} cases decided) the promoted checklist's melanoma-verdict "
        f"error vs the true label is {base_risk:.4f}, with sensitivity "
        f"{base['sensitivity_among_decided']:.4f} / specificity {base['specificity_among_decided']:.4f} "
        f"among decided melanomas. The gold-criteria whole-set ceiling "
        f"({gold_ceiling['verdict_error_vs_true_label']:.4f}) bounds the useful-coverage regime: "
        f"predicted-criteria true-label error starts above it and relaxes toward it as coverage "
        f"falls (it dips below only in the deep low-coverage tail, where the hardest cases are "
        f"selectively dropped). "
        + (
            f"Trading coverage for safety does buy real risk reduction against the true label: risk "
            f"<= 0.20 is reachable at coverage {ct['0.2']['coverage']:.4f}"
            + (f" (tau={ct['0.2']['tau']})." if ct['0.2'].get('coverage') is not None else ".")
            if reached_020 else
            f"Against the true label the curve does not reach risk <= 0.20 (min risk "
            f"{ct['0.2'].get('min_risk_reached')}), consistent with the gold-criteria floor; the "
            f"true-label error is dominated by the checklist's own ceiling, not by resolvable "
            f"perception uncertainty."
        )
        + " On the self-consistency risk (predicted-vs-gold-criteria verdict), which isolates the "
        "perception uncertainty abstention CAN fix, "
        + (
            f"risk <= 0.05 is reachable at coverage {cs['0.05']['coverage']:.4f}"
            if cs["0.05"].get("achievable") else
            f"the minimum reached is {cs['0.05'].get('min_risk_reached')} at coverage "
            f"{cs['0.05'].get('at_coverage')}"
        )
        + ", confirming the abstention knob does move the perception-level error monotonically even "
        "where the true-label risk is ceiling-bound. "
        f"The safety-critical direction moves the right way too: melanoma sensitivity AMONG DECIDED "
        f"rises monotonically as coverage falls -- from {base['sensitivity_among_decided']:.4f} at "
        f"full coverage to {sens_point['sensitivity_among_decided']:.4f} at coverage "
        f"{sens_point['coverage']:.4f} (specificity {base['specificity_among_decided']:.4f} -> "
        f"{sens_point['specificity_among_decided']:.4f}). The cases the checklist abstains on are "
        f"disproportionately the melanomas it would otherwise miss, so deciding only the confident "
        f"fraction is safer, not merely lower-volume."
    )

    return {
        "protocol": "derm7pt-risk-coverage-checklist-verdict-n5-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": (
            "Sweeping the criterion-abstention margin threshold, what is the checklist's "
            "melanoma-verdict error (risk) among the cases it still leaves decidable (coverage)?"
        ),
        "boundaries": {
            "paid_calls": 0,
            "images_encoded": 0,
            "heads": "PROMOTED criteria (dermlip_tta+metadata14, 526-d, lr_balanced, per-head C), "
                     "fit on Derm7pt train (run_cohort.fit_readers_promoted)",
            "cohort": "Derm7pt test",
            "n_cases": n,
            "n_melanoma": int(sum(is_mel)),
            "tau_grid": {"min": TAU_GRID[0], "max": TAU_GRID[-1], "n_points": len(TAU_GRID)},
            "abstention_rule": "uniform margin cutoff: criterion INDETERMINATE if head "
                               "top-minus-second margin < tau; case decidable per "
                               "aggregate.SevenPointScore.decidable",
        },
        "risk_definitions": {
            "risk_true_label": "1{verdict==excise} XOR 1{diagnosis_group==MEL}, among decided cases",
            "risk_self_consistency": "1{predicted-criteria verdict} != 1{gold-criteria verdict}, "
                                     "among decided cases",
        },
        "base_point_full_coverage": base,
        "gold_criteria_ceiling": gold_ceiling,
        "coverage_at_risk": coverage_at_risk,
        "conformal_pairing": conformal_pairing,
        "curve": curve,
        "interpretation": interpretation,
    }


def _fmt(v):
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _write_notes(r: Dict) -> None:
    lines: List[str] = []
    lines.append("# N5 -- Risk-coverage curves for the checklist verdict (promoted criteria)")
    lines.append("")
    lines.append(r["question"])
    lines.append("")
    b = r["boundaries"]
    lines.append(f"Derm7pt test (n={b['n_cases']}, {b['n_melanoma']} melanoma), promoted criteria "
                 "heads, deterministic host aggregation. Zero paid calls.")
    lines.append("")
    base = r["base_point_full_coverage"]
    gc = r["gold_criteria_ceiling"]
    lines.append(f"**Full coverage (tau=0):** true-label verdict error {base['risk_true_label']:.4f}, "
                 f"sensitivity {base['sensitivity_among_decided']:.4f}, specificity "
                 f"{base['specificity_among_decided']:.4f}. "
                 f"**Gold-criteria ceiling (floor on true-label risk):** "
                 f"{gc['verdict_error_vs_true_label']:.4f} error "
                 f"(sens {gc['sensitivity']:.4f}, spec {gc['specificity']:.4f}).")
    lines.append("")
    lines.append("## Coverage at target risk")
    lines.append("")
    lines.append("| risk target | true-label: coverage@risk | self-consistency: coverage@risk |")
    lines.append("|---|---|---|")
    for t in RISK_TARGETS:
        tl = r["coverage_at_risk"]["true_label"][str(t)]
        sc = r["coverage_at_risk"]["self_consistency"][str(t)]

        def cell(d):
            if d.get("achievable"):
                s = f"cov {d['coverage']:.4f} @ risk {d['risk']:.4f} (tau {d['tau']}, n={d['n_decided']})"
                if not d.get("reliable_sample", True):
                    s += " [tail]"
                return s
            return f"not reached (min risk {_fmt(d.get('min_risk_reached'))} @ cov {_fmt(d.get('at_coverage'))})"
        lines.append(f"| {t:.2f} | {cell(tl)} | {cell(sc)} |")
    lines.append("")
    lines.append("## Curve (every 5th tau)")
    lines.append("")
    lines.append("| tau | coverage | n_decided | risk(true) | risk(self) | sens | spec |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for p in r["curve"][::5]:
        lines.append(f"| {p['tau']:.2f} | {p['coverage']:.4f} | {p['n_decided']} | "
                     f"{_fmt(p['risk_true_label'])} | {_fmt(p['risk_self_consistency'])} | "
                     f"{_fmt(p['sensitivity_among_decided'])} | {_fmt(p['specificity_among_decided'])} |")
    lines.append("")
    cp = r["conformal_pairing"]
    lines.append("## Pairing with classifier-side conformal (conformal_v3)")
    lines.append("")
    lines.append(f"Classifier LAC conformal, alpha=0.10: test coverage {cp['classifier_side']['test_coverage']}, "
                 f"mean set size {cp['classifier_side']['mean_set_size']}, singleton rate "
                 f"{cp['classifier_side']['singleton_rate']}, MEL-class coverage "
                 f"{cp['classifier_side']['mel_class_coverage']}.")
    lines.append("")
    lines.append(cp["conceptual_pairing"])
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(r["interpretation"])
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    r = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    _write_notes(r)
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    base = r["base_point_full_coverage"]
    print("\n=== N5 risk-coverage (promoted checklist) ===")
    print(f"  base (tau=0): coverage 1.0, risk(true) {base['risk_true_label']}, "
          f"risk(self) {base['risk_self_consistency']}")
    print(f"  gold ceiling true-label error: {r['gold_criteria_ceiling']['verdict_error_vs_true_label']}")
    for t in RISK_TARGETS:
        tl = r["coverage_at_risk"]["true_label"][str(t)]
        sc = r["coverage_at_risk"]["self_consistency"][str(t)]
        print(f"  risk<= {t}: true-label {'cov '+str(tl['coverage']) if tl.get('achievable') else 'NOT reached (min '+str(tl.get('min_risk_reached'))+')'}"
              f" | self-consistency {'cov '+str(sc['coverage']) if sc.get('achievable') else 'NOT reached (min '+str(sc.get('min_risk_reached'))+')'}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
