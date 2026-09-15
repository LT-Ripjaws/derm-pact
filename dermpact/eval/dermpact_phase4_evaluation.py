"""Phase 4 scoring: RQ2 endpoints with case-clustered bootstrap."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

STATES = ("applicable", "not_applicable", "applicability_unknown", "no_catalogue_coverage")


def score_run(run: Mapping[str, Any], out: Mapping[str, Any]) -> Dict[str, Any]:
    gold: Mapping[str, Mapping[str, Any]] = run["gold"]
    per = out.get("per_rule", {})
    refused = bool(out.get("refused"))
    expected_refusal = bool(run.get("expected_refusal"))

    if expected_refusal:
        wrong_next_step = not refused and bool(out.get("released_actions"))
        return {
            "run_id": run["run_id"], "case_id": run["case_id"], "family": run["family"],
            "wrong_or_inapplicable_next_step": int(wrong_next_step or (not refused)),
            "entity_refusal_correct": int(refused),
            "usable_release_when_supported": None, "state_pairs": [], "missing_hit": [], "malformed": int(False),
        }

    gold_supported = {(rid, g["action"]) for rid, g in gold.items() if g["state"] == "applicable"}
    released = {tuple(x) for x in out.get("released_actions", [])}
    wrong = bool(released - gold_supported) or refused
    usable = None
    if gold_supported:
        usable = int(bool(released & gold_supported))
    pairs: List[Tuple[str, str]] = []
    missing_hits: List[int] = []
    malformed = 0
    for rid, g in gold.items():
        p = per.get(rid, {"state": "malformed"})
        state = p.get("state", "malformed")
        if state not in STATES:
            malformed = 1
            state = "malformed"
        pairs.append((g["state"], state))
        if g["state"] == "applicability_unknown":
            missing_hits.append(int(p.get("missing") == g["missing"]))
    return {
        "run_id": run["run_id"], "case_id": run["case_id"], "family": run["family"],
        "wrong_or_inapplicable_next_step": int(wrong),
        "entity_refusal_correct": None,
        "usable_release_when_supported": usable,
        "state_pairs": pairs, "missing_hit": missing_hits, "malformed": malformed,
    }


def macro_f1(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    per_class = {}
    f1s = []
    for s in STATES:
        tp = sum(1 for g, p in pairs if g == s and p == s)
        fp = sum(1 for g, p in pairs if g != s and p == s)
        fn = sum(1 for g, p in pairs if g == s and p != s)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(1 for g, _ in pairs if g == s)
        per_class[s] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {"macro_f1": float(np.mean(f1s)) if f1s else 0.0, "per_class": per_class,
            "accuracy": float(np.mean([g == p for g, p in pairs])) if pairs else 0.0}


def _rate(values: Sequence[Any]) -> Tuple[float, int]:
    vals = [v for v in values if v is not None]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def summarise_arm(scored: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = [p for s in scored for p in s["state_pairs"]]
    missing = [m for s in scored for m in s["missing_hit"]]
    by_family: Dict[str, List[int]] = defaultdict(list)
    for s in scored:
        by_family[s["family"]].append(s["wrong_or_inapplicable_next_step"])
    primary, n = _rate([s["wrong_or_inapplicable_next_step"] for s in scored])
    return {
        "n_runs": len(scored),
        "wrong_or_inapplicable_next_step_rate": primary,
        "wrong_or_inapplicable_next_step_count": int(sum(s["wrong_or_inapplicable_next_step"] for s in scored)),
        "entity_refusal_rate": _rate([s["entity_refusal_correct"] for s in scored])[0],
        "usable_release_when_supported_rate": _rate([s["usable_release_when_supported"] for s in scored])[0],
        "decisive_missing_fact_hit_rate": _rate(missing)[0],
        "n_missing_fact_opportunities": len(missing),
        "malformed_rate": _rate([s["malformed"] for s in scored])[0],
        "four_state": macro_f1(pairs),
        "by_family_wrong_rate": {f: float(np.mean(v)) for f, v in sorted(by_family.items())},
    }


def clustered_bootstrap_rate(scored: Sequence[Mapping[str, Any]], *, n_boot: int, seed: int) -> Dict[str, Any]:
    by_case: Dict[str, List[int]] = defaultdict(list)
    for s in scored:
        by_case[s["case_id"]].append(s["wrong_or_inapplicable_next_step"])
    cases = sorted(by_case)
    rng = random.Random(seed)
    point = float(np.mean([v for c in cases for v in by_case[c]]))
    samples = []
    for _ in range(n_boot):
        draw = [cases[rng.randrange(len(cases))] for _ in cases]
        vals = [v for c in draw for v in by_case[c]]
        samples.append(float(np.mean(vals)))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    boundary = None
    if point == 0.0:
        boundary = {"kind": "upper", "value": float(1.0 - 0.05 ** (1.0 / len(cases))),
                    "assumption": "independent_case_clusters; zero observed positive cases"}
    elif point == 1.0:
        boundary = {"kind": "lower", "value": float(0.05 ** (1.0 / len(cases))),
                    "assumption": "independent_case_clusters; all observed cases positive"}
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot,
            "n_cases": len(cases), "boundary_case_bound_one_sided_95": boundary}


def paired_risk_difference(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], *, n_boot: int, seed: int) -> Dict[str, float]:
    """Rate(a) - Rate(b) on identical runs, case-clustered percentile bootstrap."""
    a_by = {s["run_id"]: s["wrong_or_inapplicable_next_step"] for s in a}
    b_by = {s["run_id"]: s["wrong_or_inapplicable_next_step"] for s in b}
    if set(a_by) != set(b_by):
        raise ValueError("paired arms must share identical run ids")
    by_case: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    case_of = {s["run_id"]: s["case_id"] for s in a}
    for rid in a_by:
        by_case[case_of[rid]].append((a_by[rid], b_by[rid]))
    cases = sorted(by_case)
    diffs_all = [x - y for c in cases for x, y in by_case[c]]
    point = float(np.mean(diffs_all))
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [cases[rng.randrange(len(cases))] for _ in cases]
        samples.append(float(np.mean([x - y for c in draw for x, y in by_case[c]])))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "n_cases": len(cases),
            "n_runs": len(a_by)}


__all__ = ["clustered_bootstrap_rate", "macro_f1", "paired_risk_difference", "score_run", "summarise_arm"]
