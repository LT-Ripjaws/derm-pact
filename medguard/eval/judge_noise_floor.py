"""How large must an LLM-judge difference be before it means anything?

The development run judged three arms -- M, A0 and A3 -- that are **byte-identical**:
same contract hash on all 160 lesions, same evidence sets, and the same 122 RAGAS input
rows.  A0 is a declared alias of M, and A3's ablation (bounded remediation) never fired,
because remediation triggered on **0 of 160** development lesions.  Three independent
judgings of the same answers is a test-retest of the judge, obtained for free.

The judge runs at ``temperature=0`` with a fixed seed and still disagrees with itself:
on M vs A3 it scored 46/122 faithfulness samples, 38/122 context-recall samples and
72/122 answer-relevancy samples differently.  Determinism is requested from the gateway,
not guaranteed by it.

Why this matters more than a footnote: on M vs A3 the paired context-recall difference
came out at -0.0489 with a 95% clustered interval of [-0.0922, -0.0047] and a
Holm-corrected p of 0.046.  That is a **statistically significant difference between two
identical systems** -- a demonstrated false positive.  Any judge-based endpoint in this
study must therefore clear a noise floor before it is interpreted, and the floor is
measured here rather than assumed.

The floor is a property of this judge, this prompt set and this answer distribution.  It
is not a general claim about RAGAS.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard.eval.paired_endpoints import (
    CaseSeries,
    PairedEndpointError,
    paired_endpoint_difference,
)

#: Arms proven byte-identical on the development run.  Membership is verified from the
#: run report, never assumed -- see :func:`verify_identical`.
DEFAULT_IDENTICAL_ARMS: Tuple[str, ...] = ("M", "A0", "A3")

#: Judge-scored endpoints the floor applies to.  Host-computed endpoints (citation
#: integrity, contract completion, over-reliance) are deterministic and exempt.
JUDGE_SCORED_ENDPOINTS: Tuple[str, ...] = (
    "ragas_faithfulness",
    "ragas_faithfulness_literature_scoped",
    "ragas_context_recall",
    "ragas_answer_relevancy",
)


def verify_identical(
    raw_report: Mapping[str, Any], arms: Sequence[str],
) -> Dict[str, Any]:
    """Prove the arms really are the same system before using them as a null.

    A noise floor measured from arms that differ would understate real effects, so this
    refuses rather than warns: identity is checked on the contract hash and the evidence
    set of every case.
    """
    cases = raw_report.get("cases") or []
    if not cases:
        raise PairedEndpointError("run report carries no cases")
    reference = arms[0]
    mismatches: Dict[str, Dict[str, int]] = {}
    for arm in arms[1:]:
        contract = sum(
            1 for c in cases
            if (c["arms"][arm].get("contract_sha256")
                != c["arms"][reference].get("contract_sha256"))
        )
        evidence = sum(
            1 for c in cases
            if (c["arms"][arm].get("evidence_document_ids")
                != c["arms"][reference].get("evidence_document_ids"))
        )
        if contract or evidence:
            mismatches[arm] = {"contract_mismatches": contract,
                               "evidence_mismatches": evidence}
    if mismatches:
        raise PairedEndpointError(
            f"arms are not identical to {reference} and cannot form a null: {mismatches}"
        )
    return {"reference": reference, "arms": list(arms), "cases": len(cases),
            "identical": True}


def measure_floor(
    series_by_arm: Mapping[str, Mapping[str, CaseSeries]],
    *,
    endpoints: Sequence[str] = JUDGE_SCORED_ENDPOINTS,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Paired differences between identical systems: the null distribution.

    For each endpoint the floor is the largest absolute paired difference observed
    between systems that are the same.  An effect at or below it is not distinguishable
    from the judge disagreeing with itself.
    """
    arms = list(series_by_arm)
    if len(arms) < 2:
        raise PairedEndpointError("a noise floor needs at least two identical arms")
    per_endpoint: Dict[str, Any] = {}
    for endpoint in endpoints:
        nulls: List[Dict[str, Any]] = []
        for arm_a, arm_b in combinations(arms, 2):
            series_a = series_by_arm[arm_a].get(endpoint)
            series_b = series_by_arm[arm_b].get(endpoint)
            if series_a is None or series_b is None:
                continue
            result = paired_endpoint_difference(series_a, series_b, n_boot=n_boot)
            if result is None:
                continue
            nulls.append({
                "pair": f"{arm_a}-{arm_b}",
                "estimate": result["estimate"],
                "low": result["low"],
                "high": result["high"],
                "p_value": result["p_value"],
                "n_paired": result["n_paired"],
                # A "significant" difference between identical systems is a false
                # positive, and the count of them is the headline of this analysis.
                "false_positive": bool(result["excludes_zero"]),
            })
        if not nulls:
            continue
        magnitudes = [abs(n["estimate"]) for n in nulls]
        widths = [n["high"] - n["low"] for n in nulls]
        per_endpoint[endpoint] = {
            "null_comparisons": nulls,
            "max_abs_null_difference": round(max(magnitudes), 4),
            "mean_abs_null_difference": round(sum(magnitudes) / len(magnitudes), 4),
            "mean_null_interval_width": round(sum(widths) / len(widths), 4),
            "false_positives": sum(1 for n in nulls if n["false_positive"]),
            "comparisons_run": len(nulls),
            "minimum_interpretable_effect": round(max(magnitudes), 4),
        }
    return {
        "role": "measurement_instrument_characterisation",
        "identical_arms": arms,
        "endpoints": per_endpoint,
        "what_this_is": (
            "Paired differences between arms proven to be the same system. Any nonzero "
            "value is the judge disagreeing with itself, not an effect."
        ),
        "how_to_use_it": (
            "Treat minimum_interpretable_effect as a floor: a judge-based paired "
            "difference at or below it is reported as indistinguishable from judge noise "
            "regardless of its interval or its Holm-corrected p-value. Host-computed "
            "endpoints are deterministic and exempt."
        ),
        "scope_limit": (
            "A property of this judge, prompt set and answer distribution -- not a "
            "general claim about RAGAS or about LLM judges."
        ),
    }


def apply_floor(
    comparison: Mapping[str, Any], floor: Mapping[str, Any],
) -> Dict[str, Any]:
    """Annotate a real comparison with the noise-floor verdict."""
    result = dict(comparison)
    endpoint = str(result.get("metric"))
    block = (floor.get("endpoints") or {}).get(endpoint)
    if block is None:
        result["judge_noise_floor"] = None
        result["above_judge_noise_floor"] = None
        result["floor_note"] = "host-computed endpoint: deterministic, floor not applicable"
        return result
    minimum = float(block["minimum_interpretable_effect"])
    magnitude = abs(float(result.get("estimate", 0.0)))
    result["judge_noise_floor"] = minimum
    result["above_judge_noise_floor"] = magnitude > minimum
    result["effect_to_noise_ratio"] = round(magnitude / minimum, 2) if minimum else None
    if not result["above_judge_noise_floor"]:
        result["floor_note"] = (
            f"|{magnitude:.4f}| does not exceed the {minimum:.4f} difference observed "
            f"between identical systems; report as indistinguishable from judge noise"
        )
        result["favours"] = "indistinguishable_from_judge_noise"
    else:
        result["floor_note"] = (
            f"|{magnitude:.4f}| is {result['effect_to_noise_ratio']}x the largest "
            f"difference seen between identical systems"
        )
    return result
