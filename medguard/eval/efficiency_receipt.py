"""The registered efficiency endpoint, recovered from the run that actually paid.

``efficiency: calls_per_answered_case`` is a pre-registered primary outcome, but the
final v4 report shows zeros for it: v2-v4 were full cache replays (``remote_calls=0``
by design, which is what made re-measuring free).  The measured cost lives in the
**first** full run, v1, where every arm paid.

Carrying v1's cost into the v4 result is only legitimate if the two runs generated the
same things.  :func:`build_efficiency_receipt` does not assume that -- it *checks* it,
comparing per-arm answered coverage between the cost run and the result run and
refusing to emit a receipt if any arm drifted.  The v2-v4 changes were rendering-only
(they altered how a stored contract is written out, not which lesions were answered or
how many calls were made), and this check is what turns that claim into a verified one.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


class EfficiencyReceiptError(ValueError):
    """Raised when a cost run cannot be attributed to a result run."""


def _coverage(report: Mapping[str, Any], arm: str) -> Optional[float]:
    block = ((report.get("per_arm") or {}).get(arm) or {}).get("selective") or {}
    value = block.get("coverage")
    return None if value is None else float(value)


def _efficiency(report: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    return dict(((report.get("per_arm") or {}).get(arm) or {}).get("efficiency") or {})


def _alias_of(report: Mapping[str, Any], arm: str) -> Optional[str]:
    return ((report.get("per_arm") or {}).get(arm) or {}).get("alias_of")


def build_efficiency_receipt(
    cost_report: Mapping[str, Any],
    result_report: Mapping[str, Any],
    *,
    n_cases: int,
    cost_run_id: str,
    result_run_id: str,
) -> Dict[str, Any]:
    """Per-arm measured cost plus ``calls_per_answered_case``.

    Raises if any arm's answered coverage differs between the two runs, because then
    the cost run is not describing the result run's behaviour.
    """
    if n_cases <= 0:
        raise EfficiencyReceiptError("n_cases must be positive")
    cost_arms = set((cost_report.get("per_arm") or {}))
    result_arms = set((result_report.get("per_arm") or {}))
    if cost_arms != result_arms:
        raise EfficiencyReceiptError(
            f"arm rosters differ: cost-only {sorted(cost_arms - result_arms)}, "
            f"result-only {sorted(result_arms - cost_arms)}"
        )

    arms: Dict[str, Any] = {}
    for arm in sorted(result_arms):
        cost_coverage, result_coverage = _coverage(cost_report, arm), _coverage(result_report, arm)
        if cost_coverage != result_coverage:
            raise EfficiencyReceiptError(
                f"arm {arm}: answered coverage {cost_coverage} in the cost run but "
                f"{result_coverage} in the result run; cost cannot be carried forward"
            )
        measured = _efficiency(cost_report, arm)
        calls = int(measured.get("generation_calls") or 0)
        tokens = int(measured.get("total_tokens") or 0)
        answered = int(round((result_coverage or 0.0) * n_cases))
        cache_hits = int(measured.get("cache_hits") or 0)
        measured_latency = float(measured.get("latency_seconds") or 0.0)
        alias = _alias_of(result_report, arm)
        arms[arm] = {
            "answered_cases": answered,
            # A0 is M by construction and replays M's generations; reporting it as an
            # independently-costed arm would double-count M's spend and understate its
            # per-call figure.  Its cost is M's.
            "alias_of": alias,
            "cost_attributed_to": alias or arm,
            "remote_calls": calls,
            "cache_hits": cache_hits,
            "prompt_tokens": int(measured.get("prompt_tokens") or 0),
            "completion_tokens": int(measured.get("completion_tokens") or 0),
            "total_tokens": tokens,
            "latency_seconds": round(measured_latency, 1),
            "generation_errors": int(measured.get("generation_errors") or 0),
            "schema_repairs": int(measured.get("schema_repairs") or 0),
            # The registered primary.  Cache hits are excluded from the numerator on
            # purpose: this is the *marginal* cost of answering one more lesion, and a
            # replayed lesion costs nothing.  ``calls_per_answered_case_cold`` is the
            # cold-cache figure a reader would pay to reproduce the arm from scratch.
            "calls_per_answered_case": (
                None if alias else (round(calls / answered, 4) if answered else None)),
            "calls_per_answered_case_cold": (
                None if alias else
                (round((calls + cache_hits) / answered, 4) if answered else None)),
            # Per-CALL rates are the only cache-independent measurements here: an arm
            # that was partly replayed paid tokens on its cold calls only, so dividing
            # its token spend by its (larger) answered count understates it badly.
            # M is the case in point -- 50 cold calls but 122 answers.
            "tokens_per_call": round(tokens / calls, 1) if calls else None,
            "latency_seconds_per_call": round(measured_latency / calls, 2) if calls else None,
        }
        cold_rate = arms[arm]["calls_per_answered_case_cold"]
        per_call = arms[arm]["tokens_per_call"]
        latency_per_call = arms[arm]["latency_seconds_per_call"]
        # What answering one lesion costs from a cold cache: the honest comparison
        # number, and the one the matched-budget claim has to be made against.
        arms[arm]["tokens_per_answered_case_cold"] = (
            round(per_call * cold_rate, 1) if (per_call and cold_rate) else None)
        arms[arm]["latency_seconds_per_answered_case_cold"] = (
            round(latency_per_call * cold_rate, 2)
            if (latency_per_call and cold_rate) else None)
        arms[arm]["total_tokens_cold_for_full_set"] = (
            round(per_call * cold_rate * answered) if (per_call and cold_rate) else None)
    return {
        "endpoint": "calls_per_answered_case",
        "role": "registered_primary_efficiency",
        "cost_run": cost_run_id,
        "result_run": result_run_id,
        "n_cases": n_cases,
        "attribution_rule": (
            "v2-v4 were cache replays and report remote_calls=0; the measured cost is "
            "the v1 full run. Carrying it forward is verified, not assumed: every arm's "
            "answered coverage is identical in both runs (checked above), so the same "
            "lesions were answered by the same generations."
        ),
        "cold_cache_note": (
            "remote_calls counts only calls that reached the gateway. An arm with cache "
            "hits in the cost run was partly replayed even then; "
            "calls_per_answered_case_cold adds those back. Any rate whose numerator is a "
            "cold-cache measurement (tokens, latency) and whose denominator is a "
            "warm-cache count (answered cases) is meaningless -- M paid tokens on 50 "
            "calls but answered 122 lesions -- so token and latency comparisons are "
            "reported per CALL and then scaled by calls_per_answered_case_cold."
        ),
        "arms": arms,
    }
