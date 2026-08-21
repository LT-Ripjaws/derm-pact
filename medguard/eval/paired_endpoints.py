"""Paired, lesion-clustered differences on the registered evidence/process family.

``score_h2h`` emits paired CIs for *diagnostic accuracy* only.  But the registered
acceptance condition (``notes/TARGETS_TO_BEAT.md`` sec. 6, mirrored in
``research/PREREGISTRATION.json``) is an **evidence** condition:

    evidence superiority: paired improvement with a 95% clustered-bootstrap interval
    excluding zero on at least two of faithfulness, context recall, citation
    precision, or contract completion

This module supplies exactly that, offline and with **zero judge calls**, by
re-joining per-sample scores that a completed run already stored back onto their
case ids.

Reconstruction rule (asserted, never assumed)
---------------------------------------------
``h2h_scoring._ragas_block`` builds its samples from ``sorted(outcomes,
key=case_id)`` filtered to answered-with-text, and ``RagasFaithfulnessJudge.evaluate``
preserves input order (``pool.map``).  Therefore ``per_sample[i]`` belongs to the
i-th *answered* case in case-id order.  :func:`answered_case_ids` rebuilds that list
from the run report and cross-checks its length against the stored ``n_samples`` for
every arm; a drift raises rather than silently mis-joining scores to lesions.

Coverage honesty
----------------
M abstains; the baselines do not.  Any RAGAS endpoint therefore pairs only over the
cases *both* arms answered, which conditions on M having answered.  Every result
carries ``mean_a_unconditional`` / ``mean_b_unconditional`` beside the conditional
means so a reader can see whether a difference is a coverage artifact -- the
registered "no hidden coverage win" rule.  ``over_reliance`` needs no such caveat:
it is defined over *all* cases for every arm (an abstention is neither correct nor
over-reliant, but still sits in the denominator), which is why it is the fairest
single number in the family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard.eval.metrics import (
    completeness as _completeness,
    copy_reference_ratio as _copy_reference_ratio,
)
from medguard.eval.statistics import (
    bootstrap_two_sided_p,
    clustered_bootstrap_distribution,
    holm_adjust,
)


REGISTERED_BOOTSTRAP_REPLICATES = 2000
REGISTERED_BOOTSTRAP_SEED = 20260712

#: The evidence family frozen before development H2H.  Holm runs *within* this set.
REGISTERED_EVIDENCE_FAMILY: Tuple[str, ...] = (
    "ragas_faithfulness",
    "ragas_context_recall",
    "citation_integrity",
    "contract_completion",
)

#: Reported beside the family but NOT folded into it -- adding endpoints to a frozen
#: family after the fact would buy a Holm advantage the registration did not grant.
SUPPORTING_ENDPOINTS: Tuple[str, ...] = (
    "over_reliance",
    "ragas_answer_relevancy",
    "ragas_faithfulness_literature_scoped",
)

#: Spans a text-entailment judge has jurisdiction over: the arm actually produced a
#: literature-grounded claim.  ``whole`` means the judge fell back to grading prose
#: with no citation apparatus at all -- for MedGuard that is the image-derived
#: ``Visual findings:``/``Diagnosis:`` text on lesions whose condition has no licensed
#: guideline, which is a modality mismatch rather than an unfaithful answer.
LITERATURE_SCOPES: Tuple[str, ...] = ("assessment", "citations")

#: Endpoints that were NOT part of the frozen registration and must never be counted
#: toward the acceptance condition on this development set.
POST_HOC_ENDPOINTS: Tuple[str, ...] = (
    "ragas_faithfulness_literature_scoped",
    "ragas_context_recall_guideline_eligible",
)

#: The spec-v6 inferential families (see research/EXPERIMENT_SPEC_V6.json).
#:
#: The v5 family failed because two of its four members are undefined for a no-contract
#: arm.  The fix is NOT to swap in endpoints MedGuard wins by construction -- that would
#: be a test whose outcome the architecture already decided.
#:
#: Membership follows one stated rule, applied before looking at who wins: an endpoint
#: belongs if it is **defined for every arm** and its outcome is **not architecturally
#: determined**.  Applying that rule honestly admits endpoints MedGuard LOSES --
#: answer relevancy, completeness and copy-reference all favour the baselines -- and
#: they are in the family for exactly that reason.  A family containing only the three
#: MedGuard wins would be cherry-picking with a principle bolted on afterwards.
#:
#: Two families, split the way the preregistration already splits outcome families:
#: evidence *use* versus answer *behaviour*.  Holm runs within each.
V6_EVIDENCE_USE_FAMILY: Tuple[str, ...] = (
    "ragas_context_recall",
    "ragas_faithfulness_literature_scoped",
)

V6_ALIGNMENT_FAMILY: Tuple[str, ...] = (
    "over_reliance",
    "ragas_answer_relevancy",
    "completeness",
    "copy_reference",
)

V6_INFERENTIAL_FAMILY: Tuple[str, ...] = V6_EVIDENCE_USE_FAMILY + V6_ALIGNMENT_FAMILY

#: Endpoints whose outcome the architecture determines in advance.  These are the
#: paper's strongest numbers and its least inferential ones; reporting them as
#: hypothesis tests would be dishonest, so they are reported as counts with their
#: denominators and are barred from every family.
ARCHITECTURAL_ENDPOINTS: Tuple[str, ...] = (
    "citation_integrity",
    "contract_completion",
    "verifiable_citation_coverage",
    "citation_resolvability",
    "process_trust_composite",
)

#: HAM classes for which the licensed guideline corpus holds any recommendation.
#: Formal dermatology guidelines concentrate on malignancy, so the four benign classes
#: have none -- a property of the guideline literature, not a defect in retrieval.
GUIDELINE_ELIGIBLE_LABELS: Tuple[str, ...] = ("akiec", "bcc", "melanoma")

#: Endpoints where a smaller value is the better one.
LOWER_IS_BETTER: Tuple[str, ...] = ("over_reliance", "copy_reference")

#: Actions that mean "this arm produced an answer for this lesion".
ANSWERED_ACTIONS = ("PASS", "ANSWER_UNGATED")


class PairedEndpointError(ValueError):
    """Raised when a report cannot be re-joined safely enough to pair."""


@dataclass(frozen=True)
class CaseSeries:
    """One arm's per-case values for one endpoint, plus its lesion clustering."""

    arm_id: str
    endpoint: str
    values: Mapping[str, Optional[float]]
    clusters: Mapping[str, str]

    def defined(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.values.items() if v is not None}


# --------------------------------------------------------------------------------
# Re-joining a stored run
# --------------------------------------------------------------------------------
def _case_rows(raw_report: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    cases = raw_report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise PairedEndpointError("run report carries no cases")
    return sorted(cases, key=lambda c: str(c["case_id"]))


def arm_answered(case_row: Mapping[str, Any], arm_id: str) -> bool:
    """Did ``arm_id`` answer this lesion?

    Three shapes exist in the report and all three are handled explicitly rather
    than defaulted: contract arms emit ``final_action``; retrieval baselines emit an
    ``answered`` flag; B0 emits neither and answers whenever it produced a
    diagnosis without a generation error.
    """
    row = (case_row.get("arms") or {}).get(arm_id)
    if row is None:
        raise PairedEndpointError(f"arm {arm_id} missing from case {case_row.get('case_id')}")
    if row.get("generation_error"):
        return False
    if "final_action" in row:
        return str(row.get("final_action") or "") in ANSWERED_ACTIONS
    if "answered" in row:
        return bool(row.get("answered"))
    return bool(str(row.get("predicted_diagnosis") or "").strip())


def answered_case_ids(raw_report: Mapping[str, Any], arm_id: str) -> List[str]:
    """Case ids, in the order the RAGAS sample list was built for ``arm_id``."""
    return [
        str(case["case_id"]) for case in _case_rows(raw_report)
        if arm_answered(case, arm_id)
    ]


def predicted_diagnosis(case_row: Mapping[str, Any], arm_id: str) -> str:
    """The label this arm committed for this lesion.

    Only B0 diagnoses independently (prereg schema 5 unfroze it alone); every other
    arm commits the shared frozen probe recorded at case level.
    """
    row = (case_row.get("arms") or {}).get(arm_id) or {}
    own = str(row.get("predicted_diagnosis") or "").strip()
    return own or str(case_row.get("committed_diagnosis") or "").strip()


def _ragas_per_sample(
    scored_report: Mapping[str, Any], arm_id: str, metric: str,
) -> Tuple[List[Optional[float]], int]:
    block = ((scored_report.get("per_arm") or {}).get(arm_id) or {}).get("ragas_supporting") or {}
    n_samples = int(block.get("n_samples") or 0)
    scores = block.get("scores")
    if not isinstance(scores, Mapping):
        return [], n_samples
    metric_block = scores.get(metric)
    if not isinstance(metric_block, Mapping):
        return [], n_samples
    per_sample = metric_block.get("per_sample")
    if not isinstance(per_sample, list):
        return [], n_samples
    return [None if v is None else float(v) for v in per_sample], n_samples


def _sample_rows(scored_report: Mapping[str, Any], arm_id: str) -> List[Mapping[str, Any]]:
    block = ((scored_report.get("per_arm") or {}).get(arm_id) or {}).get("ragas_supporting") or {}
    rows = block.get("sample_rows")
    return [r for r in rows if isinstance(r, Mapping)] if isinstance(rows, list) else []


def _faithfulness_scopes(scored_report: Mapping[str, Any], arm_id: str) -> List[str]:
    block = ((scored_report.get("per_arm") or {}).get(arm_id) or {}).get("ragas_supporting") or {}
    scores = block.get("scores")
    if not isinstance(scores, Mapping):
        return []
    metric_block = scores.get("faithfulness")
    if not isinstance(metric_block, Mapping):
        return []
    scopes = metric_block.get("per_sample_scope")
    return [str(s) for s in scopes] if isinstance(scopes, list) else []


def load_case_series(
    raw_report: Mapping[str, Any],
    scored_report: Mapping[str, Any],
    arm_id: str,
    truth: Mapping[str, str],
) -> Dict[str, CaseSeries]:
    """Every pairable endpoint for one arm, keyed by case id.

    Raises if the reconstructed answered-case list disagrees with the sample count
    the run actually scored -- that mismatch would silently attach one lesion's
    judge score to another lesion.
    """
    cases = _case_rows(raw_report)
    clusters = {str(c["case_id"]): str(c["cluster_id"]) for c in cases}
    answered = {str(c["case_id"]): arm_answered(c, arm_id) for c in cases}
    answered_ids = [cid for cid in (str(c["case_id"]) for c in cases) if answered[cid]]

    series: Dict[str, CaseSeries] = {}
    for metric, endpoint in (
        ("faithfulness", "ragas_faithfulness"),
        ("context_recall", "ragas_context_recall"),
        ("answer_relevancy", "ragas_answer_relevancy"),
    ):
        per_sample, n_samples = _ragas_per_sample(scored_report, arm_id, metric)
        if n_samples and n_samples != len(answered_ids):
            raise PairedEndpointError(
                f"arm {arm_id}: reconstructed {len(answered_ids)} answered cases but the "
                f"report scored {n_samples} RAGAS samples; the per-sample join is unsafe"
            )
        if not per_sample:
            continue
        if len(per_sample) != len(answered_ids):
            raise PairedEndpointError(
                f"arm {arm_id} metric {metric}: {len(per_sample)} per-sample scores for "
                f"{len(answered_ids)} answered cases"
            )
        values: Dict[str, Optional[float]] = {cid: None for cid in clusters}
        values.update(dict(zip(answered_ids, per_sample)))
        series[endpoint] = CaseSeries(arm_id, endpoint, values, clusters)

    # POST-HOC secondary: faithfulness restricted to the cases where the text judge had
    # jurisdiction at all.  MedGuard's answered-mean mixes 0.97 on cited guideline
    # assessments with ~0.16 on image-derived prose for conditions that have no licensed
    # guideline -- the latter is a modality mismatch, not an unfaithful answer.  This is
    # NOT in the frozen family and cannot satisfy the acceptance condition on dev.
    scopes = _faithfulness_scopes(scored_report, arm_id)
    faith = series.get("ragas_faithfulness")
    if faith is not None and len(scopes) == len(answered_ids):
        in_scope = {
            cid: (faith.values[cid] if scope in LITERATURE_SCOPES else None)
            for cid, scope in zip(answered_ids, scopes)
        }
        restricted: Dict[str, Optional[float]] = {cid: None for cid in clusters}
        restricted.update(in_scope)
        series["ragas_faithfulness_literature_scoped"] = CaseSeries(
            arm_id, "ragas_faithfulness_literature_scoped", restricted, clusters)

    # Completeness and copy-reference, recomputed per case from the stored answer and
    # its retrieved contexts.  Both favour the BASELINES; they are here because the
    # family rule admits them, not despite it.
    rows = _sample_rows(scored_report, arm_id)
    if rows and len(rows) == len(answered_ids):
        for endpoint, scorer in (
            ("completeness", _completeness), ("copy_reference", _copy_reference_ratio),
        ):
            values: Dict[str, Optional[float]] = {cid: None for cid in clusters}
            for cid, row in zip(answered_ids, rows):
                answer = str(row.get("response") or "")
                chunks = [{"text": str(c)} for c in (row.get("retrieved_contexts") or [])]
                values[cid] = float(scorer(answer, chunks)) if answer.strip() else None
            series[endpoint] = CaseSeries(arm_id, endpoint, values, clusters)

    # POST-HOC secondary: context recall restricted to lesions whose committed condition
    # has any recommendation in the licensed corpus.  On a benign lesion the constructed
    # reference degenerates to a bare label ("Diagnosis: nevus.") and the retrieved
    # guideline context cannot contain it -- no guideline for benign nevi exists -- so
    # the metric is scoring corpus coverage rather than retrieval quality.  Eligibility
    # is fixed by the frozen probe's commitment and the frozen corpus, both settled
    # before any answer exists.  NOT in the frozen family; unusable on development.
    recall = series.get("ragas_context_recall")
    if recall is not None:
        eligible: Dict[str, Optional[float]] = {}
        for case in cases:
            cid = str(case["case_id"])
            in_scope = predicted_diagnosis(case, arm_id) in GUIDELINE_ELIGIBLE_LABELS
            eligible[cid] = recall.values.get(cid) if in_scope else None
        series["ragas_context_recall_guideline_eligible"] = CaseSeries(
            arm_id, "ragas_context_recall_guideline_eligible", eligible, clusters)

    # Verifiable-citation coverage: does this answer carry at least one citation that
    # resolves to a versioned, content-hashed source?  Defined for EVERY arm (a baseline
    # scores a real 0, not an undefined), which is exactly what citation_integrity is
    # not -- but the outcome is settled by architecture, so it is descriptive only.
    coverage: Dict[str, Optional[float]] = {}
    for case in cases:
        cid = str(case["case_id"])
        row = (case.get("arms") or {}).get(arm_id) or {}
        resolvable = len(row.get("guideline_citations") or ())
        coverage[cid] = 1.0 if resolvable else 0.0
    series["verifiable_citation_coverage"] = CaseSeries(
        arm_id, "verifiable_citation_coverage", coverage, clusters)

    # Over-reliance: answered-but-wrong, denominator = every lesion (see module docstring).
    over: Dict[str, Optional[float]] = {}
    for case in cases:
        cid = str(case["case_id"])
        gold = truth.get(clusters[cid], truth.get(cid))
        if gold is None:
            raise PairedEndpointError(f"no development label for lesion {clusters[cid]}")
        wrong = predicted_diagnosis(case, arm_id) != gold
        over[cid] = 1.0 if (answered[cid] and wrong) else 0.0
    series["over_reliance"] = CaseSeries(arm_id, "over_reliance", over, clusters)
    return series


# --------------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def paired_endpoint_difference(
    series_a: CaseSeries,
    series_b: CaseSeries,
    *,
    n_boot: int = REGISTERED_BOOTSTRAP_REPLICATES,
    seed: int = REGISTERED_BOOTSTRAP_SEED,
) -> Optional[Dict[str, Any]]:
    """Lesion-clustered paired difference ``a - b`` on one endpoint.

    Pairs only lesions where *both* arms have a defined value; an undefined score is
    never imputed as zero (an arm that cites nothing has no integrity, not integrity
    of 0.0).  Returns ``None`` when nothing pairs.
    """
    if series_a.endpoint != series_b.endpoint:
        raise PairedEndpointError(
            f"cannot pair {series_a.endpoint} against {series_b.endpoint}")
    endpoint = series_a.endpoint
    defined_a, defined_b = series_a.defined(), series_b.defined()
    shared = sorted(set(defined_a) & set(defined_b))
    if not shared:
        return None
    rows = [(defined_a[cid], defined_b[cid]) for cid in shared]
    clusters = [series_a.clusters[cid] for cid in shared]
    interval, samples = clustered_bootstrap_distribution(
        rows, clusters,
        lambda pairs: (
            sum(p[0] for p in pairs) / len(pairs) - sum(p[1] for p in pairs) / len(pairs)
        ),
        n_boot=n_boot, seed=seed,
    )
    lower_better = endpoint in LOWER_IS_BETTER
    excludes_zero = interval.low > 0.0 or interval.high < 0.0
    improvement = -interval.estimate if lower_better else interval.estimate
    if not excludes_zero:
        favours = "inconclusive"
    else:
        favours = series_a.arm_id if improvement > 0 else series_b.arm_id
    return {
        "arm_a": series_a.arm_id,
        "arm_b": series_b.arm_id,
        "metric": endpoint,
        "post_hoc": endpoint in POST_HOC_ENDPOINTS,
        "higher_is_better": not lower_better,
        "n_paired": len(shared),
        "estimate": interval.estimate,
        "low": interval.low,
        "high": interval.high,
        "p_value": bootstrap_two_sided_p(samples),
        "excludes_zero": excludes_zero,
        "improvement_for_arm_a": improvement,
        "favours": favours,
        "mean_a_paired": _mean([r[0] for r in rows]),
        "mean_b_paired": _mean([r[1] for r in rows]),
        "mean_a_unconditional": _mean(list(defined_a.values())),
        "mean_b_unconditional": _mean(list(defined_b.values())),
        "n_defined_a": len(defined_a),
        "n_defined_b": len(defined_b),
        "coverage_conditioned": len(shared) < max(len(defined_a), len(defined_b)),
    }


def architectural_endpoint(
    endpoint: str, arm_a: str, arm_b: str, value_a: Optional[float], value_b: Optional[float],
) -> Dict[str, Any]:
    """Record an endpoint that one arm defines and the other structurally cannot.

    Citation integrity and contract completion are ``not_applicable`` for a
    no-contract baseline by registration -- never imputed as 0.0 -- so they are not
    testable and must be reported as an architectural difference instead of a
    silently-won comparison.
    """
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "metric": endpoint,
        "testable": False,
        "reason": "not_applicable_for_comparator",
        "value_a": value_a,
        "value_b": value_b,
        "note": (
            f"{arm_b} produces no contract, so {endpoint} is undefined for it and is "
            "never imputed as zero; the difference is architectural, not statistical."
        ),
    }


def sealed_scale_projection(
    comparison: Mapping[str, Any], *, dev_lesions: int, sealed_lesions: int,
) -> Optional[Dict[str, Any]]:
    """Would this endpoint's interval still contain zero at the sealed sample size?

    A **projection, not a result.**  The clustered-bootstrap half-width shrinks with
    ``1/sqrt(clusters)``, so it rescales the observed half-width by
    ``sqrt(dev/sealed)`` and asks whether the observed point estimate would then clear
    zero.  It holds three things fixed that the sealed run may not: the effect size,
    the fraction of lesions on which the endpoint is defined, and the cluster
    structure.  Its only legitimate use is deciding whether an inconclusive dev
    endpoint is underpowered or genuinely null -- never as evidence of an effect.
    """
    if not comparison.get("testable", True) or "low" not in comparison:
        return None
    if dev_lesions <= 0 or sealed_lesions <= 0:
        raise PairedEndpointError("lesion counts must be positive")
    half_width = (float(comparison["high"]) - float(comparison["low"])) / 2.0
    scale = (float(dev_lesions) / float(sealed_lesions)) ** 0.5
    projected = half_width * scale
    estimate = float(comparison["estimate"])
    return {
        "metric": comparison["metric"],
        "observed_half_width": round(half_width, 4),
        "projected_half_width": round(projected, 4),
        "projected_n_paired": int(round(
            float(comparison["n_paired"]) * sealed_lesions / dev_lesions)),
        "estimate": estimate,
        "already_excludes_zero": bool(comparison.get("excludes_zero")),
        "would_exclude_zero_at_sealed_scale": abs(estimate) > projected,
        "status": "projection_not_result",
        "assumptions": [
            "effect size unchanged at sealed scale",
            "same fraction of lesions have a defined score",
            "similar lesion-cluster structure",
        ],
    }


def evidence_family_result(
    comparisons: Sequence[Mapping[str, Any]], *, alpha: float = 0.05,
) -> Dict[str, Any]:
    """Holm-correct the testable evidence-family comparisons and judge the condition.

    Holm's ``m`` is the number of tests actually performed, not the frozen family's
    nominal size: an endpoint that is architecturally undefined for the comparator was
    never a test, and inflating ``m`` with it would only make the surviving tests
    harder to pass for no inferential reason.
    """
    testable = [c for c in comparisons if c.get("testable", True) and "p_value" in c]
    raw_p = {str(c["metric"]): float(c["p_value"]) for c in testable}
    adjusted = holm_adjust(raw_p) if raw_p else {}
    enriched: List[Dict[str, Any]] = []
    for comparison in comparisons:
        item = dict(comparison)
        metric = str(item.get("metric"))
        if metric in adjusted:
            item["p_value_holm"] = adjusted[metric]
            item["significant_after_holm"] = bool(
                adjusted[metric] < alpha and item.get("excludes_zero")
            )
        enriched.append(item)
    passing = [
        c for c in enriched
        if c.get("metric") in REGISTERED_EVIDENCE_FAMILY
        and c.get("significant_after_holm")
        and c.get("improvement_for_arm_a", 0.0) > 0
    ]
    return {
        "family": list(REGISTERED_EVIDENCE_FAMILY),
        "holm_m": len(raw_p),
        "alpha": alpha,
        "comparisons": enriched,
        "endpoints_passing": [str(c["metric"]) for c in passing],
        "n_endpoints_passing": len(passing),
        "registered_condition": (
            "paired improvement with a 95% clustered-bootstrap interval excluding zero "
            "on at least two of faithfulness, context recall, citation precision, or "
            "contract completion"
        ),
        "condition_met": len(passing) >= 2,
    }
