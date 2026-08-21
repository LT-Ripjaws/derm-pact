"""Truth-joined H2H scoring: one place that turns arm outputs into endpoints.

This is the *separately joined evaluator* the protocol allows: arms run on
truth-redacted inputs and emit ``ArmCaseOutcome`` rows; here, and only here, those
rows are joined against development labels to compute the paper's endpoints.  The
arms never see a label; the runner passes this scorer the labels out of band.

Every scorer is wired, none is rebuilt:

* diagnosis -- ``metrics.classification_summary`` (7-class + per-class);
* actionable -- ``metrics.actionable_confusion`` + ``sensitivity_at_fixed_specificity``
  at the registered 0.90 specificity, AUROC over the actionable score;
* provenance (primary) -- ``citation_integrity`` over guideline citations only;
* process -- evidence-contract completion, ``not_applicable`` (never zero) for
  arms without a contract;
* selective -- ``metrics.aurc`` / ``risk_coverage_curve`` on calibrated confidence;
* efficiency -- calls/tokens/latency/errors summed from generation metadata;
* uncertainty -- lesion-clustered bootstrap CIs (2000 replicates, seed 20260712);
* RAGAS -- a supporting field-mapping only, never executed here.

Undefined endpoints are ``None``, never 0.0 or a free perfect score: an arm that
answers nothing has no accuracy, and an arm that cites nothing has no integrity.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard.config import MALIGNANT_LABELS
from medguard.eval.citation_integrity import (
    AssertedCitation,
    CaseCitationScore,
    CitationResolver,
    score_case,
    summarize as summarize_citations,
)
from medguard.eval.metrics import (
    DERM_CLASS_ORDER,
    actionable_confusion,
    aurc,
    classification_summary,
    completeness as _completeness,
    copy_reference_ratio,
    ece,
    over_reliance,
    risk_coverage_curve,
    sensitivity_at_fixed_specificity,
)
from medguard.eval.ragas_adapter import (
    RagasJudgeLock,
    RagasSupportingAdapter,
    map_arm_case_to_sample,
)
from medguard.eval.statistics import (
    Interval,
    clustered_bootstrap_ci,
    paired_clustered_bootstrap_difference,
)


REGISTERED_TARGET_SPECIFICITY = 0.90
REGISTERED_BOOTSTRAP_REPLICATES = 2000
REGISTERED_BOOTSTRAP_SEED = 20260712


@dataclass(frozen=True)
class ArmCaseOutcome:
    """One arm's truth-free result on one lesion, ready for truth-joined scoring."""

    arm_id: str
    case_id: str
    cluster_id: str
    predicted_diagnosis: str
    answered: bool
    abstained: bool
    final_action: str
    contract_applicable: bool
    calibrated_confidence: Optional[float] = None
    actionable_score: Optional[float] = None
    contract_complete: Optional[bool] = None
    structured_pass_eligible: Optional[bool] = None
    structural_completion: Optional[float] = None
    citations: Tuple[AssertedCitation, ...] = ()
    generation: Mapping[str, Any] = field(default_factory=dict)
    generation_error: Optional[str] = None
    answer_text: str = ""
    context_texts: Tuple[str, ...] = ()
    alias_of: Optional[str] = None
    # P1-P8 CATIA/GRiP trajectory-trust dimensions (contribution #4); None for
    # non-contract arms (B0/baselines), never zero.
    process_trust: Optional[Mapping[str, Optional[float]]] = None


def _interval(value: Optional[Interval]) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    return {"estimate": value.estimate, "low": value.low, "high": value.high}


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


_CITATION_RE = re.compile(r"\[\d+\]")


def _reference_answer(gold_label: Optional[str], resolver: Any) -> str:
    """Constructed, label-derived reference (DISCLOSED): the gold diagnosis plus the
    guideline recommendation(s) indicated for it (empty guidance for benign labels with
    no licensed condition). Mirrors CALIBER's templated label-derived reference and is
    used for completeness / context_recall / answer_correctness. Not a leaked answer:
    it is built from the label + the frozen guideline corpus, both host-owned."""
    label = (gold_label or "").strip()
    if not label:
        return ""
    texts: List[str] = []
    search = getattr(resolver, "search", None)
    if callable(search):
        try:
            for evidence in list(search(label))[:3]:
                text = str(getattr(evidence, "text", "") or "").strip()
                if text:
                    texts.append(text)
        except Exception:  # noqa: BLE001 -- a resolver without this label yields no guidance
            texts = []
    guidance = " ".join(texts)
    return (f"Diagnosis: {label}." + (f" {guidance}" if guidance else "")).strip()


def _alignment_block(
    outcomes: Sequence[ArmCaseOutcome], correct: Sequence[bool],
) -> Dict[str, Any]:
    """Alignment metrics via the canonical metrics.py functions (MMed-RAG /
    Rethinking-RAG family):
    - over_reliance_confident_wrong: answered-but-incorrect rate (over-relying on bad
      evidence) -- metrics.over_reliance with confident = answered;
    - copy_reference_ratio: verbatim 5-gram overlap with retrieved evidence (lower = less
      parroting);
    - completeness: recall of retrieved-evidence content tokens in the answer (uses the
      retrieved PubMed/guideline chunks, not a gold reference).
    Plus the honest note that MedGuard's *diagnosis* over-reliance is 0 by construction
    (committed label is the frozen probe's; the VLM cannot override it), reported apart
    from the descriptive no-citation-answer rate (benign lesions with no guideline to
    cite -- not VLM-prior reliance)."""
    confident = [o.answered for o in outcomes]
    over_reliance_rate = over_reliance(confident, list(correct)) if outcomes else None
    answered = [o for o in outcomes if o.answered and (o.answer_text or "").strip()]
    if not answered:
        return {
            "applicable": False, "status": "not_applicable",
            "over_reliance_confident_wrong": over_reliance_rate,
        }
    chunks = lambda o: [{"text": text} for text in o.context_texts]
    copy_vals = [copy_reference_ratio(o.answer_text, chunks(o)) for o in answered]
    completeness_vals = [_completeness(o.answer_text, chunks(o)) for o in answered]
    no_citation = sum(
        1 for o in answered
        if not _CITATION_RE.search(o.answer_text or "") and not o.citations
    )
    return {
        "applicable": True,
        "n_answered": len(answered),
        "over_reliance_confident_wrong": over_reliance_rate,
        "diagnosis_over_reliance": 0.0,  # host-committed probe Dx; VLM cannot override it
        "copy_reference_ratio": round(sum(copy_vals) / len(copy_vals), 4),
        "completeness": round(sum(completeness_vals) / len(completeness_vals), 4),
        "no_citation_answer_rate": round(no_citation / len(answered), 4),
        "note": ("diagnosis_over_reliance is 0 by construction (committed label is the "
                 "frozen probe's); over_reliance_confident_wrong = answered-but-incorrect "
                 "rate; no_citation_answer_rate rows are benign/no-guideline lesions"),
    }


def _diagnosis_block(
    y_true: Sequence[str], y_pred: Sequence[str],
) -> Optional[Dict[str, Any]]:
    valid = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t in DERM_CLASS_ORDER and p in DERM_CLASS_ORDER
    ]
    if not valid:
        return None
    summary = classification_summary([t for t, _ in valid], [p for _, p in valid])
    summary["scored_cases"] = len(valid)
    summary["unscored_cases"] = len(y_true) - len(valid)
    return summary


def _actionable_block(
    pred_labels: Sequence[str], true_labels: Sequence[str],
    scores: Sequence[Optional[float]], positive_labels: Sequence[str],
) -> Dict[str, Any]:
    have_scores = all(score is not None for score in scores) and bool(scores)
    numeric_scores = [float(s) for s in scores] if have_scores else None
    confusion = actionable_confusion(
        pred_labels, true_labels, positive_labels=positive_labels,
        scores=numeric_scores,
    )
    positive = {str(label).lower() for label in positive_labels}
    actionable_true = [(t or "").lower() in positive for t in true_labels]
    at_spec = (
        sensitivity_at_fixed_specificity(
            actionable_true, numeric_scores, REGISTERED_TARGET_SPECIFICITY,
        ) if have_scores else None
    )
    confusion["sensitivity_at_fixed_specificity"] = at_spec
    confusion["operating_specificity_target"] = REGISTERED_TARGET_SPECIFICITY
    confusion["actionable_score_available"] = have_scores
    return confusion


def _selective_block(outcomes: Sequence[ArmCaseOutcome], correct: Sequence[bool]) -> Dict[str, Any]:
    conf = [o.calibrated_confidence for o in outcomes]
    have_conf = all(value is not None for value in conf) and bool(conf)
    answered = [o.answered for o in outcomes]
    n_answered = sum(answered)
    answered_correct = [c for c, a in zip(correct, answered) if a]
    block: Dict[str, Any] = {
        "coverage": round(n_answered / len(outcomes), 4) if outcomes else None,
        "answered_cases": n_answered,
        "answered_accuracy": (
            round(sum(answered_correct) / len(answered_correct), 4)
            if answered_correct else None
        ),
        "calibrated_confidence_available": have_conf,
    }
    if have_conf:
        numeric = [float(value) for value in conf]
        block["aurc"] = aurc(numeric, list(correct))
        block["risk_coverage_curve"] = risk_coverage_curve(numeric, list(correct))
        # ECE is a probe-calibration property over ALL cases (abstention-independent):
        # the top-1 confidence vs correctness (canonical metrics.ece, 10-bin).
        block["ece"] = ece(numeric, list(correct))
    else:
        block["aurc"] = None
        block["risk_coverage_curve"] = None
        block["ece"] = None
    return block


def _citation_block(
    outcomes: Sequence[ArmCaseOutcome], resolver: CitationResolver,
) -> Dict[str, Any]:
    case_scores: List[CaseCitationScore] = [
        score_case(o.predicted_diagnosis, list(o.citations), resolver)
        for o in outcomes
    ]
    return summarize_citations(case_scores)


def _process_block(outcomes: Sequence[ArmCaseOutcome]) -> Dict[str, Any]:
    applicable = [o for o in outcomes if o.contract_applicable]
    if not applicable:
        # B1 and the other non-contract arms are not_applicable, never zero.
        return {"applicable": False, "status": "not_applicable"}
    complete = [1.0 for o in applicable if o.contract_complete]
    pass_eligible = [1.0 for o in applicable if o.structured_pass_eligible]
    completions = [
        float(o.structural_completion) for o in applicable
        if o.structural_completion is not None
    ]
    return {
        "applicable": True,
        "contracts_scored": len(applicable),
        "contract_complete_rate": round(len(complete) / len(applicable), 4),
        "structured_pass_eligible_rate": round(len(pass_eligible) / len(applicable), 4),
        "mean_structural_completion": (
            round(_mean(completions), 4) if completions else None
        ),
    }


def _process_trust_block(outcomes: Sequence[ArmCaseOutcome]) -> Dict[str, Any]:
    """Aggregate the P1-P8 CATIA/GRiP trajectory-trust dimensions per arm.

    Each dimension is the mean over the outcomes that report it (non-None); the
    dimensions stay disaggregated (the composite is a secondary compact diagnostic,
    not the primary result). Non-contract arms carry no profile -> not_applicable.
    """
    profiles = [dict(o.process_trust) for o in outcomes if o.process_trust is not None]
    if not profiles:
        return {"applicable": False, "status": "not_applicable"}
    keys = sorted({k for profile in profiles for k in profile})
    dims: Dict[str, Optional[float]] = {}
    for key in keys:
        vals = [float(p[key]) for p in profiles if p.get(key) is not None]
        dims[key] = round(sum(vals) / len(vals), 4) if vals else None
    scored = [v for v in dims.values() if v is not None]
    return {
        "applicable": True,
        "profiles_scored": len(profiles),
        "dimensions": dims,
        "composite_mean": round(sum(scored) / len(scored), 4) if scored else None,
    }


def _efficiency_block(outcomes: Sequence[ArmCaseOutcome]) -> Dict[str, Any]:
    def _sum(field_name: str) -> float:
        return sum(float(o.generation.get(field_name, 0) or 0) for o in outcomes)

    return {
        "generation_calls": int(_sum("remote_calls")),
        "prompt_tokens": int(_sum("prompt_tokens")),
        "completion_tokens": int(_sum("completion_tokens")),
        "total_tokens": int(_sum("total_tokens")),
        "latency_seconds": round(_sum("latency_seconds"), 6),
        "schema_repairs": int(_sum("schema_repairs")),
        "cache_hits": sum(1 for o in outcomes if o.generation.get("cache_hit")),
        "generation_errors": sum(1 for o in outcomes if o.generation_error),
    }


def _accuracy_ci(
    outcomes: Sequence[ArmCaseOutcome], correct: Sequence[bool], *,
    n_boot: int, seed: int,
) -> Optional[Dict[str, float]]:
    scored = [
        (o, c) for o, c in zip(outcomes, correct)
        if o.predicted_diagnosis in DERM_CLASS_ORDER
    ]
    if not scored:
        return None
    rows = [1.0 if c else 0.0 for _o, c in scored]
    clusters = [o.cluster_id for o, _c in scored]
    interval = clustered_bootstrap_ci(
        rows, clusters, lambda sample: sum(sample) / len(sample),
        n_boot=n_boot, seed=seed,
    )
    return _interval(interval)


def _ragas_block(
    outcomes: Sequence[ArmCaseOutcome], truth: Mapping[str, str],
    question: str, lock: RagasJudgeLock, judge: Any = None, resolver: Any = None,
) -> Dict[str, Any]:
    # The reference is the constructed label-derived answer (gold Dx + indicated
    # guideline text) so context_recall / answer_correctness have a real target; a bare
    # label gives them nothing to recall. Faithfulness/relevancy do not use it.
    samples = [
        map_arm_case_to_sample(
            case_id=o.case_id, arm_id=o.arm_id, question=question,
            answer=o.answer_text, contexts=list(o.context_texts),
            reference=_reference_answer(truth.get(o.cluster_id, truth.get(o.case_id)), resolver),
        )
        for o in outcomes if o.answered and o.answer_text.strip()
    ]
    return RagasSupportingAdapter(lock, judge=judge).evaluate(samples)


def _truth_for(outcome: ArmCaseOutcome, truth: Mapping[str, str]) -> Optional[str]:
    # Labels may be keyed by lesion cluster (the manifest convention) or by the
    # arm-facing case id; accept either without letting the arm see it.
    if outcome.cluster_id in truth:
        return truth[outcome.cluster_id]
    return truth.get(outcome.case_id)


def score_arm(
    outcomes: Sequence[ArmCaseOutcome],
    truth: Mapping[str, str],
    *,
    citation_resolver: CitationResolver,
    question: str,
    actionable_positive_labels: Optional[Sequence[str]] = None,
    ragas_lock: Optional[RagasJudgeLock] = None,
    ragas_judge: Any = None,
    n_boot: int = REGISTERED_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = REGISTERED_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Every endpoint for one arm, over its per-lesion outcomes."""
    if not outcomes:
        raise ValueError("cannot score an arm with no outcomes")
    positive = list(actionable_positive_labels or sorted(MALIGNANT_LABELS))
    ordered = sorted(outcomes, key=lambda o: o.case_id)
    true_labels = [_truth_for(o, truth) for o in ordered]
    missing_truth = [o.case_id for o, t in zip(ordered, true_labels) if t is None]
    if missing_truth:
        raise ValueError(f"no development label joined for cases: {missing_truth[:3]}")
    pred_labels = [o.predicted_diagnosis for o in ordered]
    correct = [p == t for p, t in zip(pred_labels, true_labels)]

    return {
        "arm_id": ordered[0].arm_id,
        "alias_of": ordered[0].alias_of,
        "cases": len(ordered),
        "diagnosis": _diagnosis_block(true_labels, pred_labels),
        "actionable": _actionable_block(
            pred_labels, true_labels, [o.actionable_score for o in ordered], positive,
        ),
        "selective": _selective_block(ordered, correct),
        "provenance_citation_integrity": _citation_block(ordered, citation_resolver),
        "process_contract_completion": _process_block(ordered),
        "process_trust": _process_trust_block(ordered),
        "efficiency": _efficiency_block(ordered),
        "diagnostic_accuracy_ci": _accuracy_ci(
            ordered, correct, n_boot=n_boot, seed=bootstrap_seed,
        ),
        "alignment": _alignment_block(ordered, correct),
        "ragas_supporting": _ragas_block(
            ordered, truth, question, ragas_lock or RagasJudgeLock(), judge=ragas_judge,
            resolver=citation_resolver,
        ),
    }


def paired_accuracy_difference(
    arm_a: Sequence[ArmCaseOutcome],
    arm_b: Sequence[ArmCaseOutcome],
    truth: Mapping[str, str],
    *,
    n_boot: int = REGISTERED_BOOTSTRAP_REPLICATES,
    seed: int = REGISTERED_BOOTSTRAP_SEED,
) -> Optional[Dict[str, Any]]:
    """Lesion-clustered paired CI for arm_a - arm_b diagnostic accuracy."""
    by_case_a = {o.case_id: o for o in arm_a}
    by_case_b = {o.case_id: o for o in arm_b}
    shared = sorted(set(by_case_a) & set(by_case_b))
    rows_a: List[float] = []
    rows_b: List[float] = []
    clusters: List[str] = []
    for case_id in shared:
        oa, ob = by_case_a[case_id], by_case_b[case_id]
        truth_label = _truth_for(oa, truth)
        if truth_label is None:
            continue
        if oa.predicted_diagnosis not in DERM_CLASS_ORDER:
            continue
        if ob.predicted_diagnosis not in DERM_CLASS_ORDER:
            continue
        rows_a.append(1.0 if oa.predicted_diagnosis == truth_label else 0.0)
        rows_b.append(1.0 if ob.predicted_diagnosis == truth_label else 0.0)
        clusters.append(oa.cluster_id)
    if not rows_a:
        return None
    interval = paired_clustered_bootstrap_difference(
        rows_a, rows_b, clusters, lambda sample: sum(sample) / len(sample),
        n_boot=n_boot, seed=seed,
    )
    return {
        "arm_a": arm_a[0].arm_id,
        "arm_b": arm_b[0].arm_id,
        "metric": "diagnostic_accuracy",
        "n_paired": len(rows_a),
        **(_interval(interval) or {}),
    }


def score_h2h(
    outcomes: Sequence[ArmCaseOutcome],
    truth: Mapping[str, str],
    *,
    citation_resolver: CitationResolver,
    question: str,
    actionable_positive_labels: Optional[Sequence[str]] = None,
    ragas_lock: Optional[RagasJudgeLock] = None,
    ragas_judge: Any = None,
    paired_comparisons: Sequence[Tuple[str, str]] = (),
    n_boot: int = REGISTERED_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = REGISTERED_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Group flat outcomes by arm and score every arm, plus paired comparisons."""
    by_arm: Dict[str, List[ArmCaseOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_arm[outcome.arm_id].append(outcome)
    per_arm = {
        arm_id: score_arm(
            arm_outcomes, truth, citation_resolver=citation_resolver, question=question,
            actionable_positive_labels=actionable_positive_labels,
            ragas_lock=ragas_lock, ragas_judge=ragas_judge,
            n_boot=n_boot, bootstrap_seed=bootstrap_seed,
        )
        for arm_id, arm_outcomes in sorted(by_arm.items())
    }
    paired: List[Dict[str, Any]] = []
    for arm_a, arm_b in paired_comparisons:
        if arm_a in by_arm and arm_b in by_arm:
            result = paired_accuracy_difference(
                by_arm[arm_a], by_arm[arm_b], truth,
                n_boot=n_boot, seed=bootstrap_seed,
            )
            if result is not None:
                paired.append(result)
    return {
        "endpoints_registered": {
            "actionable_specificity_target": REGISTERED_TARGET_SPECIFICITY,
            "bootstrap_replicates": n_boot,
            "bootstrap_seed": bootstrap_seed,
            "clustered_by": "lesion_cluster_id",
            "provenance_primary": "citation_integrity",
            "ragas_role": "supporting",
        },
        "per_arm": per_arm,
        "paired_differences": paired,
    }
