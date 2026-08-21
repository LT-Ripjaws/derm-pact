"""G1 -- can a VLM actually read the seven dermoscopic criteria off an image?

G2, G3 and G5 all used **gold** criteria, so they measured a ceiling: what perfect perception
would buy.  G1 measures how much of that ceiling is reachable, because the whole v2 architecture
rests on one unproven assumption --

    a general vision-language model can judge individual dermoscopic criteria well enough that
    the host's deterministic 7-point arithmetic lands near the expert's.

If it cannot, no amount of downstream provenance machinery saves the design: the host would be
doing exact arithmetic over noise.  This is the remaining project-killer gate.

What is measured
----------------
Per criterion, against Derm7pt's expert annotation: accuracy, macro-F1, Cohen's kappa, and the
rate at which the assessor declined (``indeterminate``).  Each is reported twice --
``determinate_only`` (abstentions excluded from the denominator, the perception question) and
``strict`` (abstentions counted as errors, the usefulness question).  Then the same, macro-averaged
across the seven.  Then the thing that actually matters clinically: agreement between the 7-point
score *derived* from the assessor's criteria via :func:`score_criteria` and the gold
``seven_point_score`` -- MAE, exact-match rate, and whether the score *range* produced under
partial information still contains the truth.

Every criterion carries a **chance baseline**: predicting that criterion's most frequent value in
the *training* split, always.  Derm7pt criteria are severely imbalanced (most are ``absent`` most
of the time), so an assessor can look impressive at 70% accuracy while being strictly worse than a
constant.  "Better than chance" is a number here, not a claim.  The same constant is pushed
through the 7-point aggregation, so the score-agreement numbers have a floor too.

Cost discipline
---------------
``--dry-run`` builds every prompt, hashes every image, estimates every token, and makes **zero**
API calls.  Run it before ever running the live path; the projection it prints is the input to
the decision about whether the full split is affordable.

The official ``test`` split is refused outright.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard_v2.criteria.aggregate import score_criteria
from medguard_v2.criteria.schema import (
    CRITERIA_BY_NAME,
    CRITERION_NAMES,
    INDETERMINATE,
    MELANOMA_SUSPICION_THRESHOLD,
)
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.metrics import classification_summary
from medguard_v2.perception.criterion_assessor import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL,
    PROMPT_VERSION,
    AssessorCache,
    CriterionAssessor,
    build_repair_prompt,
    default_client,
    estimate_text_tokens,
)

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "research" / "evidence" / "derm7pt_v1"

#: Illustrative gateway rate cards, USD per 1e6 tokens, as (input, output).  None of these is
#: this project's contracted price -- the report also emits the raw token counts, so any real rate
#: card can be applied afterwards without re-running anything.
RATE_CARD_GRID = ((0.05, 0.20), (0.10, 0.40), (0.20, 0.80), (0.50, 1.50), (1.00, 3.00))

SELECTION_STRATEGIES = ("head", "even", "score-spread")


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------
def select_cases(
    cases: Sequence[Any], limit: Optional[int], strategy: str
) -> Tuple[List[Any], Dict[str, Any]]:
    """Deterministically pick ``limit`` cases, and say how -- selection is never random.

    ``head`` is the naive prefix and is almost always the wrong choice on Derm7pt: ``meta.csv`` is
    ordered by diagnosis, so the first 19 training cases are all BCC.  ``even`` spreads evenly
    across that order.  ``score-spread`` sorts by the gold 7-point score and takes the lowest,
    median and highest, which is what a *smoke* wants -- a case with nothing to find, a middling
    one, and a case where six of seven criteria fire.  It is deliberately not distribution-like,
    and the returned record says so.
    """
    ordered = sorted(cases, key=lambda case: case.case_index)
    record: Dict[str, Any] = {
        "strategy": "all" if limit is None else strategy,
        "n_available": len(ordered),
        "n_selected": len(ordered) if limit is None else min(limit, len(ordered)),
        "representative_of_split": limit is None or strategy == "even",
    }
    if limit is None or limit >= len(ordered):
        record["note"] = "whole split, no selection applied"
        return ordered, record

    if strategy == "head":
        chosen = ordered[:limit]
        record["note"] = "first n by case_index; meta.csv is diagnosis-ordered, so this is skewed"
    elif strategy == "even":
        n = len(ordered)
        if limit == 1:
            positions = [0]
        else:
            positions = [round(i * (n - 1) / (limit - 1)) for i in range(limit)]
        chosen = [ordered[p] for p in sorted(set(positions))]
        record["note"] = "evenly spaced across case_index order"
    elif strategy == "score-spread":
        by_score = sorted(ordered, key=lambda case: (int(case.seven_point_score), case.case_index))
        n = len(by_score)
        positions = (
            [0] if limit == 1 else [round(i * (n - 1) / (limit - 1)) for i in range(limit)]
        )
        chosen = [by_score[p] for p in sorted(set(positions))]
        record["note"] = (
            "spread across the gold 7-point score (lowest .. highest). NOT a random sample: "
            "difficulty is deliberately maximised, so accuracy here does not estimate the split."
        )
    else:
        raise ValueError(f"unknown selection strategy {strategy!r}")

    record["n_selected"] = len(chosen)
    record["case_nums"] = [case.case_num for case in chosen]
    record["diagnosis_groups"] = [case.diagnosis_group for case in chosen]
    record["gold_scores"] = [int(case.seven_point_score) for case in chosen]
    return chosen, record


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def majority_by_criterion(cases: Sequence[Any]) -> Dict[str, str]:
    """Most frequent gold value per criterion. The constant an assessor must beat.

    Ties are broken by value name so the baseline is reproducible.
    """
    out: Dict[str, str] = {}
    for name in CRITERION_NAMES:
        counts = Counter(str(case.criteria[name]) for case in cases)
        best = max(counts.items(), key=lambda item: (item[1], item[0] == "absent", item[0]))
        out[name] = best[0]
    return out


def _macro_f1_over_supported(summary: Mapping[str, Any]) -> float:
    """Macro-F1 restricted to classes that actually occur in the gold labels.

    Full-vocabulary macro-F1 is the number that is comparable across splits, but on a handful of
    cases it is dominated by classes with zero support (each contributing f1=0), which says
    nothing about the assessor. Both are reported.
    """
    per_class = summary["per_class"]
    supported = [entry["f1"] for entry in per_class.values() if entry["support"] > 0]
    return sum(supported) / len(supported) if supported else 0.0


def _summary_block(summary: Mapping[str, Any], *, with_matrix: bool = False) -> Dict[str, Any]:
    block = {
        "accuracy": round(summary["accuracy"], 4),
        "balanced_accuracy": round(summary["balanced_accuracy"], 4),
        "macro_f1_full_vocabulary": round(summary["macro_f1"], 4),
        "macro_f1_supported_classes": round(_macro_f1_over_supported(summary), 4),
        "cohen_kappa": round(summary["cohen_kappa"], 4),
    }
    if with_matrix:
        block["confusion_matrix"] = summary["confusion_matrix"]
        block["labels"] = summary["labels"]
    return block


# ---------------------------------------------------------------------------
# Per-criterion scoring
# ---------------------------------------------------------------------------
def _criterion_block(
    name: str, gold: Sequence[str], pred: Sequence[str],
    train_majority: str, eval_majority: str,
) -> Dict[str, Any]:
    vocabulary = sorted(CRITERIA_BY_NAME[name].vocabulary)
    n = len(gold)

    determinate = [(g, p) for g, p in zip(gold, pred) if p != INDETERMINATE]
    n_indeterminate = n - len(determinate)

    block: Dict[str, Any] = {
        "n_assessed": n,
        "n_indeterminate": n_indeterminate,
        "indeterminate_rate": round(n_indeterminate / n, 4) if n else None,
        "n_determinate": len(determinate),
    }

    if determinate:
        block["determinate_only"] = _summary_block(
            classification_summary([g for g, _ in determinate],
                                   [p for _, p in determinate], vocabulary),
            with_matrix=True,
        )
    else:
        block["determinate_only"] = None

    # Strict view: declining counts as being wrong. Gold never contains INDETERMINATE, so that
    # label has zero support and its f1 is zero by construction -- which is the point.
    strict = classification_summary(list(gold), list(pred), vocabulary + [INDETERMINATE])
    block["strict_indeterminate_counts_as_wrong"] = _summary_block(strict)

    # The constant baseline, scored exactly like the assessor so the numbers are comparable.
    constant = classification_summary(list(gold), [train_majority] * n, vocabulary)
    eval_hits = sum(1 for g in gold if g == eval_majority)
    gold_counts = Counter(gold)
    block["gold_value_counts"] = dict(sorted(gold_counts.items()))
    block["gold_is_constant"] = len(gold_counts) <= 1
    block["chance_baseline"] = {
        "train_majority_value": train_majority,
        "train_majority": _summary_block(constant),
        "eval_majority_value": eval_majority,
        "eval_majority_accuracy_oracle": round(eval_hits / n, 4) if n else None,
        "note": (
            "train_majority is the legitimate prior (fit on train, applied to the eval cohort). "
            "Its Cohen kappa is 0 whenever the gold labels vary; when the gold column is itself "
            "constant (gold_is_constant) the constant is trivially perfect and metrics.py reports "
            "kappa 1.0 -- read the lift on that criterion as uninformative, not as a loss. "
            "eval_majority is an oracle constant shown only to bound how easy the cohort is."
        ),
    }
    block["lift_over_train_majority"] = {
        "strict_accuracy": round(strict["accuracy"] - constant["accuracy"], 4),
        "determinate_accuracy": (
            round(block["determinate_only"]["accuracy"] - constant["accuracy"], 4)
            if block["determinate_only"] else None
        ),
    }
    return block


def _score_rows(gold_scores: Sequence[int], value_maps: Sequence[Mapping[str, str]]
                ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gold_score, values in zip(gold_scores, value_maps):
        scored = score_criteria(values)
        rows.append({
            "gold_score": int(gold_score),
            "pred_score_min": scored.score_min,
            "pred_score_max": scored.score_max,
            "indeterminate": list(scored.indeterminate),
            "gold_score_in_range": scored.score_min <= int(gold_score) <= scored.score_max,
            "decidable": scored.decidable,
            "pred_meets_threshold": scored.decidable and scored.meets_threshold,
            "gold_meets_threshold": int(gold_score) >= MELANOMA_SUSPICION_THRESHOLD,
        })
    return rows


def _score_agreement(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """7-point score derived from criterion judgements vs the gold score."""
    full = [row for row in rows if not row["indeterminate"]]
    out: Dict[str, Any] = {
        "n_cases": len(rows),
        "n_fully_determinate": len(full),
        "fully_determinate_rate": round(len(full) / len(rows), 4) if rows else None,
    }
    if full:
        errors = [abs(row["pred_score_min"] - row["gold_score"]) for row in full]
        out["fully_determinate"] = {
            "mae": round(sum(errors) / len(errors), 4),
            "exact_match_rate": round(sum(1 for e in errors if e == 0) / len(errors), 4),
            "within_one_rate": round(sum(1 for e in errors if e <= 1) / len(errors), 4),
            "mean_signed_error": round(
                sum(row["pred_score_min"] - row["gold_score"] for row in full) / len(full), 4
            ),
            "max_abs_error": max(errors),
        }
    else:
        out["fully_determinate"] = None

    contains = [row["gold_score_in_range"] for row in rows]
    widths = [row["pred_score_max"] - row["pred_score_min"] for row in rows]
    out["with_abstention"] = {
        "gold_score_within_predicted_range_rate": (
            round(sum(contains) / len(contains), 4) if contains else None
        ),
        "mean_range_width": round(statistics.fmean(widths), 4) if widths else None,
        "decidable_rate": (
            round(sum(1 for row in rows if row["decidable"]) / len(rows), 4) if rows else None
        ),
    }

    decidable = [row for row in rows if row["decidable"]]
    if decidable:
        tp = sum(1 for r in decidable if r["pred_meets_threshold"] and r["gold_meets_threshold"])
        fn = sum(
            1 for r in decidable if not r["pred_meets_threshold"] and r["gold_meets_threshold"]
        )
        fp = sum(
            1 for r in decidable if r["pred_meets_threshold"] and not r["gold_meets_threshold"]
        )
        tn = sum(
            1 for r in decidable
            if not r["pred_meets_threshold"] and not r["gold_meets_threshold"]
        )
        out["threshold_verdict_on_decidable"] = {
            "threshold": MELANOMA_SUSPICION_THRESHOLD,
            "n": len(decidable),
            "accuracy": round((tp + tn) / len(decidable), 4),
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "note": "vs the gold 7-point score's own threshold verdict, not vs the diagnosis",
        }
    else:
        out["threshold_verdict_on_decidable"] = None
    return out


# ---------------------------------------------------------------------------
# Dry run: build every prompt, cost nothing
# ---------------------------------------------------------------------------
def _typical_completion_tokens() -> int:
    """Tokenise a realistic, verbose-but-legal response instead of guessing at output length."""
    example = {
        name: {
            # sorted() first so the choice is stable across processes (frozenset order is not).
            "value": max(sorted(CRITERIA_BY_NAME[name].vocabulary), key=len),
            "confidence": 0.75,
            "justification": (
                "Irregular thickened lines with unevenly sized meshes across the upper left "
                "quadrant of the lesion"
            ),
        }
        for name in CRITERION_NAMES
    }
    return estimate_text_tokens(json.dumps(example, indent=2))


def run_dry(
    cases: Sequence[Any], assessor: CriterionAssessor, *,
    price_in: Optional[float] = None, price_out: Optional[float] = None,
    selection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    requests = [assessor.build_request(case) for case in cases]
    estimates = [assessor.estimate_request_tokens(request) for request in requests]

    text_tokens = [e["text_tokens_estimated"] for e in estimates]
    image_tokens = [e["image_tokens_estimated"] for e in estimates]
    prompt_tokens = [e["prompt_tokens_estimated"] for e in estimates]

    typical_out = _typical_completion_tokens()
    ceiling_out = assessor.decoding.max_tokens
    mean_prompt = statistics.fmean(prompt_tokens) if prompt_tokens else 0.0

    cached = sum(
        1 for request in requests if assessor.cache.entry_path(request.cache_key()).exists()
    )
    uncached = len(requests) - cached

    repair_prompt_tokens = estimate_text_tokens(
        build_repair_prompt(previous="x" * 400, error="pigment_network: unknown value 'reticular'")
    ) + int(statistics.fmean(image_tokens) if image_tokens else 0)

    def _totals(n_cases: int, repair_rate: float) -> Dict[str, Any]:
        in_tokens = n_cases * mean_prompt + n_cases * repair_rate * repair_prompt_tokens
        out_tokens = n_cases * typical_out * (1.0 + repair_rate)
        block: Dict[str, Any] = {
            "cases": n_cases,
            "repair_rate_assumed": repair_rate,
            "input_tokens": int(round(in_tokens)),
            "output_tokens_typical": int(round(out_tokens)),
            "output_tokens_ceiling": int(round(n_cases * ceiling_out * (1.0 + repair_rate))),
            "input_millions": round(in_tokens / 1e6, 6),
            "output_millions": round(out_tokens / 1e6, 6),
            "cost_grid_usd": {
                f"in=${pi}/M,out=${po}/M": round(
                    in_tokens / 1e6 * pi + out_tokens / 1e6 * po, 4
                )
                for pi, po in RATE_CARD_GRID
            },
        }
        if price_in is not None and price_out is not None:
            block["cost_usd_at_supplied_rate"] = round(
                in_tokens / 1e6 * price_in + out_tokens / 1e6 * price_out, 4
            )
        return block

    return {
        "gate_id": "G1_criterion_readability",
        "mode": "dry_run",
        "boundaries": {
            "api_calls": 0,
            "paid_calls": 0,
            "images_read_for_hashing_only": sum(len(r.image_paths) for r in requests),
            "test_split_read": False,
        },
        "cohort": {
            "n_cases": len(cases),
            "splits": sorted({case.split for case in cases}),
            "selection": dict(selection or {}),
        },
        "assessor": {
            "prompt_version": PROMPT_VERSION,
            "decoding": assessor.decoding.to_dict(),
            "max_repair_attempts": assessor.max_repair_attempts,
            "include_clinical": assessor.include_clinical,
            "images_per_case": sorted(Counter(len(r.image_paths) for r in requests).items()),
            "cache_dir": str(assessor.cache.root),
        },
        "token_estimate_method": {
            "text": "tiktoken cl100k_base (proxy; the served model uses a Qwen VL tokeniser)",
            "image": "Qwen-VL smart_resize grid, 28x28 px per visual token, +2 delimiters",
            "extrapolation_basis": (
                f"per-case means computed over the {len(cases)} selected case(s); rows for other "
                "cohort sizes reuse those means"
            ),
            "warning": (
                "ESTIMATE ONLY. The live run reports observed prompt tokens and a calibration "
                "ratio; multiply this projection by that ratio before trusting it."
            ),
        },
        "per_case_tokens": {
            "prompt_text_tokens": text_tokens[0] if text_tokens else None,
            "image_tokens_mean": (
                round(statistics.fmean(image_tokens), 2) if image_tokens else None
            ),
            "image_tokens_min": min(image_tokens) if image_tokens else None,
            "image_tokens_max": max(image_tokens) if image_tokens else None,
            "prompt_tokens_mean": round(mean_prompt, 2) if prompt_tokens else None,
            "completion_tokens_typical": typical_out,
            "completion_tokens_ceiling": ceiling_out,
            "repair_call_prompt_tokens": repair_prompt_tokens,
        },
        "cache": {
            "entries_on_disk": assessor.cache.count(),
            "requests_already_cached": cached,
            "requests_needing_a_paid_call": uncached,
        },
        "projection": {
            "this_cohort_no_repairs": _totals(len(requests), 0.0),
            "this_cohort_10pct_repairs": _totals(len(requests), 0.10),
            "incremental_uncached_only": _totals(uncached, 0.10),
            "valid_split_203_cases": _totals(203, 0.10),
            "train_split_413_cases": _totals(413, 0.10),
            "full_dataset_1011_cases": _totals(1011, 0.10),
        },
    }


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------
def _usage_block(assessments: Sequence[Any], assessor: CriterionAssessor,
                 cases: Sequence[Any]) -> Dict[str, Any]:
    """Token/latency accounting, split by whether the call was actually paid for.

    A cached assessment replays the *original* call's token counts. Summing those together with
    fresh calls would make a fully-cached re-run report the same spend as the paid run, which is
    exactly the number a cost review must not see.
    """
    paid = [a for a in assessments if not a.stats.cache_hit]
    cached = [a for a in assessments if a.stats.cache_hit]

    def _sum(group: Sequence[Any]) -> Dict[str, Any]:
        return {
            "cases": len(group),
            "remote_calls": sum(a.stats.remote_calls for a in group),
            "transport_attempts": sum(a.stats.transport_attempts for a in group),
            "repair_attempts": sum(a.stats.repair_attempts for a in group),
            "prompt_tokens": sum(a.stats.prompt_tokens for a in group),
            "completion_tokens": sum(a.stats.completion_tokens for a in group),
            "total_tokens": sum(a.stats.total_tokens for a in group),
            "latency_seconds": round(sum(a.stats.latency_seconds for a in group), 3),
        }

    usage: Dict[str, Any] = {
        "cases": len(assessments),
        "paid_this_run": _sum(paid),
        "replayed_from_cache": _sum(cached),
        "cache_hit_rate": (
            round(len(cached) / len(assessments), 4) if assessments else None
        ),
        "usage_reported_by_gateway": all(a.stats.usage_available for a in paid) if paid else None,
    }

    billed = [a for a in paid if a.stats.remote_calls]
    if billed:
        calls = sum(a.stats.remote_calls for a in billed)
        observed = sum(a.stats.prompt_tokens for a in billed) / calls
        estimated = statistics.fmean(
            assessor.estimate_request_tokens(assessor.build_request(case))[
                "prompt_tokens_estimated"
            ]
            for case, assessment in zip(cases, assessments)
            if not assessment.stats.cache_hit and assessment.stats.remote_calls
        )
        usage["calibration"] = {
            "observed_prompt_tokens_per_call": round(observed, 1),
            "observed_completion_tokens_per_call": round(
                sum(a.stats.completion_tokens for a in billed) / calls, 1
            ),
            "estimated_prompt_tokens_per_call": round(estimated, 1),
            "ratio_observed_over_estimated": round(observed / estimated, 4) if estimated else None,
            "note": "multiply any dry-run input-token projection by this ratio",
        }
    return usage


def run_live(cases: Sequence[Any], assessor: CriterionAssessor, train_cases: Sequence[Any],
             *, selection: Optional[Mapping[str, Any]] = None, verbose: bool = False
             ) -> Dict[str, Any]:
    def _progress(index: int, total: int, assessment: Any) -> None:
        if verbose:
            mark = "cache" if assessment.stats.cache_hit else "paid "
            print(f"  [{index}/{total}] case {assessment.case_num}: {assessment.status} ({mark})",
                  file=sys.stderr, flush=True)

    assessments = assessor.assess_cases(cases, progress=_progress)

    scored_pairs = [(c, a) for c, a in zip(cases, assessments) if a.ok]
    parse_failures = [(c, a) for c, a in zip(cases, assessments) if a.status == "parse_failure"]
    api_failures = [(c, a) for c, a in zip(cases, assessments) if a.status == "api_failure"]

    train_majority = majority_by_criterion(train_cases)
    eval_majority = majority_by_criterion([c for c, _ in scored_pairs]) if scored_pairs else {}

    criteria_report: Dict[str, Any] = {}
    macro: Dict[str, List[float]] = {
        "accuracy": [], "macro_f1": [], "cohen_kappa": [],
        "strict_accuracy": [], "chance_accuracy": [], "indeterminate_rate": [],
    }
    for name in CRITERION_NAMES:
        if not scored_pairs:
            break
        gold = [str(case.criteria[name]) for case, _ in scored_pairs]
        pred = [assessment.values[name] for _, assessment in scored_pairs]
        block = _criterion_block(name, gold, pred, train_majority[name], eval_majority[name])
        criteria_report[name] = block
        if block["determinate_only"]:
            macro["accuracy"].append(block["determinate_only"]["accuracy"])
            macro["macro_f1"].append(block["determinate_only"]["macro_f1_supported_classes"])
            macro["cohen_kappa"].append(block["determinate_only"]["cohen_kappa"])
        macro["strict_accuracy"].append(block["strict_indeterminate_counts_as_wrong"]["accuracy"])
        macro["chance_accuracy"].append(block["chance_baseline"]["train_majority"]["accuracy"])
        macro["indeterminate_rate"].append(block["indeterminate_rate"])

    def _mean(key: str) -> Optional[float]:
        values = macro[key]
        return round(statistics.fmean(values), 4) if values else None

    rows = _score_rows(
        [case.seven_point_score for case, _ in scored_pairs],
        [assessment.values for _, assessment in scored_pairs],
    )
    chance_rows = _score_rows(
        [case.seven_point_score for case, _ in scored_pairs],
        [dict(train_majority) for _ in scored_pairs],
    )

    per_case: List[Dict[str, Any]] = []
    for (case, assessment), row in zip(scored_pairs, rows):
        per_case.append({
            "case_index": case.case_index,
            "case_num": case.case_num,
            "split": case.split,
            "diagnosis": case.diagnosis,
            "diagnosis_group": case.diagnosis_group,
            "status": assessment.status,
            "criteria": {
                name: {
                    "gold": str(case.criteria[name]),
                    "predicted": assessment.values[name],
                    "agree": str(case.criteria[name]) == assessment.values[name],
                    "confidence": assessment.confidences.get(name),
                    "justification": assessment.justifications.get(name, ""),
                }
                for name in CRITERION_NAMES
            },
            "n_criteria_agreeing": sum(
                1 for name in CRITERION_NAMES
                if str(case.criteria[name]) == assessment.values[name]
            ),
            "gold_seven_point_score": int(case.seven_point_score),
            "predicted_score_range": [row["pred_score_min"], row["pred_score_max"]],
            "cache_hit": assessment.stats.cache_hit,
            "repair_attempts": assessment.stats.repair_attempts,
            "soft_defects": assessment.soft_defects,
            "normalisations": assessment.normalisations,
        })
    for case, assessment in parse_failures + api_failures:
        per_case.append({
            "case_index": case.case_index,
            "case_num": case.case_num,
            "split": case.split,
            "diagnosis": case.diagnosis,
            "diagnosis_group": case.diagnosis_group,
            "status": assessment.status,
            "criteria": None,
            "failure_reason": assessment.failure_reason,
            "gold_seven_point_score": int(case.seven_point_score),
            "cache_hit": assessment.stats.cache_hit,
            "repair_attempts": assessment.stats.repair_attempts,
        })
    per_case.sort(key=lambda entry: entry["case_index"])

    confident_correct: List[float] = []
    confident_wrong: List[float] = []
    for case, assessment in scored_pairs:
        for name in CRITERION_NAMES:
            confidence = assessment.confidences.get(name)
            if confidence is None or assessment.values[name] == INDETERMINATE:
                continue
            target = (confident_correct
                      if str(case.criteria[name]) == assessment.values[name]
                      else confident_wrong)
            target.append(confidence)

    return {
        "gate_id": "G1_criterion_readability",
        "mode": "live",
        "question": "Can a VLM read the seven dermoscopic criteria off the image?",
        "boundaries": {
            "splits": sorted({case.split for case in cases}),
            "test_split_read": any(case.split == "test" for case in cases),
            "cache_dir": str(assessor.cache.root),
        },
        "cohort": {
            "n_cases": len(cases),
            "selection": dict(selection or {}),
            "class_counts": dict(Counter(case.diagnosis_group for case in cases)),
        },
        "assessor": {
            "prompt_version": PROMPT_VERSION,
            "decoding": assessor.decoding.to_dict(),
            "max_repair_attempts": assessor.max_repair_attempts,
            "include_clinical": assessor.include_clinical,
        },
        "usage": _usage_block(assessments, assessor, cases),
        "reliability": {
            "n_cases": len(assessments),
            "n_scored": len(scored_pairs),
            "parse_failures": len(parse_failures),
            "parse_failure_rate": (
                round(len(parse_failures) / len(assessments), 4) if assessments else None
            ),
            "api_failures": len(api_failures),
            "api_failure_rate": (
                round(len(api_failures) / len(assessments), 4) if assessments else None
            ),
            "failed_case_nums": [c.case_num for c, _ in parse_failures + api_failures],
            "failure_reasons": [a.failure_reason for _, a in parse_failures + api_failures][:20],
            "note": (
                "Parse failures carry NO criterion values. They are excluded from the accuracy "
                "denominators and reported here instead; they are never coerced to 'absent', "
                "which would fabricate a 7-point score of 0."
            ),
        },
        "per_criterion": criteria_report,
        "macro_across_seven": {
            "accuracy_determinate": _mean("accuracy"),
            "macro_f1_supported_classes": _mean("macro_f1"),
            "cohen_kappa": _mean("cohen_kappa"),
            "accuracy_strict": _mean("strict_accuracy"),
            "chance_accuracy_train_majority": _mean("chance_accuracy"),
            "lift_over_chance_strict": (
                round(_mean("strict_accuracy") - _mean("chance_accuracy"), 4)
                if macro["strict_accuracy"] and macro["chance_accuracy"] else None
            ),
            "indeterminate_rate": _mean("indeterminate_rate"),
        },
        "seven_point_score_agreement": _score_agreement(rows) if rows else None,
        "seven_point_score_agreement_chance_baseline": (
            _score_agreement(chance_rows) if chance_rows else None
        ),
        "confidence": {
            "mean_when_correct": (
                round(statistics.fmean(confident_correct), 4) if confident_correct else None
            ),
            "mean_when_wrong": (
                round(statistics.fmean(confident_wrong), 4) if confident_wrong else None
            ),
            "separation": (
                round(statistics.fmean(confident_correct) - statistics.fmean(confident_wrong), 4)
                if confident_correct and confident_wrong else None
            ),
        },
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------
def summarise(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    if report["mode"] == "dry_run":
        per_case = report["per_case_tokens"]
        lines.append(f"G1 DRY RUN -- {report['cohort']['n_cases']} case(s), 0 API calls")
        lines.append(
            f"  per case: ~{per_case['prompt_tokens_mean']} input tokens "
            f"({per_case['prompt_text_tokens']} text + ~{per_case['image_tokens_mean']} image), "
            f"~{per_case['completion_tokens_typical']} output "
            f"(ceiling {per_case['completion_tokens_ceiling']})"
        )
        lines.append(
            f"  cache: {report['cache']['requests_already_cached']} hit / "
            f"{report['cache']['requests_needing_a_paid_call']} would be paid"
        )
        for label in ("this_cohort_10pct_repairs", "valid_split_203_cases",
                      "full_dataset_1011_cases"):
            block = report["projection"][label]
            grid = block["cost_grid_usd"]
            lines.append(
                f"  {label:<28} {block['input_tokens']:>9} in / "
                f"{block['output_tokens_typical']:>7} out  -> "
                + "  ".join(f"{k} ${v}" for k, v in list(grid.items())[:3])
            )
        return "\n".join(lines)

    reliability = report["reliability"]
    macro = report["macro_across_seven"]
    lines.append(
        f"G1 LIVE -- {reliability['n_cases']} case(s), {reliability['n_scored']} scored, "
        f"{reliability['parse_failures']} parse failure(s), "
        f"{reliability['api_failures']} api failure(s)"
    )
    paid = report["usage"]["paid_this_run"]
    lines.append(
        f"  paid: {paid['cases']} case(s), {paid['remote_calls']} call(s), "
        f"{paid['prompt_tokens']} in / {paid['completion_tokens']} out tokens, "
        f"{paid['latency_seconds']}s"
    )
    calibration = report["usage"].get("calibration")
    if calibration:
        lines.append(
            f"  token estimate calibration: observed/estimated = "
            f"{calibration['ratio_observed_over_estimated']} "
            f"({calibration['observed_prompt_tokens_per_call']} observed vs "
            f"{calibration['estimated_prompt_tokens_per_call']} estimated)"
        )
    lines.append(
        f"  macro over 7: acc(determinate) {macro['accuracy_determinate']} | "
        f"acc(strict) {macro['accuracy_strict']} | chance {macro['chance_accuracy_train_majority']}"
        f" | lift {macro['lift_over_chance_strict']} | kappa {macro['cohen_kappa']} | "
        f"indeterminate {macro['indeterminate_rate']}"
    )
    lines.append(f"  {'criterion':<24} {'acc':>6} {'chance':>7} {'lift':>7} {'kappa':>7} {'ind':>6}")
    for name in CRITERION_NAMES:
        block = report["per_criterion"].get(name)
        if not block:
            continue
        strict = block["strict_indeterminate_counts_as_wrong"]
        chance = block["chance_baseline"]["train_majority"]
        determinate = block["determinate_only"] or {}
        lines.append(
            f"  {name:<24} {strict['accuracy']:>6} {chance['accuracy']:>7} "
            f"{block['lift_over_train_majority']['strict_accuracy']:>7} "
            f"{determinate.get('cohen_kappa', float('nan')):>7} "
            f"{block['indeterminate_rate']:>6}"
        )
    agreement = report.get("seven_point_score_agreement")
    baseline = report.get("seven_point_score_agreement_chance_baseline")
    if agreement and agreement.get("fully_determinate"):
        full = agreement["fully_determinate"]
        line = (f"  7-point score: MAE {full['mae']}, exact {full['exact_match_rate']}, "
                f"within-1 {full['within_one_rate']} (n={agreement['n_fully_determinate']})")
        if baseline and baseline.get("fully_determinate"):
            line += f"  [constant baseline MAE {baseline['fully_determinate']['mae']}]"
        lines.append(line)
    verdict = (agreement or {}).get("threshold_verdict_on_decidable")
    if verdict:
        lines.append(
            f"  melanoma-suspicion verdict (>= {verdict['threshold']}): "
            f"acc {verdict['accuracy']}, sens {verdict['sensitivity']}, "
            f"spec {verdict['specificity']}, n={verdict['n']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load(split: str, limit: Optional[int], strategy: str,
          case_nums: Optional[Sequence[int]]) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    if split == "test":
        raise SystemExit(
            "refusing to read the official test split: it is the confirmatory partition"
        )
    cases = load_cases()
    train = [case for case in cases if case.split == "train"]
    pool = [case for case in cases if case.split == split]

    if case_nums:
        wanted = list(dict.fromkeys(int(value) for value in case_nums))
        by_num = {case.case_num: case for case in pool}
        unknown = [value for value in wanted if value not in by_num]
        if unknown:
            raise SystemExit(f"case_num(s) not in the {split} split: {unknown}")
        chosen = [by_num[value] for value in wanted]
        record = {
            "strategy": "explicit-case-nums",
            "n_available": len(pool),
            "n_selected": len(chosen),
            "representative_of_split": False,
            "case_nums": wanted,
            "diagnosis_groups": [case.diagnosis_group for case in chosen],
            "gold_scores": [int(case.seven_point_score) for case in chosen],
            "note": "hand-specified cases; not a sample of the split",
        }
        return chosen, train, record

    chosen, record = select_cases(pool, limit, strategy)
    record["split"] = split
    return chosen, train, record


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="valid", choices=("train", "valid"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--select", default="even", choices=SELECTION_STRATEGIES,
                        help="how --limit picks its cases (default: even, never random)")
    parser.add_argument("--case-nums", default=None,
                        help="comma-separated case_num list; overrides --limit/--select")
    parser.add_argument("--dry-run", action="store_true",
                        help="build every prompt and project cost; makes zero API calls")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument("--include-clinical", action="store_true")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--cache-only", action="store_true",
                        help="fail rather than issue a paid call on a cache miss")
    parser.add_argument("--refresh", action="store_true", help="ignore cached answers and re-pay")
    parser.add_argument("--price-in", type=float, default=None, help="USD per 1e6 input tokens")
    parser.add_argument("--price-out", type=float, default=None, help="USD per 1e6 output tokens")
    parser.add_argument("--out", default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--print-json", action="store_true",
                        help="dump the whole report to stdout as well as the summary")
    parser.add_argument("--quiet", action="store_true", help="no per-case progress on stderr")
    args = parser.parse_args(argv)

    case_nums = (
        [int(part) for part in args.case_nums.split(",") if part.strip()]
        if args.case_nums else None
    )
    cases, train, selection = _load(args.split, args.limit, args.select, case_nums)

    assessor = CriterionAssessor(
        client=None,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        include_clinical=args.include_clinical,
        cache=AssessorCache(args.cache_dir),
        allow_remote=not args.cache_only,
        refresh=args.refresh,
        max_repair_attempts=args.max_repair_attempts,
    )

    if args.dry_run:
        report = run_dry(cases, assessor, price_in=args.price_in, price_out=args.price_out,
                         selection=selection)
        default_name = f"g1_dry_run_{args.split}_{len(cases)}.json"
    else:
        from medguard import config as v1_config  # lazy: only the live path needs credentials

        v1_config.load_env()
        if not v1_config.has_api_key():
            raise SystemExit(
                "no gateway credential in the environment; the live path cannot run. "
                "Use --dry-run, or set MEDGUARD_API_KEY / KILOCODE_API_KEY."
            )
        assessor.client = default_client(model=args.model)
        report = run_live(cases, assessor, train, selection=selection, verbose=not args.quiet)
        default_name = f"g1_live_{args.split}_{len(cases)}.json"

    text = json.dumps(report, indent=2)
    if args.write or args.out:
        out = Path(args.out) if args.out else OUT_DIR / default_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    print(summarise(report))
    if args.print_json:
        print(text)


if __name__ == "__main__":
    main()
