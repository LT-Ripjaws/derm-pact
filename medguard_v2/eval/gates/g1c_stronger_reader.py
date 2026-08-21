"""G1c -- was G1 measuring the *criteria*, or one weak reader?

G1 returned a verdict with a very wide blast radius: "a general VLM cannot read the seven
dermoscopic criteria", macro strict accuracy 0.5672 against a 0.6256 train-majority chance, lift
**-0.0584**, with ``pigmentation``, ``dots_and_globules`` and ``regression_structures`` all below a
constant predictor.  That conclusion was drawn from exactly **one** configuration: one mid-sized
open-weights model, zero-shot, text definitions only, a single dermoscopic image, one pass.  The
claim generalises over a space that was never sampled.

G1c samples it.  Three factors, varied so each one's contribution is separable:

* **F1, the reader.**  G1's ``qwen/qwen3-vl-32b-instruct`` against the strongest vision models the
  gateway actually serves.  ``--list-models`` reports what is on offer; the ladder below fixes what
  was run.
* **F2, visual exemplars.**  One captioned contact sheet per criterion, one TRAIN exemplar per
  permitted value, chosen as the medoid in the DermLIP feature space G1b already cached.  See
  :mod:`medguard_v2.perception.fewshot` for the design and its exact per-case token cost.
* **F3, the clinical view.**  Dermoscopic only (G1's default) against dermoscopic + clinical macro
  photograph, with the five valid-split cases whose "clinical" image is a copy of the dermoscopic
  file handled explicitly rather than sent twice.

What makes the comparison mean anything
---------------------------------------
**Every configuration runs on the same cases.**  One stratified subset of the valid split, drawn
once, deterministically, by :func:`select_subset`, and recorded by ``case_num`` in the report.
Comparing readers across different cohorts would measure the cohort.

**Every configuration is scored by G1's own code.**  This module computes no metric of its own: it
calls :func:`medguard_v2.eval.gates.g1_criterion_readability.run_live` per configuration, so
accuracy, the train-majority chance baseline, kappa, the derived 7-point score, the indeterminate
rate and the confidence separation are produced by the same functions that produced G1's headline.
A metric that differed by a rounding convention would be indistinguishable from an effect.

**The zero-shot baseline arm is G1's own answers.**  Its request identity is byte-identical, and
the assessor cache is shared, so that arm replays from disk for zero dollars and cannot drift.

**Nothing reads test, and no exemplar comes from valid.**  ``--split`` accepts only ``valid``;
:func:`~medguard_v2.perception.fewshot.select_exemplars` raises on any non-train case.

Cost discipline
---------------
``--dry-run`` prices every configuration in the ladder from the gateway's published rate card and
makes zero calls.  ``--budget-usd`` refuses to start a configuration whose projection would push
the run over the cap.  Paid and cached tokens are reported separately, per configuration, so a
re-run reports what it actually spent rather than what the first run did.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.data.derm7pt import analysis_cohort, load_cases
from medguard_v2.eval.gates import g1_criterion_readability as g1
from medguard_v2.perception.criterion_assessor import DEFAULT_CACHE_DIR, AssessorCache
from medguard_v2.perception.fewshot import (
    EXEMPLAR_SELECTION_VERSION,
    FEWSHOT_DESIGN_VERSION,
    ExemplarSheet,
    FewShotAssessor,
    build_contact_sheets,
    default_client_factory,
    exemplar_gaps,
    fewshot_image_token_cost,
    profile_for,
    select_exemplars,
)

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
FEATURES = EVIDENCE / "features" / "dermlip" / "features.npz"
SHEET_DIR = EVIDENCE / "g1c_exemplar_sheets"
OUT = EVIDENCE / "g1c_stronger_reader.json"

BASELINE_MODEL = "qwen/qwen3-vl-32b-instruct"
SUBSET_SEED = 20260810
SUBSET_N = 72

#: G1's headline, reproduced here so every row in the output table has its comparator attached.
G1_HEADLINE = {
    "config": "G1 published (full valid split, n=203)",
    "model": BASELINE_MODEL,
    "few_shot": False,
    "clinical": False,
    "n_cases": 203,
    "macro_accuracy_strict": 0.5672,
    "macro_chance_train_majority": 0.6256,
    "macro_lift_over_chance": -0.0584,
    "macro_cohen_kappa": 0.2092,
    "score_mae": 1.7143,
    "indeterminate_rate": 0.0098,
    "confidence_separation": 0.0281,
}


# ---------------------------------------------------------------------------
# The one cohort every configuration is measured on
# ---------------------------------------------------------------------------
def select_subset(
    valid_cases: Sequence[Any], n: int = SUBSET_N, seed: int = SUBSET_SEED
) -> Tuple[List[Any], Dict[str, Any]]:
    """A stratified sample of the valid split, drawn once and reused by every configuration.

    Stratified on ``diagnosis_group`` x ``level_of_diagnostic_difficulty`` jointly.  Diagnosis
    drives which criteria are present at all -- a nevus-heavy draw would make ``absent`` even more
    dominant than it already is and inflate the chance baseline the gate is scored against -- and
    difficulty is the dataset's own statement about how legible a case is, which is the property
    under test.

    Allocation across cells is by largest remainder, so the subset's cell proportions match the
    split's as closely as integers permit; the ordering *within* a cell is a seeded shuffle, so no
    cell is systematically represented by its lowest ``case_num``.  Both halves are deterministic:
    the same seed always yields the same ``case_num`` list, which is recorded in the report.
    """
    cells: "OrderedDict[Tuple[str, str], List[Any]]" = OrderedDict()
    for case in sorted(valid_cases, key=lambda c: c.case_num):
        cells.setdefault((case.diagnosis_group, case.difficulty), []).append(case)

    total = len(valid_cases)
    exact = {key: len(members) * n / total for key, members in cells.items()}
    allocation = {key: min(len(cells[key]), int(value)) for key, value in exact.items()}
    remaining = n - sum(allocation.values())
    # Largest remainder, with the cell key as a deterministic tie-break.
    order = sorted(exact, key=lambda key: (-(exact[key] - int(exact[key])), key))
    index = 0
    while remaining > 0 and index < len(order) * 4:
        key = order[index % len(order)]
        if allocation[key] < len(cells[key]):
            allocation[key] += 1
            remaining -= 1
        index += 1

    rng = random.Random(seed)
    chosen: List[Any] = []
    for key, members in cells.items():
        pool = list(members)
        rng.shuffle(pool)
        chosen.extend(pool[: allocation[key]])
    chosen.sort(key=lambda c: c.case_index)

    record = {
        "strategy": "stratified-by-diagnosis_group-x-difficulty, largest-remainder, seeded",
        "seed": seed,
        "split": "valid",
        "n_available": total,
        "n_selected": len(chosen),
        "representative_of_split": True,
        "case_nums": [c.case_num for c in chosen],
        "diagnosis_groups": dict(sorted(Counter(c.diagnosis_group for c in chosen).items())),
        "difficulty": dict(sorted(Counter(c.difficulty for c in chosen).items())),
        "split_diagnosis_groups": dict(
            sorted(Counter(c.diagnosis_group for c in valid_cases).items())
        ),
        "split_difficulty": dict(sorted(Counter(c.difficulty for c in valid_cases).items())),
        "cells": {
            f"{group}|{difficulty}": {"in_split": len(members), "selected": allocation[key]}
            for key, members in cells.items()
            for group, difficulty in [key]
        },
    }
    return chosen, record


def chance_comparability(subset: Sequence[Any], valid_cases: Sequence[Any],
                         train_cases: Sequence[Any]) -> Dict[str, Any]:
    """How close the subset's chance baseline is to the full split's.

    The gate's headline number is a *lift over chance*, and chance is the train-majority constant
    scored on whatever cohort is in front of it.  If the subset happened to be easier for a
    constant predictor than the full split, a subset lift would not be comparable to G1's.  This
    reports both floors so the reader can check rather than trust.
    """
    majority = g1.majority_by_criterion(list(train_cases))
    out: Dict[str, Any] = {"per_criterion": {}}
    subset_values: List[float] = []
    split_values: List[float] = []
    for name in CRITERION_NAMES:
        on_subset = sum(1 for c in subset if str(c.criteria[name]) == majority[name]) / len(subset)
        on_split = (
            sum(1 for c in valid_cases if str(c.criteria[name]) == majority[name]) / len(valid_cases)
        )
        subset_values.append(on_subset)
        split_values.append(on_split)
        out["per_criterion"][name] = {
            "train_majority_value": majority[name],
            "chance_on_subset": round(on_subset, 4),
            "chance_on_full_valid": round(on_split, 4),
            "delta": round(on_subset - on_split, 4),
        }
    out["macro_chance_on_subset"] = round(statistics.fmean(subset_values), 4)
    out["macro_chance_on_full_valid"] = round(statistics.fmean(split_values), 4)
    out["macro_delta"] = round(
        statistics.fmean(subset_values) - statistics.fmean(split_values), 4
    )
    return out


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------
class Config:
    """One point in the F1 x F2 x F3 space."""

    def __init__(self, name: str, model: str, *, few_shot: bool = False,
                 clinical: bool = False, note: str = ""):
        self.name = name
        self.model = model
        self.few_shot = few_shot
        self.clinical = clinical
        self.note = note
        self.profile = profile_for(model)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.name,
            "model": self.model,
            "few_shot": self.few_shot,
            "clinical": self.clinical,
            "decoding": self.profile.to_dict(),
            "note": self.note,
        }


def parse_configs(spec: str) -> List[Config]:
    """``name:model[:fewshot][:clinical]`` per comma-separated entry."""
    out: List[Config] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 2:
            raise SystemExit(f"bad --configs entry {entry!r}; want name:model[:fewshot][:clinical]")
        name, model = parts[0], parts[1]
        flags = {p.lower() for p in parts[2:]}
        unknown = flags - {"fewshot", "clinical"}
        if unknown:
            raise SystemExit(f"unknown flag(s) in {entry!r}: {sorted(unknown)}")
        out.append(Config(name, model, few_shot="fewshot" in flags, clinical="clinical" in flags))
    return out


# ---------------------------------------------------------------------------
# Running one configuration
# ---------------------------------------------------------------------------
def _build_assessor(config: Config, sheets: Sequence[ExemplarSheet], cache: AssessorCache,
                    *, workers: int, allow_remote: bool, refresh: bool) -> FewShotAssessor:
    profile = config.profile
    return FewShotAssessor(
        client=None,
        model=profile.model,
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        seed=profile.seed,
        include_clinical=config.clinical,
        cache=cache,
        allow_remote=allow_remote,
        refresh=refresh,
        max_repair_attempts=1,
        sheets=list(sheets) if config.few_shot else None,
        workers=workers,
        client_factory=default_client_factory(profile.model) if allow_remote else None,
    )


def project_cost(config: Config, assessor: FewShotAssessor, cases: Sequence[Any]) -> Dict[str, Any]:
    """Price this configuration before a single call, and say how much of it is already cached."""
    requests = [assessor.build_request(case) for case in cases]
    estimates = [assessor.estimate_request_tokens(request) for request in requests]
    prompt_tokens = [e["prompt_tokens_estimated"] for e in estimates]
    image_tokens = [e["image_tokens_estimated"] for e in estimates]

    cached = sum(1 for r in requests if assessor.cache.entry_path(r.cache_key()).exists())
    uncached = len(requests) - cached
    # Reasoning models bill their thinking as completion tokens, so the *ceiling* is the honest
    # projection for them; a 294-token estimate taken from Qwen would understate Gemini by 7x.
    completion = assessor.decoding.max_tokens
    mean_prompt = statistics.fmean(prompt_tokens) if prompt_tokens else 0.0

    worst = config.profile.cost_usd(int(mean_prompt * uncached), completion * uncached)
    return {
        "n_cases": len(cases),
        "already_cached": cached,
        "would_be_paid": uncached,
        "mean_prompt_tokens_estimated": round(mean_prompt, 1),
        "mean_image_tokens_estimated": (
            round(statistics.fmean(image_tokens), 1) if image_tokens else None
        ),
        "max_completion_tokens": completion,
        "projected_cost_usd_upper_bound": round(worst, 4) if worst is not None else None,
        "note": (
            "upper bound: assumes every uncached case emits max_tokens of completion, which is "
            "what a reasoning model approaches and a non-reasoning model does not"
        ),
    }


def correctness_by_case(report: Mapping[str, Any]) -> Dict[int, List[bool]]:
    """``case_num -> [criterion correct?]`` in canonical criterion order, scored cases only.

    ``indeterminate`` counts as wrong here, matching the ``strict`` view whose macro average is
    G1's headline.  With every scored case contributing all seven criteria, the mean of these
    flags *is* the macro strict accuracy, which is what makes a case-level bootstrap legitimate.
    """
    out: Dict[int, List[bool]] = {}
    for entry in report.get("per_case", []):
        criteria = entry.get("criteria")
        if not criteria:
            continue
        out[int(entry["case_num"])] = [bool(criteria[name]["agree"]) for name in CRITERION_NAMES]
    return out


def paired_vs_baseline(reports: Mapping[str, Any], baseline_name: str) -> Dict[str, Any]:
    """Paired, case-clustered bootstrap of each configuration's macro strict accuracy vs baseline.

    Every configuration saw the same cases, so the comparison is paired and the unpaired
    confidence interval on each arm separately would be needlessly wide.  Clustering the resample
    on ``case_num`` -- not on the 7n individual criterion judgements -- is the honest unit: seven
    judgements about one lesion are not seven independent observations, and treating them as such
    would shrink every interval by roughly sqrt(7).

    Holm across the configurations, because "is any of these readers better" is a family of tests
    and the largest of six differences is not a single pre-registered comparison.
    """
    from medguard_v2.eval.statistics import (
        bootstrap_two_sided_p,
        clustered_bootstrap_distribution,
        holm_adjust,
        mcnemar_exact,
    )

    def macro_strict(rows: Sequence[Sequence[bool]]) -> float:
        judgements = sum(len(row) for row in rows)
        return sum(sum(row) for row in rows) / judgements if judgements else 0.0

    base = reports.get(baseline_name)
    if base is None:
        return {"error": f"no baseline arm named {baseline_name!r} in this run"}
    base_flags = correctness_by_case(base)

    entries: Dict[str, Any] = {}
    raw_p: Dict[str, float] = {}
    for name, report in reports.items():
        if name == baseline_name:
            continue
        flags = correctness_by_case(report)
        shared = sorted(set(flags) & set(base_flags))
        if not shared:
            entries[name] = {"error": "no cases scored by both arms"}
            continue
        rows_a = [flags[num] for num in shared]
        rows_b = [base_flags[num] for num in shared]
        clusters = [str(num) for num in shared]

        # Resample the *pair*, never the two arms independently: the whole point of running every
        # configuration on one cohort is that the case-to-case variance cancels in the difference.
        interval, replicates = clustered_bootstrap_distribution(
            list(zip(rows_a, rows_b)), clusters,
            lambda pairs: macro_strict([p[0] for p in pairs]) - macro_strict([p[1] for p in pairs]),
            n_boot=2000, seed=SUBSET_SEED,
        )
        p_value = bootstrap_two_sided_p(replicates, 0.0)
        # McNemar over the flattened criterion judgements answers a different, coarser question
        # ("do the two readers disagree asymmetrically") and is reported alongside, not instead:
        # it ignores the case clustering the bootstrap respects, so it is the optimistic one.
        flat_a = [flag for row in rows_a for flag in row]
        flat_b = [flag for row in rows_b for flag in row]
        a_only, b_only, p_mcnemar = mcnemar_exact(flat_a, flat_b)
        raw_p[name] = float(p_value)
        entries[name] = {
            "n_paired_cases": len(shared),
            "macro_strict_accuracy_delta": round(interval.estimate, 4),
            "ci95": [round(interval.low, 4), round(interval.high, 4)],
            "bootstrap_p": round(float(p_value), 4),
            "mcnemar_criterion_level": {
                "config_only_correct": a_only,
                "baseline_only_correct": b_only,
                "p_uncorrected_ignores_case_clustering": round(p_mcnemar, 5),
            },
        }
    adjusted = holm_adjust(raw_p) if raw_p else {}
    for name, value in adjusted.items():
        entries[name]["bootstrap_p_holm"] = round(float(value), 4)
    return {
        "baseline_arm": baseline_name,
        "unit_of_resampling": "case (7 criterion judgements per case are not independent)",
        "multiplicity": "Holm across configurations",
        "per_config": entries,
    }


def _row(config: Config, report: Mapping[str, Any], *, spend: Optional[float]) -> Dict[str, Any]:
    macro = report["macro_across_seven"]
    agreement = report.get("seven_point_score_agreement") or {}
    full = agreement.get("fully_determinate") or {}
    usage = report["usage"]
    return {
        "config": config.name,
        "model": config.model,
        "few_shot": config.few_shot,
        "clinical": config.clinical,
        "n_cases": report["cohort"]["n_cases"],
        "n_scored": report["reliability"]["n_scored"],
        "parse_failures": report["reliability"]["parse_failures"],
        "api_failures": report["reliability"]["api_failures"],
        "macro_accuracy_strict": macro["accuracy_strict"],
        "macro_accuracy_determinate": macro["accuracy_determinate"],
        "macro_chance_train_majority": macro["chance_accuracy_train_majority"],
        "macro_lift_over_chance": macro["lift_over_chance_strict"],
        "macro_cohen_kappa": macro["cohen_kappa"],
        "macro_f1_supported_classes": macro["macro_f1_supported_classes"],
        "indeterminate_rate": macro["indeterminate_rate"],
        "score_mae": full.get("mae"),
        "score_exact_match": full.get("exact_match_rate"),
        "score_within_one": full.get("within_one_rate"),
        "confidence_mean_correct": report["confidence"]["mean_when_correct"],
        "confidence_mean_wrong": report["confidence"]["mean_when_wrong"],
        "confidence_separation": report["confidence"]["separation"],
        "paid_cases": usage["paid_this_run"]["cases"],
        "paid_prompt_tokens": usage["paid_this_run"]["prompt_tokens"],
        "paid_completion_tokens": usage["paid_this_run"]["completion_tokens"],
        "cached_cases": usage["replayed_from_cache"]["cases"],
        "spend_usd_this_run": round(spend, 4) if spend is not None else None,
        "per_criterion_lift": {
            name: report["per_criterion"][name]["lift_over_train_majority"]["strict_accuracy"]
            for name in CRITERION_NAMES
            if name in report["per_criterion"]
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    configs: Sequence[Config],
    *,
    subset_n: int = SUBSET_N,
    seed: int = SUBSET_SEED,
    case_nums: Optional[Sequence[int]] = None,
    workers: int = 4,
    dry_run: bool = False,
    budget_usd: Optional[float] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    cache_only: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]

    if case_nums:
        by_num = {c.case_num: c for c in valid}
        unknown = [n for n in case_nums if n not in by_num]
        if unknown:
            raise SystemExit(f"case_num(s) not in the valid split: {unknown}")
        subset = [by_num[n] for n in case_nums]
        subset.sort(key=lambda c: c.case_index)
        selection = {
            "strategy": "explicit-case-nums (nested screen)",
            "split": "valid",
            "n_available": len(valid),
            "n_selected": len(subset),
            "representative_of_split": False,
            "case_nums": [c.case_num for c in subset],
        }
    else:
        subset, selection = select_subset(valid, subset_n, seed)

    cohort, view_record = analysis_cohort(subset, require_distinct_views=False)
    assert len(cohort) == len(subset)

    # ---- F2 assets: built once, from train only, before anything is priced ----------------
    sheets: List[ExemplarSheet] = []
    exemplars: Dict[str, Dict[str, int]] = {}
    if any(config.few_shot for config in configs):
        import numpy as np

        blob = np.load(FEATURES)
        features = {str(num): vector for num, vector in zip(blob["case_num"], blob["features"])}
        exemplars = select_exemplars(train, features)
        sheets = build_contact_sheets(exemplars, {c.case_num: c for c in train}, SHEET_DIR)

    cache = AssessorCache(cache_dir)
    if not dry_run:
        from medguard import config as v1_config  # lazy: only the live path needs credentials

        v1_config.load_env()
        if not v1_config.has_api_key():
            raise SystemExit(
                "no gateway credential in the environment; the live path cannot run. "
                "Use --dry-run, or set MEDGUARD_API_KEY / KILOCODE_API_KEY."
            )

    rows: List[Dict[str, Any]] = []
    projections: List[Dict[str, Any]] = []
    reports: Dict[str, Any] = {}
    spent = 0.0
    skipped: List[Dict[str, Any]] = []

    for config in configs:
        assessor = _build_assessor(
            config, sheets, cache,
            workers=workers, allow_remote=not (dry_run or cache_only), refresh=refresh,
        )
        projection = project_cost(config, assessor, cohort)
        projection["config"] = config.name
        projection["model"] = config.model
        projections.append(projection)

        if dry_run:
            continue

        upper = projection["projected_cost_usd_upper_bound"]
        if budget_usd is not None and upper is not None and spent + upper > budget_usd:
            skipped.append({
                "config": config.name,
                "reason": "would exceed --budget-usd",
                "spent_so_far_usd": round(spent, 4),
                "projected_upper_bound_usd": upper,
                "budget_usd": budget_usd,
            })
            if verbose:
                print(f"SKIP {config.name}: projection ${upper} would exceed budget "
                      f"${budget_usd} (spent ${spent:.4f})", file=sys.stderr, flush=True)
            continue

        if verbose:
            print(f"\n=== {config.name} ({config.model}) few_shot={config.few_shot} "
                  f"clinical={config.clinical} | {projection['would_be_paid']} paid / "
                  f"{projection['already_cached']} cached | <= ${upper}", file=sys.stderr,
                  flush=True)

        assessor.client = default_client_factory(config.profile.model)()
        report = g1.run_live(cohort, assessor, train, selection=selection, verbose=False)
        paid = report["usage"]["paid_this_run"]
        spend = config.profile.cost_usd(paid["prompt_tokens"], paid["completion_tokens"])
        if spend is not None:
            spent += spend
        report["g1c_config"] = config.to_dict()
        report["g1c_spend_usd_this_run"] = round(spend, 4) if spend is not None else None
        reports[config.name] = report
        rows.append(_row(config, report, spend=spend))
        if verbose:
            print(g1.summarise(report), file=sys.stderr, flush=True)
            print(f"  spend this run: ${spend:.4f}  (cumulative ${spent:.4f})",
                  file=sys.stderr, flush=True)

    # ---- F3, scored where it can actually act -------------------------------------------
    # ``analysis_cohort`` names the cases whose "clinical" image is a copy of the dermoscopic
    # file. Adding a second view is a *no-op* on those, so leaving them in dilutes the measured
    # cross-modality effect with cases that never had two views. Every answer needed here was
    # just paid for, so this pass is cache-only and free -- and it is refused rather than
    # silently re-paid if anything is missing.
    distinct_only = [case for case in cohort if case.has_distinct_views]
    paired: Dict[str, Any] = {
        "n_cases": len(distinct_only),
        "excluded_case_nums": sorted(c.case_num for c in cohort if not c.has_distinct_views),
        "reason": (
            "adding the clinical view is a no-op on cases whose clinic image is the derm file; "
            "the F3 contrast is only interpretable on cases that have two real views"
        ),
        "rows": [],
    }
    if not dry_run and rows and len(distinct_only) < len(cohort):
        for config in configs:
            if config.name not in reports:
                continue
            replay = _build_assessor(
                config, sheets, cache, workers=1, allow_remote=False, refresh=False,
            )
            try:
                restricted = g1.run_live(distinct_only, replay, train, selection=None,
                                         verbose=False)
            except RuntimeError as exc:  # a cache-only miss: report it, never re-pay silently
                paired["rows"].append({"config": config.name, "error": str(exc)})
                continue
            paired["rows"].append(_row(config, restricted, spend=0.0))
    out_paired = paired

    baseline_row = next((r for r in rows if not r["few_shot"] and not r["clinical"]
                         and r["model"] == BASELINE_MODEL), None)

    out: Dict[str, Any] = {
        "gate_id": "G1c_stronger_reader",
        "mode": "dry_run" if dry_run else "live",
        "question": (
            "G1 concluded a VLM cannot read the seven criteria from ONE configuration. "
            "Does a stronger reader -- better model, visual exemplars, both views -- change it?"
        ),
        "boundaries": {
            "splits_evaluated": sorted({c.split for c in cohort}),
            "test_split_read": any(c.split == "test" for c in cohort),
            "exemplar_split": "train" if sheets else None,
            "cache_dir": str(cache_dir),
            "shared_cache_with_g1": Path(cache_dir) == Path(DEFAULT_CACHE_DIR),
        },
        "cohort": {
            "selection": selection,
            "clinical_view_availability": view_record,
            "chance_comparability_vs_full_valid": chance_comparability(cohort, valid, train),
        },
        "few_shot_design": {
            "design_version": FEWSHOT_DESIGN_VERSION,
            "exemplar_selection_version": EXEMPLAR_SELECTION_VERSION,
            "rule": (
                "one contact sheet per criterion; one panel per permitted value; each panel is the "
                "TRAIN case whose frozen DermLIP feature vector is closest to that value's train "
                "centroid (the medoid), centre-cropped to 252x252 with a 28px caption strip"
            ),
            "token_cost_full_vocabulary": fewshot_image_token_cost(),
            "image_tokens_per_case_actual": sum(sheet.image_tokens for sheet in sheets),
            "sheets": [sheet.to_dict() for sheet in sheets],
            "exemplar_case_nums_by_value": exemplars,
            "permitted_values_with_no_train_example": exemplar_gaps(exemplars) if sheets else {},
            "sheet_dir": str(SHEET_DIR) if sheets else None,
            "redistribution": (
                "sheets are derived Derm7pt images (CC BY-NC-ND); the directory carries its own "
                ".gitignore and only SHA-256 digests are recorded here"
            ),
        },
        "configs": [config.to_dict() for config in configs],
        "projections": projections,
        "skipped_configs": skipped,
        "results": rows,
        "results_on_cases_with_two_real_views": out_paired,
        "g1_published_headline": G1_HEADLINE,
        "total_spend_usd_this_run": round(spent, 4),
    }

    if rows:
        baseline_name = baseline_row["config"] if baseline_row else None
        out["paired_vs_baseline"] = (
            paired_vs_baseline(reports, baseline_name) if baseline_name and len(reports) > 1
            else {"error": "needs the baseline arm plus at least one other configuration"}
        )
        best = max(rows, key=lambda r: r["macro_lift_over_chance"])
        out["verdict"] = {
            "best_config": best["config"],
            "best_macro_lift_over_chance": best["macro_lift_over_chance"],
            "g1_macro_lift_over_chance": G1_HEADLINE["macro_lift_over_chance"],
            "beats_chance": best["macro_lift_over_chance"] > 0,
            "beats_g1_on_this_subset": (
                best["macro_lift_over_chance"] > baseline_row["macro_lift_over_chance"]
                if baseline_row else None
            ),
            "baseline_arm_on_this_subset": (
                baseline_row["macro_lift_over_chance"] if baseline_row else None
            ),
            "note": (
                "The comparator for 'did a stronger reader help' is the baseline arm on THIS "
                "subset, not G1's published full-split number: the subset's chance floor differs. "
                "Both are reported."
            ),
        }
        out["per_config_reports"] = reports
    return out


# ---------------------------------------------------------------------------
# Human-readable table
# ---------------------------------------------------------------------------
def summarise(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    if report["mode"] == "dry_run":
        lines.append("G1c DRY RUN -- 0 API calls")
        design = report["few_shot_design"]
        cost = design["token_cost_full_vocabulary"]
        lines.append(
            f"  few-shot sheets: {len(design['sheets'])} x {cost['panel_px']}px panels -> "
            f"{design['image_tokens_per_case_actual']} image tokens/case "
            f"(full-vocabulary ceiling {cost['total_image_tokens']})"
        )
        if design["permitted_values_with_no_train_example"]:
            lines.append(
                f"  values with no train exemplar: "
                f"{design['permitted_values_with_no_train_example']}"
            )
        header = f"  {'config':<26} {'paid':>5} {'cached':>7} {'in/case':>9} {'<= $':>9}"
        lines.append(header)
        for block in report["projections"]:
            lines.append(
                f"  {block['config']:<26} {block['would_be_paid']:>5} "
                f"{block['already_cached']:>7} {block['mean_prompt_tokens_estimated']:>9} "
                f"{str(block['projected_cost_usd_upper_bound']):>9}"
            )
        return "\n".join(lines)

    selection = report["cohort"]["selection"]
    lines.append(
        f"G1c LIVE -- {selection['n_selected']} case(s) from {selection['split']}, "
        f"{len(report['results'])} configuration(s), "
        f"${report['total_spend_usd_this_run']} spent"
    )
    header = (f"  {'config':<26} {'acc':>6} {'chance':>7} {'lift':>7} {'kappa':>7} "
              f"{'MAE':>6} {'ind':>6} {'confsep':>8} {'$':>7}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row in report["results"]:
        lines.append(
            f"  {row['config']:<26} {row['macro_accuracy_strict']:>6} "
            f"{row['macro_chance_train_majority']:>7} {row['macro_lift_over_chance']:>7} "
            f"{row['macro_cohen_kappa']:>7} {str(row['score_mae']):>6} "
            f"{row['indeterminate_rate']:>6} {str(row['confidence_separation']):>8} "
            f"{str(row['spend_usd_this_run']):>7}"
        )
    headline = report["g1_published_headline"]
    lines.append(
        f"  {'[G1 published n=203]':<26} {headline['macro_accuracy_strict']:>6} "
        f"{headline['macro_chance_train_majority']:>7} {headline['macro_lift_over_chance']:>7} "
        f"{headline['macro_cohen_kappa']:>7} {headline['score_mae']:>6} "
        f"{headline['indeterminate_rate']:>6} {headline['confidence_separation']:>8}"
    )
    stats = (report.get("paired_vs_baseline") or {}).get("per_config") or {}
    if stats:
        lines.append("  -- paired vs the baseline arm, same cases, bootstrap clustered on case --")
        for name, entry in stats.items():
            if "error" in entry:
                lines.append(f"  {name:<26} {entry['error']}")
                continue
            lines.append(
                f"  {name:<26} delta {entry['macro_strict_accuracy_delta']:+.4f} "
                f"CI95 [{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}] "
                f"p={entry['bootstrap_p']} holm={entry.get('bootstrap_p_holm')}"
            )

    paired = report.get("results_on_cases_with_two_real_views") or {}
    if paired.get("rows"):
        lines.append(
            f"  -- same table restricted to the {paired['n_cases']} cases with two real views "
            f"(excluded {paired['excluded_case_nums']}), replayed from cache --"
        )
        for row in paired["rows"]:
            if "error" in row:
                lines.append(f"  {row['config']:<26} CACHE MISS: {row['error'][:60]}")
                continue
            lines.append(
                f"  {row['config']:<26} {row['macro_accuracy_strict']:>6} "
                f"{row['macro_chance_train_majority']:>7} {row['macro_lift_over_chance']:>7} "
                f"{row['macro_cohen_kappa']:>7} {str(row['score_mae']):>6} "
                f"{row['indeterminate_rate']:>6} {str(row['confidence_separation']):>8}"
            )
    verdict = report.get("verdict")
    if verdict:
        lines.append(
            f"  best: {verdict['best_config']} lift {verdict['best_macro_lift_over_chance']} "
            f"(baseline arm on same subset {verdict['baseline_arm_on_this_subset']})"
        )
    for entry in report.get("skipped_configs", []):
        lines.append(f"  SKIPPED {entry['config']}: {entry['reason']} "
                     f"(projected ${entry['projected_upper_bound_usd']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_LADDER = (
    "baseline:qwen/qwen3-vl-32b-instruct,"
    "baseline_fewshot:qwen/qwen3-vl-32b-instruct:fewshot,"
    "qwen235b:qwen/qwen3-vl-235b-a22b-instruct,"
    "gemini31pro:google/gemini-3.1-pro-preview,"
    "claudeopus5:anthropic/claude-opus-5,"
    "gpt56terra:openai/gpt-5.6-terra,"
    "grok45:x-ai/grok-4.5"
)


def list_models() -> None:
    """Ask the gateway what it serves.  One unauthenticated-shaped GET, no completion."""
    from medguard import config as v1_config

    v1_config.load_env()
    from openai import OpenAI

    client = OpenAI(api_key=v1_config.GATEWAY_API_KEY, base_url=v1_config.GATEWAY_BASE_URL)
    data = client.models.list().model_dump()["data"]
    vision = []
    for entry in data:
        modalities = (entry.get("architecture") or {}).get("input_modalities") or []
        if "image" not in modalities:
            continue
        pricing = entry.get("pricing") or {}
        try:
            price_in = float(pricing.get("prompt", "0")) * 1e6
            price_out = float(pricing.get("completion", "0")) * 1e6
        except (TypeError, ValueError):
            price_in = price_out = float("nan")
        vision.append((entry["id"], price_in, price_out,
                       sorted(entry.get("supported_parameters") or [])))
    vision.sort()
    print(f"{len(data)} models served, {len(vision)} accept image input")
    for model, price_in, price_out, params in vision:
        flags = ",".join(p for p in ("temperature", "seed") if p in params) or "-"
        print(f"  {model:<46} in=${price_in:>8.3f}/M out=${price_out:>8.3f}/M  accepts:{flags}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=DEFAULT_LADDER)
    parser.add_argument("--subset-n", type=int, default=SUBSET_N)
    parser.add_argument("--seed", type=int, default=SUBSET_SEED)
    parser.add_argument("--case-nums", default=None,
                        help="comma-separated valid-split case_nums; overrides --subset-n")
    parser.add_argument("--full-valid", action="store_true",
                        help="run on all 203 valid cases instead of the stratified subset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.list_models:
        list_models()
        return

    case_nums = (
        [int(part) for part in args.case_nums.split(",") if part.strip()]
        if args.case_nums else None
    )
    subset_n = 203 if args.full_valid else args.subset_n

    report = run(
        parse_configs(args.configs),
        subset_n=subset_n,
        seed=args.seed,
        case_nums=case_nums,
        workers=args.workers,
        dry_run=args.dry_run,
        budget_usd=args.budget_usd,
        cache_dir=Path(args.cache_dir),
        refresh=args.refresh,
        cache_only=args.cache_only,
        verbose=not args.quiet,
    )

    text = json.dumps(report, indent=2, default=str)
    if args.write or args.out:
        out = Path(args.out) if args.out else OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    print(summarise(report))


if __name__ == "__main__":
    main()
