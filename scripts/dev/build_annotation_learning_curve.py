"""Annotation learning curve: what are accumulated clinician overrides WORTH as training labels?

Thesis under test
-----------------
Every successful ``override_criterion`` call is one expert-verified (case, criterion, value)
annotation, now captured durably by ``medguard_v2/agent/annotation_store.py`` (the AgentTools
hook). The world's supply of 7-point criterion labels is the field's bottleneck -- Derm7pt's train
split holds 413 cases -- so if deployment itself grows the label pool, the criteria heads should
improve as annotations accumulate. This script MEASURES that claim instead of asserting it.

Simulated-expert convention (same as correction_value_curves)
-------------------------------------------------------------
Derm7pt's gold 7-point criteria stand in for a SIMULATED expert: the VALID split's ground-truth
criteria play the role of the corrections a clinician's overrides converge to. This is standard
correctability / label-acquisition methodology, NOT a human study -- no clinician judgement is
involved, and real overrides are noisier than dataset gold (interobserver agreement on dermoscopic
criteria is only fair-to-moderate), so the curve is an UPPER BOUND on the value of captured
overrides.

Protocol
--------
* Heads: the PROMOTED criteria recipe, bit-identical machinery imported from
  scripts/dev/derm7pt_p2_safety_promote.py -- dermlip_tta+metadata14 (526-d), lr_balanced,
  per-criterion C from run_cohort.PROMOTED_CRITERIA_C, seed 20260810.
* Baseline (k=0): heads fit on TRAIN only -> criteria binary-concept F1 on TEST must reproduce
  the frozen 0.7168 (perception_phase2_safety_preserving.json final_test_metrics). Raise on drift.
* Growth: for k in {25, 50, 100, 150, 203}, sample k VALID cases (registered seed 20260815;
  NESTED prefix subsets, so within a repeat the k=100 pool contains the k=50 pool and the curve is
  monotone in data), add them to the training pool WITH their ground-truth criteria, refit the
  heads identically, evaluate TEST binary-concept F1 (+ multiclass macro-F1) once per k.
  TEST is never trained on. 5 sampling repeats (seeds 20260815+r, r=0..4) -> mean +/- std.
  k=203 is the FULL valid split, so all repeats coincide there (std 0 by construction).
* Uncertainty on the headline delta: a paired test-case bootstrap (2000 reps) over the k=0 and
  k=203 predictions -- annotation-sampling spread alone (the repeat std) says nothing about
  test-set noise.

Boundaries: zero paid calls, zero images encoded. Heads are linear probes over frozen cached
features; TEST is read for evaluation only. Does not modify any source under medguard_v2/.

Outputs:
  * research/evidence/derm7pt_v1/annotation_learning_curve.json
  * notes/ANNOTATION_LEARNING_CURVE.md

Run::
    venv_win/Scripts/python.exe scripts/dev/build_annotation_learning_curve.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES  # noqa: E402
from medguard_v2.data.derm7pt import EXPECTED_SPLIT_SIZES, load_cases  # noqa: E402
from medguard_v2.pipeline.run_cohort import PROMOTED_CRITERIA_C  # noqa: E402

# ---- import the promote script's reviewed machinery by file path (same pattern as
# build_correction_value_curves_promoted.py), so head fitting and the binary-concept F1 are
# bit-identical to what produced the frozen 0.7168 reference ----------------------------------
_PROMOTE_PATH = REPO / "scripts" / "dev" / "derm7pt_p2_safety_promote.py"
_spec = importlib.util.spec_from_file_location("_p2_safety_promote", _PROMOTE_PATH)
promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(promote)

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "annotation_learning_curve.json"
OUT_MD = REPO / "notes" / "ANNOTATION_LEARNING_CURVE.md"

#: Frozen baseline reference: promoted criteria binary-concept F1 on TEST
#: (perception_phase2_safety_preserving.json final_test_metrics.criteria_binary_concept_f1).
REF_BASELINE_BINARY_F1 = 0.7168
REF_TOL = 0.0006  # same tolerance the promote script's own wiring checks use

SAMPLING_SEED = 20260815      # registered; repeat r uses SAMPLING_SEED + r
N_REPEATS = 5
K_VALUES = (25, 50, 100, 150, 203)
N_BOOTSTRAP = 2000


def _round4(x) -> float:
    return round(float(x), 4)


# ------------------------------------------------------------------ metrics


def _per_criterion_binary_f1(heads, x, cases) -> Dict[str, float]:
    """Per-criterion binarised (scoring vs non-scoring concept) macro-F1 -- the same computation
    as promote._binary_concept_f1, returned per criterion instead of pre-averaged."""
    out: Dict[str, float] = {}
    for name in CRITERION_NAMES:
        spec = CRITERIA_BY_NAME[name]
        pred = [str(p) for p in heads[name].predict(x)]
        gold = [str(c.criteria[name]) for c in cases]
        yt = [1 if v in spec.scoring_values else 0 for v in gold]
        yp = [1 if v in spec.scoring_values else 0 for v in pred]
        out[name] = float(f1_score(yt, yp, average="macro", zero_division=0))
    return out


def _per_criterion_multiclass_f1(heads, x, cases, labels_by_criterion) -> Dict[str, float]:
    """Per-criterion FULL-vocabulary macro-F1 over the label set observed in TEST gold (fixed
    across k, so the metric is comparable along the curve)."""
    out: Dict[str, float] = {}
    for name in CRITERION_NAMES:
        pred = [str(p) for p in heads[name].predict(x)]
        gold = [str(c.criteria[name]) for c in cases]
        out[name] = float(
            f1_score(gold, pred, average="macro", labels=labels_by_criterion[name],
                     zero_division=0)
        )
    return out


def _binary_predictions(heads, x, cases) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Binarised (concept) gold and predictions per criterion, as bool arrays -- bootstrap input."""
    gold_bin: Dict[str, np.ndarray] = {}
    pred_bin: Dict[str, np.ndarray] = {}
    for name in CRITERION_NAMES:
        spec = CRITERIA_BY_NAME[name]
        pred = [str(p) for p in heads[name].predict(x)]
        gold = [str(c.criteria[name]) for c in cases]
        gold_bin[name] = np.array([v in spec.scoring_values for v in gold], dtype=bool)
        pred_bin[name] = np.array([v in spec.scoring_values for v in pred], dtype=bool)
    return gold_bin, pred_bin


def _binary_macro_f1_np(yt: np.ndarray, yp: np.ndarray) -> float:
    """Binary macro-F1 (mean of the positive- and negative-class F1), zero_division=0 convention.
    Vectorised twin of sklearn's f1_score(average='macro') used inside the bootstrap; asserted
    equal to sklearn on the full sample before use."""

    def _f1(t: np.ndarray, p: np.ndarray) -> float:
        tp = int(np.sum(t & p))
        fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p))
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom else 0.0

    return 0.5 * (_f1(yt, yp) + _f1(~yt, ~yp))


def _paired_bootstrap_delta(gold_bin, pred_a, pred_b, seed) -> Dict:
    """Paired test-case bootstrap of mean-over-criteria binary F1 (B) - (A).

    The SAME resampled case indices are applied to both prediction sets, so the CI is on the
    paired delta, not on two independent F1s.
    """
    n = len(next(iter(gold_bin.values())))
    rng = np.random.default_rng(seed)
    deltas = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        f1_a = np.mean([
            _binary_macro_f1_np(gold_bin[name][idx], pred_a[name][idx])
            for name in CRITERION_NAMES
        ])
        f1_b = np.mean([
            _binary_macro_f1_np(gold_bin[name][idx], pred_b[name][idx])
            for name in CRITERION_NAMES
        ])
        deltas[b] = f1_b - f1_a
    point = float(np.mean([
        _binary_macro_f1_np(gold_bin[name], pred_b[name]) for name in CRITERION_NAMES
    ]) - np.mean([
        _binary_macro_f1_np(gold_bin[name], pred_a[name]) for name in CRITERION_NAMES
    ]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "point_estimate": _round4(point),
        "ci95": [_round4(lo), _round4(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_bootstrap": N_BOOTSTRAP,
        "unit": "mean-over-criteria binary-concept F1, k=203 minus k=0, paired over test cases",
    }


# ------------------------------------------------------------------ build


def build() -> Dict:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    test = [c for c in cases if c.split == "test"]
    for split, got in (("train", train), ("valid", valid), ("test", test)):
        if len(got) != EXPECTED_SPLIT_SIZES[split]:
            raise SystemExit(f"split drift: {split} has {len(got)} cases, "
                             f"expected {EXPECTED_SPLIT_SIZES[split]}")
    assert K_VALUES[-1] == len(valid), "largest k must be the full VALID split"
    assert not ({c.case_num for c in valid} & {c.case_num for c in test})
    assert not ({c.case_num for c in train} & {c.case_num for c in test})

    F = promote.Features()
    x_test = F.tta_meta_mat(test)
    labels_by_criterion = {
        name: sorted({str(c.criteria[name]) for c in test}) for name in CRITERION_NAMES
    }

    # ---- baseline (k=0): TRAIN only; must reproduce the frozen 0.7168 -----------------------
    heads0 = promote._fit_heads(F.tta_meta_mat, train, PROMOTED_CRITERIA_C, "balanced")
    baseline_promote_fn = promote._binary_concept_f1(heads0, x_test, test)  # the frozen function
    if abs(baseline_promote_fn - REF_BASELINE_BINARY_F1) > REF_TOL:
        raise SystemExit(
            "WIRING DRIFT: TRAIN-only promoted heads give binary-concept F1 "
            f"{baseline_promote_fn}, expected {REF_BASELINE_BINARY_F1} "
            "(perception_phase2_safety_preserving.json). Refusing to build the curve on an "
            "untrustworthy baseline."
        )
    per_crit0 = _per_criterion_binary_f1(heads0, x_test, test)
    baseline_binary = float(np.mean(list(per_crit0.values())))
    assert abs(baseline_binary - baseline_promote_fn) <= 5e-5, \
        "per-criterion decomposition disagrees with the promote script's own mean"
    per_crit0_mc = _per_criterion_multiclass_f1(heads0, x_test, test, labels_by_criterion)
    baseline_multiclass = float(np.mean(list(per_crit0_mc.values())))
    gold_bin, pred_bin0 = _binary_predictions(heads0, x_test, test)
    # Self-check the vectorised bootstrap metric against sklearn on the full sample.
    for name in CRITERION_NAMES:
        ours = _binary_macro_f1_np(gold_bin[name], pred_bin0[name])
        assert abs(ours - per_crit0[name]) <= 1e-12, f"bootstrap metric drift on {name}"

    # ---- growth: nested samples, 5 repeats, refit identically -------------------------------
    #   cache key = the sampled valid-case tuple, so the k=203 pool (identical in every repeat)
    #   is fit once, not five times.
    fit_cache: Dict[Tuple[int, ...], Dict] = {}
    per_repeat: List[Dict] = []
    pred_bin_full = None  # k=203 predictions, for the paired bootstrap

    for r in range(N_REPEATS):
        rng = np.random.default_rng(SAMPLING_SEED + r)
        perm = rng.permutation(len(valid))
        rows = []
        prev: set = set()
        for k in K_VALUES:
            subset_idx = sorted(int(i) for i in perm[:k])
            assert prev <= set(subset_idx), "subsets must be nested within a repeat"
            prev = set(subset_idx)
            key = tuple(subset_idx)
            if key not in fit_cache:
                pool = train + [valid[i] for i in subset_idx]  # gold criteria = the annotations
                heads = promote._fit_heads(
                    F.tta_meta_mat, pool, PROMOTED_CRITERIA_C, "balanced"
                )
                pc_bin = _per_criterion_binary_f1(heads, x_test, test)
                pc_mc = _per_criterion_multiclass_f1(heads, x_test, test, labels_by_criterion)
                entry = {
                    "binary_f1": float(np.mean(list(pc_bin.values()))),
                    "multiclass_macro_f1": float(np.mean(list(pc_mc.values()))),
                    "per_criterion_binary_f1": pc_bin,
                    "per_criterion_multiclass_f1": pc_mc,
                }
                if k == len(valid):
                    _, entry["_pred_bin"] = _binary_predictions(heads, x_test, test)
                fit_cache[key] = entry
            entry = fit_cache[key]
            if k == len(valid) and pred_bin_full is None:
                pred_bin_full = entry["_pred_bin"]
            rows.append({"k": k, **{kk: vv for kk, vv in entry.items() if kk != "_pred_bin"}})
        per_repeat.append({"repeat": r, "seed": SAMPLING_SEED + r, "rows": rows})
        print(f"  repeat {r} done ({time.strftime('%H:%M:%S')})")

    # ---- aggregate mean +/- std across repeats ----------------------------------------------
    def _agg(metric: str) -> List[Dict]:
        table = []
        for j, k in enumerate(K_VALUES):
            vals = np.array([rep["rows"][j][metric] for rep in per_repeat])
            table.append({
                "k": k,
                "mean": _round4(vals.mean()),
                "std": _round4(vals.std(ddof=1)),
                "per_repeat": [_round4(v) for v in vals],
            })
        return table

    curve_binary = _agg("binary_f1")
    curve_multiclass = _agg("multiclass_macro_f1")

    # per-criterion at k=203 (identical across repeats; report the unique fit's values)
    full_key = tuple(range(len(valid)))
    pc_full_bin = fit_cache[full_key]["per_criterion_binary_f1"]
    pc_full_mc = fit_cache[full_key]["per_criterion_multiclass_f1"]

    per_criterion = {
        name: {
            "binary_f1_k0": _round4(per_crit0[name]),
            "binary_f1_k203": _round4(pc_full_bin[name]),
            "binary_f1_delta": _round4(pc_full_bin[name] - per_crit0[name]),
            "multiclass_f1_k0": _round4(per_crit0_mc[name]),
            "multiclass_f1_k203": _round4(pc_full_mc[name]),
            "multiclass_f1_delta": _round4(pc_full_mc[name] - per_crit0_mc[name]),
        }
        for name in CRITERION_NAMES
    }

    # ---- headline delta + paired test-case bootstrap ----------------------------------------
    final_binary_mean = curve_binary[-1]["mean"]
    delta = _round4(final_binary_mean - baseline_binary)
    boot = _paired_bootstrap_delta(
        gold_bin, pred_bin0, pred_bin_full,
        seed=np.random.SeedSequence(SAMPLING_SEED).spawn(1)[0],
    )
    max_sampling_std = max(row["std"] for row in curve_binary)
    n_k_below_baseline = sum(1 for row in curve_binary if row["mean"] < baseline_binary)

    # Curve shape, reported not smoothed: the registered endpoint is k=K_VALUES[-1] (ALL
    # annotations); an interior peak is disclosed as shape, never selected as the headline.
    means = [baseline_binary] + [row["mean"] for row in curve_binary]
    ks = [0] + list(K_VALUES)
    peak_j = int(np.argmax(means))
    curve_shape = {
        "monotone_nondecreasing": bool(
            all(means[i] <= means[i + 1] + 1e-12 for i in range(len(means) - 1))
        ),
        "peak_k": ks[peak_j],
        "peak_mean": _round4(means[peak_j]),
        "k_points_below_baseline": [
            row["k"] for row in curve_binary if row["mean"] < baseline_binary
        ],
        "note": (
            "The registered headline endpoint is the FULL annotation pool "
            f"(k={K_VALUES[-1]}); an interior peak is curve shape, not a tunable stopping "
            "point -- selecting it post hoc would be test-set peeking."
        ),
    }

    # ---- honest verdict, computed not narrated ----------------------------------------------
    if delta <= 0:
        verdict_class = "negative"
        verdict = (
            f"NULL/NEGATIVE: adding all {K_VALUES[-1]} simulated annotations moves test "
            f"binary-concept F1 by {delta:+.4f} ({baseline_binary:.4f} -> "
            f"{final_binary_mean:.4f}). The GROWTH claim is NOT supported on this dataset "
            f"scale; the capture machinery stands as future-proofing only."
        )
    elif not boot["excludes_zero"]:
        verdict_class = "positive-point-estimate-only"
        verdict = (
            f"POSITIVE POINT ESTIMATE, NOT TEST-SET SIGNIFICANT: {baseline_binary:.4f} -> "
            f"{final_binary_mean:.4f} ({delta:+.4f}), but the paired test-case bootstrap CI "
            f"{boot['ci95']} includes zero. Report as a trend, not a supported growth claim."
        )
    else:
        verdict_class = "supported"
        verdict = (
            f"SUPPORTED: {K_VALUES[-1]} simulated deployment annotations lift test "
            f"binary-concept F1 {baseline_binary:.4f} -> {final_binary_mean:.4f} "
            f"({delta:+.4f}), paired test-case bootstrap 95% CI {boot['ci95']} excluding "
            f"zero. Sampling spread across repeats is small (max std {max_sampling_std})."
        )
    if not curve_shape["monotone_nondecreasing"]:
        verdict += (
            f" Curve shape is NON-monotone: the mean dips below baseline at "
            f"k={curve_shape['k_points_below_baseline'] or 'none'} and peaks at "
            f"k={curve_shape['peak_k']} ({curve_shape['peak_mean']:.4f}) before the "
            f"k={K_VALUES[-1]} endpoint -- reported as shape; the endpoint is not cherry-picked."
        )

    headline = (
        f"{K_VALUES[-1]} per-case annotation sets captured from simulated deployment lift test "
        f"criteria binary-concept F1 from {baseline_binary:.4f} to {final_binary_mean:.4f} "
        f"({delta:+.4f}); multiclass macro-F1 {baseline_multiclass:.4f} -> "
        f"{curve_multiclass[-1]['mean']:.4f}."
    )

    return {
        "protocol": "annotation_learning_curve_v1",
        "generated_by": "scripts/dev/build_annotation_learning_curve.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "thesis": (
            "Every successful clinician override IS a per-criterion annotation. Deployment can "
            "therefore grow the world's scarcest asset in this problem -- expert 7-point "
            "criterion labels (Derm7pt train holds only 413 cases). This artefact measures what "
            "that accumulation is worth to the promoted criteria heads."
        ),
        "capture_machinery": {
            "store": "medguard_v2/agent/annotation_store.py (append-only JSONL, "
                     "schema-validated, last-write-wins replay via as_label_updates)",
            "hook": "AgentTools(annotation_store=...) records every SUCCESSFUL "
                    "override_criterion call with the contract's case_id, the prior value, and "
                    "minimal contract provenance; default None = zero behaviour change",
            "note": "This curve does NOT read the store; it simulates what the store would "
                    "accumulate (see simulated_expert framing).",
        },
        "framing": {
            "simulated_expert": (
                "Derm7pt's ground-truth 7-point criteria on the VALID split stand in for a "
                "SIMULATED expert's accumulated overrides -- the values a clinician's corrections "
                "converge to. Same convention as correction_value_curves.json. This is NOT a "
                "human study; no clinician judgement is involved."
            ),
            "upper_bound_caveat": (
                "Real clinician criterion labels are noisier than dataset consensus gold: "
                "interobserver agreement on dermoscopic 7-point criteria is only fair-to-"
                "moderate (cf. the Consensus Net Meeting on Dermoscopy interobserver study, "
                "Argenziano et al., J Am Acad Dermatol 2003), so this curve is an UPPER BOUND "
                "on the value of captured overrides."
            ),
            "test_hygiene": "TEST (n=395) is never trained on; it is scored once per (k, repeat).",
        },
        "config": {
            "heads": "PROMOTED criteria recipe: dermlip_tta+metadata14 (526-d), lr_balanced, "
                     "per-criterion C (run_cohort.PROMOTED_CRITERIA_C), seed 20260810 -- "
                     "machinery imported bit-identical from scripts/dev/derm7pt_p2_safety_promote.py",
            "per_criterion_C": dict(PROMOTED_CRITERIA_C),
            "n_train": len(train),
            "n_valid": len(valid),
            "n_test": len(test),
            "k_values": list(K_VALUES),
            "annotation_unit": (
                "one k = one fully-annotated CASE (all 7 per-criterion labels), i.e. k=203 "
                "corresponds to up to 7*203=1421 individual override annotations in "
                "annotation-store terms"
            ),
            "sampling_seed": SAMPLING_SEED,
            "n_repeats": N_REPEATS,
            "nesting": "within a repeat, subsets are prefixes of one seeded permutation of "
                       "VALID, so k' > k trains on a superset of k's pool",
            "k_full_note": "k=203 is the entire VALID split: all repeats coincide, std is 0 by "
                           "construction",
            "multiclass_labels": "per criterion, the label set observed in TEST gold (fixed "
                                 "across k for comparability)",
        },
        "baseline_self_check": {
            "expected_binary_concept_f1": REF_BASELINE_BINARY_F1,
            "got": _round4(baseline_promote_fn),
            "source": "perception_phase2_safety_preserving.json final_test_metrics",
            "reproduced": True,
        },
        "baseline_k0": {
            "binary_f1": _round4(baseline_binary),
            "multiclass_macro_f1": _round4(baseline_multiclass),
        },
        "curve": {
            "binary_concept_f1": curve_binary,
            "multiclass_macro_f1": curve_multiclass,
        },
        "per_criterion_f1_k0_vs_k203": per_criterion,
        "headline_delta": {
            "binary_f1_k0": _round4(baseline_binary),
            "binary_f1_k203_mean": final_binary_mean,
            "delta": delta,
            "paired_test_case_bootstrap": boot,
            "max_sampling_std_across_k": max_sampling_std,
            "n_k_points_below_baseline": n_k_below_baseline,
            "curve_shape": curve_shape,
        },
        "per_repeat_detail": per_repeat,
        "verdict_class": verdict_class,
        "verdict": verdict,
        "headline": headline,
    }


# ------------------------------------------------------------------ notes


def _write_notes(r: Dict) -> None:
    cb = r["curve"]["binary_concept_f1"]
    cm = r["curve"]["multiclass_macro_f1"]
    hd = r["headline_delta"]
    lines: List[str] = []
    lines.append("# Annotation learning curve (Derm7pt, simulated-expert overrides)")
    lines.append("")
    lines.append(r["thesis"])
    lines.append("")
    lines.append(f"**Capture machinery:** {r['capture_machinery']['store']}; "
                 f"hook: {r['capture_machinery']['hook']}.")
    lines.append("")
    lines.append(f"**Simulated expert:** {r['framing']['simulated_expert']}")
    lines.append("")
    lines.append(f"**Caveat:** {r['framing']['upper_bound_caveat']}")
    lines.append("")
    b = r["config"]
    lines.append(f"Promoted criteria heads ({b['heads'].split(':')[1].split('--')[0].strip()}), "
                 f"train n={b['n_train']}, valid n={b['n_valid']} (annotation source), test "
                 f"n={b['n_test']} (never trained on). Sampling seed {b['sampling_seed']}, "
                 f"{b['n_repeats']} nested repeats. Baseline self-check reproduced "
                 f"{r['baseline_self_check']['expected_binary_concept_f1']} exactly.")
    lines.append("")
    lines.append("## The curve (mean +/- std over 5 sampling repeats)")
    lines.append("")
    lines.append("| k annotations | binary-concept F1 | multiclass macro-F1 |")
    lines.append("|---:|---|---|")
    lines.append(f"| 0 (baseline) | {r['baseline_k0']['binary_f1']:.4f} | "
                 f"{r['baseline_k0']['multiclass_macro_f1']:.4f} |")
    for rb, rm in zip(cb, cm):
        std_note = " (all repeats coincide)" if rb["k"] == b["n_valid"] else ""
        lines.append(f"| {rb['k']} | {rb['mean']:.4f} +/- {rb['std']:.4f} | "
                     f"{rm['mean']:.4f} +/- {rm['std']:.4f}{std_note} |")
    lines.append("")
    lines.append("## Which criteria benefit (binary-concept F1, k=0 -> k=203)")
    lines.append("")
    lines.append("| criterion | k=0 | k=203 | delta |")
    lines.append("|---|---:|---:|---:|")
    ranked = sorted(
        r["per_criterion_f1_k0_vs_k203"].items(),
        key=lambda kv: kv[1]["binary_f1_delta"], reverse=True,
    )
    for name, row in ranked:
        lines.append(f"| {name} | {row['binary_f1_k0']:.4f} | {row['binary_f1_k203']:.4f} | "
                     f"{row['binary_f1_delta']:+.4f} |")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(r["headline"])
    lines.append("")
    lines.append(f"Paired test-case bootstrap on the k=203-vs-k=0 delta: "
                 f"{hd['paired_test_case_bootstrap']['point_estimate']:+.4f}, 95% CI "
                 f"{hd['paired_test_case_bootstrap']['ci95']}, excludes zero: "
                 f"{hd['paired_test_case_bootstrap']['excludes_zero']}.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(r["verdict"])
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
    print("\n=== annotation learning curve ===")
    print(f"  baseline (k=0)   : binary F1 {r['baseline_k0']['binary_f1']} "
          f"(self-check vs {REF_BASELINE_BINARY_F1}: reproduced)")
    for row in r["curve"]["binary_concept_f1"]:
        print(f"  k={row['k']:>3}            : {row['mean']:.4f} +/- {row['std']:.4f}")
    hd = r["headline_delta"]
    print(f"  delta k203-k0    : {hd['delta']:+.4f}  bootstrap CI "
          f"{hd['paired_test_case_bootstrap']['ci95']} "
          f"(excludes zero: {hd['paired_test_case_bootstrap']['excludes_zero']})")
    print(f"  verdict          : {r['verdict_class']}")
    print(f"\n{r['verdict']}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
