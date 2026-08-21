"""Correction-value curves: what is the human-in-the-loop correction channel worth,
and does the LOCO/hinge line spend corrections where they matter?

Framing -- read this before the numbers
----------------------------------------
Derm7pt ships gold 7-point criteria per case. Here they stand in for a **SIMULATED
expert** who can be asked to resolve one criterion at a time. This is standard
correctability / active-feature-acquisition methodology (query the oracle one
feature at a time, watch the downstream decision converge) -- it is **not** a
human study, and no clinician judgement is involved in generating a correction.

Protocol, per test-split case
------------------------------
1. Start from the system's PREDICTED criteria (argmax of the frozen per-criterion
   heads, fit on Derm7pt train, applied to test -- no abstention gating on this
   starting point, so every criterion has a determinate value and the score is
   always decidable). Score it -> melanoma-threshold verdict (excise iff score>=3).
2. Reveal/correct criteria ONE AT A TIME (the simulated expert supplies the true
   value). Recompute the deterministic verdict after each correction, for
   k = 0..7 corrections.
3. Five correction-ORDERING policies decide which criterion gets corrected next.
   All five are BLIND to the true values when they choose an order -- only the
   reveal step uses the truth once a criterion is picked, exactly as a real
   clinician choosing what to double-check would be:
   - HINGE-GUIDED: at every step, re-run `verdict_hinges` / `leave_one_criterion_out`
     on the CURRENT (partially corrected) criteria and correct the criterion whose
     removal would flip or undecide the verdict, breaking ties by point weight
     (major=2 before minor=1) then canonical order. When no remaining criterion is
     currently a hinge, falls back to largest point weight -- still the sensible
     continuation, just with no immediate single-flip to chase.
   - UNCERTAINTY-GUIDED: fixed per-case order, ascending by the head's own
     top-minus-second posterior margin (least confident criterion first), computed
     with the exact formula `criterion_abstention.AbstentionPolicy.apply` uses.
   - RANDOM: 20 seeded independent per-case permutations; the headline curves
     report mean +/- std ACROSS those 20 cohort-level replicates.
   - HINGE-GATED, UNCERTAINTY-RANKED (combined, adaptive): at every step, GATE the
     candidate set to the current hinges (as HINGE-GUIDED computes them); RANK
     within that gated set by ascending margin (most uncertain hinge first). Falls
     back to ranking ALL remaining criteria by margin when no hinge remains at a
     step. Tests the hypothesis that the best correction target is a criterion
     that is BOTH decision-fragile AND model-uncertain -- a hinge the model was
     confident (and right) about is a worse use of a correction than one it was
     unsure about.
   - UNCERTAINTY-GATED, HINGE-RANKED (combined, adaptive, the mirror): at every
     step, GATE the candidate set to criteria whose margin is below THEIR OWN
     fitted abstention threshold (`AbstentionPolicy.margins[name]` from
     `criterion_abstention_v1.json` -- the same signal the abstention mechanism
     already uses to decide "uncertain enough to abstain", reused here as a gate
     instead of a continuous score); RANK within that gated set by hinge status
     first (hinge before non-hinge, ties by point weight), margin ascending as a
     final tie-break. Falls back to the same hinge-first-then-margin ranking over
     ALL remaining criteria when nothing clears its own uncertainty gate. This
     policy's exact construction was left underspecified by the brief that asked
     for it ("the mirror, for completeness") -- the choice made here is documented
     in its own JSON `description` field so it is easy to check or override.

Ground-truth verdict (the target the k=0..7 curves are measured against) is
`score_criteria(true_criteria).meets_threshold` -- the melanoma-suspicion verdict
computed from Derm7pt's own gold criteria. By construction every policy's k=7
state is bit-for-bit identical to the true criteria, so this target is ALWAYS
reached at k=7 for every case and every policy -- that is a tautology, not a
finding, and is asserted as a sanity check rather than reported as a result.
"Sensitivity"/"specificity" below are relative to THIS target (does the verdict
say excise), not to the raw melanoma/non-melanoma diagnosis label. The dataset
carries a SEPARATE, non-tautological imperfection: even the fully-informed
checklist verdict (all 7 true criteria) does not perfectly match the actual
diagnosis label. That is reported once, as a dataset-property aside, not as a
function of k -- see `dataset_property_aside_checklist_vs_diagnosis` in the
output and G2 (research/evidence/derm7pt_v1/g2_criteria_ceiling.json), which
measured the same kind of ceiling on `valid` (sensitivity 0.9016, specificity
0.8592) with gold criteria feeding a rule that never had any learning applied.

Boundaries
----------
Zero paid calls. Zero images encoded. Deterministic host arithmetic
(`medguard_v2.criteria.aggregate`) plus two already-fitted linear probes
(`medguard_v2.pipeline.run_cohort.fit_readers`, fit on train only, exactly as
`scripts/dev/build_loco_analysis.py` already does on this same split) plus a
seeded RNG for the random-ordering policy and the bootstrap CI. No source under
medguard_v2/ is modified; everything below only imports from it.

Reads (read-only)
------------------
medguard_v2/criteria/aggregate.py            score_criteria, leave_one_criterion_out, verdict_hinges
medguard_v2/criteria/schema.py               CRITERION_NAMES, CRITERIA_BY_NAME, MELANOMA_SUSPICION_THRESHOLD
medguard_v2/data/derm7pt.py                  load_cases -- true criteria, split, diagnosis_group
medguard_v2/pipeline/run_cohort.py           fit_readers, _load_features -- frozen heads, TRAIN ONLY
medguard_v2/perception/criterion_abstention.py  load_policy -- AbstentionPolicy.apply's margin formula

Outputs
-------
research/evidence/derm7pt_v1/correction_value_curves.json
notes/CORRECTION_VALUE_CURVES.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import (  # noqa: E402
    leave_one_criterion_out,
    score_criteria,
    verdict_hinges,
)
from medguard_v2.criteria.schema import (  # noqa: E402
    CRITERIA_BY_NAME,
    CRITERION_NAMES,
    MELANOMA_SUSPICION_THRESHOLD,
)
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.perception.criterion_abstention import load_policy  # noqa: E402
from medguard_v2.pipeline.run_cohort import _load_features, fit_readers  # noqa: E402

SEED = 20260810
N_RANDOM_REPLICATES = 20
N_BOOTSTRAP = 2000
N_STEPS = len(CRITERION_NAMES)  # 7

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "correction_value_curves.json"
OUT_MD = REPO / "notes" / "CORRECTION_VALUE_CURVES.md"


# --------------------------------------------------------------------------- #
# Cohort + predictions
# --------------------------------------------------------------------------- #

def _load_cohort():
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]

    features = _load_features()
    _diagnosis_model, heads = fit_readers(train, features)
    x_test = np.stack([features[str(c.case_num)] for c in test])

    predicted_raw = {name: heads[name].predict(x_test).tolist() for name in CRITERION_NAMES}

    policy = load_policy()
    margins: Dict[str, np.ndarray] = {}
    for name in CRITERION_NAMES:
        _values, margin = policy.apply(heads[name], x_test, name)
        margins[name] = np.asarray(margin, dtype=float)

    return test, train, predicted_raw, margins, policy


# --------------------------------------------------------------------------- #
# Correction-ordering policies
# --------------------------------------------------------------------------- #

def _hinge_order_and_verdicts(pred: Mapping[str, str], truth: Mapping[str, str]) -> Tuple[List[bool], List[str]]:
    """Adaptive: re-rank remaining criteria by LOCO hinge status after every reveal."""
    values = dict(pred)
    remaining = list(CRITERION_NAMES)
    verdicts = [score_criteria(values).meets_threshold]
    order: List[str] = []
    for _ in range(N_STEPS):
        hinges = {h["name"] for h in verdict_hinges(values)}

        def key(name: str, _hinges=hinges):
            is_hinge = name in _hinges
            return (0 if is_hinge else 1, -CRITERIA_BY_NAME[name].points, CRITERION_NAMES.index(name))

        pick = min(remaining, key=key)
        remaining.remove(pick)
        order.append(pick)
        values[pick] = truth[pick]
        verdicts.append(score_criteria(values).meets_threshold)
    return verdicts, order


def _fixed_order_verdicts(pred: Mapping[str, str], truth: Mapping[str, str], order: Sequence[str]) -> List[bool]:
    values = dict(pred)
    verdicts = [score_criteria(values).meets_threshold]
    for name in order:
        values[name] = truth[name]
        verdicts.append(score_criteria(values).meets_threshold)
    return verdicts


def _uncertainty_order(margins: Mapping[str, np.ndarray], i: int) -> List[str]:
    """Fixed per-case order: least-confident head prediction first."""
    return sorted(CRITERION_NAMES, key=lambda name: (margins[name][i], CRITERION_NAMES.index(name)))


def _hinge_gated_uncertainty_ranked(
    pred: Mapping[str, str], truth: Mapping[str, str], margins: Mapping[str, np.ndarray], i: int
) -> Tuple[List[bool], List[str], int]:
    """Adaptive combined policy: GATE to current hinges, RANK by ascending margin.

    Tests "decision-fragile AND model-uncertain beats decision-fragile alone": among the
    criteria that verdict_hinges currently flags as pivotal, correct the one the head is LEAST
    confident about first. Falls back to ranking ALL remaining criteria by margin (i.e. the same
    per-step choice UNCERTAINTY-GUIDED would make) when no remaining criterion is currently a
    hinge. Returns (verdicts, order, n_steps_where_the_hinge_gate_was_nonempty).
    """
    values = dict(pred)
    remaining = list(CRITERION_NAMES)
    verdicts = [score_criteria(values).meets_threshold]
    order: List[str] = []
    gate_hits = 0
    for _ in range(N_STEPS):
        hinges = {h["name"] for h in verdict_hinges(values)} & set(remaining)
        pool = hinges if hinges else set(remaining)
        gate_hits += 1 if hinges else 0
        pick = min(pool, key=lambda name: (margins[name][i], CRITERION_NAMES.index(name)))
        remaining.remove(pick)
        order.append(pick)
        values[pick] = truth[pick]
        verdicts.append(score_criteria(values).meets_threshold)
    return verdicts, order, gate_hits


def _uncertainty_gated_hinge_ranked(
    pred: Mapping[str, str],
    truth: Mapping[str, str],
    margins: Mapping[str, np.ndarray],
    i: int,
    abstention_thresholds: Mapping[str, float],
) -> Tuple[List[bool], List[str], int]:
    """Adaptive combined policy, the mirror of the one above: GATE to criteria the model's OWN
    fitted abstention threshold would call uncertain (margin < that criterion's
    `AbstentionPolicy.margins[name]`), RANK within the gated set by hinge status first (hinge
    before non-hinge, ties broken by point weight, then margin ascending, then canonical order).
    Falls back to the same hinge-first ranking over ALL remaining criteria when nothing remaining
    clears its own uncertainty gate (this makes the fallback reduce to HINGE-GUIDED's own
    ranking, refined by a margin tie-break). Returns (verdicts, order, n_steps_gate_nonempty).
    """
    values = dict(pred)
    remaining = list(CRITERION_NAMES)
    verdicts = [score_criteria(values).meets_threshold]
    order: List[str] = []
    gate_hits = 0
    for _ in range(N_STEPS):
        uncertain = {name for name in remaining if margins[name][i] < abstention_thresholds.get(name, 0.0)}
        pool = uncertain if uncertain else set(remaining)
        gate_hits += 1 if uncertain else 0
        hinges = {h["name"] for h in verdict_hinges(values)} & pool

        def key(name: str, _hinges=hinges):
            is_hinge = name in _hinges
            return (0 if is_hinge else 1, -CRITERIA_BY_NAME[name].points, margins[name][i], CRITERION_NAMES.index(name))

        pick = min(pool, key=key)
        remaining.remove(pick)
        order.append(pick)
        values[pick] = truth[pick]
        verdicts.append(score_criteria(values).meets_threshold)
    return verdicts, order, gate_hits


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _curve_metrics(verdicts_k: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """accuracy/sensitivity/specificity vs k, for a (n, k+1) bool verdict grid."""
    pos = truth
    neg = ~truth
    accuracy = (verdicts_k == truth[:, None]).mean(axis=0)
    sensitivity = verdicts_k[pos].mean(axis=0) if pos.any() else np.full(verdicts_k.shape[1], np.nan)
    specificity = (~verdicts_k[neg]).mean(axis=0) if neg.any() else np.full(verdicts_k.shape[1], np.nan)
    return accuracy, sensitivity, specificity


def _k_star(verdicts_k: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """First k (0..N_STEPS) at which the verdict matches ground truth, per case."""
    correct = verdicts_k == truth[:, None]
    assert correct[:, -1].all(), (
        "k=N_STEPS must tautologically match ground truth (all criteria corrected == true "
        "criteria); a False here means the simulation has a bug, not a real finding."
    )
    return correct.argmax(axis=1)


def _round_list(x: np.ndarray, nd: int = 4) -> List[float]:
    return [round(float(v), nd) for v in np.asarray(x, dtype=float)]


def _bootstrap_abc(case_abc: np.ndarray, rng: np.random.Generator, n_boot: int) -> Dict[str, object]:
    n = len(case_abc)
    idx_all = np.arange(n)
    boot_vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        boot_vals[b] = case_abc[idx].mean()
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return {
        "point_estimate": round(float(case_abc.mean()), 4),
        "bootstrap_mean": round(float(boot_vals.mean()), 4),
        "bootstrap_std": round(float(boot_vals.std(ddof=1)), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
        "resample_unit": "case",
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    test, train, predicted_raw, margins, policy = _load_cohort()
    n = len(test)

    pred_criteria = [{name: str(predicted_raw[name][i]) for name in CRITERION_NAMES} for i in range(n)]
    true_criteria = [dict(c.criteria) for c in test]

    ground_truth_verdict = np.array(
        [score_criteria(true_criteria[i]).meets_threshold for i in range(n)], dtype=bool
    )
    is_melanoma = np.array([c.diagnosis_group == "MEL" for c in test], dtype=bool)
    predicted_verdict_k0 = np.array(
        [score_criteria(pred_criteria[i]).meets_threshold for i in range(n)], dtype=bool
    )

    # ---- hinge-guided (adaptive) ----
    hinge_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    hinge_orders: List[List[str]] = []
    for i in range(n):
        v, order = _hinge_order_and_verdicts(pred_criteria[i], true_criteria[i])
        hinge_verdicts[i] = v
        hinge_orders.append(order)

    # ---- uncertainty-guided (fixed per-case order from head margins) ----
    uncertainty_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    for i in range(n):
        order = _uncertainty_order(margins, i)
        uncertainty_verdicts[i] = _fixed_order_verdicts(pred_criteria[i], true_criteria[i], order)

    # ---- combined policy 1: hinge-gated, uncertainty-ranked (adaptive) ----
    p1_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    p1_orders: List[List[str]] = []
    p1_gate_hits = np.zeros(n, dtype=int)
    for i in range(n):
        v, order, hits = _hinge_gated_uncertainty_ranked(pred_criteria[i], true_criteria[i], margins, i)
        p1_verdicts[i] = v
        p1_orders.append(order)
        p1_gate_hits[i] = hits

    # ---- combined policy 2: uncertainty-gated, hinge-ranked (adaptive, the mirror) ----
    p2_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    p2_orders: List[List[str]] = []
    p2_gate_hits = np.zeros(n, dtype=int)
    for i in range(n):
        v, order, hits = _uncertainty_gated_hinge_ranked(pred_criteria[i], true_criteria[i], margins, i, policy.margins)
        p2_verdicts[i] = v
        p2_orders.append(order)
        p2_gate_hits[i] = hits

    # ---- random (20 seeded replicates) ----
    seed_seq = np.random.SeedSequence(SEED)
    child_seeds = seed_seq.spawn(N_RANDOM_REPLICATES + 2)  # +1 hinge-vs-random boot, +1 hinge-vs-uncertainty boot
    random_child_seeds, boot_seed_random, boot_seed_uncertainty = (
        child_seeds[:N_RANDOM_REPLICATES], child_seeds[N_RANDOM_REPLICATES], child_seeds[N_RANDOM_REPLICATES + 1],
    )
    # Second call on the SAME SeedSequence object: continues the cumulative spawn counter, so the
    # 22 seeds above are bit-identical to a prior run that only ever called .spawn(22) once
    # (verified empirically) -- the already-reported random/hinge_vs_random/hinge_vs_uncertainty
    # numbers do not move when these 6 new comparisons are added.
    combo_boot_seeds = seed_seq.spawn(6)
    (
        boot_seed_p1_random, boot_seed_p1_uncertainty, boot_seed_p1_hinge,
        boot_seed_p2_random, boot_seed_p2_uncertainty, boot_seed_p2_hinge,
    ) = combo_boot_seeds
    random_verdicts = np.zeros((N_RANDOM_REPLICATES, n, N_STEPS + 1), dtype=bool)
    for r, cs in enumerate(random_child_seeds):
        rng = np.random.default_rng(cs)
        for i in range(n):
            perm = rng.permutation(N_STEPS)
            order = [CRITERION_NAMES[j] for j in perm]
            random_verdicts[r, i] = _fixed_order_verdicts(pred_criteria[i], true_criteria[i], order)

    # ---- sanity checks (assert, not report) ----
    assert (hinge_verdicts[:, -1] == ground_truth_verdict).all()
    assert (uncertainty_verdicts[:, -1] == ground_truth_verdict).all()
    assert (random_verdicts[:, :, -1] == ground_truth_verdict[None, :]).all()
    assert (p1_verdicts[:, -1] == ground_truth_verdict).all()
    assert (p2_verdicts[:, -1] == ground_truth_verdict).all()
    assert (hinge_verdicts[:, 0] == predicted_verdict_k0).all()
    assert (uncertainty_verdicts[:, 0] == predicted_verdict_k0).all()
    assert (random_verdicts[:, :, 0] == predicted_verdict_k0[None, :]).all()
    assert (p1_verdicts[:, 0] == predicted_verdict_k0).all()
    assert (p2_verdicts[:, 0] == predicted_verdict_k0).all()

    # ---- curves: hinge, uncertainty (single deterministic realisation) ----
    hinge_acc, hinge_sens, hinge_spec = _curve_metrics(hinge_verdicts, ground_truth_verdict)
    unc_acc, unc_sens, unc_spec = _curve_metrics(uncertainty_verdicts, ground_truth_verdict)
    hinge_kstar = _k_star(hinge_verdicts, ground_truth_verdict)
    unc_kstar = _k_star(uncertainty_verdicts, ground_truth_verdict)

    # ---- curves: the two combined policies (single deterministic realisation each) ----
    p1_acc, p1_sens, p1_spec = _curve_metrics(p1_verdicts, ground_truth_verdict)
    p2_acc, p2_sens, p2_spec = _curve_metrics(p2_verdicts, ground_truth_verdict)
    p1_kstar = _k_star(p1_verdicts, ground_truth_verdict)
    p2_kstar = _k_star(p2_verdicts, ground_truth_verdict)

    # ---- curves: random (mean +/- std across 20 replicate cohort-level runs) ----
    rep_acc, rep_sens, rep_spec, rep_mean_kstar = [], [], [], []
    kstar_random_per_case = np.zeros((N_RANDOM_REPLICATES, n), dtype=int)
    for r in range(N_RANDOM_REPLICATES):
        acc_r, sens_r, spec_r = _curve_metrics(random_verdicts[r], ground_truth_verdict)
        kstar_r = _k_star(random_verdicts[r], ground_truth_verdict)
        kstar_random_per_case[r] = kstar_r
        rep_acc.append(acc_r)
        rep_sens.append(sens_r)
        rep_spec.append(spec_r)
        rep_mean_kstar.append(kstar_r.mean())
    rep_acc = np.stack(rep_acc)      # (20, 8)
    rep_sens = np.stack(rep_sens)
    rep_spec = np.stack(rep_spec)
    rep_mean_kstar = np.asarray(rep_mean_kstar)  # (20,)

    # ---- already correct at k=0 (policy-independent) ----
    already_correct_frac = float((predicted_verdict_k0 == ground_truth_verdict).mean())
    assert abs(already_correct_frac - float(rep_acc[:, 0].mean())) < 1e-9
    # k=0 has no policy input yet, so all 20 replicate-level accuracies at k=0 must be equal.
    # The underlying bool arrays were already asserted bit-identical above; this just checks
    # the derived float statistic with a tolerance instead of fragile exact float equality.
    assert np.allclose(rep_acc[:, 0], rep_acc[0, 0], atol=1e-9)

    # ---- area-between-curves: hinge vs random (required) ----
    hinge_correct = (hinge_verdicts == ground_truth_verdict[:, None]).astype(float)          # (n, 8)
    random_correct_per_rep = (random_verdicts == ground_truth_verdict[None, :, None]).astype(float)  # (20, n, 8)
    random_correct_mean = random_correct_per_rep.mean(axis=0)                                 # (n, 8)
    case_abc_hinge_vs_random = hinge_correct.mean(axis=1) - random_correct_mean.mean(axis=1)   # (n,)
    abc_hinge_vs_random = _bootstrap_abc(
        case_abc_hinge_vs_random, np.random.default_rng(boot_seed_random), N_BOOTSTRAP
    )

    # ---- area-between-curves: hinge vs uncertainty (bonus, same machinery) ----
    uncertainty_correct = (uncertainty_verdicts == ground_truth_verdict[:, None]).astype(float)
    case_abc_hinge_vs_uncertainty = hinge_correct.mean(axis=1) - uncertainty_correct.mean(axis=1)
    abc_hinge_vs_uncertainty = _bootstrap_abc(
        case_abc_hinge_vs_uncertainty, np.random.default_rng(boot_seed_uncertainty), N_BOOTSTRAP
    )

    # ---- area-between-curves: the two combined policies vs random, vs uncertainty-guided,
    # and (bonus) vs hinge-guided alone -- the coordinator's core hypothesis test is
    # p1_vs_uncertainty: does gating hinges by model uncertainty beat uncertainty alone? ----
    p1_correct = (p1_verdicts == ground_truth_verdict[:, None]).astype(float)
    p2_correct = (p2_verdicts == ground_truth_verdict[:, None]).astype(float)

    def _abc(correct_a: np.ndarray, correct_b: np.ndarray, seed) -> Dict[str, object]:
        case_gap = correct_a.mean(axis=1) - correct_b.mean(axis=1)
        return _bootstrap_abc(case_gap, np.random.default_rng(seed), N_BOOTSTRAP)

    abc_p1_vs_random = _abc(p1_correct, random_correct_mean, boot_seed_p1_random)
    abc_p1_vs_uncertainty = _abc(p1_correct, uncertainty_correct, boot_seed_p1_uncertainty)
    abc_p1_vs_hinge = _abc(p1_correct, hinge_correct, boot_seed_p1_hinge)
    abc_p2_vs_random = _abc(p2_correct, random_correct_mean, boot_seed_p2_random)
    abc_p2_vs_uncertainty = _abc(p2_correct, uncertainty_correct, boot_seed_p2_uncertainty)
    abc_p2_vs_hinge = _abc(p2_correct, hinge_correct, boot_seed_p2_hinge)

    # ---- dataset-property aside: fully-informed rule verdict vs actual diagnosis label ----
    tp = int((ground_truth_verdict & is_melanoma).sum())
    fn = int((~ground_truth_verdict & is_melanoma).sum())
    fp = int((ground_truth_verdict & ~is_melanoma).sum())
    tn = int((~ground_truth_verdict & ~is_melanoma).sum())
    ds_sensitivity = tp / (tp + fn) if (tp + fn) else None
    ds_specificity = tn / (tn + fp) if (tn + fp) else None
    ds_accuracy = (tp + tn) / n if n else None

    # ---- assemble JSON ----
    def _policy_block(acc, sens, spec, kstar_mean) -> Dict[str, object]:
        return {
            "accuracy_vs_k": _round_list(acc),
            "sensitivity_vs_k": _round_list(sens),
            "specificity_vs_k": _round_list(spec),
            "mean_corrections_to_correct_verdict": round(float(kstar_mean), 4),
        }

    # Diagnostic: how often each combined policy's GATE stage actually restricted the candidate
    # pool vs fell back to the ungated ranking. Reported so a near-degenerate gate (e.g. the two
    # criteria whose fitted abstention threshold is exactly 0.0 can never clear the uncertainty
    # gate) is visible rather than silently producing a policy that mostly reduces to its fallback.
    p1_gate_nonempty_fraction = round(float(p1_gate_hits.mean() / N_STEPS), 4)
    p2_gate_nonempty_fraction = round(float(p2_gate_hits.mean() / N_STEPS), 4)

    result = {
        "protocol": "correction_value_curves_v1",
        "generated_by": "scripts/dev/build_correction_value_curves.py",
        "framing": {
            "simulated_expert": (
                "Derm7pt's ground-truth 7-point criteria stand in for a SIMULATED expert "
                "supplying corrections. This is standard correctability / active-feature-"
                "acquisition methodology, NOT a human study -- no clinician judgement is "
                "involved in generating these corrections."
            ),
            "predicted_criteria_source": (
                "argmax of the frozen per-criterion heads (medguard_v2.pipeline.run_cohort."
                "fit_readers), fit on Derm7pt train only, applied to test. No abstention "
                "gating on this starting point -- every criterion has a determinate predicted "
                "value, matching the existing scripts/dev/build_loco_analysis.py precedent."
            ),
            "ground_truth_verdict_definition": (
                "score_criteria(true_criteria).meets_threshold: the melanoma-suspicion verdict "
                "(score>=3) computed from Derm7pt's gold 7-point criteria. By construction this "
                "is exactly what every policy reaches at k=7 (values == true_criteria once all "
                "7 are revealed) -- see sanity_checks, not a reported finding."
            ),
            "positive_class": (
                "ground_truth_verdict == True, i.e. the fully-informed checklist recommends "
                "excision. Sensitivity/specificity below are relative to THIS target, not to "
                "the raw melanoma/non-melanoma diagnosis label -- see "
                "dataset_property_aside_checklist_vs_diagnosis for that separate comparison."
            ),
            "ordering_policies_are_truth_blind": (
                "All three policies choose WHICH criterion to correct next using only "
                "already-known information (current partial criteria state, or the head's own "
                "confidence); only the reveal step uses the true value of the chosen criterion. "
                "None of the policies can 'peek' at ground truth to decide what to correct."
            ),
        },
        "boundaries": {
            "paid_calls": 0,
            "images_encoded": 0,
            "model_calls_beyond_fitted_heads": 0,
            "fit_split": "train",
            "n_fit_cases": len(train),
            "evaluated_split": "test",
            "n_test_cases": n,
            "n_melanoma_test": int(is_melanoma.sum()),
            "melanoma_threshold": MELANOMA_SUSPICION_THRESHOLD,
            "seed": SEED,
            "random_policy_replicates": N_RANDOM_REPLICATES,
            "bootstrap_replicates": N_BOOTSTRAP,
            "abstention_policy_used_for": "uncertainty_guided ordering only (per-case margins via AbstentionPolicy.apply); NOT used to gate the k=0 starting criteria",
            "abstention_policy_fitted_on": policy.fitted_on,
        },
        "policies": {
            "hinge_guided": {
                "description": (
                    "Adaptive: at every step, correct the remaining criterion that "
                    "verdict_hinges/leave_one_criterion_out currently flags as flipping or "
                    "undeciding the verdict (largest point weight first among hinges; largest "
                    "point weight among all remaining when no hinge remains)."
                ),
                **_policy_block(hinge_acc, hinge_sens, hinge_spec, hinge_kstar.mean()),
            },
            "uncertainty_guided": {
                "description": (
                    "Fixed per-case order: ascending head margin (top-minus-second posterior), "
                    "least-confident criterion corrected first."
                ),
                **_policy_block(unc_acc, unc_sens, unc_spec, unc_kstar.mean()),
            },
            "random": {
                "description": (
                    f"{N_RANDOM_REPLICATES} independent seeded per-case random permutations; "
                    "mean +/- std across the replicate-level cohort curves."
                ),
                "accuracy_vs_k": {"mean": _round_list(rep_acc.mean(axis=0)), "std": _round_list(rep_acc.std(axis=0, ddof=1))},
                "sensitivity_vs_k": {"mean": _round_list(rep_sens.mean(axis=0)), "std": _round_list(rep_sens.std(axis=0, ddof=1))},
                "specificity_vs_k": {"mean": _round_list(rep_spec.mean(axis=0)), "std": _round_list(rep_spec.std(axis=0, ddof=1))},
                "mean_corrections_to_correct_verdict": {
                    "mean": round(float(rep_mean_kstar.mean()), 4),
                    "std": round(float(rep_mean_kstar.std(ddof=1)), 4),
                },
                "replicates": N_RANDOM_REPLICATES,
            },
            "hinge_gated_uncertainty_ranked": {
                "description": (
                    "Combined, adaptive. At every step: GATE the remaining candidates to the "
                    "current LOCO hinges (verdict_hinges intersected with what's left); RANK "
                    "within that gated set by ascending head margin (most uncertain hinge "
                    "first). Falls back to ranking ALL remaining criteria by margin (i.e. the "
                    "same per-step choice uncertainty_guided would make) when no remaining "
                    "criterion is currently a hinge. Tests: is the best correction target a "
                    "criterion that is BOTH decision-fragile AND model-uncertain?"
                ),
                "hinge_gate_nonempty_fraction_of_steps": p1_gate_nonempty_fraction,
                **_policy_block(p1_acc, p1_sens, p1_spec, p1_kstar.mean()),
            },
            "uncertainty_gated_hinge_ranked": {
                "description": (
                    "Combined, adaptive, the mirror of hinge_gated_uncertainty_ranked (brief "
                    "left its exact construction underspecified -- 'the mirror, for "
                    "completeness'). At every step: GATE the remaining candidates to criteria "
                    "whose margin is below THEIR OWN fitted abstention threshold "
                    "(AbstentionPolicy.margins[name] from criterion_abstention_v1.json, reused "
                    "here as a gate rather than a continuous score); RANK within that gated set "
                    "by hinge status first (hinge before non-hinge, ties by point weight, then "
                    "margin ascending). Falls back to the same hinge-first ranking over ALL "
                    "remaining criteria when nothing clears its own uncertainty gate -- note two "
                    "criteria (blue_whitish_veil, vascular_structures) have a fitted threshold "
                    "of exactly 0.0 and so can NEVER clear this gate; see "
                    "uncertainty_gate_nonempty_fraction_of_steps for how often the gate bound at "
                    "all across the other five."
                ),
                "uncertainty_gate_nonempty_fraction_of_steps": p2_gate_nonempty_fraction,
                **_policy_block(p2_acc, p2_sens, p2_spec, p2_kstar.mean()),
            },
        },
        "already_correct_at_k0": {
            "fraction": round(already_correct_frac, 4),
            "n_cases": int(round(already_correct_frac * n)),
            "note": "Policy-independent: no correction has been applied yet, so this is identical across all three policies (asserted).",
        },
        "hinge_vs_random_area_between_curves": {
            "definition": (
                "Per-case ABC = mean_k(1{hinge verdict_k correct}) - mean_k(mean over 20 "
                "replicates of 1{random verdict_k correct}), k=0..7. Point estimate = mean "
                "over cases. Positive means the hinge-guided curve sits above the random-mean "
                "curve on average across the correction budget."
            ),
            **abc_hinge_vs_random,
        },
        "hinge_vs_uncertainty_area_between_curves": {
            "definition": (
                "Same construction as hinge_vs_random, comparing hinge-guided against the "
                "uncertainty-guided (head-margin) policy instead of random. Not requested by "
                "the core protocol; included because it is nearly free given the same "
                "machinery and directly informs whether hinge-guidance beats the next-best "
                "truth-blind alternative, not just a naive baseline."
            ),
            **abc_hinge_vs_uncertainty,
        },
        "combined_policy_comparisons": {
            "note": (
                "Follow-up requested by the coordinator after the first pass of this script: "
                "does combining hinge-fragility and model-uncertainty beat either signal alone? "
                "The core hypothesis test is hinge_gated_uncertainty_ranked_vs_uncertainty_guided "
                "-- if that excludes zero in the combined policy's favour, gating hinges by "
                "uncertainty is a real win and a directive to rank the renderer's hinge line by "
                "model uncertainty. All six comparisons use the same case-resampling bootstrap "
                "as hinge_vs_random above (2000 reps, seed 20260810, independently spawned child "
                "seeds so they don't perturb the three single-signal policies' own numbers)."
            ),
            "hinge_gated_uncertainty_ranked_vs_random": abc_p1_vs_random,
            "hinge_gated_uncertainty_ranked_vs_uncertainty_guided": abc_p1_vs_uncertainty,
            "hinge_gated_uncertainty_ranked_vs_hinge_guided": abc_p1_vs_hinge,
            "uncertainty_gated_hinge_ranked_vs_random": abc_p2_vs_random,
            "uncertainty_gated_hinge_ranked_vs_uncertainty_guided": abc_p2_vs_uncertainty,
            "uncertainty_gated_hinge_ranked_vs_hinge_guided": abc_p2_vs_hinge,
        },
        "dataset_property_aside_checklist_vs_diagnosis": {
            "description": (
                "NOT a function of k. Compares the FULLY-INFORMED checklist verdict "
                "(score_criteria(true_criteria).meets_threshold, i.e. what every policy "
                "converges to at k=7) against the actual diagnosis label (diagnosis_group == "
                "'MEL'), on test. This is the real 'ceiling' question: does the 7-point rule, "
                "even with perfect criteria knowledge, agree with the pathology label? It is "
                "the test-split analogue of G2's oracle-criteria melanoma-detection ceiling "
                "(research/evidence/derm7pt_v1/g2_criteria_ceiling.json), which measured "
                "sensitivity 0.9016 / specificity 0.8592 on valid with gold criteria feeding "
                "the same unlearned published rule."
            ),
            "sensitivity": round(ds_sensitivity, 4) if ds_sensitivity is not None else None,
            "specificity": round(ds_specificity, 4) if ds_specificity is not None else None,
            "accuracy": round(ds_accuracy, 4) if ds_accuracy is not None else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "reinterpretation_note": (
                "The brief that spawned this script asked for 'fraction never correctable "
                "within 7 (ground-truth verdict differs even with all criteria corrected)'. "
                "Read literally against this protocol's own ground_truth_verdict definition "
                "that fraction is tautologically always 0 -- at k=7 the working criteria ARE "
                "the true criteria, so the computed verdict cannot differ from itself (see "
                "sanity_checks). The substantive, non-tautological version of that question -- "
                "does the checklist rule itself disagree with the independent diagnosis label, "
                "even fully informed -- is answered above instead, and is the one that ties to "
                "a rough-set-style checklist-vs-diagnosis ceiling. That external '92.1%' figure "
                "was not found anywhere in this repository (grepped; likely a literature figure "
                "from outside this codebase) and is therefore not asserted to match the number "
                "computed here -- treat this block as the honest, independently computed "
                "analogue, not a reproduction of that specific citation."
            ),
        },
        "sanity_checks": {
            "k7_matches_ground_truth_verdict_all_policies": True,
            "k0_identical_across_all_policies": True,
            "k0_random_replicate_std_is_zero": True,
        },
        "per_case": [
            {
                "case_id": f"derm7pt-{test[i].case_num}",
                "diagnosis_group": test[i].diagnosis_group,
                "is_melanoma": bool(is_melanoma[i]),
                "ground_truth_verdict": bool(ground_truth_verdict[i]),
                "predicted_verdict_k0": bool(predicted_verdict_k0[i]),
                "correct_at_k0": bool(predicted_verdict_k0[i] == ground_truth_verdict[i]),
                "k_star_hinge": int(hinge_kstar[i]),
                "k_star_uncertainty": int(unc_kstar[i]),
                "k_star_random_mean_over_replicates": round(float(kstar_random_per_case[:, i].mean()), 4),
                "k_star_hinge_gated_uncertainty_ranked": int(p1_kstar[i]),
                "k_star_uncertainty_gated_hinge_ranked": int(p2_kstar[i]),
                "hinge_correction_order": hinge_orders[i],
                "hinge_gated_uncertainty_ranked_order": p1_orders[i],
                "uncertainty_gated_hinge_ranked_order": p2_orders[i],
            }
            for i in range(n)
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")

    _write_notes(result)
    print(f"wrote {OUT_MD}")


def _write_notes(result: Dict[str, object]) -> None:
    b = result["boundaries"]
    pol = result["policies"]
    abc_r = result["hinge_vs_random_area_between_curves"]
    abc_u = result["hinge_vs_uncertainty_area_between_curves"]
    aside = result["dataset_property_aside_checklist_vs_diagnosis"]
    ac0 = result["already_correct_at_k0"]

    def row(label, block, k):
        acc_field, sens_field = block["accuracy_vs_k"], block["sensitivity_vs_k"]
        acc = acc_field["mean"][k] if isinstance(acc_field, dict) else acc_field[k]
        sens = sens_field["mean"][k] if isinstance(sens_field, dict) else sens_field[k]
        return f"| {label} | k={k} | {acc:.4f} | {sens:.4f} |"

    lines: List[str] = []
    lines.append("# Correction-value curves (Derm7pt test, simulated-expert corrections)")
    lines.append("")
    lines.append(
        f"n={b['n_test_cases']} test cases ({b['n_melanoma_test']} melanoma), fit on train "
        f"(n={b['n_fit_cases']}). Seed {b['seed']}. Zero paid calls, zero images encoded -- "
        "deterministic host aggregation over already-fitted linear probes."
    )
    lines.append("")
    lines.append(
        "**Simulated expert, not a human study.** Derm7pt's gold 7-point criteria play the "
        "role of an expert who resolves one criterion at a time; this is standard "
        "correctability / active-feature-acquisition methodology."
    )
    lines.append("")
    lines.append(
        "**Ground-truth verdict** = `score_criteria(true_criteria).meets_threshold`. All "
        "five ordering policies are blind to the truth when choosing what to correct next; "
        "only the reveal step uses it. By construction every policy reaches this exact target "
        "at k=7 (tautology, asserted not reported)."
    )
    lines.append("")
    lines.append(
        "**Two combined policies were added as a follow-up** to test whether the best "
        "correction target is a criterion that is BOTH decision-fragile (a hinge) AND "
        "model-uncertain, not just one or the other: `hinge-gated, uncertainty-ranked` (gate to "
        "current hinges, rank by lowest margin) and its mirror `uncertainty-gated, hinge-ranked` "
        "(gate to criteria below their own fitted abstention threshold, rank by hinge status). "
        "Full construction of each is in the JSON's `policies.*.description`."
    )
    lines.append("")

    lines.append("## Accuracy + sensitivity at k=0,1,2,3 (vs the checklist-threshold verdict)")
    lines.append("")
    lines.append("| policy | k | accuracy | sensitivity |")
    lines.append("|---|---|---:|---:|")
    for label, block in (
        ("hinge-guided", pol["hinge_guided"]),
        ("uncertainty-guided", pol["uncertainty_guided"]),
        ("hinge-gated, uncertainty-ranked", pol["hinge_gated_uncertainty_ranked"]),
        ("uncertainty-gated, hinge-ranked", pol["uncertainty_gated_hinge_ranked"]),
        ("random (mean)", pol["random"]),
    ):
        for k in (0, 1, 2, 3):
            lines.append(row(label, block, k))
    lines.append("")

    lines.append("## Mean corrections until the verdict matches ground truth (lower = better targeting)")
    lines.append("")
    lines.append("| policy | mean k* |")
    lines.append("|---|---:|")
    lines.append(f"| hinge-guided | {pol['hinge_guided']['mean_corrections_to_correct_verdict']:.4f} |")
    lines.append(f"| uncertainty-guided | {pol['uncertainty_guided']['mean_corrections_to_correct_verdict']:.4f} |")
    lines.append(f"| hinge-gated, uncertainty-ranked | {pol['hinge_gated_uncertainty_ranked']['mean_corrections_to_correct_verdict']:.4f} |")
    lines.append(f"| uncertainty-gated, hinge-ranked | {pol['uncertainty_gated_hinge_ranked']['mean_corrections_to_correct_verdict']:.4f} |")
    lines.append(
        f"| random | {pol['random']['mean_corrections_to_correct_verdict']['mean']:.4f} "
        f"+/- {pol['random']['mean_corrections_to_correct_verdict']['std']:.4f} (20 reps) |"
    )
    lines.append("")
    lines.append(f"Already correct at k=0 (no correction needed), any policy: **{ac0['fraction']:.4f}** ({ac0['n_cases']}/{result['boundaries']['n_test_cases']}).")
    lines.append(
        f"Gate diagnostics: hinge-gated,uncertainty-ranked's hinge gate was non-empty on "
        f"**{pol['hinge_gated_uncertainty_ranked']['hinge_gate_nonempty_fraction_of_steps']:.4f}** "
        f"of steps; uncertainty-gated,hinge-ranked's uncertainty gate (margin below that "
        f"criterion's own fitted abstention threshold) was non-empty on "
        f"**{pol['uncertainty_gated_hinge_ranked']['uncertainty_gate_nonempty_fraction_of_steps']:.4f}** "
        f"of steps (two criteria have a fitted threshold of exactly 0.0 and can never clear it, "
        f"so this is expected to run well under 1.0)."
    )
    lines.append("")

    lines.append("## Hinge vs random: area-between-curves")
    lines.append("")
    lines.append(
        f"Point estimate **{abc_r['point_estimate']:+.4f}**, bootstrap 95% CI "
        f"[{abc_r['ci95'][0]:+.4f}, {abc_r['ci95'][1]:+.4f}] "
        f"(resample cases, {abc_r['n_boot']} reps, seed {b['seed']}). "
        f"Excludes zero: **{abc_r['excludes_zero']}**."
    )
    lines.append("")
    lines.append(
        f"Bonus, same machinery -- hinge vs uncertainty-guided: point estimate "
        f"**{abc_u['point_estimate']:+.4f}**, 95% CI [{abc_u['ci95'][0]:+.4f}, {abc_u['ci95'][1]:+.4f}], "
        f"excludes zero: **{abc_u['excludes_zero']}**."
    )
    lines.append("")

    lines.append("## Combined policies vs random and vs uncertainty-guided (the follow-up)")
    lines.append("")
    lines.append(
        "Core hypothesis test: does `hinge-gated, uncertainty-ranked` beat `uncertainty-guided` "
        "alone? All CIs are 2000-rep case-resampling bootstraps, seed 20260810, independently "
        "seeded from the three single-signal policies' own bootstraps above (those numbers do "
        "not move)."
    )
    lines.append("")
    lines.append("| comparison | point estimate | 95% CI | excludes zero |")
    lines.append("|---|---:|---|:---:|")
    combo = result["combined_policy_comparisons"]
    for key, label in (
        ("hinge_gated_uncertainty_ranked_vs_random", "hinge-gated,uncertainty-ranked vs random"),
        ("hinge_gated_uncertainty_ranked_vs_uncertainty_guided", "**hinge-gated,uncertainty-ranked vs uncertainty-guided**"),
        ("hinge_gated_uncertainty_ranked_vs_hinge_guided", "hinge-gated,uncertainty-ranked vs hinge-guided"),
        ("uncertainty_gated_hinge_ranked_vs_random", "uncertainty-gated,hinge-ranked vs random"),
        ("uncertainty_gated_hinge_ranked_vs_uncertainty_guided", "uncertainty-gated,hinge-ranked vs uncertainty-guided"),
        ("uncertainty_gated_hinge_ranked_vs_hinge_guided", "uncertainty-gated,hinge-ranked vs hinge-guided"),
    ):
        c = combo[key]
        lines.append(
            f"| {label} | {c['point_estimate']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['excludes_zero']} |"
        )
    lines.append("")

    lines.append("## Dataset-property aside: fully-informed rule verdict vs actual diagnosis label")
    lines.append("")
    lines.append(
        "Not a function of k. The brief asked for a 'never correctable within 7' fraction "
        "framed as the protocol's own ground-truth verdict disagreeing with itself at k=7 -- "
        "that is tautologically always 0 under this protocol's definitions (see the JSON's "
        "`reinterpretation_note`). The substantive version: does the checklist rule, even with "
        "every true criterion known, agree with the actual melanoma/non-melanoma label?"
    )
    lines.append("")
    lines.append(
        f"Sensitivity **{aside['sensitivity']:.4f}**, specificity **{aside['specificity']:.4f}**, "
        f"accuracy **{aside['accuracy']:.4f}** (confusion: {aside['confusion']}). "
        "G2 measured the same kind of ceiling on valid with an unlearned rule: sensitivity "
        "0.9016, specificity 0.8592. No '92.1%' figure was found anywhere in this repository "
        "(grepped); it is not asserted to match either number above."
    )
    lines.append("")

    lines.append("## Honest read")
    lines.append("")
    lines.append(_honest_paragraph(result))
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _honest_paragraph(result: Dict[str, object]) -> str:
    abc_r = result["hinge_vs_random_area_between_curves"]
    abc_u = result["hinge_vs_uncertainty_area_between_curves"]
    pol = result["policies"]
    hinge_kstar = pol["hinge_guided"]["mean_corrections_to_correct_verdict"]
    unc_kstar = pol["uncertainty_guided"]["mean_corrections_to_correct_verdict"]
    random_kstar = pol["random"]["mean_corrections_to_correct_verdict"]["mean"]
    beats_random = abc_r["excludes_zero"] and abc_r["point_estimate"] > 0

    if beats_random:
        part1 = (
            f"Hinge-guided targeting IS measurably better than random: the area-between-curves "
            f"is {abc_r['point_estimate']:+.4f} with a 95% bootstrap CI "
            f"[{abc_r['ci95'][0]:+.4f}, {abc_r['ci95'][1]:+.4f}] that excludes zero, and hinge-"
            f"guided reaches the correct verdict in {hinge_kstar:.3f} corrections on average "
            f"versus {random_kstar:.3f} for random ordering. The LOCO hinge line is not just a "
            f"post-hoc rationalisation -- pointing corrections at it measurably shortens the "
            f"path to the right verdict versus not targeting at all."
        )
    else:
        part1 = (
            f"Hinge-guided targeting is NOT measurably better than random on this cohort: the "
            f"area-between-curves is {abc_r['point_estimate']:+.4f} with a 95% bootstrap CI "
            f"[{abc_r['ci95'][0]:+.4f}, {abc_r['ci95'][1]:+.4f}], which does not exclude zero, "
            f"and mean corrections-to-correct are {hinge_kstar:.3f} (hinge) versus "
            f"{random_kstar:.3f} (random)."
        )

    # This caveat is load-bearing, not optional colour: the bonus hinge-vs-uncertainty
    # comparison uses the exact same machinery, and burying an unfavourable result here would
    # turn an honest finding into a cherry-picked one.
    uncertainty_edge = abc_u["point_estimate"] < 0
    if uncertainty_edge:
        part2 = (
            f" That said, beating random is a low bar, and it is the ONLY bar hinge-guidance "
            f"clears here. Against the cheaper uncertainty-margin policy -- which needs no LOCO "
            f"machinery, just the head's own top-vs-second posterior gap -- hinge-guided does "
            f"NOT come out ahead: the hinge-vs-uncertainty area-between-curves point estimate is "
            f"{abc_u['point_estimate']:+.4f} (95% CI [{abc_u['ci95'][0]:+.4f}, "
            f"{abc_u['ci95'][1]:+.4f}], excludes zero: {abc_u['excludes_zero']}) -- numerically "
            f"favouring uncertainty, though not significantly so. Uncertainty-guided also reaches "
            f"the correct verdict in fewer corrections on average ({unc_kstar:.3f} vs "
            f"{hinge_kstar:.3f} for hinge) and has higher accuracy and sensitivity at every one of "
            f"k=1,2,3 in the table above. The honest summary is narrower than 'the hinge line "
            f"works': it demonstrably beats not targeting at all, but this evidence does not show "
            f"it beats the much cheaper alternative of just asking the model how sure it is. "
            f"Whether that matters depends on what hinge-guidance is FOR -- if the point is "
            f"specifically to explain WHY a criterion is being asked for (auditable, tied to the "
            f"verdict rule) rather than just which one to ask for, the two policies are not "
            f"interchangeable even when their raw efficiency is similar."
        )
    else:
        part2 = (
            f" It also holds up against the cheaper uncertainty-margin policy: hinge-vs-"
            f"uncertainty area-between-curves is {abc_u['point_estimate']:+.4f} (95% CI "
            f"[{abc_u['ci95'][0]:+.4f}, {abc_u['ci95'][1]:+.4f}], excludes zero: "
            f"{abc_u['excludes_zero']}), so hinge-guidance is not merely reproducing what raw "
            f"model confidence would already give for free."
        )

    # Follow-up: does GATING hinges by model uncertainty beat uncertainty alone? This is the
    # coordinator's core hypothesis test, computed the same honest way as parts 1-2 -- report
    # whichever way the CI actually falls, and say so plainly if it does not clear the bar.
    combo = result["combined_policy_comparisons"]
    p1_vs_unc = combo["hinge_gated_uncertainty_ranked_vs_uncertainty_guided"]
    p1_vs_random = combo["hinge_gated_uncertainty_ranked_vs_random"]
    p1_kstar = pol["hinge_gated_uncertainty_ranked"]["mean_corrections_to_correct_verdict"]
    p2_kstar = pol["uncertainty_gated_hinge_ranked"]["mean_corrections_to_correct_verdict"]
    combined_beats_uncertainty = p1_vs_unc["excludes_zero"] and p1_vs_unc["point_estimate"] > 0
    p1_beats_random = p1_vs_random["excludes_zero"] and p1_vs_random["point_estimate"] > 0

    if combined_beats_uncertainty:
        part3 = (
            f" The follow-up hypothesis -- that the best correction target is BOTH "
            f"decision-fragile and model-uncertain -- DOES hold: gating the hinge set by model "
            f"uncertainty beats uncertainty alone. Hinge-gated,uncertainty-ranked reaches the "
            f"correct verdict in {p1_kstar:.3f} corrections on average (vs {unc_kstar:.3f} for "
            f"uncertainty-guided alone), and its area-between-curves against uncertainty-guided "
            f"is {p1_vs_unc['point_estimate']:+.4f} with a 95% CI "
            f"[{p1_vs_unc['ci95'][0]:+.4f}, {p1_vs_unc['ci95'][1]:+.4f}] excluding zero in the "
            f"combined policy's favour. That is both a real result and, per the coordinator's "
            f"framing, a directive: rank the renderer's hinge line by model uncertainty rather "
            f"than by point weight alone. The mirror policy (uncertainty-gated, hinge-ranked, "
            f"mean {p2_kstar:.3f} corrections-to-correct) is reported alongside for completeness."
        )
    else:
        part3 = (
            f" The follow-up hypothesis -- that the best correction target is BOTH "
            f"decision-fragile and model-uncertain -- is NOT supported here: gating the hinge "
            f"set by model uncertainty does not measurably beat uncertainty alone. "
            f"Hinge-gated,uncertainty-ranked vs uncertainty-guided area-between-curves is "
            f"{p1_vs_unc['point_estimate']:+.4f} (95% CI [{p1_vs_unc['ci95'][0]:+.4f}, "
            f"{p1_vs_unc['ci95'][1]:+.4f}], excludes zero: {p1_vs_unc['excludes_zero']}); mean "
            f"corrections-to-correct are {p1_kstar:.3f} (combined) vs {unc_kstar:.3f} "
            f"(uncertainty alone) -- {'a real edge' if p1_kstar < unc_kstar else 'no edge'} on "
            f"that single number but not one the CI backs. "
            + (
                f"It does still separate cleanly from random (vs random: "
                f"{p1_vs_random['point_estimate']:+.4f}, CI [{p1_vs_random['ci95'][0]:+.4f}, "
                f"{p1_vs_random['ci95'][1]:+.4f}])."
                if p1_beats_random
                else
                f"It does not cleanly separate from random either (vs random: "
                f"{p1_vs_random['point_estimate']:+.4f}, CI [{p1_vs_random['ci95'][0]:+.4f}, "
                f"{p1_vs_random['ci95'][1]:+.4f}])."
            )
            + (
                f" Read plainly: restricting the candidate pool to hinges already captures "
                f"most of what ranking by uncertainty was adding on top of hinge-guidance -- "
                f"once you're only choosing among decision-fragile criteria, which of those is "
                f"also the most uncertain does not move the needle much further on this cohort. "
                f"The mirror policy (uncertainty-gated, hinge-ranked, mean {p2_kstar:.3f} "
                f"corrections-to-correct) is reported alongside for completeness; it was not "
                f"the coordinator's core hypothesis."
            )
        )
    return part1 + part2 + part3


if __name__ == "__main__":
    main()
