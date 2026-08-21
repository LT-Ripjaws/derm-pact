"""N2 (promoted) -- correction-value curves re-run on the PROMOTED criteria heads.

The base analysis (scripts/dev/build_correction_value_curves.py ->
research/evidence/derm7pt_v1/correction_value_curves.json) ran on the ORIGINAL 512-d dermlip
criterion heads. The safety-preserving promotion changed the criteria heads to
dermlip_tta+metadata14 (526-d, lr_balanced, per-head C), which changes the PREDICTED criteria the
correction protocol starts from and the head margins the uncertainty/combined policies rank by. So
the whole correction-value comparison must be re-run on that promoted base before its headline can
be quoted for the promoted system.

This script re-runs the IDENTICAL five-policy comparison on the promoted heads. To guarantee the
policy logic is bit-for-bit the reviewed logic (the subtle part -- the adaptive hinge gating and the
combined policies), it IMPORTS the policy/metric/bootstrap helpers straight from the base script and
only swaps the head source (fit_readers_promoted instead of fit_readers) and re-implements the
straightforward orchestration. The random-seed construction mirrors the base exactly so the promoted
run is itself reproducible.

Core hypothesis (unchanged): does HINGE-GATED, UNCERTAINTY-RANKED still beat random, uncertainty-
guided, and hinge-guided -- with bootstrap CIs excluding zero -- on the promoted base? Reported
honestly as holds / weakens / strengthens against the base numbers.

CAVEAT on the mirror policy (uncertainty-gated, hinge-ranked): its uncertainty GATE reuses
criterion_abstention_v1.json's per-criterion thresholds, which were fit on the ORIGINAL heads and
are on a different margin scale than the promoted balanced heads. That gate is therefore mis-scaled
on promoted margins -- reported for parity with the base only, NOT as a clean comparison. The core
policy (hinge-gated, uncertainty-ranked) does NOT use those thresholds; it gates on live LOCO hinges
and ranks by the promoted heads' own margins, so it is clean on the promoted base.

Boundaries: zero paid calls, zero images. Promoted heads fit on Derm7pt train only, evaluated on
Derm7pt test. Deterministic host aggregation + seeded bootstrap. Does NOT overwrite the base JSON.
Output: research/evidence/derm7pt_v1/correction_value_curves_promoted.json + notes/CORRECTION_VALUE_CURVES_PROMOTED.md
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import score_criteria  # noqa: E402
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.perception.criterion_abstention import load_policy  # noqa: E402
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    _head_values_and_margins,
    _promoted_feature_matrices,
    fit_readers_promoted,
)

# ---- import the base script's reviewed policy/metric/bootstrap helpers by file path ----------
_BASE_PATH = REPO / "scripts" / "dev" / "build_correction_value_curves.py"
_spec = importlib.util.spec_from_file_location("_base_cvc", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

SEED = base.SEED
N_RANDOM_REPLICATES = base.N_RANDOM_REPLICATES
N_BOOTSTRAP = base.N_BOOTSTRAP
N_STEPS = base.N_STEPS

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "correction_value_curves_promoted.json"
OUT_MD = REPO / "notes" / "CORRECTION_VALUE_CURVES_PROMOTED.md"
BASE_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "correction_value_curves.json"


def _load_cohort_promoted():
    """Same shape as base._load_cohort, but with the PROMOTED criteria heads."""
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    diag_feats, crit_feats = _promoted_feature_matrices(cases)
    _diagnosis, heads = fit_readers_promoted(train, diag_feats, crit_feats)
    x_test = np.stack([crit_feats[str(c.case_num)] for c in test])

    predicted_raw: Dict[str, List[str]] = {}
    margins: Dict[str, np.ndarray] = {}
    for name in CRITERION_NAMES:
        values, margin = _head_values_and_margins(heads[name], x_test)  # argmax values + top-2 margin
        predicted_raw[name] = values
        margins[name] = np.asarray(margin, dtype=float)

    policy = load_policy()  # only for the mirror policy p2's (mis-scaled) gate; documented caveat
    return test, train, predicted_raw, margins, policy


def build() -> Dict:
    test, train, predicted_raw, margins, policy = _load_cohort_promoted()
    n = len(test)

    pred_criteria = [{name: str(predicted_raw[name][i]) for name in CRITERION_NAMES} for i in range(n)]
    true_criteria = [dict(c.criteria) for c in test]

    ground_truth_verdict = np.array(
        [score_criteria(true_criteria[i]).meets_threshold for i in range(n)], dtype=bool)
    is_melanoma = np.array([c.diagnosis_group == "MEL" for c in test], dtype=bool)
    predicted_verdict_k0 = np.array(
        [score_criteria(pred_criteria[i]).meets_threshold for i in range(n)], dtype=bool)

    # ---- run the five policies (imported logic) ----
    hinge_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    hinge_orders: List[List[str]] = []
    for i in range(n):
        v, order = base._hinge_order_and_verdicts(pred_criteria[i], true_criteria[i])
        hinge_verdicts[i] = v
        hinge_orders.append(order)

    uncertainty_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    for i in range(n):
        order = base._uncertainty_order(margins, i)
        uncertainty_verdicts[i] = base._fixed_order_verdicts(pred_criteria[i], true_criteria[i], order)

    p1_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    p1_orders: List[List[str]] = []
    p1_gate_hits = np.zeros(n, dtype=int)
    for i in range(n):
        v, order, hits = base._hinge_gated_uncertainty_ranked(pred_criteria[i], true_criteria[i], margins, i)
        p1_verdicts[i] = v
        p1_orders.append(order)
        p1_gate_hits[i] = hits

    p2_verdicts = np.zeros((n, N_STEPS + 1), dtype=bool)
    p2_orders: List[List[str]] = []
    p2_gate_hits = np.zeros(n, dtype=int)
    for i in range(n):
        v, order, hits = base._uncertainty_gated_hinge_ranked(
            pred_criteria[i], true_criteria[i], margins, i, policy.margins)
        p2_verdicts[i] = v
        p2_orders.append(order)
        p2_gate_hits[i] = hits

    # ---- random (20 seeded replicates), seed structure mirrors base exactly ----
    seed_seq = np.random.SeedSequence(SEED)
    child_seeds = seed_seq.spawn(N_RANDOM_REPLICATES + 2)
    random_child_seeds = child_seeds[:N_RANDOM_REPLICATES]
    boot_seed_random = child_seeds[N_RANDOM_REPLICATES]
    boot_seed_uncertainty = child_seeds[N_RANDOM_REPLICATES + 1]
    combo_boot_seeds = seed_seq.spawn(6)
    (boot_seed_p1_random, boot_seed_p1_uncertainty, boot_seed_p1_hinge,
     boot_seed_p2_random, boot_seed_p2_uncertainty, boot_seed_p2_hinge) = combo_boot_seeds

    random_verdicts = np.zeros((N_RANDOM_REPLICATES, n, N_STEPS + 1), dtype=bool)
    for r, cs in enumerate(random_child_seeds):
        rng = np.random.default_rng(cs)
        for i in range(n):
            perm = rng.permutation(N_STEPS)
            order = [CRITERION_NAMES[j] for j in perm]
            random_verdicts[r, i] = base._fixed_order_verdicts(pred_criteria[i], true_criteria[i], order)

    # ---- sanity (tautologies) ----
    assert (hinge_verdicts[:, -1] == ground_truth_verdict).all()
    assert (uncertainty_verdicts[:, -1] == ground_truth_verdict).all()
    assert (p1_verdicts[:, -1] == ground_truth_verdict).all()
    assert (p2_verdicts[:, -1] == ground_truth_verdict).all()
    assert (random_verdicts[:, :, -1] == ground_truth_verdict[None, :]).all()
    assert (hinge_verdicts[:, 0] == predicted_verdict_k0).all()
    assert (p1_verdicts[:, 0] == predicted_verdict_k0).all()

    # ---- curves + k* ----
    hinge_acc, hinge_sens, hinge_spec = base._curve_metrics(hinge_verdicts, ground_truth_verdict)
    unc_acc, unc_sens, unc_spec = base._curve_metrics(uncertainty_verdicts, ground_truth_verdict)
    p1_acc, p1_sens, p1_spec = base._curve_metrics(p1_verdicts, ground_truth_verdict)
    p2_acc, p2_sens, p2_spec = base._curve_metrics(p2_verdicts, ground_truth_verdict)
    hinge_kstar = base._k_star(hinge_verdicts, ground_truth_verdict)
    unc_kstar = base._k_star(uncertainty_verdicts, ground_truth_verdict)
    p1_kstar = base._k_star(p1_verdicts, ground_truth_verdict)
    p2_kstar = base._k_star(p2_verdicts, ground_truth_verdict)

    rep_acc, rep_sens, rep_spec, rep_mean_kstar = [], [], [], []
    for r in range(N_RANDOM_REPLICATES):
        acc_r, sens_r, spec_r = base._curve_metrics(random_verdicts[r], ground_truth_verdict)
        kstar_r = base._k_star(random_verdicts[r], ground_truth_verdict)
        rep_acc.append(acc_r)
        rep_sens.append(sens_r)
        rep_spec.append(spec_r)
        rep_mean_kstar.append(kstar_r.mean())
    rep_acc = np.stack(rep_acc)
    rep_sens = np.stack(rep_sens)
    rep_spec = np.stack(rep_spec)
    rep_mean_kstar = np.asarray(rep_mean_kstar)

    already_correct_frac = float((predicted_verdict_k0 == ground_truth_verdict).mean())

    # ---- area-between-curves comparisons (same bootstrap machinery as base) ----
    hinge_correct = (hinge_verdicts == ground_truth_verdict[:, None]).astype(float)
    uncertainty_correct = (uncertainty_verdicts == ground_truth_verdict[:, None]).astype(float)
    p1_correct = (p1_verdicts == ground_truth_verdict[:, None]).astype(float)
    p2_correct = (p2_verdicts == ground_truth_verdict[:, None]).astype(float)
    random_correct_mean = (random_verdicts == ground_truth_verdict[None, :, None]).astype(float).mean(axis=0)

    def _abc(correct_a, correct_b, seed):
        case_gap = correct_a.mean(axis=1) - correct_b.mean(axis=1)
        return base._bootstrap_abc(case_gap, np.random.default_rng(seed), N_BOOTSTRAP)

    abc_hinge_vs_random = _abc(hinge_correct, random_correct_mean, boot_seed_random)
    abc_hinge_vs_uncertainty = _abc(hinge_correct, uncertainty_correct, boot_seed_uncertainty)
    abc_p1_vs_random = _abc(p1_correct, random_correct_mean, boot_seed_p1_random)
    abc_p1_vs_uncertainty = _abc(p1_correct, uncertainty_correct, boot_seed_p1_uncertainty)
    abc_p1_vs_hinge = _abc(p1_correct, hinge_correct, boot_seed_p1_hinge)
    abc_p2_vs_random = _abc(p2_correct, random_correct_mean, boot_seed_p2_random)
    abc_p2_vs_uncertainty = _abc(p2_correct, uncertainty_correct, boot_seed_p2_uncertainty)
    abc_p2_vs_hinge = _abc(p2_correct, hinge_correct, boot_seed_p2_hinge)

    # ---- dataset-property aside (promoted argmax has no bearing; same gold-criteria ceiling) ----
    tp = int((ground_truth_verdict & is_melanoma).sum())
    fn = int((~ground_truth_verdict & is_melanoma).sum())
    fp = int((ground_truth_verdict & ~is_melanoma).sum())
    tn = int((~ground_truth_verdict & ~is_melanoma).sum())

    def _policy_block(acc, sens, spec, kstar_mean):
        return {
            "accuracy_vs_k": base._round_list(acc),
            "sensitivity_vs_k": base._round_list(sens),
            "specificity_vs_k": base._round_list(spec),
            "mean_corrections_to_correct_verdict": round(float(kstar_mean), 4),
        }

    result = {
        "protocol": "correction_value_curves_PROMOTED_v1",
        "generated_by": "scripts/dev/build_correction_value_curves_promoted.py",
        "base_reference": "research/evidence/derm7pt_v1/correction_value_curves.json (ORIGINAL heads)",
        "difference_from_base": (
            "Identical five-policy protocol and imported policy logic; the ONLY change is the "
            "criteria head source -- promoted dermlip_tta+metadata14 (526-d, lr_balanced, per-head "
            "C) via run_cohort.fit_readers_promoted, instead of the original 512-d dermlip heads. "
            "Predicted starting criteria and head margins therefore change; the deterministic "
            "aggregation, the correction protocol, and the bootstrap are unchanged."
        ),
        "boundaries": {
            "paid_calls": 0,
            "images_encoded": 0,
            "criteria_heads": "PROMOTED dermlip_tta+metadata14 (526-d, lr_balanced, per-head C)",
            "fit_split": "train",
            "n_fit_cases": len(train),
            "evaluated_split": "test",
            "n_test_cases": n,
            "n_melanoma_test": int(is_melanoma.sum()),
            "melanoma_threshold": MELANOMA_SUSPICION_THRESHOLD,
            "seed": SEED,
            "random_policy_replicates": N_RANDOM_REPLICATES,
            "bootstrap_replicates": N_BOOTSTRAP,
        },
        "mirror_policy_caveat": (
            "uncertainty_gated_hinge_ranked reuses criterion_abstention_v1.json thresholds (fit on "
            "the ORIGINAL heads) as its uncertainty gate; on the promoted balanced-head margin scale "
            "that gate is mis-calibrated. Reported for parity with the base, NOT as a clean "
            "comparison. The core policy hinge_gated_uncertainty_ranked does not use those "
            "thresholds and is clean on the promoted base."
        ),
        "policies": {
            "hinge_guided": _policy_block(hinge_acc, hinge_sens, hinge_spec, hinge_kstar.mean()),
            "uncertainty_guided": _policy_block(unc_acc, unc_sens, unc_spec, unc_kstar.mean()),
            "random": {
                "accuracy_vs_k": {"mean": base._round_list(rep_acc.mean(axis=0)),
                                  "std": base._round_list(rep_acc.std(axis=0, ddof=1))},
                "sensitivity_vs_k": {"mean": base._round_list(rep_sens.mean(axis=0)),
                                     "std": base._round_list(rep_sens.std(axis=0, ddof=1))},
                "specificity_vs_k": {"mean": base._round_list(rep_spec.mean(axis=0)),
                                     "std": base._round_list(rep_spec.std(axis=0, ddof=1))},
                "mean_corrections_to_correct_verdict": {
                    "mean": round(float(rep_mean_kstar.mean()), 4),
                    "std": round(float(rep_mean_kstar.std(ddof=1)), 4),
                },
                "replicates": N_RANDOM_REPLICATES,
            },
            "hinge_gated_uncertainty_ranked": {
                "hinge_gate_nonempty_fraction_of_steps": round(float(p1_gate_hits.mean() / N_STEPS), 4),
                **_policy_block(p1_acc, p1_sens, p1_spec, p1_kstar.mean()),
            },
            "uncertainty_gated_hinge_ranked": {
                "uncertainty_gate_nonempty_fraction_of_steps": round(float(p2_gate_hits.mean() / N_STEPS), 4),
                **_policy_block(p2_acc, p2_sens, p2_spec, p2_kstar.mean()),
            },
        },
        "already_correct_at_k0": {
            "fraction": round(already_correct_frac, 4),
            "n_cases": int(round(already_correct_frac * n)),
        },
        "hinge_vs_random_area_between_curves": abc_hinge_vs_random,
        "hinge_vs_uncertainty_area_between_curves": abc_hinge_vs_uncertainty,
        "combined_policy_comparisons": {
            "hinge_gated_uncertainty_ranked_vs_random": abc_p1_vs_random,
            "hinge_gated_uncertainty_ranked_vs_uncertainty_guided": abc_p1_vs_uncertainty,
            "hinge_gated_uncertainty_ranked_vs_hinge_guided": abc_p1_vs_hinge,
            "uncertainty_gated_hinge_ranked_vs_random": abc_p2_vs_random,
            "uncertainty_gated_hinge_ranked_vs_uncertainty_guided": abc_p2_vs_uncertainty,
            "uncertainty_gated_hinge_ranked_vs_hinge_guided": abc_p2_vs_hinge,
        },
        "dataset_property_aside_checklist_vs_diagnosis": {
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "note": "Gold-criteria verdict vs diagnosis label. Independent of which heads produced "
                    "the predicted criteria, so identical to the base (a wiring check).",
        },
    }
    result["verdict_vs_base"] = _verdict_vs_base(result)
    return result


def _load_base():
    if BASE_JSON.exists():
        return json.loads(BASE_JSON.read_text(encoding="utf-8"))
    return None


def _holds(pt, ci, base_pt):
    """Classify a comparison vs its base counterpart."""
    excl0 = ci[0] > 0 or ci[1] < 0
    if base_pt is None:
        return "holds" if (excl0 and pt > 0) else "not-significant"
    if excl0 and pt > 0:
        return "strengthens" if pt > base_pt else "holds"
    return "weakens"


def _verdict_vs_base(result: Dict) -> Dict:
    base_data = _load_base()
    combo = result["combined_policy_comparisons"]

    def base_combo(key):
        if not base_data:
            return None
        return base_data.get("combined_policy_comparisons", {}).get(key, {}).get("point_estimate")

    core = {}
    for key, label in (
        ("hinge_gated_uncertainty_ranked_vs_random", "vs random"),
        ("hinge_gated_uncertainty_ranked_vs_uncertainty_guided", "vs uncertainty"),
        ("hinge_gated_uncertainty_ranked_vs_hinge_guided", "vs hinge"),
    ):
        c = combo[key]
        bpt = base_combo(key)
        core[label] = {
            "promoted_point_estimate": c["point_estimate"],
            "promoted_ci95": c["ci95"],
            "promoted_excludes_zero": c["excludes_zero"],
            "base_point_estimate": bpt,
            "classification": _holds(c["point_estimate"], c["ci95"], bpt),
        }

    all_win = all(
        combo[k]["excludes_zero"] and combo[k]["point_estimate"] > 0
        for k in (
            "hinge_gated_uncertainty_ranked_vs_random",
            "hinge_gated_uncertainty_ranked_vs_uncertainty_guided",
            "hinge_gated_uncertainty_ranked_vs_hinge_guided",
        )
    )
    p1_kstar = result["policies"]["hinge_gated_uncertainty_ranked"]["mean_corrections_to_correct_verdict"]
    unc_kstar = result["policies"]["uncertainty_guided"]["mean_corrections_to_correct_verdict"]
    base_p1_kstar = (
        base_data["policies"]["hinge_gated_uncertainty_ranked"]["mean_corrections_to_correct_verdict"]
        if base_data else None
    )

    # honest nuances: is the vs-hinge edge marginal, and does mean-k* disagree with the ABC verdict?
    hinge_kstar = result["policies"]["hinge_guided"]["mean_corrections_to_correct_verdict"]
    vs_hinge = combo["hinge_gated_uncertainty_ranked_vs_hinge_guided"]
    vs_hinge_marginal = vs_hinge["excludes_zero"] and vs_hinge["ci95"][0] < 0.002
    kstar_disagrees = hinge_kstar < p1_kstar  # hinge-guided reaches correct verdict in fewer steps
    nuances = []
    if vs_hinge_marginal:
        nuances.append(
            f"The edge over hinge-guided ALONE is marginal on promoted heads: point "
            f"{vs_hinge['point_estimate']:+.4f}, 95% CI [{vs_hinge['ci95'][0]:+.4f}, "
            f"{vs_hinge['ci95'][1]:+.4f}] -- the lower bound sits essentially at zero, within "
            f"bootstrap noise, so 'beats hinge-guided' is not robust here (it WAS robust on the base "
            f"at {base_combo('hinge_gated_uncertainty_ranked_vs_hinge_guided') or 'n/a'})."
        )
    if kstar_disagrees:
        nuances.append(
            f"By the mean-corrections metric the two even swap: hinge-guided ALONE reaches the "
            f"correct verdict in {hinge_kstar} corrections vs {p1_kstar} for the combined policy. "
            f"The promoted heads give a better starting point (already-correct@k0 "
            f"{result['already_correct_at_k0']['fraction']}), which strengthens hinge-guidance on "
            f"its own; the combined policy's remaining, robust advantage is over UNCERTAINTY-guided "
            f"(which it strengthens vs base), not over hinge-guided."
        )

    if all_win:
        headline = (
            "HOLDS on the promoted heads: hinge-gated, uncertainty-ranked still beats random, "
            "uncertainty-guided, AND hinge-guided, all three bootstrap CIs excluding zero in its "
            f"favour. Mean corrections-to-correct-verdict {p1_kstar} (vs {unc_kstar} for "
            f"uncertainty-guided alone; base was {base_p1_kstar}). Its edge over UNCERTAINTY-guided "
            f"STRENGTHENS vs the base "
            f"({combo['hinge_gated_uncertainty_ranked_vs_uncertainty_guided']['point_estimate']:+.4f} "
            f"vs base {base_combo('hinge_gated_uncertainty_ranked_vs_uncertainty_guided') or 'n/a'})."
            + (" NUANCES: " + " ".join(nuances) if nuances else "")
        )
    else:
        losers = [lbl for lbl, d in core.items() if d["classification"] == "weakens"]
        headline = (
            "WEAKENS on the promoted heads: hinge-gated, uncertainty-ranked no longer clears every "
            f"bar. It fails to significantly beat: {losers}. Mean corrections-to-correct "
            f"{p1_kstar} (uncertainty-guided {unc_kstar}; base p1 {base_p1_kstar}). See the "
            "per-comparison CIs -- report the specific bars it still clears and the ones it does not."
        )

    return {
        "core_comparisons_vs_base": core,
        "still_beats_all_three": bool(all_win),
        "nuances": nuances,
        "headline": headline,
    }


def _write_notes(r: Dict) -> None:
    pol = r["policies"]
    combo = r["combined_policy_comparisons"]
    v = r["verdict_vs_base"]
    lines: List[str] = []
    lines.append("# N2 (promoted) -- correction-value curves on the PROMOTED criteria heads")
    lines.append("")
    lines.append(r["difference_from_base"])
    lines.append("")
    b = r["boundaries"]
    lines.append(f"n={b['n_test_cases']} test ({b['n_melanoma_test']} melanoma), promoted heads fit "
                 f"on train (n={b['n_fit_cases']}). Seed {b['seed']}, {b['bootstrap_replicates']} "
                 "bootstrap reps. Zero paid calls.")
    lines.append("")
    lines.append(f"Already correct at k=0: **{r['already_correct_at_k0']['fraction']:.4f}** "
                 f"({r['already_correct_at_k0']['n_cases']}/{b['n_test_cases']}).")
    lines.append("")
    lines.append("## Mean corrections until verdict matches ground truth (lower = better targeting)")
    lines.append("")
    lines.append("| policy | mean k* |")
    lines.append("|---|---:|")
    for name, label in (
        ("hinge_guided", "hinge-guided"),
        ("uncertainty_guided", "uncertainty-guided"),
        ("hinge_gated_uncertainty_ranked", "hinge-gated, uncertainty-ranked"),
        ("uncertainty_gated_hinge_ranked", "uncertainty-gated, hinge-ranked (mirror, caveat)"),
    ):
        lines.append(f"| {label} | {pol[name]['mean_corrections_to_correct_verdict']:.4f} |")
    rr = pol["random"]["mean_corrections_to_correct_verdict"]
    lines.append(f"| random | {rr['mean']:.4f} +/- {rr['std']:.4f} |")
    lines.append("")
    lines.append("## Core hypothesis: hinge-gated, uncertainty-ranked vs the alternatives")
    lines.append("")
    lines.append("| comparison | promoted point | promoted 95% CI | excl 0 | base point | vs base |")
    lines.append("|---|---:|---|:---:|---:|---|")
    for key, label in (
        ("hinge_gated_uncertainty_ranked_vs_random", "vs random"),
        ("hinge_gated_uncertainty_ranked_vs_uncertainty_guided", "vs uncertainty-guided"),
        ("hinge_gated_uncertainty_ranked_vs_hinge_guided", "vs hinge-guided"),
    ):
        c = combo[key]
        d = v["core_comparisons_vs_base"][{"hinge_gated_uncertainty_ranked_vs_random": "vs random",
            "hinge_gated_uncertainty_ranked_vs_uncertainty_guided": "vs uncertainty",
            "hinge_gated_uncertainty_ranked_vs_hinge_guided": "vs hinge"}[key]]
        bpt = d["base_point_estimate"]
        lines.append(f"| {label} | {c['point_estimate']:+.4f} | "
                     f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['excludes_zero']} | "
                     f"{bpt:+.4f} | {d['classification']} |")
    lines.append("")
    lines.append("## Secondary (mirror policy, mis-scaled gate -- parity only)")
    lines.append("")
    for key, label in (
        ("uncertainty_gated_hinge_ranked_vs_random", "mirror vs random"),
        ("uncertainty_gated_hinge_ranked_vs_uncertainty_guided", "mirror vs uncertainty-guided"),
        ("uncertainty_gated_hinge_ranked_vs_hinge_guided", "mirror vs hinge-guided"),
    ):
        c = combo[key]
        lines.append(f"- {label}: {c['point_estimate']:+.4f} "
                     f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], excludes zero {c['excludes_zero']}")
    lines.append("")
    lines.append("## Verdict vs base")
    lines.append("")
    lines.append(v["headline"])
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    r = build()
    assert OUT_JSON.name != BASE_JSON.name, "must not overwrite the base curves"
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    _write_notes(r)
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    pol = r["policies"]
    combo = r["combined_policy_comparisons"]
    print("\n=== N2 promoted correction-value curves ===")
    print(f"  already-correct@k0: {r['already_correct_at_k0']['fraction']}")
    print(f"  meanK*: hinge {pol['hinge_guided']['mean_corrections_to_correct_verdict']} | "
          f"uncertainty {pol['uncertainty_guided']['mean_corrections_to_correct_verdict']} | "
          f"p1 {pol['hinge_gated_uncertainty_ranked']['mean_corrections_to_correct_verdict']} | "
          f"random {pol['random']['mean_corrections_to_correct_verdict']['mean']}")
    for key in ("hinge_gated_uncertainty_ranked_vs_random",
                "hinge_gated_uncertainty_ranked_vs_uncertainty_guided",
                "hinge_gated_uncertainty_ranked_vs_hinge_guided"):
        c = combo[key]
        print(f"  {key:55s} {c['point_estimate']:+.4f} CI={c['ci95']} excl0={c['excludes_zero']}")
    print(f"\n  {r['verdict_vs_base']['headline']}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
