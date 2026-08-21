"""Deterministic, offline evaluation of the case-based (faithful-by-construction) XAI arm.

The clinician surface now carries an ON-DEMAND case-based explanation: for a case, the nearest
confirmed TRAIN cases in the frozen DermLIP feature space, plus the nearest different-label case as a
boundary counterfactual. This script measures whether that retrieval is actually SANE -- i.e. whether
"nearest in the model's own feature space" tracks the clinical label -- because an explanation that
retrieves semantically unrelated cases would be faithful-to-the-model but useless-to-the-clinician.

Two retrieval-sanity endpoints, over the frozen TRAIN-only image-embedding bank:

  (a) same-label neighbour rate  -- of the k nearest TRAIN cases, the fraction whose CONFIRMED
      (ground-truth) diagnosis matches the query's true diagnosis. Reported against a random-retrieval
      baseline (the label-prior collision rate) so the lift is interpretable, and per-class so the
      SK/MISC tail is visible.
  (b) boundary separation gap    -- nearest-same-label similarity minus nearest-different-label
      similarity, per case. Positive => the case sits closer to its own class than to the nearest
      other class; this is exactly the "nearest different-label case" the render surfaces.

Leak discipline: the bank is TRAIN-only. The primary read is over VALID (disjoint from train, so
retrieval is leak-free by construction). A second read is TRAIN-on-TRAIN with leave-one-out exclusion
-- which additionally PROVES the exclusion works: no case may retrieve itself.

Retrieval feature space: the pure frozen 512-d DermLIP image embedding (NOT the 526-d
dermlip+metadata concat the linear probe reads). Same argued deviation as the P2 kNN ensemble
(knn_ensemble_p2.json): in RAW cosine the 14-d one-hot metadata block swamps the unit-norm image
block, so a metadata-concatenated bank would retrieve by demographics, not lesion appearance. The
linear probe reweights each dim and is immune; cosine kNN is not.

A handful of production-style example contracts (committed diagnosis from the promoted reader) are
rendered WITH the on-demand appendix, dogfooding the exact clinician surface.

Zero paid API calls; everything is local over cached features.

Run::
    py -3.13 scripts/dev/build_case_based_xai.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.case_memory import build_similar_cases_context  # noqa: E402
from medguard_v2.criteria.schema import CRITERION_NAMES  # noqa: E402
from medguard_v2.data.derm7pt import DERM7PT_CLASS_ORDER, load_cases  # noqa: E402
from medguard_v2.evidence.contract import build_contract, contract_to_dict  # noqa: E402
from medguard_v2.evidence.guideline_binding import GuidelineCatalog  # noqa: E402
from medguard_v2.evidence.render import render_markdown  # noqa: E402
from medguard_v2.perception.knn_memory import build_memory  # noqa: E402
from medguard_v2.pipeline.run_cohort import (  # noqa: E402
    PROMOTED_OPERATING_POINT,
    _promoted_feature_matrices,
    apply_operating_point,
    fit_readers_promoted,
)

EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
OUT_JSON = EVIDENCE / "case_based_xai.json"
OUT_NOTE = REPO / "notes" / "CASE_BASED_XAI.md"

# Pure frozen 512-d DermLIP image embedding -- the retrieval space (see module docstring).
FEATURES_DERMLIP = EVIDENCE / "features" / "dermlip" / "features.npz"

SEED = 20260810
K_GRID = (1, 3, 5, 7, 10)
K_MAIN = 5
LABELS = list(DERM7PT_CLASS_ORDER)
FEATURE_SPACE = "dermlip (512-d frozen DermLIP image embedding; image-only cosine)"


def _round4(x) -> float:
    return round(float(x), 4)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {str(c): v for c, v in zip(d["case_num"], d["features"])}


def _full_ranking(bank, vec: np.ndarray, exclude_id: str | None):
    """Whole-bank (id, confirmed_label, cosine) ranking for one query, descending similarity."""
    idx, sims = bank.retrieve(
        vec.reshape(1, -1), len(bank), exclude_ids=[exclude_id] if exclude_id else None
    )
    ids = bank._case_ids
    labels = bank.label_values("diagnosis")
    return [
        (str(ids[int(j)]), str(labels[int(j)]), float(s))
        for j, s in zip(idx[0], sims[0])
        if np.isfinite(s)
    ]


def _case_metrics(ranked, true_label: str, k: int) -> Dict:
    """Per-case retrieval-sanity quantities from a full ranking against the query's TRUE label."""
    topk = ranked[:k]
    same_in_topk = [sim for cid, lab, sim in topk if lab == true_label]
    nearest_same = next((sim for cid, lab, sim in ranked if lab == true_label), None)
    nearest_other = next((sim for cid, lab, sim in ranked if lab != true_label), None)
    return {
        "same_label_rate": len(same_in_topk) / k,
        "nearest_same_sim": nearest_same,
        "nearest_other_sim": nearest_other,
        "boundary_gap": (
            None if (nearest_same is None or nearest_other is None) else nearest_same - nearest_other
        ),
        "one_nn_same_label": bool(ranked and ranked[0][1] == true_label),
        "self_retrieved": None,  # filled by caller for the LOO arm
    }


def _aggregate(per_case: List[Dict], cases, k: int) -> Dict:
    gaps = [m["boundary_gap"] for m in per_case if m["boundary_gap"] is not None]
    rates = [m["same_label_rate"] for m in per_case]
    one_nn = [m["one_nn_same_label"] for m in per_case]
    pos_gap = [g for g in gaps if g > 0]
    near_same = [m["nearest_same_sim"] for m in per_case if m["nearest_same_sim"] is not None]
    near_other = [m["nearest_other_sim"] for m in per_case if m["nearest_other_sim"] is not None]

    # Per-class same-label rate (the tail SK/MISC is the point of the retrieval arm).
    by_class: Dict[str, List[float]] = {c: [] for c in LABELS}
    for m, case in zip(per_case, cases):
        by_class[case.diagnosis_group].append(m["same_label_rate"])

    return {
        "k": k,
        "n": len(per_case),
        "same_label_neighbor_rate_mean": _round4(np.mean(rates)),
        "same_label_neighbor_rate_std": _round4(np.std(rates)),
        "boundary_gap_mean": _round4(np.mean(gaps)) if gaps else None,
        "boundary_gap_median": _round4(np.median(gaps)) if gaps else None,
        "nearest_same_sim_mean": _round4(np.mean(near_same)) if near_same else None,
        "nearest_other_sim_mean": _round4(np.mean(near_other)) if near_other else None,
        # NOTE: fraction_positive_gap == one_nn_same_label_accuracy by construction (gap > 0 iff the
        # single nearest case is same-label). Reported once, not twice, to avoid double-counting.
        "fraction_cases_positive_gap": _round4(len(pos_gap) / len(gaps)) if gaps else None,
        "one_nn_same_label_accuracy": _round4(np.mean(one_nn)),
        "per_class_same_label_rate": {
            c: _round4(np.mean(v)) for c, v in by_class.items() if v
        },
        "per_class_support": {c: len(v) for c, v in by_class.items() if v},
    }


def _random_baseline_same_label_rate(query_cases, bank_cases) -> float:
    """Expected same-label rate of a RANDOM neighbour: sum_c p_query(c) * p_bank(c).

    The chance level any retrieval must beat -- a bank that just returned random train cases would
    score this by the label-prior collision alone.
    """
    q_counts = Counter(c.diagnosis_group for c in query_cases)
    b_counts = Counter(c.diagnosis_group for c in bank_cases)
    n_q, n_b = len(query_cases), len(bank_cases)
    return sum((q_counts.get(c, 0) / n_q) * (b_counts.get(c, 0) / n_b) for c in LABELS)


def _k_sweep(feat, query_cases, bank, exclude_self: bool) -> List[Dict]:
    out = []
    for k in K_GRID:
        per_case = []
        for c in query_cases:
            ranked = _full_ranking(
                bank, feat[str(c.case_num)], str(c.case_num) if exclude_self else None
            )
            per_case.append(_case_metrics(ranked, c.diagnosis_group, k))
        out.append(_aggregate(per_case, query_cases, k))
    return out


def _example_contracts(cases, train, valid, feat, bank, catalog) -> List[Dict]:
    """Production-style contracts (committed diagnosis from the promoted reader) rendered WITH the
    on-demand appendix -- dogfooding the exact clinician surface the render/agent expose."""
    diag_feats, crit_feats = _promoted_feature_matrices(cases)
    diagnosis, heads = fit_readers_promoted(train, diag_feats, crit_feats)
    xd = np.stack([diag_feats[str(c.case_num)] for c in valid])
    xc = np.stack([crit_feats[str(c.case_num)] for c in valid])
    posterior = diagnosis.predict_proba(xd)
    predicted, confidences = apply_operating_point(posterior, diagnosis.classes_, PROMOTED_OPERATING_POINT)
    head_pred = {name: [str(v) for v in head.predict(xc)] for name, head in heads.items()}

    # One example per committed class where available, capped, for a spread of surfaces.
    chosen: Dict[str, int] = {}
    for i, c in enumerate(valid):
        committed = str(predicted[i])
        if committed not in chosen:
            chosen[committed] = i
        if len(chosen) == len(LABELS):
            break

    examples = []
    for committed, i in sorted(chosen.items()):
        case = valid[i]
        criteria = {name: head_pred[name][i] for name in CRITERION_NAMES}
        ctx = build_similar_cases_context(
            query_embedding=feat[str(case.case_num)],
            reference_diagnosis=committed,          # PRODUCTION: same_label vs the COMMITTED diagnosis
            bank=bank,
            query_case_id=str(case.case_num),
            k=K_MAIN,
            feature_space=FEATURE_SPACE,
        )
        contract = build_contract(
            case_id=f"derm7pt-{case.case_num}",
            classifier_diagnosis=committed,
            classifier_confidence=float(confidences[i]),
            criteria=criteria,
            catalog=catalog,
            similar_cases=ctx,
        )
        examples.append({
            "case_num": case.case_num,
            "true_diagnosis": case.diagnosis_group,
            "committed_diagnosis": committed,
            "safe_to_display": contract.safe_to_display,
            "default_view_hides_appendix": "Similar confirmed cases" not in render_markdown(contract),
            "similar_cases": contract_to_dict(contract)["similar_cases"],
            "rendered_on_demand": render_markdown(contract, include_appendix=True),
        })
    return examples


def build() -> Dict:
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    feat = _load_npz(FEATURES_DERMLIP)

    catalog = GuidelineCatalog.load()

    # Frozen, hash-versioned TRAIN-only diagnosis bank. TEST is never touched; VALID is disjoint.
    bank = build_memory(
        feat, train, {"diagnosis": [c.diagnosis_group for c in train]}, name="diagnosis@dermlip",
    )

    # ---- primary read: VALID vs TRAIN (leak-free by disjointness) ---------------------------
    valid_sweep = _k_sweep(feat, valid, bank, exclude_self=False)
    valid_main = next(r for r in valid_sweep if r["k"] == K_MAIN)

    # ---- second read: TRAIN-on-TRAIN with leave-one-out (proves the exclusion works) --------
    train_sweep = _k_sweep(feat, train, bank, exclude_self=True)
    train_main = next(r for r in train_sweep if r["k"] == K_MAIN)

    # Leak-free proof: over every train query with exclusion, the case itself is never retrieved.
    self_leaks = 0
    for c in train:
        ranked = _full_ranking(bank, feat[str(c.case_num)], str(c.case_num))
        if any(cid == str(c.case_num) for cid, _lab, _sim in ranked):
            self_leaks += 1

    valid_baseline = _random_baseline_same_label_rate(valid, train)
    train_baseline = _random_baseline_same_label_rate(train, train)

    examples = _example_contracts(cases, train, valid, feat, bank, catalog)

    report = {
        "protocol": "derm7pt-case-based-xai-retrieval-sanity-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "purpose": (
            "Measures whether the ON-DEMAND case-based explanation retrieves clinically coherent "
            "neighbours: same-label neighbour rate (retrieval sanity) and same-vs-nearest-other-label "
            "similarity gap (boundary separation), over a frozen TRAIN-only DermLIP image bank."
        ),
        "faithful_by_construction_note": (
            "The neighbours ARE the query's nearest points in the frozen feature space the perception "
            "reader itself reads, so the explanation is faithful by construction (identity in the "
            "model's representation), not a post-hoc rationalisation. This eval does NOT test "
            "faithfulness (which is definitional); it tests whether that faithful retrieval is "
            "clinically SANE."
        ),
        "retrieval_feature_space": {
            "space": FEATURE_SPACE,
            "argued_deviation_from_brief": (
                "The brief specified dermlip+metadata for the bank. Same verified premise failure as "
                "knn_ensemble_p2.json: in RAW cosine the 14-d one-hot metadata block swamps the "
                "unit-norm image block, so a metadata-concatenated bank retrieves by demographics, not "
                "lesion appearance. A linear probe reweights each dimension and is immune; cosine kNN "
                "is not. Image-only retrieval is the faithful and stronger choice."
            ),
        },
        "bank": {**bank.summary(), "split": "train", "feature_space": FEATURE_SPACE},
        "cohort": {
            "train": len(train), "valid": len(valid),
            "valid_class_support": dict(Counter(c.diagnosis_group for c in valid)),
        },
        "leak_discipline": {
            "primary_read": "VALID queries vs TRAIN bank (disjoint splits -> leak-free by construction)",
            "second_read": "TRAIN-on-TRAIN with leave-one-out exclusion",
            "train_loo_self_leaks": self_leaks,
            "leak_free": self_leaks == 0,
        },
        "same_label_neighbor_rate": {
            "definition": "fraction of the k nearest TRAIN cases whose CONFIRMED diagnosis matches "
                          "the query's true diagnosis",
            "k_main": K_MAIN,
            "valid": {
                "rate_mean": valid_main["same_label_neighbor_rate_mean"],
                "random_baseline_label_prior": _round4(valid_baseline),
                "lift_over_baseline": _round4(
                    valid_main["same_label_neighbor_rate_mean"] - valid_baseline),
                "per_class_rate": valid_main["per_class_same_label_rate"],
                "per_class_support": valid_main["per_class_support"],
                "one_nn_same_label_accuracy": valid_main["one_nn_same_label_accuracy"],
            },
            "train_loo": {
                "rate_mean": train_main["same_label_neighbor_rate_mean"],
                "random_baseline_label_prior": _round4(train_baseline),
                "lift_over_baseline": _round4(
                    train_main["same_label_neighbor_rate_mean"] - train_baseline),
                "per_class_rate": train_main["per_class_same_label_rate"],
                "one_nn_same_label_accuracy": train_main["one_nn_same_label_accuracy"],
            },
        },
        "boundary_separation_gap": {
            "definition": "per-case (nearest-same-label cosine) minus (nearest-different-label "
                          "cosine); positive => the case is closer to its own class than to any other",
            "identity_note": (
                "fraction_cases_positive_gap equals one_nn_same_label_accuracy by construction (the "
                "gap is positive iff the single nearest case is same-label). The nearest_*_sim_mean "
                "pair shows the absolute proximities: both are high, so lesions sit in a DENSE region "
                "and near class boundaries in raw DermLIP cosine space -- which is precisely why "
                "surfacing the nearest different-label case is clinically informative (it flags a "
                "genuinely close rival, not a far-away strawman). The linear probe reweights "
                "dimensions to separate the classes; raw cosine kNN cannot (cf. knn_ensemble_p2.json)."
            ),
            "valid": {
                "gap_mean": valid_main["boundary_gap_mean"],
                "gap_median": valid_main["boundary_gap_median"],
                "nearest_same_sim_mean": valid_main["nearest_same_sim_mean"],
                "nearest_other_sim_mean": valid_main["nearest_other_sim_mean"],
                "fraction_cases_positive_gap": valid_main["fraction_cases_positive_gap"],
            },
            "train_loo": {
                "gap_mean": train_main["boundary_gap_mean"],
                "gap_median": train_main["boundary_gap_median"],
                "nearest_same_sim_mean": train_main["nearest_same_sim_mean"],
                "nearest_other_sim_mean": train_main["nearest_other_sim_mean"],
                "fraction_cases_positive_gap": train_main["fraction_cases_positive_gap"],
            },
        },
        "k_sweep": {"valid": valid_sweep, "train_loo": train_sweep},
        "example_contracts": examples,
        "surface_wiring": {
            "contract_field": "DecisionContract.similar_cases (Optional[SimilarCasesContext])",
            "render": "render_markdown(contract, include_appendix=True) -- ON-DEMAND appendix; the "
                      "default 10-second view never shows it (MG52 progressive disclosure)",
            "agent": "clinician_agent.explain_reasoning() step 9 (case_based_explanation), on-demand",
            "saliency": "caveated pointer only (faithful by deletion/insertion but spatially diffuse; "
                        "not a substitute for clinician localization); attribution_faithfulness_v1.json",
        },
    }
    return report


def _fmt_pc(d: Dict[str, float]) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items())


def _write_note(report: Dict) -> str:
    sl = report["same_label_neighbor_rate"]
    bg = report["boundary_separation_gap"]
    bank = report["bank"]
    lines = [
        "# Case-based explanation (faithful-by-construction XAI): retrieval-sanity evidence",
        "",
        f"_Generated {report['generated_at']} · seed {report['seed']} · zero paid API calls._",
        "",
        "## What this is",
        "",
        "The clinician surface now carries an **on-demand** case-based explanation: the nearest "
        "confirmed TRAIN cases to a lesion in the frozen DermLIP feature space, plus the nearest "
        "different-label case as a boundary counterfactual. It is **faithful by construction** -- the "
        "neighbours are the query's own nearest points in the representation the perception reader "
        "itself reads, i.e. the identity of the case in the model's feature space, not a post-hoc "
        "rationalisation. This evidence does not test faithfulness (that is definitional); it tests "
        "whether that faithful retrieval is clinically **sane**.",
        "",
        "Progressive disclosure (MG52): a displayed explanation measurably raises clinician cognitive "
        "load, so the block is **never** in the default 10-second view -- it appears only when the "
        "agent (or a UI affordance) explicitly requests it AND the context is present.",
        "",
        "## Retrieval bank",
        "",
        f"- `{bank['name']}` · **{bank['n_entries']}** TRAIN cases · {bank['feature_dim']}-d · "
        f"`{bank['bank_hash'][:16]}` (sha256)",
        f"- Feature space: {report['retrieval_feature_space']['space']}",
        f"- Leak discipline: {report['leak_discipline']['primary_read']}; "
        f"train-LOO self-leaks = **{report['leak_discipline']['train_loo_self_leaks']}** "
        f"(leak-free = {report['leak_discipline']['leak_free']})",
        "",
        "## (a) Same-label neighbour rate — retrieval sanity",
        "",
        f"Of the k={sl['k_main']} nearest TRAIN cases, the fraction whose confirmed diagnosis matches "
        "the query's true diagnosis. A random-retrieval bank would score only the label-prior "
        "collision rate.",
        "",
        "| split | same-label rate | random baseline | lift | 1-NN same-label acc |",
        "|---|---:|---:|---:|---:|",
        f"| VALID (leak-free) | **{sl['valid']['rate_mean']}** | "
        f"{sl['valid']['random_baseline_label_prior']} | "
        f"+{sl['valid']['lift_over_baseline']} | {sl['valid']['one_nn_same_label_accuracy']} |",
        f"| TRAIN (leave-one-out) | {sl['train_loo']['rate_mean']} | "
        f"{sl['train_loo']['random_baseline_label_prior']} | "
        f"+{sl['train_loo']['lift_over_baseline']} | {sl['train_loo']['one_nn_same_label_accuracy']} |",
        "",
        f"Per-class same-label rate (VALID): {_fmt_pc(sl['valid']['per_class_rate'])}.",
        "",
        "Honest read: retrieval is coherent well above chance for the majority classes (NEV strongest), "
        "but the **SK/MISC tail retrieves poorly** -- the same minority-class difficulty seen in "
        "`knn_ensemble_p2.json`, where a training-free kNN readout did not beat the linear probe on "
        "tail recall. The case-based explanation is therefore reported as an auditability/coherence "
        "surface, not an accuracy claim.",
        "",
        "## (b) Boundary separation gap",
        "",
        "Per case: nearest-same-label cosine minus nearest-different-label cosine. Positive => the "
        "lesion sits closer to its own class than to any other -- exactly the boundary the render's "
        "\"nearest different-label case\" surfaces.",
        "",
        "| split | gap (mean) | gap (median) | nearest-same | nearest-other | fraction positive |",
        "|---|---:|---:|---:|---:|---:|",
        f"| VALID | **{bg['valid']['gap_mean']}** | {bg['valid']['gap_median']} | "
        f"{bg['valid']['nearest_same_sim_mean']} | {bg['valid']['nearest_other_sim_mean']} | "
        f"{bg['valid']['fraction_cases_positive_gap']} |",
        f"| TRAIN (LOO) | {bg['train_loo']['gap_mean']} | {bg['train_loo']['gap_median']} | "
        f"{bg['train_loo']['nearest_same_sim_mean']} | {bg['train_loo']['nearest_other_sim_mean']} | "
        f"{bg['train_loo']['fraction_cases_positive_gap']} |",
        "",
        "`fraction positive` equals the 1-NN same-label accuracy above **by construction** (the gap is "
        "positive iff the single nearest case is same-label) -- reported once, not as an independent "
        "confirmation. Both nearest-same and nearest-other similarities are high: lesions sit in a "
        "dense region near class boundaries in raw DermLIP cosine space, which is why surfacing the "
        "nearest different-label case is clinically useful -- it flags a genuinely close rival, not a "
        "far-away strawman.",
        "",
        "## k sweep (VALID)",
        "",
        "Same-label rate stays well above the 0.384 chance baseline across k (it is the k-dependent "
        "quantity; the boundary gap uses the full ranking and is k-invariant, so it is not swept here).",
        "",
        "| k | same-label rate | lift over baseline |",
        "|---:|---:|---:|",
    ]
    _base = report["same_label_neighbor_rate"]["valid"]["random_baseline_label_prior"]
    for r in report["k_sweep"]["valid"]:
        lines.append(
            f"| {r['k']} | {r['same_label_neighbor_rate_mean']} | "
            f"+{_round4(r['same_label_neighbor_rate_mean'] - _base)} |"
        )
    lines += [
        "",
        "## Surface wiring",
        "",
        f"- **Contract**: `{report['surface_wiring']['contract_field']}` (optional, round-trips in "
        "`contract_to_dict`, host-verified leak-free).",
        f"- **Render**: {report['surface_wiring']['render']}.",
        f"- **Agent**: {report['surface_wiring']['agent']}.",
        f"- **Saliency**: {report['surface_wiring']['saliency']}.",
        "",
        f"{len(report['example_contracts'])} production-style example contracts (committed diagnosis "
        "from the promoted reader) are rendered with the on-demand appendix in "
        "`research/evidence/derm7pt_v1/case_based_xai.json` (`example_contracts[*].rendered_on_demand`).",
        "",
    ]
    text = "\n".join(lines)
    OUT_NOTE.write_text(text, encoding="utf-8")
    return text


def _print_summary(report: Dict) -> None:
    sl = report["same_label_neighbor_rate"]
    bg = report["boundary_separation_gap"]
    print("\n=== case-based XAI retrieval sanity ===")
    print(f"  bank            : {report['bank']['name']} n={report['bank']['n_entries']} "
          f"{report['bank']['bank_hash'][:12]}...")
    print(f"  leak-free (LOO) : {report['leak_discipline']['leak_free']} "
          f"(self-leaks={report['leak_discipline']['train_loo_self_leaks']})")
    print(f"  same-label rate : VALID {sl['valid']['rate_mean']} "
          f"(baseline {sl['valid']['random_baseline_label_prior']}, "
          f"lift +{sl['valid']['lift_over_baseline']}) | 1-NN {sl['valid']['one_nn_same_label_accuracy']}")
    print(f"  per-class VALID : {_fmt_pc(sl['valid']['per_class_rate'])}")
    print(f"  boundary gap    : VALID mean {bg['valid']['gap_mean']} "
          f"({bg['valid']['fraction_cases_positive_gap']} of cases positive)")
    print(f"  examples        : {len(report['example_contracts'])} rendered (appendix hidden by "
          f"default: {all(e['default_view_hides_appendix'] for e in report['example_contracts'])})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the JSON evidence + notes")
    args = parser.parse_args()
    started = time.perf_counter()
    report = build()
    _print_summary(report)
    print(f"\ndone in {time.perf_counter() - started:.1f}s")
    if args.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT_JSON.relative_to(REPO)}")
        _write_note(report)
        print(f"wrote {OUT_NOTE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
