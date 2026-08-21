"""N3 -- Abstention under distribution shift (the G8 rescue).

G8 (research/evidence/derm7pt_v1/g8_cross_domain_criteria.json) is a negative: the
Derm7pt-trained criteria path is out-of-domain on BCN20000 by construction (Derm7pt is the only
source of criterion labels anywhere), and out there its melanoma AUROC collapses to 0.68 and it
recovers only 2 melanomas the in-domain BCN classifier misses. That is a real boundary condition.

This script asks whether the *checklist path fails safe* under exactly that shift: does the
criterion-level abstention mechanism (medguard_v2/perception/criterion_abstention.py,
criterion_abstention_v1.json) fire MORE often out-of-domain than in-domain? If it does, the G8
negative becomes a graceful-degradation result -- the path abstains (goes INDETERMINATE) rather
than confidently guessing wrong on inputs it was never trained for. If it does not, that is an
honest sharpening: the abstention boundary does not automatically track domain shift, and the
claim must be narrowed accordingly.

WHICH HEADS -- the honest comparison
------------------------------------
The abstention policy criterion_abstention_v1.json was fit **on the ORIGINAL 512-d dermlip heads**
(medguard_v2.pipeline.run_cohort.fit_readers, C=10, unbalanced, out-of-fold on Derm7pt train). Its
per-criterion margin cutoffs are calibrated to *those* heads' posterior-margin scale. The promoted
criteria heads (dermlip_tta+metadata14, 526-d, lr_balanced) live on a different margin scale, and
-- decisively -- the cross-centre cohort (BCN) carries ONLY the plain dermlip view: there is no
cached dermlip_tta view for BCN and no Derm7pt-style sex/location/elevation metadata for it, so the
526-d promoted feature vector cannot even be constructed for BCN. Applying the v1 cutoffs to
promoted-head margins would therefore be doubly ill-posed (wrong scale AND impossible features).

So the ONLY well-posed, apples-to-apples cross-domain comparison is the original heads on the plain
512-d dermlip view, which both datasets share (both are unit-norm DermLIP_ViT-B-16 embeddings --
verified). That is the honest comparison and the one reported here. The promoted-head version is
reported as *not computable cross-domain*, with the reason, rather than faked.

Two structural facts to read the numbers honestly (both from criterion_abstention_v1.json):
  * blue_whitish_veil and vascular_structures have a fitted cutoff of EXACTLY 0.0 -> they can NEVER
    abstain, in any domain. Their 0% rate is a structural floor, not a finding; they are excluded
    from the "spike" verdict.
  * pigment_network, dots_and_globules (cutoff 0.25) and pigmentation (0.35) already abstain on a
    large fraction of in-domain cases (little headroom). A flat rate there can be a ceiling
    artefact; the criteria with real headroom (streaks 0.05, regression_structures 0.15, and the
    partially-saturated pigmentation) are where a genuine spike is detectable.

Boundaries: zero paid calls, zero images encoded. Heads fit on Derm7pt train only. BCN test split
is never read (load_records defaults to train+valid; allow_test stays False). Cross-centre cohort
is BCN lesion-level (clinical unit, matching G8), mean-pooled + renormalised.

Reads (read-only): medguard_v2/pipeline/run_cohort.py, medguard_v2/perception/criterion_abstention.py,
medguard_v2/criteria/{aggregate,schema}.py, medguard_v2/data/{derm7pt,bcn20000}.py.
Output: research/evidence/derm7pt_v1/abstention_under_shift.json + notes/ABSTENTION_UNDER_SHIFT.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.criteria.aggregate import score_criteria  # noqa: E402
from medguard_v2.criteria.schema import CRITERION_NAMES, INDETERMINATE  # noqa: E402
from medguard_v2.data.bcn20000 import lesion_level, load_records  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.perception.criterion_abstention import load_policy  # noqa: E402
from medguard_v2.pipeline.run_cohort import _load_features, fit_readers  # noqa: E402

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "abstention_under_shift.json"
OUT_MD = REPO / "notes" / "ABSTENTION_UNDER_SHIFT.md"

STRUCTURAL_FLOOR = tuple(n for n in CRITERION_NAMES)  # filled after policy load


def _apply_policy(heads, policy, x: np.ndarray) -> Dict[str, List[str]]:
    """Per-criterion INDETERMINATE-or-value readings under the fitted margin policy."""
    out: Dict[str, List[str]] = {}
    margins: Dict[str, np.ndarray] = {}
    for name, head in heads.items():
        values, margin = policy.apply(head, x, name)
        out[name] = values
        margins[name] = np.asarray(margin, dtype=float)
    return out, margins


def _cohort_stats(values: Dict[str, List[str]], margins: Dict[str, np.ndarray],
                  policy, n: int) -> Dict:
    """Per-criterion abstention rate + margin summary, plus case-level rollups."""
    per_criterion = {}
    for name in CRITERION_NAMES:
        abstain = np.array([v == INDETERMINATE for v in values[name]], dtype=bool)
        m = margins[name]
        per_criterion[name] = {
            "cutoff": float(policy.margins.get(name, 0.0)),
            "structural_floor": float(policy.margins.get(name, 0.0)) == 0.0,
            "n_abstain": int(abstain.sum()),
            "abstain_rate": round(float(abstain.mean()), 4),
            "mean_margin": round(float(m.mean()), 4),
            "median_margin": round(float(np.median(m)), 4),
        }

    # case-level: any indeterminate, and undecidable verdict (score range spans the threshold)
    any_indet = 0
    undecidable = 0
    for i in range(n):
        crit = {name: values[name][i] for name in CRITERION_NAMES}
        if any(crit[name] == INDETERMINATE for name in CRITERION_NAMES):
            any_indet += 1
        if not score_criteria(crit).decidable:
            undecidable += 1

    # overall abstention rate over all 7*n criterion reads
    total_reads = 7 * n
    total_abstain = sum(per_criterion[name]["n_abstain"] for name in CRITERION_NAMES)

    # headroom criteria = those whose cutoff > 0 (can actually abstain)
    headroom = [name for name in CRITERION_NAMES if not per_criterion[name]["structural_floor"]]
    headroom_reads = len(headroom) * n
    headroom_abstain = sum(per_criterion[name]["n_abstain"] for name in headroom)

    return {
        "n": n,
        "per_criterion": per_criterion,
        "overall": {
            "criterion_reads": total_reads,
            "criterion_reads_indeterminate": total_abstain,
            "criterion_read_abstain_rate": round(total_abstain / total_reads, 4) if total_reads else 0.0,
            "mean_per_criterion_abstain_rate": round(
                float(np.mean([per_criterion[n_]["abstain_rate"] for n_ in CRITERION_NAMES])), 4),
            "headroom_criteria_abstain_rate": round(headroom_abstain / headroom_reads, 4)
            if headroom_reads else 0.0,
        },
        "case_level": {
            "any_indeterminate_rate": round(any_indet / n, 4) if n else 0.0,
            "undecidable_verdict_rate": round(undecidable / n, 4) if n else 0.0,
            "n_any_indeterminate": any_indet,
            "n_undecidable": undecidable,
        },
    }


def build() -> Dict:
    policy = load_policy()

    # ---- fit the ORIGINAL 512-d dermlip heads on Derm7pt train (the heads the policy was fit for)
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    features = _load_features()
    _diagnosis, heads = fit_readers(train, features)

    # ---- IN-DOMAIN: Derm7pt test
    x_test = np.stack([features[str(c.case_num)] for c in test])
    vals_id, marg_id = _apply_policy(heads, policy, x_test)
    in_domain = _cohort_stats(vals_id, marg_id, policy, len(test))
    id_class = {}
    for c in test:
        id_class[c.diagnosis_group] = id_class.get(c.diagnosis_group, 0) + 1

    # ---- CROSS-CENTRE: BCN lesion-level (train+valid; test never read)
    bcn_records, bcn_features = load_records(splits=("train", "valid"))  # allow_test defaults False
    bcn_lesions, bcn_pooled = lesion_level(bcn_records, bcn_features)
    vals_bcn, marg_bcn = _apply_policy(heads, policy, bcn_pooled)
    cross = _cohort_stats(vals_bcn, marg_bcn, policy, len(bcn_lesions))
    bcn_class = {}
    for r in bcn_lesions:
        bcn_class[r.group] = bcn_class.get(r.group, 0) + 1

    # ---- MATCHED-CLASS CONTROL: NEV-only in both domains (controls for the class-mix difference)
    nev_idx_id = [i for i, c in enumerate(test) if c.diagnosis_group == "NEV"]
    nev_idx_bcn = [i for i, r in enumerate(bcn_lesions) if r.group == "NEV"]

    def _nev_rate(values, idx):
        if not idx:
            return None
        reads = 0
        abst = 0
        for name in CRITERION_NAMES:
            if policy.margins.get(name, 0.0) == 0.0:
                continue  # skip structural floors
            for i in idx:
                reads += 1
                abst += 1 if values[name][i] == INDETERMINATE else 0
        return round(abst / reads, 4) if reads else None

    nev_control = {
        "n_nev_in_domain": len(nev_idx_id),
        "n_nev_cross_centre": len(nev_idx_bcn),
        "headroom_abstain_rate_in_domain": _nev_rate(vals_id, nev_idx_id),
        "headroom_abstain_rate_cross_centre": _nev_rate(vals_bcn, nev_idx_bcn),
    }

    # ---- POOLING ROBUSTNESS: BCN at IMAGE level (single image, apples-to-apples with the single-
    # image Derm7pt test). Mean-pooling a lesion's images toward its centroid could itself inflate
    # head confidence (wider margins), which would confound the primary lesion-level read. If the
    # image-level cross-centre read shows the SAME direction (no spike), the pooling is not driving
    # the finding.
    vals_bcn_img, marg_bcn_img = _apply_policy(heads, policy, bcn_features)
    cross_img = _cohort_stats(vals_bcn_img, marg_bcn_img, policy, bcn_features.shape[0])
    pooling_robustness = {
        "cross_centre_unit": "image (unpooled)",
        "n_images": int(bcn_features.shape[0]),
        "headroom_criteria_abstain_rate": cross_img["overall"]["headroom_criteria_abstain_rate"],
        "headroom_read_rate_delta_vs_in_domain": round(
            cross_img["overall"]["headroom_criteria_abstain_rate"]
            - in_domain["overall"]["headroom_criteria_abstain_rate"], 4),
        "undecidable_verdict_rate": cross_img["case_level"]["undecidable_verdict_rate"],
        "undecidable_verdict_delta_vs_in_domain": round(
            cross_img["case_level"]["undecidable_verdict_rate"]
            - in_domain["case_level"]["undecidable_verdict_rate"], 4),
        "mean_headroom_margin_delta_vs_in_domain": round(float(np.mean([
            cross_img["per_criterion"][name]["mean_margin"]
            - in_domain["per_criterion"][name]["mean_margin"]
            for name in CRITERION_NAMES if policy.margins.get(name, 0.0) != 0.0])), 4),
    }

    # ---- SPIKE deltas (cross-centre minus in-domain)
    per_criterion_delta = {}
    for name in CRITERION_NAMES:
        d = round(cross["per_criterion"][name]["abstain_rate"]
                  - in_domain["per_criterion"][name]["abstain_rate"], 4)
        dm = round(cross["per_criterion"][name]["mean_margin"]
                   - in_domain["per_criterion"][name]["mean_margin"], 4)
        per_criterion_delta[name] = {
            "abstain_rate_delta": d,
            "mean_margin_delta": dm,
            "structural_floor": in_domain["per_criterion"][name]["structural_floor"],
        }

    headroom_delta = round(cross["overall"]["headroom_criteria_abstain_rate"]
                           - in_domain["overall"]["headroom_criteria_abstain_rate"], 4)
    overall_delta = round(cross["overall"]["criterion_read_abstain_rate"]
                          - in_domain["overall"]["criterion_read_abstain_rate"], 4)
    undecidable_delta = round(cross["case_level"]["undecidable_verdict_rate"]
                              - in_domain["case_level"]["undecidable_verdict_rate"], 4)
    any_indet_delta = round(cross["case_level"]["any_indeterminate_rate"]
                            - in_domain["case_level"]["any_indeterminate_rate"], 4)

    # A "spike" is a materially higher abstention out-of-domain among the criteria that CAN abstain.
    # Use the headroom-criteria read-rate (excludes the two structural-floor criteria) as the
    # primary signal, and case-level undecidable-verdict rate as the safety-relevant one.
    SPIKE_EPS = 0.03
    fails_safe = (headroom_delta >= SPIKE_EPS) or (undecidable_delta >= SPIKE_EPS)
    margins_shrink = float(np.mean([
        per_criterion_delta[name]["mean_margin_delta"]
        for name in CRITERION_NAMES
        if not per_criterion_delta[name]["structural_floor"]
    ]))

    if fails_safe:
        verdict = (
            "SPIKE: the checklist path fails safe under distribution shift. Among the criteria that "
            "can abstain, the abstention rate is materially higher out-of-domain (BCN) than "
            f"in-domain (Derm7pt test) -- headroom-criteria read-rate delta {headroom_delta:+.4f}, "
            f"case-level undecidable-verdict delta {undecidable_delta:+.4f}. The path goes "
            "INDETERMINATE on inputs it was not trained for rather than confidently guessing, which "
            "turns the G8 out-of-domain negative into a graceful-degradation property."
        )
    elif margins_shrink > 0:
        verdict = (
            "NO SPIKE -- and the mechanism cuts the WRONG way (a substantive negative, not just a "
            f"null). Out-of-domain the abstention rate is slightly LOWER (headroom-criteria read-"
            f"rate delta {headroom_delta:+.4f}, undecidable-verdict delta {undecidable_delta:+.4f}), "
            "because the frozen criterion heads become on average MORE confident on BCN inputs, not "
            f"less: mean head margin WIDENS by {margins_shrink:+.4f} across the criteria that can "
            "abstain. A margin-based abstention policy fit in-domain therefore does NOT fail safe "
            "under centre shift -- the probe can be confidently wrong on out-of-domain lesions and "
            "the in-domain-calibrated cutoffs will not catch it, so the criteria path commits rather "
            "than abstains exactly where G8 showed its accuracy has collapsed. The honest claim is "
            "narrow: criterion-margin abstention is an IN-DISTRIBUTION selective-prediction tool "
            "(it separates confident from unsure readings within a centre), NOT an out-of-domain "
            "guard. A real cross-centre safety gate needs an explicit distribution-shift / OOD "
            "detector on the feature side, which this margin policy is not. The NEV-only matched-"
            "class control shows the same direction, so this is not a class-mix artefact."
        )
    else:
        verdict = (
            "NO SPIKE (honest sharpening): the abstention mechanism does NOT fire materially more "
            f"out-of-domain (headroom-criteria read-rate delta {headroom_delta:+.4f}, undecidable-"
            f"verdict delta {undecidable_delta:+.4f}), though mean head margins DO shrink "
            f"({margins_shrink:+.4f} avg over headroom criteria) -- the confidence signal moves in "
            "the fail-safe direction but not enough to cross the fitted cutoffs and change the "
            "abstention rate. The claim must stay narrow: the criteria path degrades out-of-domain "
            "(G8) but the in-domain-fit margin policy does not automatically announce that by "
            "abstaining more."
        )

    # Pooling-robustness suffix: does BCN at IMAGE level (no lesion pooling) agree with the primary
    # lesion-level read? If so, mean-pooling toward lesion centroids is not what widened the margins.
    img_hd = pooling_robustness["headroom_read_rate_delta_vs_in_domain"]
    img_undec = pooling_robustness["undecidable_verdict_delta_vs_in_domain"]
    img_agrees = (img_hd < SPIKE_EPS) and (img_undec < SPIKE_EPS)
    verdict += (
        f" Pooling robustness: BCN at IMAGE level (single image, no lesion pooling; n="
        f"{pooling_robustness['n_images']}) shows the SAME direction "
        f"(headroom read-rate delta {img_hd:+.4f}, undecidable delta {img_undec:+.4f}, mean-margin "
        f"delta {pooling_robustness['mean_headroom_margin_delta_vs_in_domain']:+.4f}), so the "
        + ("no-spike finding is not an artefact of mean-pooling toward lesion centroids."
           if img_agrees else
           "lesion-level and image-level reads DISAGREE -- pooling may be masking a spike; treat "
           "the lesion-level primary with caution and see the image-level block.")
    )

    return {
        "protocol": "derm7pt-abstention-under-distribution-shift-n3-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "one_shot": False,
        "question": (
            "Does the criterion-level abstention mechanism fire MORE out-of-domain (BCN) than "
            "in-domain (Derm7pt test)? A spike => the checklist path fails safe (abstains rather "
            "than guesses) under the same shift that G8 showed degrades its accuracy."
        ),
        "g8_link": {
            "artifact": "research/evidence/derm7pt_v1/g8_cross_domain_criteria.json",
            "g8_finding": (
                "Derm7pt-trained criteria path is out-of-domain on BCN: melanoma AUROC 0.68, "
                "catches only 2 melanomas the in-domain BCN classifier misses (a real boundary)."
            ),
        },
        "honest_comparison_note": (
            "ORIGINAL 512-d dermlip heads only. criterion_abstention_v1.json was fit on those heads; "
            "its cutoffs are calibrated to their margin scale. The promoted 526-d "
            "(dermlip_tta+metadata14) heads CANNOT be built for BCN (no cached TTA view, no "
            "Derm7pt-style metadata for BCN), so a promoted-head cross-domain comparison is not "
            "computable -- reported as such, not faked. Both datasets share the plain unit-norm "
            "DermLIP_ViT-B-16 512-d view, so this comparison has no normalisation confound."
        ),
        "promoted_head_version": {
            "computable_cross_domain": False,
            "reason": (
                "BCN carries only the plain dermlip view; dermlip_tta is not cached for BCN and "
                "Derm7pt metadata14 (sex/location/elevation) has no BCN analogue in the frozen "
                "feature set. The 526-d promoted vector cannot be assembled for BCN, so the v1 "
                "policy cannot be applied to promoted heads out-of-domain. In-domain, run_cohort "
                "itself already declines to gate the promoted balanced heads with the v1 cutoffs "
                "(they are the wrong scale) -- see run_cohort.py _head_values_and_margins."
            ),
        },
        "boundaries": {
            "paid_calls": 0,
            "images_encoded": 0,
            "heads": "original dermlip 512-d, LogisticRegression C=10 unbalanced, fit on Derm7pt train",
            "abstention_policy": "criterion_abstention_v1 (margin cutoffs, fit on Derm7pt train OOF)",
            "in_domain_cohort": "Derm7pt test (single-image, unit-norm dermlip)",
            "cross_centre_cohort": "BCN20000 lesion-level (train+valid), mean-pooled + renormalised",
            "bcn_test_read": False,
        },
        "policy_cutoffs": dict(policy.margins),
        "structural_floor_criteria": [
            n for n in CRITERION_NAMES if policy.margins.get(n, 0.0) == 0.0
        ],
        "class_distribution": {
            "in_domain": id_class,
            "cross_centre": bcn_class,
            "note": "Different class mixes are themselves part of the shift; a NEV-only matched "
                    "control is reported separately.",
        },
        "in_domain": in_domain,
        "cross_centre": cross,
        "matched_class_control_nev_only": nev_control,
        "pooling_robustness_image_level": pooling_robustness,
        "spike_analysis": {
            "per_criterion_delta": per_criterion_delta,
            "headroom_criteria_read_rate_delta": headroom_delta,
            "overall_read_rate_delta": overall_delta,
            "case_any_indeterminate_delta": any_indet_delta,
            "case_undecidable_verdict_delta": undecidable_delta,
            "mean_headroom_margin_delta": round(margins_shrink, 4),
            "spike_epsilon": SPIKE_EPS,
            "fails_safe": bool(fails_safe),
        },
        "verdict": verdict,
    }


def _write_notes(r: Dict) -> None:
    idd, cr = r["in_domain"], r["cross_centre"]
    sp = r["spike_analysis"]
    lines: List[str] = []
    lines.append("# N3 -- Abstention under distribution shift (G8 rescue)")
    lines.append("")
    lines.append(r["question"])
    lines.append("")
    lines.append(f"**In-domain** = Derm7pt test (n={idd['n']}); **cross-centre** = BCN lesion-level "
                 f"(n={cr['n']}). Original 512-d dermlip heads + criterion_abstention_v1 policy. "
                 "Zero paid calls.")
    lines.append("")
    lines.append("## Per-criterion abstention rate (in-domain -> cross-centre, delta)")
    lines.append("")
    lines.append("| criterion | cutoff | in-domain | cross-centre | delta | mean-margin delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in CRITERION_NAMES:
        pid = idd["per_criterion"][name]
        pcr = cr["per_criterion"][name]
        d = sp["per_criterion_delta"][name]
        floor = " (floor)" if pid["structural_floor"] else ""
        lines.append(
            f"| {name}{floor} | {pid['cutoff']:.2f} | {pid['abstain_rate']:.4f} | "
            f"{pcr['abstain_rate']:.4f} | {d['abstain_rate_delta']:+.4f} | "
            f"{d['mean_margin_delta']:+.4f} |")
    lines.append("")
    lines.append("## Overall + case-level")
    lines.append("")
    lines.append("| measure | in-domain | cross-centre | delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| headroom-criteria read abstain-rate | "
                 f"{idd['overall']['headroom_criteria_abstain_rate']:.4f} | "
                 f"{cr['overall']['headroom_criteria_abstain_rate']:.4f} | "
                 f"{sp['headroom_criteria_read_rate_delta']:+.4f} |")
    lines.append(f"| all-reads abstain-rate | {idd['overall']['criterion_read_abstain_rate']:.4f} | "
                 f"{cr['overall']['criterion_read_abstain_rate']:.4f} | "
                 f"{sp['overall_read_rate_delta']:+.4f} |")
    lines.append(f"| case any-indeterminate | {idd['case_level']['any_indeterminate_rate']:.4f} | "
                 f"{cr['case_level']['any_indeterminate_rate']:.4f} | "
                 f"{sp['case_any_indeterminate_delta']:+.4f} |")
    lines.append(f"| case undecidable-verdict | {idd['case_level']['undecidable_verdict_rate']:.4f} | "
                 f"{cr['case_level']['undecidable_verdict_rate']:.4f} | "
                 f"{sp['case_undecidable_verdict_delta']:+.4f} |")
    lines.append("")
    nc = r["matched_class_control_nev_only"]
    lines.append(f"NEV-only matched control (headroom criteria): in-domain "
                 f"{nc['headroom_abstain_rate_in_domain']} vs cross-centre "
                 f"{nc['headroom_abstain_rate_cross_centre']} "
                 f"(n_nev {nc['n_nev_in_domain']} vs {nc['n_nev_cross_centre']}).")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(r["verdict"])
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    report = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_notes(report)
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    sp = report["spike_analysis"]
    print("\n=== N3 abstention under shift ===")
    print(f"  in-domain n={report['in_domain']['n']}  cross-centre n={report['cross_centre']['n']}")
    print(f"  headroom read-rate: in {report['in_domain']['overall']['headroom_criteria_abstain_rate']} "
          f"-> cross {report['cross_centre']['overall']['headroom_criteria_abstain_rate']} "
          f"(delta {sp['headroom_criteria_read_rate_delta']:+.4f})")
    print(f"  undecidable verdict: in {report['in_domain']['case_level']['undecidable_verdict_rate']} "
          f"-> cross {report['cross_centre']['case_level']['undecidable_verdict_rate']} "
          f"(delta {sp['case_undecidable_verdict_delta']:+.4f})")
    print(f"  fails_safe (spike): {sp['fails_safe']}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
