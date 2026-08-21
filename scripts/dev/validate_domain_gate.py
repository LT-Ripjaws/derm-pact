"""R-A -- Domain-gate validation: can TRAIN-bank similarity see what margins cannot?

N3 (research/evidence/derm7pt_v1/abstention_under_shift.json) is a negative with a sharp edge:
under the measured Derm7pt->BCN centre shift the criterion heads get MORE confident, so a
margin-based abstention policy can never gate out-of-domain inputs. The mechanism that could is
feature-space similarity to the frozen TRAIN bank -- medguard_v2/perception/domain_check.py.
This script measures whether it actually does, with the populations kept honest:

  * IN-DOMAIN POSITIVES : Derm7pt VALID (n=203). Also the ONLY population the threshold is fit
    on (p1 of its domain scores -- ~99% of in-domain images pass by construction).
  * SHIFT POPULATION    : BCN20000 train+valid images (test never read). BCN is dermoscopy too,
    so this asks specifically whether CENTRE shift is visible to bank similarity. Image level is
    primary (the gate's deployment unit is one image); lesion-pooled is a secondary view.
  * SPILL CHECK         : Derm7pt TEST domain scores -- what fraction of legitimate in-domain
    images the gate would wrongly flag. This is a domain-score histogram over the frozen feature
    cache; no label is used and no model is evaluated, so it is a legitimate test read (recorded
    in boundaries).
  * GROSS-OOD SANITY    : synthetic non-dermoscopic images (seeded uniform noise, flat grey) and
    a few in-repo matplotlib figure renderings, freshly encoded with the frozen DermLIP encoder
    (local, zero paid calls). The figure sheets CONTAIN dermoscopy crops, so they are a
    conservative probe -- labelled as such, not oversold. Skipped gracefully (recorded) if the
    encoder cannot load; pass --skip-sanity to skip explicitly.

k SELECTION HONESTY: the gate's k is selected from {1,3,5} by the valid-vs-BCN separation AUROC
here -- so BCN DOES touch the k choice (a pre-registered 3-point grid, disclosed, with all three
candidates' numbers reported so the sensitivity is visible). BCN NEVER touches the threshold:
that is p1 of valid scores at the chosen k. Nothing else is tuned on BCN, and the verdict is
reported plainly whichever way the separation lands -- a weak AUROC here does not get massaged,
it SHARPENS N3 ("centre shift is not detectable by bank similarity either") and the paper says
so; the gate still earns its place against gross non-dermoscopic inputs.

Outputs: research/evidence/derm7pt_v1/domain_gate_validation.json + notes/DOMAIN_GATE_VALIDATION.md
Run with the feature venv (venv_win\\Scripts\\python.exe): the sanity block loads the encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.data.bcn20000 import lesion_level, load_records  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.perception.domain_check import (  # noqa: E402
    DOMAIN_GATE_K,
    DOMAIN_GATE_VALID_PERCENTILE,
    _load_dermlip_features,
    build_train_bank,
    domain_scores,
    fit_valid_threshold,
    load_domain_gate,
)

OUT_JSON = REPO / "research" / "evidence" / "derm7pt_v1" / "domain_gate_validation.json"
OUT_MD = REPO / "notes" / "DOMAIN_GATE_VALIDATION.md"

K_GRID = (1, 3, 5)
SANITY_SEED = 20260815

#: In-repo figure renderings used as conservative gross-OOD probes (they are matplotlib
#: canvases, not raw dermoscopy photographs -- but they CONTAIN dermoscopy crops).
FIGURE_PROBES = (
    REPO / "research" / "evidence" / "derm7pt_v1" / "g1c_exemplar_sheets" / "pigment_network.png",
    REPO / "research" / "evidence" / "derm7pt_v1" / "g1c_exemplar_sheets" / "streaks.png",
    REPO / "research" / "evidence" / "derm7pt_v1" / "attribution_samples" / "case_2_diagnosis.png",
)


def rank_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC = P(pos > neg) + 0.5*P(tie), via tie-averaged ranks."""
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    n = scores.shape[0]
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0  # average rank, 1-based
        i = j
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _stats(scores: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(scores.shape[0]),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "min": round(float(scores.min()), 4),
        "p1": round(float(np.percentile(scores, 1)), 4),
        "p5": round(float(np.percentile(scores, 5)), 4),
        "median": round(float(np.median(scores)), 4),
        "max": round(float(scores.max()), 4),
    }


def _flagged(scores: np.ndarray, threshold: float) -> Dict[str, object]:
    below = scores < threshold
    return {"n_flagged": int(below.sum()), "flagged_fraction": round(float(below.mean()), 4)}


def _sanity_probes(bank, k: int, threshold: float, skip: bool) -> Dict[str, object]:
    """Encode synthetic non-dermoscopic images + in-repo figure PNGs and score them."""
    if skip:
        return {"status": "skipped (--skip-sanity)", "probes": []}
    try:
        from PIL import Image

        from medguard_v2.perception.encode import load_encoder

        encode, provenance = load_encoder("dermlip")
    except Exception as exc:  # encoder or PIL unavailable -- record, do not fail the run
        return {"status": f"skipped (encoder unavailable: {exc})", "probes": []}

    import tempfile

    probes: List[Tuple[str, Path]] = []
    tmp = Path(tempfile.mkdtemp(prefix="domain_gate_sanity_"))
    rng = np.random.default_rng(SANITY_SEED)
    noise = (rng.uniform(0, 255, size=(448, 448, 3))).astype(np.uint8)
    noise_path = tmp / "uniform_noise.png"
    Image.fromarray(noise).save(noise_path)
    probes.append(("synthetic: seeded uniform RGB noise", noise_path))
    grey = np.full((448, 448, 3), 128, dtype=np.uint8)
    grey_path = tmp / "flat_grey.png"
    Image.fromarray(grey).save(grey_path)
    probes.append(("synthetic: flat mid-grey", grey_path))
    for fig in FIGURE_PROBES:
        if fig.exists():
            probes.append((f"in-repo figure rendering: {fig.relative_to(REPO)}", fig))

    results = []
    for label, path in probes:
        vec = np.asarray(encode([path])[0], dtype=np.float64)
        score = float(domain_scores(bank, vec, k)[0])
        results.append({
            "probe": label,
            "domain_score": round(score, 4),
            "flagged_ood": bool(score < threshold),
        })
    return {
        "status": "ran",
        "encoder": {k_: provenance[k_] for k_ in ("model_id",) if k_ in provenance} or "dermlip",
        "note": (
            "The figure probes are matplotlib canvases CONTAINING dermoscopy crops -- a "
            "conservative gross-OOD probe, deliberately harder than a photo of a cat. The "
            "synthetic probes contain no dermoscopic content at all."
        ),
        "probes": results,
    }


def build(skip_sanity: bool) -> Dict:
    cases = load_cases()
    features = _load_dermlip_features()
    bank = build_train_bank(features, cases)

    valid = np.stack([features[str(c.case_num)] for c in cases if c.split == "valid"])
    test = np.stack([features[str(c.case_num)] for c in cases if c.split == "test"])

    bcn_records, bcn_images = load_records(splits=("train", "valid"))  # test never read
    bcn_lesion_records, bcn_lesions = lesion_level(bcn_records, bcn_images)

    # ---- k grid: scores, valid-p1 threshold, separation, flag fractions ----------------------
    k_results = {}
    for k in K_GRID:
        s_valid = domain_scores(bank, valid, k)
        s_test = domain_scores(bank, test, k)
        s_bcn_img = domain_scores(bank, bcn_images, k)
        s_bcn_les = domain_scores(bank, bcn_lesions, k)
        fit = fit_valid_threshold(bank, valid, k=k)
        thr = fit["threshold"]
        k_results[k] = {
            "threshold_p1_valid": round(thr, 6),
            "auroc_valid_vs_bcn_image": round(rank_auroc(s_valid, s_bcn_img), 4),
            "auroc_valid_vs_bcn_lesion": round(rank_auroc(s_valid, s_bcn_les), 4),
            "scores": {
                "derm7pt_valid": _stats(s_valid),
                "derm7pt_test": _stats(s_test),
                "bcn_image": _stats(s_bcn_img),
                "bcn_lesion_pooled": _stats(s_bcn_les),
            },
            "flagged_at_threshold": {
                "derm7pt_valid": _flagged(s_valid, thr),
                "derm7pt_test": _flagged(s_test, thr),
                "bcn_image": _flagged(s_bcn_img, thr),
                "bcn_lesion_pooled": _flagged(s_bcn_les, thr),
            },
        }

    # ---- k selection: argmax image-level separation AUROC (ties -> smaller k) ----------------
    selected_k = max(K_GRID, key=lambda k: (k_results[k]["auroc_valid_vs_bcn_image"], -k))
    registered_matches = selected_k == DOMAIN_GATE_K

    gate = load_domain_gate()
    reg = k_results[DOMAIN_GATE_K]
    auroc = reg["auroc_valid_vs_bcn_image"]
    bcn_flag = reg["flagged_at_threshold"]["bcn_image"]["flagged_fraction"]
    test_flag = reg["flagged_at_threshold"]["derm7pt_test"]["flagged_fraction"]
    valid_flag = reg["flagged_at_threshold"]["derm7pt_valid"]["flagged_fraction"]

    sanity = _sanity_probes(bank, DOMAIN_GATE_K, gate.threshold, skip_sanity)
    sanity_flagged = [p for p in sanity.get("probes", []) if p["flagged_ood"]]
    synthetic = [p for p in sanity.get("probes", []) if p["probe"].startswith("synthetic")]
    synthetic_all_flagged = bool(synthetic) and all(p["flagged_ood"] for p in synthetic)

    # ---- the honest verdict, whichever way it lands ------------------------------------------
    # Two facts are kept separate: how well the score RANKS centre shift (AUROC), and how much
    # of the shift population the registered valid-p1 operating point actually GATES (the
    # threshold is chosen to keep ~99% of in-domain images, not to maximise BCN capture).
    if auroc >= 0.90:
        centre_shift = (
            f"CENTRE SHIFT IS DETECTABLE IN RANK: valid-vs-BCN image AUROC {auroc:.4f} -- "
            "bank similarity sees the cross-centre shift that N3 showed head margins cannot "
            "(margins move the WRONG way; this score moves the right way). At the registered "
            f"valid-p1 operating point (kept deliberately conservative so ~99% of in-domain "
            f"images pass), {bcn_flag:.1%} of BCN images fall below threshold and are "
            "auto-downgraded to the single-view report. "
            + ("The majority of the measured shift population is gated."
               if bcn_flag >= 0.5 else
               "The upper tail of the BCN score distribution overlaps the in-domain "
               "distribution, so the conservative threshold converts strong rank separation "
               "into a partial gate: the closest-to-domain BCN images pass. The gate "
               "materially NARROWS the N3 boundary for centre shift; it does not close it, "
               "and the paper should quote both numbers rather than the AUROC alone.")
        )
    elif auroc >= 0.75:
        centre_shift = (
            f"CENTRE SHIFT IS PARTIALLY DETECTABLE: valid-vs-BCN image AUROC {auroc:.4f}, and "
            f"{bcn_flag:.1%} of BCN images fall below the valid-p1 threshold. The score "
            "distributions are shifted but overlap substantially. The gate softens, not "
            "solves, the N3 boundary for centre shift."
        )
    else:
        centre_shift = (
            f"CENTRE SHIFT IS NOT (or only weakly) DETECTABLE by bank similarity either: "
            f"valid-vs-BCN image AUROC {auroc:.4f}, and only {bcn_flag:.1%} of BCN images fall "
            "below the valid-p1 threshold. BCN is dermoscopy too -- same modality, different "
            "centre -- and in frozen DermLIP space that difference is mostly invisible to "
            "top-k cosine. This SHARPENS N3 rather than fixing it: neither head margins (N3) "
            "nor train-bank similarity (here) announce the cross-centre shift under which G8 "
            "showed the checklist path degrades, and the paper must say so."
        )
    if sanity.get("status") == "ran" and synthetic:
        gross = (
            f" Gross non-dermoscopic inputs: {len(sanity_flagged)}/{len(sanity['probes'])} "
            f"sanity probes flagged; the fully synthetic probes (no dermoscopic content) were "
            + ("ALL flagged, so the gate does catch inputs that are not dermoscopy at all."
               if synthetic_all_flagged else
               "NOT all flagged -- the gate's gross-OOD claim is weaker than expected and must "
               "be stated with that caveat.")
        )
    else:
        gross = " Gross-OOD sanity probes were skipped (see sanity_probes.status)."
    verdict = centre_shift + gross + (
        f" Spill on legitimate in-domain images: {test_flag:.2%} of Derm7pt TEST flagged "
        f"(valid, the fitting population, {valid_flag:.2%} by construction)."
    )

    bcn_split_counts = {}
    for r in bcn_lesion_records:
        bcn_split_counts[r.group] = bcn_split_counts.get(r.group, 0) + 1

    return {
        "protocol": "derm7pt-domain-gate-validation-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": (
            "Does mean top-k cosine to the frozen TRAIN dermlip bank separate in-domain "
            "dermoscopy (Derm7pt valid) from the measured cross-centre shift population "
            "(BCN20000), and from gross non-dermoscopic inputs? N3 showed criterion margins "
            "move the WRONG way under this shift; this is the mechanism that can gate, if "
            "anything can."
        ),
        "n3_link": {
            "artifact": "research/evidence/derm7pt_v1/abstention_under_shift.json",
            "n3_finding": (
                "criterion-head margins WIDEN on BCN (mean headroom-margin delta +, abstention "
                "rate flat/lower): margin-based abstention cannot gate out-of-domain inputs"
            ),
        },
        "boundaries": {
            "paid_calls": 0,
            "images_encoded": (
                0 if sanity.get("status") != "ran" else len(sanity.get("probes", []))
            ),
            "encoding_note": "only the sanity probes are freshly encoded (local frozen DermLIP)",
            "bank": "Derm7pt TRAIN only, 512-d plain dermlip image embedding, hash-versioned",
            "threshold_fit": "Derm7pt VALID only (p1); BCN NEVER touches the threshold",
            "k_selection": (
                "k in {1,3,5} selected by valid-vs-BCN image AUROC -- BCN touches the k choice "
                "(pre-registered 3-point grid, all candidates reported) but not the threshold"
            ),
            "bcn_test_read": False,
            "derm7pt_test_read": (
                "feature cache only, for the flagged-fraction histogram; no label used, no "
                "model evaluated (authorized domain-score read)"
            ),
        },
        "populations": {
            "derm7pt_valid": {"n": int(valid.shape[0]), "role": "in-domain positives + threshold fit"},
            "derm7pt_test": {"n": int(test.shape[0]), "role": "in-domain spill check"},
            "bcn_image": {
                "n": int(bcn_images.shape[0]),
                "role": "cross-centre shift population, image level (PRIMARY: the gate's "
                        "deployment unit is a single image)",
            },
            "bcn_lesion_pooled": {
                "n": int(bcn_lesions.shape[0]),
                "role": "cross-centre, lesion mean-pooled + renormalised (secondary view, "
                        "matches N3's primary unit)",
                "group_counts": dict(sorted(bcn_split_counts.items())),
            },
        },
        "k_grid": {str(k): k_results[k] for k in K_GRID},
        "k_selection": {
            "grid": list(K_GRID),
            "criterion": "argmax valid-vs-BCN image-level AUROC (ties -> smaller k)",
            "selected_k": int(selected_k),
            "registered_k": int(DOMAIN_GATE_K),
            "registered_matches_selected": bool(registered_matches),
        },
        "registered_gate": {
            **gate.summary(),
            "valid_percentile": DOMAIN_GATE_VALID_PERCENTILE,
        },
        "headline_at_registered_gate": {
            "auroc_valid_vs_bcn_image": auroc,
            "auroc_valid_vs_bcn_lesion": reg["auroc_valid_vs_bcn_lesion"],
            "bcn_image_flagged_fraction": bcn_flag,
            "bcn_lesion_flagged_fraction":
                reg["flagged_at_threshold"]["bcn_lesion_pooled"]["flagged_fraction"],
            "derm7pt_test_flagged_fraction": test_flag,
            "derm7pt_valid_flagged_fraction": valid_flag,
        },
        "sanity_probes": sanity,
        "verdict": verdict,
    }


def _write_notes(r: Dict) -> None:
    head = r["headline_at_registered_gate"]
    reg = r["registered_gate"]
    lines: List[str] = []
    lines.append("# Domain-gate validation (R-A: the gate N3 said margins cannot be)")
    lines.append("")
    lines.append(r["question"])
    lines.append("")
    lines.append(
        f"**Gate**: mean top-{reg['k']} cosine to the TRAIN dermlip bank "
        f"(n={reg['bank']['n_entries']}, hash `{reg['bank']['bank_hash'][:16]}...`); "
        f"threshold {reg['threshold']:.4f} = p1 of Derm7pt VALID scores (BCN never fit on)."
    )
    lines.append("")
    lines.append("## Separation + flag fractions (registered k)")
    lines.append("")
    lines.append("| measure | value |")
    lines.append("|---|---:|")
    lines.append(f"| AUROC valid vs BCN (image, primary) | {head['auroc_valid_vs_bcn_image']:.4f} |")
    lines.append(f"| AUROC valid vs BCN (lesion-pooled) | {head['auroc_valid_vs_bcn_lesion']:.4f} |")
    lines.append(f"| BCN images flagged OOD | {head['bcn_image_flagged_fraction']:.4f} |")
    lines.append(f"| BCN lesions flagged OOD | {head['bcn_lesion_flagged_fraction']:.4f} |")
    lines.append(f"| Derm7pt TEST flagged (spill) | {head['derm7pt_test_flagged_fraction']:.4f} |")
    lines.append(f"| Derm7pt VALID flagged (by construction ~1%) | {head['derm7pt_valid_flagged_fraction']:.4f} |")
    lines.append("")
    lines.append("## k sensitivity (threshold refit at each k; selection disclosed)")
    lines.append("")
    lines.append("| k | AUROC (image) | AUROC (lesion) | BCN img flagged | test flagged |")
    lines.append("|---:|---:|---:|---:|---:|")
    for k in r["k_selection"]["grid"]:
        row = r["k_grid"][str(k)]
        mark = " (registered)" if k == r["k_selection"]["registered_k"] else ""
        lines.append(
            f"| {k}{mark} | {row['auroc_valid_vs_bcn_image']:.4f} | "
            f"{row['auroc_valid_vs_bcn_lesion']:.4f} | "
            f"{row['flagged_at_threshold']['bcn_image']['flagged_fraction']:.4f} | "
            f"{row['flagged_at_threshold']['derm7pt_test']['flagged_fraction']:.4f} |"
        )
    lines.append("")
    sanity = r["sanity_probes"]
    lines.append(f"## Gross-OOD sanity probes ({sanity['status']})")
    lines.append("")
    for p in sanity.get("probes", []):
        lines.append(
            f"- {p['probe']}: score {p['domain_score']:.4f} -> "
            f"{'FLAGGED OOD' if p['flagged_ood'] else 'passed'}"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(r["verdict"])
    lines.append("")
    lines.append(
        "Wiring: medguard_v2/pipeline/predict_one.py now scores every input against this gate "
        "before the heads; below-threshold inputs are auto-downgraded to the single-view "
        "report with the bank-similarity disclosure (force_dual_view=True bypasses, recorded)."
    )
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-sanity", action="store_true",
                        help="skip the encoder-dependent gross-OOD sanity probes")
    args = parser.parse_args()

    t0 = time.perf_counter()
    report = build(skip_sanity=args.skip_sanity)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_notes(report)
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")

    head = report["headline_at_registered_gate"]
    sel = report["k_selection"]
    print("\n=== domain gate validation ===")
    print(f"  k grid AUROC (image): " + ", ".join(
        f"k={k}: {report['k_grid'][str(k)]['auroc_valid_vs_bcn_image']:.4f}" for k in sel["grid"]))
    print(f"  selected k={sel['selected_k']}  registered k={sel['registered_k']} "
          f"(match: {sel['registered_matches_selected']})")
    print(f"  threshold (p1 valid) @ registered k: {report['registered_gate']['threshold']:.4f}")
    print(f"  BCN image flagged: {head['bcn_image_flagged_fraction']:.4f}   "
          f"test flagged: {head['derm7pt_test_flagged_fraction']:.4f}")
    print(f"  sanity: {report['sanity_probes']['status']}")
    for p in report["sanity_probes"].get("probes", []):
        print(f"    {p['probe']}: {p['domain_score']:.4f} "
              f"{'FLAGGED' if p['flagged_ood'] else 'passed'}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
