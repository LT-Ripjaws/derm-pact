"""Deletion/insertion faithfulness of GradCAM saliency maps on frozen DermLIP.

``medguard_v2/perception/attribution.py`` produces GradCAM heatmaps for the
diagnosis classifier and each criterion head, but nothing has ever measured
whether those heatmaps are *faithful*: whether the regions GradCAM marks as
important are actually the regions driving the head's decision, as opposed to
a plausible-looking map that happens to correlate with nothing.  A heatmap
that looks anatomically sensible to a clinician but is uncorrelated with what
the linear head actually used would be worse than no explanation at all.

Protocol (deletion / insertion, Petsiuk et al. 2018 "RISE"-style)
-------------------------------------------------------------------
For one (case, head) pair:

1.  Rank the 14x14 = 196 ViT patches by the head's GradCAM saliency.
2.  **Deletion**: starting from the original image, progressively replace the
    most-salient-first patches with a blurred baseline; re-encode with the
    SAME frozen DermLIP; re-score with the SAME fitted head; track
    P(originally-predicted class).  A faithful map makes this probability
    collapse *fast* -- low area under the curve (AUC).
3.  **Insertion**: the mirror image -- start fully blurred, progressively
    reveal the most-salient-first patches, track the same probability. A
    faithful map makes this rise *fast* -- high AUC.
4.  Both are compared against a **random-order baseline**: the mean AUC of 5
    fixed random patch orders (same seeds every run). If GradCAM does not
    beat random, it is not localising the evidence the head actually uses --
    that is a reportable negative, not a bug in this script.

Masking happens in **preprocessed-tensor space** (the 224x224 image DermLIP's
``preprocess`` produces), not in original-image pixel space: that is the
space the ViT actually tokenises into 16x16 patches, and it is the space
``gradcam_dermlip``'s patch-grid saliency already lives in. Resizing a patch
mask into original-image pixel coordinates and back would go through a lossy,
avoidable round trip through PIL's bilinear resize.

The random-order baseline does not depend on the head, so its 5x21
masked-image *encodings* are computed once per case and re-scored (cheap,
linear) against all three heads, instead of being re-encoded per head.  Only
the saliency-order curve is genuinely head-specific and gets its own encode.

Runtime guard
--------------
A timed pilot on the first 3 sampled cases projects the full-run wall time;
if the projection exceeds ``--budget-minutes`` (default 40), the case count
is cut to 25 and the reduction is written into the output artifact -- never
silently.

CLI::

    python scripts/dev/validate_attribution_faithfulness.py
    python scripts/dev/validate_attribution_faithfulness.py --n-cases 3 --out /tmp/pilot.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import auc as sklearn_auc  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from medguard_v2.data.derm7pt import Derm7ptCase, DIAGNOSIS_GROUP_ORDER, load_cases  # noqa: E402
from medguard_v2.perception.encode import encode_cases  # noqa: E402
from medguard_v2.perception.attribution import (  # noqa: E402
    _load_dermlip_model,
    gradcam_dermlip,
    save_attribution_overlay,
)
from medguard_v2.pipeline.run_cohort import fit_readers  # noqa: E402

SEED = 20260810  # matches probe.py / run_cohort.py's registered seed
N_CASES_DEFAULT = 40
N_MASK_STEPS = 20
N_RANDOM_ORDERS = 5
BUDGET_MINUTES_DEFAULT = 40.0
REDUCED_N_CASES = 25
PILOT_CASES = 3

HEAD_NAMES = ("diagnosis", "pigment_network", "blue_whitish_veil")

OUT_JSON = PROJECT_ROOT / "research" / "evidence" / "derm7pt_v1" / "attribution_faithfulness_v1.json"
OUT_OVERLAY_DIR = PROJECT_ROOT / "research" / "evidence" / "derm7pt_v1" / "attribution_samples"

PROTOCOL = "derm7pt-attribution-faithfulness-v1"


# --------------------------------------------------------------------------
# deterministic, diagnosis-spread case sampling
# --------------------------------------------------------------------------

def _stratified_quota(sizes: Dict[str, int], total: int) -> Dict[str, int]:
    """Water-filling allocation: as equal as possible across groups, capped by
    each group's availability, redistributing shortfall to groups with room.

    Plain proportional-to-prevalence sampling of Derm7pt's valid split would
    give ~1 BCC case out of 40 (BCC is 7/203 of valid) -- barely enough to
    say anything about that diagnosis.  This spreads the budget instead:
    every group gets an equal share up to what it actually has.
    """
    groups = sorted(sizes)
    quota = {g: 0 for g in groups}
    remaining = total
    active = set(groups)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for g in sorted(active):
            if remaining <= 0:
                break
            room = sizes[g] - quota[g]
            if room <= 0:
                active.discard(g)
                continue
            take = min(share, room, remaining)
            quota[g] += take
            remaining -= take
            progressed = True
            if quota[g] >= sizes[g]:
                active.discard(g)
        if not progressed:
            break
    return quota


def select_cases(cases: Sequence[Derm7ptCase], n: int, seed: int) -> List[Derm7ptCase]:
    """Deterministically sample ``n`` valid-split cases, spread across diagnoses."""
    import random

    valid = [c for c in cases if c.split == "valid"]
    by_group: Dict[str, List[Derm7ptCase]] = {}
    for case in valid:
        by_group.setdefault(case.diagnosis_group, []).append(case)

    rng = random.Random(seed)
    for group in by_group:
        by_group[group].sort(key=lambda c: c.case_num)  # deterministic order pre-shuffle
        rng.shuffle(by_group[group])

    sizes = {g: len(cs) for g, cs in by_group.items()}
    quota = _stratified_quota(sizes, n)

    selected: List[Derm7ptCase] = []
    for group in sorted(by_group):
        selected.extend(by_group[group][: quota.get(group, 0)])
    selected.sort(key=lambda c: c.case_num)
    return selected


# --------------------------------------------------------------------------
# masking mechanics (tensor space, matching the ViT's own patch grid)
# --------------------------------------------------------------------------

def _blurred_baseline(tensor) -> "torch.Tensor":  # noqa: F821 - torch imported lazily
    """Strongly Gaussian-blurred version of a preprocessed (C,H,W) tensor.

    Gaussian blur is a normalised convolution (weights sum to 1), so blurring
    the already-normalised tensor is mathematically identical to blurring the
    raw pixels and re-normalising -- normalisation is an affine per-channel
    map, and a sum-to-1 convolution commutes with it. No need to invert
    ``preprocess``'s Normalize step.
    """
    import torchvision.transforms.functional as TF

    size = tensor.shape[-1]
    k = size // 3
    k = k + 1 if k % 2 == 0 else k  # gaussian_blur requires an odd kernel size
    sigma = size / 6.0
    return TF.gaussian_blur(tensor, kernel_size=[k, k], sigma=[sigma, sigma])


def _patch_mask_tensor(original, baseline, keep_original: np.ndarray):
    """``keep_original``: bool (grid, grid). True keeps that patch's pixels
    from ``original``; False sources them from ``baseline``."""
    import torch

    c, h, w = original.shape
    grid = keep_original.shape[0]
    ph, pw = h // grid, w // grid
    pixel_mask = np.repeat(np.repeat(keep_original, ph, axis=0), pw, axis=1)
    pixel_mask = pixel_mask[:h, :w]  # guard against non-exact division remainder
    mask_t = torch.from_numpy(pixel_mask).to(dtype=torch.bool).unsqueeze(0).expand(c, -1, -1)
    return torch.where(mask_t, original, baseline)


def _random_patch_order(seed: int, case_num: int, order_idx: int, n_patches: int) -> np.ndarray:
    """Fixed per-(case, order_idx) random permutation, independent of case
    processing order or which other cases are in the run."""
    rng = np.random.default_rng([seed, case_num, order_idx])
    return rng.permutation(n_patches)


def _groups_from_order(order: np.ndarray, n_steps: int) -> List[np.ndarray]:
    return [g for g in np.array_split(order, n_steps) if g.size > 0]


def _encode_batch(model, tensors) -> np.ndarray:
    """Batched forward pass, L2-normalised the same way encode.py does.

    Uses the raw model directly (no gradients needed for scoring masked
    images), rather than going through encode.py's path-based ``encode_fn``
    closure, which only accepts on-disk image paths -- masked images here are
    synthetic, in-memory tensors that were never written to disk.
    """
    import torch

    batch = torch.stack(list(tensors))
    with torch.inference_mode():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.cpu().numpy().astype("float32")


def _mask_curve_features(
    model, orig_tensor, baseline_tensor, order: np.ndarray, n_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode every point on the deletion and insertion curves for one patch
    order. Returns ``(fractions, deletion_features, insertion_features)``,
    where ``fractions[k]`` is the cumulative share of patches processed by
    step ``k`` (uneven when 196 does not divide evenly by ``n_steps``)."""
    grid = int(round(np.sqrt(order.size)))
    assert grid * grid == order.size, f"order size {order.size} is not a perfect square"

    groups = _groups_from_order(order, n_steps)
    n_patches = order.size
    cum_counts = np.cumsum([0] + [g.size for g in groups])
    fractions = cum_counts / n_patches

    deletion_tensors = []
    insertion_tensors = []
    masked_so_far = np.zeros(n_patches, dtype=bool)
    for step in range(len(groups) + 1):
        keep_original_deletion = (~masked_so_far).reshape(grid, grid)
        keep_original_insertion = masked_so_far.reshape(grid, grid)
        deletion_tensors.append(
            _patch_mask_tensor(orig_tensor, baseline_tensor, keep_original_deletion)
        )
        insertion_tensors.append(
            _patch_mask_tensor(orig_tensor, baseline_tensor, keep_original_insertion)
        )
        if step < len(groups):
            masked_so_far = masked_so_far.copy()
            masked_so_far[groups[step]] = True

    deletion_features = _encode_batch(model, deletion_tensors)
    insertion_features = _encode_batch(model, insertion_tensors)
    return fractions, deletion_features, insertion_features


# --------------------------------------------------------------------------
# per-case, per-head evaluation
# --------------------------------------------------------------------------

def _paired_stats(saliency: np.ndarray, random_mean: np.ndarray) -> dict:
    """Per-case paired comparison of saliency vs random-order AUC."""
    gap = saliency - random_mean
    result = {
        "saliency_mean": round(float(saliency.mean()), 6),
        "random_mean": round(float(random_mean.mean()), 6),
        "gap_mean": round(float(gap.mean()), 6),
        "gap_median": round(float(np.median(gap)), 6),
        "gap_std": round(float(gap.std(ddof=1)), 6) if len(gap) > 1 else None,
    }
    try:
        t_stat, t_p = scipy_stats.ttest_rel(saliency, random_mean)
        result["ttest_p"] = round(float(t_p), 6)
        result["ttest_stat"] = round(float(t_stat), 6)
    except Exception as exc:  # noqa: BLE001
        result["ttest_p"] = None
        result["ttest_error"] = str(exc)
    try:
        if np.allclose(gap, 0.0):
            raise ValueError("all paired differences are exactly zero")
        w_stat, w_p = scipy_stats.wilcoxon(saliency, random_mean)
        result["wilcoxon_p"] = round(float(w_p), 6)
        result["wilcoxon_stat"] = round(float(w_stat), 6)
    except Exception as exc:  # noqa: BLE001
        result["wilcoxon_p"] = None
        result["wilcoxon_error"] = str(exc)
    return result


def _head_verdict(deletion_stats: dict, insertion_stats: dict) -> str:
    """Fixed decision rule, stated before results are inspected (see module
    docstring / task spec): faithful requires BOTH curves to beat random with
    the correct sign AND paired significance; marginal is right-signed but
    not significant (or only one curve clears); unfaithful is anything else,
    including a wrong-signed mean gap."""
    del_gap = deletion_stats["gap_mean"]  # want < 0: saliency AUC lower than random
    ins_gap = insertion_stats["gap_mean"]  # want > 0: saliency AUC higher than random
    del_p = deletion_stats.get("wilcoxon_p")
    ins_p = insertion_stats.get("wilcoxon_p")
    del_sig = del_p is not None and del_p < 0.05 and del_gap < 0
    ins_sig = ins_p is not None and ins_p < 0.05 and ins_gap > 0
    right_direction = del_gap < 0 and ins_gap > 0

    if del_sig and ins_sig:
        return "faithful"
    if right_direction and (del_sig or ins_sig):
        return "marginal"
    if right_direction:
        return "marginal (right direction, not significant)"
    return "unfaithful"


def evaluate_case_head(
    model,
    case: Derm7ptCase,
    head_name: str,
    head,
    orig_feature: np.ndarray,
    orig_tensor,
    baseline_tensor,
    random_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    pred_class = head.predict(orig_feature.reshape(1, -1))[0]
    class_idx = list(head.classes_).index(pred_class)

    attr = gradcam_dermlip(str(case.derm_path), head, head_name, return_patch_grid=True)
    if attr.predicted_value != pred_class:
        raise RuntimeError(
            f"case {case.case_num} {head_name}: gradcam_dermlip predicted "
            f"{attr.predicted_value!r} but the fresh head.predict gave {pred_class!r} "
            "-- the two should be computed from the same features and cannot disagree."
        )
    patch_heatmap = attr.patch_heatmap
    degenerate = bool(np.isclose(float(patch_heatmap.max()), float(patch_heatmap.min())))
    saliency_order = np.argsort(-patch_heatmap.ravel())

    fractions, del_feats, ins_feats = _mask_curve_features(
        model, orig_tensor, baseline_tensor, saliency_order, N_MASK_STEPS
    )
    del_probs = head.predict_proba(del_feats)[:, class_idx]
    ins_probs = head.predict_proba(ins_feats)[:, class_idx]
    deletion_auc_saliency = float(sklearn_auc(fractions, del_probs))
    insertion_auc_saliency = float(sklearn_auc(fractions, ins_probs))

    deletion_auc_random = []
    insertion_auc_random = []
    for r_fractions, r_del_feats, r_ins_feats in random_curves:
        r_del_probs = head.predict_proba(r_del_feats)[:, class_idx]
        r_ins_probs = head.predict_proba(r_ins_feats)[:, class_idx]
        deletion_auc_random.append(float(sklearn_auc(r_fractions, r_del_probs)))
        insertion_auc_random.append(float(sklearn_auc(r_fractions, r_ins_probs)))

    return {
        "case_num": int(case.case_num),
        "diagnosis_group": case.diagnosis_group,
        "predicted_class": str(pred_class),
        "degenerate_saliency_map": degenerate,
        "deletion_auc_saliency": round(deletion_auc_saliency, 6),
        "deletion_auc_random_all": [round(v, 6) for v in deletion_auc_random],
        "deletion_auc_random_mean": round(float(np.mean(deletion_auc_random)), 6),
        "insertion_auc_saliency": round(insertion_auc_saliency, 6),
        "insertion_auc_random_all": [round(v, 6) for v in insertion_auc_random],
        "insertion_auc_random_mean": round(float(np.mean(insertion_auc_random)), 6),
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def _infer_patch_px(model, sample_tensor) -> int:
    """Patch pixel size from the model's own conv/patch-embed layer, so the
    masking grid matches the encoder's real tokenisation instead of an
    assumed constant."""
    visual = model.visual
    conv = getattr(visual, "conv1", None)
    if conv is not None and hasattr(conv, "kernel_size"):
        return int(conv.kernel_size[0])
    trunk = getattr(visual, "trunk", None)
    patch_embed = getattr(trunk, "patch_embed", None) if trunk is not None else None
    proj = getattr(patch_embed, "proj", None) if patch_embed is not None else None
    if proj is not None and hasattr(proj, "kernel_size"):
        return int(proj.kernel_size[0])
    raise RuntimeError("cannot determine patch pixel size from the model's visual tower")


def run(n_cases: int, budget_minutes: float, out_json: Path, overlay_dir: Path, verbose: bool = True) -> dict:
    t_start = time.time()

    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    bundle = encode_cases(cases)  # cached; near-instant
    features = {str(int(cn)): bundle.vector_for(int(cn)) for cn in bundle.case_nums}
    diagnosis_head, criterion_heads = fit_readers(train, features)
    heads = {
        "diagnosis": diagnosis_head,
        "pigment_network": criterion_heads["pigment_network"],
        "blue_whitish_veil": criterion_heads["blue_whitish_veil"],
    }
    if verbose:
        for name, head in heads.items():
            print(f"head {name}: classes={list(head.classes_)} coef_shape={head.coef_.shape}")

    model, preprocess = _load_dermlip_model("dermlip")

    sampled = select_cases(cases, n_cases, SEED)
    group_counts = {g: sum(1 for c in sampled if c.diagnosis_group == g) for g in DIAGNOSIS_GROUP_ORDER}
    if verbose:
        print(f"sampled {len(sampled)} valid-split cases, spread: {group_counts}")

    # --- runtime guard: time a pilot on the first PILOT_CASES cases, project, reduce if needed ---
    pilot_n = min(PILOT_CASES, len(sampled))
    pilot_start = time.time()
    from PIL import Image as PILImage

    def _process_case(case: Derm7ptCase) -> Dict[str, dict]:
        img = PILImage.open(case.derm_path).convert("RGB")
        orig_tensor = preprocess(img)
        assert orig_tensor.shape[-2] == orig_tensor.shape[-1], (
            f"expected a square preprocessed tensor, got {tuple(orig_tensor.shape)}"
        )
        baseline_tensor = _blurred_baseline(orig_tensor)
        # Fresh single-image encode, NOT the cached corpus feature from features.npz:
        # gradcam_dermlip also encodes a fresh single image (batch=1) to get its own
        # prediction, and the cached cache was built in a separate session at batch
        # size 32. Batched vs. single-image matmuls are not bit-identical (ordinary
        # floating-point non-associativity), and on a genuinely close-to-boundary
        # case that is enough to flip an argmax -- caught empirically on case 20's
        # blue_whitish_veil head (a binary decision near p=0.5) during a pilot run.
        # Every point on this curve, including the "0 patches masked" endpoint, must
        # come from the same encode path for the curve to be internally consistent.
        orig_feature = _encode_batch(model, [orig_tensor])[0]

        # Random orders are head-independent, so they are built once per case and
        # re-scored per head below instead of re-encoded per head. Patch grid size
        # comes from the model's own patch-embed geometry (verified empirically as
        # 14x14 for DermLIP ViT-B/16 before writing this script) rather than a
        # hardcoded constant, so a different --encoder would not silently misgrid.
        patch_px = _infer_patch_px(model, orig_tensor)
        grid = orig_tensor.shape[-1] // patch_px
        n_patches = grid * grid

        random_curves = []
        for order_idx in range(N_RANDOM_ORDERS):
            order = _random_patch_order(SEED, case.case_num, order_idx, n_patches)
            random_curves.append(
                _mask_curve_features(model, orig_tensor, baseline_tensor, order, N_MASK_STEPS)
            )

        per_head = {}
        for head_name, head in heads.items():
            per_head[head_name] = evaluate_case_head(
                model, case, head_name, head, orig_feature, orig_tensor, baseline_tensor, random_curves
            )
        return per_head

    results: Dict[str, List[dict]] = {name: [] for name in heads}
    processed: List[Derm7ptCase] = []

    for case in sampled[:pilot_n]:
        per_head = _process_case(case)
        for name in heads:
            results[name].append(per_head[name])
        processed.append(case)

    pilot_elapsed = time.time() - pilot_start
    per_case_seconds = pilot_elapsed / max(pilot_n, 1)
    projected_total_minutes = per_case_seconds * len(sampled) / 60.0
    reduced = False
    if projected_total_minutes > budget_minutes and len(sampled) > REDUCED_N_CASES:
        reduced = True
        # Re-stratify at the smaller n via select_cases, NOT sampled[:REDUCED_N_CASES]:
        # `sampled` is sorted by case_num at this point, so a plain slice would keep
        # whichever diagnosis groups happen to have low case_nums and silently drop
        # others -- exactly the "spread across diagnoses" property select_cases exists
        # to guarantee. Any already-processed pilot cases that survive into the new,
        # smaller sample are kept (no wasted work); everything else in the new sample
        # still needs processing below.
        new_n = max(REDUCED_N_CASES, pilot_n)
        sampled = select_cases(cases, new_n, SEED)
        sampled_nums = {c.case_num for c in sampled}
        keep_nums = {c.case_num for c in processed} & sampled_nums
        for name in heads:
            results[name] = [r for r in results[name] if r["case_num"] in keep_nums]
        processed = [c for c in processed if c.case_num in keep_nums]
        if verbose:
            print(
                f"RUNTIME GUARD: pilot projected {projected_total_minutes:.1f} min for "
                f"{n_cases} cases (budget {budget_minutes:.0f} min); reducing to "
                f"{len(sampled)} cases (re-stratified, not truncated; "
                f"{len(processed)} pilot case(s) carried over)."
            )
    if verbose:
        print(f"pilot: {pilot_n} cases in {pilot_elapsed:.1f}s "
              f"({per_case_seconds:.1f}s/case) -> projected {projected_total_minutes:.1f} min total")

    remaining = [c for c in sampled if c.case_num not in {p.case_num for p in processed}]
    for i, case in enumerate(remaining):
        per_head = _process_case(case)
        for name in heads:
            results[name].append(per_head[name])
        processed.append(case)
        if verbose and (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  {len(processed)}/{len(sampled)} cases done, {elapsed:.0f}s elapsed")

    # --- overlay PNGs: first sampled case (by case_num), one per head; not cherry-picked ---
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths = []
    overlay_case = processed[0]
    for head_name, head in heads.items():
        attr = gradcam_dermlip(str(overlay_case.derm_path), head, head_name)
        out_path = overlay_dir / f"case_{overlay_case.case_num}_{head_name}.png"
        save_attribution_overlay(str(overlay_case.derm_path), attr, str(out_path))
        try:
            recorded_path = str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            recorded_path = str(out_path)  # out_path is outside the repo (e.g. a scratch dir)
        overlay_paths.append(recorded_path)
    if verbose:
        print(f"overlays written for case {overlay_case.case_num}: {overlay_paths}")

    # --- aggregate ---
    heads_out = {}
    verdicts = []
    for name in heads:
        rows = results[name]
        saliency_del = np.array([r["deletion_auc_saliency"] for r in rows])
        random_del = np.array([r["deletion_auc_random_mean"] for r in rows])
        saliency_ins = np.array([r["insertion_auc_saliency"] for r in rows])
        random_ins = np.array([r["insertion_auc_random_mean"] for r in rows])

        deletion_stats = _paired_stats(saliency_del, random_del)
        insertion_stats = _paired_stats(saliency_ins, random_ins)
        verdict = _head_verdict(deletion_stats, insertion_stats)
        verdicts.append(verdict)
        n_degenerate = sum(1 for r in rows if r["degenerate_saliency_map"])

        heads_out[name] = {
            "n_classes": int(len(heads[name].classes_)),
            "n_degenerate_saliency_maps": n_degenerate,
            "per_case": rows,
            "summary": {
                "deletion": deletion_stats,
                "insertion": insertion_stats,
            },
            "verdict": verdict,
        }
        if verbose:
            print(
                f"\n{name}: deletion saliency={deletion_stats['saliency_mean']:.4f} "
                f"random={deletion_stats['random_mean']:.4f} gap={deletion_stats['gap_mean']:+.4f} "
                f"p={deletion_stats.get('wilcoxon_p')}"
            )
            print(
                f"{name}: insertion saliency={insertion_stats['saliency_mean']:.4f} "
                f"random={insertion_stats['random_mean']:.4f} gap={insertion_stats['gap_mean']:+.4f} "
                f"p={insertion_stats.get('wilcoxon_p')}"
            )
            print(f"{name}: verdict = {verdict} ({n_degenerate}/{len(rows)} degenerate maps)")

    if all(v == "faithful" for v in verdicts):
        overall = "faithful: GradCAM beats random on both deletion and insertion for every head tested"
    elif all(v == "unfaithful" for v in verdicts):
        overall = "unfaithful: GradCAM does not beat random ordering for any head tested"
    else:
        overall = (
            "mixed: " + "; ".join(f"{name}={heads_out[name]['verdict']}" for name in heads)
        )

    runtime_seconds = time.time() - t_start
    artifact = {
        "protocol": PROTOCOL,
        "purpose": (
            "measures whether GradCAM saliency on the diagnosis classifier and 2 criterion "
            "heads (pigment_network, blue_whitish_veil) is faithful enough to ship as "
            "criterion-level visual evidence, via deletion/insertion AUC vs a random-order "
            "baseline (mean of 5 fixed random orders)"
        ),
        "seed": SEED,
        "n_cases_requested": n_cases,
        "n_cases_run": len(processed),
        "runtime_guard": {
            "budget_minutes": budget_minutes,
            "pilot_cases": pilot_n,
            "pilot_seconds_per_case": round(per_case_seconds, 2),
            "projected_minutes_for_requested_n": round(projected_total_minutes, 1),
            "reduced": reduced,
        },
        "diagnosis_group_counts": group_counts if not reduced else {
            g: sum(1 for c in processed if c.diagnosis_group == g) for g in DIAGNOSIS_GROUP_ORDER
        },
        "case_nums": [int(c.case_num) for c in processed],
        "n_mask_steps": N_MASK_STEPS,
        "n_random_orders": N_RANDOM_ORDERS,
        "baseline": "gaussian_blur (kernel ~ size/3, sigma = size/6, applied in normalised tensor space)",
        "masking_space": "preprocessed 224x224 tensor, patch-aligned to the ViT's own tokenisation",
        "metric": "P(originally-predicted class) on the SAME fitted head, re-encoded by the SAME frozen DermLIP",
        "overlay_case_num": int(overlay_case.case_num),
        "overlay_paths": overlay_paths,
        "heads": heads_out,
        "verdict": overall,
        "runtime_seconds": round(runtime_seconds, 1),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(out_json)
    if verbose:
        print(f"\nwrote {out_json}")
        print(f"total runtime: {runtime_seconds:.1f}s ({runtime_seconds/60:.1f} min)")
        print(f"\nVERDICT: {overall}")
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-cases", type=int, default=N_CASES_DEFAULT)
    ap.add_argument("--budget-minutes", type=float, default=BUDGET_MINUTES_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--overlay-dir", type=Path, default=OUT_OVERLAY_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run(
        n_cases=args.n_cases,
        budget_minutes=args.budget_minutes,
        out_json=args.out,
        overlay_dir=args.overlay_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
