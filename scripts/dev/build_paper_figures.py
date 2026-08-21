"""Build the four IEEE Access result figures for MedGuard v2 from sealed evidence JSON.

Reads (read-only, never modified):
  research/evidence/derm7pt_v1/roc_curve_data.json
  research/evidence/derm7pt_v1/complementarity_venn_data.json
  research/evidence/derm7pt_v1/confusion_matrices.json
  research/evidence/derm7pt_v1/paper_metrics_summary.json

Writes:
  our_paper/figs/fig_roc.png
  our_paper/figs/fig_venn.png
  our_paper/figs/fig_confusion.png
  our_paper/figs/fig_crossdata.png

Every plotted number is read directly from the JSON evidence files (printed to
stdout at generation time for traceability) -- nothing is hand-typed into the
plotting calls except axis cosmetics.

Run with the pinned interpreter only:
    ./venv_win/Scripts/python.exe scripts/dev/build_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
OUT_DIR = REPO / "our_paper" / "figs"

# ---------------------------------------------------------------------------
# IEEE-safe styling: fixed font family, 300 dpi, colorblind-safe (Okabe-Ito)
# palette chosen so series also separate in greyscale via distinct luminance
# and (where relevant) hatch/marker shape.
# ---------------------------------------------------------------------------
# NOTE: "Helvetica" is in the IEEE-safe candidate set but is not installed as a distinct
# family on this machine (font_manager confirmed: Arial, DejaVu Serif, Times New Roman only).
# Arial is metric-compatible with Helvetica, so it is used as that slot's fallback instead
# of listing "Helvetica" and generating findfont warnings for a family that will never match.
plt.rcParams["font.family"] = ["Times New Roman", "Arial", "DejaVu Serif"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8.5
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

COL_CURVE = "#0072B2"    # blue    -- ROC curve
COL_ONCURVE = "#009E73"  # green   -- on-curve operating points
COL_UNION = "#D55E00"    # vermillion -- off-curve union point
COL_CHANCE = "#888888"   # grey dashed -- chance / reference lines
COL_A = "#0072B2"        # venn circle A (classifier)
COL_B = "#D55E00"        # venn circle B (checklist)

WRITTEN = []  # (name, path, (w,h)) for the final report


def load_json(name: str) -> dict:
    path = EVIDENCE / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(out_path) as im:
        dims = im.size
    WRITTEN.append((name, str(out_path), dims))
    print(f"[saved] {name}: {out_path}  {dims[0]}x{dims[1]}px")


# ---------------------------------------------------------------------------
# Figure 1: Derm7pt melanoma ROC (promoted classifier), union point off-curve
# ---------------------------------------------------------------------------
def fig_roc() -> bool:
    roc_data = load_json("roc_curve_data.json")
    summary = load_json("paper_metrics_summary.json")

    roc = roc_data["promoted_classifier_roc"]
    points = roc["points"]
    fpr = np.array([p["fpr"] for p in points], dtype=float)
    tpr = np.array([p["sensitivity_tpr"] for p in points], dtype=float)
    auroc = roc["auroc"]
    print(f"[fig_roc] promoted_classifier_roc: n_points={len(points)}, auroc={auroc}, "
          f"auroc_ci={roc.get('auroc_ci')}")

    order = np.argsort(fpr, kind="stable")
    fpr_s, tpr_s = fpr[order], tpr[order]

    thresholds_meta = roc_data["promoted_clinical_thresholds"]
    spec90 = thresholds_meta["specificity_0.9"]
    spec95 = thresholds_meta["specificity_0.95"]
    union = roc_data["promoted_union_operating_point"]
    argmax_block = summary["perception_derm7pt"]["operating_point_bootstrap"]["classifier"]
    argmax_sens = argmax_block["sensitivity"]
    argmax_spec = argmax_block["specificity"]

    print(f"[fig_roc] argmax operating point (paper_metrics_summary.json): "
          f"sens={argmax_sens}, spec={argmax_spec}")
    print(f"[fig_roc] spec>=0.90 point (roc_curve_data.json): {spec90}")
    print(f"[fig_roc] spec>=0.95 point (roc_curve_data.json): {spec95}")
    print(f"[fig_roc] union operating point (roc_curve_data.json): {union}")

    def fpr_of(spec):
        return round(1.0 - spec, 4)

    argmax_fpr = fpr_of(argmax_spec)
    spec90_fpr = fpr_of(spec90["specificity"])
    spec95_fpr = fpr_of(spec95["specificity"])
    union_fpr = fpr_of(union["specificity"])

    def curve_sens_at_fpr(target_fpr: float) -> float:
        """Upper envelope: highest sensitivity the curve attains at this FPR (handles
        vertical step ties where several thresholds share the same FPR)."""
        idx = np.where(np.isclose(fpr_s, target_fpr, atol=5e-5))[0]
        if len(idx) == 0:
            raise ValueError(f"No ROC curve point found at fpr={target_fpr}")
        return float(tpr_s[idx].max())

    def point_on_curve(target_fpr: float, target_sens: float) -> bool:
        """Exact membership: does (fpr, sens) appear as one of the actual plotted
        threshold points? (A named operating point can legitimately sit at the
        BOTTOM of a vertical step-tie, not just the top, so this must not use max().)"""
        return bool(np.any(np.isclose(fpr_s, target_fpr, atol=5e-5) & np.isclose(tpr_s, target_sens, atol=5e-5)))

    for label, tfpr, tsens in [
        ("argmax", argmax_fpr, argmax_sens),
        ("spec90", spec90_fpr, spec90["sensitivity"]),
        ("spec95", spec95_fpr, spec95["sensitivity"]),
    ]:
        n_at_fpr = int(np.sum(np.isclose(fpr_s, tfpr, atol=5e-5)))
        ok = point_on_curve(tfpr, tsens)
        print(f"[fig_roc] check {label}: target sens={tsens} at fpr={tfpr} -> on_curve={ok} "
              f"({n_at_fpr} curve points share this fpr)")
        assert ok, f"{label} point ({tfpr}, {tsens}) does not appear on the plotted curve"

    union_curve_sens = curve_sens_at_fpr(union_fpr)
    n_at_union_fpr = int(np.sum(np.isclose(fpr_s, union_fpr, atol=5e-5)))
    print(f"[fig_roc] classifier curve at union's fpr={union_fpr}: max sens={union_curve_sens} "
          f"({n_at_union_fpr} curve points share this fpr)")
    below = union["sensitivity"] < union_curve_sens
    print(f"[fig_roc] UNION-BELOW-CURVE CHECK: union sens={union['sensitivity']} at fpr={union_fpr} "
          f"vs classifier curve sens={union_curve_sens} at same fpr -> below_curve={below}")
    assert below, "Union point is not below the ROC curve at equal specificity -- refusing to save a wrong figure."

    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot(fpr_s, tpr_s, color=COL_CURVE, lw=1.8, zorder=2,
            label=f"Classifier ROC (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COL_CHANCE, lw=1.0, zorder=1, label="Chance")

    ax.scatter([argmax_fpr], [argmax_sens], marker="o", s=60, color=COL_ONCURVE,
               edgecolor="black", linewidth=0.6, zorder=4,
               label=f"Operating point (argmax): sens {argmax_sens:.3f} @ spec {argmax_spec:.3f}")
    ax.scatter([spec90_fpr], [spec90["sensitivity"]], marker="s", s=55, color=COL_ONCURVE,
               edgecolor="black", linewidth=0.6, zorder=4,
               label=f"Sens @ Spec = 0.90: {spec90['sensitivity']:.3f}")
    ax.scatter([spec95_fpr], [spec95["sensitivity"]], marker="^", s=62, color=COL_ONCURVE,
               edgecolor="black", linewidth=0.6, zorder=4,
               label=f"Sens @ Spec = 0.95: {spec95['sensitivity']:.3f}")

    ax.scatter([union_fpr], [union["sensitivity"]], marker="D", s=72, color=COL_UNION,
               edgecolor="black", linewidth=0.7, zorder=5,
               label=f"Union rule (classifier OR checklist): sens {union['sensitivity']:.3f}")
    ax.plot([union_fpr, union_fpr], [union["sensitivity"], union_curve_sens],
            color=COL_UNION, lw=1.1, linestyle=":", zorder=3)

    ax.annotate(
        f"union ({union['sensitivity']:.2f}) < classifier ROC\n({union_curve_sens:.2f}) at equal specificity",
        xy=(union_fpr, (union["sensitivity"] + union_curve_sens) / 2.0),
        xytext=(union_fpr + 0.16, union["sensitivity"] - 0.20),
        fontsize=8, color=COL_UNION,
        arrowprops=dict(arrowstyle="->", color=COL_UNION, lw=1.0),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("Derm7pt melanoma detection: classifier ROC")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.6)

    save(fig, "fig_roc.png")
    return below


# ---------------------------------------------------------------------------
# Figure 2: two-circle Venn of melanoma catches (classifier vs checklist)
# ---------------------------------------------------------------------------
def fig_venn() -> None:
    data = load_json("complementarity_venn_data.json")
    v = data["venn_counts"]
    melanomas = v["melanomas"]
    both = v["both"]
    a_only = v["classifier_only"]
    b_only = v["criteria_only"]
    neither = v["neither"]
    print(f"[fig_venn] venn_counts: {v}")
    assert a_only + b_only + both + neither == melanomas, "Venn counts do not sum to total melanomas"

    a_total = a_only + both
    b_total = b_only + both

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Centers spaced further apart than in the first draft: the initial version placed the
    # "Classifier catches" / "Checklist catches" headers directly above each circle's centre,
    # which collided into unreadable run-together text at this circle spacing. Fixed by both
    # widening the centre-to-centre gap and anchoring each header out past its own circle.
    r = 2.3
    cx_a, cx_b, cy = 4.2, 6.8, 3.9

    circle_a = plt.Circle((cx_a, cy), r, color=COL_A, alpha=0.38, ec="black", lw=1.2, zorder=1)
    circle_b = plt.Circle((cx_b, cy), r, color=COL_B, alpha=0.38, ec="black", lw=1.2, zorder=1)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    ax.text(cx_a - 1.15, cy, f"{a_only}", fontsize=16, ha="center", va="center", fontweight="bold")
    ax.text((cx_a + cx_b) / 2, cy, f"{both}", fontsize=16, ha="center", va="center", fontweight="bold")
    ax.text(cx_b + 1.15, cy, f"{b_only}", fontsize=16, ha="center", va="center", fontweight="bold")

    ax.text(cx_a - 1.0, cy + r + 0.5, f"Classifier catches\n(n = {a_total})", fontsize=9.5, ha="center", va="bottom")
    ax.text(cx_b + 1.0, cy + r + 0.5, f"Checklist catches\n(n = {b_total})", fontsize=9.5, ha="center", va="bottom")

    ax.text((cx_a + cx_b) / 2, cy - r - 0.45,
            f"{neither} missed by both (of {melanomas} melanomas)",
            fontsize=9.5, ha="center", va="top", style="italic")

    ax.set_title(f"Melanoma detection complementarity, Derm7pt (n = {melanomas} melanomas)", pad=14)

    save(fig, "fig_venn.png")


# ---------------------------------------------------------------------------
# Figure 3: Derm7pt 5-class confusion matrix
# ---------------------------------------------------------------------------
def fig_confusion() -> None:
    data = load_json("confusion_matrices.json")
    block = data["derm7pt_test"]
    json_labels = block["labels"]
    matrix = np.array(block["matrix"], dtype=float)
    print(f"[fig_confusion] confusion_matrices.json derm7pt_test: n={block['n']}, labels={json_labels}")

    target_order = ["BCC", "MEL", "NEV", "SK", "MISC"]
    idx = [json_labels.index(lbl) for lbl in target_order]
    mat = matrix[np.ix_(idx, idx)]

    row_sums = mat.sum(axis=1, keepdims=True)
    row_norm = mat / row_sums
    acc = float(np.trace(mat) / mat.sum())
    bal_acc = float(np.mean(np.diag(mat) / row_sums.flatten()))
    print(f"[fig_confusion] computed from matrix (reordered BCC/MEL/NEV/SK/MISC): "
          f"accuracy={acc:.4f}, balanced_accuracy={bal_acc:.4f}, n={int(mat.sum())}")

    # Cross-check against paper_metrics_summary.json classifier_5class arms (report-only,
    # not enforced): this surfaces whether the raw matrix matches the promoted argmax/
    # operating-point arms or the pre-promotion base arm.
    summary = load_json("paper_metrics_summary.json")
    c5 = summary["perception_derm7pt"]["classifier_5class"]
    print(f"[fig_confusion] paper_metrics_summary.json classifier_5class.argmax.accuracy = "
          f"{c5['argmax']['accuracy']}")
    print(f"[fig_confusion] paper_metrics_summary.json classifier_5class.operating_point.accuracy = "
          f"{c5['operating_point']['accuracy']}")
    print(f"[fig_confusion] paper_metrics_summary.json classifier_5class.base_arm_reference.accuracy = "
          f"{c5['base_arm_reference']['accuracy']}  (balanced_accuracy="
          f"{c5['base_arm_reference']['balanced_accuracy']})")
    if abs(acc - c5["base_arm_reference"]["accuracy"]) < 1e-4:
        print("[fig_confusion] NOTE: matrix reconciles to the BASE (pre-promotion) arm, "
              "not the promoted argmax/operating-point arms. Using it as-is (only full matrix "
              "available in confusion_matrices.json); title deliberately does not claim "
              "'operating point' or 'promoted'. Flagged in final report.")

    fig, ax = plt.subplots(figsize=(5.1, 4.5))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(target_order)))
    ax.set_yticks(range(len(target_order)))
    ax.set_xticklabels(target_order)
    ax.set_yticklabels(target_order)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Derm7pt 5-class confusion matrix (test, n = {int(mat.sum())}; accuracy = {acc:.3f})")

    for i in range(len(target_order)):
        for j in range(len(target_order)):
            val = int(mat[i, j])
            shade = row_norm[i, j]
            color = "white" if shade > 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised proportion")

    ax.set_xticks(np.arange(-0.5, len(target_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    save(fig, "fig_confusion.png")


# ---------------------------------------------------------------------------
# Figure 4: melanoma AUROC across datasets / evaluation regimes
# ---------------------------------------------------------------------------
def fig_crossdata() -> None:
    summary = load_json("paper_metrics_summary.json")

    derm7pt = summary["perception_derm7pt"]["melanoma_auroc"]["point"]
    bcn_in = summary["perception_bcn"]["in_domain"]["mel_auroc"]
    bcn_cross = summary["perception_bcn"]["cross_centre"]["mel_auroc"]
    ham_e3 = summary["perception_ham"]["e3_split"]["five_group"]["mel_auroc"]
    ham_sealed = summary["perception_ham"]["sealed_test"]["mel_auroc"]

    values = [derm7pt, bcn_in, bcn_cross, ham_e3, ham_sealed]
    labels = ["Derm7pt\nin-domain", "BCN20000\nin-domain", "BCN20000\ncross-centre",
              "HAM10000\nE3", "HAM10000\nsealed"]
    print(f"[fig_crossdata] AUROC values from paper_metrics_summary.json: "
          f"{dict(zip([l.replace(chr(10), ' ') for l in labels], values))}")

    expected = [0.9151, 0.8641, 0.7962, 0.9303, 0.9444]
    for v, e, lbl in zip(values, expected, labels):
        ok = abs(v - e) < 1e-6
        print(f"[fig_crossdata] check {lbl.replace(chr(10), ' ')}: json={v} task-given={e} match={ok}")
        assert ok, f"{lbl}: JSON value {v} does not match task-given value {e}"

    colors = ["#0072B2", "#56B4E9", "#56B4E9", "#009E73", "#009E73"]
    hatches = ["", "", "//", "", "\\\\"]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8, width=0.6, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    for xi, v in zip(x, values):
        ax.text(xi, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=8.8)

    ax.axhline(0.5, color=COL_CHANCE, linestyle="--", lw=1.1, zorder=2, label="Chance (AUROC = 0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("Melanoma AUROC")
    # Extra headroom above the tallest value labels (max bar 0.9444 -> label ~0.96) so the
    # legend box (anchored to the axes' top-left corner) doesn't crowd the Derm7pt bar's
    # "0.9151" label directly beneath it -- confirmed by a pixel crop that the first draft's
    # ylim=1.06 left them touching.
    ax.set_ylim(0.0, 1.22)
    ax.set_title("Melanoma AUROC across datasets and evaluation regimes")
    ax.legend(loc="upper left", frameon=True, fontsize=8.3)
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "fig_crossdata.png")


def main() -> None:
    print(f"[build_paper_figures] REPO={REPO}")
    print(f"[build_paper_figures] EVIDENCE={EVIDENCE}")
    print(f"[build_paper_figures] OUT_DIR={OUT_DIR}")

    below_curve_ok = fig_roc()
    fig_venn()
    fig_confusion()
    fig_crossdata()

    print("\n=== SUMMARY ===")
    print(f"union-below-curve check passed: {below_curve_ok}")
    for name, path, dims in WRITTEN:
        print(f"{name}: {path}  {dims[0]}x{dims[1]}px")


if __name__ == "__main__":
    main()
