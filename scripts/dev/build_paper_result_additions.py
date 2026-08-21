"""Build two paper-only IEEE result figures from the latest sealed evidence.

This script does not fit a model, choose a threshold, or recompute a reported
classifier or assistant endpoint.  It visualizes values already recorded in:

  research/evidence/derm7pt_v1/criteria_regrouped_kawahara.json
  research/evidence/derm7pt_v1/redteam_v1.json

The rule-of-three limits in the robustness figure apply the approximation that
is already defined in the manuscript to the recorded zero-success counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "research" / "evidence" / "derm7pt_v1"
OUT_DIR = REPO / "our_paper" / "figs"

plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "Arial", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 180,
    }
)

BLUE = "#0072B2"
VERMILLION = "#D55E00"
DARK = "#222222"
MID = "#777777"
GRID = "#D9D9D9"


def load_json(name: str) -> dict:
    with (EVIDENCE / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_both(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {pdf}")
    print(f"[saved] {png}")


def criterion_profile() -> None:
    data = load_json("criteria_regrouped_kawahara.json")
    promoted = data["per_criterion"]["promoted"]
    order = [
        "pigment_network",
        "blue_whitish_veil",
        "vascular_structures",
        "streaks",
        "pigmentation",
        "dots_and_globules",
        "regression_structures",
    ]

    labels = [promoted[name]["kawahara_abbreviation"] for name in order]
    accuracy = np.array([promoted[name]["grouped_accuracy"] for name in order])
    acc_low = np.array(
        [promoted[name]["grouped_accuracy_ci95"]["ci95_low"] for name in order]
    )
    acc_high = np.array(
        [promoted[name]["grouped_accuracy_ci95"]["ci95_high"] for name in order]
    )
    auroc = np.array(
        [promoted[name]["melanoma_indicative_auroc"] for name in order]
    )
    auc_low = np.array(
        [promoted[name]["melanoma_indicative_auroc_ci95"]["ci95_low"] for name in order]
    )
    auc_high = np.array(
        [promoted[name]["melanoma_indicative_auroc_ci95"]["ci95_high"] for name in order]
    )

    n_test = data["protocol"]["n_test"]
    n_boot = data["protocol"]["bootstrap"]["n_resamples"]
    assert n_test == 395 and n_boot == 1000
    assert np.all((accuracy >= 0) & (accuracy <= 1))
    assert np.all((auroc >= 0) & (auroc <= 1))

    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), sharey=True)

    panels = [
        (axes[0], accuracy, acc_low, acc_high, BLUE, "o", "(a) Grouped-category accuracy"),
        (axes[1], auroc, auc_low, auc_high, VERMILLION, "s", "(b) Melanoma-indicative AUROC"),
    ]
    for ax, values, low, high, color, marker, title in panels:
        xerr = np.vstack((values - low, high - values))
        ax.errorbar(
            values,
            y,
            xerr=xerr,
            fmt=marker,
            ms=5.5,
            mfc=color,
            mec=DARK,
            mew=0.55,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.4,
            capthick=1.0,
            zorder=3,
        )
        for yi, value in zip(y, values):
            ax.text(min(value + 0.022, 0.95), yi, f"{value:.3f}", va="center", fontsize=7.8)
        ax.set_xlim(0.5, 1.0)
        ax.set_xticks(np.arange(0.5, 1.01, 0.1))
        ax.set_xlabel("Performance")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.18, wspace=0.18)
    save_both(fig, "fig_criteria_profile")

    for label, acc, lo, hi, auc, alo, ahi in zip(
        labels, accuracy, acc_low, acc_high, auroc, auc_low, auc_high
    ):
        print(
            f"[criterion] {label}: accuracy={acc:.4f} [{lo:.4f}, {hi:.4f}], "
            f"AUROC={auc:.4f} [{alo:.4f}, {ahi:.4f}]"
        )


def robustness_bounds() -> None:
    data = load_json("redteam_v1.json")
    attacks = [
        (
            "Output tampering",
            int(data["attack_a_corpus_tampering"]["n_trials"]),
            int(data["attack_a_corpus_tampering"]["n_escapes"]),
            "o",
        ),
        (
            "Prompt injection",
            int(data["attack_b_prompt_injection"]["n_scored"]),
            int(round(data["attack_b_prompt_injection"]["injection_success_rate"] * data["attack_b_prompt_injection"]["n_scored"])),
            "s",
        ),
        (
            "Evidence poisoning",
            int(data["attack_c_poisoned_retrieval"]["n_scored"]),
            int(round(data["attack_c_poisoned_retrieval"]["poisoned_content_selection_rate"] * data["attack_c_poisoned_retrieval"]["n_scored"])),
            "^",
        ),
    ]
    assert [(name, n, successes) for name, n, successes, _ in attacks] == [
        ("Output tampering", 132, 0),
        ("Prompt injection", 40, 0),
        ("Evidence poisoning", 29, 0),
    ]

    y = np.arange(len(attacks))
    limits = np.array([100.0 * 3.0 / n for _, n, _, _ in attacks])
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for yi, ((name, n, successes, marker), upper) in enumerate(zip(attacks, limits)):
        ax.hlines(yi, 0, upper, color=BLUE, linewidth=1.5, zorder=2)
        ax.plot(0, yi, marker=marker, ms=6, mfc=BLUE, mec=DARK, mew=0.6, zorder=3)
        ax.plot(upper, yi, marker="|", ms=9, color=BLUE, mew=1.2, zorder=3)
        ax.text(upper + 0.35, yi, f"0/{n}; upper {upper:.1f}%", va="center", fontsize=7.8)

    ax.set_yticks(y, [name for name, _, _, _ in attacks])
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 12.0)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.set_xlabel("Attack-success rate (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.97, bottom=0.23)
    save_both(fig, "fig_robustness_bounds")

    for (name, n, _, _), upper in zip(attacks, limits):
        print(f"[robustness] {name}: 0/{n}, rule-of-three upper={upper:.4f}%")


if __name__ == "__main__":
    criterion_profile()
    robustness_bounds()
