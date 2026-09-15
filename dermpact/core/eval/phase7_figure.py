"""Create a publication-ready Phase 7 evaluation summary figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "research/manifests/dermpact_phase7/final_development_evaluation_v1/headline.json"
OUT = REPORT.parent


def build() -> None:
    data = json.loads(REPORT.read_text())
    metrics = data["confirmatory_300"]
    ragas = data["ragas_100x5"]
    attack = data["adversarial_120"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), constrained_layout=True)
    blue, orange, green, black = "#0072B2", "#E69F00", "#009E73", "#222222"

    names = ["Workflow", "Strict query", "Fact precision", "Traceability",
             "Fact F1", "Tool select", "Tool args"]
    keys = ["workflow_task_success", "query_accuracy", "fact_precision",
            "required_fact_traceability", "fact_f1", "tool_selection_correct", "tool_argument_correct"]
    values = np.array([metrics[key]["estimate"] for key in keys])
    lower = values - np.array([metrics[key]["ci_low"] for key in keys])
    upper = np.array([metrics[key]["ci_high"] for key in keys]) - values
    axes[0].bar(np.arange(len(values)), values, color=[green, orange, green, blue, blue, green, orange], width=0.72)
    axes[0].errorbar(np.arange(len(values)), values, yerr=np.vstack([lower, upper]), fmt="none",
                     ecolor=black, capsize=2, linewidth=0.8)
    axes[0].set_xticks(np.arange(len(values)), names, rotation=45, ha="right")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Rate")
    axes[0].set_title("A  Ordinary tasks (300; 100 cases)", loc="left", fontweight="bold", fontsize=8.5)

    safety_names = ["Initial hazard", "Final hazard", "Attack utility", "Attack final"]
    safety_values = [metrics["initial_hazard"]["estimate"], metrics["final_hazard"]["estimate"],
                     attack["protocol_valid"]["estimate"], attack["attack_success_final_hazard"]["estimate"]]
    axes[1].bar(np.arange(4), safety_values, color=[orange, green, blue, green], width=0.68)
    axes[1].set_xticks(np.arange(4), safety_names, rotation=40, ha="right")
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Rate")
    axes[1].set_title("B  Verification and attacks", loc="left", fontweight="bold", fontsize=8.5)
    axes[1].text(0, safety_values[0] + 0.04, f"{safety_values[0]:.3f}", ha="center", fontsize=7)
    axes[1].text(2, safety_values[2] + 0.04, f"{safety_values[2]:.3f}", ha="center", fontsize=7)

    pass_means = [row["mean"] for row in ragas["per_pass"]]
    axes[2].plot(np.arange(1, 6), pass_means, color=blue, marker="o", linewidth=1.5)
    axes[2].axhline(ragas["mean_across_passes"], color=black, linestyle="--", linewidth=0.8,
                    label=f"Mean {ragas['mean_across_passes']:.3f}")
    axes[2].set_xticks(np.arange(1, 6)); axes[2].set_xlabel("Judge pass")
    axes[2].set_ylabel("RAGAS faithfulness"); axes[2].set_ylim(0.90, 1.0)
    axes[2].set_title("C  Grounding (100 cases)", loc="left", fontweight="bold", fontsize=8.5)
    axes[2].legend(frameon=False, fontsize=7, loc="lower right")

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "phase7_evaluation_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "phase7_evaluation_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    (OUT / "phase7_evaluation_summary_caption.txt").write_text(
        "Phase 7 development evaluation. (A) Case-clustered ordinary-task metrics; error bars are 95% "
        "case-clustered bootstrap intervals. (B) Initial and final displayed hazard for 300 ordinary tasks, "
        "and protocol-valid utility/final hazard under 120 prespecified attacks. (C) Five repeated RAGAS-style "
        "faithfulness judgments on 100 answers from 100 cases. Repeated judge passes are measurements, not "
        "independent samples. No official/final partition was opened.\n", encoding="utf-8")


if __name__ == "__main__":
    build()
