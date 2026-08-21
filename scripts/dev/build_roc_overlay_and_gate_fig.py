"""Rebuild two Results figures from cached DermLIP features by deterministic replay.

(1) fig_roc_all.png -- melanoma ROC curves of all five diagnosis evaluations in one axes
    (Derm7pt test, HAM10000 image-level, HAM10000 lesion-level, BCN20000 same-centre,
    BCN20000 cross-centre).
(2) fig_gate_scores.png -- unfamiliar-image warning score distributions (Derm7pt valid+test
    vs BCN20000 development images) with the registered threshold.

No paid calls, no new images encoded, no new labels read beyond the already-guarded reads the
source scripts perform. Every replayed score vector is accepted only if its summary statistic
reproduces the stored evidence artifact to 4 decimals; the script aborts otherwise.

Sources of truth being reproduced:
  research/evidence/derm7pt_v1/promoted_two_view_ablation.json  (Derm7pt ROC, stored curve)
  research/evidence/derm7pt_v1/paper_metrics_summary.json       (HAM image-level 0.9303)
  research/evidence/sealed_v2/sealed_v2_evaluation.json         (HAM lesion-level 0.9444)
  research/evidence/bcn_v1/bcn_test_evaluation.json             (BCN 0.8641 / 0.7962)
  research/evidence/derm7pt_v1/domain_gate_validation.json      (gate stats, threshold 0.835128)

Usage:  python scripts/dev/build_roc_overlay_and_gate_fig.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts" / "dev", REPO / "scripts" / "sealed"):
    sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

import build_bcn_test_evaluation as bbte  # noqa: E402
import build_ham_evaluation as bhe  # noqa: E402
import run_sealed_v2 as rsv  # noqa: E402

FIGS = REPO / "our_paper" / "figs"
OUT_CURVES = REPO / "research" / "evidence" / "derm7pt_v1" / "roc_overlay_curves.json"
plt.rcParams.update({"font.family": "serif", "font.size": 10})


def _assert_close(name: str, got: float, expected: float, tol: float = 5e-5) -> None:
    if abs(got - expected) > tol:
        raise SystemExit(f"REPRODUCTION FAILED for {name}: got {got:.6f}, expected {expected:.6f}")
    print(f"  [ok] {name}: {got:.4f} == {expected:.4f}")


def derm7pt_curve():
    abl = json.loads((REPO / "research/evidence/derm7pt_v1/promoted_two_view_ablation.json").read_text())
    stored = abl["roc_curve"]
    fpr = [pt["fpr"] for pt in stored]
    tpr = [pt["sensitivity_tpr"] for pt in stored]
    auc = float(abl["melanoma_auroc"]["point_estimate"])
    _assert_close("Derm7pt stored AUROC", auc, 0.9151)
    return np.asarray(fpr), np.asarray(tpr), auc


def ham_image_level_curve():
    features, records = bhe.load_records()
    cluster_label = bhe.lesion_labels(records)
    train_clusters, test_clusters = bhe.make_split(cluster_label)
    train_set, test_set = set(train_clusters), set(test_clusters)
    train_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in train_set]
    test_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in test_set]
    x_train, x_test = features[train_idx], features[test_idx]
    clusters_train = [str(records[i]["cluster_id"]) for i in train_idx]
    y5_train = [bhe.HAM_TO_GROUP[str(records[i]["label"])] for i in train_idx]
    y5_test = [bhe.HAM_TO_GROUP[str(records[i]["label"])] for i in test_idx]
    assert len(test_idx) == 1595, f"e3 test is {len(test_idx)} images, expected 1595"

    c5 = bhe._select_c(x_train, y5_train, clusters_train, seed=bhe.SEED)
    model5 = LogisticRegression(max_iter=5000, C=c5, random_state=bhe.SEED)
    model5.fit(x_train, y5_train)
    mel = list(model5.classes_).index("MEL")
    scores = model5.predict_proba(x_test)[:, mel]
    truth = np.asarray([t == "MEL" for t in y5_test])
    auc = float(roc_auc_score(truth, scores))
    _assert_close("HAM image-level AUROC", auc, 0.9303)
    fpr, tpr, _ = roc_curve(truth, scores)
    return fpr, tpr, auc


def ham_lesion_level_curve():
    features, records, _ = rsv.load_sealed_data()
    x_lesion, lesion_records = rsv.lesion_pool(features, records)
    assert len(lesion_records) == 1495
    clf, _, _ = rsv.fit_ham_classifier(features, records)
    mel = list(clf.classes_).index("MEL")
    scores = clf.predict_proba(x_lesion)[:, mel]
    truth = np.asarray([r["group"] == "MEL" for r in lesion_records])
    auc = float(roc_auc_score(truth, scores))
    _assert_close("HAM lesion-level AUROC", auc, 0.9444)
    fpr, tpr, _ = roc_curve(truth, scores)
    return fpr, tpr, auc


def bcn_curves():
    train_records, train_features = bbte.load_records(splits=("train",))
    train_lesions, x_train = bbte.lesion_level(train_records, train_features)
    test_records, test_features = bbte.load_records(splits=("test",), allow_test=True)
    test_lesions, x_test = bbte.lesion_level(test_records, test_features)
    assert len(test_lesions) == 712
    y_train = [r.group for r in train_lesions]
    truth = np.asarray([r.group == "MEL" for r in test_lesions])

    cv1 = bbte._select_bcn_c(x_train, y_train, [r.lesion_id for r in train_lesions])
    arm1 = LogisticRegression(max_iter=5000, C=cv1["selected_C"], random_state=bbte.BCN_SEED)
    arm1.fit(x_train, y_train)
    s1 = arm1.predict_proba(x_test)[:, list(arm1.classes_).index("MEL")]
    auc1 = float(roc_auc_score(truth, s1))
    _assert_close("BCN same-centre AUROC", auc1, 0.8641)

    arm2, _, _ = bbte._fit_derm7pt_classifier_and_heads()
    s2 = arm2.predict_proba(x_test)[:, list(arm2.classes_).index("MEL")]
    auc2 = float(roc_auc_score(truth, s2))
    _assert_close("BCN cross-centre AUROC", auc2, 0.7962)

    f1, t1, _ = roc_curve(truth, s1)
    f2, t2, _ = roc_curve(truth, s2)
    return (f1, t1, auc1), (f2, t2, auc2)


def draw_overlay(curves):
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    styles = [
        ("Derm7pt test (AUROC 0.915)", "#1f77b4", "-"),
        ("HAM10000, image level (0.930)", "#2ca02c", "-"),
        ("HAM10000, lesion level (0.944)", "#17becf", "-"),
        ("BCN20000, same centre (0.864)", "#ff7f0e", "--"),
        ("BCN20000, cross centre (0.796)", "#d62728", "--"),
    ]
    for (fpr, tpr, _), (label, color, ls) in zip(curves, styles):
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.7, label=label)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("Melanoma ROC across the five evaluations", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = FIGS / "fig_roc_all.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def gate_figure():
    from medguard_v2.data.bcn20000 import load_records as bcn_load
    from medguard_v2.data.derm7pt import load_cases
    from medguard_v2.perception.domain_check import _load_dermlip_features, build_train_bank, domain_scores

    stored = json.loads((REPO / "research/evidence/derm7pt_v1/domain_gate_validation.json").read_text())
    threshold = float(stored["registered_gate"]["threshold"])

    cases = load_cases()
    features = _load_dermlip_features()
    bank = build_train_bank(features, cases)
    valid = np.stack([features[str(c.case_num)] for c in cases if c.split == "valid"])
    test = np.stack([features[str(c.case_num)] for c in cases if c.split == "test"])
    bcn_records, bcn_images = bcn_load(splits=("train", "valid"))

    s_valid = domain_scores(bank, valid, 1)
    s_test = domain_scores(bank, test, 1)
    s_bcn = domain_scores(bank, bcn_images, 1)

    ref = stored["k_grid"]["1"]["scores"]
    _assert_close("gate valid mean", float(np.mean(s_valid)), ref["derm7pt_valid"]["mean"], 5e-5)
    _assert_close("gate test mean", float(np.mean(s_test)), ref["derm7pt_test"]["mean"], 5e-5)
    _assert_close("gate BCN image mean", float(np.mean(s_bcn)), ref["bcn_image"]["mean"], 5e-5)
    _assert_close("gate BCN flagged fraction", float(np.mean(s_bcn < threshold)),
                  stored["k_grid"]["1"]["flagged_at_threshold"]["bcn_image"]["flagged_fraction"], 5e-5)
    _assert_close("gate Derm7pt-test flagged fraction", float(np.mean(s_test < threshold)),
                  stored["k_grid"]["1"]["flagged_at_threshold"]["derm7pt_test"]["flagged_fraction"], 5e-5)

    derm = np.concatenate([s_valid, s_test])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = np.linspace(0.66, 1.0, 55)
    ax.hist(s_bcn, bins=bins, density=True, alpha=0.55, color="#d62728",
            label=f"BCN20000 images (n = {len(s_bcn):,})")
    ax.hist(derm, bins=bins, density=True, alpha=0.60, color="#1f77b4",
            label=f"Derm7pt valid + test (n = {len(derm)})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.3)
    ax.text(threshold - 0.004, ax.get_ylim()[1] * 0.58, "warning threshold\n(fixed on Derm7pt)",
            ha="right", va="top", fontsize=8)
    ax.set_xlabel("Similarity to the Derm7pt training bank (domain score)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = FIGS / "fig_gate_scores.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def main():
    print("[1/5] Derm7pt stored curve")
    c1 = derm7pt_curve()
    print("[2/5] HAM image-level replay")
    c2 = ham_image_level_curve()
    print("[3/5] HAM lesion-level replay")
    c3 = ham_lesion_level_curve()
    print("[4/5] BCN replays")
    c4, c5 = bcn_curves()

    curves = [c1, c2, c3, c4, c5]
    OUT_CURVES.write_text(json.dumps({
        "protocol": "roc-overlay-deterministic-replay-v1",
        "note": "fpr/tpr replayed from cached features; accepted only after point-AUROC matched "
                "the stored evidence artifact to 4 decimals (see build_roc_overlay_and_gate_fig.py)",
        "series": [
            {"name": n, "auroc": round(a, 4), "fpr": [round(float(x), 5) for x in f],
             "tpr": [round(float(x), 5) for x in t]}
            for n, (f, t, a) in zip(
                ["derm7pt_test", "ham_image_level", "ham_lesion_level", "bcn_same_centre", "bcn_cross_centre"],
                curves)
        ],
    }, indent=1))
    print("saved", OUT_CURVES)
    draw_overlay(curves)
    print("[5/5] gate score figure")
    gate_figure()


if __name__ == "__main__":
    main()
