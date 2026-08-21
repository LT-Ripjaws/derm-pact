"""Offline, free prediction of which axes MedGuard can win before we spend.

The paper's acceptability rests on a matched H2H table where MedGuard beats
baselines on axes reviewers respect. Diagnosis is tied by construction -- every
arm shares the same frozen perception commitment -- so the win has to come from
elsewhere. This module asks, using only artifacts already on disk, whether the
axes we have chosen actually carry signal, so that a no-go is discovered now
rather than after a paid generation run.

It answers four questions and returns a go / go-conditional / no-go per axis:

  A. Selective prediction (AURC). Is the frozen perception confidence
     discriminative enough that abstention can help at all? If confidence does
     not separate right from wrong, no architecture wins AURC and the headline
     must change.

  B. Over-reliance (the headline). MedGuard binds the diagnosis to the image and
     forbids retrieved evidence from changing it. Is that protection worth
     anything -- i.e. would a naive arm that fuses retrieved exemplar labels
     actually corrupt cases the perception layer already gets right? If the net
     effect is protective, over-reliance is a real, shared-axis win.

  C. Provenance (exclusive). On how many lesions can MedGuard produce a
     source-exact citation that no baseline architecture can produce at all?

  D. Prior corroboration. What did the retired H2H run show on these axes, with
     the caveat that the architecture and case set differ.

This uses development labels deliberately: choosing the study's headline on
development data is what development data is for. The sealed test is not touched.
No generation, no network. The report is byte-reproducible.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from medguard.eval.metrics import (
    DERM_CLASS_ORDER,
    aurc,
    classification_summary,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)


ROOT = Path(__file__).resolve().parents[2]

PERCEPTION = ROOT / "research/experiments/medguard_h2h_v1/perception/dermlip_v1/predictions.jsonl"
DEV_MANIFEST = ROOT / "research/manifests/medguard_h2h_development_v1.json"
SPLIT = ROOT / "research/manifests/ham10000_split_v2.json"
PREFLIGHT = ROOT / "research/experiments/medguard_h2h_v1/development_preflight.json"
FEASIBILITY = ROOT / "research/experiments/medguard_h2h_v1/guideline_catalog_feasibility_v1.json"
REPORT = ROOT / "research/experiments/medguard_h2h_v1/win_feasibility_v1.json"

ACTIONABLE_HAM = ("akiec", "bcc", "melanoma")

# From H2H_RUN2_RESULTS.md. The architecture (retired learned gate) and case set
# (n=200, different seed) differ from the current deterministic system, so these
# are corroboration of where the mechanism family helps, not a prediction of the
# current system's numbers.
H2H_RUN2_PRIOR = {
    "source": "notes/H2H_RUN2_RESULTS.md",
    "architecture": "retired learned depth-two gate; not the current deterministic hierarchy",
    "case_set": "n=200 seed=1, not the frozen 160-lesion development manifest",
    "learned_over_reliance": 0.010,
    "learned_over_reliance_rank": "lowest of 14 variants (clean win)",
    "learned_aurc": 0.065,
    "learned_aurc_rank": "lowest (clean win); next best 0.103",
    "learned_selective_accuracy": 0.846,
    "learned_selective_accuracy_rank": "highest; +8pts over best rival at 65% coverage",
    "caveat": "The same analysis found single-operating-point harm/ECE/calls were "
              "coverage-confounded. The defensible wins were AURC, selective accuracy, "
              "over-reliance, and Pareto-optimality -- not a raw clean sweep.",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dev_labels() -> Dict[str, str]:
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    labels: Dict[str, str] = {}
    for record in manifest["records"]:
        labels.setdefault(record["cluster_id"], record["label"])
    return labels


def _isic_labels() -> Dict[str, str]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    return {record["case_id"]: record["label"] for record in split["records"]}


def axis_a_selective_signal(
    commitments: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> Dict[str, Any]:
    """Does perception confidence separate right from wrong? (AURC axis)"""
    conf = [float(row["calibrated_confidence"]) for row in commitments]
    correct = [row["committed_diagnosis"] == labels[row["cluster_id"]] for row in commitments]
    overall = sum(correct) / len(correct)

    observed_aurc = aurc(conf, correct)
    coverage_points = {
        f"selective_accuracy_at_{int(c * 100)}pct_coverage":
            selective_accuracy_at_coverage(conf, correct, c)
        for c in (0.2, 0.3, 0.5, 0.7, 1.0)
    }
    headroom = coverage_points["selective_accuracy_at_50pct_coverage"] - overall

    # Permutation floor: if confidence is uninformative, shuffling it leaves AURC
    # unchanged. A materially lower observed AURC means the ranking carries signal.
    rng = random.Random(0)
    shuffled = []
    for _ in range(1000):
        order = conf[:]
        rng.shuffle(order)
        shuffled.append(aurc(order, correct))
    shuffled_mean = sum(shuffled) / len(shuffled)

    # Go if abstention genuinely raises accuracy and the ranking beats chance.
    signal = headroom >= 0.05 and observed_aurc < shuffled_mean - 0.01
    return {
        "overall_accuracy": overall,
        "aurc_observed": observed_aurc,
        "aurc_shuffled_confidence_mean": shuffled_mean,
        "aurc_beats_chance_by": shuffled_mean - observed_aurc,
        "selective_accuracy_by_coverage": coverage_points,
        "headroom_selacc50_minus_overall": headroom,
        "risk_coverage_curve": risk_coverage_curve(conf, correct),
        "verdict": "GO" if signal else "NO-GO",
        "criterion": "headroom >= 0.05 AND observed AURC < shuffled AURC - 0.01",
        "reading": (
            "The AURC curve is shared by every arm (identical confidence, identical "
            "correctness), so this establishes that the axis carries signal, not that "
            "MedGuard's curve differs. MedGuard's edge on this axis is a better "
            "operating point and not corrupting the commitment (axis B), which needs "
            "B2/B3 plus generation to measure."
        ),
    }


def axis_b_over_reliance(
    commitments: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    per_case: Sequence[Mapping[str, Any]],
    isic_labels: Mapping[str, str],
) -> Dict[str, Any]:
    """Would naive exemplar-label fusion corrupt cases perception gets right?

    MedGuard keeps the diagnosis image-bound: retrieved exemplar labels cannot
    change it. A MedDr-style arm fuses those labels. If fusion flips more correct
    cases to wrong than it rescues, image-binding is net protective and the
    over-reliance win is real.
    """
    commitment_by_cluster = {row["cluster_id"]: row for row in commitments}

    perception_correct: List[bool] = []
    fusion_correct: List[bool] = []
    protected = 0     # perception right, exemplar-fusion would be wrong
    sacrificed = 0    # perception wrong, exemplar-fusion would be right
    disagree_with_commitment = 0

    for case in per_case:
        cluster = case["cluster_id"]
        commitment = commitment_by_cluster[cluster]
        committed = commitment["committed_diagnosis"]
        truth = labels[cluster]

        exemplar_labels = []
        for evidence in case["retrieval"]["evidence"]:
            if evidence.get("retrieval_channel") != "image_exemplar":
                continue
            isic = str(evidence["source_id"]).split(":")[-1]
            if isic in isic_labels:
                exemplar_labels.append(isic_labels[isic])
        if not exemplar_labels:
            continue

        # Majority vote; ties resolved by the nearest (first-listed) exemplar.
        counts = collections.Counter(exemplar_labels)
        top = counts.most_common()
        best = top[0][1]
        tied = [lbl for lbl, n in top if n == best]
        fused = exemplar_labels[0] if len(tied) > 1 else top[0][0]

        p_ok = committed == truth
        f_ok = fused == truth
        perception_correct.append(p_ok)
        fusion_correct.append(f_ok)
        if fused != committed:
            disagree_with_commitment += 1
        if p_ok and not f_ok:
            protected += 1
        if (not p_ok) and f_ok:
            sacrificed += 1

    n = len(perception_correct)
    perception_acc = sum(perception_correct) / n
    fusion_acc = sum(fusion_correct) / n
    net_protection = protected - sacrificed

    # Go if binding the diagnosis to the image is net protective and the attack
    # surface is not negligible.
    signal = net_protection > 0 and disagree_with_commitment >= 0.15 * n
    return {
        "lesions_scored": n,
        "perception_accuracy": perception_acc,
        "naive_exemplar_fusion_accuracy": fusion_acc,
        "accuracy_delta_perception_minus_fusion": perception_acc - fusion_acc,
        "exemplars_disagree_with_commitment": disagree_with_commitment,
        "attack_surface_fraction": disagree_with_commitment / n,
        "protected_correct_cases": protected,
        "sacrificed_recoverable_cases": sacrificed,
        "net_protection": net_protection,
        "verdict": "GO" if signal else ("GO-CONDITIONAL" if net_protection > 0 else "NO-GO"),
        "criterion": "net_protection > 0 AND attack_surface_fraction >= 0.15",
        "reading": (
            "This bounds the diagnostic over-reliance a fusion arm faces from the image "
            "channel. The text channel is an additional over-reliance vector that only "
            "generation can measure; MedGuard is structurally immune to both because the "
            "diagnosis never leaves the frozen commitment. A positive net protection means "
            "the immunity is not free -- a naive arm genuinely loses accuracy it did not "
            "have to."
        ),
    }


def axis_c_provenance(feasibility: Mapping[str, Any]) -> Dict[str, Any]:
    """On how many lesions can MedGuard cite a source-exact statement at all?"""
    reach = feasibility["summary"]["reachability"]
    total = feasibility["summary"]["lesions"]["total"]
    capable = reach["any_condition_matched_record"]
    return {
        "lesions_total": total,
        "medguard_source_exact_capable": capable,
        "baseline_source_exact_capable": 0,
        "reading": (
            "Baselines have no source-exact binding: B0 cites nothing, B1 cites PubMed "
            "abstracts by id but cannot resolve a statement to a page/section/content-hash "
            "of a guideline PDF. This axis is exclusive by construction, so it differentiates "
            "but cannot be a fair head-to-head 'win' on its own -- it must accompany a "
            "shared-axis result (A or B)."
        ),
        "verdict": "GO" if capable > 0 else "NO-GO",
        "criterion": "medguard_source_exact_capable > 0",
    }


def build_report() -> Dict[str, Any]:
    commitments = _load_jsonl(PERCEPTION)
    for row in commitments:
        if "label" in row or "benign_malignant" in row:
            raise ValueError("perception artifact carries a truth field; refusing to proceed")
    labels = _dev_labels()
    isic_labels = _isic_labels()
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    feasibility = json.loads(FEASIBILITY.read_text(encoding="utf-8"))

    # Shared perception floor every arm inherits.
    order = list(DERM_CLASS_ORDER)
    truth = [labels[row["cluster_id"]] for row in commitments]
    predicted = [row["committed_diagnosis"] for row in commitments]
    summary = classification_summary(truth, predicted, order)
    actionable_truth = [labels[row["cluster_id"]] in ACTIONABLE_HAM for row in commitments]
    actionable_pred = [row["committed_diagnosis"] in ACTIONABLE_HAM for row in commitments]
    tp = sum(1 for t, p in zip(actionable_truth, actionable_pred) if t and p)
    fn = sum(1 for t, p in zip(actionable_truth, actionable_pred) if t and not p)
    tn = sum(1 for t, p in zip(actionable_truth, actionable_pred) if not t and not p)
    fp = sum(1 for t, p in zip(actionable_truth, actionable_pred) if not t and p)

    axis_a = axis_a_selective_signal(commitments, labels)
    axis_b = axis_b_over_reliance(commitments, labels, preflight["per_case"], isic_labels)
    axis_c = axis_c_provenance(feasibility)

    winnable = [a for a in (axis_a, axis_b, axis_c) if a["verdict"] == "GO"]
    go = axis_b["verdict"] in ("GO", "GO-CONDITIONAL") and axis_a["verdict"] == "GO"

    return {
        "schema_version": "medguard-win-feasibility-v1",
        "status": "offline_free_prediction_not_a_paper_result",
        "headline_under_test": (
            "Given a fixed dermatology perception system, MedGuard makes the safest and "
            "most auditable use of it: near-zero over-reliance, competitive selective risk, "
            "and source-exact evidence, at diagnostic non-inferiority."
        ),
        "inputs": {
            "perception": {"path": str(PERCEPTION.relative_to(ROOT)).replace("\\", "/"),
                           "sha256": _sha256_file(PERCEPTION)},
            "development_manifest": {"path": str(DEV_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
                                     "sha256": _sha256_file(DEV_MANIFEST)},
            "preflight": {"path": str(PREFLIGHT.relative_to(ROOT)).replace("\\", "/"),
                          "sha256": _sha256_file(PREFLIGHT)},
            "guideline_feasibility": {"path": str(FEASIBILITY.relative_to(ROOT)).replace("\\", "/"),
                                      "sha256": _sha256_file(FEASIBILITY)},
        },
        "boundaries": {
            "generation_calls": 0,
            "network_access": False,
            "sealed_test_access": False,
            "development_labels_used": True,
            "development_labels_rationale": (
                "Choosing the study headline and confirming axis signal is a development-time "
                "decision; development labels exist for exactly this. The sealed test is untouched."
            ),
        },
        "shared_perception_floor": {
            "note": "Every arm inherits this. Diagnosis cannot be a head-to-head win.",
            "accuracy": summary["accuracy"],
            "balanced_accuracy": summary["balanced_accuracy"],
            "macro_f1": summary["macro_f1"],
            "actionable_binary": {
                "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                "sensitivity": tp / (tp + fn) if (tp + fn) else None,
                "specificity": tn / (tn + fp) if (tn + fp) else None,
            },
        },
        "axes": {
            "A_selective_prediction_aurc": axis_a,
            "B_over_reliance_headline": axis_b,
            "C_provenance_exclusive": axis_c,
        },
        "prior_corroboration": H2H_RUN2_PRIOR,
        "overall": {
            "winnable_axes_offline": [
                name for name, axis in (
                    ("A_selective_prediction_aurc", axis_a),
                    ("B_over_reliance_headline", axis_b),
                    ("C_provenance_exclusive", axis_c),
                ) if axis["verdict"] in ("GO", "GO-CONDITIONAL")
            ],
            "recommendation": "PROCEED to B2/B3 and matched H2H" if go else
                              "REFRAME the headline before building further",
            "what_this_does_not_prove": (
                "This predicts that the chosen axes carry signal and that MedGuard's "
                "architecture is positioned to win them. It does NOT establish the arm-to-arm "
                "deltas, which require B2/B3, the missing scorers, and one authorized generation "
                "run. A GO here means the paper is worth building, not that it is already won."
            ),
        },
    }


def main() -> int:
    report = build_report()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    REPORT.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (REPORT.parent / (REPORT.name + ".sha256")).write_text(digest + "\n", encoding="ascii", newline="\n")

    floor = report["shared_perception_floor"]
    print(f"shared perception floor (all arms tie): acc={floor['accuracy']:.3f} "
          f"balacc={floor['balanced_accuracy']:.3f} macroF1={floor['macro_f1']:.3f}")  # noqa: E501
    for name, axis in report["axes"].items():
        print(f"  [{axis['verdict']:15s}] {name}")
    a = report["axes"]["A_selective_prediction_aurc"]
    b = report["axes"]["B_over_reliance_headline"]
    print(f"  A: AURC {a['aurc_observed']:.3f} vs chance {a['aurc_shuffled_confidence_mean']:.3f}; "
          f"headroom {a['headroom_selacc50_minus_overall']:+.3f}")
    print(f"  B: perception {b['perception_accuracy']:.3f} vs naive fusion "
          f"{b['naive_exemplar_fusion_accuracy']:.3f}; protected {b['protected_correct_cases']}, "
          f"sacrificed {b['sacrificed_recoverable_cases']}, net {b['net_protection']:+d}")
    print(f"  recommendation: {report['overall']['recommendation']}")
    print(f"report: {REPORT.relative_to(ROOT)}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
