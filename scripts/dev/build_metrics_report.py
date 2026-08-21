"""Render a completed H2H report JSON into a paper-ready complete-metrics markdown.

Surfaces every scored axis for the M (method) arm plus B0 (VLM baseline), including the
newly-wired confusion matrix / per-class P-R-F1 / Cohen kappa / binary accuracy / ECE /
alignment / RAGAS answer-relevancy + context-recall. Honest positioning vs CALIBER
(same frozen DermLIP probe -> accuracy is inherited, not a win) is left to prose; this
file is the number surface.

    python scripts/dev/build_metrics_report.py --report <path.json> --out notes/METRICS_COMPLETE.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd=4):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def confusion_md(diag):
    labels = diag.get("labels", [])
    cm = diag.get("confusion_matrix", [])
    head = "| true \\ pred | " + " | ".join(labels) + " | support |"
    sep = "|" + "---|" * (len(labels) + 2)
    rows = [head, sep]
    for i, lab in enumerate(labels):
        support = sum(cm[i]) if i < len(cm) else 0
        cells = " | ".join(str(cm[i][j]) for j in range(len(labels))) if i < len(cm) else ""
        rows.append(f"| **{lab}** | {cells} | {support} |")
    return "\n".join(rows)


def per_class_md(diag):
    pc = diag.get("per_class", {})
    rows = ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for lab, d in pc.items():
        rows.append(f"| {lab} | {fmt(d['precision'],3)} | {fmt(d['recall'],3)} | "
                    f"{fmt(d['f1'],3)} | {d['support']} |")
    return "\n".join(rows)


def arm_section(name, blk):
    diag = blk.get("diagnosis", {}) or {}
    act = blk.get("actionable", {}) or {}
    sel = blk.get("selective", {}) or {}
    prov = blk.get("provenance_citation_integrity", {}) or {}
    proc = blk.get("process_contract_completion", {}) or {}
    trust = blk.get("process_trust", {}) or {}
    align = blk.get("alignment", {}) or {}
    rag = (blk.get("ragas_supporting", {}) or {}).get("scores") or {}
    faith = rag.get("faithfulness") or {}
    ar = rag.get("answer_relevancy") or {}
    cr = rag.get("context_recall") or {}
    out = [f"## {name}  (n={blk.get('cases')})", ""]
    out += ["### Classification (7-way)",
            f"- accuracy **{fmt(diag.get('accuracy'))}** · balanced-acc {fmt(diag.get('balanced_accuracy'))} "
            f"· macro-F1 {fmt(diag.get('macro_f1'))} · weighted-F1 {fmt(diag.get('weighted_f1'))} "
            f"· MCC {fmt(diag.get('mcc'))} · Cohen κ **{fmt(diag.get('cohen_kappa'))}**", ""]
    if diag.get("per_class"):
        out += ["**Per-class precision / recall / F1:**", "", per_class_md(diag), ""]
    if diag.get("confusion_matrix"):
        out += ["**Confusion matrix:**", "", confusion_md(diag), ""]
    out += ["### Binary (malignant vs benign)",
            f"- accuracy **{fmt(act.get('accuracy'))}** · sensitivity {fmt(act.get('sensitivity'))} "
            f"· specificity {fmt(act.get('specificity'))} · AUROC **{fmt(act.get('auroc'))}** "
            f"· PPV {fmt(act.get('ppv'))} · NPV {fmt(act.get('npv'))}",
            f"- sensitivity@90%spec: {json.dumps(act.get('sensitivity_at_fixed_specificity'), default=str)}", ""]
    out += ["### Calibration & selective",
            f"- **ECE {fmt(sel.get('ece'))}** · AURC {fmt(sel.get('aurc'))} · coverage {fmt(sel.get('coverage'))} "
            f"· answered-accuracy **{fmt(sel.get('answered_accuracy'))}**", ""]
    out += ["### Provenance & process",
            f"- citation integrity **{fmt(prov.get('integrity_rate'))}** "
            f"({prov.get('intact_citations')}/{prov.get('total_citations')} intact; "
            f"{prov.get('cases_with_citations')}/{prov.get('cases_scored')} cited)",
            f"- contract completion {fmt(proc.get('contract_complete_rate'))} · "
            f"pass-eligible {fmt(proc.get('structured_pass_eligible_rate'))} · "
            f"trust P1–P8 composite **{fmt(trust.get('composite_mean'))}**", ""]
    out += ["### Alignment (MMed-RAG / Rethinking-RAG family)",
            f"- over-reliance (answered-but-wrong) {fmt(align.get('over_reliance_confident_wrong'))} · "
            f"**diagnosis over-reliance {fmt(align.get('diagnosis_over_reliance'))}** (architectural) · "
            f"copy-reference {fmt(align.get('copy_reference_ratio'))} · "
            f"completeness {fmt(align.get('completeness'))} · "
            f"no-citation-answer rate {fmt(align.get('no_citation_answer_rate'))}", ""]
    out += ["### RAGAS (supporting; independent judge)",
            f"- faithfulness (assessment) **{fmt(faith.get('assessment_mean'))}** "
            f"(answered {fmt(faith.get('answered_mean'))}, n_assessment={faith.get('n_assessment_scored')})",
            f"- answer-relevancy {fmt(ar.get('mean'))} (n={ar.get('n_scored')}) · "
            f"context-recall {fmt(cr.get('mean'))} (n={cr.get('n_scored')})", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="M,B0")
    args = ap.parse_args()
    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    per_arm = rep["scored"]["per_arm"]
    paired = rep["scored"].get("paired_differences", [])
    lines = ["# MedGuard — complete metric surface (dev-160)", "",
             "> Auto-generated from the H2H report. Same frozen DermLIP probe as the prior",
             "> derm-RAG work, so classification/AUC are **inherited, not a win**; the",
             "> contribution is the host-verifiable provenance/contract/alignment axis.", ""]
    for pd in paired:
        lines.append(f"- **{pd['arm_a']}−{pd['arm_b']} {pd['metric']}**: "
                     f"{fmt(pd['estimate'])} [{fmt(pd['low'])}, {fmt(pd['high'])}] (n={pd['n_paired']})")
    lines.append("")
    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm in per_arm:
            lines.append(arm_section(arm, per_arm[arm]))
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
