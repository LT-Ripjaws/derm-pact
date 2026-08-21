"""Rescore the LLM-verifier catch-rate study against the corrected counterfactual oracle.

``llm_verifier_study_v1.json`` (scripts/dev/run_llm_verifier_study.py) scored the LLM verifier
and ``deterministic_checks`` against ground truth taken from each override row's
``counterfactual`` block in agent_eval_report_v4.json -- i.e. the OLD counterfactual scorer.
That scorer had a context-extraction bug fixed after the study ran. The fix's effect on the 193
override rows is already computed and recorded per-row in
``agent_eval_report_v4_rescored.json`` (``old_correct`` / ``new_correct``, host-computed exact
arithmetic, no judge involved).

This script does no new scoring. It joins the study's 193 override-with-oracle rows to the
rescored file by ``case_id``, then recomputes the same confusion matrices
(``llm_verifier_study_v1.json`` already reports) using ``new_correct`` in place of the old
oracle, so the old and new detector performance can be read side by side.

Both `old_*` matrices below are recomputed from ``per_row`` data here (not copied from the
study file) and cross-checked against the study's own stored top-level confusion matrices, and
against the rescored file's ``old_correct`` field -- any mismatch is surfaced under
``join.old_oracle_consistency_mismatches`` rather than silently assumed away.

Output: research/evidence/derm7pt_v1/llm_verifier_study_v1_rescored_oracle.json

Usage:
    py -3.13 scripts/dev/recompute_verifier_study_oracle.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STUDY = REPO / "research" / "evidence" / "derm7pt_v1" / "llm_verifier_study_v1.json"
RESCORED = REPO / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_v4_rescored.json"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "llm_verifier_study_v1_rescored_oracle.json"

EXPECTED_N_OVERRIDE_ROWS = 193


def _confusion(pairs: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    """pairs = [(flagged, oracle_wrong), ...] -> counts + precision/recall/F1.

    Mirrors scripts/dev/run_llm_verifier_study.py::_confusion exactly so old/new numbers are
    computed the same way.
    """
    tp = sum(1 for f, o in pairs if f and o)
    fp = sum(1 for f, o in pairs if f and not o)
    fn = sum(1 for f, o in pairs if not f and o)
    tn = sum(1 for f, o in pairs if not f and not o)
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "base_rate_oracle_wrong": round(sum(1 for _, o in pairs if o) / len(pairs), 4) if pairs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--rescored", type=Path, default=RESCORED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    study = json.loads(args.study.read_text(encoding="utf-8"))
    rescored = json.loads(args.rescored.read_text(encoding="utf-8"))

    override_rows = [
        r for r in study["per_row"]
        if r.get("kind") == "override" and r.get("oracle") is not None
    ]
    rescored_rows = rescored["per_row"]
    rescored_map = {r["case_id"]: r for r in rescored_rows}

    study_ids = [r["case_id"] for r in override_rows]
    study_id_set = set(study_ids)
    rescored_id_set = set(rescored_map)

    # --------------------------------------------------------------- join verification

    duplicate_study_ids = sorted({cid for cid in study_ids if study_ids.count(cid) > 1})
    missing_in_rescored = sorted(study_id_set - rescored_id_set)
    missing_in_study = sorted(rescored_id_set - study_id_set)

    join_ok = (
        len(override_rows) == EXPECTED_N_OVERRIDE_ROWS
        and len(rescored_rows) == EXPECTED_N_OVERRIDE_ROWS
        and not duplicate_study_ids
        and not missing_in_rescored
        and not missing_in_study
    )

    if not join_ok:
        print("JOIN VERIFICATION FAILED:")
        print(f"  study override-with-oracle rows: {len(override_rows)} (expected {EXPECTED_N_OVERRIDE_ROWS})")
        print(f"  rescored per_row rows: {len(rescored_rows)} (expected {EXPECTED_N_OVERRIDE_ROWS})")
        print(f"  duplicate case_ids in study subset: {duplicate_study_ids[:10]}")
        print(f"  case_ids in study but not rescored: {missing_in_rescored[:10]}")
        print(f"  case_ids in rescored but not study: {missing_in_study[:10]}")
        raise SystemExit(1)

    # Cross-check: the study's own OLD oracle (per-row oracle["correct"], sourced from v4's
    # stored `counterfactual` block) should agree with the rescored file's `old_correct` for
    # every joined row -- both claim to describe the same pre-fix scorer output.
    old_oracle_consistency_mismatches = []
    for r in override_rows:
        cid = r["case_id"]
        study_old = bool(r["oracle"]["correct"])
        rescored_old = bool(rescored_map[cid]["old_correct"])
        if study_old != rescored_old:
            old_oracle_consistency_mismatches.append(
                {"case_id": cid, "study_oracle_correct": study_old, "rescored_old_correct": rescored_old}
            )

    # --------------------------------------------------------------- build pairs

    # "matched subset" = rows where the LLM verifier call succeeded (no error), matching the
    # subset the study's own confusion_matrix_llm_verifier_vs_oracle was computed over.
    old_verifier_pairs: List[Tuple[bool, bool]] = []
    new_verifier_pairs: List[Tuple[bool, bool]] = []
    old_det_pairs_matched: List[Tuple[bool, bool]] = []
    new_det_pairs_matched: List[Tuple[bool, bool]] = []
    old_det_pairs_all: List[Tuple[bool, bool]] = []
    new_det_pairs_all: List[Tuple[bool, bool]] = []

    both_wrong = fixed_by_rescorer = newly_broken_by_rescorer = both_correct = 0
    n_verifier_errors = 0

    for r in override_rows:
        cid = r["case_id"]
        old_wrong = not bool(r["oracle"]["correct"])
        new_wrong = not bool(rescored_map[cid]["new_correct"])

        if old_wrong and new_wrong:
            both_wrong += 1
        elif old_wrong and not new_wrong:
            fixed_by_rescorer += 1
        elif not old_wrong and new_wrong:
            newly_broken_by_rescorer += 1
        else:
            both_correct += 1

        det_flag = bool(r["deterministic"]["flagged"])
        old_det_pairs_all.append((det_flag, old_wrong))
        new_det_pairs_all.append((det_flag, new_wrong))

        v = r["verifier"]
        if "error" in v:
            n_verifier_errors += 1
            continue

        v_flag = bool(v["flagged"])
        old_verifier_pairs.append((v_flag, old_wrong))
        new_verifier_pairs.append((v_flag, new_wrong))
        old_det_pairs_matched.append((det_flag, old_wrong))
        new_det_pairs_matched.append((det_flag, new_wrong))

    confusion_verifier_old = _confusion(old_verifier_pairs)
    confusion_verifier_new = _confusion(new_verifier_pairs)
    confusion_det_matched_old = _confusion(old_det_pairs_matched)
    confusion_det_matched_new = _confusion(new_det_pairs_matched)
    confusion_det_all_old = _confusion(old_det_pairs_all)
    confusion_det_all_new = _confusion(new_det_pairs_all)

    # Internal consistency: recomputed OLD matrices should equal the study file's own stored
    # top-level matrices (same rows, same oracle, same logic). A mismatch would mean this
    # script's join/selection diverges from the original study run.
    stored_old = {
        "confusion_matrix_llm_verifier_vs_oracle": study.get("confusion_matrix_llm_verifier_vs_oracle"),
        "confusion_matrix_deterministic_vs_oracle_matched_subset": study.get(
            "confusion_matrix_deterministic_vs_oracle_matched_subset"
        ),
        "confusion_matrix_deterministic_vs_oracle_all_override_rows": study.get(
            "confusion_matrix_deterministic_vs_oracle_all_override_rows"
        ),
    }
    recomputed_old = {
        "confusion_matrix_llm_verifier_vs_oracle": confusion_verifier_old,
        "confusion_matrix_deterministic_vs_oracle_matched_subset": confusion_det_matched_old,
        "confusion_matrix_deterministic_vs_oracle_all_override_rows": confusion_det_all_old,
    }
    stored_old_matrix_mismatches = [
        key for key in stored_old
        if stored_old[key] is not None and stored_old[key] != recomputed_old[key]
    ]

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    output = {
        "protocol": "verifier_study_oracle_rescore_v1",
        "source_study": _rel(args.study),
        "source_rescored_oracle": _rel(args.rescored),
        "note": (
            "old_* matrices use the study's original oracle (agent_eval_report_v4.json "
            "counterfactual.correct, i.e. the pre-fix scorer). new_* matrices substitute "
            "agent_eval_report_v4_rescored.json's new_correct (context-anchored extraction "
            "fix) for the same 193 override rows, same verifier/deterministic flags -- only "
            "the oracle label changes."
        ),
        "join": {
            "n_study_override_with_oracle_rows": len(override_rows),
            "n_rescored_rows": len(rescored_rows),
            "expected_n": EXPECTED_N_OVERRIDE_ROWS,
            "n_joined": len(override_rows) if join_ok else len(study_id_set & rescored_id_set),
            "join_ok": join_ok,
            "duplicate_case_ids_in_study_subset": duplicate_study_ids,
            "case_ids_missing_in_rescored": missing_in_rescored,
            "case_ids_missing_in_study": missing_in_study,
            "old_oracle_consistency_mismatches": old_oracle_consistency_mismatches,
            "stored_old_matrix_mismatches": stored_old_matrix_mismatches,
        },
        "oracle_base_rates": {
            "old_wrong": both_wrong + fixed_by_rescorer,
            "old_wrong_rate": round((both_wrong + fixed_by_rescorer) / len(override_rows), 4),
            "new_wrong": both_wrong + newly_broken_by_rescorer,
            "new_wrong_rate": round((both_wrong + newly_broken_by_rescorer) / len(override_rows), 4),
        },
        "oracle_transition": {
            "description": (
                "2x2 breakdown of old_correct x new_correct over the 193 override rows: how "
                "the scorer fix changed each row's ground-truth label, not detector output."
            ),
            "both_wrong_old_and_new": both_wrong,
            "fixed_by_rescorer_old_wrong_new_correct": fixed_by_rescorer,
            "newly_broken_by_rescorer_old_correct_new_wrong": newly_broken_by_rescorer,
            "both_correct_old_and_new": both_correct,
            "rows_changed": fixed_by_rescorer + newly_broken_by_rescorer,
        },
        "n_verifier_errors_excluded_from_matched_subsets": n_verifier_errors,
        "confusion_matrix_llm_verifier_vs_oracle": {
            "old": confusion_verifier_old,
            "new": confusion_verifier_new,
        },
        "confusion_matrix_deterministic_vs_oracle_matched_subset": {
            "old": confusion_det_matched_old,
            "new": confusion_det_matched_new,
        },
        "confusion_matrix_deterministic_vs_oracle_all_override_rows": {
            "old": confusion_det_all_old,
            "new": confusion_det_all_new,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"Join: {len(override_rows)} study rows x {len(rescored_rows)} rescored rows -> "
          f"{output['join']['n_joined']} joined (ok={join_ok})")
    if old_oracle_consistency_mismatches:
        print(f"  WARNING: {len(old_oracle_consistency_mismatches)} old-oracle consistency mismatches")
    if stored_old_matrix_mismatches:
        print(f"  WARNING: recomputed-old differs from study's stored old matrices for: "
              f"{stored_old_matrix_mismatches}")
    print()
    print(f"Oracle wrong-rate: old {output['oracle_base_rates']['old_wrong']}/193 "
          f"({output['oracle_base_rates']['old_wrong_rate']:.1%})  ->  "
          f"new {output['oracle_base_rates']['new_wrong']}/193 "
          f"({output['oracle_base_rates']['new_wrong_rate']:.1%})")
    print(f"  transition: both_wrong={both_wrong}  fixed={fixed_by_rescorer}  "
          f"newly_broken={newly_broken_by_rescorer}  both_correct={both_correct}")
    print()
    print(f"{'detector':<38}{'oracle':<6}{'n':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
          f"{'precision':>11}{'recall':>9}{'F1':>7}")
    for name, old_cm, new_cm in (
        ("LLM verifier (matched subset)", confusion_verifier_old, confusion_verifier_new),
        ("deterministic (matched subset)", confusion_det_matched_old, confusion_det_matched_new),
        ("deterministic (all override)", confusion_det_all_old, confusion_det_all_new),
    ):
        for tag, cm in (("old", old_cm), ("new", new_cm)):
            print(f"{name:<38}{tag:<6}{cm['n']:>5}{cm['tp']:>5}{cm['fp']:>5}{cm['fn']:>5}{cm['tn']:>5}"
                  f"{str(cm['precision']):>11}{str(cm['recall']):>9}{str(cm['f1']):>7}")


if __name__ == "__main__":
    main()
