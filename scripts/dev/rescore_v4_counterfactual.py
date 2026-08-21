"""Re-score the stored v4 agent answers with the fixed counterfactual scorer.

The fix wave replaced bare number extraction with context-anchored extraction
(shared helper in verification.py). This re-reads agent_eval_report_v4.json's
stored answers and recomputes the counterfactual endpoint with the fixed
scorer -- no agent calls, no API cost. Writes a small sidecar so the original
report stays as generated.

Output: research/evidence/derm7pt_v1/agent_eval_report_v4_rescored.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.clinician_agent import AgentAnswer
from medguard_v2.eval.agent_eval import _build_contracts, build_questions, score_counterfactual

REPORT = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v4.json"
OUT = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v4_rescored.json"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = report["rows"]

    contracts = {c.case_id: c for c in _build_contracts(203, "all")}
    questions = {}
    for contract in contracts.values():
        for q in build_questions(contract):
            questions[(q.case_id, q.kind, q.text)] = q

    rescored = []
    changed = 0
    for row in rows:
        if row.get("kind") != "override" or row.get("counterfactual") is None:
            continue
        q = questions.get((row["case_id"], row["kind"], row["question"]))
        if q is None:
            raise SystemExit(f"question not reconstructable for {row['case_id']}")
        answer = AgentAnswer(
            kind=row["answer_kind"],
            text=row.get("answer", "") or "",
            citations=tuple(row.get("citations_list", []) or []),
            overrides_applied=dict(row.get("overrides_applied", {}) or {}),
        )
        # The stored row keeps the applied overrides; the scorer needs them to
        # judge override_applied exactly as the original run did.
        if not answer.overrides_applied and row["counterfactual"].get("override_applied"):
            criterion, value = q.override
            answer = AgentAnswer(
                kind=answer.kind, text=answer.text, citations=answer.citations,
                overrides_applied={criterion: value},
            )
        new_cf = score_counterfactual(q, answer, contracts[row["case_id"]])
        old_cf = row["counterfactual"]
        if bool(new_cf["correct"]) != bool(old_cf["correct"]):
            changed += 1
        rescored.append({
            "case_id": row["case_id"],
            "old_correct": bool(old_cf["correct"]),
            "new_correct": bool(new_cf["correct"]),
            "old_asserts_wrong": bool(old_cf["asserts_a_wrong_score"]),
            "new_asserts_wrong": bool(new_cf["asserts_a_wrong_score"]),
        })

    n = len(rescored)
    out = {
        "protocol": "v4-counterfactual-rescore-with-context-anchored-extraction",
        "source_report": REPORT.name,
        "n_override_rows": n,
        "old_correct": sum(1 for r in rescored if r["old_correct"]),
        "new_correct": sum(1 for r in rescored if r["new_correct"]),
        "old_accuracy": round(sum(1 for r in rescored if r["old_correct"]) / n, 4),
        "new_accuracy": round(sum(1 for r in rescored if r["new_correct"]) / n, 4),
        "rows_changed": changed,
        "note": (
            "Same stored v4 answers, fixed scorer only (context-anchored number "
            "extraction, shared with the in-loop check). The delta measures scorer "
            "artifacts, not model behaviour."
        ),
        "per_row": rescored,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"n={n}  old={out['old_accuracy']}  new={out['new_accuracy']}  changed={changed}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
