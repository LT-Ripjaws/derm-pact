"""Deterministic template-report baseline for the agent evaluation (reviewer ask C1).

Answers the SAME 591 questions on the SAME 203 contracts as agent_eval_report_v5_4.json, but by
string formatting from the decision record and the bound guidance -- no language model, no
sampling, zero generation calls. Scored with the SAME endpoint functions (score_citations,
score_refusal, score_counterfactual, score_fabrication) and checked with the SAME deterministic
draft checker, then aggregated with the SAME _build_report.

Purpose: separate what the record guarantees from what the 32B model adds. If the template matches
the agent on the structured endpoints, those endpoints are properties of the record-and-checks
architecture; the model buys conversational flexibility, not endpoint reliability.

No paid calls. Writes research/evidence/derm7pt_v1/template_baseline_v1.json.

Usage:  python scripts/dev/build_template_baseline_agent.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.clinician_agent import AgentAnswer, AgentTools, _draft_issues  # noqa: E402
from medguard_v2.eval.agent_eval import (  # noqa: E402
    Question,
    _build_contracts,
    _build_report,
    build_questions,
    score_citations,
    score_counterfactual,
    score_fabrication,
    score_refusal,
)
from medguard_v2.evidence.guideline_binding import GuidelineCatalog  # noqa: E402

OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "template_baseline_v1.json"
OUT_TEST = REPO / "research" / "evidence" / "derm7pt_v1" / "template_baseline_test_v1.json"
REFERENCE = REPO / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_v5_4.json"
REFERENCE_TEST = REPO / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_test_v1.json"


def _stub_chat(messages):  # pragma: no cover - never called (llm_verifier=False)
    raise RuntimeError("template baseline must not call a language model")


def _criteria_lines(contract) -> List[str]:
    lines = []
    for row in contract.score.per_criterion:
        if not row.determinate:
            lines.append(f"{row.name.replace('_', ' ')}: indeterminate")
        elif row.points:
            lines.append(f"{row.name.replace('_', ' ')} = {contract.criteria[row.name]} (+{row.points})")
    return lines


def template_answer(question: Question, contract, catalog: GuidelineCatalog) -> AgentAnswer:
    tools = AgentTools(contract, catalog, gated_retrieval=True)
    score = contract.score
    rng = (f"{score.score_min}" if score.score_min == score.score_max
           else f"{score.score_min} to {score.score_max}")

    if question.kind == "out_of_scope":
        text = ("I cannot answer that. This system explains the recorded assessment and the "
                "licensed guidance for it; it does not estimate prognosis.")
        return AgentAnswer(kind="refusal", text=text, turns=0,
                           verification={}, retrieval=tools.retrieval_summary())

    if question.kind == "why":
        contributing = _criteria_lines(contract)
        if contract.conflict.present:
            body = (f"The classifier commits to {contract.classifier_diagnosis}, while the checklist "
                    f"score of {rng} meets the suspicion threshold of 3. Contributing findings: "
                    + "; ".join(contributing) + ". Re-examine the listed findings on the image; "
                    "a clinician correction to any of them is recomputed through the same arithmetic.")
        else:
            indet = [r.name.replace("_", " ") for r in score.per_criterion if not r.determinate]
            body = (f"The following criteria could not be assessed: {', '.join(indet)}. "
                    f"The checklist score is therefore the range {rng} against the threshold of 3. "
                    "Recorded findings: " + ("; ".join(contributing) or "none scoring") + ".")
        text = f"Diagnosis: recorded assessment. {body}"
        citations: tuple = ()
    elif question.kind == "override":
        criterion, value = question.override
        result = tools.override_criterion(criterion, value)
        p = result.payload
        text = (f"Your correction is applied: {criterion.replace('_', ' ')} is now '{value}' (was '{p['was']}'). "
                f"The checklist score changes from {p['score_before']['min']} to "
                f"{p['score_after']['min']} against the threshold of {p['threshold']}. "
                f"The recommendation is {p['verdict_after']}"
                + (", changed from " + p["verdict_before"] if p["verdict_changed"] else ", unchanged")
                + ". Computed by the host arithmetic, not generated.")
        citations = ()
    else:  # guideline
        cond = contract.classifier_diagnosis
        result = tools.search_guidelines(cond, "initial_local_management")
        p = result.payload
        recs = p.get("recommendations") or []
        if recs:
            r = recs[0]
            text = (f"Assessment: For the recorded diagnosis ({cond}), the guideline states: "
                    f"\"{r['text']}\" [{r['id']}]")
            citations = (r["id"],)
        else:
            reason = p.get("absence_reason") or result.note or ""
            text = (f"The guideline collection holds no recommendation for {cond}. "
                    f"{reason}").strip()
            citations = ()

    issues, source = _draft_issues(
        contract=contract, catalog=catalog, question=question.text, text=text,
        citations=citations, tools=tools, chat=_stub_chat, llm_verifier=False,
    )
    return AgentAnswer(
        kind="answer", text=text, citations=citations,
        tool_calls=tuple(tools.calls), turns=0,
        overrides_applied=dict(tools.overrides),
        verification={"issues_first_draft": list(issues), "revised": False,
                      "revisions": 0, "issues_final": list(issues),
                      "issue_source": source},
        retrieval=tools.retrieval_summary(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="use the frozen TEST contracts artifact (no test re-read) instead of the validation cohort")
    args = parser.parse_args()

    catalog = GuidelineCatalog.load()
    if args.test:
        from medguard_v2.evidence.contract import build_contract
        raw = json.loads((REPO / "research/evidence/derm7pt_v1/cdss_contracts_test_promoted.json")
                         .read_text(encoding="utf-8"))
        entries = next(v for v in raw.values() if isinstance(v, list)) if isinstance(raw, dict) else raw
        contracts = [build_contract(
            case_id=e["case_id"],
            classifier_diagnosis=str(e["classifier"]["diagnosis"]),
            classifier_confidence=float(e["classifier"]["confidence"]),
            criteria={c["name"]: c["value"] for c in e["criteria"]},
            catalog=catalog,
            criterion_margins=e.get("criterion_margins"),
        ) for e in entries]
    else:
        contracts = _build_contracts(10_000, cohort="all", promoted=True)
    print(f"{len(contracts)} contracts")

    rows: List[Dict] = []
    for contract in contracts:
        for question in build_questions(contract):
            answer = template_answer(question, contract, catalog)
            rows.append({
                "case_id": question.case_id,
                "kind": question.kind,
                "question": question.text,
                "answer_kind": answer.kind,
                "answer": answer.text,
                "turns": answer.turns,
                "tool_calls": [dict(c) for c in answer.tool_calls],
                "citations": score_citations(answer, catalog),
                "refusal": score_refusal(question, answer),
                "counterfactual": score_counterfactual(question, answer, contract),
                "fabrication": score_fabrication(answer, catalog),
                "verification": dict(answer.verification),
                "retrieval": dict(answer.retrieval),
            })

    report = _build_report(rows, n_cases=len(contracts), model="deterministic-template-baseline",
                           cohort="all", llm_verifier=False, gated_retrieval=True, promoted=True)
    report["protocol"] = "medguard-v2-template-baseline-v1"
    report["note"] = ("Deterministic rendering from the decision record; same questions, contracts, "
                      "scorers, and deterministic draft checker as agent_eval_report_v5_4.json; "
                      "zero language-model calls.")

    ref = json.loads((REFERENCE_TEST if args.test else REFERENCE).read_text(encoding="utf-8"))
    print(f"\nquestions: {len(rows)} (reference {ref['n_questions']})")
    print(f"{'endpoint':38s} {'template':>10s} {'agent v5.4':>10s}")
    for name, path in [
        ("citation_resolvability", ("citation_resolvability", "resolvability")),
        ("refusal_correctness", ("refusal_correctness", "overall")),
        ("override_consistency", ("counterfactual_accuracy", "correct")),
        ("absence_disclosure", ("absence_disclosure", "states_absence")),
        ("management_fabrication", ("management_fabrication", "rate")),
    ]:
        a = report["endpoints"][path[0]].get(path[1])
        b = ref["endpoints"][path[0]].get(path[1])
        print(f"{name:38s} {str(a):>10s} {str(b):>10s}")
    checked = [r for r in rows if r["verification"].get("issues_first_draft") is not None]
    n_bad = sum(1 for r in checked if r["verification"]["issues_first_draft"])
    print(f"first drafts with issues: {n_bad} of {len(checked)} checked")
    for r in checked:
        if r["verification"]["issues_first_draft"]:
            print("  ISSUE:", r["case_id"], r["kind"], r["verification"]["issues_first_draft"][:2])
            break

    if args.write:
        target = OUT_TEST if args.test else OUT
        target.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("wrote", target)


if __name__ == "__main__":
    main()
