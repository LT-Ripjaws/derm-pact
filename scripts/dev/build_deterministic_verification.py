"""Run deterministic_checks() on all agent answers in a report.

Loads each answer from an agent_eval_report_*.json (default: v3), reconstructs
the contract from the pipeline, and runs the zero-cost deterministic verifier.

Output: research/evidence/derm7pt_v1/deterministic_verification.json (default)

Usage:
    py -3.13 scripts/dev/build_deterministic_verification.py
    py -3.13 scripts/dev/build_deterministic_verification.py \\
        --report research/evidence/derm7pt_v1/agent_eval_report_v4.json \\
        --out research/evidence/derm7pt_v1/deterministic_verification_v4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.verification import deterministic_checks
from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.evidence.contract import build_contract
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.perception.criterion_abstention import load_policy
from medguard_v2.pipeline.run_cohort import _load_features, fit_readers

SEED = 20260810
REPORT = REPO / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_v3.json"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "deterministic_verification.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT,
                        help="agent_eval_report_*.json to check (default: v3)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output path for deterministic_verification json")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["rows"]

    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    features = _load_features()
    diagnosis, heads = fit_readers(train, features)
    catalog = GuidelineCatalog.load()
    policy = load_policy()

    case_map = {f"derm7pt-{c.case_num}": c for c in valid}
    contract_cache: dict[str, object] = {}

    def get_contract(case_id: str):
        if case_id in contract_cache:
            return contract_cache[case_id]
        case = case_map[case_id]
        x = features[str(case.case_num)].reshape(1, -1)
        criteria = {}
        for name, head in heads.items():
            values, _ = policy.apply(head, x, name)
            criteria[name] = values[0]
        c = build_contract(
            case_id=case_id,
            classifier_diagnosis=str(diagnosis.predict(x)[0]),
            classifier_confidence=float(diagnosis.predict_proba(x).max()),
            criteria=criteria,
            catalog=catalog,
        )
        contract_cache[case_id] = c
        return c

    results = []
    issue_counter = Counter()
    answers_with_issues = 0
    total_answers = 0
    total_refusals = 0

    for row in rows:
        case_id = row["case_id"]
        answer_text = row.get("answer", "")
        answer_kind = row.get("answer_kind", "")

        if answer_kind == "refusal" or not answer_text:
            total_refusals += 1
            results.append({
                "case_id": case_id,
                "kind": row.get("kind", ""),
                "answer_kind": answer_kind,
                "skipped": True,
                "reason": "refusal or empty answer",
                "issues": [],
            })
            continue

        total_answers += 1
        contract = get_contract(case_id)
        issues = deterministic_checks(contract, answer_text, catalog)

        if issues:
            answers_with_issues += 1
            for issue in issues:
                issue_counter[issue.split(":")[0] if ":" in issue else issue[:60]] += 1

        results.append({
            "case_id": case_id,
            "kind": row.get("kind", ""),
            "question": row.get("question", "")[:100],
            "answer_kind": answer_kind,
            "skipped": False,
            "issues": issues,
            "n_issues": len(issues),
        })

    output = {
        "protocol": "deterministic_verification_v1",
        "source_report": str(args.report.relative_to(REPO)) if args.report.is_absolute() else str(args.report),
        "n_rows": len(rows),
        "n_answers_checked": total_answers,
        "n_refusals_skipped": total_refusals,
        "n_answers_with_issues": answers_with_issues,
        "issue_rate": round(answers_with_issues / total_answers, 4) if total_answers else 0,
        "issue_type_counts": dict(issue_counter),
        "total_issues_found": sum(issue_counter.values()),
        "per_answer": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  {total_answers} answers checked, {total_refusals} refusals skipped")
    print(f"  {answers_with_issues} answers with issues ({answers_with_issues/total_answers:.1%})")
    print(f"  Issue types: {dict(issue_counter)}")


if __name__ == "__main__":
    main()
