"""Frozen-configuration agent evaluation on the 395 Derm7pt TEST lesions (reviewer ask C2).

Configuration is frozen at the shipped system: PROMPT_VERSION as imported from
medguard_v2.agent.clinician_agent, qwen/qwen3-vl-32b-instruct at temperature 0, 700 tokens,
gated retrieval, deterministic checks, the same question templates and endpoint scorers as
agent_eval_report_v5_4.json. Nothing is tuned on this run and nothing may be re-run after it.

Test governance: the promoted perception contracts for the test split are built once via
run_cohort('test', allow_test=True, promoted=True), written to
research/evidence/derm7pt_v1/cdss_contracts_test_promoted.json, and the read is appended to the
hash-chained access log research/V2_TEST_ACCESS_LOG.jsonl. If the contracts file already exists
it is reloaded; the test split is not re-read.

Usage:
    python scripts/dev/run_agent_eval_test.py --fake            # wiring check, no paid calls
    python scripts/dev/run_agent_eval_test.py --write           # the real one-shot (needs MEDGUARD_API_KEY)
    python scripts/dev/run_agent_eval_test.py --limit 5         # paid smoke on 5 cases (no --write)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.clinician_agent import PROMPT_VERSION, run_agent  # noqa: E402
from medguard_v2.eval.agent_eval import (  # noqa: E402
    CACHE_DIR,
    _build_report,
    build_questions,
    score_citations,
    score_counterfactual,
    score_fabrication,
    score_refusal,
)
from medguard_v2.eval.confirmatory import _append_log, verify_log  # noqa: E402
from medguard_v2.evidence.contract import build_contract  # noqa: E402
from medguard_v2.evidence.guideline_binding import GuidelineCatalog  # noqa: E402

MODEL = "qwen/qwen3-vl-32b-instruct"
CONTRACTS_OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "cdss_contracts_test_promoted.json"
REPORT_OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_test_v1.json"


def load_or_build_test_contracts() -> List[Dict]:
    if CONTRACTS_OUT.exists():
        raw = json.loads(CONTRACTS_OUT.read_text(encoding="utf-8"))
        entries = next(v for v in raw.values() if isinstance(v, list)) if isinstance(raw, dict) else raw
        print(f"reusing {CONTRACTS_OUT.name} ({len(entries)} cases); test split NOT re-read")
        return entries

    from medguard_v2.pipeline.run_cohort import run_cohort

    print("building promoted test contracts (single governed read of the test split)...")
    report = run_cohort("test", allow_test=True, promoted=True)
    CONTRACTS_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    entries = next(v for v in report.values() if isinstance(v, list))
    _append_log(
        "agent_confirmatory_test_run",
        "Reviewer-requested frozen-configuration agent evaluation on the 395 test lesions: promoted "
        "perception contracts built once via run_cohort('test', promoted=True); agent prompts, model, "
        "scorers, and checks identical to the v5.4 validation evaluation; nothing tuned on this run.",
        {"n_cases": len(entries), "prompt_version": PROMPT_VERSION, "model": MODEL,
         "artifact": str(CONTRACTS_OUT.relative_to(REPO)).replace("\\", "/")},
    )
    print("access log:", verify_log())
    return entries


def entries_to_contracts(entries: List[Dict], catalog: GuidelineCatalog):
    out = []
    for e in entries:
        criteria = {c["name"]: c["value"] for c in e["criteria"]}
        out.append(build_contract(
            case_id=e["case_id"],
            classifier_diagnosis=str(e["classifier"]["diagnosis"]),
            classifier_confidence=float(e["classifier"]["confidence"]),
            criteria=criteria,
            catalog=catalog,
            criterion_margins=e.get("criterion_margins"),
        ))
    return out


def make_chat(fake: bool, stats: Dict):
    if fake:
        def chat(messages):
            stats["calls"] += 0
            return json.dumps({"action": "answer", "text": "Diagnosis: wiring check only.",
                               "citations": []})
        return chat

    from medguard import config
    from medguard.pipeline.llm import LLMClient

    if not config.load_env():
        raise SystemExit("No LLM credential available (set MEDGUARD_API_KEY).")
    client = LLMClient(model=MODEL)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def chat(messages):
        key = hashlib.sha256(json.dumps(
            {"messages": messages, "model": MODEL, "t": 0.0, "m": 700}, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        path = CACHE_DIR / key[:2] / f"{key}.json"
        if path.exists():
            stats["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))["reply"]
        started = time.time()
        response = client.chat(messages, temperature=0.0, max_tokens=700)
        reply = response["content"] if isinstance(response, dict) else str(response)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        stats["calls"] += 1
        stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        stats["latency_s"] += time.time() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"reply": reply, "usage": usage, "model": MODEL}, indent=2),
                        encoding="utf-8")
        return reply

    return chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="wiring check; no credentials, no paid calls")
    parser.add_argument("--limit", type=int, default=0, help="first N cases only (paid smoke)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    catalog = GuidelineCatalog.load()
    if args.fake:
        # Wiring check must not touch the test split: reuse the existing VALID contracts artifact.
        from medguard_v2.eval.agent_eval import PROMOTED_CONTRACTS
        raw = json.loads(PROMOTED_CONTRACTS.read_text(encoding="utf-8"))
        entries = (next(v for v in raw.values() if isinstance(v, list))
                   if isinstance(raw, dict) else raw)[:3]
        print("FAKE mode: wiring against 3 VALID contracts; test split untouched")
    else:
        entries = load_or_build_test_contracts()
    contracts = entries_to_contracts(entries, catalog)
    if args.limit:
        contracts = contracts[: args.limit]
    print(f"{len(contracts)} test contracts; prompt version {PROMPT_VERSION}")

    stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
    chat = make_chat(args.fake, stats)

    rows: List[Dict] = []
    for k, contract in enumerate(contracts, 1):
        for question in build_questions(contract):
            answer = run_agent(contract=contract, catalog=catalog, question=question.text,
                               chat=chat, model=MODEL, llm_verifier=False, gated_retrieval=True)
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
        if k % 25 == 0:
            print(f"  {k}/{len(contracts)} cases ({stats['calls']} paid, {stats['cache_hits']} cached)")

    report = _build_report(rows, n_cases=len(contracts), model=MODEL if not args.fake else "FAKE",
                           cohort="test-frozen", llm_verifier=False, gated_retrieval=True, promoted=True)
    report["protocol"] = "medguard-v2-agent-confirmatory-test-v1"
    report["prompt_version"] = PROMPT_VERSION
    report["usage"] = {**stats, "latency_s": round(stats["latency_s"], 1)}
    report["note"] = ("One-shot frozen-configuration agent evaluation on the Derm7pt test split; "
                      "questions, scorers, prompts, model, and checks identical to the v5.4 "
                      "validation evaluation. Nothing was tuned on this run.")

    print(f"\nquestions: {report['n_questions']}")
    for name, block in report["endpoints"].items():
        vals = {k2: v for k2, v in block.items() if k2 not in ("note", "exact")}
        print(f"  {name}: {vals}")
    print(f"usage: {stats['calls']} paid calls, {stats['cache_hits']} cached")

    if args.write and not args.fake:
        REPORT_OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("wrote", REPORT_OUT)
    elif args.fake:
        print("(fake mode: nothing written)")


if __name__ == "__main__":
    main()
