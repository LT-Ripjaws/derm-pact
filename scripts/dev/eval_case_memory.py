"""A/B evaluation: clinician agent with and without case memory on dev split.

Builds a case memory from all Derm7pt train contracts (413 cases), then runs
the agent eval on the dev cohort twice: once without case memory (baseline,
should reproduce existing numbers from cache) and once with case memory enabled.

Compares all endpoints side by side. The case memory does NOT change the
deterministic pipeline — it only gives the agent context about similar past
evaluations when it chooses to call retrieve_similar_cases().

Cost: ~$1-2 for the case-memory arm (baseline arm is fully cached).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.case_memory import CaseMemory, build_memory_from_contracts
from medguard_v2.agent.clinician_agent import run_agent
from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.agent_eval import (
    CACHE_DIR,
    SEED,
    BASE_C,
    Question,
    agent_cohort_indices,
    build_questions,
    score_citations,
    score_counterfactual,
    score_fabrication,
    score_refusal,
)
from medguard_v2.evidence.contract import DecisionContract, build_contract
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.perception.criterion_abstention import load_policy

FEATURES_PATH = REPO / "research/evidence/derm7pt_v1/features/dermlip/features.npz"
OUT = REPO / "research/evidence/derm7pt_v1/case_memory_ab_eval.json"


def _load_features():
    data = np.load(FEATURES_PATH)
    return {str(c): v for c, v in zip(data["case_num"], data["features"])}, data


def _fit_heads_and_classifier(train, features_dict):
    from sklearn.linear_model import LogisticRegression
    x = np.stack([features_dict[str(c.case_num)] for c in train])
    diagnosis = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
    diagnosis.fit(x, [c.diagnosis_group for c in train])
    policy = load_policy()
    heads = {}
    for name in CRITERION_NAMES:
        head = LogisticRegression(max_iter=5000, C=BASE_C, random_state=SEED)
        head.fit(x, [str(c.criteria[name]) for c in train])
        heads[name] = head
    return diagnosis, heads, policy


def _build_contracts_and_memory(train, valid, features_dict, diagnosis, heads, policy, catalog):
    """Build contracts for train (memory) and dev-valid (evaluation)."""

    def _predict_criteria(x_batch):
        criteria_list = [{} for _ in range(len(x_batch))]
        for name in CRITERION_NAMES:
            values, _ = policy.apply(heads[name], x_batch, name)
            for i, v in enumerate(values):
                criteria_list[i][name] = v
        return criteria_list

    # Train contracts -> case memory
    x_train = np.stack([features_dict[str(c.case_num)] for c in train])
    train_criteria = _predict_criteria(x_train)
    train_posterior = diagnosis.predict_proba(x_train)
    train_predicted = diagnosis.predict(x_train)

    train_contracts = []
    for i, case in enumerate(train):
        train_contracts.append(build_contract(
            case_id=f"derm7pt-{case.case_num}",
            classifier_diagnosis=str(train_predicted[i]),
            classifier_confidence=float(train_posterior[i].max()),
            criteria=train_criteria[i],
            catalog=catalog,
        ))

    memory = build_memory_from_contracts(train_contracts, features_dict)

    # Dev-valid contracts
    allowed = agent_cohort_indices(len(valid), "dev")
    order = sorted(allowed, key=lambda i: (i * 7919) % len(valid))

    x_valid = np.stack([features_dict[str(c.case_num)] for c in valid])
    valid_criteria = _predict_criteria(x_valid)
    valid_posterior = diagnosis.predict_proba(x_valid)
    valid_predicted = diagnosis.predict(x_valid)

    dev_contracts = []
    dev_embeddings = []
    for i in order:
        dev_contracts.append(build_contract(
            case_id=f"derm7pt-{valid[i].case_num}",
            classifier_diagnosis=str(valid_predicted[i]),
            classifier_confidence=float(valid_posterior[i].max()),
            criteria=valid_criteria[i],
            catalog=catalog,
        ))
        dev_embeddings.append(features_dict[str(valid[i].case_num)])

    return memory, dev_contracts, dev_embeddings


def _run_arm(
    arm_name: str,
    contracts: List[DecisionContract],
    embeddings: list,
    catalog: GuidelineCatalog,
    chat,
    model: str,
    memory: CaseMemory = None,
    limit: int = None,
):
    if limit:
        contracts = contracts[:limit]
        embeddings = embeddings[:limit]

    rows = []
    for idx, contract in enumerate(contracts):
        emb = embeddings[idx] if embeddings else None
        for question in build_questions(contract):
            answer = run_agent(
                contract=contract,
                catalog=catalog,
                question=question.text,
                chat=chat,
                model=model,
                case_memory=memory,
                query_embedding=emb,
            )
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
                "fabrication": score_fabrication(answer),
                "used_case_memory": any(
                    c.get("tool") == "retrieve_similar_cases" for c in answer.tool_calls
                ),
            })

    return _summarize(arm_name, rows)


def _summarize(arm_name: str, rows: List[Dict]) -> Dict:
    def _rate(pred):
        vals = [r for r in rows if pred(r) is not None]
        return round(sum(1 for r in vals if pred(r)) / len(vals), 4) if vals else None

    cf = [r for r in rows if r["counterfactual"] is not None]
    cited_rows = [r for r in rows if r["citations"]["cited"] > 0]
    memory_used = sum(1 for r in rows if r.get("used_case_memory", False))

    return {
        "arm": arm_name,
        "n_cases": len(set(r["case_id"] for r in rows)),
        "n_questions": len(rows),
        "memory_calls": memory_used,
        "memory_call_rate": round(memory_used / len(rows), 4) if rows else 0,
        "endpoints": {
            "citation_resolvability": {
                "answers_with_citations": len(cited_rows),
                "resolvability": round(
                    sum(r["citations"]["resolved"] for r in cited_rows)
                    / max(1, sum(r["citations"]["cited"] for r in cited_rows)), 4
                ) if cited_rows else None,
                "all_intact": all(
                    r["citations"]["resolved"] == r["citations"]["intact"] for r in cited_rows
                ),
            },
            "refusal_correctness": {
                "overall": _rate(lambda r: r["refusal"]["correct"]),
                "on_answerable": round(
                    sum(1 for r in rows if r["refusal"]["answerable"] and r["refusal"]["correct"])
                    / max(1, sum(1 for r in rows if r["refusal"]["answerable"])), 4),
                "on_unanswerable": round(
                    sum(1 for r in rows if not r["refusal"]["answerable"] and r["refusal"]["correct"])
                    / max(1, sum(1 for r in rows if not r["refusal"]["answerable"])), 4),
            },
            "counterfactual_accuracy": {
                "n": len(cf),
                "override_applied": round(
                    sum(1 for r in cf if r["counterfactual"]["override_applied"]) / max(1, len(cf)), 4),
                "states_correct_score": round(
                    sum(1 for r in cf if r["counterfactual"]["states_correct_score"]) / max(1, len(cf)), 4),
                "correct": round(
                    sum(1 for r in cf if r["counterfactual"]["correct"]) / max(1, len(cf)), 4),
            },
            "management_fabrication": {
                "rate": _rate(lambda r: r["fabrication"]["fabricated_management"]),
            },
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit dev cases (default: all dev cases)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from medguard import config
    from medguard.pipeline.llm import LLMClient

    if not config.load_env():
        raise SystemExit("No LLM credential available.")

    client = LLMClient(model=args.model)
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    stats = {"baseline": {"calls": 0, "cache_hits": 0, "latency_s": 0.0},
             "with_memory": {"calls": 0, "cache_hits": 0, "latency_s": 0.0}}

    def make_chat(arm_stats):
        def chat(messages):
            key = hashlib.sha256(
                json.dumps(
                    {"messages": messages, "model": args.model,
                     "t": args.temperature, "m": args.max_tokens},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            path = cache_dir / key[:2] / f"{key}.json"
            if path.exists():
                arm_stats["cache_hits"] += 1
                return json.loads(path.read_text(encoding="utf-8"))["reply"]

            started = time.time()
            response = client.chat(messages, temperature=args.temperature,
                                   max_tokens=args.max_tokens)
            reply = response["content"] if isinstance(response, dict) else str(response)
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            arm_stats["calls"] += 1
            arm_stats["latency_s"] += time.time() - started

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"reply": reply, "usage": usage, "model": args.model}, indent=2),
                encoding="utf-8",
            )
            return reply
        return chat

    # Load data
    print("Loading features and cases...")
    features_dict, _ = _load_features()
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    catalog = GuidelineCatalog.load()

    print(f"Fitting classifier + {len(CRITERION_NAMES)} criterion heads on {len(train)} train cases...")
    diagnosis, heads, policy = _fit_heads_and_classifier(train, features_dict)

    print("Building case memory from train contracts...")
    memory, dev_contracts, dev_embeddings = _build_contracts_and_memory(
        train, valid, features_dict, diagnosis, heads, policy, catalog
    )
    print(f"Case memory: {len(memory)} entries. Dev contracts: {len(dev_contracts)} cases.")

    # Arm 1: baseline (no case memory) — should be fully cached
    print("\n=== ARM 1: Baseline (no case memory) ===")
    baseline = _run_arm(
        "baseline", dev_contracts, dev_embeddings, catalog,
        make_chat(stats["baseline"]), args.model,
        memory=None, limit=args.limit,
    )
    baseline["usage"] = stats["baseline"]
    _print_endpoints("Baseline", baseline)

    # Arm 2: with case memory — will have new cache misses (different system prompt)
    print("\n=== ARM 2: With case memory ===")
    with_memory = _run_arm(
        "with_memory", dev_contracts, dev_embeddings, catalog,
        make_chat(stats["with_memory"]), args.model,
        memory=memory, limit=args.limit,
    )
    with_memory["usage"] = stats["with_memory"]
    _print_endpoints("With memory", with_memory)

    # Comparison
    print("\n=== COMPARISON ===")
    for ep_name in baseline["endpoints"]:
        b = baseline["endpoints"][ep_name]
        m = with_memory["endpoints"][ep_name]
        for key in b:
            bv = b[key]
            mv = m[key]
            if isinstance(bv, (int, float)) and isinstance(mv, (int, float)):
                delta = mv - bv
                marker = "  " if abs(delta) < 0.001 else (" +" if delta > 0 else " ")
                print(f"  {ep_name}.{key}: baseline={bv}  memory={mv}  delta={marker}{delta:.4f}")

    print(f"\nMemory usage: {with_memory['memory_calls']}/{with_memory['n_questions']} "
          f"questions used retrieve_similar_cases ({with_memory['memory_call_rate']:.1%})")

    # Strip per-row data for the summary (keep it in the full output)
    output = {
        "protocol": "case_memory_ab_eval",
        "model": args.model,
        "n_memory_entries": len(memory),
        "baseline": {k: v for k, v in baseline.items() if k != "rows"},
        "with_memory": {k: v for k, v in with_memory.items() if k != "rows"},
        "baseline_rows": baseline["rows"],
        "with_memory_rows": with_memory["rows"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")


def _print_endpoints(label, arm):
    print(f"\n{label} ({arm['n_cases']} cases, {arm['n_questions']} questions):")
    for name, block in arm["endpoints"].items():
        vals = "  ".join(f"{k}={v}" for k, v in block.items())
        print(f"  {name}: {vals}")
    u = arm.get("usage", {})
    print(f"  usage: {u.get('calls', 0)} paid, {u.get('cache_hits', 0)} cached")


if __name__ == "__main__":
    main()
