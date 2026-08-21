"""Combined re-run: agent eval (v3 prompt) → RAGAS faithfulness scoring.

The v3 prompt adds structured Diagnosis:/Assessment: sections to agent answers.
The RAGAS judge's _extract_scored_span() will detect the Assessment: section and
only score guideline-grounded claims, instead of scoring the whole answer (which
includes contract-derived facts that drag faithfulness down).

Expected improvement: mixed-grounding cases go from ~0.097 to ~0.85+ because
contract-derived statements are no longer scored against guideline contexts.

Cost: ~$2 total (agent calls + RAGAS judge calls).

v5.3 additions to ``run_ragas`` (both measurement fixes, not generation changes):
  - Contract-context: the Assessment-scoping trick above only fixes GUIDELINE-scoped answers.
    A 'mixed' (or 'contract', where already scored) row's Diagnosis section states real
    contract-derived facts -- score, criteria, classifier output -- that no retrieved guideline
    context can support, so judging it against guideline contexts alone still mismeasures it
    (v2's 0.0972 mixed faithfulness had exactly this cause). Those rows now also get the case's
    own compact decision-contract facts as one more context string. Guideline-only rows are
    untouched.
  - Median-of-3 judging: the judge is nondeterministic at temperature 0 (measured noise floor
    0.0489). Every sample is now judged N_JUDGE_ATTEMPTS (3) times, cached independently per
    attempt, and the reported faithfulness/answer_relevancy are the per-sample MEDIAN, with a
    judge_dispersion summary of how much the attempts disagreed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def run_agent_eval(model: str, limit: int, cohort: str, out: Path) -> dict:
    """Step 1: Run agent eval with the v3 prompt."""
    from medguard_v2.eval.agent_eval import CACHE_DIR, run

    from medguard import config
    from medguard.pipeline.llm import LLMClient

    if not config.load_env():
        raise SystemExit("No LLM credential available.")

    client = LLMClient(model=model)
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "calls": 0, "cache_hits": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
    }

    def chat(messages):
        key = hashlib.sha256(
            json.dumps(
                {"messages": messages, "model": model, "t": 0.0, "m": 700},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        path = cache_dir / key[:2] / f"{key}.json"
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
        path.write_text(
            json.dumps({"reply": reply, "usage": usage, "model": model}, indent=2),
            encoding="utf-8",
        )
        return reply

    print(f"=== Agent eval: model={model} cohort={cohort} limit={limit} ===")
    report = run(chat, limit=limit, model=model, cohort=cohort)
    report["usage"] = {
        **stats,
        "latency_s": round(stats["latency_s"], 1),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"  cases={report['n_cases']}  questions={report['n_questions']}")
    print(f"  usage: {stats['calls']} paid, {stats['cache_hits']} cached")
    for name, block in report["endpoints"].items():
        vals = {k: v for k, v in block.items() if k not in ("note", "exact")}
        print(f"  {name}: {vals}")
    print(f"  wrote {out}")
    return report


#: D: how many times each sample is judged. The reported faithfulness/answer_relevancy are the
#: per-sample MEDIAN across attempts, not any single call -- the judge is nondeterministic at
#: temperature 0 (measured noise floor 0.0489, research/evidence/derm7pt_v1 judge-noise note).
N_JUDGE_ATTEMPTS = 3


def _median(values: List[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _combine_judge_attempts(
    attempts: List[List[Optional[float]]],
) -> Tuple[List[Optional[float]], List[Optional[float]], List[List[Optional[float]]]]:
    """Transpose N_JUDGE_ATTEMPTS parallel per-sample score lists (one list per attempt, same
    row order as `scorable`) into per-sample (median, mean-absolute-deviation, raw-attempts).

    A sample where every attempt is None (the judge failed to parse on all three tries) stays
    None; a sample where only some attempts are None takes the median/MAD over whichever
    attempts did return a score.
    """
    n_samples = len(attempts[0]) if attempts else 0
    medians: List[Optional[float]] = []
    mads: List[Optional[float]] = []
    per_sample_attempts: List[List[Optional[float]]] = []
    for i in range(n_samples):
        values = [attempt[i] if i < len(attempt) else None for attempt in attempts]
        per_sample_attempts.append(values)
        present = [v for v in values if v is not None]
        if not present:
            medians.append(None)
            mads.append(None)
            continue
        med = _median(present)
        medians.append(round(med, 4))
        mads.append(round(sum(abs(v - med) for v in present) / len(present), 4))
    return medians, mads, per_sample_attempts


def _aggregate_faithfulness(per_sample: List[Optional[float]], per_scope: List[str]) -> Dict[str, Any]:
    """Re-derives RagasFaithfulnessJudge.evaluate()'s own faithfulness aggregation (mean,
    coverage, assessment_mean, ...) from an already-combined per-sample list -- the median-of-N
    values from _combine_judge_attempts -- instead of one call's raw per_sample. Same shape as
    the judge's own output, so every existing consumer of faithfulness_detail is unaffected by
    where the per-sample numbers came from.
    """
    from medguard.eval.ragas_judge import SCOPE_ASSESSMENT

    scored = [s for s in per_sample if s is not None]
    assessment_scored = [
        s for s, scope in zip(per_sample, per_scope)
        if s is not None and scope == SCOPE_ASSESSMENT
    ]
    answered_mean = round(sum(scored) / len(scored), 4) if scored else None
    overall_mean = round(sum(scored) / len(per_sample), 4) if per_sample else None
    assessment_mean = (
        round(sum(assessment_scored) / len(assessment_scored), 4) if assessment_scored else None
    )
    return {
        "mean": answered_mean,
        "answered_mean": answered_mean,
        "assessment_mean": assessment_mean,
        "n_assessment_scored": len(assessment_scored),
        "overall_mean": overall_mean,
        "coverage": round(len(scored) / len(per_sample), 4) if per_sample else None,
        "n_scored": len(scored),
        "n_samples": len(per_sample),
        "per_sample": per_sample,
        "per_sample_scope": per_scope,
    }


def _aggregate_metric(per_sample: List[Optional[float]]) -> Dict[str, Any]:
    """Same idea as _aggregate_faithfulness, mirroring RagasFaithfulnessJudge.evaluate()'s
    generic _metric_block (used for answer_relevancy: no scope breakdown)."""
    scored = [v for v in per_sample if v is not None]
    return {
        "mean": round(sum(scored) / len(scored), 4) if scored else None,
        "coverage": round(len(scored) / len(per_sample), 4) if per_sample else None,
        "n_scored": len(scored),
        "n_samples": len(per_sample),
        "per_sample": per_sample,
    }


def _dispersion_summary(
    mads: List[Optional[float]], attempts: List[List[Optional[float]]],
) -> Dict[str, Any]:
    """Judge-noise summary (mean/max absolute deviation of each attempt from that sample's own
    median) -- measures the judge's own inconsistency across N_JUDGE_ATTEMPTS repeats of the
    identical question, not model quality."""
    present = [m for m in mads if m is not None]
    return {
        "n_attempts": N_JUDGE_ATTEMPTS,
        "n_samples_with_dispersion": len(present),
        "mean_absolute_deviation": round(sum(present) / len(present), 4) if present else None,
        "max_absolute_deviation": round(max(present), 4) if present else None,
        "n_samples_all_attempts_agreed_exactly": sum(1 for m in present if m == 0.0),
        "note": (
            "Mean absolute deviation of each judge attempt from that sample's own median, "
            "averaged over samples with >=2 non-null attempts. Quantifies judge noise (measured "
            "single-call noise floor: 0.0489), not model quality."
        ),
    }


def _contract_context_string(contract: Any) -> str:
    """Compact decision-contract facts for the RAGAS judge (C).

    A 'mixed' or 'contract' grounded answer's Diagnosis section states system-computed facts
    (score, criteria, classifier output, conflict, implied action) that no retrieved guideline
    context can support -- judging faithfulness against guideline contexts alone is what made
    v2's mixed faithfulness read 0.0972. Deliberately NOT the full contract_to_dict() blob (that
    also carries verbatim guideline recommendation text and a receipt block -- redundant here,
    and irrelevant to what a Diagnosis section draws on): just the fields the spec names --
    criteria table, score range and threshold, classifier diagnosis+confidence, conflict
    description, implied management action.
    """
    from medguard_v2.evidence.contract import contract_to_dict

    data = contract_to_dict(contract)
    payload = {
        "source": "decision_contract (system-computed, not guideline text)",
        "case_id": data["case_id"],
        "classifier_diagnosis": data["classifier"]["diagnosis"],
        "classifier_confidence": data["classifier"]["confidence"],
        "criteria": {c["name"]: c["value"] for c in data["criteria"]},
        "seven_point_score_min": data["seven_point_score"]["min"],
        "seven_point_score_max": data["seven_point_score"]["max"],
        "melanoma_suspicion_threshold": data["seven_point_score"]["threshold"],
        "conflict_description": data["conflict"]["description"],
        "management_action": data["criteria_implied_action"],
    }
    return json.dumps(payload, sort_keys=True)


def _load_contracts_by_case_id(agent_report: dict) -> Dict[str, Any]:
    """Rebuild the exact DecisionContract per case_id that produced this agent report -- same
    features, same fitted heads (fixed SEED/C, deterministic), same cohort/limit -- the same way
    medguard_v2.eval.agent_eval.run() built them originally. Needed so C can hand the judge the
    real contract facts rather than only what happens to be re-derivable from the report's rows.
    """
    from medguard_v2.eval.agent_eval import _build_contracts

    n_cases = agent_report.get("n_cases", 0)
    cohort = agent_report.get("cohort", "all")
    # Match the perception the answers were generated on: a promoted-perception agent report must
    # be judged against the PROMOTED contracts, or the judge is handed base-perception facts
    # (different diagnosis/score/conflict) and flags the answer's correct, promoted-derived
    # statements as unfaithful -- which is what depressed the v5.4 mixed/conflict faithfulness.
    promoted = str(agent_report.get("perception", "")).startswith("promoted")
    if not n_cases:
        return {}
    contracts = _build_contracts(n_cases, cohort, promoted=promoted)
    return {c.case_id: c for c in contracts}


def run_ragas(agent_results_path: Path, out: Path, model: str) -> dict:
    """Step 2: Score agent answers for RAGAS faithfulness.

    C: rows grounded 'mixed' or 'contract' that are ALREADY going to be judged (non-empty
    guideline contexts) additionally get the case's own compact decision-contract facts as one
    more context string, so a faithful contract-derived statement is no longer judged unfaithful
    for lacking a guideline citation it was never supposed to have. This does not change which
    rows enter `scorable` (a row with zero guideline contexts stays un-scored, verified exactly
    by the host instead); only what the ones already scored are judged against. Guideline-only
    rows are unchanged.
    D: every sample is judged N_JUDGE_ATTEMPTS times (the judge is nondeterministic at temp 0;
    measured noise floor 0.0489); the reported faithfulness/answer_relevancy are the per-sample
    MEDIAN across attempts, with a judge_dispersion summary of how much attempts disagreed.
    """
    from medguard import config
    from medguard.pipeline.llm import LLMClient
    from medguard.eval.ragas_judge import (
        JudgeCache,
        RagasFaithfulnessJudge,
    )
    from medguard_v2.evidence.guideline_binding import (
        GROUP_TO_CONDITION,
        GuidelineCatalog,
    )

    if not config.load_env():
        raise SystemExit("No LLM credential available.")

    data = json.loads(agent_results_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    catalog = GuidelineCatalog.load()
    print(f"\n=== RAGAS scoring: {len(rows)} rows ===")

    contracts_by_case = _load_contracts_by_case_id(data)
    print(f"  rebuilt {len(contracts_by_case)} case contracts for contract-context enrichment "
          f"(cohort={data.get('cohort')!r}, n_cases={data.get('n_cases')})")

    def reconstruct_contexts(tool_calls):
        contexts = []
        seen = set()
        for tc in tool_calls:
            if tc.get("tool") != "search_guidelines":
                continue
            condition = tc.get("condition", "")
            qt = tc.get("question_type", "")
            key = (condition, qt)
            if key in seen:
                continue
            seen.add(key)
            if condition not in GROUP_TO_CONDITION:
                continue
            try:
                result = catalog.bind(condition, question_types=(qt,)) if qt else catalog.bind(condition)
            except KeyError:
                continue
            for rec in result.recommendations:
                if rec.text not in contexts:
                    contexts.append(rec.text)
        return contexts

    scorable = []
    grounding_counts = {"guideline": 0, "contract": 0, "mixed": 0}
    contract_context_added_n = 0
    contract_lookup_misses = 0
    for row in rows:
        answer = row.get("answer", "")
        if not answer or not answer.strip():
            continue
        kind = row.get("kind", "")
        if kind == "out_of_scope":
            continue
        tool_calls = row.get("tool_calls", [])
        contexts = reconstruct_contexts(tool_calls)
        has_guidelines = len(contexts) > 0
        if kind == "guideline" and has_guidelines:
            grounding = "guideline"
        elif kind == "override":
            grounding = "contract"
        elif kind == "why":
            grounding = "mixed" if has_guidelines else "contract"
        else:
            grounding = "guideline" if has_guidelines else "contract"
        grounding_counts[grounding] = grounding_counts.get(grounding, 0) + 1

        # C: only rows ALREADY headed for the judge (non-empty guideline contexts) and not pure
        # guideline grounding get the contract-facts context appended. A row with zero guideline
        # contexts (the common override case) stays un-added and un-scored -- see the docstring.
        contract_context_added = False
        if contexts and grounding in ("mixed", "contract"):
            case_contract = contracts_by_case.get(row.get("case_id", ""))
            if case_contract is not None:
                contexts = contexts + [_contract_context_string(case_contract)]
                contract_context_added = True
                contract_context_added_n += 1
            else:
                contract_lookup_misses += 1

        if not contexts:
            continue
        scorable.append({
            "case_id": row["case_id"],
            "kind": kind,
            "question": row["question"],
            "answer": answer,
            "contexts": contexts,
            "grounding": grounding,
            "contract_context_added": contract_context_added,
        })

    n_out_of_scope = sum(1 for r in rows if r.get("kind") == "out_of_scope")
    n_contract = grounding_counts.get("contract", 0)
    n_contract_scored = sum(1 for s in scorable if s["grounding"] == "contract")
    print(f"  scorable: {len(scorable)} (guideline+mixed+contract-with-context)")
    print(f"  contract-only, unscored: {n_contract - n_contract_scored} (verified exactly, not by judge)")
    print(f"  grounding: {grounding_counts}")
    print(f"  contract-context added to {contract_context_added_n} rows "
          f"({contract_lookup_misses} lookup misses)")

    cache_dir = REPO / "research/evidence/derm7pt_v1/ragas_judge_cache"
    session_id = f"v3_ragas_{int(time.time())}"
    cache = JudgeCache(cache_dir, session_id)

    client = LLMClient(model=config.JUDGE_MODEL)
    # D: seed is already part of JudgeCache._key, so a distinct judge per attempt (seeded
    # 0/1/2) buckets each attempt's calls into its own cache entry with no change to
    # ragas_judge.py. Attempt 0 (seed=0) is the exact construction every prior single-shot run
    # used, so existing cache entries are hit and reused as attempt 0 at zero new cost.
    judges = [
        RagasFaithfulnessJudge(client, model=config.JUDGE_MODEL, max_workers=2, cache=cache, seed=attempt)
        for attempt in range(N_JUDGE_ATTEMPTS)
    ]

    ragas_rows = [
        {
            "user_input": s["question"],
            "response": s["answer"],
            "retrieved_contexts": s["contexts"],
        }
        for s in scorable
    ]

    print(f"  judging {len(ragas_rows)} samples x {N_JUDGE_ATTEMPTS} attempts with {config.JUDGE_MODEL}...")
    attempt_results = [
        judge.evaluate(ragas_rows, metrics=["faithfulness", "answer_relevancy"])
        for judge in judges
    ]

    faith_attempts = [r.get("faithfulness", {}).get("per_sample", []) for r in attempt_results]
    rel_attempts = [r.get("answer_relevancy", {}).get("per_sample", []) for r in attempt_results]
    faith_scopes = (
        attempt_results[0].get("faithfulness", {}).get("per_sample_scope", [])
        if attempt_results else []
    )

    faith_median, faith_mad, faith_attempts_per_sample = _combine_judge_attempts(faith_attempts)
    rel_median, rel_mad, rel_attempts_per_sample = _combine_judge_attempts(rel_attempts)

    faith_block = _aggregate_faithfulness(faith_median, faith_scopes)
    rel_block = _aggregate_metric(rel_median)
    faith_per = faith_block["per_sample"]
    rel_per = rel_block["per_sample"]

    output = {
        "protocol": "E5c_ragas_agent_metrics_contract_context_median3",
        "agent_results_file": str(agent_results_path.name),
        "prompt_version": data.get("prompt_version", "unknown"),
        "n_judge_attempts": N_JUDGE_ATTEMPTS,
        "n_rows_total": len(rows),
        "n_out_of_scope": n_out_of_scope,
        "n_contract_grounded": n_contract,
        "n_guideline_scorable": len(scorable),
        "n_scored_faithfulness": len([s for s in faith_per if s is not None]),
        "n_scored_relevancy": len([s for s in rel_per if s is not None]),
        "faithfulness": faith_block.get("mean"),
        "faithfulness_detail": faith_block,
        "answer_relevancy": rel_block.get("mean"),
        "answer_relevancy_detail": {
            k: v for k, v in rel_block.items() if k != "per_sample"
        },
        "grounding_analysis": {
            "guideline_grounded": sum(1 for s in scorable if s["grounding"] == "guideline"),
            "mixed_grounded": sum(1 for s in scorable if s["grounding"] == "mixed"),
            "contract_only": n_contract,
            "contract_context_added": contract_context_added_n,
            "contract_context_lookup_misses": contract_lookup_misses,
        },
        "judge_model": config.JUDGE_MODEL,
        "per_sample": [
            {
                "case_id": scorable[i]["case_id"],
                "kind": scorable[i]["kind"],
                "grounding": scorable[i]["grounding"],
                "contract_context_added": scorable[i]["contract_context_added"],
                "faithfulness": faith_per[i] if i < len(faith_per) else None,
                "faithfulness_attempts": (
                    faith_attempts_per_sample[i] if i < len(faith_attempts_per_sample) else []
                ),
                "faithfulness_mad": faith_mad[i] if i < len(faith_mad) else None,
                "scope": faith_scopes[i] if i < len(faith_scopes) else None,
                "answer_relevancy": rel_per[i] if i < len(rel_per) else None,
                "answer_relevancy_attempts": (
                    rel_attempts_per_sample[i] if i < len(rel_attempts_per_sample) else []
                ),
                "answer_relevancy_mad": rel_mad[i] if i < len(rel_mad) else None,
                "n_contexts": len(scorable[i]["contexts"]),
            }
            for i in range(len(scorable))
        ],
        "faithfulness_by_grounding": {},
        "judge_dispersion": {
            "faithfulness": _dispersion_summary(faith_mad, faith_attempts_per_sample),
            "answer_relevancy": _dispersion_summary(rel_mad, rel_attempts_per_sample),
        },
        "cache_stats": {
            "hits": cache.hits,
            "misses": cache.misses,
            "session_id": session_id,
        },
    }

    # "contract" added alongside the pre-existing guideline/mixed breakdown: C's fix applies to
    # already-scored contract-grounding rows too (an override conversation that also happened to
    # search), and this is the only place their aggregate becomes visible rather than only
    # appearing individually in per_sample. Purely additive -- existing readers of
    # faithfulness_by_grounding["guideline"/"mixed"] are unaffected.
    for gtype in ("guideline", "mixed", "contract"):
        scores = [
            faith_per[i]
            for i in range(len(scorable))
            if scorable[i]["grounding"] == gtype
            and i < len(faith_per) and faith_per[i] is not None
        ]
        if scores:
            output["faithfulness_by_grounding"][gtype] = {
                "mean": round(sum(scores) / len(scores), 4),
                "n": len(scores),
            }

    # Scope analysis: how many answers used Assessment: sections?
    scope_counts = {}
    for s in faith_scopes:
        scope_counts[s] = scope_counts.get(s, 0) + 1
    output["scope_distribution"] = scope_counts

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"\n  === RAGAS Results (median of {N_JUDGE_ATTEMPTS} judge attempts) ===")
    print(f"  faithfulness: {output['faithfulness']}")
    print(f"  answer_relevancy: {output['answer_relevancy']}")
    print(f"  scope distribution: {scope_counts}")
    print(f"  by grounding: {output['faithfulness_by_grounding']}")
    print(f"  judge dispersion (faithfulness): {output['judge_dispersion']['faithfulness']}")
    print(f"  judge calls: {cache.hits} cached + {cache.misses} fresh")
    print(f"  wrote {out}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    parser.add_argument("--limit", type=int, default=101)
    parser.add_argument("--cohort", default="report")
    args = parser.parse_args()

    agent_out = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v3.json"
    ragas_out = REPO / "research/evidence/derm7pt_v1/agent_ragas_metrics_v3.json"

    report = run_agent_eval(args.model, args.limit, args.cohort, agent_out)
    ragas = run_ragas(agent_out, ragas_out, args.model)

    # Summary comparison with v2
    v2_path = REPO / "research/evidence/derm7pt_v1/agent_ragas_metrics.json"
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        print(f"\n  === v2 → v3 comparison ===")
        print(f"  faithfulness: {v2.get('faithfulness')} → {ragas.get('faithfulness')}")
        print(f"  answer_relevancy: {v2.get('answer_relevancy')} → {ragas.get('answer_relevancy')}")
        v2_by_g = v2.get("faithfulness_by_grounding", {})
        v3_by_g = ragas.get("faithfulness_by_grounding", {})
        for g in ("guideline", "mixed"):
            v2_val = v2_by_g.get(g, {}).get("mean", "N/A")
            v3_val = v3_by_g.get(g, {}).get("mean", "N/A")
            print(f"  {g}: {v2_val} → {v3_val}")


if __name__ == "__main__":
    main()
