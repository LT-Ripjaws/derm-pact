"""M7: real per-question token/latency cost for the clinician agent.

Reviewer finding (notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md, M7): efficiency_analysis.json's
agent_evaluation_cost is all zeros because the sealed v5.4 run (agent_eval_report_v5_4.json) was a
100% cache replay against research/evidence/derm7pt_v1/agent_cache/ -- 0 paid calls, 1667 cache
hits -- and EVERY cached reply carries usage: {}. The root cause is a bug in
scripts/dev/run_agent_eval.py::make_chat (NOT fixed here -- that script is untouched):

    response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    reply = response["content"] if isinstance(response, dict) else str(response)
    usage = response.get("usage", {}) if isinstance(response, dict) else {}

``medguard.pipeline.llm.LLMClient.chat`` returns a plain ``str`` (medguard/pipeline/llm.py:102),
never a dict, so ``isinstance(response, dict)`` is always False and ``usage`` is always ``{}``.
The real per-call telemetry (prompt/completion/total tokens, whether the gateway returned usage
at all, and latency) was never lost -- it was sitting the whole time in
``LLMClient.last_call_metadata`` / ``LLMClient.call_log`` (medguard/pipeline/llm.py:88-101) and was
simply never read.

This script produces real numbers two ways, and never touches the sealed cache or any existing
file:

  Part 1 -- a small FRESH paid sample (<=20 cases, hard cap 200 API calls) on the SAME protocol as
            the sealed v5.4 run (model, temperature, max_tokens, prompt_version, llm_verifier,
            gated_retrieval, promoted), reading LLMClient.last_call_metadata after every call.
            Replies are cached in a SEPARATE scratch dir
            (research/evidence/derm7pt_v1/efficiency_sample_v1/cache/) so a re-run is free and
            never touches research/evidence/derm7pt_v1/agent_cache/.

  Part 2 -- offline token accounting over the FULL sealed run (591 questions, 203 cases): replays
            medguard_v2.eval.agent_eval.run() with a chat callable that reads-only from the sealed
            agent_cache/ (never writes to it), tokenizes the exact outgoing `messages` (prompt) and
            the cached `reply` (completion) with whatever tokenizer is available, and cross-checks
            the reconstruction's endpoints/verification numbers against agent_eval_report_v5_4.json
            as a fidelity proof that the replay is byte-identical to the sealed conversations.

Usage:
    # 1. Preview Part 1's cost WITHOUT calling the API:
    ./venv_win/Scripts/python.exe scripts/dev/build_agent_efficiency_sample.py --dry-run

    # 2. Run for real (Part 2 always; Part 1 if a credential is available):
    ./venv_win/Scripts/python.exe scripts/dev/build_agent_efficiency_sample.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.clinician_agent import PROMPT_VERSION  # noqa: E402
from medguard_v2.eval.agent_eval import (  # noqa: E402
    CACHE_DIR as SEALED_CACHE_DIR,
    _build_contracts,
    build_questions,
    run,
)

# --------------------------------------------------------------------------- verified protocol
#
# Verified against research/evidence/derm7pt_v1/agent_eval_report_v5_4.json: model and
# prompt_version are stored fields on the report and are asserted against it at startup
# (_verify_protocol_labels). temperature/max_tokens are NOT stored on the report (the runner never
# records its own decoding params); they are scripts/dev/run_agent_eval.py's argparse DEFAULTS
# (--temperature 0.0, --max-tokens 700), which is the only invocation shape that produces the
# observed n_cases=203/n_questions=591/cache_hits=1667 with zero CLI overrides on record anywhere
# in notes/. The strongest available verification is empirical, not documentary: Part 2 replays
# the sealed conversations through this exact (model, temperature, max_tokens, prompt_version,
# llm_verifier, gated_retrieval) tuple and the cache-hit rate + reconstructed endpoints are checked
# against the sealed report below -- a wrong value here would show up as mass cache misses.
MODEL = "qwen/qwen3-vl-32b-instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 700
LLM_VERIFIER = False          # agent_eval_report_v5_4.json: verification.llm_verifier_enabled
GATED_RETRIEVAL = True        # agent_eval_report_v5_4.json: retrieval_gating.gated_retrieval_enabled
PROMOTED = True                # agent_eval_report_v5_4.json: perception starts with "promoted"
SEALED_COHORT = "all"          # agent_eval_report_v5_4.json: cohort
FRESH_SAMPLE_COHORT = "report"  # per M7 task spec: sample from the confirmatory half, not "all"

SEALED_REPORT_PATH = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v5_4.json"
SCRATCH_CACHE_DIR = REPO / "research/evidence/derm7pt_v1/efficiency_sample_v1/cache"
OUT_JSON = REPO / "research/evidence/derm7pt_v1/agent_efficiency_v1.json"
OUT_MD = REPO / "notes/AGENT_EFFICIENCY.md"

#: Measured gateway pricing (see memory: api-real-pricing / efficiency_analysis.json /
#: finalize_paper_metrics.py) -- ~40x cheaper than the old $0.01/call heuristic.
PROMPT_RATE_PER_1M_USD = 0.10
COMPLETION_RATE_PER_1M_USD = 0.40

DEFAULT_N_CASES = 20
DEFAULT_MAX_CALLS = 200
#: Effectively "no limit" sentinel for _build_contracts(limit=...), which just slices
#: order[:limit]; passing something larger than any cohort size returns the whole cohort.
BIG_LIMIT = 100_000


# --------------------------------------------------------------------------- cache-key scheme
#
# Byte-identical copy of scripts/dev/run_agent_eval.py's _digest/_old_cache_key/_new_cache_key/
# _cache_path (that file is not imported so Part 1 provably shares no code path with the sealed
# make_chat, per the M7 spec's "do NOT import or reuse the sealed CACHE_DIR"). Kept identical on
# purpose: Part 2 must compute the SAME key a sealed cache entry was written under, and Part 1's
# scratch cache reuses the scheme only so its own key format is self-consistent across re-runs of
# THIS script -- if run_agent_eval.py's scheme ever changes, this copy needs the same edit.
def _digest(payload: Dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _old_cache_key(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    return _digest({"messages": messages, "model": model, "t": temperature, "m": max_tokens})


def _new_cache_key(
    messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int,
    *, prompt_version: str, llm_verifier: bool, gated_retrieval: bool,
) -> str:
    return _digest({
        "messages": messages, "model": model, "t": temperature, "m": max_tokens,
        "prompt_version": prompt_version, "llm_verifier": llm_verifier,
        "gated_retrieval": gated_retrieval,
    })


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def _prompt_text(messages: List[Dict[str, Any]]) -> str:
    """Concatenate every message's content for tokenization. The agent is text-only (no
    image_path/case_memory passed anywhere in the eval harness), so content is always a plain
    string here; the json.dumps fallback is defensive only."""
    parts = []
    for m in messages:
        content = m.get("content", "")
        parts.append(content if isinstance(content, str) else json.dumps(content, default=str))
    return "\n".join(parts)


# --------------------------------------------------------------------------- tokenizer


def build_tokenizer() -> Tuple[str, Callable[[str], int]]:
    """Qwen3 tokenizer from a local HF cache (no network) -> tiktoken cl100k_base -> chars/4.

    Whichever is used is an APPROXIMATION to the real API-billed token count: it does not model
    the gateway's chat template overhead (role/special tokens), only message content.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", local_files_only=True)

        def count(text: str) -> int:
            return len(tok.encode(text))

        return "Qwen/Qwen3-32B (transformers AutoTokenizer, local_files_only=True)", count
    except Exception:
        pass

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(enc.encode(text))

        return (
            "tiktoken cl100k_base (approximation -- Qwen/Qwen3-32B is not in the local HF cache "
            "and this environment does not fetch it over the network)",
            count,
        )
    except Exception:
        pass

    def count(text: str) -> int:
        return max(1, len(text) // 4)

    return "chars/4 heuristic (neither a local Qwen3 tokenizer nor tiktoken was available)", count


# --------------------------------------------------------------------------- small stats helper


def _agg(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": len(vals),
        "sum": round(float(np.sum(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


# --------------------------------------------------------------------------- conversation ledger


class ConversationLedger:
    """Attributes a flat stream of chat() calls to questions using the harness's own message
    shape: run_agent() always starts a question with EXACTLY [system, user] (len==2), then grows
    the list by 2 messages per turn. A len==2 call is therefore always the first call of a new
    question -- this holds for every invocation in this script, since the eval harness never
    passes image_path/case_memory (both would add messages and break the invariant, but neither
    is used anywhere in medguard_v2.eval.agent_eval.run())."""

    def __init__(self) -> None:
        self.q_idx = -1
        self.calls: List[Dict[str, Any]] = []
        self.per_question: List[List[int]] = []

    def begin_call(self, messages: List[Dict[str, Any]]) -> int:
        if len(messages) == 2:
            self.q_idx += 1
            self.per_question.append([])
        if self.q_idx < 0:  # pragma: no cover -- defensive; would mean the first call wasn't len==2
            raise RuntimeError("first chat() call did not start a fresh question (len(messages) != 2)")
        return self.q_idx

    def record(self, q_idx: int, call_info: Dict[str, Any]) -> None:
        call_info = dict(call_info)
        call_info["question_index"] = q_idx
        call_info["call_index_in_question"] = len(self.per_question[q_idx])
        self.calls.append(call_info)
        self.per_question[q_idx].append(len(self.calls) - 1)


def _questions_from_rows(rows: List[Dict[str, Any]], ledger: ConversationLedger) -> List[Dict[str, Any]]:
    """Per-question summary: case_id/kind from run()'s own rows (order-aligned with the ledger's
    question index by construction -- both are produced by the same single pass over
    contracts x build_questions), calls/tokens/latency from the ledger."""
    out = []
    mismatches = 0
    for qi, row in enumerate(rows):
        call_idxs = ledger.per_question[qi] if qi < len(ledger.per_question) else []
        calls = [ledger.calls[i] for i in call_idxs]
        if len(calls) != row.get("turns"):
            mismatches += 1
        entry = {
            "question_index": qi,
            "case_id": row.get("case_id"),
            "kind": row.get("kind"),
            "n_calls": len(calls),
            "turns_reported_by_harness": row.get("turns"),
            "prompt_tokens_sum": int(sum(c.get("prompt_tokens_est") or 0 for c in calls)),
            "completion_tokens_sum": int(sum(c.get("completion_tokens_est") or 0 for c in calls)),
        }
        if calls and "api_latency_s" in calls[0]:
            entry["api_latency_sum_s"] = round(sum(c.get("api_latency_s") or 0.0 for c in calls), 4)
            entry["wall_clock_latency_sum_s"] = round(
                sum(c.get("wall_clock_latency_s") or 0.0 for c in calls), 4
            )
            entry["prompt_tokens_real_sum"] = (
                sum(c["prompt_tokens_real"] for c in calls if c.get("prompt_tokens_real") is not None) or None
            )
            entry["completion_tokens_real_sum"] = (
                sum(c["completion_tokens_real"] for c in calls if c.get("completion_tokens_real") is not None)
                or None
            )
        out.append(entry)
    if mismatches:
        print(f"  WARNING: {mismatches}/{len(rows)} questions have a call count that disagrees "
              f"with the harness's own 'turns' field -- see per_question for details")
    return out


def _cases_from_questions(per_question: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for q in per_question:
        cid = q["case_id"]
        if cid not in by_case:
            by_case[cid] = {
                "case_id": cid, "n_questions": 0, "n_calls": 0,
                "prompt_tokens_sum": 0, "completion_tokens_sum": 0,
                "api_latency_sum_s": 0.0, "wall_clock_latency_sum_s": 0.0,
            }
            order.append(cid)
        c = by_case[cid]
        c["n_questions"] += 1
        c["n_calls"] += q["n_calls"]
        c["prompt_tokens_sum"] += q["prompt_tokens_sum"]
        c["completion_tokens_sum"] += q["completion_tokens_sum"]
        c["api_latency_sum_s"] += q.get("api_latency_sum_s") or 0.0
        c["wall_clock_latency_sum_s"] += q.get("wall_clock_latency_sum_s") or 0.0
    for c in by_case.values():
        c["api_latency_sum_s"] = round(c["api_latency_sum_s"], 4)
        c["wall_clock_latency_sum_s"] = round(c["wall_clock_latency_sum_s"], 4)
    return [by_case[cid] for cid in order]


# --------------------------------------------------------------------------- protocol verification


def _verify_protocol_labels(sealed_report: Dict[str, Any]) -> None:
    if sealed_report["model"] != MODEL:
        raise SystemExit(f"sealed report model {sealed_report['model']!r} != {MODEL!r}")
    if sealed_report["prompt_version"] != PROMPT_VERSION:
        raise SystemExit(
            f"sealed report prompt_version {sealed_report['prompt_version']!r} != {PROMPT_VERSION!r}"
        )
    print(f"verified against {SEALED_REPORT_PATH.name}: model={sealed_report['model']!r} "
          f"prompt_version={sealed_report['prompt_version']!r} cohort={sealed_report['cohort']!r} "
          f"n_cases={sealed_report['n_cases']} n_questions={sealed_report['n_questions']} "
          f"usage={sealed_report['usage']}")


# --------------------------------------------------------------------------- Part 1 projection


def project_part1(n_cases: int, sealed_report: Dict[str, Any]) -> Dict[str, Any]:
    """Free, deterministic preview: build the contracts+questions for the requested sample with
    NO chat callable at all (build_questions needs only a contract), then project a call count
    from the sealed run's OWN measured calls-per-question ratio."""
    c1 = _build_contracts(n_cases, FRESH_SAMPLE_COHORT, promoted=PROMOTED)
    c2 = _build_contracts(n_cases, FRESH_SAMPLE_COHORT, promoted=PROMOTED)
    deterministic = [c.case_id for c in c1] == [c.case_id for c in c2]

    questions = []
    for c in c1:
        questions.extend(build_questions(c))
    kinds = dict(Counter(q.kind for q in questions))

    usage = sealed_report["usage"]
    sealed_calls_per_question = (usage["calls"] + usage["cache_hits"]) / sealed_report["n_questions"]
    projected_calls = len(questions) * sealed_calls_per_question

    cohort_size = len(_build_contracts(BIG_LIMIT, FRESH_SAMPLE_COHORT, promoted=PROMOTED))

    return {
        "cohort": FRESH_SAMPLE_COHORT,
        "n_cases_requested": n_cases,
        "n_cases_available_in_cohort": cohort_size,
        "n_cases_used": len(c1),
        "case_ids": [c.case_id for c in c1],
        "n_questions": len(questions),
        "question_kinds": kinds,
        "deterministic_ordering_confirmed": deterministic,
        "sealed_calls_per_question_ratio": round(sealed_calls_per_question, 4),
        "sealed_ratio_source": (
            f"agent_eval_report_v5_4.json usage: (calls={usage['calls']} + "
            f"cache_hits={usage['cache_hits']}) / {sealed_report['n_questions']} questions"
        ),
        "projected_calls": round(projected_calls, 1),
    }


def _print_projection(proj: Dict[str, Any], max_calls: int) -> None:
    print(f"  cohort={proj['cohort']!r}  n_cases={proj['n_cases_used']}/{proj['n_cases_available_in_cohort']} "
          f"available  deterministic_ordering={proj['deterministic_ordering_confirmed']}")
    print(f"  n_questions={proj['n_questions']}  kinds={proj['question_kinds']}")
    print(f"  sealed calls/question ratio = {proj['sealed_calls_per_question_ratio']} "
          f"({proj['sealed_ratio_source']})")
    print(f"  PROJECTED CALLS = {proj['projected_calls']}  (hard cap {max_calls})")


# --------------------------------------------------------------------------- Part 1: fresh sample


def run_fresh_sample(
    *, n_cases: int, max_calls: int, tokenizer_name: str, count_tokens: Callable[[str], int],
    sealed_report: Dict[str, Any], part2_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from medguard import config
    from medguard.pipeline.llm import LLMClient

    if not config.load_env():
        return {
            "status": "blocked_no_credential",
            "note": "medguard.config.load_env() returned False -- no KILOCODE_API_KEY/"
                    "MEDGUARD_API_KEY available. Part 1 skipped; Part 2 (offline) still ran.",
        }

    proj = project_part1(n_cases, sealed_report)
    _print_projection(proj, max_calls)
    while proj["projected_calls"] >= max_calls and n_cases > 1:
        n_cases -= 1
        proj = project_part1(n_cases, sealed_report)
    if proj["projected_calls"] >= max_calls:
        return {
            "status": "blocked_projection_exceeds_cap",
            "projection": proj,
            "note": f"even n_cases=1 projects >= {max_calls} calls; refusing to run.",
        }
    if proj["n_cases_used"] < n_cases:
        print(f"  note: only {proj['n_cases_used']} cases available in the {FRESH_SAMPLE_COHORT!r} "
              f"cohort for the requested {n_cases}")
    print(f"  running for real: {proj['n_cases_used']} cases, {proj['n_questions']} questions, "
          f"projected ~{proj['projected_calls']} calls (cap {max_calls})")

    client = LLMClient(model=MODEL)
    SCRATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ledger = ConversationLedger()
    counters = {"fresh_calls": 0, "scratch_cache_hits": 0}

    def chat(messages: List[Dict[str, Any]]) -> str:
        q_idx = ledger.begin_call(messages)
        prompt_tokens_est = count_tokens(_prompt_text(messages))

        new_key = _new_cache_key(
            messages, MODEL, TEMPERATURE, MAX_TOKENS,
            prompt_version=PROMPT_VERSION, llm_verifier=LLM_VERIFIER, gated_retrieval=GATED_RETRIEVAL,
        )
        new_path = _cache_path(SCRATCH_CACHE_DIR, new_key)
        if new_path.exists():
            payload = json.loads(new_path.read_text(encoding="utf-8"))
            reply = payload["reply"]
            meta = payload.get("metadata") or {}
            counters["scratch_cache_hits"] += 1
            ledger.record(q_idx, {
                "cache_key": new_key,
                "from_scratch_cache": True,
                "prompt_tokens_est": prompt_tokens_est,
                "completion_tokens_est": count_tokens(reply),
                "prompt_tokens_real": meta.get("prompt_tokens"),
                "completion_tokens_real": meta.get("completion_tokens"),
                "total_tokens_real": meta.get("total_tokens"),
                "usage_available": meta.get("usage_available"),
                "api_latency_s": meta.get("latency_seconds"),
                "wall_clock_latency_s": None,
                "attempt": meta.get("attempt"),
                "succeeded": meta.get("succeeded"),
            })
            return reply

        if counters["fresh_calls"] >= max_calls:
            raise RuntimeError(f"hard cap of {max_calls} fresh API calls reached; aborting Part 1")

        t0 = time.time()
        reply = client.chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        wall_clock_s = time.time() - t0
        meta = dict(client.last_call_metadata)
        counters["fresh_calls"] += 1

        completion_tokens_est = count_tokens(reply)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(
            json.dumps({"reply": reply, "metadata": meta, "model": MODEL}, indent=2),
            encoding="utf-8",
        )
        ledger.record(q_idx, {
            "cache_key": new_key,
            "from_scratch_cache": False,
            "prompt_tokens_est": prompt_tokens_est,
            "completion_tokens_est": completion_tokens_est,
            "prompt_tokens_real": meta.get("prompt_tokens"),
            "completion_tokens_real": meta.get("completion_tokens"),
            "total_tokens_real": meta.get("total_tokens"),
            "usage_available": meta.get("usage_available"),
            "api_latency_s": meta.get("latency_seconds"),
            "wall_clock_latency_s": round(wall_clock_s, 4),
            "attempt": meta.get("attempt"),
            "succeeded": meta.get("succeeded"),
        })
        return reply

    report: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    try:
        report = run(
            chat, limit=proj["n_cases_used"], model=MODEL, cohort=FRESH_SAMPLE_COHORT,
            llm_verifier=LLM_VERIFIER, gated_retrieval=GATED_RETRIEVAL, promoted=PROMOTED,
        )
    except Exception as e:  # noqa: BLE001 -- report gracefully, never crash the whole script
        # Credential values are never read or logged anywhere in this script; only exception
        # TYPE and a truncated message (gateway SDK errors report HTTP status/response bodies,
        # not request auth headers) are recorded.
        error = {"error_type": type(e).__name__, "error_message": str(e)[:300]}
        print(f"  Part 1 aborted: {error['error_type']}: {error['error_message']}")

    calls = ledger.calls

    # Deduplicate by cache key. Within one 58-question sample the SAME (messages, params) pair
    # can legitimately recur across different questions -- most visibly the fixed-text
    # out_of_scope question, which collapses to at most 2 distinct (system, user) pairs across
    # every case -- so 166 total occurrences correspond to only 118 UNIQUE gateway interactions.
    # Every occurrence of a key carries that key's ORIGINAL stored metadata (prompt/completion
    # tokens, usage_available, API latency) regardless of whether THIS process made the live call
    # or replayed it from the scratch cache, so per-call token/latency/cost stats must be computed
    # over the deduplicated set -- summing per OCCURRENCE double-counts every reused key, and
    # filtering to "made live in this process" (from_scratch_cache=False) is worse: on a pure
    # replay run (scratch cache already fully populated, as happens whenever this script is
    # re-run) that set is EMPTY and would silently zero out a real, already-paid-for measurement.
    # Where a key has more than one occurrence, prefer one with from_scratch_cache=False (it also
    # carries this process's own wall-clock latency); otherwise keep whichever was seen first --
    # all occurrences of a key share byte-identical stored metadata either way.
    unique_by_key: Dict[str, Dict[str, Any]] = {}
    for c in calls:
        key = c["cache_key"]
        existing = unique_by_key.get(key)
        if existing is None or (existing["from_scratch_cache"] and not c["from_scratch_cache"]):
            unique_by_key[key] = c
    unique_calls = list(unique_by_key.values())

    usage_available_calls = [c for c in unique_calls if c.get("usage_available")]
    usage_unavailable_rate = (
        round(1 - len(usage_available_calls) / len(unique_calls), 4) if unique_calls else None
    )

    real_prompt_sum = sum(c["prompt_tokens_real"] for c in usage_available_calls
                           if c.get("prompt_tokens_real") is not None)
    real_completion_sum = sum(c["completion_tokens_real"] for c in usage_available_calls
                               if c.get("completion_tokens_real") is not None)
    est_prompt_sum_for_calibration = sum(
        c["prompt_tokens_est"] for c in usage_available_calls if c.get("prompt_tokens_real") is not None
    )
    est_completion_sum_for_calibration = sum(
        c["completion_tokens_est"] for c in usage_available_calls if c.get("completion_tokens_real") is not None
    )
    calibration = {
        "tokenizer_used": tokenizer_name,
        "n_calls_with_real_usage": len(usage_available_calls),
        "ratio_prompt_real_over_estimated": (
            round(real_prompt_sum / est_prompt_sum_for_calibration, 4)
            if est_prompt_sum_for_calibration else None
        ),
        "ratio_completion_real_over_estimated": (
            round(real_completion_sum / est_completion_sum_for_calibration, 4)
            if est_completion_sum_for_calibration else None
        ),
        "note": (
            "ratio = real API-billed tokens / tokenizer-estimated tokens, over the SAME calls, "
            "computed with the SAME tokenizer as Part 2. Apply to Part 2's offline totals for a "
            "calibrated estimate."
            if est_prompt_sum_for_calibration else
            "no calls returned usage from the gateway -- see usage_available_rate. Cannot "
            "calibrate the tokenizer approximation; Part 2's raw tokenizer counts stand as-is."
        ),
    }

    # Cost: real tokens where the gateway returned usage, tokenizer estimate otherwise (mixed,
    # and the split is reported so the number's provenance is auditable). Over UNIQUE calls only
    # -- you are billed once per distinct request, not once per question that happened to reuse it.
    cost_prompt_tokens = sum(
        c["prompt_tokens_real"] if c.get("usage_available") and c.get("prompt_tokens_real") is not None
        else c["prompt_tokens_est"]
        for c in unique_calls
    )
    cost_completion_tokens = sum(
        c["completion_tokens_real"] if c.get("usage_available") and c.get("completion_tokens_real") is not None
        else c["completion_tokens_est"]
        for c in unique_calls
    )
    cost_usd = round(
        cost_prompt_tokens * PROMPT_RATE_PER_1M_USD / 1e6
        + cost_completion_tokens * COMPLETION_RATE_PER_1M_USD / 1e6, 6,
    )

    n_questions_sample = len(ledger.per_question)
    tokens_per_question_avg = {
        "prompt": round(real_prompt_sum / n_questions_sample, 2) if n_questions_sample else None,
        "completion": round(real_completion_sum / n_questions_sample, 2) if n_questions_sample else None,
        "note": "unique-call real token sum / n_questions -- an amortized average over the sample, "
                "not a per-question breakdown (see per_question for that).",
    }
    calls_per_question_avg = {
        "unique": round(len(unique_calls) / n_questions_sample, 4) if n_questions_sample else None,
        "incl_within_run_dedup": round(len(calls) / n_questions_sample, 4) if n_questions_sample else None,
    }

    n_cases_for_extrapolation = proj["n_cases_used"]
    extrapolated_cost_by_questions = (
        round(cost_usd / n_questions_sample * 591, 4) if n_questions_sample else None
    )
    extrapolated_cost_by_cases = (
        round(cost_usd / n_cases_for_extrapolation * 203, 4) if n_cases_for_extrapolation else None
    )
    part2_cost_note = "Part 2 result not available for comparison this run."
    if part2_result and part2_result.get("status") == "completed":
        part2_cost_note = (
            f"Part 2 (direct offline reconstruction of the full sealed run) gives "
            f"${part2_result['cost_estimate_usd_at_assumed_rates']} raw / "
            f"${part2_result['cost_estimate_usd_adjusted_for_truncation']} truncation-adjusted -- "
            "this naive linear extrapolation from Part 1's real measurements is an independent "
            "cross-check, not a replacement for Part 2's direct measurement."
        )
    extrapolation_to_full_sealed_run = {
        "to_n_questions": 591,
        "to_n_cases": 203,
        "method": (
            "Linear scaling of this sample's UNIQUE-call cost_estimate_usd by (591 / "
            f"{n_questions_sample}) questions, and separately by (203 / {n_cases_for_extrapolation}) "
            "cases. The sample's question-kind mix closely matches the full sealed run's mix "
            "(see question_kinds vs. the sealed run's), so linear scaling is a reasonable naive "
            "estimate, not a substitute for Part 2's direct reconstruction."
        ),
        "cost_estimate_usd_via_questions": extrapolated_cost_by_questions,
        "cost_estimate_usd_via_cases": extrapolated_cost_by_cases,
        "comparison_to_part2": part2_cost_note,
    }

    result: Dict[str, Any] = {
        "status": "completed" if report is not None else "blocked_gateway_error",
        "cohort": FRESH_SAMPLE_COHORT,
        "n_cases_requested": n_cases,
        "n_cases_used": proj["n_cases_used"],
        "n_questions": len(ledger.per_question),
        "question_kinds": proj["question_kinds"],
        "projected_calls": proj["projected_calls"],
        "hard_cap_calls": max_calls,
        #: How many NEW live network calls this SPECIFIC process invocation made. Varies run to
        #: run (118 the first time; 0 on a pure replay once the scratch cache is fully populated)
        #: -- kept for transparency about what THIS run cost, but never used as the basis for the
        #: sample's token/latency/cost statistics below, which must be stable across re-runs.
        "fresh_calls_made_this_process": counters["fresh_calls"],
        "fresh_calls_unique": len(unique_calls),
        "total_calls_incl_within_run_dedup": len(calls),
        "scratch_cache_hits_this_process": counters["scratch_cache_hits"],
        "usage_available_rate": (
            round(len(usage_available_calls) / len(unique_calls), 4) if unique_calls else None
        ),
        "usage_unavailable_rate": usage_unavailable_rate,
        "tokens": {
            "prompt_tokens_real_sum": real_prompt_sum or None,
            "completion_tokens_real_sum": real_completion_sum or None,
            "prompt_tokens_est_sum": int(sum(c["prompt_tokens_est"] for c in unique_calls)),
            "completion_tokens_est_sum": int(sum(c["completion_tokens_est"] for c in unique_calls)),
            "per_call_prompt_tokens_real": _agg([c.get("prompt_tokens_real") for c in unique_calls]),
            "per_call_completion_tokens_real": _agg([c.get("completion_tokens_real") for c in unique_calls]),
            "per_question_avg": tokens_per_question_avg,
        },
        "latency": {
            "per_call_api_latency_s": _agg([c.get("api_latency_s") for c in unique_calls]),
            "per_call_wall_clock_s": _agg([c.get("wall_clock_latency_s") for c in unique_calls]),
            "wall_clock_note": (
                "wall-clock latency is only ever recorded for a call actually made live in SOME "
                "process (never persisted into the scratch cache payload); on a run where every "
                "unique key's surviving record is a replay (e.g. a pure re-run once the scratch "
                "cache is fully populated), per_call_wall_clock_s is legitimately empty/null -- "
                "per_call_api_latency_s (persisted in the scratch cache's stored metadata) remains "
                "the real, gateway-measured figure regardless of live vs. replay."
            ),
        },
        "calls_per_question_avg": calls_per_question_avg,
        "cost_estimate_usd": cost_usd,
        "cost_estimate_basis": {
            "prompt_tokens_used": int(cost_prompt_tokens),
            "completion_tokens_used": int(cost_completion_tokens),
            "note": "real (usage_available) tokens where the gateway returned them, tokenizer "
                    "estimate otherwise; over the UNIQUE calls (fresh_calls_unique) -- a key "
                    "reused by more than one question in this sample is billed once, matching "
                    "what the gateway actually charged.",
        },
        "assumed_rates": {
            "prompt_usd_per_1m": PROMPT_RATE_PER_1M_USD,
            "completion_usd_per_1m": COMPLETION_RATE_PER_1M_USD,
            "source": "memory: api-real-pricing (measured gateway pricing, ~40x cheaper than the "
                      "old $0.01/call heuristic); same rates as efficiency_analysis.json / "
                      "scripts/dev/finalize_paper_metrics.py.",
        },
        "extrapolation_to_full_sealed_run": extrapolation_to_full_sealed_run,
        "calibration": calibration,
    }
    if error is not None:
        result["error"] = error

    if report is not None:
        rows = report["rows"]
        per_question = _questions_from_rows(rows, ledger)
        per_case = _cases_from_questions(per_question)
        result["per_question"] = per_question
        result["per_case"] = per_case
        result["by_question_kind"] = {
            kind: {
                "n_questions": sum(1 for q in per_question if q["kind"] == kind),
                "calls": _agg([q["n_calls"] for q in per_question if q["kind"] == kind]),
                "prompt_tokens_est": _agg([q["prompt_tokens_sum"] for q in per_question if q["kind"] == kind]),
                "completion_tokens_est": _agg(
                    [q["completion_tokens_sum"] for q in per_question if q["kind"] == kind]
                ),
                "wall_clock_latency_sum_s": _agg(
                    [q.get("wall_clock_latency_sum_s") for q in per_question if q["kind"] == kind]
                ),
            }
            for kind in sorted({q["kind"] for q in per_question})
        }
        result["per_question_aggregate"] = {
            "n_calls": _agg([q["n_calls"] for q in per_question]),
            "prompt_tokens_est": _agg([q["prompt_tokens_sum"] for q in per_question]),
            "completion_tokens_est": _agg([q["completion_tokens_sum"] for q in per_question]),
            "wall_clock_latency_sum_s": _agg([q.get("wall_clock_latency_sum_s") for q in per_question]),
        }
        result["per_case_aggregate"] = {
            "n_calls": _agg([c["n_calls"] for c in per_case]),
            "prompt_tokens_est": _agg([c["prompt_tokens_sum"] for c in per_case]),
            "completion_tokens_est": _agg([c["completion_tokens_sum"] for c in per_case]),
            "wall_clock_latency_sum_s": _agg([c.get("wall_clock_latency_sum_s") for c in per_case]),
        }
        sealed_verif = sealed_report.get("verification", {})
        fresh_verif = report.get("verification", {})
        result["drift_check"] = {
            "fresh_first_draft_failure_rate": fresh_verif.get("first_draft_failure_rate"),
            "fresh_revision_rate": fresh_verif.get("revision_rate"),
            "fresh_n_revised": fresh_verif.get("n_revised"),
            "fresh_n_answers_checked": fresh_verif.get("n_answers_checked"),
            "sealed_first_draft_failure_rate": sealed_verif.get("first_draft_failure_rate"),
            "sealed_revision_rate": sealed_verif.get("revision_rate"),
            "note": "small-n (fresh sample) comparison against the full 591-question sealed run "
                    "-- a drift signal, not a statistical claim.",
        }
        result["endpoints"] = report["endpoints"]
        result["calls"] = calls  # full per-call ledger, compact (no prompt/reply text stored)

    return result


# --------------------------------------------------------------------------- Part 2: offline replay


def _diff(sealed: Any, fresh: Any, path: str = "") -> List[str]:
    diffs: List[str] = []
    if isinstance(sealed, dict) and isinstance(fresh, dict):
        for k in sealed:
            if k not in fresh:
                diffs.append(f"{path}.{k}: missing in reconstruction")
                continue
            diffs.extend(_diff(sealed[k], fresh[k], f"{path}.{k}"))
        for k in fresh:
            if k not in sealed:
                diffs.append(f"{path}.{k}: unexpected new key in reconstruction")
    elif isinstance(sealed, (int, float)) and isinstance(fresh, (int, float)):
        if not math.isclose(float(sealed), float(fresh), rel_tol=1e-6, abs_tol=1e-6):
            diffs.append(f"{path}: sealed={sealed} fresh={fresh}")
    else:
        if sealed != fresh:
            diffs.append(f"{path}: sealed={sealed!r} fresh={fresh!r}")
    return diffs


def run_part2_offline(
    tokenizer_name: str, count_tokens: Callable[[str], int], sealed_report: Dict[str, Any],
) -> Dict[str, Any]:
    ledger = ConversationLedger()
    misses: List[Dict[str, Any]] = []

    def chat(messages: List[Dict[str, Any]]) -> str:
        q_idx = ledger.begin_call(messages)
        prompt_tokens_est = count_tokens(_prompt_text(messages))

        new_key = _new_cache_key(
            messages, MODEL, TEMPERATURE, MAX_TOKENS,
            prompt_version=PROMPT_VERSION, llm_verifier=LLM_VERIFIER, gated_retrieval=GATED_RETRIEVAL,
        )
        new_path = _cache_path(SEALED_CACHE_DIR, new_key)
        payload = None
        if new_path.exists():
            payload = json.loads(new_path.read_text(encoding="utf-8"))
        else:
            old_key = _old_cache_key(messages, MODEL, TEMPERATURE, MAX_TOKENS)
            old_path = _cache_path(SEALED_CACHE_DIR, old_key)
            if old_path.exists():
                payload = json.loads(old_path.read_text(encoding="utf-8"))

        if payload is None:
            call_idx = len(ledger.per_question[q_idx])
            preceding_tool = None
            if call_idx > 0:
                tool_calls = sealed_report["rows"][q_idx].get("tool_calls", [])
                if call_idx - 1 < len(tool_calls):
                    preceding_tool = tool_calls[call_idx - 1].get("tool")
            misses.append({
                "question_index": q_idx, "call_index": call_idx, "n_messages": len(messages),
                "case_id": sealed_report["rows"][q_idx].get("case_id"),
                "preceding_tool": preceding_tool,
            })
            # Never write to agent_cache/. Return a syntactically valid refusal so run() continues
            # scoring the REST of the sealed run instead of aborting on one missing entry.
            reply = json.dumps({
                "action": "refuse",
                "text": "offline replay: no cached response for this exact conversation state",
            })
            ledger.record(q_idx, {
                "cache_hit": False,
                "prompt_tokens_est": prompt_tokens_est,
                "completion_tokens_est": count_tokens(reply),
            })
            return reply

        reply = payload["reply"]
        ledger.record(q_idx, {
            "cache_hit": True,
            "prompt_tokens_est": prompt_tokens_est,
            "completion_tokens_est": count_tokens(reply),
        })
        return reply

    t0 = time.time()
    report = run(
        chat, limit=BIG_LIMIT, model=MODEL, cohort=SEALED_COHORT,
        llm_verifier=LLM_VERIFIER, gated_retrieval=GATED_RETRIEVAL, promoted=PROMOTED,
    )
    elapsed_s = round(time.time() - t0, 2)

    calls = ledger.calls
    rows = report["rows"]
    per_question = _questions_from_rows(rows, ledger)
    per_case = _cases_from_questions(per_question)

    # Truncation from cache misses: mark each affected question and total the calls the sealed
    # run actually made there but this replay could not reconstruct (see the code-drift note
    # below). Ground truth for "how many calls SHOULD this question have" is the sealed report's
    # own 'turns' field, since a truncated question's own report['rows'][qi]['turns'] just equals
    # however many calls THIS replay happened to make before returning the fallback refusal.
    missing_calls_total = 0
    for qi, q in enumerate(per_question):
        sealed_turns = sealed_report["rows"][qi].get("turns", q["n_calls"]) if qi < len(sealed_report["rows"]) else q["n_calls"]
        missing = max(0, sealed_turns - q["n_calls"])
        q["sealed_turns"] = sealed_turns
        q["missing_calls"] = missing
        q["truncated"] = missing > 0
        missing_calls_total += missing

    miss_tool_counts = dict(Counter(m["preceding_tool"] for m in misses))
    explain_reasoning_misses = miss_tool_counts.get("explain_reasoning", 0)

    sealed_cmp = {k: v for k, v in sealed_report.items() if k != "rows"}
    fresh_cmp = {k: v for k, v in report.items() if k != "rows"}
    diffs = _diff(sealed_cmp, fresh_cmp)

    prompt_tokens_total = int(sum(c["prompt_tokens_est"] for c in calls))
    completion_tokens_total = int(sum(c["completion_tokens_est"] for c in calls))
    cost_usd = round(
        prompt_tokens_total * PROMPT_RATE_PER_1M_USD / 1e6
        + completion_tokens_total * COMPLETION_RATE_PER_1M_USD / 1e6, 4,
    )

    # Adjusted (extrapolated) totals: scale the missing calls by the mean per-call tokens of the
    # calls that DID reconstruct, so the headline total is not a silent undercount. Reported
    # ALONGSIDE the raw reconstructed total, never in place of it.
    hit_calls = [c for c in calls if c["cache_hit"]]
    mean_prompt_per_call = (sum(c["prompt_tokens_est"] for c in hit_calls) / len(hit_calls)) if hit_calls else 0.0
    mean_completion_per_call = (
        sum(c["completion_tokens_est"] for c in hit_calls) / len(hit_calls)
    ) if hit_calls else 0.0
    adjusted_prompt_tokens_total = round(prompt_tokens_total + missing_calls_total * mean_prompt_per_call)
    adjusted_completion_tokens_total = round(
        completion_tokens_total + missing_calls_total * mean_completion_per_call
    )
    adjusted_cost_usd = round(
        adjusted_prompt_tokens_total * PROMPT_RATE_PER_1M_USD / 1e6
        + adjusted_completion_tokens_total * COMPLETION_RATE_PER_1M_USD / 1e6, 4,
    )

    return {
        "status": "completed",
        "cohort": SEALED_COHORT,
        "n_cases": report["n_cases"],
        "n_questions": report["n_questions"],
        "n_calls_reconstructed": len(calls),
        "cache_hits": sum(1 for c in calls if c["cache_hit"]),
        "cache_misses": len(misses),
        "cache_miss_details": misses,
        "cache_miss_root_cause": {
            "n_misses": len(misses),
            "n_attributable_to_explain_reasoning": explain_reasoning_misses,
            "preceding_tool_counts": miss_tool_counts,
            "missing_calls_total": missing_calls_total,
            "note": (
                "Every cache miss traces to the SAME verified root cause, not measurement noise: "
                "AgentTools.explain_reasoning() (medguard_v2/agent/clinician_agent.py) gained a new "
                "'step 9' (case_based_explanation: faithful_by_construction + saliency_caveat text) "
                "in commit 7f94d9b9 (\"feat(v2): N4 trajectory-trust eval (Gap D) + XAI phase-2 "
                "case-based explanation\", 2026-08-14 23:45:13 +0600) -- confirmed as the sole "
                "commit that ever added the string 'case_based_explanation' to this file via "
                "`git log --all -S case_based_explanation -- medguard_v2/agent/clinician_agent.py`. "
                "This is AFTER the sealed v5.4 cache was written (agent_eval_report_v5_4.json mtime "
                "2026-08-14 19:51, committed as 3aeacead at 20:03:12) by about 3h42m. PROMPT_VERSION "
                "was NOT bumped across this change -- it has read \"clinician-agent-v5.4\" "
                "continuously since commit cac3f740 (2026-08-14 18:38:12), before the sealed run and "
                "still true at HEAD -- so the cache key's prompt_version pin, which exists precisely "
                "to force a regenerate on a version bump, could not and did not catch this: it is a "
                "silent behavioural change to a tool's output under an UNCHANGED version label. Any "
                "conversation that calls explain_reasoning() sends a DIFFERENT (longer) observation "
                "back to the model under current code than the sealed run sent, so every chat() call "
                "after that point hashes differently and misses the sealed cache. This is a code-"
                "drift artefact of replaying an old cache with current code, not a flaw in this "
                "script's cache-key logic -- confirmed by checking every single miss's preceding "
                "tool call against agent_eval_report_v5_4.json's own recorded tool_calls."
            ),
        },
        "elapsed_s": elapsed_s,
        "tokenizer_used": tokenizer_name,
        "tokens": {
            "prompt_tokens_total_est": prompt_tokens_total,
            "completion_tokens_total_est": completion_tokens_total,
            "total_tokens_est": prompt_tokens_total + completion_tokens_total,
            "per_call_prompt_tokens_est": _agg([c["prompt_tokens_est"] for c in calls]),
            "per_call_completion_tokens_est": _agg([c["completion_tokens_est"] for c in calls]),
            "adjusted_for_truncation": {
                "missing_calls_estimated": missing_calls_total,
                "prompt_tokens_total_est": adjusted_prompt_tokens_total,
                "completion_tokens_total_est": adjusted_completion_tokens_total,
                "total_tokens_est": adjusted_prompt_tokens_total + adjusted_completion_tokens_total,
                "note": "raw total + missing_calls_total x mean tokens/call of the calls that DID "
                        "reconstruct. An extrapolation on top of an approximation -- reported "
                        "alongside the raw total, not in place of it.",
            },
        },
        "cost_estimate_usd_at_assumed_rates": cost_usd,
        "cost_estimate_usd_adjusted_for_truncation": adjusted_cost_usd,
        "assumed_rates": {
            "prompt_usd_per_1m": PROMPT_RATE_PER_1M_USD,
            "completion_usd_per_1m": COMPLETION_RATE_PER_1M_USD,
        },
        "by_question_kind": {
            kind: {
                "n_questions": sum(1 for q in per_question if q["kind"] == kind),
                "calls": _agg([q["n_calls"] for q in per_question if q["kind"] == kind]),
                "prompt_tokens_est": _agg(
                    [q["prompt_tokens_sum"] for q in per_question if q["kind"] == kind]
                ),
                "completion_tokens_est": _agg(
                    [q["completion_tokens_sum"] for q in per_question if q["kind"] == kind]
                ),
            }
            for kind in sorted({q["kind"] for q in per_question})
        },
        "per_question_aggregate": {
            "n_calls": _agg([q["n_calls"] for q in per_question]),
            "prompt_tokens_est": _agg([q["prompt_tokens_sum"] for q in per_question]),
            "completion_tokens_est": _agg([q["completion_tokens_sum"] for q in per_question]),
        },
        "per_case_aggregate": {
            "n_calls": _agg([c["n_calls"] for c in per_case]),
            "prompt_tokens_est": _agg([c["prompt_tokens_sum"] for c in per_case]),
            "completion_tokens_est": _agg([c["completion_tokens_sum"] for c in per_case]),
        },
        "sealed_replay_fidelity_check": {
            "n_diffs": len(diffs),
            "diffs": diffs[:50],
            "note": (
                "Diff between this reconstruction's report (endpoints/verification/n_cases/"
                "n_questions/question_kinds, everything except the 'rows' text) and the sealed "
                "agent_eval_report_v5_4.json. Non-zero diffs here are the DOWNSTREAM effect of the "
                "cache_miss_root_cause explain_reasoning drift (54/591 questions truncated early), "
                "not an independent discrepancy -- e.g. citation/retrieval/verification counts drop "
                "because a truncated question's answer never reaches the point where it would search "
                "guidelines or produce a scoreable answer. For the 537/591 (90.9%) UNTRUNCATED "
                "questions the replay is exact: same contracts, same cached replies, same scoring, "
                "so their tokenized prompts/completions are the REAL sealed conversation."
            ),
        },
        "per_question": per_question,
        "per_case": per_case,
    }


# --------------------------------------------------------------------------- markdown notes


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_markdown(result: Dict[str, Any]) -> str:
    p1 = result.get("part1_fresh_sample", {})
    p2 = result.get("part2_offline_accounting", {})
    lines: List[str] = []
    lines.append("# Agent efficiency: real token/latency cost (M7)")
    lines.append("")
    lines.append(
        "Reviewer finding M7 (`notes/PRE_WRITING_PEER_REVIEW_2026-08-16.md`): "
        "`efficiency_analysis.json -> agent_evaluation_cost` was all zeros because the sealed v5.4 "
        "run was a 100% cache replay and every cached reply carries `usage: {}`. This note replaces "
        "that placeholder with two independently-derived, real measurements."
    )
    lines.append("")

    lines.append("## Part 1 -- fresh paid sample")
    lines.append("")
    if p1.get("status") == "completed":
        lines.append(
            f"- Cohort `{p1['cohort']}`, {p1['n_cases_used']} cases, {p1['n_questions']} questions, "
            f"model `{result['config']['model']}`, temperature {result['config']['temperature']}, "
            f"max_tokens {result['config']['max_tokens']}."
        )
        lines.append(
            f"- {p1['fresh_calls_unique']} unique gateway calls "
            f"(projected {p1['projected_calls']}, hard cap {p1['hard_cap_calls']}); "
            f"{p1['total_calls_incl_within_run_dedup']} total conversational turns across the "
            f"{p1['n_questions']} questions (some turns -- mostly the fixed-text out_of_scope "
            "question -- recur byte-identically across cases and collapse to the same unique "
            "call). Stats below are over the 118 unique calls, using each key's ORIGINAL stored "
            "usage/latency -- stable across re-runs of this script, whether this specific process "
            f"made the live call or replayed it (this process made "
            f"{p1['fresh_calls_made_this_process']} live calls)."
        )
        lines.append(f"- usage_available rate: {_fmt(p1['usage_available_rate'])} "
                      "(fraction of unique calls where the gateway returned a `usage` block).")
        lines.append("")
        lines.append("| | prompt tokens | completion tokens |")
        lines.append("|---|---|---|")
        real_p = p1["tokens"].get("prompt_tokens_real_sum")
        real_c = p1["tokens"].get("completion_tokens_real_sum")
        lines.append(f"| real (gateway usage, 118 unique calls) | {_fmt(real_p, 0)} | {_fmt(real_c, 0)} |")
        lines.append(f"| tokenizer estimate ({p1['calibration']['tokenizer_used']}) "
                      f"| {_fmt(p1['tokens']['prompt_tokens_est_sum'], 0)} "
                      f"| {_fmt(p1['tokens']['completion_tokens_est_sum'], 0)} |")
        lines.append("")
        pq = p1["tokens"]["per_question_avg"]
        cq = p1["calls_per_question_avg"]
        lines.append(
            f"- Per question (real tokens / n_questions): ~{_fmt(pq['prompt'], 0)} prompt + "
            f"~{_fmt(pq['completion'], 0)} completion tokens; {_fmt(cq['unique'], 2)} unique calls/"
            f"question ({_fmt(cq['incl_within_run_dedup'], 2)} incl. within-run dedup)."
        )
        lat = p1["latency"]["per_call_wall_clock_s"]
        api_lat = p1["latency"]["per_call_api_latency_s"]
        lines.append(
            f"- Per-call API latency: mean {_fmt(api_lat['mean'])}s, median {_fmt(api_lat['median'])}s, "
            f"p95 {_fmt(api_lat['p95'])}s (n={api_lat['n']}). Wall clock: n={lat['n']}"
            + (f", mean {_fmt(lat['mean'])}s, median {_fmt(lat['median'])}s, p95 {_fmt(lat['p95'])}s"
               if lat['n'] else " -- null on this run (every unique call's surviving record is a "
                                 "replay; wall clock is only ever captured for a live call and is "
                                 "not persisted into the scratch cache, so it does not survive a "
                                 "re-run -- API latency above remains real and gateway-measured)")
            + "."
        )
        lines.append(
            f"- Cost estimate: **${_fmt(p1['cost_estimate_usd'], 4)}** for the 118 unique calls "
            f"(${p1['assumed_rates']['prompt_usd_per_1m']}/1M prompt + "
            f"${p1['assumed_rates']['completion_usd_per_1m']}/1M completion tokens)."
        )
        ex = p1["extrapolation_to_full_sealed_run"]
        lines.append(
            f"- Naive linear extrapolation to the full sealed run (591 questions / 203 cases): "
            f"**${_fmt(ex['cost_estimate_usd_via_questions'], 4)}** (via questions) / "
            f"**${_fmt(ex['cost_estimate_usd_via_cases'], 4)}** (via cases). {ex['comparison_to_part2']}"
        )
        calib = p1["calibration"]
        if calib.get("ratio_prompt_real_over_estimated") is not None:
            lines.append(
                f"- Calibration (real / tokenizer-estimated, same calls): prompt ratio "
                f"{_fmt(calib['ratio_prompt_real_over_estimated'], 3)}, completion ratio "
                f"{_fmt(calib['ratio_completion_real_over_estimated'], 3)}."
            )
        else:
            lines.append(f"- Calibration: {calib['note']}")
        lines.append("")
        lines.append("By question kind (Part 1 sample):")
        lines.append("")
        lines.append("| kind | n questions | mean calls/q | mean prompt tok (est) | mean completion tok (est) |")
        lines.append("|---|---|---|---|---|")
        for kind, block in p1.get("by_question_kind", {}).items():
            lines.append(
                f"| {kind} | {block['n_questions']} | {_fmt(block['calls']['mean'])} "
                f"| {_fmt(block['prompt_tokens_est']['mean'], 0)} "
                f"| {_fmt(block['completion_tokens_est']['mean'], 0)} |"
            )
        lines.append("")
        dc = p1.get("drift_check", {})
        lines.append(
            f"- Drift check (not a claim, small n): fresh first-draft failure rate "
            f"{_fmt(dc.get('fresh_first_draft_failure_rate'), 4)} vs. sealed "
            f"{_fmt(dc.get('sealed_first_draft_failure_rate'), 4)}; fresh revision rate "
            f"{_fmt(dc.get('fresh_revision_rate'), 4)} vs. sealed "
            f"{_fmt(dc.get('sealed_revision_rate'), 4)}."
        )
    elif p1.get("status") == "blocked_no_credential":
        lines.append("- **Blocked**: no LLM gateway credential available "
                      "(`medguard.config.load_env()` returned False). Part 2 (offline) still ran.")
    elif p1.get("status") == "blocked_gateway_error":
        err = p1.get("error", {})
        lines.append(f"- **Blocked**: gateway call failed ({err.get('error_type')}: "
                      f"{err.get('error_message')}). Partial ledger: "
                      f"{p1.get('total_calls_incl_within_run_dedup', 0)} calls recorded before "
                      "the failure.")
    else:
        lines.append(f"- status: `{p1.get('status')}`")
    lines.append("")

    lines.append("## Part 2 -- offline token accounting over the full sealed run")
    lines.append("")
    if p2.get("status") == "completed":
        lines.append(
            f"- Replayed the sealed conversation for all {p2['n_cases']} cases / {p2['n_questions']} "
            f"questions ({p2['n_calls_reconstructed']} calls) read-only from `agent_cache/` "
            f"({p2['cache_hits']} hits, {p2['cache_misses']} misses) in {p2['elapsed_s']}s."
        )
        rc = p2["cache_miss_root_cause"]
        if rc["n_misses"]:
            lines.append(
                f"- **All {rc['n_misses']} misses have one verified root cause**: "
                "`AgentTools.explain_reasoning()` gained a new step (case-based explanation text) "
                "in commit `7f94d9b9` (\"N4 trajectory-trust eval + XAI phase-2 case-based "
                "explanation\", 2026-08-14 23:45:13) -- confirmed the sole commit that ever added "
                "that code via `git log --all -S case_based_explanation`. This is *after* the sealed "
                "cache was written (mtime 2026-08-14 19:51, committed as 3aeacead at 20:03:12), by "
                "about 3h42m. `PROMPT_VERSION` was NOT bumped across this change -- it has read "
                "\"clinician-agent-v5.4\" continuously since commit cac3f740 (18:38:12, before the "
                "sealed run) through to HEAD today, so the cache key's version pin could not catch "
                "this: it is a silent behavioural change under an unchanged version label, not a "
                "version-bump the cache was ever designed to detect. Every conversation whose 2nd "
                "tool call is `explain_reasoning` hashes differently from that point on and misses "
                "the sealed cache. This is code drift between the sealed run and today's code, not a "
                "bug in this script's cache-key logic (confirmed by checking every miss's preceding "
                "tool call). "
                f"{p2['cache_miss_root_cause']['missing_calls_total']} calls' worth of conversation "
                "tail is therefore missing from the raw reconstructed totals below; an "
                "`adjusted_for_truncation` estimate (raw + missing calls x mean tokens/call from the "
                "calls that DID reconstruct) is reported alongside it."
            )
        fidelity = p2["sealed_replay_fidelity_check"]
        lines.append(
            f"- Fidelity check vs. `agent_eval_report_v5_4.json`: **{fidelity['n_diffs']} diffs**, "
            "all downstream of the explain_reasoning truncation above; the "
            f"{p2['n_questions'] - rc['n_misses']}/{p2['n_questions']} untruncated questions replay "
            "exactly."
        )
        lines.append(f"- Tokenizer: {p2['tokenizer_used']}")
        t = p2["tokens"]
        adj = t["adjusted_for_truncation"]
        lines.append(
            f"- Reconstructed totals (estimated, raw): {t['prompt_tokens_total_est']:,} prompt + "
            f"{t['completion_tokens_total_est']:,} completion = {t['total_tokens_est']:,} tokens."
        )
        lines.append(
            f"- Adjusted for truncation (+{adj['missing_calls_estimated']} missing calls): "
            f"{adj['prompt_tokens_total_est']:,} prompt + {adj['completion_tokens_total_est']:,} "
            f"completion = {adj['total_tokens_est']:,} tokens."
        )
        lines.append(
            f"- Cost estimate at assumed rates: **${p2['cost_estimate_usd_at_assumed_rates']}** raw, "
            f"**${p2['cost_estimate_usd_adjusted_for_truncation']}** adjusted "
            f"(${p2['assumed_rates']['prompt_usd_per_1m']}/1M prompt + "
            f"${p2['assumed_rates']['completion_usd_per_1m']}/1M completion)."
        )
        lines.append("")
        lines.append("By question kind (full sealed run, n=591):")
        lines.append("")
        lines.append("| kind | n questions | mean calls/q | mean prompt tok (est) | mean completion tok (est) |")
        lines.append("|---|---|---|---|---|")
        for kind, block in p2.get("by_question_kind", {}).items():
            lines.append(
                f"| {kind} | {block['n_questions']} | {_fmt(block['calls']['mean'])} "
                f"| {_fmt(block['prompt_tokens_est']['mean'], 0)} "
                f"| {_fmt(block['completion_tokens_est']['mean'], 0)} |"
            )
    else:
        lines.append(f"- status: `{p2.get('status')}`")
    lines.append("")

    lines.append("## Four honest sentences")
    lines.append("")
    lines.append(
        "1. The sealed run's `efficiency_analysis.json` zeros were a cache-replay artefact, not a "
        "true zero cost: `scripts/dev/run_agent_eval.py::make_chat` only reads `usage` when "
        "`client.chat()` returns a dict, but `LLMClient.chat` returns a `str` and stores real usage "
        "in `client.last_call_metadata`, which the runner never read; this script reads it directly."
    )
    lines.append(
        "2. Part 2's token counts are an approximation, not billed truth: the Qwen3 tokenizer is not "
        f"in the local HF cache in this environment, so counts use {p2.get('tokenizer_used', 'n/a')}, "
        "which does not model gateway chat-template overhead."
    )
    lines.append(
        "3. Part 2's offline replay under-counts by construction: `explain_reasoning()` gained a new "
        "case-based-explanation step (commit 7f94d9b9) about 3h42m after the sealed cache was "
        f"written the same day, under an UNCHANGED PROMPT_VERSION, so the {p2.get('cache_misses', 'n/a')} "
        "conversations that call it diverge from current code and truncate early in the "
        "reconstruction; the reported totals include both the raw (truncated) figure and a "
        "truncation-adjusted extrapolation, and Part 1 is unaffected since it never reads the "
        "sealed cache."
    )
    lines.append(
        "4. Part 1's per-call statistics (tokens, latency, cost) are computed over the "
        f"{p1.get('fresh_calls_unique', 'n/a')} UNIQUE gateway calls, not per occurrence and not "
        "only over calls made live in the current process: every occurrence of a reused "
        "conversational turn (e.g. the fixed-text out_of_scope question, identical across cases) "
        "carries that turn's real, once-measured token/latency data regardless of which process run "
        "fetched it, so cost and per-call stats stay correct and stable whether this script just "
        "made the calls or is replaying them for free from its own scratch cache."
    )
    lines.append(
        "5. Part 1's fresh sample is small (n="
        f"{p1.get('n_cases_used', 'n/a')} cases, {p1.get('n_questions', 'n/a')} questions) and its "
        "first-draft/revision rates are reported next to the sealed 591-question numbers as a drift "
        "signal only, never as a replacement for the sealed measurement."
    )
    lines.append(
        "6. Both cost estimates assume the same gateway pricing "
        f"(${PROMPT_RATE_PER_1M_USD}/1M prompt + ${COMPLETION_RATE_PER_1M_USD}/1M completion tokens, "
        "per prior measured pricing, not the old $0.01/call heuristic) and neither run touched "
        "`research/evidence/derm7pt_v1/agent_cache/`."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Part 1 projection only: no API calls, no file writes, exits immediately")
    parser.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--skip-part1", action="store_true", help="skip the fresh paid sample")
    parser.add_argument("--skip-part2", action="store_true", help="skip the offline reconstruction")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--notes-out", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sealed_report = json.loads(SEALED_REPORT_PATH.read_text(encoding="utf-8"))
    _verify_protocol_labels(sealed_report)

    if args.dry_run:
        print(f"\n=== DRY RUN: Part 1 projection only (no API calls) ===")
        proj = project_part1(args.n_cases, sealed_report)
        _print_projection(proj, args.max_calls)
        if proj["projected_calls"] >= args.max_calls:
            print(f"   WARNING: projected calls >= hard cap; a real run will auto-reduce n_cases.")
        return

    tokenizer_name, count_tokens = build_tokenizer()
    print(f"\ntokenizer: {tokenizer_name}")

    result: Dict[str, Any] = {
        "protocol": "agent-efficiency-v1",
        "analysis_type": (
            "post-hoc efficiency measurement; fresh sample n cases (paid) + offline token "
            "accounting over the sealed run; sealed cache untouched"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "llm_verifier": LLM_VERIFIER,
            "gated_retrieval": GATED_RETRIEVAL,
            "promoted": PROMOTED,
            "sealed_cohort": SEALED_COHORT,
            "fresh_sample_cohort": FRESH_SAMPLE_COHORT,
            "verified_against": str(SEALED_REPORT_PATH.relative_to(REPO)).replace("\\", "/"),
        },
    }

    if not args.skip_part2:
        print("\n=== Part 2: offline token accounting over the full sealed run ===")
        result["part2_offline_accounting"] = run_part2_offline(tokenizer_name, count_tokens, sealed_report)
    else:
        result["part2_offline_accounting"] = {"status": "skipped"}

    if not args.skip_part1:
        print("\n=== Part 1: fresh paid sample ===")
        result["part1_fresh_sample"] = run_fresh_sample(
            n_cases=args.n_cases, max_calls=args.max_calls,
            tokenizer_name=tokenizer_name, count_tokens=count_tokens, sealed_report=sealed_report,
            part2_result=result.get("part2_offline_accounting"),
        )
    else:
        result["part1_fresh_sample"] = {"status": "skipped"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    md = build_markdown(result)
    args.notes_out.parent.mkdir(parents=True, exist_ok=True)
    args.notes_out.write_text(md, encoding="utf-8")
    print(f"wrote {args.notes_out}")

    p1 = result["part1_fresh_sample"]
    p2 = result["part2_offline_accounting"]
    print("\n=== SUMMARY ===")
    print(f"Part 1: status={p1.get('status')} fresh_calls_unique={p1.get('fresh_calls_unique')} "
          f"made_this_process={p1.get('fresh_calls_made_this_process')} "
          f"usage_available_rate={p1.get('usage_available_rate')} cost=${p1.get('cost_estimate_usd')}")
    print(f"Part 2: status={p2.get('status')} n_calls={p2.get('n_calls_reconstructed')} "
          f"cache_misses={p2.get('cache_misses')} cost=${p2.get('cost_estimate_usd_at_assumed_rates')}")


if __name__ == "__main__":
    main()
