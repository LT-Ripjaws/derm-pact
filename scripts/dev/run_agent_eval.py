"""Run the clinician-agent evaluation against a real model, with an on-disk cache.

Text-only: the agent never sees an image, so this is far cheaper than the G1 assessor runs.
Every model reply is cached by a hash of the full message list plus the decoding parameters, so a
re-run costs nothing and reproduces exactly.
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

from medguard_v2.agent.clinician_agent import PROMPT_VERSION  # noqa: E402
from medguard_v2.eval.agent_eval import CACHE_DIR, OUT, run  # noqa: E402


def _digest(payload: Dict) -> str:
    """The hash both cache-key schemes share. Key IDENTITY is the payload dict alone, so the two
    schemes below differ only in what they put in it -- never in how it is serialised or hashed.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _old_cache_key(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    """The cache key as computed before F7 -- messages + decoding params only."""
    return _digest({"messages": messages, "model": model, "t": temperature, "m": max_tokens})


def _new_cache_key(
    messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int,
    *, prompt_version: str, llm_verifier: bool, gated_retrieval: bool,
) -> str:
    """F7: fold PROMPT_VERSION and the llm_verifier/gated_retrieval flag values into the hashed
    dict. The full message list already captures most behaviour changes (a different system
    prompt, a different tool observation), but a host-side-only behaviour change -- one that
    alters whether a REVISION turn gets appended, without changing the messages sent up to the
    point that decision is made -- would otherwise replay a completion generated under a
    different behavioural configuration as if it were current. PROMPT_VERSION is redundant with
    the system prompt text most of the time, but pinning it explicitly means a version bump is
    never silently absorbed by a coincidental prompt-text match.
    """
    return _digest({
        "messages": messages, "model": model, "t": temperature, "m": max_tokens,
        "prompt_version": prompt_version, "llm_verifier": llm_verifier,
        "gated_retrieval": gated_retrieval,
    })


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def make_chat(
    model: str, temperature: float, max_tokens: int, cache_dir: Path, stats: Dict,
    *, prompt_version: str, llm_verifier: bool, gated_retrieval: bool,
):
    from medguard import config
    from medguard.pipeline.llm import LLMClient

    # The config module loads the credential itself and reports only whether one exists; the
    # value is never read, printed or logged here.
    if not config.load_env():
        raise SystemExit(
            "No LLM gateway credential available. Set KILOCODE_API_KEY in the environment or "
            "populate medguard/.env, then re-run."
        )

    client = LLMClient(model=model)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def chat(messages: List[Dict[str, str]]) -> str:
        new_key = _new_cache_key(
            messages, model, temperature, max_tokens,
            prompt_version=prompt_version, llm_verifier=llm_verifier, gated_retrieval=gated_retrieval,
        )
        new_path = _cache_path(cache_dir, new_key)
        if new_path.exists():
            stats["cache_hits"] += 1
            return json.loads(new_path.read_text(encoding="utf-8"))["reply"]

        # Compatibility path -- this IS the durable migration, not a one-off patch. Caches
        # written before F7 used the narrower _old_cache_key. Read under the old key if present,
        # so the existing ~600-question v4 cache keeps replaying at zero cost, and write the hit
        # forward under the new key too: every replay migrates its own entry, so the cache
        # completes its own migration over the course of a run instead of needing a separate
        # one-off script.
        old_key = _old_cache_key(messages, model, temperature, max_tokens)
        old_path = _cache_path(cache_dir, old_key)
        if old_path.exists():
            stats["cache_hits"] += 1
            payload = old_path.read_text(encoding="utf-8")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(payload, encoding="utf-8")
            return json.loads(payload)["reply"]

        started = time.time()
        response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        reply = response["content"] if isinstance(response, dict) else str(response)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}

        stats["calls"] += 1
        stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        stats["latency_s"] += time.time() - started

        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(
            json.dumps({"reply": reply, "usage": usage, "model": model}, indent=2), encoding="utf-8"
        )
        return reply

    return chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--cohort", default="dev", choices=["dev", "report", "all"],
                        help="dev may be iterated on; report is scored once")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--llm-verifier", action="store_true",
                        help="enable the LLM verifier gate (default off; see "
                             "research/evidence/derm7pt_v1/llm_verifier_study_v1.json for the "
                             "catch-rate study that keeps it off by default)")
    parser.add_argument("--ungated-retrieval", action="store_true",
                        help="disable router-gated retrieval -- the ablation arm that reproduces "
                             "clinician-agent-v4's ungated search_guidelines. Default is gated (on).")
    parser.add_argument("--promoted", action="store_true",
                        help="build contracts from the safety-preserving promoted perception "
                             "(dermlip+metadata C=100 diagnosis + promoted criteria heads, with "
                             "per-case margins) instead of the base dermlip heads. Use with the "
                             "v5.4 PROMPT_VERSION so the cache regenerates against promoted contracts.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    gated_retrieval = not args.ungated_retrieval

    stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
    chat = make_chat(
        args.model, args.temperature, args.max_tokens, CACHE_DIR, stats,
        prompt_version=PROMPT_VERSION, llm_verifier=args.llm_verifier, gated_retrieval=gated_retrieval,
    )

    report = run(
        chat, limit=args.limit, model=args.model, cohort=args.cohort,
        llm_verifier=args.llm_verifier, gated_retrieval=gated_retrieval,
        promoted=args.promoted,
    )
    report["usage"] = {
        **stats,
        "latency_s": round(stats["latency_s"], 1),
        "note": "calls are paid; cache_hits are free replays",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    endpoints = report["endpoints"]
    print(f"model={report['model']}  cohort={report['cohort']}  cases={report['n_cases']}  questions={report['n_questions']}")
    print(f"usage: {stats['calls']} paid calls, {stats['cache_hits']} cached, "
          f"{stats['prompt_tokens']}+{stats['completion_tokens']} tokens")
    print()
    for name, block in endpoints.items():
        print(f"{name}:")
        for k, v in block.items():
            if k != "note":
                print(f"    {k}: {v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
