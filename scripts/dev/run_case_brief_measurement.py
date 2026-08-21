"""Measure generation-under-verification on the real model: 203 case briefs.

The `generate_case_brief` module (evidence/summary.py) is built and unit-tested with
fake chat, but has never run against the real model -- so the paper's "generation under
deterministic verification" claim has code but no measured numbers. This closes that:
for every promoted valid contract, generate a brief with qwen3-vl-32b, and record the
verification outcome. The measured claim is the GATE, not a judge score: X% of briefs
verify against the decision record (Y% first-try), Z% fall back to the deterministic
headline, and unverified prose is never shown by construction.

Output: research/evidence/derm7pt_v1/case_brief_measurement.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from medguard_v2.agent.clinician_agent import PROMPT_VERSION
from medguard_v2.eval.agent_eval import _build_contracts
from medguard_v2.evidence.summary import generate_case_brief
from run_agent_eval import make_chat

OUT = REPO / "research/evidence/derm7pt_v1/case_brief_measurement.json"
CACHE = REPO / "research/evidence/derm7pt_v1/agent_cache"
MODEL = "qwen/qwen3-vl-32b-instruct"


def main() -> None:
    contracts = _build_contracts(203, "all", promoted=True)
    stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
    chat = make_chat(
        MODEL, 0.0, 400, CACHE, stats,
        prompt_version=PROMPT_VERSION + "+casebrief", llm_verifier=False, gated_retrieval=True,
    )

    rows = []
    issue_counter: Counter = Counter()
    for i, contract in enumerate(contracts):
        brief = generate_case_brief(contract, chat)
        for issue in brief.issues:
            issue_counter[issue.split(":")[0][:60]] += 1
        rows.append({
            "case_id": contract.case_id,
            "verified": brief.verified,
            "attempts": brief.attempts,
            "fallback_used": brief.fallback_used,
            "n_issues": len(brief.issues),
            "issues": list(brief.issues),
            "chars": len(brief.text or ""),
            "text": brief.text,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/203] paid={stats['calls']} cached={stats['cache_hits']}")

    n = len(rows)
    verified = [r for r in rows if r["verified"]]
    first_try = [r for r in verified if r["attempts"] == 1]
    fallback = [r for r in rows if r["fallback_used"]]
    out = {
        "protocol": "case_brief_measurement_v1",
        "perception": "promoted (v5.4 contracts)",
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "n": n,
        "verified_rate": round(len(verified) / n, 4),
        "verified_first_try_rate": round(len(first_try) / n, 4),
        "verified_after_revision_rate": round((len(verified) - len(first_try)) / n, 4),
        "fallback_rate": round(len(fallback) / n, 4),
        "mean_attempts": round(sum(r["attempts"] for r in rows) / n, 3),
        "mean_chars_verified": round(
            sum(r["chars"] for r in verified) / max(1, len(verified)), 1
        ),
        # CaseBrief carries only the FINAL issue list (empty on a verified brief); first-draft
        # issue text is not retained by the dataclass, so revisions are visible via attempts>1,
        # not via issue types. Labelled accordingly -- do not read this as first-draft data.
        "issue_types_final": dict(issue_counter),
        "guarantee_note": (
            "The measured claim is the gate: a brief is shown only if it passes the same "
            "deterministic host checks the agent faces (score quotes, criterion values, "
            "citation/specifics traceability, action statement); on double failure the "
            "deterministic headline renders instead. Unverified generated prose is never "
            "displayed, by construction."
        ),
        "usage": {**stats, "latency_s": round(stats["latency_s"], 1)},
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"n={n} verified={out['verified_rate']} (first-try {out['verified_first_try_rate']}, "
          f"post-revision {out['verified_after_revision_rate']}) fallback={out['fallback_rate']}")
    print(f"issues: {dict(issue_counter)}")
    print(f"paid={stats['calls']} cached={stats['cache_hits']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
