"""RG-003 evidence-influence probe: does retrieved evidence CAUSALLY change the answer?

Citation resolvability 1.0 proves cited records exist; it does NOT prove the evidence was
load-bearing. This probe measures causal influence: for guideline-grounded answers that cited
real records, re-run the SAME case with evidence SUPPRESSED (a NoEvidenceCatalog that returns
not-covered for every condition, so both the contract's pre-bound guideline AND the agent's
search_guidelines return nothing), and diff the answer. If the answer loses its management
specifics / citations / switches to an explicit absence disclosure, the evidence was load-bearing.

Deterministic aggregation makes the diff interpretable; the answer cache makes the baseline free.
Bounded to a seed-sampled subset to cap cost. Promoted (v5.4) perception.

Output: research/evidence/derm7pt_v1/evidence_influence_probe.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from medguard_v2.agent.clinician_agent import PROMPT_VERSION, run_agent
from medguard_v2.agent.verification import ADVICE_SPECIFICS
from medguard_v2.eval.agent_eval import _build_contracts, build_questions
from medguard_v2.evidence.guideline_binding import BindingResult, GuidelineCatalog
from run_agent_eval import make_chat  # reuse the exact cached chat client

REPORT = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v5_4.json"
OUT = REPO / "research/evidence/derm7pt_v1/evidence_influence_probe.json"
CACHE = REPO / "research/evidence/derm7pt_v1/agent_cache"
MODEL = "qwen/qwen3-vl-32b-instruct"
SEED = 20260814
N = 40


class NoEvidenceCatalog:
    """Wraps the real catalog but reports NO coverage for every condition -- the counterfactual
    where the licensed corpus is unavailable. bind() mirrors a genuinely-uncovered result."""

    def __init__(self, real: GuidelineCatalog):
        self._real = real

    def bind(self, diagnosis_group, **kw) -> BindingResult:
        return BindingResult(
            condition_group=diagnosis_group,
            catalog_condition=None,
            covered=False,
            recommendations=(),
            absence_reason=(
                f"[evidence-influence probe] guideline corpus suppressed for '{diagnosis_group}'."
            ),
        )

    def __getattr__(self, name):
        return getattr(self._real, name)


def _specifics(text: str):
    t = (text or "").lower()
    return {m for m in ADVICE_SPECIFICS if m in t}


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = {(r["case_id"], r["kind"]): r for r in report["rows"]}

    # Baseline evidence-using answers: guideline/why answers that cited >=1 real record.
    candidates = [
        r for r in report["rows"]
        if r["kind"] in ("guideline", "why")
        and r["answer_kind"] == "answer"
        and r["citations"]["cited"] > 0
    ]
    random.Random(SEED).shuffle(candidates)
    sample = candidates[:N]
    want = {(r["case_id"], r["kind"]) for r in sample}

    contracts = {c.case_id: c for c in _build_contracts(report["n_cases"], report["cohort"], promoted=True)}
    real_catalog = GuidelineCatalog.load()
    no_ev_catalog = NoEvidenceCatalog(real_catalog)

    stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
    chat = make_chat(MODEL, 0.0, 700, CACHE, stats,
                     prompt_version=PROMPT_VERSION + "+noevidence", llm_verifier=False, gated_retrieval=True)

    results = []
    for cid, contract in contracts.items():
        for q in build_questions(contract):
            key = (q.case_id, q.kind)
            if key not in want:
                continue
            base = rows[key]
            base_specifics = _specifics(base.get("answer", ""))
            base_cited = base["citations"]["cited"]

            # No-evidence contract: rebuild with the suppressed catalog so the pre-bound guideline
            # is gone too, then run the agent (its search_guidelines will also return not-covered).
            from medguard_v2.evidence.contract import build_contract
            ne_contract = build_contract(
                case_id=contract.case_id,
                classifier_diagnosis=contract.classifier_diagnosis,
                classifier_confidence=contract.classifier_confidence,
                criteria=dict(contract.criteria),
                catalog=no_ev_catalog,
                criterion_margins=getattr(contract, "criterion_margins", None),
            )
            ans = run_agent(contract=ne_contract, catalog=no_ev_catalog, question=q.text,
                            chat=chat, model=MODEL)
            ne_specifics = _specifics(ans.text or "")
            ne_cited = len(ans.citations)
            ne_absence = any(p in (ans.text or "").lower()
                             for p in ("no recommendation", "no guidance", "holds no", "corpus suppressed", "not available"))

            lost_specifics = sorted(base_specifics - ne_specifics)
            changed = bool(base_specifics and (ne_specifics != base_specifics or ne_cited == 0)) or \
                      (base_cited > 0 and ne_cited == 0) or (ans.kind != "answer")
            results.append({
                "case_id": q.case_id, "kind": q.kind,
                "baseline_cited": base_cited, "baseline_specifics": sorted(base_specifics),
                "no_evidence_kind": ans.kind, "no_evidence_cited": ne_cited,
                "no_evidence_specifics": sorted(ne_specifics), "no_evidence_states_absence": ne_absence,
                "lost_specifics": lost_specifics,
                "evidence_changed_answer": changed,
            })

    n = len(results)
    changed_n = sum(1 for r in results if r["evidence_changed_answer"])
    lost_any = sum(1 for r in results if r["lost_specifics"])
    became_absence = sum(1 for r in results if r["no_evidence_states_absence"] or r["no_evidence_kind"] != "answer")
    out = {
        "protocol": "evidence_influence_probe_v1_rg003",
        "perception": "promoted (v5.4)",
        "prompt_version": PROMPT_VERSION,
        "n": n,
        "evidence_changed_answer_rate": round(changed_n / n, 4) if n else None,
        "lost_specifics_rate": round(lost_any / n, 4) if n else None,
        "became_absence_or_refusal_rate": round(became_absence / n, 4) if n else None,
        "note": (
            "Counterfactual: same case with the guideline corpus SUPPRESSED (contract pre-bind + "
            "agent search both return not-covered). A changed answer => the retrieved evidence was "
            "load-bearing, not decorative. Turns citation-resolvability=1.0 into a causal claim."
        ),
        "usage": {**stats, "latency_s": round(stats["latency_s"], 1)},
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"n={n} | evidence_changed={out['evidence_changed_answer_rate']} | "
          f"lost_specifics={out['lost_specifics_rate']} | became_absence/refusal={out['became_absence_or_refusal_rate']}")
    print(f"paid calls={stats['calls']} cache_hits={stats['cache_hits']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
