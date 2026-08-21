"""Development H2H driver harness.

Modes:
  dry    -- build the REAL frozen-artifact production dependencies but inject a FAKE
            generator; run a few real dev cases. ZERO paid calls. Validates the
            untested h2h_production assembly end to end (runtime load, concept links,
            guideline catalog, real retrieval, all 11 arms, scorers).
  sanity -- REAL generator (config.load_env, paid) on --n dev lesions. Verifies M
            coverage + records real token usage before the full run.
  full   -- REAL generator on all 160 dev lesions; the paid RESULT run.

Prints a compact per-arm summary and, for paid modes, the aggregate token usage so
cost can be computed from the gateway's per-token price.

Usage:
  python run_dev_h2h.py --mode dry    --cache <dir> --n 3
  python run_dev_h2h.py --mode sanity --cache <dir> --n 10 --out <path> --nboot 500
  python run_dev_h2h.py --mode full   --cache <dir> --out <path> --nboot 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve()
# project root is E:\Studies\Research\MedGuard-RAG-Journal
PROJECT_ROOT = Path("E:/Studies/Research/MedGuard-RAG-Journal")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_fake_generator():
    """A VLMGenerator wrapping a scripted client -> valid, PASSing contracts, no cost.

    The contract branch parses the runner's host-findings block from the on-wire
    prompt and echoes the first finding id, so bind_evidence_contract accepts it and
    the full generate->cache->verify->score path (the smoke's bug #2/#3 surface) runs.
    """
    from medguard.pipeline.generator import VLMGenerator

    class FakeLLMClient:
        def __init__(self):
            self.model = "fake/vlm-32b"
            self.call_log = []

        def vision(self, prompt, image_path=None, *, system="", max_tokens=512,
                   temperature=0.2, seed=None):
            self.call_log.append({
                "succeeded": True, "usage_available": True, "prompt_tokens": 5,
                "completion_tokens": 7, "total_tokens": 12, "latency_seconds": 0.01,
                "seed": seed,
            })
            if "single most likely diagnosis" in prompt:
                return json.dumps({"diagnosis": "melanoma", "rationale": "asymmetric network"})
            if "EvidenceContract" in prompt:
                findings = []
                parts = prompt.split("image_finding_ids): ", 1)
                if len(parts) == 2:
                    seg = parts[1].split("\nHost committed diagnosis:", 1)[0]
                    try:
                        findings = json.loads(seg)
                    except Exception:
                        findings = []
                claims = []
                if findings:
                    f = findings[0]
                    claims = [{
                        "claim_id": "c1", "claim_type": "diagnosis",
                        "text": "The lesion is most consistent with the committed diagnosis.",
                        "image_support": "supports",
                        "target_concept": f.get("concept_code", ""),
                        "target_state": f.get("state", "present"),
                        "image_finding_ids": [f.get("finding_id", "")],
                        "evidence_ids": [], "entailment": "not_assessed",
                        "contradiction": False,
                    }]
                return json.dumps({"claims": claims, "unresolved_uncertainties": []})
            return "Grounded answer consistent with the committed diagnosis."

    return VLMGenerator(FakeLLMClient(), max_tokens=1536, max_schema_repairs=1, seed=0)


def summarize(report, cases, truth=None):
    print("\n==== SUMMARY ====")
    b = report["boundaries"]
    print(f"cases={report['aggregate']['cases']} arm_rows={report['aggregate']['arm_rows']} "
          f"generation_calls={b['generation_calls']} "
          f"dev_only={b['development_only']} sealed_access={b['sealed_test_access']} "
          f"truth_in_inputs={b['truth_fields_in_arm_inputs']}")

    # Per-arm coverage roll-up straight from the audit cases.
    arms = report["arms"]
    tally = {a: {"answered": 0, "abstain": 0, "err": 0, "calls": 0,
                 "ptok": 0, "ctok": 0, "ttok": 0, "cache_hits": 0} for a in arms}
    m_pass = m_total = 0
    for case in report["cases"]:
        for a, row in case["arms"].items():
            g = row.get("generation", {}) or {}
            fa = row.get("final_action", "")
            t = tally[a]
            t["calls"] += int(g.get("remote_calls", 0) or 0)
            t["ptok"] += int(g.get("prompt_tokens", 0) or 0)
            t["ctok"] += int(g.get("completion_tokens", 0) or 0)
            t["ttok"] += int(g.get("total_tokens", 0) or 0)
            t["cache_hits"] += 1 if g.get("cache_hit") else 0
            has_err = bool(row.get("generation_error"))
            if has_err:
                t["err"] += 1
            if a == "M":
                m_total += 1
                if fa in ("PASS", "ANSWER_UNGATED"):
                    m_pass += 1
            # Contract arms emit final_action; B0/baselines don't -> infer from calls.
            if fa in ("PASS", "ANSWER_UNGATED"):
                t["answered"] += 1
            elif fa == "ABSTAIN":
                t["abstain"] += 1
            elif not fa:  # B0 / baseline row: answered iff it produced a call w/o error
                if (int(g.get("remote_calls", 0) or 0) > 0 or g.get("cache_hit")) and not has_err:
                    t["answered"] += 1
    print(f"\nM PASS coverage: {m_pass}/{m_total}")
    print(f"{'arm':>4} {'ans':>4} {'abst':>4} {'err':>4} {'calls':>6} "
          f"{'ptok':>8} {'ctok':>8} {'ttok':>8} {'cacheHit':>8}")
    for a in arms:
        t = tally[a]
        print(f"{a:>4} {t['answered']:>4} {t['abstain']:>4} {t['err']:>4} {t['calls']:>6} "
              f"{t['ptok']:>8} {t['ctok']:>8} {t['ttok']:>8} {t['cache_hits']:>8}")

    tot_p = sum(t["ptok"] for t in tally.values())
    tot_c = sum(t["ctok"] for t in tally.values())
    tot_t = sum(t["ttok"] for t in tally.values())
    tot_calls = sum(t["calls"] for t in tally.values())
    print(f"\nTOTAL remote_calls={tot_calls} prompt_tokens={tot_p} "
          f"completion_tokens={tot_c} total_tokens={tot_t}")

    # A quick B0-vs-M diagnosis peek + any generation errors surfaced.
    errs = []
    for case in report["cases"]:
        for a, row in case["arms"].items():
            if row.get("generation_error"):
                errs.append((case["cluster_id"], a, row["generation_error"]))
    if errs:
        print("\nGENERATION ERRORS:")
        for cid, a, e in errs[:20]:
            print(f"  lesion {cid} arm {a}: {e}")

    # B0 (VLM) diagnosis vs truth vs host commitment -- the competed-diagnosis signal.
    if truth is not None:
        print("\nB0-VLM diagnosis vs truth vs host-committed (M-family probe):")
        print(f"  {'lesion':>10} {'truth':>10} {'host':>10} {'B0':>10}  match")
        b0_correct = host_correct = n = 0
        for case in report["cases"]:
            cid = case["cluster_id"]
            gold = truth.get(str(cid), "?")
            host = case.get("committed_diagnosis", "?")
            b0row = case["arms"].get("B0", {})
            b0 = b0row.get("predicted_diagnosis", "") or "(none)"
            n += 1
            b0_correct += 1 if b0 == gold else 0
            host_correct += 1 if host == gold else 0
            flag = "ok" if b0 == gold else "MISS"
            print(f"  {cid:>10} {gold:>10} {host:>10} {b0:>10}  {flag}")
        if n:
            print(f"  B0 acc={b0_correct}/{n}  host(probe) acc={host_correct}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "sanity", "full"], required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm ids (e.g. 'M' or 'M,A0,A1'); default = all")
    ap.add_argument("--ragas", action="store_true",
                    help="wire the real RAGAS faithfulness judge (paid; sanity/full only)")
    ap.add_argument("--abstain-coverage", type=float, default=None,
                    help="fit a label-free calibrated abstention gate to answer this "
                         "fraction of lesions (e.g. 0.90); omit to disable the gate")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel generation workers over lesions (the gateway is "
                         "~20s/call and sequential; >1 fans cases out over threads, "
                         "result-invariant). Each worker gets its own LLM client.")
    ap.add_argument("--ragas-workers", type=int, default=1,
                    help="parallel workers for the RAGAS judge's per-sample scoring "
                         "(paid; sanity/full only). Defaults to --workers if unset via 0.")
    args = ap.parse_args()

    from medguard.eval.h2h_production import (
        build_production_dependencies,
        load_development_run_cases,
        load_development_truth,
    )
    from medguard.eval.h2h_protocol import ArmId
    from medguard.eval.matched_full_h2h_run import run_h2h, DEFAULT_ARMS

    if args.arms:
        arms = tuple(ArmId(a.strip()) for a in args.arms.split(",") if a.strip())
    else:
        arms = DEFAULT_ARMS

    gen = None
    ragas_judge = None
    generator_factory = None  # per-thread generators when --workers > 1
    ragas_workers = args.ragas_workers if args.ragas_workers > 0 else args.workers
    if args.mode == "dry":
        gen = build_fake_generator()
        print("[dry] fake generator -> ZERO paid calls")
        if args.workers > 1:
            generator_factory = build_fake_generator  # each thread gets its own fake
        if args.ragas:
            print("[dry] --ragas ignored (judge calls are paid; use sanity/full)")
    else:
        from medguard import config
        ok = config.load_env()  # user-authorized; value never surfaced
        print(f"[{args.mode}] api_key_present={ok}")
        if not ok:
            sys.exit("no gateway API key present after load_env(); aborting before any call")
        if args.workers > 1:
            # Factory must mirror build_production_dependencies' generator exactly
            # (VLMGenerator(LLMClient(), max_tokens=1536)) so every worker shares the
            # model identity the cache namespace binds.
            from medguard.pipeline.llm import LLMClient
            from medguard.pipeline.generator import VLMGenerator
            generator_factory = lambda: VLMGenerator(LLMClient(), max_tokens=1536)
            print(f"[parallel] generation workers = {args.workers} "
                  f"(each with its own LLM client)")
        if args.ragas:
            from medguard.pipeline.llm import LLMClient
            from medguard.eval.ragas_judge import RagasFaithfulnessJudge
            ragas_judge = RagasFaithfulnessJudge(
                LLMClient(model=config.JUDGE_MODEL), model=config.JUDGE_MODEL,
                max_tokens=2048, max_workers=ragas_workers)
            print(f"[ragas] judge model = {ragas_judge.model} (paid); "
                  f"workers = {ragas_workers}")

    cases = load_development_run_cases()
    truth = load_development_truth()
    print(f"loaded cases={len(cases)} truth_labels={len(truth)}")

    # Verify image paths on disk for the cases we will actually run.
    subset = cases if args.mode == "full" else cases[: args.n]
    missing = 0
    for c in subset:
        for p in c.ordered_image_paths:
            if not Path(p).exists():
                missing += 1
                if missing <= 5:
                    print(f"  MISSING IMAGE: lesion {c.cluster_id}: {p}")
    print(f"running {len(subset)} lesions; missing_images={missing}")
    if missing:
        sys.exit("image paths missing; aborting")

    print("building production dependencies (heavy offline load)...")
    deps = build_production_dependencies(args.cache, generator=gen)

    # Calibrated selective abstention (Fix 4): fit the diagnostic-abstention tau on the
    # FULL dev confidence distribution, label-free (quantiles only, no truth), so the
    # threshold is stable regardless of the run subset. Save it for a sealed reuse.
    if args.abstain_coverage is not None:
        from medguard.eval.selective_gate import fit_diagnostic_abstention_gate
        confidences = [c.perception.commitment.confidence for c in cases]
        gate = fit_diagnostic_abstention_gate(
            confidences, target_answer_coverage=args.abstain_coverage)
        deps.abstention_gate = gate
        print(f"[abstain] gate fitted on {gate.n_calibration} dev confidences: "
              f"tau={gate.diagnostic_confidence_tau:.4f} "
              f"target_answer_coverage={gate.target_answer_coverage}")
        if args.out:
            gate_path = str(args.out).replace(".json", ".abstention_gate.json")
            gate.save(gate_path)
            print(f"[abstain] wrote gate -> {gate_path}")
    print("deps built:",
          f"concept_links={len(deps.concept_links)}",
          f"corpus_receipt={deps.corpus_receipt[:12]}",
          f"perception_receipt={deps.perception_receipt[:12]}",
          f"final_k={deps.final_k} text_k={deps.text_candidate_k} image_k={deps.image_candidate_k}")

    import time as _time

    def _progress(done, total, elapsed):
        rate = done / (elapsed / 60.0) if elapsed > 0 else 0.0
        eta_min = (total - done) / rate if rate > 0 else float("nan")
        print(f"[progress] case {done}/{total}  elapsed={elapsed/60.0:.1f}min  "
              f"rate={rate:.2f} lesions/min  eta={eta_min:.1f}min", flush=True)

    print(f"generating {len(subset)} lesions x {len(arms)} arms "
          f"(workers={args.workers})...", flush=True)
    report = run_h2h(subset, deps, truth, arms=arms,
                     n_boot=args.nboot, output_path=args.out, ragas_judge=ragas_judge,
                     max_workers=args.workers, generator_factory=generator_factory,
                     on_case_done=_progress)
    summarize(report, subset, truth)

    # Scored provenance / process / diagnosis blocks (host-deterministic for M-family).
    per_arm = report.get("scored", {}).get("per_arm", {})
    print("\n==== SCORED (per arm) ====")
    for a in report["arms"]:
        blk = per_arm.get(a)
        if not blk:
            continue
        diag = blk.get("diagnosis", {})
        prov = blk.get("provenance_citation_integrity", {})
        proc = blk.get("process_contract_completion", {})
        trust = blk.get("process_trust", {})
        sel = blk.get("selective", {})
        act = blk.get("actionable", {})
        print(f"\n[{a}] cases={blk.get('cases')}")
        align = blk.get("alignment", {})
        print(f"  diagnosis: acc={diag.get('accuracy')} macroF1={diag.get('macro_f1')} "
              f"balAcc={diag.get('balanced_accuracy')} mcc={diag.get('mcc')} "
              f"kappa={diag.get('cohen_kappa')}")
        print(f"  provenance_citation_integrity: {json.dumps(prov, default=str)}")
        print(f"  process_contract_completion:   {json.dumps(proc, default=str)}")
        print(f"  process_trust (P1-P8):         {json.dumps(trust, default=str)}")
        print(f"  selective: aurc={sel.get('aurc')} coverage={sel.get('coverage')} "
              f"answered_accuracy={sel.get('answered_accuracy')} ece={sel.get('ece')}")
        print(f"  actionable(binary mal/benign): acc={act.get('accuracy')} "
              f"sens={act.get('sensitivity')} spec={act.get('specificity')} "
              f"auroc={act.get('auroc')} sens@spec={act.get('sensitivity_at_fixed_specificity')}")
        print(f"  alignment: {json.dumps(align, default=str)}")
        rag = blk.get("ragas_supporting", {})
        rag_scores = rag.get("scores") if isinstance(rag.get("scores"), dict) else None
        faith = (rag_scores or {}).get("faithfulness") if rag_scores else None
        ar = (rag_scores or {}).get("answer_relevancy") if rag_scores else None
        cr = (rag_scores or {}).get("context_recall") if rag_scores else None
        print(f"  ragas: status={rag.get('status')} faithfulness={faith}")
        print(f"         answer_relevancy={ar}")
        print(f"         context_recall={cr}")

    # Coverage / abstention for contract arms, straight from the audit rows.
    for a in report["arms"]:
        actions = [c["arms"][a].get("final_action") for c in report["cases"] if a in c["arms"]]
        from collections import Counter
        passed = sum(1 for x in actions if x in ("PASS", "ANSWER_UNGATED"))
        print(f"[{a}] coverage(PASS)={passed}/{len(actions)}  actions={dict(Counter(actions))}")

    if args.out:
        # also dump just the scored block for easy reuse
        scored_path = str(args.out).replace(".json", ".scored.json")
        Path(scored_path).write_text(json.dumps(report.get("scored", {}), indent=2, default=str), encoding="utf-8")
        print(f"\nwrote report -> {args.out}\nwrote scored -> {scored_path}")


if __name__ == "__main__":
    main()
