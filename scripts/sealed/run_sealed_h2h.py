"""The single confirmatory evaluation on the HAM10000 sealed test partition.

Deliberately a SEPARATE script rather than another ``--mode`` on the development
harness. A sealed run must not be one typo away from a development one.

Every registered sealed requirement is enforced here in code, not by discipline:

* the runtime lock is verified BEFORE anything else -- a drifted runtime would not
  measure what was frozen, so the run refuses;
* the abstention gate is LOADED BY PATH and never refit;
* case loading goes through the one-shot guard, so a second run raises;
* truth is loaded separately and later, under its own logged purpose, so the access
  log records that generation happened before any label was read;
* every arm is judged in ONE session, because the measured cross-session judge drift
  (up to 0.058 faithfulness / 0.067 context recall) does not support otherwise;
* the sealed access grant is written into the report, so the manuscript can quote the
  log entry that authorised the number it prints.

Usage (the confirmation flag is required and unabbreviated):

    python run_sealed_h2h.py --cache <dir> --out <path> \
        --gate research/evidence/dev_h2h_full/dev_h2h_full_160x11_v5.abstention_gate.json \
        --workers 6 --ragas-workers 6 --i-understand-this-runs-once
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", required=True,
                    help="frozen abstention gate JSON; LOADED, never refit")
    ap.add_argument("--lock", default="research/RUNTIME_LOCK_V1.json")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ragas-workers", type=int, default=6)
    ap.add_argument("--arms", default=None, help="default = all registered arms")
    ap.add_argument("--override-token", default=None,
                    help="required only to re-open a spent sealed purpose")
    ap.add_argument("--i-understand-this-runs-once", action="store_true",
                    help="required; the protocol allows the sealed test ONCE")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify every precondition and exit WITHOUT opening sealed access")
    ap.add_argument("--sealed-features",
                    default="research/experiments/medguard_h2h_v1/features/dermlip_sealed_v1",
                    help="frozen sealed QUERY feature artifact directory")
    ap.add_argument("--smoke", type=int, default=None,
                    help=("pipeline check on N sealed lesions. Logged under its OWN "
                          "purpose so it does NOT consume the confirmatory one, and it "
                          "produces no reportable result. Use a THROWAWAY --cache: the "
                          "confirmatory run must regenerate, not replay smoke output."))
    ap.add_argument("--resume", action="store_true",
                    help=("continue an interrupted confirmatory run. Requires the SAME "
                          "--cache: completed generations replay free from it. Logged "
                          "under sealed_h2h_resume, NOT as an override of the "
                          "confirmatory purpose, because continuing an evaluation is "
                          "not starting a second one. NOTE: the RAGAS judge has no "
                          "cache, so judging always restarts from scratch."))
    args = ap.parse_args()

    if args.smoke is not None and args.i_understand_this_runs_once:
        sys.exit("--smoke and --i-understand-this-runs-once are mutually exclusive: a "
                 "smoke test is not the confirmatory run and must not be labelled one.")
    if args.resume and args.smoke is not None:
        sys.exit("--resume and --smoke are mutually exclusive.")
    if args.resume and args.i_understand_this_runs_once:
        sys.exit("--resume does not need --i-understand-this-runs-once: the decision to "
                 "spend the sealed evaluation was already made and logged.")

    from medguard.eval.h2h_protocol import ArmId
    from medguard.eval.h2h_sealed import (
        EXPECTED_LESIONS, load_sealed_run_cases, load_sealed_truth,
        sealed_partition_summary, sealed_perception_status,
    )
    from medguard.eval.matched_full_h2h_run import DEFAULT_ARMS, run_h2h
    from medguard.eval.ragas_judge import judge_specification_sha256
    from medguard.eval.runtime_lock import verify_judge_specification, verify_lock
    from medguard.eval.selective_gate import CalibratedAbstentionGate

    # 1. The runtime must be exactly what was frozen, and so must the judge. Check both
    #    FIRST: everything after this costs money or spends the one shot.
    lock = json.loads((PROJECT_ROOT / args.lock).read_text(encoding="utf-8"))
    lock_state = verify_lock(PROJECT_ROOT, lock)
    verify_judge_specification(lock, judge_specification_sha256())
    print(f"[lock] VERIFIED {lock_state['source_files']} files "
          f"{lock_state['source_digest'][:16]}; judge specification unchanged")

    # 2. The gate is loaded, not fitted. Refitting on the sealed set would tune the
    #    operating point on the confirmatory data -- the exact thing a sealed test forbids.
    gate = CalibratedAbstentionGate.load(args.gate)
    print(f"[gate] loaded from {args.gate}: tau={gate.diagnostic_confidence_tau:.6f} "
          f"target_coverage={gate.target_answer_coverage} "
          f"(fitted on {gate.n_calibration} DEVELOPMENT confidences; NOT refit here)")

    summary = sealed_partition_summary()
    print(f"[partition] {summary['images']} images over {summary['lesions']} lesions")

    arms = (tuple(ArmId(a.strip()) for a in args.arms.split(",") if a.strip())
            if args.arms else DEFAULT_ARMS)
    print(f"[arms] {[a.value for a in arms]}")

    # The precondition most likely to be missing, checked WITHOUT opening the guard:
    # discovering the artifact is absent only after access was opened would waste the
    # single run.
    perception = sealed_perception_status()
    print(f"[perception] {perception}")

    if args.dry_run:
        if not perception["ready"]:
            sys.exit(f"[dry-run] NOT READY: {perception['reason']}")
        print("\n[dry-run] all preconditions verified; sealed access NOT opened.")
        return

    if not perception["ready"]:
        sys.exit(f"refusing: sealed perception artifact unusable -- {perception['reason']}")

    if args.smoke is None and not args.resume and not args.i_understand_this_runs_once:
        sys.exit("refusing: pass --i-understand-this-runs-once. The sealed test is "
                 "evaluated ONCE and this run cannot be taken back.")
    if args.smoke is not None:
        print(f"\n[SMOKE] pipeline check on {args.smoke} sealed lesion(s). This is NOT "
              f"the confirmatory run: it is logged under a separate purpose, it does not "
              f"consume the confirmatory one, and its output is NOT a result.")

    # 3. Paid generation is user-authorized per run; fail before spending if absent.
    from medguard import config
    if not config.load_env():
        sys.exit("no gateway API key after load_env(); aborting before any call")

    from medguard.eval.h2h_production import build_production_dependencies
    from medguard.eval.ragas_judge import RagasFaithfulnessJudge
    from medguard.pipeline.generator import VLMGenerator
    from medguard.pipeline.llm import LLMClient

    print("building production dependencies (heavy offline load)...")
    sealed_features = PROJECT_ROOT / args.sealed_features / "feature_manifest.json"
    if not sealed_features.is_file():
        sys.exit(f"refusing: sealed query features missing at {sealed_features}. "
                 f"Freeze them with scripts/dev/freeze_sealed_features.py -- the "
                 f"exemplar channel encodes the patient image through a frozen lookup "
                 f"that covers train+calibration only, so it cannot resolve a sealed "
                 f"image without them.")
    deps = build_production_dependencies(args.cache, sealed_feature_manifest=sealed_features)
    deps.abstention_gate = gate

    # 4. Sealed cases. This opens and logs the one-shot guard. It happens AFTER the
    #    dependencies build, so a build failure cannot burn the run.
    from medguard.eval.h2h_sealed import CONFIRMATORY_PURPOSE, SMOKE_PURPOSE
    if args.smoke is not None:
        purpose = SMOKE_PURPOSE
    elif args.resume:
        purpose = "sealed_h2h_resume"
    else:
        purpose = CONFIRMATORY_PURPOSE
    detail = (
        (f"PIPELINE SMOKE on {args.smoke} sealed lesion(s), arms {[a.value for a in arms]}, "
         f"cache {args.cache}. Not a result and not the confirmatory evaluation.")
        if args.smoke is not None else
        (f"RESUME of the interrupted confirmatory H2H from cache {args.cache}; completed "
         f"generations replay free and no second evaluation is authorised by this entry")
        if args.resume else
        (f"confirmatory H2H over the sealed HAM10000 test partition "
         f"({summary['images']} images / {summary['lesions']} lesions), {len(arms)} arms, "
         f"runtime lock {lock_state['source_digest'][:16]}, gate loaded by path from "
         f"{args.gate}")
    )
    cases, grant = load_sealed_run_cases(
        detail=detail, override_token=args.override_token,
        purpose=purpose, limit=args.smoke,
    )
    print(f"[sealed] access granted seq={grant['sequence']} "
          f"entry={grant['entry_sha256'][:16]}")

    missing = [p for c in cases for p in c.ordered_image_paths if not Path(p).exists()]
    if missing:
        sys.exit(f"{len(missing)} sealed image paths missing; aborting before generation")

    # The conditional belongs OUTSIDE the lambda: written inside, the factory is never
    # None and instead returns None per call, which run_h2h would hand to a worker.
    generator_factory = (
        (lambda: VLMGenerator(LLMClient(), max_tokens=1536)) if args.workers > 1 else None)
    # The judging phase has no natural checkpoint and a dropped connection discards
    # all of it. Cache it so an interruption costs minutes, not hours -- at the price
    # of mixing judging sessions, which is recorded and reported rather than hidden.
    from medguard.eval.ragas_judge import JudgeCache
    import datetime as _dt
    judge_session = _dt.datetime.now(_dt.timezone.utc).strftime("judge-%Y%m%dT%H%M%SZ")
    judge_cache = JudgeCache(Path(args.cache).parent / "judge_cache", judge_session)
    judge = RagasFaithfulnessJudge(
        LLMClient(model=config.JUDGE_MODEL), model=config.JUDGE_MODEL,
        max_tokens=2048, max_workers=args.ragas_workers, cache=judge_cache)
    print(f"[ragas] judge cache at {judge_cache.root} (session {judge_session})")
    judge.assert_matches_specification()
    print(f"[ragas] judge {judge.model} pinned to the frozen specification")

    import time
    started = time.time()

    def progress(done, total, elapsed):
        rate = done / (elapsed / 60.0) if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("nan")
        print(f"[progress] case {done}/{total} elapsed={elapsed/60:.1f}min "
              f"rate={rate:.2f}/min eta={eta:.1f}min", flush=True)

    # 5. Truth is joined out of band by the scorer, and its access is logged separately
    #    so the log shows generation preceded any label read.
    if args.smoke is not None:
        truth_purpose, truth_detail = "sealed_smoke_scoring", "label join for a PIPELINE SMOKE, not a result"
    elif args.resume:
        truth_purpose, truth_detail = "sealed_scoring_resume", "label join on RESUME of the interrupted confirmatory run"
    else:
        truth_purpose, truth_detail = "sealed_scoring", "out-of-band label join for the scorer, after sealed generation"
    truth = load_sealed_truth(
        detail=truth_detail, override_token=args.override_token, purpose=truth_purpose)

    report = run_h2h(
        cases, deps, truth, arms=arms, n_boot=args.nboot, output_path=args.out,
        ragas_judge=judge, max_workers=args.workers,
        generator_factory=generator_factory if args.workers > 1 else None,
        on_case_done=progress,
    )
    # run_h2h stamps development boundaries by default. On a sealed run those assertions
    # are FALSE, and a report claiming sealed_test_access=False while evaluating the
    # sealed set would be a provenance lie inside our own audit artifact. Overwrite them
    # truthfully rather than appending contradicting fields beside them.
    boundaries = report.setdefault("boundaries", {})
    boundaries["development_only"] = False
    boundaries["sealed_test_access"] = True
    boundaries["test_records_loaded"] = summary["images"]
    boundaries["partition"] = "ham10000_sealed_test"
    boundaries["lesions_evaluated"] = len(cases)
    boundaries["is_confirmatory"] = purpose in (CONFIRMATORY_PURPOSE, "sealed_h2h_resume")
    boundaries["resumed"] = args.resume
    boundaries["smoke_test"] = args.smoke is not None
    boundaries["sealed_access_grant"] = grant
    boundaries["runtime_lock_digest"] = lock_state["source_digest"]
    boundaries["abstention_gate_path"] = args.gate
    boundaries["abstention_gate_refit"] = False
    boundaries["sealed_query_features"] = str(sealed_features)
    boundaries["judge_session_id"] = judge_session
    boundaries["judge_cache_hits"] = judge_cache.hits
    boundaries["judge_cache_misses"] = judge_cache.misses
    # If judging spanned more than one session, the mixing must be visible: measured
    # cross-session drift on this judge reaches 0.058 faithfulness / 0.067 context recall.
    boundaries["judge_session_composition"] = judge_cache.session_composition()
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    scored = str(args.out).replace(".json", ".scored.json")
    Path(scored).write_text(
        json.dumps(report.get("scored", {}), indent=2, default=str), encoding="utf-8")

    print(f"\n[done] {(time.time() - started)/3600:.2f} h")
    print(f"wrote {args.out}\nwrote {scored}")
    print(f"[sealed] this partition is now spent; the access log records entry "
          f"{grant['entry_sha256'][:16]}")


if __name__ == "__main__":
    main()
