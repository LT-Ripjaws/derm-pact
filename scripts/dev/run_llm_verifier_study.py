"""LLM verifier catch-rate study: what does a second-agent auditor add?

Measures ``medguard_v2.agent.verification.verify_answer`` (a second LLM call that audits
the primary clinician agent's answer against its decision contract) on the answers already
recorded in ``agent_eval_report_v4.json``. **No answers are regenerated** -- this only adds
one auditor call per selected row, on top of answers the primary agent already produced.

Ground truth comes from override rows' ``counterfactual`` block, which is host-computed exact
arithmetic (the aggregator recomputes the score under the clinician's stated correction; no
judge is involved). That lets the verifier's catch rate be scored as a real precision/recall
detector problem, not just an agreement statistic.

Row selection (~270, cap 320):
  1. ALL override rows with answer_kind == "answer"                              (193)
  2. ALL guideline rows with answer_kind == "answer" and citations.cited > 0       (27)
  3. A seed-20260813 random sample of 50 further answer rows (guideline-uncited + why)

NOTE on (2): the task brief that specified this study estimated "~72" for this bucket. 72 is
the report's own `endpoints.citation_resolvability.answers_with_citations` figure, which counts
answer rows WITH CITATIONS ACROSS ALL KINDS (override 39 + guideline 27 + why 6 = 72), not
guideline rows alone. Restricted to kind == "guideline" specifically, the true count is 27. This
script implements the literal, unambiguous filter as specified (kind == "guideline"); the
discrepancy is recorded in the output under `selection.note_on_guideline_count_discrepancy`.

Contract reconstruction mirrors ``scripts/dev/build_deterministic_verification.py`` exactly:
load_cases (valid split) -> _load_features -> fit_readers (train-fit) -> load_policy ->
build_contract per case, no receipt. This has been empirically checked against the batched
construction ``medguard_v2/eval/agent_eval.py::_build_contracts`` actually used to produce the
v4 report (which fits with a different literal random_state, 20260811 vs fit_readers' 20260810):
0/203 diagnosis-label mismatches, max confidence delta 4.44e-16 (identical after round(.,4)),
0/1421 criterion-value mismatches after the abstention policy. LogisticRegression's default
lbfgs solver is deterministic for this data regardless of random_state, so the two harnesses
reconstruct bit-identical contracts in every way that matters to verification.

Verifier responses are cached on disk under `research/evidence/derm7pt_v1/verifier_cache/` keyed
by sha256(case_id + question + answer_text + model), so re-runs replay for free.

Output: research/evidence/derm7pt_v1/llm_verifier_study_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.agent.verification import VerificationResult, deterministic_checks, verify_answer  # noqa: E402
from medguard_v2.data.derm7pt import load_cases  # noqa: E402
from medguard_v2.evidence.contract import DecisionContract, build_contract  # noqa: E402
from medguard_v2.evidence.guideline_binding import GuidelineCatalog  # noqa: E402
from medguard_v2.perception.criterion_abstention import load_policy  # noqa: E402
from medguard_v2.pipeline.run_cohort import _load_features, fit_readers  # noqa: E402

REPORT = REPO / "research" / "evidence" / "derm7pt_v1" / "agent_eval_report_v4.json"
CACHE_DIR = REPO / "research" / "evidence" / "derm7pt_v1" / "verifier_cache"
OUT = REPO / "research" / "evidence" / "derm7pt_v1" / "llm_verifier_study_v1.json"

SEED = 20260813
OTHER_SAMPLE_SIZE = 50
MAX_TOTAL = 320
PROTOCOL = "llm_verifier_catch_rate_v1"


# --------------------------------------------------------------------- selection


def select_rows(rows: List[dict]) -> Tuple[List[Tuple[int, dict, str]], dict]:
    """Return [(index, row, selection_reason), ...] plus a report of the selection logic."""
    override_idx = [
        i for i, r in enumerate(rows)
        if r.get("kind") == "override" and r.get("answer_kind") == "answer"
    ]
    guideline_cited_idx = [
        i for i, r in enumerate(rows)
        if r.get("kind") == "guideline" and r.get("answer_kind") == "answer"
        and (r.get("citations") or {}).get("cited", 0) > 0
    ]
    mandatory = set(override_idx) | set(guideline_cited_idx)

    pool_idx = [
        i for i, r in enumerate(rows)
        if r.get("answer_kind") == "answer" and i not in mandatory
    ]
    rng = random.Random(SEED)
    other_idx = sorted(rng.sample(pool_idx, min(OTHER_SAMPLE_SIZE, len(pool_idx))))

    selected = (
        [(i, rows[i], "override_all") for i in override_idx]
        + [(i, rows[i], "guideline_cited_all") for i in guideline_cited_idx]
        + [(i, rows[i], "other_random_sample") for i in other_idx]
    )

    if len(selected) > MAX_TOTAL:
        keep_other = max(0, MAX_TOTAL - len(override_idx) - len(guideline_cited_idx))
        selected = selected[: len(override_idx) + len(guideline_cited_idx)] + selected[
            len(override_idx) + len(guideline_cited_idx): len(override_idx) + len(guideline_cited_idx) + keep_other
        ]

    selection_report = {
        "override_answer_all": len(override_idx),
        "guideline_cited_answer_all": len(guideline_cited_idx),
        "other_pool_size": len(pool_idx),
        "other_answer_random_sample": len(other_idx),
        "total_selected": len(selected),
        "cap": MAX_TOTAL,
        "seed": SEED,
        "note_on_guideline_count_discrepancy": (
            "The task brief estimated ~72 for 'guideline rows with citations.cited>0'. That "
            "figure is the v4 report's own endpoints.citation_resolvability.answers_with_citations, "
            "which counts answer rows with citations across ALL kinds (override 39 + guideline 27 "
            "+ why 6 = 72). Restricted literally to kind=='guideline', the count is 27. This "
            "script uses the literal per-kind filter as specified."
        ),
    }
    return selected, selection_report


# --------------------------------------------------------------------- contracts


def build_contract_lookup():
    """Mirrors scripts/dev/build_deterministic_verification.py exactly."""
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    valid = [c for c in cases if c.split == "valid"]
    features = _load_features()
    diagnosis, heads = fit_readers(train, features)
    catalog = GuidelineCatalog.load()
    policy = load_policy()

    case_map = {f"derm7pt-{c.case_num}": c for c in valid}
    contract_cache: Dict[str, DecisionContract] = {}

    def get_contract(case_id: str) -> DecisionContract:
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

    return get_contract, catalog


# --------------------------------------------------------------------- verifier chat + cache


def _cache_key(case_id: str, question: str, answer_text: str, model: str) -> str:
    body = json.dumps(
        {"case_id": case_id, "question": question, "answer_text": answer_text, "model": model},
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def make_chat(client, temperature: float, max_tokens: int):
    def chat(messages: List[Dict[str, Any]]) -> str:
        return client.chat(messages, temperature=temperature, max_tokens=max_tokens)

    return chat


def cached_verify(
    *,
    contract: DecisionContract,
    question: str,
    answer_text: str,
    case_id: str,
    model: str,
    chat_fn,
    client,
    catalog: GuidelineCatalog,
    cache_dir: Path,
    stats: Dict[str, Any],
) -> Tuple[Optional[VerificationResult], bool, Optional[str]]:
    """Returns (result, from_cache, error). result is None iff error is not None."""
    key = _cache_key(case_id, question, answer_text, model)
    path = cache_dir / key[:2] / f"{key}.json"

    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        stats["cached_calls"] += 1
        return (
            VerificationResult(
                passed=bool(cached["passed"]),
                issues=tuple(cached.get("issues", [])),
                confidence=str(cached.get("confidence", "low")),
                raw_response=cached.get("raw_response", ""),
            ),
            True,
            None,
        )

    calls_before = len(client.call_log)
    try:
        started = time.time()
        result = verify_answer(
            contract=contract, question=question, answer_text=answer_text,
            chat=chat_fn, catalog=catalog,
        )
        elapsed = time.time() - started
        stats["paid_calls"] += 1
        stats["latency_s"] += elapsed

        # LLMClient.chat() returns the raw text only; usage/telemetry lives in call_log.
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        if len(client.call_log) > calls_before:
            meta = client.call_log[-1]
            usage = {
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "total_tokens": meta.get("total_tokens"),
            }
            for k in ("prompt_tokens", "completion_tokens"):
                if usage[k]:
                    stats[k] += usage[k]

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "passed": result.passed,
                    "issues": list(result.issues),
                    "confidence": result.confidence,
                    "raw_response": result.raw_response,
                    "model": model,
                    "elapsed_s": round(elapsed, 3),
                    "usage": usage,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result, False, None
    except Exception as e:  # noqa: BLE001 -- per-row isolation, recorded and skipped
        stats["errors"] += 1
        return None, False, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------- scoring helpers


def _confusion(pairs: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    """pairs = [(flagged, oracle_wrong), ...] -> counts + precision/recall/F1."""
    tp = sum(1 for f, o in pairs if f and o)
    fp = sum(1 for f, o in pairs if f and not o)
    fn = sum(1 for f, o in pairs if not f and o)
    tn = sum(1 for f, o in pairs if not f and not o)
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "base_rate_oracle_wrong": round(sum(1 for _, o in pairs if o) / len(pairs), 4) if pairs else None,
    }


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="truncate the final selected row list (debug/smoke-test only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print selection + cache-hit estimate; make no API calls")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["rows"]

    selected, selection_report = select_rows(rows)
    if args.limit is not None:
        selected = selected[: args.limit]
        selection_report["debug_limit_applied"] = args.limit

    get_contract, catalog = build_contract_lookup()

    # Fail fast, before any paid call, if a selected case_id cannot be reconstructed.
    valid_case_ids = set()
    for _, row, _ in selected:
        valid_case_ids.add(row["case_id"])
    unresolved = []
    for cid in valid_case_ids:
        try:
            get_contract(cid)
        except KeyError:
            unresolved.append(cid)
    if unresolved:
        raise SystemExit(f"{len(unresolved)} selected case_id(s) do not resolve in the valid "
                          f"split's case map: {unresolved[:10]}")

    if args.dry_run:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        would_call = 0
        would_hit = 0
        for _, row, _ in selected:
            key = _cache_key(row["case_id"], row["question"], row["answer"], args.model)
            path = args.cache_dir / key[:2] / f"{key}.json"
            if path.exists():
                would_hit += 1
            else:
                would_call += 1
        print(f"Selection: {selection_report}")
        print(f"Would call API: {would_call}, would replay from cache: {would_hit}")
        print(f"Estimated cost at ~$0.01/call: ~${would_call * 0.01:.2f}")
        return

    from medguard import config
    from medguard.pipeline.llm import LLMClient

    if not config.load_env():
        raise SystemExit(
            "No LLM gateway credential available. Set KILOCODE_API_KEY in the environment or "
            "populate medguard/.env, then re-run."
        )

    client = LLMClient(model=args.model)
    chat_fn = make_chat(client, args.temperature, args.max_tokens)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    stats = {"paid_calls": 0, "cached_calls": 0, "errors": 0, "latency_s": 0.0,
              "prompt_tokens": 0, "completion_tokens": 0}
    per_row: List[Dict[str, Any]] = []
    t0 = time.time()

    for n, (idx, row, reason) in enumerate(selected):
        case_id = row["case_id"]
        kind = row["kind"]
        question = row["question"]
        answer_text = row.get("answer", "")

        if not answer_text.strip():
            per_row.append({
                "case_id": case_id, "kind": kind, "selection_reason": reason,
                "skipped": True, "reason": "empty answer text",
            })
            continue

        contract = get_contract(case_id)

        det_issues = deterministic_checks(contract, answer_text, catalog)

        result, from_cache, error = cached_verify(
            contract=contract, question=question, answer_text=answer_text, case_id=case_id,
            model=args.model, chat_fn=chat_fn, client=client, catalog=catalog,
            cache_dir=args.cache_dir, stats=stats,
        )

        cf = row.get("counterfactual")
        oracle = None
        if kind == "override" and cf is not None:
            oracle = {
                "available": True,
                "correct": bool(cf.get("correct")),
                "states_correct_score": cf.get("states_correct_score"),
                "asserts_a_wrong_score": cf.get("asserts_a_wrong_score"),
                "host_score_before": cf.get("host_score_before"),
                "host_score_after": cf.get("host_score_after"),
                "host_verdict_before": cf.get("host_verdict_before"),
                "host_verdict_after": cf.get("host_verdict_after"),
            }

        record: Dict[str, Any] = {
            "case_id": case_id,
            "kind": kind,
            "selection_reason": reason,
            "question_snippet": question[:100],
            "deterministic": {"issues": det_issues, "n_issues": len(det_issues),
                              "flagged": len(det_issues) > 0},
            "oracle": oracle,
        }
        if error is not None:
            record["verifier"] = {"error": error}
        else:
            unparseable = (
                len(result.issues) == 1
                and "unparseable" in result.issues[0].lower()
            )
            # Well-formed per the prompt contract is passed<->empty-issues in lockstep. Flag the
            # cases where a syntactically valid JSON reply is semantically self-contradictory
            # (e.g. passed=false with no stated issue) -- distinct from outright unparseable.
            inconsistent = (not unparseable) and (
                (result.passed and len(result.issues) > 0)
                or (not result.passed and len(result.issues) == 0)
            )
            record["verifier"] = {
                "passed": result.passed,
                "issues": list(result.issues),
                "confidence": result.confidence,
                "flagged": not result.passed,
                "from_cache": from_cache,
                "unparseable": unparseable,
                "internally_inconsistent": inconsistent,
            }
        per_row.append(record)

        done = n + 1
        if done % 25 == 0 or done == len(selected):
            elapsed = time.time() - t0
            print(f"  [{done}/{len(selected)}] {elapsed:.0f}s  paid={stats['paid_calls']} "
                  f"cached={stats['cached_calls']} errors={stats['errors']}")

    # ------------------------------------------------------------- scoring

    override_pairs_verifier: List[Tuple[bool, bool]] = []
    override_pairs_det_matched: List[Tuple[bool, bool]] = []
    override_pairs_det_all: List[Tuple[bool, bool]] = []

    both = verifier_only = deterministic_only = neither = 0
    n_overlap_considered = 0

    verifier_unparseable_ids: List[str] = []
    verifier_inconsistent_ids: List[str] = []
    verifier_error_rows: List[Dict[str, str]] = []

    for r in per_row:
        if r.get("skipped"):
            continue
        det_flag = r["deterministic"]["flagged"]
        oracle = r.get("oracle")

        if oracle is not None:
            override_pairs_det_all.append((det_flag, not oracle["correct"]))

        v = r["verifier"]
        if "error" in v:
            verifier_error_rows.append({"case_id": r["case_id"], "kind": r["kind"], "error": v["error"]})
            continue

        if v.get("unparseable"):
            verifier_unparseable_ids.append(r["case_id"])
        if v.get("internally_inconsistent"):
            verifier_inconsistent_ids.append(r["case_id"])

        v_flag = v["flagged"]

        n_overlap_considered += 1
        if v_flag and det_flag:
            both += 1
        elif v_flag and not det_flag:
            verifier_only += 1
        elif det_flag and not v_flag:
            deterministic_only += 1
        else:
            neither += 1

        if oracle is not None:
            oracle_wrong = not oracle["correct"]
            override_pairs_verifier.append((v_flag, oracle_wrong))
            override_pairs_det_matched.append((det_flag, oracle_wrong))

    confusion_verifier = _confusion(override_pairs_verifier)
    confusion_det_matched = _confusion(override_pairs_det_matched)
    confusion_det_all = _confusion(override_pairs_det_all)

    output = {
        "protocol": PROTOCOL,
        "model": args.model,
        "temperature": args.temperature,
        "source_report": str(args.report.relative_to(REPO)),
        "seed": SEED,
        "selection": selection_report,
        "contract_reconstruction": {
            "approach": (
                "mirrors scripts/dev/build_deterministic_verification.py exactly: load_cases "
                "(valid split) -> _load_features -> fit_readers(train, features) [SEED=20260810, "
                "BASE_C=10.0, from medguard_v2.pipeline.run_cohort] -> load_policy -> "
                "build_contract per case (no receipt)."
            ),
            "fidelity_check": (
                "Empirically verified against medguard_v2/eval/agent_eval.py::_build_contracts "
                "(SEED=20260811, batched) which actually produced the v4 report: 0/203 diagnosis "
                "label mismatches, max classifier-confidence delta 4.44e-16 (identical after "
                "round(.,4)), 0/1421 per-case criterion-value mismatches after the abstention "
                "policy. LogisticRegression's default lbfgs solver is deterministic for this data "
                "regardless of random_state, so the reconstructed contracts used for verification "
                "here match what the primary agent actually saw when it produced these answers."
            ),
        },
        "usage": {
            "paid_calls": stats["paid_calls"],
            "cached_calls": stats["cached_calls"],
            "errors": stats["errors"],
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "latency_s": round(stats["latency_s"], 1),
            "elapsed_s": round(time.time() - t0, 1),
            "note": "paid_calls are real LLM calls this run made; cached_calls replayed free from "
                    "research/evidence/derm7pt_v1/verifier_cache/. Token counts are only as "
                    "reliable as the gateway's reported usage field for this model/provider.",
        },
        "confusion_matrix_llm_verifier_vs_oracle": confusion_verifier,
        "confusion_matrix_deterministic_vs_oracle_matched_subset": confusion_det_matched,
        "confusion_matrix_deterministic_vs_oracle_all_override_rows": confusion_det_all,
        "overlap_all_studied_rows": {
            "n": n_overlap_considered,
            "both_flagged": both,
            "verifier_only": verifier_only,
            "deterministic_only": deterministic_only,
            "neither_flagged": neither,
        },
        "anomalies": {
            "verifier_unparseable_count": len(verifier_unparseable_ids),
            "verifier_unparseable_case_ids": verifier_unparseable_ids,
            "verifier_internally_inconsistent_count": len(verifier_inconsistent_ids),
            "verifier_internally_inconsistent_case_ids": verifier_inconsistent_ids,
            "verifier_api_errors_count": len(verifier_error_rows),
            "verifier_api_errors": verifier_error_rows,
        },
        "per_row": per_row,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- summary print

    print()
    print(f"wrote {args.out}")
    print(f"selected {selection_report['total_selected']} rows "
          f"(override={selection_report['override_answer_all']}, "
          f"guideline_cited={selection_report['guideline_cited_answer_all']}, "
          f"other_sample={selection_report['other_answer_random_sample']})")
    print(f"paid calls: {stats['paid_calls']}   cached: {stats['cached_calls']}   "
          f"errors: {stats['errors']}")
    print(f"tokens: {stats['prompt_tokens']} prompt + {stats['completion_tokens']} completion "
          f"({stats['prompt_tokens'] + stats['completion_tokens']} total)")
    print()
    print(f"{'detector':<32}{'n':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
          f"{'precision':>11}{'recall':>9}{'F1':>7}")
    for name, cm in (
        ("LLM verifier (matched subset)", confusion_verifier),
        ("deterministic (matched subset)", confusion_det_matched),
        ("deterministic (all override)", confusion_det_all),
    ):
        print(f"{name:<32}{cm['n']:>5}{cm['tp']:>5}{cm['fp']:>5}{cm['fn']:>5}{cm['tn']:>5}"
              f"{str(cm['precision']):>11}{str(cm['recall']):>9}{str(cm['f1']):>7}")
    print()
    print(f"overlap (all {n_overlap_considered} studied rows w/ a verifier result): "
          f"both={both} verifier_only={verifier_only} deterministic_only={deterministic_only} "
          f"neither={neither}")
    if verifier_unparseable_ids:
        print(f"unparseable verifier outputs: {len(verifier_unparseable_ids)} "
              f"{verifier_unparseable_ids[:10]}")
    if verifier_inconsistent_ids:
        print(f"internally-inconsistent verifier outputs: {len(verifier_inconsistent_ids)} "
              f"{verifier_inconsistent_ids[:10]}")
    if verifier_error_rows:
        print(f"verifier API errors: {len(verifier_error_rows)} (see anomalies.verifier_api_errors)")


if __name__ == "__main__":
    main()
