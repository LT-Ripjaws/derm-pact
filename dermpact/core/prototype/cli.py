"""WP-D: ``python -m dermpact.core.prototype`` argparse CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dermpact.core.episode import (
    acknowledge_review, confirm_outcome, derive_state, mark_unresolved, resolve as episode_resolve,
)
from dermpact.core.inference_client import WorkerCrashed, WorkerRequestError, WorkerTimeout
from dermpact.core.prototype import graph as graph_mod
from dermpact.core.prototype import inventory as inv
from dermpact.core.prototype import report as report_mod
from dermpact.core.prototype import scenarios as scen
from dermpact.core.prototype.host import DEFAULT_STATE_DIR, HostServices, INFERENCE_KINDS, MODES
from dermpact.core.prototype.state import now_iso
from dermpact.core.store import StoreErrorV2
from dermpact.core.record_v2 import GLOBAL_CLASSES
from dermpact.core.memory import OUTCOME_GRADES

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REJECTED = 3
EXIT_UNAVAILABLE = 4
EXIT_CANCELLED = 5


def _build_host(args: argparse.Namespace) -> HostServices:
    host = HostServices.build(args.state_dir, inventory_dir=args.inventory_dir, mode=args.mode, inference=args.inference)
    host.live_model = getattr(args, "model", None)
    return host

_WORKER_ERRORS = (WorkerTimeout, WorkerCrashed, WorkerRequestError, ConnectionError, OSError)

_REJECTED_TERMINAL_STATES = frozenset({"rejected"})
_CANCELLED_TERMINAL_STATES = frozenset({"cancelled"})

_REVIEW_COMMANDS = ("acknowledge", "resolve", "unresolved", "confirm-outcome")


def _add_common_host_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR),
                    help="case store + checkpoint directory (default: %(default)s). For "
                         "'run'/'resume'/'status'/'list-cases' this directory IS the persistent "
                         "case store, so a later 'resume' in a separate process against the same "
                         "--state-dir can pause/resume the run. 'run-all' instead treats this as a "
                         "PARENT directory: it creates one fresh, isolated <state-dir>/<scenario>/ "
                         "case store per scenario, wiping each on every run-all invocation, so no "
                         "scenario (or repeated run-all call) ever sees another's committed data")
    p.add_argument("--inventory-dir", default=str(inv.DEFAULT_INVENTORY_DIR),
                    help="WP-X inventory directory (default: %(default)s)")
    p.add_argument("--mode", default="offline", choices=MODES,
                    help="'offline' runs the offline deterministic controller; "
                         "'live-deferred' constructs the live provider with allow_live=False "
                         "(never sends a request) and falls back to offline; "
                         "'live' constructs it with allow_live=True but still sends nothing unless the "
                         "environment has DERMPACT_ALLOW_LIVE_LLM=1 (otherwise falls back to offline, "
                         "labelled live_gate_closed) (default: %(default)s)")
    p.add_argument("--model", default=None,
                    help="live provider model id override (default: DERMPACT_GENERATOR_MODEL from dermpact.config)")
    p.add_argument("--inference", default="cached", choices=INFERENCE_KINDS,
                    help="'cached' uses WP-X's already-computed inventory predictions; "
                         "'worker' runs the real frozen model in a subprocess (default: %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dermpact.core.prototype",
        description="DermPact Phase 5-6 connected Python prototype: a LangGraph state workflow "
                    "over the record/gate/ledger/episode/clarification/XAI/memory/agent modules, "
                    "with a CLI, a tagged case inventory and nine end-to-end scenarios. "
                    "No HTTP server, frontend, or browser step exists anywhere in this package.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-cases", help="list the allowlisted development cases and tagged synthetic fixtures")
    _add_common_host_args(p_list)

    p_scn = sub.add_parser("scenarios", help="list the nine executable scenarios")

    p_run = sub.add_parser(
        "run", help="start one graph run for a case and scenario",
        description="Starts one run, persisting its checkpoint and case store directly under "
                    "--state-dir (not isolated, not wiped) so a later 'resume' in a separate "
                    "process against the same --state-dir can pause/resume it. Contrast "
                    "'run-all', which isolates each of the nine scenarios in its own fresh "
                    "<state-dir>/<scenario>/ sub-directory.",
    )
    _add_common_host_args(p_run)
    p_run.add_argument("--case", default=None, help="case id (development_allowlisted_case or a fixture id); "
                                                       "defaults to the scenario's own representative case")
    p_run.add_argument("--scenario", required=True, choices=scen.SCENARIO_NAMES)
    p_run.add_argument("--output", required=True, help="directory the run's artefact bundle is written under")
    p_run.add_argument("--request", default=None, help="override the scenario's default opening clinician message")
    p_run.add_argument("--ask-first", action="store_true",
                        help="route straight to the clarify node's own decision-relevant question, regardless of "
                             "whether --request matches one of its own ask-first phrases (WP-D gap-fix addition)")

    p_resume = sub.add_parser(
        "resume", help="resume a paused run with an explicit host confirmation/answer/review",
        description="Resumes a run previously started with 'run' (or a scenario run inside "
                    "'run-all'). Pass the SAME --state-dir the run used: for a plain 'run' that "
                    "is the persistent --state-dir itself; for a run-all scenario that is "
                    "<state-dir>/<scenario>/ (run-all's SUMMARY.md and per-run report.json name "
                    "the run_id but not this sub-directory, so resuming a run-all run manually "
                    "requires passing --state-dir <original-state-dir>/<scenario-name>).",
    )
    _add_common_host_args(p_resume)
    p_resume.add_argument("--run-id", required=True)
    p_resume.add_argument("--output", required=True)
    group = p_resume.add_mutually_exclusive_group(required=True)
    group.add_argument("--confirm", metavar="PROPOSAL_ID", help="confirm the pending proposal")
    group.add_argument("--cancel", metavar="PROPOSAL_ID", help="cancel the pending proposal")
    group.add_argument("--answer", metavar="FACT=VALUE", help="answer a clarification question "
                                                                "(VALUE is true/false/unknown)")
    group.add_argument("--review", metavar="COMMAND", choices=_REVIEW_COMMANDS,
                        help="a direct episode review command (acknowledge/resolve/unresolved/confirm-outcome) -- "
                             "operates on the episode directly (WP-D addition; see the WP-D report), "
                             "not on the graph's own confirmation/question interrupt")
    p_resume.add_argument("--acknowledge", metavar="SHA256", default=None,
                           help="the previewed consequence sha256, required when the pending "
                                "confirmation changes the displayed next step")
    p_resume.add_argument("--evidence-ref", metavar="REF", default=None,
                           help="required by '--review confirm-outcome' (episode.confirm_outcome rejects an "
                                "empty evidence_ref); ignored by every other --review command")
    p_resume.add_argument("--outcome-diagnosis", choices=GLOBAL_CLASSES, default=None,
                          help="explicit confirmed diagnosis; required by '--review confirm-outcome' and never copied from the model")
    p_resume.add_argument("--evidence-grade",
                          choices=tuple(sorted(OUTCOME_GRADES - {"unknown", "dataset_reference_label"})), default=None,
                          help="explicit evidence grade required by '--review confirm-outcome'")

    p_status = sub.add_parser("status", help="show a run's current node and pending interrupt, if any")
    _add_common_host_args(p_status)
    p_status.add_argument("--run-id", required=True)

    p_all = sub.add_parser(
        "run-all", help="run all nine scenarios on the tagged inventory and write SUMMARY.md",
        description="Runs all nine scenarios. Each scenario gets its own fresh, isolated case "
                    "store under <state-dir>/<scenario-name>/, wiped at the start of every "
                    "run-all invocation, so no scenario ever sees another scenario's committed "
                    "amendments and two consecutive run-all calls against the same --state-dir "
                    "produce the same terminal states. SUMMARY.md records, per scenario: the "
                    "terminal state; whether a graph interrupt occurred and how it was resolved "
                    "(confirm/cancel/answer); the record version before and after; the gate "
                    "disposition; and the mode/source-label.",
    )
    _add_common_host_args(p_all)
    p_all.add_argument("--output", required=True)

    from dermpact.core.prototype.luna_cli import register
    register(sub)
    return p




def _print_header(host: HostServices, case_id: str) -> None:
    print(f"mode={host.mode} inference={host.inference}", file=sys.stderr)
    try:
        record = host.store.load_case(case_id).current_record
        print(f"case_id={case_id} source_label={record.identity.source_label}", file=sys.stderr)
    except StoreErrorV2:
        pass


def _describe_interrupt(payload: Dict[str, Any]) -> str:
    if payload.get("kind") == "confirmation":
        ack = " (requires --acknowledge)" if payload.get("requires_acknowledgement") else ""
        return (f"PAUSED: awaiting confirmation -- proposal_id={payload.get('proposal_id')}{ack} "
                f"case_id={payload.get('case_id')} record_version={payload.get('record_version')}")
    if payload.get("kind") == "question":
        return (f"PAUSED: clarification question -- fact={payload.get('fact')!r} "
                f"\"{payload.get('question_text')}\" case_id={payload.get('case_id')} "
                f"record_version={payload.get('record_version')}")
    return f"PAUSED: {payload}"


def _report_run_outcome(run_id: str, case_id: str, result: Dict[str, Any], output_dir: str) -> int:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        report_mod.write_pause_snapshot(Path(output_dir), run_id, state=result, interrupt_payload=payload)
        print(f"run_id={run_id}")
        print(_describe_interrupt(payload))
        return EXIT_OK
    terminal = result.get("terminal_state", "unknown")
    print(f"run_id={run_id}")
    print(f"terminal_state={terminal} record_version={result.get('record_version')} "
          f"record_sha256={result.get('record_sha256')}")
    paths = result.get("report_paths") or {}
    if paths:
        print(f"report={paths.get('report.md')}")
    if terminal in _CANCELLED_TERMINAL_STATES:
        return EXIT_CANCELLED
    if terminal in _REJECTED_TERMINAL_STATES:
        return EXIT_REJECTED
    return EXIT_OK




def cmd_list_cases(args: argparse.Namespace) -> int:
    host = _build_host(args)
    if host.manifest is None:
        print("no inventory manifest found; build the WP-X inventory first", file=sys.stderr)
        return EXIT_UNAVAILABLE
    print(f"{'case_id':<28} {'source_label':<28} {'reliability':<12} {'abstention':<10} clinical")
    for row in host.manifest.demo_subset:
        cid = row["case_id"]
        try:
            record = scen._preview_record(host, cid)
        except Exception as exc:  # noqa: BLE001
            print(f"{cid:<28} <could not build: {exc}>")
            continue
        print(f"{cid:<28} {record.identity.source_label:<28} {record.reliability.disposition:<12} "
              f"{record.reliability.selective_abstention:<10} {record.inputs.clinical_availability} -- {row.get('reason', '')}")
    print()
    print("synthetic workflow fixtures (never real cases):")
    for fid in inv.FIXTURE_CASE_IDS:
        print(f"  {fid:<28} {inv.FIXTURE_LABELS.get(fid, '')}")
    if inv.SKIPPED_FIXTURES:
        print()
        print("fixtures intentionally not built:")
        for name, reason in inv.SKIPPED_FIXTURES.items():
            print(f"  {name}: {reason}")
    return EXIT_OK


def cmd_scenarios(args: argparse.Namespace) -> int:
    for name in scen.SCENARIO_NAMES:
        fn = scen.SCENARIOS[name]
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        print(f"{name:<22} {doc}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    try:
        host = _build_host(args)
    except Exception as exc:  # noqa: BLE001
        print(f"could not build host services: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    try:
        case_id = args.case or scen.default_case_id(host, args.scenario)
        request = args.request if args.request is not None else scen.opening_request(host, case_id, args.scenario)
    except scen.ScenarioError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    _print_header(host, case_id)
    run_id = scen._new_run_id(args.scenario)
    try:
        result = graph_mod.start_run(
            host, run_id=run_id, case_id=case_id, scenario=args.scenario, request=request, output_dir=args.output,
            ask_first=args.ask_first,
        )
    except _WORKER_ERRORS as exc:
        print(f"worker unavailable: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except StoreErrorV2 as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return EXIT_REJECTED
    return _report_run_outcome(run_id, case_id, result, args.output)


def _parse_answer(spec: str) -> Optional[Dict[str, Any]]:
    if "=" not in spec:
        return None
    fact, _, raw = spec.partition("=")
    raw_l = raw.strip().lower()
    if raw_l == "true":
        value: Optional[bool] = True
    elif raw_l == "false":
        value = False
    elif raw_l == "unknown":
        value = None
    else:
        return None
    return {"fact": fact.strip(), "value": value}


def cmd_resume(args: argparse.Namespace) -> int:
    try:
        host = _build_host(args)
    except Exception as exc:  # noqa: BLE001
        print(f"could not build host services: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    if args.review is not None:
        return _cmd_resume_review(args, host)

    if args.confirm is not None:
        resume_value: Dict[str, Any] = {"proposal_id": args.confirm, "action": "confirm", "acknowledge": args.acknowledge}
    elif args.cancel is not None:
        resume_value = {"proposal_id": args.cancel, "action": "cancel"}
    else:
        parsed = _parse_answer(args.answer)
        if parsed is None:
            print("usage error: --answer must be FACT=true|false|unknown", file=sys.stderr)
            return EXIT_USAGE
        resume_value = parsed

    prior_state = graph_mod.get_state(host, args.run_id)
    if prior_state is None:
        print(f"rejected: no such run_id {args.run_id!r} (nothing checkpointed under this --state-dir)", file=sys.stderr)
        return EXIT_REJECTED

    case_id = prior_state.get("case_id", "?")
    try:
        result = graph_mod.resume_run(host, run_id=args.run_id, output_dir=args.output, resume_value=resume_value)
    except _WORKER_ERRORS as exc:
        print(f"worker unavailable: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    _print_header(host, case_id)
    return _report_run_outcome(args.run_id, case_id, result, args.output)


def _cmd_resume_review(args: argparse.Namespace, host: HostServices) -> int:
    """A direct episode command (WP-D addition -- see the WP-D report for why this bypasses the graph's own interrupt/resume protocol: the graph has no dedicated 'review' pause node, only 'await_confirmation'/'clarify')."""
    prior_state = graph_mod.get_state(host, args.run_id)
    if prior_state is None:
        print(f"rejected: no such run_id {args.run_id!r}", file=sys.stderr)
        return EXIT_REJECTED
    case_id = prior_state.get("case_id")
    try:
        case_state = host.store.load_case(case_id)
    except StoreErrorV2 as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return EXIT_REJECTED
    record = case_state.current_record
    episode_id = f"episode-{case_id}"
    ep_state = derive_state(episode_id, case_id, case_state.episode_events)
    now = now_iso()
    session_id, actor = args.run_id, "clinician"
    idempotency_key = f"{args.run_id}-review-{args.review}"
    if args.review == "acknowledge":
        outcome = acknowledge_review(ep_state, actor=actor, session_id=session_id, idempotency_key=idempotency_key,
                                      at=now, current_record_version=record.identity.record_version)
    elif args.review == "resolve":
        outcome = episode_resolve(ep_state, reason="resolved via CLI --review resolve", actor=actor, session_id=session_id,
                                   idempotency_key=idempotency_key, at=now, current_record_version=record.identity.record_version)
    elif args.review == "confirm-outcome":
        if not args.outcome_diagnosis or not args.evidence_grade or not args.evidence_ref:
            print("rejected: confirm-outcome requires --outcome-diagnosis, --evidence-grade and --evidence-ref",
                  file=sys.stderr)
            return EXIT_REJECTED
        outcome_payload = {"diagnosis": args.outcome_diagnosis, "evidence_grade": args.evidence_grade,
                           "source": args.evidence_ref}
        outcome = confirm_outcome(ep_state, outcome=outcome_payload, evidence_ref=args.evidence_ref or "",
                                   actor=actor, session_id=session_id, idempotency_key=idempotency_key, at=now,
                                   current_record_version=record.identity.record_version)
    else:
        outcome = mark_unresolved(ep_state, reason="marked unresolved via CLI --review unresolved", actor=actor,
                                   session_id=session_id, idempotency_key=idempotency_key, at=now,
                                   current_record_version=record.identity.record_version)
    if outcome.status != "ok":
        print(f"rejected: {outcome.reason} {outcome.detail}", file=sys.stderr)
        return EXIT_REJECTED
    if outcome.is_new and outcome.event is not None:
        host.store.append_episode_event(case_id, outcome.event)
    print(f"run_id={args.run_id} case_id={case_id} review_command={args.review} status=ok")
    print(f"episode_status={outcome.state.to_dict()['status']}")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    host = _build_host(args)
    state = graph_mod.get_state(host, args.run_id)
    if state is None:
        print(f"no such run_id {args.run_id!r}", file=sys.stderr)
        return EXIT_REJECTED
    print(f"run_id={args.run_id} case_id={state.get('case_id')} scenario={state.get('scenario')}")
    print(f"record_version={state.get('record_version')} record_sha256={state.get('record_sha256')}")
    print(f"terminal_state={state.get('terminal_state')}")
    pending = state.get("pending_interrupt")
    if pending:
        print(_describe_interrupt(pending))
    return EXIT_OK


def cmd_run_all(args: argparse.Namespace) -> int:
    summary = scen.run_all(
        state_dir=args.state_dir, inventory_dir=args.inventory_dir, output_dir=args.output,
        mode=args.mode, inference=args.inference,
    )
    for r in summary["results"]:
        print(f"{r['scenario']:<22} case={r['case_id']:<28} terminal_state={r['terminal_state']}")
    if summary["errors"]:
        for name, detail in summary["errors"]:
            print(f"ERROR {name}: {detail}", file=sys.stderr)
    print(f"summary={summary['summary_path']}")
    return EXIT_OK if not summary["errors"] else EXIT_REJECTED


_COMMANDS = {
    "list-cases": cmd_list_cases, "scenarios": cmd_scenarios, "run": cmd_run, "resume": cmd_resume,
    "status": cmd_status, "run-all": cmd_run_all,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "luna-clarify":
        from dermpact.core.prototype.luna_cli import run
        return run(args)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_usage(sys.stderr)
        return EXIT_USAGE
    return handler(args)


__all__ = ["build_parser", "main", "EXIT_OK", "EXIT_USAGE", "EXIT_REJECTED", "EXIT_UNAVAILABLE", "EXIT_CANCELLED"]
