"""WP-D: the connected LangGraph state workflow."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from dermpact.core import renderer_v2
from dermpact.core.agent.controller import run_turn
from dermpact.core.agent.offline_controller import classify_intent
from dermpact.core.agent.providers import LiveGatewayProvider
from dermpact.core.clarification import decision_relevant_questions
from dermpact.core.episode import derive_state, mark_awaiting_information, mark_awaiting_review, request_review
from dermpact.core.episode import record_recomputed as ep_record_recomputed
from dermpact.core.episode import record_unsolicited_memory_reminder
from dermpact.core.ledger_v2 import AmendmentProposal
from dermpact.core.prototype import report as report_mod
from dermpact.core.prototype.host import HostServices
from dermpact.core.prototype.state import PrototypeState, now_iso, trace_entry

CHECKPOINT_FILENAME = "checkpoints.sqlite"

def _record(host: HostServices, case_id: str):
    return host.store.load_case(case_id).current_record


def _find_proposal_data(tool_trace: List[Dict[str, Any]], proposal_id: str) -> Optional[Dict[str, Any]]:
    for entry in reversed(tool_trace):
        if entry.get("tool") in ("propose_fact_amendment", "propose_criterion_amendment") and entry.get("ok"):
            data = entry.get("data") or {}
            if data.get("proposal_id") == proposal_id:
                return data
    return None


def make_graph(host: HostServices) -> StateGraph:
    def load_case(state: PrototypeState) -> Dict[str, Any]:
        case_id = state["case_id"]
        case_state = host.ensure_case(case_id)
        record = case_state.current_record
        return {
            "record_version": record.identity.record_version, "record_sha256": record.record_sha256,
            "trace": [trace_entry("load_case", record_version=record.identity.record_version,
                                   inputs={"case_id": case_id},
                                   outputs_summary={"source_label": record.identity.source_label})],
        }

    def build_record(state: PrototypeState) -> Dict[str, Any]:
        record = _record(host, state["case_id"])
        return {
            "record_version": record.identity.record_version, "record_sha256": record.record_sha256,
            "trace": [trace_entry("build_record", record_version=record.identity.record_version,
                                   inputs={"case_id": state["case_id"]},
                                   outputs_summary={"record_sha256": record.record_sha256})],
        }

    def read_reliability(state: PrototypeState) -> Dict[str, Any]:
        record = _record(host, state["case_id"])
        rel = record.reliability
        return {
            "trace": [trace_entry("read_reliability", record_version=record.identity.record_version,
                                   inputs={"record_sha256": record.record_sha256},
                                   outputs_summary={"disposition": rel.disposition, "iqa_state": rel.iqa_state,
                                                     "ood_state": rel.ood_state, "selective_abstention": rel.selective_abstention})],
        }

    def assess_applicability(state: PrototypeState) -> Dict[str, Any]:
        record = _record(host, state["case_id"])
        verdict = host.gate.verify(record, expected_case_id=state["case_id"], now=date.today())
        return {
            "trace": [trace_entry("assess_applicability", record_version=record.identity.record_version,
                                   inputs={"record_sha256": record.record_sha256},
                                   outputs_summary={"gate_disposition": verdict.disposition,
                                                     "next_step_status": record.next_step.status})],
        }

    def route_request(state: PrototypeState) -> Dict[str, Any]:
        """Route on the controller's own intent classifier (single source of truth): out-of-scope -> refuse; an ask-first request (or ``--ask-first``) -> the ``clarify`` question interrupt; an unrecognised request while a decisive fact is still missing -> ``clarify``; everything else, including review/reacquisition requests, -> ``run_tools`` where the controller proposes and this host applies the episode proposal."""
        record = _record(host, state["case_id"])
        request = state.get("request") or ""
        intent = classify_intent(request)
        if intent == "out_of_scope":
            route = "refuse_out_of_scope"
        elif intent == "ask_first" or bool(state.get("ask_first")):
            route = "clarify"
        elif intent == "unknown" and record.next_step.status == "conditional":
            route = "clarify"
        else:
            route = "run_tools"
        return {
            "route": route,
            "trace": [trace_entry("route_request", record_version=record.identity.record_version,
                                   inputs={"request": request, "intent": intent}, outputs_summary={"route": route})],
        }

    def _offline_run(case_id: str, request: str, session_id: str, actor: str):
        memory_explicit = classify_intent(request) == "similar_cases"
        ctx, ledger, n0 = host.tool_context(case_id, session_id=session_id, actor=actor, agent_mode="offline_deterministic",
                                            memory_request_explicit=memory_explicit)
        result = run_turn(ctx, request)
        host.flush_ledger_events(case_id, ledger, n0)
        trace = result.to_dict()["tool_trace"]
        if any(t.get("tool") == "request_measured_xai" and t.get("ok") for t in trace):
            ctx2, ledger2, n2 = host.tool_context(case_id, session_id=session_id, actor=actor, agent_mode="offline_deterministic",
                                                  memory_request_explicit=memory_explicit)
            result = run_turn(ctx2, request)
            host.flush_ledger_events(case_id, ledger2, n2)
        return result

    def _apply_episode_proposal(case_id: str, proposal: Dict[str, Any], *, also_mark_awaiting_review: bool) -> None:
        """WP-D addition (gap-fix report, Refinement 3): the TOOL (agent/tools.py, frozen) never appends an episode event -- it only returns a computed ``proposed_event`` in ``ToolResult.data`` for the host to decide whether to persist (see that module's own docstring)."""
        proposed = proposal.get("proposed_event")
        if not proposed:
            return
        case_state = host.store.load_case(case_id)
        ep_state = derive_state(f"episode-{case_id}", case_id, case_state.episode_events)
        result = request_review(
            ep_state, reason=proposed["payload"].get("reason", ""), actor=proposed["actor"],
            session_id=proposed["session_id"], idempotency_key=proposed["idempotency_key"], at=proposed["at"],
            current_record_version=proposed["record_version_at_event"],
        )
        if result.is_new and result.event is not None:
            host.store.append_episode_event(case_id, result.event)
            ep_state = result.state
        if also_mark_awaiting_review:
            aw = mark_awaiting_review(
                ep_state, actor=proposed["actor"], session_id=proposed["session_id"],
                idempotency_key=f"{proposed['idempotency_key']}-awaiting", at=proposed["at"],
                current_record_version=proposed["record_version_at_event"],
            )
            if aw.is_new and aw.event is not None:
                host.store.append_episode_event(case_id, aw.event)

    def _run_agent_turn(state: PrototypeState, *, node_name: str) -> Dict[str, Any]:
        case_id, session_id, actor = state["case_id"], state["session_id"], state["actor"]
        request = state.get("request") or ""
        live_status = None
        memory_explicit = classify_intent(request) == "similar_cases"
        if node_name == "refuse_out_of_scope":
            result = _offline_run(case_id, request, session_id, actor)
            if state.get("mode") in ("live-deferred", "live"):
                live_status = "not_required_scope_gate"
        elif host.provider_factory is not None:
            ctx, ledger, n0 = host.tool_context(case_id, session_id=session_id, actor=actor, agent_mode="live_provider",
                                                memory_request_explicit=memory_explicit)
            result = run_turn(ctx, request, provider=host.provider_factory(request, ctx))
            host.flush_ledger_events(case_id, ledger, n0)
        elif state.get("mode") in ("live-deferred", "live"):
            allow_live = state.get("mode") == "live"
            ctx, ledger, n0 = host.tool_context(case_id, session_id=session_id, actor=actor, agent_mode="live_provider",
                                                memory_request_explicit=memory_explicit)
            result = run_turn(ctx, request, provider=LiveGatewayProvider(allow_live=allow_live, model=host.live_model))
            host.flush_ledger_events(case_id, ledger, n0)
            if result.terminal_state == "provider_failed":
                live_status = "deferred_by_user" if not allow_live else (
                    "live_gate_closed" if os.environ.get("DERMPACT_ALLOW_LIVE_LLM") != "1" else "live_provider_failed")
                result = _offline_run(case_id, request, session_id, actor)
            else:
                live_status = "live_provider_used"
        else:
            result = _offline_run(case_id, request, session_id, actor)

        for proposal in result.episode_proposals:
            if proposal.get("status") == "ok":
                _apply_episode_proposal(case_id, proposal, also_mark_awaiting_review=(proposal.get("kind") == "review"))

        if result.terminal_state == "answered":
            for tool_result in result.tool_trace:
                data = tool_result.get("data") or {}
                if tool_result.get("tool") == "retrieve_case_comparisons" and data.get("unsolicited_reminder_proposed"):
                    receipt = data.get("receipt") or {}
                    receipt_hash = hashlib.sha256(
                        json.dumps(receipt, sort_keys=True, separators=(",", ":"), default=str).encode()
                    ).hexdigest()
                    case_state = host.store.load_case(case_id)
                    ep_state = derive_state(f"episode-{case_id}", case_id, case_state.episode_events)
                    event = record_unsolicited_memory_reminder(
                        ep_state, comparison_receipt_sha256=receipt_hash, actor=actor, session_id=session_id,
                        idempotency_key=f"memory-reminder:{receipt_hash}", at=now_iso(),
                        current_record_version=_record(host, case_id).identity.record_version,
                    )
                    if event.is_new and event.event is not None:
                        host.store.append_episode_event(case_id, event.event)

        record = _record(host, case_id)
        result_dict = result.to_dict()
        pending_interrupt = None
        if result.pending_proposal_id:
            data = _find_proposal_data(result_dict["tool_trace"], result.pending_proposal_id)
            pending_interrupt = {
                "kind": "confirmation", "proposal_id": result.pending_proposal_id,
                "preview_sha256": (data or {}).get("consequence_preview_sha256"),
                "requires_acknowledgement": bool((data or {}).get("next_step_changed")),
                "case_id": case_id, "record_version": record.identity.record_version, "expires_at": None,
            }
        checker_ok = bool(result_dict["checker_verdict"].get("ok", True))
        next_route = "await_confirmation" if pending_interrupt is not None else "verify_response"
        return {
            "turn_result": result_dict, "pending_interrupt": pending_interrupt, "checker_ok": checker_ok,
            "live_status": live_status, "record_version": record.identity.record_version,
            "record_sha256": record.record_sha256, "route": next_route, "terminal_state": result.terminal_state,
            "trace": [trace_entry(node_name, record_version=record.identity.record_version,
                                   inputs={"request": request},
                                   outputs_summary={"terminal_state": result.terminal_state, "mode": result.mode})],
        }

    def run_tools(state: PrototypeState) -> Dict[str, Any]:
        return _run_agent_turn(state, node_name="run_tools")

    def refuse_out_of_scope(state: PrototypeState) -> Dict[str, Any]:
        update = _run_agent_turn(state, node_name="refuse_out_of_scope")
        if update.get("terminal_state") == "answered":
            update["terminal_state"] = "refused_out_of_scope"
            turn = dict(update.get("turn_result") or {})
            turn["terminal_state"] = "refused_out_of_scope"
            update["turn_result"] = turn
            trace = list(update.get("trace") or [])
            if trace:
                trace[-1] = {**trace[-1], "outputs_summary": {
                    **trace[-1].get("outputs_summary", {}), "terminal_state": "refused_out_of_scope",
                }}
                update["trace"] = trace
        return update

    def clarify(state: PrototypeState) -> Dict[str, Any]:
        case_id = state["case_id"]
        record = _record(host, case_id)
        case_state = host.store.load_case(case_id)
        analysis = decision_relevant_questions(
            record, engine=host.engine, catalogue=host.catalogue, calibrators=host.bundle.calibration,
            bundle=None, episode_events=case_state.episode_events,
        )
        relevant = [q for q in analysis.questions if q.decision_relevant]
        if not relevant:
            return {
                "route": "render_report", "clarification": analysis.to_dict(), "terminal_state": "answered", "checker_ok": True,
                "trace": [trace_entry("clarify", record_version=record.identity.record_version, inputs={"case_id": case_id},
                                       outputs_summary={"note": "no decision-relevant missing fact found"})],
            }
        question = relevant[0]
        payload = {
            "kind": "question", "fact": question.fact, "question_text": question.question_text,
            "permitted_values": list(question.permitted_values), "affected_rules": list(question.affected_rules),
            "case_id": case_id, "record_version": record.identity.record_version, "expires_at": None,
            "requires_acknowledgement": False,
        }
        answer = interrupt(payload)
        fact = answer.get("fact")
        value = answer.get("value")
        session_id, actor = state["session_id"], state["actor"]
        now = now_iso()
        if fact != question.fact:
            return {
                "route": "render_report", "terminal_state": "rejected", "checker_ok": True, "clarification": analysis.to_dict(),
                "trace": [trace_entry("clarify", record_version=record.identity.record_version, inputs={"answer": answer},
                                       outputs_summary={"rejected": "fact_mismatch"})],
            }
        if value is None:
            episode_id = f"episode-{case_id}"
            ep_state = derive_state(episode_id, case_id, case_state.episode_events)
            result = mark_awaiting_information(
                ep_state, fact=fact, unavailable=True, actor=actor, session_id=session_id,
                idempotency_key=f"{case_id}-unknown-{fact}", at=now, current_record_version=record.identity.record_version,
            )
            if result.is_new and result.event is not None:
                host.store.append_episode_event(case_id, result.event)
            return {
                "route": "render_report", "terminal_state": "awaiting_information", "checker_ok": True,
                "clarification": analysis.to_dict(), "episode": result.state.to_dict(),
                "trace": [trace_entry("clarify", record_version=record.identity.record_version, inputs={"answer": answer},
                                       outputs_summary={"fact": fact, "unavailable": True})],
            }
        ledger, n0 = host.ledger_for_case(case_id)
        proposal = ledger.propose(
            record, AmendmentProposal(kind="clinician_fact", field=fact, value=bool(value)),
            actor=actor, session_id=session_id, proposed_at=now, statement=f"clarify answer: {fact}={value!r}",
        )
        host.flush_ledger_events(case_id, ledger, n0)
        if proposal.status != "proposed":
            return {
                "route": "render_report", "terminal_state": "rejected", "checker_ok": True, "clarification": analysis.to_dict(),
                "trace": [trace_entry("clarify", record_version=record.identity.record_version, inputs={"answer": answer},
                                       outputs_summary={"rejected": proposal.reason})],
            }
        pending_interrupt = {
            "kind": "confirmation", "proposal_id": proposal.proposal_id, "preview_sha256": proposal.consequence_preview_sha256,
            "requires_acknowledgement": bool(proposal.next_step_changed), "case_id": case_id,
            "record_version": record.identity.record_version, "expires_at": None,
        }
        return {
            "route": "await_confirmation", "pending_interrupt": pending_interrupt, "clarification": analysis.to_dict(),
            "trace": [trace_entry("clarify", record_version=record.identity.record_version, inputs={"answer": answer},
                                   outputs_summary={"proposed": proposal.proposal_id})],
        }

    def await_confirmation(state: PrototypeState) -> Dict[str, Any]:
        case_id = state["case_id"]
        pending = state["pending_interrupt"] or {}
        payload = {
            "kind": "confirmation", "proposal_id": pending.get("proposal_id"), "preview_sha256": pending.get("preview_sha256"),
            "requires_acknowledgement": pending.get("requires_acknowledgement", False), "case_id": case_id,
            "record_version": pending.get("record_version"), "expires_at": pending.get("expires_at"),
        }
        action = interrupt(payload)
        session_id, actor = state["session_id"], state["actor"]
        now = now_iso()
        ledger, n0 = host.ledger_for_case(case_id)
        case_state = host.store.load_case(case_id)
        current_record = case_state.current_record
        proposal_id = action.get("proposal_id")
        outcome, reason = "rejected", None
        if proposal_id != pending.get("proposal_id"):
            reason = "wrong_proposal"
        elif action.get("action") == "cancel":
            r = ledger.cancel(proposal_id, session_id=session_id, now=now)
            outcome, reason = r.status, r.reason
        elif action.get("action") == "confirm":
            token = ledger.issue_confirmation_token(proposal_id, issued_at=now, session_id=session_id)
            if token.status != "issued":
                outcome, reason = "rejected", token.reason
            else:
                commit = ledger.commit(
                    current_record, token.token, now=now, actor=actor, session_id=session_id,
                    acknowledged_consequence_sha256=action.get("acknowledge"),
                )
                if commit.status == "committed":
                    host.store.append_record(case_id, commit.record, expected_record_version=current_record.identity.record_version)
                    episode_id = f"episode-{case_id}"
                    ep_state = derive_state(episode_id, case_id, case_state.episode_events)
                    rec_result = ep_record_recomputed(
                        ep_state, record_version=commit.record.identity.record_version, actor=actor, session_id=session_id,
                        idempotency_key=f"{case_id}-recomputed-{commit.record.record_sha256}", at=now,
                    )
                    if rec_result.is_new and rec_result.event is not None:
                        host.store.append_episode_event(case_id, rec_result.event)
                    outcome = "committed"
                else:
                    outcome, reason = "rejected", commit.reason
        else:
            reason = "unknown_action"
        host.flush_ledger_events(case_id, ledger, n0)
        final_record = host.store.load_case(case_id).current_record
        return {
            "record_version": final_record.identity.record_version, "record_sha256": final_record.record_sha256,
            "pending_interrupt": None, "terminal_state": outcome, "checker_ok": True, "route": "verify_response",
            "trace": [trace_entry("await_confirmation", record_version=final_record.identity.record_version,
                                   inputs={"action": action}, outputs_summary={"outcome": outcome, "reason": reason})],
        }

    def verify_response(state: PrototypeState) -> Dict[str, Any]:
        checker_ok = state.get("checker_ok", True)
        route = "render_report" if checker_ok else "repair"
        return {
            "route": route,
            "trace": [trace_entry("verify_response", record_version=state.get("record_version"),
                                   inputs={"terminal_state": state.get("terminal_state")},
                                   outputs_summary={"checker_ok": checker_ok, "route": route})],
        }

    def repair(state: PrototypeState) -> Dict[str, Any]:
        return {
            "route": "render_report",
            "trace": [trace_entry("repair", record_version=state.get("record_version"),
                                   inputs={"terminal_state": state.get("terminal_state")},
                                   outputs_summary={"note": "run_turn's own repair budget (max 1) already exhausted; "
                                                             "rendering the host fallback template"})],
        }

    def render_report(state: PrototypeState) -> Dict[str, Any]:
        record = _record(host, state["case_id"])
        verdict = host.gate.verify(record, expected_case_id=state["case_id"], now=date.today())
        terminal = state.get("terminal_state") or "answered"
        return {
            "terminal_state": terminal,
            "trace": [trace_entry("render_report", record_version=record.identity.record_version,
                                   inputs={"terminal_state": terminal}, outputs_summary={"gate_disposition": verdict.disposition})],
        }

    def persist(state: PrototypeState) -> Dict[str, Any]:
        case_id, run_id = state["case_id"], state["run_id"]
        output_dir = Path(state["output_dir"])
        case_state = host.store.load_case(case_id)
        record = case_state.current_record
        verdict = host.gate.verify(record, expected_case_id=case_id, now=date.today(), lineage=list(case_state.records))
        if verdict.disposition == "refusal":
            rendered = renderer_v2.render_refusal(verdict, case_id)
        elif verdict.disposition == "display":
            rendered = renderer_v2.render(record, verdict, lineage=case_state.records)
        else:
            rendered = renderer_v2.render_partial(record, verdict)
        view = renderer_v2.view_model(record, verdict)
        episode_state = derive_state(f"episode-{case_id}", case_id, case_state.episode_events)
        report = report_mod.build_report(
            state=state, record=record, verdict=verdict, rendered=rendered, view=view,
            episode_state=episode_state, case_state=case_state, host=host,
        )
        paths = report_mod.write_report(output_dir, run_id, report, record=record, case_state=case_state, rendered=rendered)
        return {
            "report_paths": paths, "episode": episode_state.to_dict(),
            "trace": [trace_entry("persist", record_version=record.identity.record_version, inputs={"run_id": run_id},
                                   outputs_summary={"files": sorted(paths.keys())})],
        }

    def _edge(state: PrototypeState) -> str:
        return state["route"]

    g = StateGraph(PrototypeState)
    for name, fn in (
        ("load_case", load_case), ("build_record", build_record), ("read_reliability", read_reliability),
        ("assess_applicability", assess_applicability), ("route_request", route_request), ("clarify", clarify),
        ("await_confirmation", await_confirmation), ("run_tools", run_tools),
        ("refuse_out_of_scope", refuse_out_of_scope), ("verify_response", verify_response), ("repair", repair),
        ("render_report", render_report), ("persist", persist),
    ):
        g.add_node(name, fn)

    g.add_edge(START, "load_case")
    g.add_edge("load_case", "build_record")
    g.add_edge("build_record", "read_reliability")
    g.add_edge("read_reliability", "assess_applicability")
    g.add_edge("assess_applicability", "route_request")
    g.add_conditional_edges("route_request", _edge, {
        "clarify": "clarify", "refuse_out_of_scope": "refuse_out_of_scope", "run_tools": "run_tools",
    })
    g.add_conditional_edges("run_tools", _edge, {"await_confirmation": "await_confirmation", "verify_response": "verify_response"})
    g.add_conditional_edges("clarify", _edge, {"await_confirmation": "await_confirmation", "render_report": "render_report"})
    g.add_edge("refuse_out_of_scope", "verify_response")
    g.add_edge("await_confirmation", "verify_response")
    g.add_conditional_edges("verify_response", _edge, {"render_report": "render_report", "repair": "repair"})
    g.add_edge("repair", "render_report")
    g.add_edge("render_report", "persist")
    g.add_edge("persist", END)
    return g


def compile_app(host: HostServices):
    """One compiled app and one SQLite checkpoint connection per host (see ``HostServices.compiled_app``/``close``)."""
    return host.compiled_app(make_graph, CHECKPOINT_FILENAME)


def start_run(
    host: HostServices, *, run_id: str, case_id: str, scenario: str, request: str, output_dir: Any,
    session_id: Optional[str] = None, actor: str = "clinician", ask_first: bool = False,
) -> Dict[str, Any]:
    app = compile_app(host)
    config = {"configurable": {"thread_id": run_id}}
    initial_state: PrototypeState = {
        "run_id": run_id, "case_id": case_id, "scenario": scenario, "mode": host.mode, "inference": host.inference,
        "output_dir": str(output_dir), "session_id": session_id or run_id, "actor": actor, "request": request,
        "ask_first": bool(ask_first), "trace": [],
    }
    return app.invoke(initial_state, config)


def resume_run(host: HostServices, *, run_id: str, output_dir: Any, resume_value: Any) -> Dict[str, Any]:
    app = compile_app(host)
    config = {"configurable": {"thread_id": run_id}}
    return app.invoke(Command(resume=resume_value, update={"output_dir": str(output_dir)}), config)


def get_state(host: HostServices, run_id: str) -> Optional[Dict[str, Any]]:
    app = compile_app(host)
    snapshot = app.get_state({"configurable": {"thread_id": run_id}})
    return dict(snapshot.values) if snapshot.values else None


__all__ = ["CHECKPOINT_FILENAME", "compile_app", "get_state", "make_graph", "resume_run", "start_run"]
