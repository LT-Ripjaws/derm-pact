"""Persisted entry point for the Luna workflow; legacy prototype graph is unchanged."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver
from dermpact.core.agent.redesign.clarification import ClarificationAgent
from dermpact.core.agent.redesign.amendment import AmendmentAgent
from dermpact.core.agent.redesign.evidence import EvidenceError
from dermpact.core.agent.redesign.provider import JsonGatewayClient

CHECKPOINT_FILE = "luna_clarification.sqlite"


class Conversation(TypedDict, total=False):
    case_id: str
    session_id: str
    actor: str
    request: str
    rule_id: str
    workflow: str
    agent: dict
    result: dict


def make_graph(host, client_factory=JsonGatewayClient):
    def begin(state):
        host.ensure_case(state["case_id"])
        cls = AmendmentAgent if state.get("workflow") == "amendment" else ClarificationAgent
        agent = cls(host, case_id=state["case_id"], session_id=state["session_id"],
                                   actor=state["actor"], client_factory=client_factory)
        result = agent.begin(state["request"], rule_id=state["rule_id"])
        return {"agent": agent.snapshot(), "result": result}

    def turn(state):
        # Resume starts here again. No model or ledger call precedes interrupt.
        event = interrupt(state["result"])
        agent = ClarificationAgent.restore(host, state["agent"], client_factory=client_factory)
        if event["action"] == "reply":
            result = agent.respond(event["text"])
        else:
            result = agent.confirm(proposal_id=event["proposal_id"], preview_sha256=event["preview_sha256"],
                                   approved=event["action"] == "confirm")
        return {"agent": agent.snapshot(), "result": result}

    def route(state):
        return "turn" if state["agent"]["phase"] in {"awaiting_information", "awaiting_confirmation"} else END

    graph = StateGraph(Conversation)
    graph.add_node("begin", begin); graph.add_node("turn", turn)
    graph.add_edge(START, "begin")
    graph.add_conditional_edges("begin", route); graph.add_conditional_edges("turn", route)
    return graph


def execute(host, *, run_id: str, action: str, case_id=None, rule_id=None, request=None,
            actor="clinician", text=None, proposal_id=None, preview_sha256=None,
            client_factory=JsonGatewayClient, workflow="clarification"):
    if workflow not in {"clarification", "amendment"}:
        raise ValueError("unknown workflow")
    if not run_id.strip() or len(run_id) > 200:
        raise ValueError("run_id must contain 1-200 characters")
    db = Path(host.state_dir) / CHECKPOINT_FILE
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db), check_same_thread=False)
    try:
        app = make_graph(host, client_factory).compile(checkpointer=SqliteSaver(connection))
        config = {"configurable": {"thread_id": run_id}}
        snapshot = app.get_state(config)
        if action == "start":
            if snapshot.values:
                raise EvidenceError("run_id_already_exists")
            if not case_id or not request or (workflow == "clarification" and not rule_id):
                raise ValueError("start requires case and request; clarification also requires rule")
            app.invoke({"case_id": case_id, "session_id": run_id, "actor": actor,
                        "request": request, "rule_id": rule_id or "", "workflow": workflow}, config)
        elif action in {"reply", "confirm", "cancel"}:
            if not snapshot.values or "agent" not in snapshot.values:
                raise EvidenceError("no_saved_luna_conversation")
            agent = ClarificationAgent.restore(host, snapshot.values["agent"], client_factory=client_factory)
            expected = "awaiting_information" if action == "reply" else "awaiting_confirmation"
            if agent.phase != expected or not snapshot.next:
                raise EvidenceError("no_matching_pending_interaction")
            if action == "reply":
                if not text:
                    raise ValueError("reply requires text")
                event = {"action": action, "text": text}
            else:
                if not agent.pending or proposal_id != agent.pending.proposal_id or preview_sha256 != agent.pending.consequence_preview_sha256:
                    raise EvidenceError("confirmation_does_not_match_displayed_proposal")
                event = {"action": action, "proposal_id": proposal_id, "preview_sha256": preview_sha256}
            app.invoke(Command(resume=event), config)
        elif action != "status":
            raise ValueError("unsupported Luna conversation action")
        final = app.get_state(config)
        if not final.values:
            raise EvidenceError("no_saved_luna_conversation")
        result = {"run_id": run_id, "paused": bool(final.next),
                  "result": final.values.get("result", {"outcome": "incomplete"})}
        if "agent" in final.values:
            try:
                ClarificationAgent.restore(host, final.values["agent"], client_factory=client_factory)
            except EvidenceError as exc:
                result["resume_error"] = str(exc)
        return result
    finally:
        connection.close()
