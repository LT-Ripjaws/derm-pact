"""WP-AG: one bounded LLM controller loop."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from dermpact.core.agent.checker import CheckerVerdict, check_explanation
from dermpact.core.agent.config import WORKFLOW_CONFIG, WorkflowConfig
from dermpact.core.agent.offline_controller import OfflineDeterministicProvider, classify_intent
from dermpact.core.agent.prompts import PROMPT_SHA256, SYSTEM_PROMPT, TOOLS_SCHEMA
from dermpact.core.agent.tools import ToolContext, ToolRegistry, ToolResult

TERMINAL_STATES = frozenset({
    "answered", "awaiting_confirmation", "awaiting_information", "refused_out_of_scope",
    "fallback_template", "provider_failed", "budget_exhausted", "cancelled",
})

_INTENT_TOOL_NAMES = {
    "explain": {"read_record_section"},
    "why_deferred": {"read_reliability", "read_record_section", "propose_review", "propose_reacquisition"},
    "supply_assertion": {"propose_fact_amendment", "preview_consequences"},
    "correct_criterion": {"propose_criterion_amendment", "preview_consequences"},
    "xai_region": {"request_measured_xai", "read_measured_xai"},
    "clinical_photo": {"request_measured_xai", "read_measured_xai"},
    "similar_cases": {"retrieve_case_comparisons"},
    "source_quote": {"retrieve_guideline_evidence", "search_guideline_library"},
    "what_remains": {"read_review_history", "find_decisive_missing_facts"},
    "review_request": {"propose_review", "read_review_history"},
    "reacquisition_request": {"propose_reacquisition", "read_review_history"},
    "ask_first": {"find_decisive_missing_facts"},
}

_REQUIRED_READ_TOOL_BY_INTENT = {
    "explain": "read_record_section",
    "why_deferred": "read_reliability",
    "similar_cases": "retrieve_case_comparisons",
    "supply_assertion": "propose_fact_amendment",
    "correct_criterion": "propose_criterion_amendment",
}

_COMPLEX_ROUTE_TOOLS = frozenset({
    "propose_fact_amendment", "propose_criterion_amendment", "preview_consequences",
    "find_decisive_missing_facts", "propose_review", "propose_reacquisition",
})
_PROPOSAL_TOOLS = frozenset({"propose_fact_amendment", "propose_criterion_amendment"})
_EPISODE_PROPOSAL_TOOLS = frozenset({"propose_review", "propose_reacquisition"})

@dataclass(frozen=True)
class TurnResult:
    mode: str
    text: str
    claims: Tuple[str, ...]
    tool_trace: Tuple[Dict[str, Any], ...]
    pending_proposal_id: Optional[str]
    episode_proposals: Tuple[Dict[str, Any], ...]
    checker_verdict: Dict[str, Any]
    checker_history: Tuple[Dict[str, Any], ...]
    repairs_used: int
    usage: Dict[str, Any]
    latency_s: float
    terminal_state: str
    prompt_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "text": self.text, "claims": list(self.claims),
            "tool_trace": list(self.tool_trace), "pending_proposal_id": self.pending_proposal_id,
            "episode_proposals": list(self.episode_proposals), "checker_verdict": self.checker_verdict,
            "checker_history": list(self.checker_history),
            "repairs_used": self.repairs_used, "usage": self.usage, "latency_s": self.latency_s,
            "terminal_state": self.terminal_state, "prompt_sha256": self.prompt_sha256,
        }


def _classify_route(tool_trace: Sequence[ToolResult]) -> str:
    if tool_trace and tool_trace[0].tool in _COMPLEX_ROUTE_TOOLS:
        return "complex"
    return "simple"


def _verdict_dict(verdict: Optional[CheckerVerdict]) -> Dict[str, Any]:
    if verdict is None:
        return {"ok": True, "rejections": [], "matched_claims": []}
    return {
        "ok": verdict.ok, "rejections": [dataclasses.asdict(r) for r in verdict.rejections],
        "matched_claims": list(verdict.matched_claims),
    }


def _fallback_text(record: Any) -> str:
    return (
        f"Case {record.identity.case_id} v{record.identity.record_version}: "
        f"{record.next_step.display_text} Scope: {record.next_step.scope_note}"
    )


def _summary_claims(result: ToolResult, keep: Callable[[str], bool]) -> List[str]:
    return [str(p["id"]) for p in result.propositions if keep(str(p.get("path_or_ref", "")))]


def _grounded_tool_summary(result: ToolResult) -> Optional[Tuple[str, List[str]]]:
    """Compile the exact facts required by a read intent from its verified receipt."""
    error_code = (result.error or {}).get("code")
    if not result.ok and result.tool == "retrieve_case_comparisons" and error_code == "memory_unavailable":
        return (
            "Historical comparison memory is not available for this case. "
            "No historical case is used and the current prediction is unchanged.",
            [],
        )
    if not result.ok and result.tool == "request_measured_xai" and error_code == "xai_unavailable":
        return (
            "Measured image-sensitivity evidence is unavailable for this case. "
            "No attribution claim is made and the current prediction is unchanged.",
            [],
        )
    if not result.ok or not isinstance(result.data, Mapping):
        return None
    data = result.data
    if result.tool == "read_reliability":
        parts = [
            "Reliability status:", f"disposition {data.get('disposition')};",
            f"image-quality state {data.get('iqa_state')};", f"distribution state {data.get('ood_state')};",
            f"view-stability state {data.get('view_stability_state')};",
            f"selective-abstention state {data.get('selective_abstention')}."
        ]
        for warning in data.get("warnings") or []:
            parts.append(
                f"Warning {warning.get('code')}: {warning.get('reason')}. "
                f"Recovery action: {warning.get('recovery_action')}."
            )
        if not data.get("warnings"):
            parts.append("No reliability warning or recovery action is recorded for this case.")
        suffixes = (".iqa_state", ".ood_state", ".view_stability_state", ".selective_abstention", ".disposition",
                    ".code", ".reason", ".recovery_action")
        return " ".join(parts), _summary_claims(result, lambda path: path.endswith(suffixes))
    if result.tool == "read_record_section":
        heads = data.get("heads") if isinstance(data.get("heads"), Mapping) else data
        diagnosis = heads.get("diagnosis_head") or {}
        probabilities = diagnosis.get("probabilities") or {}
        provisional = diagnosis.get("provisional_class")
        parts = [
            f"Model evidence: provisional diagnosis {provisional} with class probability {probabilities.get(provisional)}; "
            f"separate melanoma-head probability {(heads.get('melanoma_head') or {}).get('probability')}.",
            "Criterion probabilities: " + "; ".join(
                f"{name} {value}" for name, value in sorted((heads.get("criterion_probabilities") or {}).items())
            ) + ".",
        ]
        reliability = data.get("reliability")
        if isinstance(reliability, Mapping):
            parts.append(
                "Reliability: "
                f"disposition {reliability.get('disposition')}; image-quality {reliability.get('iqa_state')}; "
                f"distribution {reliability.get('ood_state')}; view stability {reliability.get('view_stability_state')}; "
                f"selective abstention {reliability.get('selective_abstention')}."
            )
        checklist = data.get("checklist")
        if isinstance(checklist, Mapping):
            parts.append(
                f"Checklist score range {checklist.get('score_min')} to {checklist.get('score_max')}; "
                f"decidable {checklist.get('decidable')}."
            )
        agreement = data.get("agreement")
        if isinstance(agreement, Mapping):
            parts.append(f"Direct-versus-checklist agreement recorded: {agreement.get('present')}.")
        next_step = data.get("next_step")
        if isinstance(next_step, Mapping):
            parts.append(
                f"Next-step status {next_step.get('status')}; action {next_step.get('action_code')}. "
                f"{next_step.get('display_text')} Scope: {next_step.get('scope_note')}"
            )
        required_suffixes = (
            "diagnosis_head.provisional_class", "melanoma_head.probability", "reliability.disposition",
            "reliability.iqa_state", "reliability.ood_state", "reliability.view_stability_state",
            "reliability.selective_abstention", "checklist.score_min", "checklist.score_max",
            "checklist.decidable", "agreement.present", "next_step.status", "next_step.action_code",
        )
        return " ".join(parts), _summary_claims(
            result, lambda path: path.endswith(required_suffixes) or ".criterion_probabilities." in path
        )
    if result.tool == "retrieve_case_comparisons":
        text = data.get("grounded_summary_text")
        if not text:
            return None
        keep = lambda path: (
            path == "memory.status" or path.endswith("descriptive_reference_label_counts.n_retrieved")
            or ".descriptive_reference_label_counts.counts." in path
            or path.endswith("descriptive_reference_label_counts.diagnostic_effect")
            or ("memory.matches[" in path and path.endswith((".similarity", ".outcome.diagnosis")))
            or path in {"memory.counterexample_status", "memory.reason", "memory.counterexample.similarity",
                        "memory.counterexample.outcome.diagnosis"}
            or path.endswith("memory.receipt.policy.min_similarity")
        )
        return str(text), _summary_claims(result, keep)
    return None


def _failure_path_grounded_summary(tool_trace: Sequence[ToolResult]) -> Optional[Tuple[str, List[str]]]:
    for result in reversed(tool_trace):
        summary = _grounded_tool_summary(result)
        if summary is not None and result.tool in {"read_reliability", "read_record_section"}:
            return summary
    return None


def provider_tool_payload(result: ToolResult) -> Dict[str, Any]:
    """Compact wire view; the full ToolResult remains in the checker/audit trace."""
    binding = result.bindings or {}
    data = result.data
    propositions = list(result.propositions)
    if result.tool == "retrieve_case_comparisons" and isinstance(data, Mapping):
        def compact_match(match: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
            if not match:
                return None
            outcome = match.get("outcome") or {}
            return {
                "episode_id": match.get("episode_id"), "case_id": match.get("case_id"),
                "similarity": match.get("similarity"), "matched_fields": match.get("matched_fields"),
                "different_fields": match.get("different_fields"), "unknown_fields": match.get("unknown_fields"),
                "diagnosis": match.get("diagnosis"),
                "outcome": {key: outcome.get(key) for key in ("diagnosis", "evidence_grade", "source", "date")},
                "confirmed_amendments": match.get("confirmed_amendments"),
            }
        receipt = data.get("receipt") or {}
        data = {
            "status": data.get("status"), "matches": [compact_match(match) for match in data.get("matches", [])],
            "counterexample": compact_match(data.get("counterexample")),
            "counterexample_status": data.get("counterexample_status"), "reason": data.get("reason"),
            "dropped_after_revocation": data.get("dropped_after_revocation"),
            "descriptive_reference_label_counts": data.get("descriptive_reference_label_counts"),
            "grounded_summary_text": data.get("grounded_summary_text"),
            "policy": receipt.get("policy"), "identity_exclusion": receipt.get("identity_exclusion"),
            "decision_effect": "comparison only; no vote, probability blend, or diagnosis rewrite",
        }
        keep_fragments = (
            ".similarity", ".diagnosis", ".evidence_grade", ".matched_fields", ".different_fields",
            ".unknown_fields", ".confirmed_amendments", "memory.status", "memory.counterexample_status",
            "memory.reason", "memory.dropped_after_revocation", "memory.receipt.policy",
            "memory.receipt.identity_exclusion", "memory.descriptive_reference_label_counts",
        )
        propositions = [p for p in propositions if any(fragment in str(p.get("path_or_ref", "")) for fragment in keep_fragments)]
    return {
        "tool": result.tool, "ok": result.ok, "data": data, "error": result.error,
        "binding": {key: binding.get(key) for key in ("case_id", "record_version")},
        "propositions": [
            {key: proposition.get(key) for key in ("id", "path_or_ref", "value")}
            for proposition in propositions
        ],
    }


def _tools_schema_for_intent(intent: str) -> List[Dict[str, Any]]:
    allowed_names = _INTENT_TOOL_NAMES.get(intent, set())
    schema = [dict(spec) for spec in TOOLS_SCHEMA if spec["name"] in allowed_names]
    if intent == "explain":
        for spec in schema:
            if spec["name"] == "read_record_section":
                spec["args"] = {**spec["args"], "section": "overview"}
    return schema


def _intent_policy_rejection(
    intent: str, tool_name: Any, tool_args: Mapping[str, Any], *, case_id: str, record_version: int,
    runtime_allowed_names: Optional[Sequence[str]] = None,
) -> Optional[ToolResult]:
    """Enforce provider-facing tool policy again at runtime."""
    allowed_names = set(runtime_allowed_names) if runtime_allowed_names is not None else _INTENT_TOOL_NAMES.get(intent, set())
    if tool_name not in allowed_names:
        return ToolResult(
            ok=False, tool=str(tool_name), args=dict(tool_args),
            bindings={"case_id": case_id, "record_version": record_version},
            error={
                "code": "tool_not_allowed_for_intent",
                "detail": f"tool {tool_name!r} is not allowed for request_intent {intent!r}",
                "repairable": True,
            },
        )
    if intent == "explain" and tool_name == "read_record_section" and tool_args.get("section") != "overview":
        return ToolResult(
            ok=False, tool=str(tool_name), args=dict(tool_args),
            bindings={"case_id": case_id, "record_version": record_version},
            error={
                "code": "tool_arguments_not_allowed_for_intent",
                "detail": "explain permits read_record_section only with section='overview'",
                "repairable": True,
            },
        )
    return None


def run_turn(
    ctx: ToolContext, message: str, *, provider: Any = None, budgets: WorkflowConfig = WORKFLOW_CONFIG,
    cancel_token: Optional[Callable[[], bool]] = None,
) -> TurnResult:
    registry = ToolRegistry(ctx)
    offline_mode = ctx.mode == "offline_deterministic"
    active_provider = OfflineDeterministicProvider(message, ctx) if offline_mode else provider
    if active_provider is None:
        raise ValueError("run_turn requires a provider unless ctx.mode == 'offline_deterministic'")

    tool_trace: List[ToolResult] = []
    repairs_used = 0
    duplicate_count = 0
    proposal_calls = 0
    pending_proposal_id: Optional[str] = None
    episode_proposals: List[Dict[str, Any]] = []
    usage_total: Dict[str, Any] = {}
    latency_total = 0.0
    checker_history: List[Dict[str, Any]] = []

    def _current_record():
        return ctx.store.load_case(ctx.case_id).current_record

    initial_record = _current_record()
    request_intent = classify_intent(message)
    active_tools_schema = _tools_schema_for_intent(request_intent)
    enforce_xai_sequence = (not offline_mode and request_intent in {"xai_region", "clinical_photo"})
    expected_xai_tool: Optional[str] = "request_measured_xai" if enforce_xai_sequence else None
    expected_read_tool: Optional[str] = None if offline_mode else _REQUIRED_READ_TOOL_BY_INTENT.get(request_intent)
    required_tool = expected_xai_tool or expected_read_tool
    if required_tool is not None:
        active_tools_schema = [spec for spec in active_tools_schema if spec["name"] == required_tool]
    messages: List[Dict[str, Any]] = [{
        "role": "user",
        "content": json.dumps({
            "host_context": {
                "case_id": ctx.case_id,
                "record_version": initial_record.identity.record_version,
                "source_label": initial_record.identity.source_label,
                "request_intent": request_intent,
            },
            "clinician_message": message,
        }, sort_keys=True),
    }]

    def _finish(text: str, claims: Sequence[str], terminal_state: str, *, checker_verdict: Optional[CheckerVerdict] = None) -> TurnResult:
        return TurnResult(
            mode=ctx.mode, text=text, claims=tuple(claims), tool_trace=tuple(t.to_dict() for t in tool_trace),
            pending_proposal_id=pending_proposal_id, episode_proposals=tuple(episode_proposals),
            checker_verdict=_verdict_dict(checker_verdict), checker_history=tuple(checker_history),
            repairs_used=repairs_used, usage=dict(usage_total),
            latency_s=latency_total, terminal_state=terminal_state,
            prompt_sha256=(None if offline_mode else PROMPT_SHA256),
        )

    def _budget_exhausted(reason: str) -> TurnResult:
        try:
            record = _current_record()
            text = f"{reason}\n{_fallback_text(record)}"
        except Exception:  # noqa: BLE001 - never let a fallback attempt mask the real termination
            text = reason
        return _finish(text, (), "budget_exhausted")

    def _provider_failed() -> TurnResult:
        try:
            record = _current_record()
            text = _fallback_text(record)
        except Exception:  # noqa: BLE001
            text = "provider failed and no fallback record is available"
        return _finish(text, (), "provider_failed")

    for _ in range(budgets.max_turns):
        if cancel_token is not None and cancel_token():
            return _finish("", (), "cancelled")

        try:
            reply = active_provider.complete(SYSTEM_PROMPT, messages, active_tools_schema)
        except Exception:  # noqa: BLE001 - LiveCallsDisabled and transport failures land here
            return _provider_failed()

        if reply.usage:
            for k, v in reply.usage.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    usage_total[k] = usage_total.get(k, 0) + v
        latency_total += reply.latency_s

        if reply.error:
            return _provider_failed()

        if reply.tool_calls:
            call = reply.tool_calls[0]
            tool_name, tool_args = call.get("tool"), dict(call.get("args", {}))
            record_version = _current_record().identity.record_version
            provider_requested_case_id = tool_args.get("case_id")
            provider_requested_record_version = tool_args.get("record_version")
            binding_overridden = (
                provider_requested_case_id not in {None, ctx.case_id}
                or provider_requested_record_version not in {None, record_version}
            )
            tool_args["case_id"] = ctx.case_id
            tool_args["record_version"] = record_version
            result = _intent_policy_rejection(
                request_intent, tool_name, tool_args,
                case_id=ctx.case_id, record_version=record_version,
                runtime_allowed_names=({required_tool} if required_tool is not None else None),
            ) or registry.call(tool_name, tool_args)
            if binding_overridden:
                result = dataclasses.replace(result, args={
                    **result.args,
                    "_binding_overridden": True,
                    "_provider_requested_case_id": provider_requested_case_id,
                    "_provider_requested_record_version": provider_requested_record_version,
                })
            tool_trace.append(result)
            messages.append({"role": "assistant", "tool_call": {"tool": tool_name, "args": tool_args}})
            messages.append({"role": "tool", "tool": tool_name, "result": provider_tool_payload(result)})
            if hasattr(active_provider, "note_result"):
                active_provider.note_result(result)

            if result.duplicate:
                duplicate_count += 1
                if duplicate_count > budgets.max_duplicate_calls:
                    return _budget_exhausted("duplicate tool calls exceeded the episode budget")

            if result.ok and tool_name in _PROPOSAL_TOOLS:
                proposal_calls += 1
                if proposal_calls > budgets.pending_confirmations:
                    return _budget_exhausted("more than one pending confirmation is not permitted")
                pending_proposal_id = (result.data or {}).get("proposal_id")
            if result.ok and tool_name in _EPISODE_PROPOSAL_TOOLS:
                episode_proposals.append(result.data or {})
            if enforce_xai_sequence and result.ok and tool_name == "request_measured_xai":
                expected_xai_tool = "read_measured_xai"
                active_tools_schema = [spec for spec in _tools_schema_for_intent(request_intent)
                                       if spec["name"] == expected_xai_tool]
            elif enforce_xai_sequence and result.ok and tool_name == "read_measured_xai":
                expected_xai_tool = None
                active_tools_schema = []
            if expected_read_tool is not None and result.ok and tool_name == expected_read_tool:
                expected_read_tool = None
                active_tools_schema = []
            required_tool = expected_xai_tool or expected_read_tool

            route = _classify_route(tool_trace)
            limit = budgets.complex_route_max_tools if route == "complex" else budgets.simple_route_max_tools
            if len(tool_trace) > limit:
                return _budget_exhausted("tool-call budget exhausted for this route")
            grounded = (
                _grounded_tool_summary(result)
                if (not offline_mode or result.tool == "retrieve_case_comparisons") else None
            )
            if grounded is None and result.ok and tool_name == "read_measured_xai" and isinstance(result.data, Mapping):
                text = result.data.get("grounded_summary_text")
                if text:
                    grounded = (str(text), [str(p["id"]) for p in result.propositions])
            if grounded is not None:
                grounded_summary, summary_claims = grounded
                verdict = check_explanation(grounded_summary, summary_claims, _current_record(), tool_trace, budgets)
                checker_history.append(_verdict_dict(verdict))
                if verdict.ok:
                    return _finish(grounded_summary, summary_claims, "answered", checker_verdict=verdict)
            continue

        text = reply.text or ""
        claims = list(reply.claims)
        if required_tool is not None:
            messages.append({
                "role": "system",
                "content": f"required_tool_sequence_incomplete: call {required_tool} before a final answer",
            })
            continue
        record = _current_record()
        verdict = check_explanation(text, claims, record, tool_trace, budgets)
        checker_history.append(_verdict_dict(verdict))
        if verdict.ok:
            hint = getattr(active_provider, "terminal_hint", None)
            if hint:
                return _finish(text, claims, hint, checker_verdict=verdict)
            if pending_proposal_id is not None:
                return _finish(text, claims, "awaiting_confirmation", checker_verdict=verdict)
            return _finish(text, claims, "answered", checker_verdict=verdict)

        if repairs_used < budgets.max_explanation_repairs:
            repairs_used += 1
            messages.append({
                "role": "system",
                "content": (
                    "repair_request: rewrite the entire answer. Do not repeat any rejected span, "
                    "even while denying, quoting, or explaining it. For a comparison-vote rejection, "
                    "say only that retrieved cases are qualitative comparisons and cannot determine "
                    "the current case; describe matched/different fields if useful."
                ),
                "rejections": [dataclasses.asdict(r) for r in verdict.rejections],
            })
            continue

        grounded_rescue = _failure_path_grounded_summary(tool_trace)
        if grounded_rescue is not None:
            rescue_text, rescue_claims = grounded_rescue
            rescue_verdict = check_explanation(rescue_text, rescue_claims, record, tool_trace, budgets)
            checker_history.append(_verdict_dict(rescue_verdict))
            if rescue_verdict.ok:
                return _finish(rescue_text, rescue_claims, "answered", checker_verdict=rescue_verdict)

        fallback_text = _fallback_text(record)
        fallback_verdict = check_explanation(fallback_text, (), record, tool_trace, budgets)
        return _finish(fallback_text, (), "fallback_template", checker_verdict=fallback_verdict)

    return _budget_exhausted("turn budget exhausted")


__all__ = ["TurnResult", "TERMINAL_STATES", "provider_tool_payload", "run_turn"]
