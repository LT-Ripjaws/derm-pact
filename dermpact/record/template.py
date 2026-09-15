"""Phase 5: deterministic three-layer renderer (CONTEXT.md S15.7)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from dermpact.record.case_record import StructuredCaseRecord
from dermpact.record.gate import (
    ALL_SECTIONS,
    GateVerdict,
    allowed_claims,
    partial_report_scope,
    required_warnings,
)

#: Which of the three physical layers each logical section renders into.
SECTION_LAYER: Mapping[str, str] = {
    "identity": "top", "warnings": "top", "checklist": "top", "next_step": "top", "agreement": "top",
    "criterion_evidence": "evidence", "rule_evidence": "evidence",
    "source_spans": "audit", "amendment_history": "audit", "gate_receipt": "audit", "versions": "audit",
}
#: Display order within each layer.
SECTION_ORDER: Tuple[str, ...] = (
    "identity", "checklist", "agreement", "next_step", "warnings",
    "criterion_evidence", "rule_evidence",
    "source_spans", "versions", "amendment_history", "gate_receipt",
)


@dataclass(frozen=True)
class RenderedOutput:
    layers: Dict[str, str]
    claims: Tuple[Tuple[str, Any], ...]
    warnings: Tuple[str, ...]
    text_sha256: str




def _path_matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")


def _select(allowed: FrozenSet[Tuple[str, Any]], *, include: Sequence[str], exclude_contains: Sequence[str] = ()) -> Tuple[Tuple[str, Any], ...]:
    out: List[Tuple[str, Any]] = []
    for path, value in allowed:
        if not any(_path_matches(path, pre) for pre in include):
            continue
        if any(sub in path for sub in exclude_contains):
            continue
        out.append((path, value))
    return tuple(sorted(out))


def _claims_for_section(record: StructuredCaseRecord, section: str) -> Tuple[Tuple[str, Any], ...]:
    allowed = allowed_claims(record).claims
    if section == "identity":
        return _select(allowed, include=("identity",))
    if section == "warnings":
        return ()  # warning text is host-computed, not a record claim
    if section == "checklist":
        return _select(
            allowed,
            include=("checklist", "perception.provisional_class", "perception.melanoma_probability", "perception.abstained"),
            exclude_contains=(".per_criterion",),
        )
    if section == "next_step":
        return _select(allowed, include=("next_step",))
    if section == "agreement":
        return _select(allowed, include=("agreement",))
    if section == "criterion_evidence":
        return _select(
            allowed,
            include=(
                "checklist.per_criterion", "perception.criterion_probabilities",
                "perception.criterion_values", "perception.model_criterion_values",
            ),
        )
    if section == "rule_evidence":
        return _select(allowed, include=("applicability.results",), exclude_contains=(".source.",))
    if section == "source_spans":
        out = [(p, v) for p, v in allowed if p.startswith("applicability.results[") and ".source." in p]
        return tuple(sorted(out))
    if section == "versions":
        return _select(allowed, include=("versions", "record_sha256"))
    if section == "amendment_history":
        return _select(allowed, include=("amendments",))
    if section == "gate_receipt":
        return ()  # the verdict is not part of the record; nothing here is a record claim
    raise KeyError(section)  # pragma: no cover




def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _section_identity(record: StructuredCaseRecord) -> str:
    return (
        f"Case {record.identity.case_id} -- record version {record.identity.record_version} "
        f"(created {record.identity.created_at}, schema {record.identity.schema_version})"
    )


def _section_checklist(record: StructuredCaseRecord) -> str:
    lines = [
        f"Provisional assessment: {record.perception.provisional_class} "
        f"(uncalibrated melanoma probability: {_fmt(record.perception.melanoma_probability)})",
    ]
    c = record.checklist
    if c.decidable:
        lines.append(f"Seven-point checklist: score {c.score_min} (meets_threshold={c.meets_threshold})")
    else:
        lines.append(
            f"Seven-point checklist: undecidable, score range [{c.score_min},{c.score_max}]; "
            f"indeterminate criteria: {', '.join(c.indeterminate_criteria) if c.indeterminate_criteria else 'none'}"
        )
    lines.append(
        "Reliability: iqa={iqa}, ood={ood}, view_stability={vs}, selective_abstention={sa}, "
        "calibration={cal}".format(
            iqa=record.reliability.iqa_state, ood=record.reliability.ood_state,
            vs=record.reliability.view_stability_state, sa=record.reliability.selective_abstention,
            cal=record.reliability.calibration_state,
        )
    )
    return "\n".join(lines)


def _section_agreement(record: StructuredCaseRecord) -> str:
    a = record.agreement
    return f"Direct-classifier vs checklist agreement: present={a.present} escalate={a.escalate}\n  {a.description}"


def _section_next_step(record: StructuredCaseRecord) -> str:
    return record.next_step.display_text


def _section_warnings(record: StructuredCaseRecord) -> str:
    warnings = required_warnings(record)
    if not warnings:
        return "Warnings: none"
    lines = ["Warnings:"]
    for w in warnings:
        lines.append(f"  - [{w.code}] {w.reason} | remaining usable output: {w.remaining_usable_output} | recovery: {w.recovery_action}")
    return "\n".join(lines)


def _section_criterion_evidence(record: StructuredCaseRecord) -> str:
    lines = ["Per-criterion checklist evidence:"]
    model_values = record.perception.model_criterion_values
    for c in record.checklist.per_criterion:
        prob = record.perception.criterion_probabilities.get(c.name)
        prob_text = f" probability={_fmt(prob)}" if prob is not None else ""
        source = "model" if model_values.get(c.name) == c.value else "amended"
        lines.append(
            f"  - {c.name}: value={c.value} points={c.points}/{c.max_points} "
            f"determinate={c.determinate} source={source}{prob_text}"
        )
    action = record.checklist.management_implication.get("action", "")
    reason = record.checklist.management_implication.get("reason", "")
    lines.append(f"Checklist-implied management (informational only, not the displayed next step): {action} -- {reason}")
    lines.append(
        "Model's own criterion reading (frozen, never amended): "
        + ", ".join(f"{name}={model_values[name]}" for name in sorted(model_values))
    )
    return "\n".join(lines)


def _section_rule_evidence(record: StructuredCaseRecord) -> str:
    lines = ["Guideline rule evidence:"]
    for row in record.applicability.get("results", []):
        lines.append(
            f"  - {row['rule_id']}: state={row['state']} reason={row['reason_code']} "
            f"decisive_missing_fact={row.get('decisive_missing_fact')}"
        )
        path = row.get("rule_path") or []
        if path:
            lines.append("      rule_path: " + "; ".join(f"{p['predicate']}={p['outcome']}" for p in path))
    return "\n".join(lines)


def _section_source_spans(record: StructuredCaseRecord) -> str:
    lines = ["Rule source spans (verbatim):"]
    for row in record.applicability.get("results", []):
        src = row.get("source", {})
        lines.append(
            f"  - {row['rule_id']} :: document={src.get('document_id')} page={src.get('pdf_page')} "
            f"span_sha256={src.get('span_sha256')}"
        )
        lines.append(f"      {src.get('span_text', '')}")
    return "\n".join(lines)


def _section_versions(record: StructuredCaseRecord) -> str:
    lines = ["Version pins:"]
    for name, value in record.versions.to_dict().items():
        lines.append(f"  - {name}: {value}")
    lines.append(f"record_sha256={record.record_sha256}")
    lines.append(f"parent_record_sha256={record.identity.parent_record_sha256}")
    return "\n".join(lines)


def _section_amendment_history(record: StructuredCaseRecord) -> str:
    n = len(record.amendments)
    lines = [f"Amendment history ({n} entr{'y' if n == 1 else 'ies'}):"]
    for a in record.amendments:
        lines.append(
            f"  - amendment_id={a.amendment_id} kind={a.kind} field={a.field} "
            f"before={a.before!r} after={a.after!r} actor={a.actor} "
            f"proposed_at={a.proposed_at} committed_at={a.committed_at} token_sha256={a.token_sha256} "
            f"consequence_preview_sha256={a.consequence_preview_sha256}"
        )
        rc = a.recomputed_consequences or {}
        lines.append(
            "      consequence diff: "
            f"next_step {rc.get('next_step_before')} -> {rc.get('next_step_after')} "
            f"(next_step_changed={rc.get('next_step_changed')}); "
            f"checklist {rc.get('checklist_before')} -> {rc.get('checklist_after')}; "
            f"agreement {rc.get('agreement_before')} -> {rc.get('agreement_after')}"
        )
    return "\n".join(lines)


def _section_gate_receipt(verdict: GateVerdict) -> str:
    lines = [
        f"Gate receipt: policy_version={verdict.policy_version} checked_at={verdict.checked_at} "
        f"disposition={verdict.disposition} passed={verdict.passed}",
    ]
    for f in verdict.failures:
        lines.append(f"  * FAILURE[{f.code}]: {f.detail}")
    lines.append(f"  receipt_sha256={verdict.receipt_sha256}")
    return "\n".join(lines)


_SECTION_TEXT = {
    "identity": lambda r, v: _section_identity(r),
    "checklist": lambda r, v: _section_checklist(r),
    "agreement": lambda r, v: _section_agreement(r),
    "next_step": lambda r, v: _section_next_step(r),
    "warnings": lambda r, v: _section_warnings(r),
    "criterion_evidence": lambda r, v: _section_criterion_evidence(r),
    "rule_evidence": lambda r, v: _section_rule_evidence(r),
    "source_spans": lambda r, v: _section_source_spans(r),
    "versions": lambda r, v: _section_versions(r),
    "amendment_history": lambda r, v: _section_amendment_history(r),
    "gate_receipt": lambda r, v: _section_gate_receipt(v),
}


def _build(
    record: StructuredCaseRecord, verdict: Optional[GateVerdict], *, sections: Sequence[str], leading: str = "",
) -> RenderedOutput:
    layers: Dict[str, List[str]] = {"top": [], "evidence": [], "audit": []}
    if leading:
        layers["top"].append(leading)
    claims: List[Tuple[str, Any]] = []
    warning_codes: Tuple[str, ...] = ()
    for name in SECTION_ORDER:
        if name not in sections:
            continue
        text = _SECTION_TEXT[name](record, verdict)
        layers[SECTION_LAYER[name]].append(text)
        claims.extend(_claims_for_section(record, name))
        if name == "warnings":
            warning_codes = tuple(w.code for w in required_warnings(record))
    layer_text = {k: "\n\n".join(v) for k, v in layers.items()}
    combined = "\n\n".join([layer_text["top"], layer_text["evidence"], layer_text["audit"]])
    text_sha256 = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return RenderedOutput(
        layers=layer_text, claims=tuple(sorted(set(claims))), warnings=warning_codes, text_sha256=text_sha256,
    )


def render(record: StructuredCaseRecord, verdict: GateVerdict, *, lineage: Sequence[StructuredCaseRecord] = ()) -> RenderedOutput:
    """Full three-layer render (every section)."""
    return _build(record, verdict, sections=ALL_SECTIONS)


def render_partial(record: StructuredCaseRecord, verdict: GateVerdict) -> RenderedOutput:
    """Only the sections :func:`~dermpact.record.gate.partial_report_scope` allows, with a leading line naming the failed check(s)."""
    scope = partial_report_scope(verdict, record)
    allowed_sections = {s for s, ok in scope.items() if ok}
    failed_codes = sorted({f.code for f in verdict.failures})
    leading = f"PARTIAL REPORT -- failed check(s): {', '.join(failed_codes) if failed_codes else 'none'}"
    return _build(record, verdict, sections=allowed_sections, leading=leading)


def render_refusal(verdict: GateVerdict, expected_case_id: str) -> RenderedOutput:
    """A refusal carrying no case content beyond the requested id."""
    failed_codes = sorted({f.code for f in verdict.failures})
    text = (
        f"REFUSAL -- cannot display case {expected_case_id}.\n"
        f"Failed check(s): {', '.join(failed_codes) if failed_codes else 'none'}\n"
        f"policy_version={verdict.policy_version} checked_at={verdict.checked_at} receipt_sha256={verdict.receipt_sha256}"
    )
    return RenderedOutput(
        layers={"top": text, "evidence": "", "audit": ""}, claims=(), warnings=(),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def render_question(record: StructuredCaseRecord) -> Optional[str]:
    """The single targeted question, or ``None`` when the record carries no pending question."""
    return record.next_step.targeted_question


__all__ = [
    "RenderedOutput",
    "SECTION_LAYER",
    "SECTION_ORDER",
    "render",
    "render_partial",
    "render_question",
    "render_refusal",
]
