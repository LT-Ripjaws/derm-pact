"""WP-A1: deterministic three-layer renderer v2."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from dermpact.criteria.aggregate import leave_one_criterion_out

from dermpact.core.record_v2 import CHECKLIST_IMPLICATION_PREFIX, StructuredCaseRecordV2
from dermpact.core.gate_v2 import (
    ALL_SECTIONS,
    GateVerdict,
    allowed_claims_v2,
    partial_report_scope,
    required_warnings_v2,
)

#: Which of the three physical layers each logical section renders into.
SECTION_LAYER: Mapping[str, str] = {
    "identity": "top", "warnings": "top", "checklist": "top", "next_step": "top", "agreement": "top",
    "clinician_assertions": "top", "image_status": "top",
    "criterion_evidence": "evidence", "rule_evidence": "evidence",
    "source_spans": "audit", "amendment_history": "audit", "gate_receipt": "audit", "versions": "audit",
}
#: Display order within each layer.
SECTION_ORDER: Tuple[str, ...] = (
    "identity", "image_status", "checklist", "agreement", "clinician_assertions", "next_step", "warnings",
    "criterion_evidence", "rule_evidence",
    "source_spans", "versions", "amendment_history", "gate_receipt",
)

#: Text that must never appear anywhere in a rendered output.
FORBIDDEN_PHRASES: Tuple[str, ...] = (
    "Assessment: excision", "verified diagnosis", "familiar image", "checklist applies",
    "consider", "margin", "survival", "staging",
)

_WARNING_AFFECTED_OUTPUTS: Mapping[str, str] = {
    "image_quality": "the perception head's probabilities, computed from an image the corruption detector flagged before any usability judgement",
    "unfamiliar_image": "none of the diagnostic outputs are changed by this; it is an advisory unfamiliarity note only",
    "unfamiliarity_unassessed": "none of the diagnostic outputs are changed by this; the unfamiliarity check itself could not be run",
    "view_instability": "the recommended view's probabilities, which vary across this case's assessed alternate views",
    "selective_deferral": "the ensemble-calibrated probability average, which is unvalidated for this case",
    "incomplete_reliability": "a required reliability check that was not run for this case",
}

_DEFERRAL_NOT_A_QUEUE_POSITION = (
    "A deferral is a per-member uncertainty flag requesting clinician review; it is not a "
    "position in a review queue and not a completed review."
)


def _member_exceedances(
    values: Optional[Sequence[float]], thresholds_by_fold: Optional[Mapping[str, Any]],
) -> List[Tuple[int, float, float]]:
    """Pairs each per-member/per-fold score with its own frozen threshold (same index -> same fold, exactly how ``ReliabilityBundle.assess_selected_prediction`` built both lists) and returns only the ``(index, score, threshold)`` triples where the record's own score exceeds the record's own threshold -- every number here is a value already stored on the record, never computed."""
    out: List[Tuple[int, float, float]] = []
    for i, score in enumerate(values or ()):
        entry = (thresholds_by_fold or {}).get(str(i))
        threshold = entry.get("threshold") if isinstance(entry, Mapping) else entry
        if isinstance(threshold, (int, float)) and isinstance(score, (int, float)) and score > threshold:
            out.append((i, score, threshold))
    return out


def _warning_measurement_line(code: str, scores: Mapping[str, Any], thresholds: Mapping[str, Any]) -> str:
    """The measured score(s) versus the frozen threshold(s) this warning code is actually keyed to, using only values already present in ``record.reliability`` -- so every number printed is a record leaf :func:`~dermpact.core.gate_v2.allowed_claims_v2` already admits."""
    if code == "image_quality":
        score, threshold = scores.get("synthetic_corruption_probability"), thresholds.get("quality_threshold")
        if score is None or threshold is None:
            return "measured score/threshold not available in this record"
        return f"measured synthetic_corruption_probability={_fmt(score)} vs frozen quality_threshold={_fmt(threshold)}"
    if code in ("unfamiliar_image", "unfamiliarity_unassessed"):
        score, threshold = scores.get("unfamiliarity"), thresholds.get("ood_threshold")
        if score is None or threshold is None:
            return "unfamiliarity was not assessed for this case; no score/threshold to compare"
        return f"measured unfamiliarity={_fmt(score)} vs frozen ood_threshold={_fmt(threshold)}"
    if code == "selective_deferral":
        exceeded = _member_exceedances(scores.get("member_uncertainty"), thresholds.get("per_member_acceptance_thresholds"))
        if not exceeded:
            return "measured member uncertainty did not exceed any frozen per-member acceptance threshold in this record's own scores"
        return "measured " + "; ".join(
            f"member {i} uncertainty={_fmt(s)} vs its frozen acceptance_threshold={_fmt(t)}" for i, s, t in exceeded
        )
    if code == "view_instability":
        exceeded = _member_exceedances(scores.get("member_view_shifts"), thresholds.get("per_fold_view_thresholds"))
        if not exceeded:
            return "measured view shift did not exceed any frozen per-fold view threshold in this record's own scores"
        return "measured " + "; ".join(
            f"member {i} view_shift={_fmt(s)} vs its frozen view threshold={_fmt(t)}" for i, s, t in exceeded
        )
    if code == "incomplete_reliability":
        return "no measured score applies; a required reliability check was not run for this case"
    return "no measured score/threshold mapping is defined for this warning code"


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


def _claims_for_section(record: StructuredCaseRecordV2, section: str) -> Tuple[Tuple[str, Any], ...]:
    allowed = allowed_claims_v2(record).claims
    if section == "identity":
        return _select(allowed, include=("identity",))
    if section == "warnings":
        return ()  # warning text is host-computed, not a record claim
    if section == "image_status":
        return _select(allowed, include=("image_status",))
    if section == "checklist":
        return _select(
            allowed,
            include=("checklist", "perception.diagnosis_head", "perception.melanoma_head", "perception.abstained",
                     "derived.diagnosis_head", "derived.melanoma_head"),
            exclude_contains=(".per_criterion",),
        )
    if section == "next_step":
        return _select(allowed, include=("next_step",))
    if section == "agreement":
        return _select(allowed, include=("agreement",))
    if section == "clinician_assertions":
        return _select(allowed, include=("clinician_assertions",))
    if section == "criterion_evidence":
        return _select(
            allowed,
            include=("checklist.per_criterion", "perception.criterion_probabilities", "perception.criterion_values",
                     "perception.model_criterion_values", "derived.criterion_probabilities"),
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


def _section_identity(record: StructuredCaseRecordV2) -> str:
    i = record.identity
    return (
        f"Case {i.case_id} -- record version {i.record_version} (created {i.created_at}, "
        f"schema {i.schema_version}, source {i.source_label})"
    )


def _section_image_status(record: StructuredCaseRecordV2) -> str:
    s = record.image_status
    return (
        f"Image status: {s.state} | reasons: {', '.join(s.reasons) if s.reasons else 'none'} "
        f"| recovery: {s.recovery_action}"
    )


def _section_checklist(record: StructuredCaseRecordV2) -> str:
    dh = record.perception.diagnosis_head
    mh = record.perception.melanoma_head
    lines = [
        f"Direct diagnosis head: {dh['provisional_class']} (aggregation: {dh['aggregation']})",
        f"Melanoma head probability: {_fmt(mh['probability'])} (aggregation: {mh['aggregation']}; "
        "a separate binary head, distinct from the diagnosis head's own melanoma-class probability)",
    ]
    c = record.checklist
    if c.decidable:
        lines.append(f"Seven-point checklist: score {c.score_min} (meets_threshold={c.meets_threshold})")
    else:
        lines.append(
            f"Seven-point checklist: undecidable, score range [{c.score_min},{c.score_max}]; "
            f"indeterminate criteria: {', '.join(c.indeterminate_criteria) if c.indeterminate_criteria else 'none'}"
        )
    return "\n".join(lines)


def _section_agreement(record: StructuredCaseRecordV2) -> str:
    a = record.agreement
    return f"Direct-classifier vs checklist agreement: present={a.present} escalate={a.escalate}\n  {a.description}"


def _section_clinician_assertions(record: StructuredCaseRecordV2) -> str:
    lines = ["Clinician assertions:"]
    for fact, entry in sorted(record.clinician_assertions.items()):
        prov = entry.get("provenance", {})
        lines.append(f"  - {fact}: value={entry.get('value')!r} source={prov.get('source')} statement={prov.get('statement', '')!r}")
    return "\n".join(lines)


def _section_next_step(record: StructuredCaseRecordV2) -> str:
    return f"{record.next_step.display_text}\nScope: {record.next_step.scope_note}"


def _warning_entries(record: StructuredCaseRecordV2) -> List[Dict[str, Any]]:
    """Every active warning enriched with its measured-score-vs-threshold line and fixed affected-outputs text."""
    scores = record.reliability.scores
    thresholds = record.reliability.thresholds
    entries: List[Dict[str, Any]] = []
    for w in required_warnings_v2(record):
        entry: Dict[str, Any] = {
            "code": w.code,
            "reason": w.reason,
            "measurement": _warning_measurement_line(w.code, scores, thresholds),
            "affected_outputs": _WARNING_AFFECTED_OUTPUTS.get(w.code, "not documented for this code"),
            "recovery_action": w.recovery_action,
            "remaining_usable_output": w.remaining_usable_output,
        }
        if w.code == "selective_deferral":
            entry["deferral_note"] = _DEFERRAL_NOT_A_QUEUE_POSITION
        entries.append(entry)
    return entries


def _section_warnings(record: StructuredCaseRecordV2) -> str:
    entries = _warning_entries(record)
    if not entries:
        return "Warnings: none"
    lines = ["Warnings:"]
    for e in entries:
        lines.append(
            f"  - [{e['code']}] {e['reason']} | {e['measurement']} | affected outputs: {e['affected_outputs']} "
            f"| remaining usable output: {e['remaining_usable_output']} | recovery: {e['recovery_action']}"
        )
        if "deferral_note" in e:
            lines.append(f"      {e['deferral_note']}")
    return "\n".join(lines)


def _section_criterion_evidence(record: StructuredCaseRecordV2) -> str:
    lines = ["Per-criterion checklist evidence:"]
    model_values = record.perception.model_criterion_values
    loco = leave_one_criterion_out(record.perception.criterion_values) if record.checklist.decidable else {}
    for c in record.checklist.per_criterion:
        prob = record.perception.criterion_probabilities.get(c.name)
        prob_text = f" probability={_fmt(prob)}" if prob is not None else ""
        source = "model" if model_values.get(c.name) == c.value else "amended"
        sensitivity = ""
        if c.name in loco:
            entry = loco[c.name]
            sensitivity = (
                f" | checklist arithmetic if read otherwise: score_range={entry['score_range']} "
                f"still_decidable={entry['still_decidable']} verdict_changed={entry['verdict_changed']}"
            )
        lines.append(
            f"  - {c.name}: value={c.value} points={c.points}/{c.max_points} "
            f"determinate={c.determinate} source={source}{prob_text}{sensitivity}"
        )
    imp = record.checklist.management_implication
    lines.append(
        f"Checklist arithmetic implication (not a clinical next step): action={imp['action']} reason={imp['reason']}"
    )
    lines.append(
        "Model's own criterion reading (frozen, never amended): "
        + ", ".join(f"{name}={model_values[name]}" for name in sorted(model_values))
    )
    return "\n".join(lines)


def _section_rule_evidence(record: StructuredCaseRecordV2) -> str:
    lines = ["Guideline rule evidence:"]
    for row in record.applicability.get("results", []):
        lines.append(
            f"  - {row['rule_id']}: state={row['state']} reason={row['reason_code']} "
            f"decisive_missing_fact={row.get('decisive_missing_fact')} question={row.get('question')}"
        )
        path = row.get("rule_path") or []
        if path:
            lines.append("      rule_path: " + "; ".join(f"{p['predicate']}={p['outcome']}" for p in path))
    return "\n".join(lines)


def _section_source_spans(record: StructuredCaseRecordV2) -> str:
    lines = ["Rule source spans (verbatim):"]
    for row in record.applicability.get("results", []):
        src = row.get("source", {})
        lines.append(
            f"  - {row['rule_id']} :: document={src.get('document_id')} page={src.get('pdf_page')} "
            f"span_sha256={src.get('span_sha256')}"
        )
        lines.append(f"      {src.get('span_text', '')}")
    return "\n".join(lines)


def _section_versions(record: StructuredCaseRecordV2) -> str:
    lines = ["Version pins:"]
    for name, value in record.versions.to_dict().items():
        lines.append(f"  - {name}: {value}")
    lines.append(f"record_sha256={record.record_sha256}")
    lines.append(f"parent_record_sha256={record.identity.parent_record_sha256}")
    return "\n".join(lines)


def _section_amendment_history(record: StructuredCaseRecordV2) -> str:
    n = len(record.amendments)
    lines = [f"Amendment history ({n} entr{'y' if n == 1 else 'ies'}):"]
    for a in record.amendments:
        lines.append(
            f"  - amendment_id={a.amendment_id} kind={a.kind} field={a.field} before={a.before!r} after={a.after!r} "
            f"actor={a.actor} session_id={a.session_id} proposed_at={a.proposed_at} committed_at={a.committed_at} "
            f"token_sha256={a.token_sha256} consequence_preview_sha256={a.consequence_preview_sha256}"
        )
        rc = a.recomputed_consequences or {}
        lines.append(
            "      consequence diff: "
            f"next_step {rc.get('next_step_before')} -> {rc.get('next_step_after')} "
            f"(next_step_changed={rc.get('next_step_changed')}); "
            f"checklist {rc.get('checklist_before')} -> {rc.get('checklist_after')}"
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
    "image_status": lambda r, v: _section_image_status(r),
    "checklist": lambda r, v: _section_checklist(r),
    "agreement": lambda r, v: _section_agreement(r),
    "clinician_assertions": lambda r, v: _section_clinician_assertions(r),
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
    record: StructuredCaseRecordV2, verdict: Optional[GateVerdict], *, sections: Sequence[str], leading: str = "",
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
            warning_codes = tuple(w.code for w in required_warnings_v2(record))
    layer_text = {k: "\n\n".join(v) for k, v in layers.items()}
    combined = "\n\n".join([layer_text["top"], layer_text["evidence"], layer_text["audit"]])
    text_sha256 = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return RenderedOutput(layers=layer_text, claims=tuple(sorted(set(claims))), warnings=warning_codes, text_sha256=text_sha256)


def render(record: StructuredCaseRecordV2, verdict: GateVerdict, *, lineage: Sequence[StructuredCaseRecordV2] = ()) -> RenderedOutput:
    """Full three-layer render (every section)."""
    return _build(record, verdict, sections=ALL_SECTIONS)


def render_partial(record: StructuredCaseRecordV2, verdict: GateVerdict) -> RenderedOutput:
    """Only the sections :func:`~dermpact.core.gate_v2.partial_report_scope` allows."""
    scope = partial_report_scope(verdict, record)
    allowed_sections = {s for s, ok in scope.items() if ok}
    failed_codes = sorted({f.code for f in verdict.failures})
    leading = f"PARTIAL REPORT -- failed check(s): {', '.join(failed_codes) if failed_codes else 'none'}"
    return _build(record, verdict, sections=allowed_sections, leading=leading)


def render_refusal(verdict: GateVerdict, expected_case_id: str) -> RenderedOutput:
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


def render_question(record: StructuredCaseRecordV2) -> Optional[str]:
    return record.next_step.targeted_question




def view_model(record: StructuredCaseRecordV2, verdict: GateVerdict) -> Dict[str, Any]:
    """JSON-shaped UI view model. Pure function of ``(record, verdict)``."""
    per_criterion = []
    loco = leave_one_criterion_out(record.perception.criterion_values) if record.checklist.decidable else {}
    model_values = record.perception.model_criterion_values
    for c in record.checklist.per_criterion:
        entry: Dict[str, Any] = {
            "name": c.name, "value": c.value, "points": c.points, "max_points": c.max_points,
            "determinate": c.determinate, "source": ("model" if model_values.get(c.name) == c.value else "amended"),
            "probability": record.perception.criterion_probabilities.get(c.name),
        }
        if c.name in loco:
            h = loco[c.name]
            entry["sensitivity"] = {
                "label": "checklist arithmetic, not diagnosis attribution",
                "score_range_if_read_otherwise": list(h["score_range"]),
                "still_decidable": h["still_decidable"],
                "verdict_changed": h["verdict_changed"],
            }
        else:
            entry["sensitivity"] = None
        per_criterion.append(entry)

    applicability_rows = []
    for row in record.applicability.get("results", []):
        applicability_rows.append({
            "rule_id": row["rule_id"], "state": row["state"], "reason": row["reason_code"],
            "decisive_fact": row.get("decisive_missing_fact"), "question": row.get("question"),
            "source_span_id": (row.get("source", {}) or {}).get("span_sha256"),
        })

    sources = []
    for row in record.applicability.get("results", []):
        src = row.get("source", {}) or {}
        sources.append({
            "rule_id": row["rule_id"], "span_text": src.get("span_text"), "page": src.get("pdf_page"),
            "span_sha256": src.get("span_sha256"),
            "validity_window": {"valid_from": (row.get("checks") or {}).get("source_current")},
        })

    return {
        "identity": record.identity.to_dict(),
        "heads": {
            "diagnosis_head": record.perception.diagnosis_head, "melanoma_head": record.perception.melanoma_head,
            "perception_mode": record.perception.perception_mode,
        },
        "checklist": {
            "score_min": record.checklist.score_min, "score_max": record.checklist.score_max,
            "decidable": record.checklist.decidable, "meets_threshold": record.checklist.meets_threshold,
            "per_criterion": per_criterion, "checklist_implication_label": "checklist arithmetic",
        },
        "clinician_assertions": record.clinician_assertions,
        "reliability": {
            "iqa_state": record.reliability.iqa_state, "ood_state": record.reliability.ood_state,
            "view_stability_state": record.reliability.view_stability_state,
            "selective_abstention": record.reliability.selective_abstention,
            "disposition": record.reliability.disposition, "thresholds": record.reliability.thresholds,
            "meaning": record.reliability.meaning,
        },
        "image_status": record.image_status.to_dict(),
        "next_step": record.next_step.to_dict(),
        "applicability": applicability_rows,
        "sources": sources,
        "versions": record.versions.to_dict(),
        "amendments": [a.to_dict() for a in record.amendments],
        "warnings": _warning_entries(record),
    }


__all__ = [
    "FORBIDDEN_PHRASES",
    "RenderedOutput",
    "SECTION_LAYER",
    "SECTION_ORDER",
    "render",
    "render_partial",
    "render_question",
    "render_refusal",
    "view_model",
]
