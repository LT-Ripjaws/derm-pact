"""WP-AG: deterministic offline intent router."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from dermpact.core.agent.providers import ProviderReply
from dermpact.core.agent.tools import ToolContext, ToolResult

INTENT_NAMES: Tuple[str, ...] = (
    "out_of_scope", "reacquisition_request", "review_request", "ask_first", "supply_assertion",
    "correct_criterion", "clinical_photo", "xai_region", "similar_cases", "what_remains", "source_quote",
    "why_deferred", "explain", "unknown",
)

_REACQUISITION_PHRASES: Tuple[str, ...] = (
    "image is poor", "poor image", "bad image quality", "image quality is poor",
    "retake the image", "reacquire the image", "should we retake", "should we reacquire",
    "image needs to be retaken", "need a new image", "image is not usable",
)
_REVIEW_PHRASES: Tuple[str, ...] = (
    "flag this case for clinician review", "flag this case for review", "flag it for review",
    "flag for review", "please flag this case", "please review this case",
    "send this case for review", "escalate this case for review", "needs clinician review",
    "requires clinician review", "request clinician review", "clinician should review this case",
    "this case needs review",
)
_ASK_FIRST_PHRASES: Tuple[str, ...] = (
    "what do you need", "what do you still need", "what information do you need",
    "what is missing", "what's missing", "which question", "what should i tell you",
    "what should i answer", "what do you want to know",
)

_TOPIC_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("suspect", "clinician.melanoma_suspected"),
    ("equivocal", "clinician.equivocal"),
    ("high total nevus", "patient.high_total_nevus_count"),
    ("high nevus", "patient.high_total_nevus_count"),
    ("high risk", "patient.high_risk_status"),
    ("personal history of melanoma", "patient.personal_melanoma_history"),
    ("melanoma history", "patient.personal_melanoma_history"),
    ("genetic predisposition", "patient.genetic_predisposition"),
    ("rcm", "site.rcm_available"),
    ("confocal", "site.rcm_available"),
    ("under assessment", "lesion.under_assessment"),
)

_VALUE_PHRASES: Tuple[Tuple[str, str, bool], ...] = (
    ("melanoma is not suspected", "clinician.melanoma_suspected", False),
    ("melanoma isn't suspected", "clinician.melanoma_suspected", False),
    ("melanoma not suspected", "clinician.melanoma_suspected", False),
    ("melanoma is suspected", "clinician.melanoma_suspected", True),
    ("melanoma suspected", "clinician.melanoma_suspected", True),
    ("not equivocal", "clinician.equivocal", False),
    ("isn't equivocal", "clinician.equivocal", False),
    ("equivocal", "clinician.equivocal", True),
    ("not high risk", "patient.high_risk_status", False),
    ("high risk", "patient.high_risk_status", True),
    ("not a high total nevus count", "patient.high_total_nevus_count", False),
    ("no high total nevus count", "patient.high_total_nevus_count", False),
    ("high total nevus count", "patient.high_total_nevus_count", True),
    ("high nevus count", "patient.high_total_nevus_count", True),
    ("no personal history of melanoma", "patient.personal_melanoma_history", False),
    ("personal history of melanoma", "patient.personal_melanoma_history", True),
    ("no melanoma history", "patient.personal_melanoma_history", False),
    ("melanoma history", "patient.personal_melanoma_history", True),
    ("no genetic predisposition", "patient.genetic_predisposition", False),
    ("genetic predisposition", "patient.genetic_predisposition", True),
    ("rcm is not available", "site.rcm_available", False),
    ("rcm not available", "site.rcm_available", False),
    ("rcm available", "site.rcm_available", True),
    ("reflectance confocal microscopy is not available", "site.rcm_available", False),
    ("reflectance confocal microscopy", "site.rcm_available", True),
    ("not under assessment", "lesion.under_assessment", False),
    ("under assessment", "lesion.under_assessment", True),
)

_CRITERION_PHRASES: Tuple[Tuple[str, str, str], ...] = (
    ("veil is absent", "blue_whitish_veil", "absent"),
    ("veil is present", "blue_whitish_veil", "present"),
    ("streaks are irregular", "streaks", "irregular"),
    ("streaks irregular", "streaks", "irregular"),
    ("pigment network is atypical", "pigment_network", "atypical"),
    ("pigment network atypical", "pigment_network", "atypical"),
    ("pigment network is typical", "pigment_network", "typical"),
    ("pigment network typical", "pigment_network", "typical"),
)

_UNKNOWN_WORDS = ("don't know", "do not know", "not sure", "unknown", "unavailable", "no idea")
_HYPOTHETICAL_WORDS = ("what if", "suppose", "hypothetically", "hypothetical")

_RULE_ID_RE = re.compile(r"eado-mel-p1-2024:r\d+", re.IGNORECASE)

_INTENT_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("out_of_scope", ("treatment", "margin", "staging", "prognosis", "survival")),
    ("reacquisition_request", _REACQUISITION_PHRASES),
    ("review_request", _REVIEW_PHRASES),
    ("ask_first", _ASK_FIRST_PHRASES),
    ("supply_assertion", tuple(k for k, _ in _TOPIC_KEYWORDS)),
    ("correct_criterion", tuple(p for p, _, _ in _CRITERION_PHRASES)),
    ("clinical_photo", ("clinical photo", "photo contribution")),
    ("xai_region", ("heatmap", "which region", "measured influence", "which area", "sensitivity map")),
    ("similar_cases", ("similar case", "previous case", "comparable case", "prior case",
                       "retrieved case", "majority diagnosis", "majority verdict",
                       "consensus", "vote", "average the outcomes")),
    ("what_remains", ("what remains", "resume", "status of this case", "what's left")),
    ("source_quote", ("source", "quote", "which guideline", "cite the")),
    ("why_deferred", ("deferred", "reliability", "why review", "flagged")),
    ("explain", ("why", "explain", "evidence", "proposition", "melanoma probability")),
)


def classify_intent(message: str) -> str:
    lower = message.lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if any(k in lower for k in keywords):
            return intent
    return "unknown"


def _parse_fact_statement(message: str) -> Optional[Tuple[str, Optional[bool], bool, bool]]:
    """Returns ``(fact, value_or_None, is_hypothetical, is_unknown)``."""
    lower = message.lower()
    is_hypothetical = any(w in lower for w in _HYPOTHETICAL_WORDS)
    is_unknown = any(w in lower for w in _UNKNOWN_WORDS)

    for phrase, fact, value in _VALUE_PHRASES:
        if phrase in lower:
            if is_unknown:
                return fact, None, is_hypothetical, True
            return fact, value, is_hypothetical, False

    if is_unknown:
        for keyword, fact in _TOPIC_KEYWORDS:
            if keyword in lower:
                return fact, None, is_hypothetical, True
    return None


def _parse_criterion_statement(message: str) -> Optional[Tuple[str, str]]:
    lower = message.lower()
    for phrase, criterion, value in _CRITERION_PHRASES:
        if phrase in lower:
            return criterion, value
    return None


def _parse_rule_id(message: str) -> Optional[str]:
    m = _RULE_ID_RE.search(message)
    if m is None:
        return None
    return "eado-mel-p1-2024:" + m.group(0).split(":")[-1].upper()


_WARNING_AFFECTED_OUTPUTS: Dict[str, str] = {
    "image_quality": "the perception head's probabilities, computed from an image the corruption detector flagged before any usability judgement",
    "unfamiliar_image": "none of the diagnostic outputs are changed by this; it is an advisory unfamiliarity note only",
    "unfamiliarity_unassessed": "none of the diagnostic outputs are changed by this; the unfamiliarity check itself could not be run",
    "view_instability": "the recommended view's probabilities, which vary across this case's assessed alternate views",
    "selective_deferral": "the ensemble-calibrated probability average, which is unvalidated for this case",
    "incomplete_reliability": "a required reliability check that was not run for this case",
}


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
    """The measured score(s) versus the frozen threshold(s) this warning code is actually keyed to, using only values already present in ``record.reliability`` -- so every number the caller prints is a leaf ``allowed_claims_v2`` already admits."""
    if code == "image_quality":
        score, threshold = scores.get("synthetic_corruption_probability"), thresholds.get("quality_threshold")
        if score is None or threshold is None:
            return "measured score/threshold not available in this record"
        return f"measured synthetic_corruption_probability={score} vs frozen quality_threshold={threshold}"
    if code in ("unfamiliar_image", "unfamiliarity_unassessed"):
        score, threshold = scores.get("unfamiliarity"), thresholds.get("ood_threshold")
        if score is None or threshold is None:
            return "unfamiliarity was not assessed for this case; no score/threshold to compare"
        return f"measured unfamiliarity={score} vs frozen ood_threshold={threshold}"
    if code == "selective_deferral":
        exceeded = _member_exceedances(scores.get("member_uncertainty"), thresholds.get("per_member_acceptance_thresholds"))
        if not exceeded:
            return "measured member uncertainty did not exceed any frozen per-member acceptance threshold in this record's own scores"
        return "measured " + "; ".join(f"member {i} uncertainty={s} vs its frozen acceptance_threshold={t}" for i, s, t in exceeded)
    if code == "view_instability":
        exceeded = _member_exceedances(scores.get("member_view_shifts"), thresholds.get("per_fold_view_thresholds"))
        if not exceeded:
            return "measured view shift did not exceed any frozen per-fold view threshold in this record's own scores"
        return "measured " + "; ".join(f"member {i} view_shift={s} vs its frozen view threshold={t}" for i, s, t in exceeded)
    if code == "incomplete_reliability":
        return "no measured score applies; a required reliability check was not run for this case"
    return "no measured score/threshold mapping is defined for this warning code"


def _proposition_lookup(propositions: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """``path_or_ref`` -> proposition id, so a finalize function can cite the exact proposition backing a value it prints without re-deriving ids."""
    return {str(p["path_or_ref"]): str(p["id"]) for p in propositions}


def _comparison_match_line(
    path_prefix: str, match: Mapping[str, Any], lookup: Mapping[str, str], claims: List[str],
) -> str:
    """One line for a ``retrieve_case_comparisons`` match or counterexample: similarity plus matched/different fields together, and the historical outcome's own evidence grade -- never a claim that it proves or supports the current diagnosis."""
    sim_pid = lookup.get(f"{path_prefix}.similarity")
    if sim_pid:
        claims.append(sim_pid)
    outcome = match.get("outcome") or {}
    return (
        f"  - episode {match['episode_id']} (case {match.get('case_id')}), similarity={match['similarity']}: "
        f"matched fields {match['matched_fields']}, different fields {match['different_fields']}; "
        f"historical outcome={outcome.get('diagnosis')} (evidence_grade={outcome.get('evidence_grade')})"
    )


@dataclass
class OfflineTurn:
    intent: str
    next_step: Callable[[List[ToolResult]], Optional[Dict[str, Any]]]
    finalize: Callable[[List[ToolResult]], Tuple[str, Tuple[str, ...]]]
    terminal_hint: Optional[str] = None


def _fixed_sequence(steps: Sequence[Dict[str, Any]]) -> Callable[[List[ToolResult]], Optional[Dict[str, Any]]]:
    steps = list(steps)

    def next_step(collected: List[ToolResult]) -> Optional[Dict[str, Any]]:
        i = len(collected)
        return steps[i] if i < len(steps) else None

    return next_step


def _snapshot(ctx: ToolContext) -> Tuple[Any, int, str]:
    record = ctx.store.load_case(ctx.case_id).current_record
    return record, record.identity.record_version, ctx.case_id


def plan_for(intent: str, message: str, ctx: ToolContext) -> OfflineTurn:
    record, rv, case_id = _snapshot(ctx)

    if intent == "out_of_scope":
        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            text = (
                f"Out of scope: {record.next_step.scope_note}. "
                "This system does not provide guidance on that topic. "
                f"In-scope alternative: {record.next_step.display_text}"
            )
            return text, ()
        return OfflineTurn("out_of_scope", _fixed_sequence(()), finalize, terminal_hint="refused_out_of_scope")

    if intent in ("review_request", "reacquisition_request"):
        tool = "propose_review" if intent == "review_request" else "propose_reacquisition"
        steps = ({"tool": tool, "args": {"case_id": case_id, "record_version": rv, "reason": message.strip()}},)

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            if r is None or not r.ok:
                why = (r.error or {}).get("detail") if (r is not None and isinstance(r.error, dict)) else None
                return f"Could not record the {tool.replace('propose_', '')} proposal{': ' + why if why else ''}.", ()
            if intent == "review_request":
                return (f"Clinician review proposed for case {case_id}. The host records the request in the review "
                        "episode; this is a proposal, not a completed review and not a review-queue position. "
                        "The record is unchanged."), ()
            return (f"Image reacquisition proposed for case {case_id}. The host records it in the review episode as "
                    "awaiting information; the current model outputs stay displayed with their reliability flags. "
                    "The record is unchanged."), ()
        return OfflineTurn(intent, _fixed_sequence(steps), finalize)

    if intent == "ask_first":
        steps = ({"tool": "find_decisive_missing_facts", "args": {"case_id": case_id, "record_version": rv}},)

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            questions = [q for q in ((r.data or {}).get("questions", []) if (r is not None and r.ok) else [])
                         if q.get("decision_relevant")]
            if not questions:
                return "No decision-relevant fact is missing for this case; nothing further is needed from you.", ()
            q = questions[0]
            return (f"One decision-relevant fact is missing: {q.get('fact')}. {q.get('question_text')} "
                    f"Permitted answers: {', '.join(str(v) for v in q.get('permitted_values', []))}, or unknown. "
                    f"It affects {', '.join(q.get('affected_rules', []))}."), ()
        return OfflineTurn("ask_first", _fixed_sequence(steps), finalize)

    if intent == "supply_assertion":
        parsed = _parse_fact_statement(message)
        if parsed is None:
            return plan_for("unknown", message, ctx)
        fact, value, is_hypothetical, is_unknown = parsed
        current_value = record.clinician_assertions.get(fact, {}).get("value")

        if is_unknown:
            steps = ({"tool": "propose_review", "args": {
                "case_id": case_id, "record_version": rv, "reason": f"information unavailable: {fact}",
            }},)

            def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
                return (
                    f"Noted: {fact} is unknown. This will not be re-asked in this episode; "
                    "a clinician review has been proposed to help resolve it."
                ), ()
            return OfflineTurn("supply_assertion", _fixed_sequence(steps), finalize, terminal_hint="awaiting_information")

        if current_value is not None and current_value != value and not is_hypothetical:
            def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
                return (
                    f"{fact} was previously recorded as {current_value!r}; you are now stating {value!r}. "
                    "Please confirm this change explicitly before it is proposed."
                ), ()
            return OfflineTurn("supply_assertion", _fixed_sequence(()), finalize)

        steps = ({"tool": "propose_fact_amendment", "args": {
            "case_id": case_id, "record_version": rv, "fact": fact, "value": value, "statement": message,
        }},)

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            if r is None or not r.ok:
                detail = (r.error or {}).get("detail") if r is not None else "no tool result"
                return f"Could not propose {fact}={value!r}: {detail}", ()
            proposal_id = (r.data or {}).get("proposal_id")
            if is_hypothetical:
                return (
                    f"Hypothetical only: if {fact}={value!r}, proposal_id={proposal_id}. "
                    "Nothing is committed; this preview proposal is left for the host to cancel."
                ), ()
            return (
                f"Proposed {fact} = {value!r} (proposal_id={proposal_id}). "
                "Awaiting clinician confirmation before this is committed."
            ), ()
        return OfflineTurn("supply_assertion", _fixed_sequence(steps), finalize)

    if intent == "correct_criterion":
        parsed = _parse_criterion_statement(message)
        if parsed is None:
            return plan_for("unknown", message, ctx)
        criterion, value = parsed
        steps = ({"tool": "propose_criterion_amendment", "args": {
            "case_id": case_id, "record_version": rv, "criterion": criterion, "value": value, "statement": message,
        }},)

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            if r is None or not r.ok:
                detail = (r.error or {}).get("detail") if r is not None else "no tool result"
                return f"Could not propose {criterion}={value!r}: {detail}", ()
            proposal_id = (r.data or {}).get("proposal_id")
            return (
                f"Proposed checklist criterion {criterion} = {value!r} (proposal_id={proposal_id}). "
                "This is checklist evidence only; it does not by itself establish clinical suspicion."
            ), ()
        return OfflineTurn("correct_criterion", _fixed_sequence(steps), finalize)

    if intent == "clinical_photo":
        clinical_available = record.inputs.clinical_availability == "present"

        def next_step(collected: List[ToolResult]) -> Optional[Dict[str, Any]]:
            if not clinical_available:
                return None
            if not collected:
                return {"tool": "read_measured_xai", "args": {"case_id": case_id, "record_version": rv, "kind": "clinical"}}
            if len(collected) == 1 and not collected[0].ok:
                return {"tool": "request_measured_xai", "args": {"case_id": case_id, "record_version": rv, "kind": "clinical"}}
            return None

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            if not clinical_available:
                return f"Not applicable: clinical_availability={record.inputs.clinical_availability!r}; no clinical photo is present for this case.", ()
            read = collected[0] if collected else None
            if read is not None and read.ok:
                return "Measured clinical-photo contribution is available for this case (see the stored receipt).", ()
            started = collected[-1] if len(collected) > 1 else None
            if started is not None and started.ok:
                return f"No cached measurement yet; a clinical-photo contribution job has been started (job_id={(started.data or {}).get('job_id')}).", ()
            return "Measured clinical-photo contribution is not available and could not be started.", ()
        return OfflineTurn("clinical_photo", next_step, finalize)

    if intent == "xai_region":
        def next_step(collected: List[ToolResult]) -> Optional[Dict[str, Any]]:
            if not collected:
                return {"tool": "read_measured_xai", "args": {"case_id": case_id, "record_version": rv, "kind": "region"}}
            if len(collected) == 1 and not collected[0].ok:
                return {"tool": "request_measured_xai", "args": {"case_id": case_id, "record_version": rv, "kind": "region"}}
            return None

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            read = collected[0] if collected else None
            if read is not None and read.ok:
                return f"Measured region sensitivity is available (receipt status={(read.data or {}).get('status')}).", ()
            started = collected[-1] if len(collected) > 1 else None
            if started is not None and started.ok:
                return f"No cached measurement yet; a region-sensitivity job has been started (job_id={(started.data or {}).get('job_id')}).", ()
            return "Measured region sensitivity is not available and could not be started.", ()
        return OfflineTurn("xai_region", next_step, finalize)

    if intent == "similar_cases":
        steps = ({"tool": "retrieve_case_comparisons", "args": {"case_id": case_id, "record_version": rv}},)

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            if r is None or not r.ok:
                detail = (r.error or {}).get("detail") if r is not None else "no tool result"
                return f"Case comparisons are not available for this case: {detail}.", ()
            data = r.data or {}
            lookup = _proposition_lookup(r.propositions)
            claims: List[str] = []
            receipt = data.get("receipt") or {}
            identity_exclusion = receipt.get("identity_exclusion")
            identity_line = (
                f"Identity policy: {(receipt.get('policy') or {}).get('identity_policy')} -- "
                + (identity_exclusion if identity_exclusion else "same-patient exclusion verified")
            )
            status = data.get("status")
            if status != "ok" or not data.get("matches"):
                reason = data.get("reason") or f"comparison status {status!r}"
                return (
                    f"No eligible historical comparison case was found (status={status}): {reason}. "
                    f"{identity_line}."
                ), ()
            lines = [
                "Historical comparison (non-diagnostic reminder; a retrieved case does not by itself "
                "establish this case's diagnosis):",
            ]
            count_payload = data.get("descriptive_reference_label_counts") or {}
            counts = count_payload.get("counts") or {}
            if counts:
                count_claim_paths = ["memory.descriptive_reference_label_counts.n_retrieved"]
                count_claim_paths.extend(
                    f"memory.descriptive_reference_label_counts.counts.{label}" for label in counts
                )
                claims.extend(lookup[path] for path in count_claim_paths if path in lookup)
                ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
                rendered = [
                    f"{count} {'has' if int(count) == 1 else 'have'} dataset label {label}"
                    for label, count in ordered
                ]
                if len(rendered) == 1:
                    count_text = rendered[0]
                else:
                    count_text = ", ".join(rendered[:-1]) + " and " + rendered[-1]
                lines.append(
                    f"Among the {count_payload.get('n_retrieved')} retrieved reference cases, {count_text}. "
                    "This is a descriptive historical count and is not used to alter the current prediction."
                )
            for i, m in enumerate(data["matches"]):
                lines.append(_comparison_match_line(f"memory.matches[{i}]", m, lookup, claims))
            counterexample = data.get("counterexample")
            if counterexample is not None:
                lines.append("Similar-but-discordant counterexample:")
                lines.append(_comparison_match_line("memory.counterexample", counterexample, lookup, claims))
            else:
                lines.append(f"No counterexample available (counterexample_status={data.get('counterexample_status')}).")
            lines.append(identity_line)
            return "\n".join(lines), tuple(dict.fromkeys(claims))
        return OfflineTurn("similar_cases", _fixed_sequence(steps), finalize)

    if intent == "what_remains":
        steps = ({"tool": "read_review_history", "args": {"case_id": case_id, "record_version": rv}},)

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            if r is None or not r.ok:
                return "Review history is not available for this case.", ()
            wr = (r.data or {}).get("what_remains", {})
            return f"What remains: {wr.get('sentence')}", ()
        return OfflineTurn("what_remains", _fixed_sequence(steps), finalize)

    if intent == "source_quote":
        rule_id = _parse_rule_id(message) or record.next_step.rule_id
        if rule_id is None:
            rows = (record.applicability or {}).get("results", [])
            rule_id = rows[0]["rule_id"] if rows else None
        if rule_id is None:
            return OfflineTurn("source_quote", _fixed_sequence(()),
                               lambda collected: ("No catalogue rule is available to cite for this case.", ()))
        steps = ({"tool": "retrieve_guideline_evidence", "args": {"case_id": case_id, "record_version": rv, "rule_id": rule_id}},)

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            if r is None or not r.ok:
                return f"Could not retrieve guideline evidence for {rule_id}.", ()
            src = (r.data or {}).get("source", {})
            prop_id = r.propositions[0]["id"] if r.propositions else None
            return (
                f"{rule_id} ({(r.data or {}).get('status')}, jurisdiction={(r.data or {}).get('jurisdiction')}, "
                f"guideline_version={(r.data or {}).get('guideline_version')}): \"{src.get('span_text')}\" (page {src.get('pdf_page')})"
            ), ((prop_id,) if prop_id else ())
        return OfflineTurn("source_quote", _fixed_sequence(steps), finalize)

    if intent == "why_deferred":
        steps = (
            {"tool": "read_reliability", "args": {"case_id": case_id, "record_version": rv}},
            {"tool": "read_record_section", "args": {"case_id": case_id, "record_version": rv, "section": "image_status"}},
        )

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            r = collected[0] if collected else None
            rel = (r.data or {}) if r is not None and r.ok else {}
            img_r = collected[1] if len(collected) > 1 else None
            img = (img_r.data or {}) if img_r is not None and img_r.ok else {}
            scores = rel.get("scores") or {}
            thresholds = rel.get("thresholds") or {}
            warnings = rel.get("warnings") or []
            lines = [
                f"Reliability disposition: {rel.get('disposition')}.",
                f"IQA state: {rel.get('iqa_state')} ({(rel.get('meaning') or {}).get('iqa')}).",
                f"OOD state: {rel.get('ood_state')} ({(rel.get('meaning') or {}).get('ood')}).",
                f"View stability: {rel.get('view_stability_state')}.",
                f"Selective abstention: {rel.get('selective_abstention')}.",
                f"Image status: {img.get('state')} (reasons={list(img.get('reasons') or ())}, "
                f"recovery_action={img.get('recovery_action')}).",
            ]
            if warnings:
                lines.append("Active reliability flag(s):")
                for w in warnings:
                    code = w.get("code")
                    lines.append(
                        f"  - [{code}] {w.get('reason')} -- {_warning_measurement_line(code, scores, thresholds)}. "
                        f"Affected outputs: {_WARNING_AFFECTED_OUTPUTS.get(code, 'not documented for this code')}. "
                        f"Recovery action: {w.get('recovery_action')}. "
                        f"Remaining usable output if unresolved: {w.get('remaining_usable_output')}."
                    )
            else:
                lines.append("No reliability flag is active; nothing to recover.")
            lines.append(
                "A deferral is a per-member uncertainty flag requesting clinician review; it is not a "
                "position in a review queue and not a completed review. This reliability display is "
                "separate from the episode's own review-queue position; ask what remains / resume to see "
                "where this case stands in clinician review."
            )
            return "\n".join(lines), ()
        return OfflineTurn("why_deferred", _fixed_sequence(steps), finalize)

    if intent == "explain":
        steps = (
            {"tool": "read_record_section", "args": {"case_id": case_id, "record_version": rv, "section": "overview"}},
        )

        def finalize(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
            overview = (collected[0].data or {}) if collected and collected[0].ok else {}
            heads = overview.get("heads") or {}
            checklist = overview.get("checklist") or {}
            lines = [
                "Model prediction: the direct diagnosis head's top class is "
                f"{(heads.get('diagnosis_head') or {}).get('provisional_class')} "
                f"(melanoma_head.probability={(heads.get('melanoma_head') or {}).get('probability')}).",
                "Checklist arithmetic (not a clinical next step): "
                f"score range [{checklist.get('score_min')},{checklist.get('score_max')}], "
                f"decidable={checklist.get('decidable')}, meets_threshold={checklist.get('meets_threshold')}.",
            ]
            lines.append(
                "Measured image sensitivity is available only through the separate measured-XAI request; "
                "none is claimed in this evidence summary."
            )
            lines.append("Guideline status: see this case's own next-step and applicability sections for the guideline-derived recommendation.")
            return "\n".join(lines), ()
        return OfflineTurn("explain", _fixed_sequence(steps), finalize)

    def finalize_unknown(collected: List[ToolResult]) -> Tuple[str, Tuple[str, ...]]:
        return (
            "I did not recognise a supported request. I can: explain the prediction, explain why a case is "
            "deferred/flagged, record a clinical assertion or checklist correction, report measured image or "
            "clinical-photo sensitivity, retrieve similar cases, report what remains, or quote the exact "
            "guideline source for a rule."
        ), ()
    return OfflineTurn("unknown", _fixed_sequence(()), finalize_unknown)


class OfflineDeterministicProvider:
    """Drives :func:`~dermpact.core.agent.controller.run_turn`'s tool loop exactly like a live provider, but every reply is computed by :func:`plan_for` from the clinician's message and the tool results actually observed so far -- no model call, ever."""

    def __init__(self, message: str, ctx: ToolContext):
        self.provider_name = "offline_deterministic"
        self.intent = classify_intent(message)
        self._turn = plan_for(self.intent, message, ctx)
        self._collected: List[ToolResult] = []

    @property
    def terminal_hint(self) -> Optional[str]:
        return self._turn.terminal_hint

    def complete(self, system: str, messages: Sequence[Mapping[str, Any]], tools_schema: Sequence[Mapping[str, Any]]) -> ProviderReply:
        step = self._turn.next_step(self._collected)
        if step is not None:
            return ProviderReply(tool_calls=(step,), provider_name=self.provider_name)
        text, claims = self._turn.finalize(self._collected)
        return ProviderReply(text=text, claims=claims, provider_name=self.provider_name)

    def note_result(self, result: ToolResult) -> None:
        self._collected.append(result)


__all__ = [
    "INTENT_NAMES", "OfflineTurn", "OfflineDeterministicProvider", "classify_intent", "plan_for",
]
