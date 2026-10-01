"""Interpret explicit clinician corrections; reuse the existing host-only commit gate."""
from dataclasses import asdict
from datetime import date
from typing import Literal

from pydantic import Field
from dermpact.core.record_v2 import CLINICIAN_ASSERTION_FACTS
from dermpact.core.ledger_v2 import AmendmentProposal
from dermpact.criteria.schema import CRITERIA_BY_NAME
from .clarification import ClarificationAgent, InterpretationCheck, VERIFIER, now
from .evidence import EvidenceError, bind_receipt, wire_receipt
from .verification import StrictModel, Draft, SemanticVerifier, SemanticVerdict, binding_issues, display_answer


class CorrectionIntent(StrictModel):
    status: Literal["clear", "uncertain", "out_of_scope"]
    kind: Literal["clinician_fact", "criterion_value"] | None
    field: str | None
    value: bool | int | str | None
    quote: str = Field(max_length=4000)


def prediction_identity(record):
    p = record.perception.to_dict()
    return {k:p[k] for k in ("members", "diagnosis_head", "melanoma_head", "criterion_probabilities", "model_criterion_values")}


class AmendmentAgent(ClarificationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflow_kind = "amendment"

    def begin(self, request, *, rule_id=""):
        if self.phase not in {"ready", "answered", "cancelled", "awaiting_information"} or self.pending is not None:
            raise EvidenceError("an interaction is already pending")
        if not request.strip() or len(request) > 4000:
            raise ValueError("request must contain 1-4000 characters")
        case = self.host.store.load_case(self.case_id)
        gate = self.host.gate.verify(case.current_record, expected_case_id=self.case_id, now=date.today(), lineage=case.records)
        if not gate.passed:
            raise EvidenceError("source_record_invalid_before_amendment")
        self.request, self.rule_id, self.bound = request, "", self._identity(case.current_record)
        vocabulary = {name: sorted(spec.vocabulary) for name,spec in CRITERIA_BY_NAME.items()}
        payload = {"active_case": self.bound, "clinician_request": request,
                   "clinician_fact_fields": list(CLINICIAN_ASSERTION_FACTS), "criterion_values": vocabulary}
        intent = self._model(CorrectionIntent,
            "Extract one explicitly requested correction for this active case. Preserve negation and uncertainty. "
            "Use only supplied field/value vocabulary; clinician facts are boolean except total_nevus_count (integer). "
            "Do not infer clinical truth from predictions. Use uncertain for ambiguous/multiple unspecified changes. "
            "Copy an exact supporting quote; return out_of_scope for prohibited/unrecognized changes.", payload)
        if intent.status != "clear":
            self.phase = "awaiting_information" if intent.status == "uncertain" else "cancelled"
            return self._result(text="Specify one explicit, supported record correction." if intent.status == "uncertain" else "This requested change is outside the supported record fields.")
        if not intent.quote or intent.quote not in request:
            raise EvidenceError("correction_quote_not_in_request")
        if intent.kind == "clinician_fact":
            valid = intent.field in CLINICIAN_ASSERTION_FACTS and (
                type(intent.value) is int and intent.value >= 0 if intent.field == "patient.total_nevus_count" else type(intent.value) is bool)
        else:
            valid = intent.kind == "criterion_value" and intent.field in vocabulary and intent.value in vocabulary[intent.field]
        if not valid:
            raise EvidenceError("invalid_correction_field_or_value")
        check = self._model(InterpretationCheck,
            "Check that the full clinician request explicitly supports this ONE field/value change in the active case. "
            "Reject uncertainty, missed negation, another case, or instructions to invent a value. This checks interpretation, not medical truth.",
            {**payload, "interpretation": intent.model_dump()}, verifier=True)
        if not check.supported:
            self.phase = "awaiting_information"
            return self._result(text=check.reason)
        record = self._current()
        ledger, n0 = self.host.ledger_for_case(self.case_id)
        proposal = ledger.propose(record, AmendmentProposal(kind=intent.kind, field=intent.field, value=intent.value),
            actor=self.actor, session_id=self.session_id, proposed_at=now(), statement=request)
        self.host.flush_ledger_events(self.case_id, ledger, n0)
        self.trace.append({"stage": "proposal", "receipt": asdict(proposal)})
        if proposal.status != "proposed":
            self.phase = "awaiting_information"
            return self._result(reason=proposal.reason)
        self.pending, self.phase = proposal, "awaiting_confirmation"
        self.question = None
        return self._result(proposed_value=proposal.after, proposal_id=proposal.proposal_id,
            preview_sha256=proposal.consequence_preview_sha256, consequence_preview=proposal.consequence_preview,
            requires_acknowledgement=bool(proposal.next_step_changed), text="Review and explicitly confirm this correction before it is saved.")

    def respond(self, text):
        return self.begin(text)

    def _advance(self):
        record = self._current()
        case = self.host.store.load_case(self.case_id)
        if not record.amendments or len(case.records) < 2:
            raise EvidenceError("no_saved_amendment_to_explain")
        entry, prior = record.amendments[-1], case.records[-2]
        checklist = lambda item: {k:item.checklist.to_dict()[k] for k in ("decidable","score_min","score_max","meets_threshold")}
        next_step = lambda item: {k:item.next_step.to_dict()[k] for k in ("status","action_code","display_text")}
        data = {"saved_change": {"kind": entry.kind, "field": entry.field.replace("_", " ").replace(".", " "),
                                  "before": entry.before, "after": entry.after},
                "checklist_before": checklist(prior), "checklist_after": checklist(record),
                "checklist_unchanged": checklist(prior) == checklist(record),
                "next_step_before": next_step(prior), "next_step_after": next_step(record),
                "raw_model_predictions_unchanged": prediction_identity(record) == prediction_identity(prior),
                "scope": "The clinician-confirmed record correction is not a new model prediction or pathological confirmation. A recorded recommendation does not establish that a clinical procedure was performed."}
        receipt = bind_receipt({**self.bound, "tool": "read_saved_amendment", "ok": True, "data": data,
                               "error": None, "detail": "full"}, self.bound, 1)
        question = "Explain the saved correction, any recorded checklist/next-step change, and the unchanged model-prediction scope."
        payload = {"request": question, "evidence": wire_receipt(receipt)}
        verifier = SemanticVerifier(self.client_factory(VERIFIER, allow_live=True, response_schema=SemanticVerdict.model_json_schema()))
        for _ in range(2):
            draft = self._model(Draft, "Explain the saved change concisely using only the supplied before/after record facts. "
                "Do not claim a diagnosis, a new prediction, or an action was clinically performed. Use plain field names. "
                "Put fact IDs only in references, not text. Say unchanged without relisting unchanged details. "
                "Use separate short statements for the correction, checklist, next step, and prediction limits. Use the provided display names for actions. "
                "Preserve the recorded status: recommended does not mean unresolved or performed. Cite every factual clause.", payload)
            issues = binding_issues(draft, [receipt], self.bound)
            verdict = {"ok": False, "issues": issues} if issues else verifier.verify(question, draft, [receipt])
            self._current()
            self.trace.append({"stage": "explanation_check", "draft": draft.model_dump(), "verdict": verdict})
            if verdict["ok"]:
                self.phase = "answered"
                return self._result(text=display_answer(draft), evidence=[receipt], commit_status="committed")
            payload["repair_feedback"] = verdict["issues"]
        self.phase = "safe_fallback"
        return self._result(text="The correction was saved, but its explanation could not be verified.", commit_status="committed")
