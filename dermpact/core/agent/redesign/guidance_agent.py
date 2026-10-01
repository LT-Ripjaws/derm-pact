"""Request-led guideline retrieval and clarification, without a supplied rule ID."""
from dataclasses import asdict
from datetime import date
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field
from dermpact.pipeline.hybrid_retriever import BM25Retriever
from dermpact.core.record_v2 import CLINICIAN_ASSERTION_FACTS
from dermpact.core.ledger_v2 import AmendmentProposal, ProposalReceipt
from .clarification import ClarificationAgent, InterpretationCheck, GENERATOR, VERIFIER, now
from .evidence import EvidenceError, bind_receipt, digest, wire_receipt, fact_catalog
from .provider import JsonGatewayClient
from .verification import StrictModel, Statement, Draft, SemanticVerifier, SemanticVerdict, binding_issues, display_answer

VERSION = "request-led-guidance-v4"
SYSTEM = """You are a clinician-facing dermoscopic diagnostic decision-support agent.
Work out the clinician's actual goal. Choose your own guideline searches, relevant source
passages, clinical questions and final explanation. No target guideline has been selected
for you. The available case fields are a storage vocabulary, not a list of questions to ask.
Search the guideline source text before giving source-based guidance. Search again if the
returned passages do not answer the request. Do not substitute unrelated guidance.
When missing clinical facts materially affect the requested assessment, ask one relevant
question at a time. Do not infer clinical suspicion, equivocal status or patient risk from
model predictions. A clinician's uncertain/unavailable answer remains unknown. If sufficient
information cannot be obtained, explain the conditional guidance and unresolved limitation.
After an explicit clinician reply, propose saving the supported field/value using an exact
quote from that reply. The host obtains separate confirmation; you cannot confirm or commit.
For a general question that needs no patient-specific decision, answer without unnecessary
questions. Explain source-supported diagnostic assessment options and limits; do not prescribe
treatment, select excision margins, claim a diagnosis is confirmed, or claim a procedure occurred.
Only the latest evidence below is citable. It excludes precomputed guideline applicability,
next-step recommendations, rule predicates and candidate-question lists. Clinical facts carry
their own provenance; model predictions are not clinical assertions or pathology results.
Return one action: search(query), ask(question,fact), propose(fact,value,quote), or answer(statements).
fact is an available clinical field, or null for a question that cannot be stored in that schema.
Use null/empty values for irrelevant action fields. An answer contains short statements with
supplied fact_id references for every factual clause. Cite the retrieved source text AND the
clinical facts supporting patient-specific conclusions. State source limitations concisely.
Use plain language; keep identifiers in references, not prose. Do not narrate your reasoning.
Answer only the requested clinical question: the first statement directly answers what was
asked, naming the requested item and the recorded clinical fact that makes the guidance apply
to this patient. Attribute each clinical fact to who recorded it, as its provenance states
(for example, "the clinician recorded that melanoma is suspected, so the guideline supports
histopathological confirmation"). Do not assert a property that appears only in the request
(such as an adjective describing a procedure) unless the cited evidence states it. Match the
request's cardinality: "which approach" or "what should" asks for the one item whose stated
conditions all match the recorded facts; do not add other items that also apply. Refer to the
model prediction generically unless the request asks what it predicted. Every statement must
resolve a requested part or state a limitation the request asked about. Do not add other
recommendations, techniques, procedures, evidence grades, consensus rates, extra retrieved
passages, or remarks about what a passage does not say, even when retrieved and true; a
retrieved passage is not a reason to mention it. Two or three statements are usually enough.
A recorded false value means the condition is absent, not unknown; only null or missing
means unknown.
When repairing a draft, retain supported content and remove unsupported clauses. Do not
turn a removed claim into its negation or add new claims to explain the correction.
"""
VERIFICATION_CONTEXT = """
This workflow may explain source-supported diagnostic assessment options after clarification.
It must not prescribe treatment or claim procedures occurred. Guideline conditions and patient
facts support a conditional assessment; they do not establish a confirmed diagnosis.
TRUSTED SCHEMA DEFINITION: applicability/prerequisites describe eligibility only, not performance
or completion. 'These facts do not establish completion' denies an inference, not the event.
That qualification needs no observation proving non-completion. Neither completion nor
non-completion may be asserted merely because the procedure's status is absent.
For coverage, use the smallest sufficient set of supported statements. Still reject any
unsupported extra factual claim elsewhere in the answer.
The trusted workflow_execution object reports actions actually completed by the host.
Evaluate instructions to ask questions or save only after confirmation against those events,
not against whether the final prose repeats the actions. Include only informational answer
requirements in question_parts. Report an operational violation if the supplied events show
one; never assume an action happened from the draft alone.
"""


class Action(StrictModel):
    kind: Literal["search", "ask", "propose", "answer"]
    query: str | None = None
    question: str | None = None
    fact: str | None = None
    value: bool | int | str | None = None
    quote: str | None = None
    statements: list[Statement] = Field(default_factory=list)


class GuidanceVerifierClient:
    def __init__(self, factory, workflow_context):
        self.client = factory(VERIFIER, allow_live=True, response_schema=SemanticVerdict.model_json_schema())
        self.workflow_context = workflow_context

    def complete(self, system, messages):
        payload = json.loads(messages[0]["content"])
        payload["workflow_execution"] = self.workflow_context
        reply = self.client.complete(system + VERIFICATION_CONTEXT,
            [{**messages[0], "content": json.dumps(payload)}, *messages[1:]])
        if not isinstance(reply.value, dict):
            return reply
        statements = {s["statement_index"]: s["text"] for s in payload["statements"]}
        for part in reply.value.get("question_parts", []):
            if not isinstance(part, dict) or not isinstance(part.get("answer_quote"), str) or not isinstance(part.get("statement_indices"), list) or any(type(i) is not int for i in part["statement_indices"]):
                continue
            text = " ".join(" ".join(statements.get(i, "").split()) for i in part.get("statement_indices", []))
            quote = " ".join(part.get("answer_quote", "").split())
            if not quote or quote in text:
                continue
            start, end = None, 0
            for excerpt in re.split(r"(?<=[.!?])\s+", quote):
                position = text.find(excerpt, end)
                if position < 0:
                    break
                start = position if start is None else start
                end = position + len(excerpt)
            else:
                part["answer_quote"] = text[start:end]
                reply.metadata.setdefault("coverage_quote_rebindings", []).append(
                    {"original": quote, "bound_span": part["answer_quote"]})
        return reply


class GuidanceAgent(ClarificationAgent):
    """LLM selects guidance and dialogue; the inherited ledger enforces confirmed writes."""
    def __init__(self, host, *, case_id, session_id, actor, state_path, client_factory=JsonGatewayClient):
        super().__init__(host, case_id=case_id, session_id=session_id, actor=actor, client_factory=client_factory)
        self.state_path = Path(state_path)
        self.documents = [{"id": r.rule_id, "title": r.title, "text": r.title + "\n" + r.source.span_text,
            "source": {k: r.source.as_dict()[k] for k in ("document_id", "doi", "pdf_page", "span_sha256")},
            "guideline_version": r.guideline_version} for r in host.catalogue.records]
        self.retriever = BM25Retriever(self.documents)
        self.dialogue, self.sources, self.searches = [], {}, []
        self.last_reply, self.action_count, self.repairs = "", 0, 0
        self.workflow_kind = "request_led_guidance"

    def _save(self):
        state = {"version": VERSION, "models": {"generator": asdict(GENERATOR), "verifier": asdict(VERIFIER)},
            "catalogue_sha256": self.host.catalogue.catalogue_sha256, "case_id": self.case_id,
            "session_id": self.session_id, "actor": self.actor, "request": self.request,
            "phase": self.phase, "bound": self.bound, "question": self.question,
            "pending": asdict(self.pending) if self.pending else None, "confirmed": self.confirmed,
            "dialogue": self.dialogue, "sources": self.sources, "searches": self.searches,
            "last_reply": self.last_reply, "action_count": self.action_count, "repairs": self.repairs,
            "trace": self.trace}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".pending")
        temporary.write_text(json.dumps({**state, "sha256": digest(state)}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.state_path)

    @classmethod
    def restore(cls, host, state_path):
        state = json.loads(Path(state_path).read_text(encoding="utf-8")); signature = state.pop("sha256")
        if signature != digest(state) or state["version"] != VERSION:
            raise EvidenceError("guidance_checkpoint_integrity_mismatch")
        if state["models"] != {"generator": asdict(GENERATOR), "verifier": asdict(VERIFIER)} or state["catalogue_sha256"] != host.catalogue.catalogue_sha256:
            raise EvidenceError("guidance_checkpoint_configuration_mismatch")
        agent = cls(host, case_id=state["case_id"], session_id=state["session_id"], actor=state["actor"], state_path=state_path)
        for key in ("request", "phase", "bound", "question", "confirmed", "dialogue", "sources", "searches", "last_reply", "action_count", "repairs", "trace"):
            setattr(agent, key, state[key])
        agent.pending = ProposalReceipt(**state["pending"]) if state["pending"] else None
        agent._current()
        if agent.pending:
            events = host.store.load_case(agent.case_id).ledger_events
            if any(e.get("proposal_id") == agent.pending.proposal_id and e.get("type") in {"committed", "cancelled", "expired"} for e in events):
                raise EvidenceError("guidance_pending_proposal_closed")
            if not any(e.get("type") == "proposed" and e.get("proposal_id") == agent.pending.proposal_id
                       and e.get("record_sha256") == agent.bound["record_sha256"]
                       and e.get("session_id") == agent.session_id and e.get("actor") == agent.actor
                       and e.get("consequence_preview_sha256") == agent.pending.consequence_preview_sha256 for e in events):
                raise EvidenceError("guidance_pending_proposal_mismatch")
        return agent

    def _result(self, **values):
        self._save()
        return {"version": VERSION, "outcome": self.phase, "identity": self.bound, **values}

    def _receipts(self):
        record = self._current()
        perception = record.perception.to_dict()
        data = {"clinical_facts": record.clinician_assertions,
            "model_predictions": {k: perception[k] for k in ("diagnosis_head", "melanoma_head", "criterion_probabilities")},
            "scope": "Model outputs are not clinician assessments or confirmed diagnoses. No clinical procedure completion is recorded by this interface."}
        raw = [{"tool": "read_case_facts", "ok": True, "data": data, "error": None, "detail": "full"}]
        if self.sources:
            raw.append({"tool": "search_guidelines", "ok": True, "data": {"passages": list(self.sources.values())}, "error": None, "detail": "full"})
        if self.last_reply:
            raw.append({"tool": "read_clinician_reply", "ok": True, "data": {
                "question": self.question["question_text"] if self.question else None,
                "reply": self.last_reply, "scope": "Clinician-provided conversation information, not yet a confirmed stored change or pathology finding."},
                "error": None, "detail": "full"})
        return [bind_receipt({**self.bound, **r}, self.bound, i + 1) for i, r in enumerate(raw)]

    def begin(self, request):
        if self.phase not in {"ready", "answered", "cancelled"} or not request.strip():
            raise EvidenceError("guidance_episode_already_pending_or_empty_request")
        record = self._record(); case = self.host.store.load_case(self.case_id)
        if not self.host.gate.verify(record, expected_case_id=self.case_id, now=date.today(), lineage=case.records).passed:
            raise EvidenceError("guidance_source_record_invalid")
        self.request, self.bound = request, self._identity(record)
        self.sources, self.searches, self.confirmed = {}, [], []
        self.last_reply, self.action_count, self.repairs = "", 0, 0
        self.dialogue = [{"role": "clinician", "text": request}]
        return self._loop()

    def respond(self, text):
        if self.phase != "awaiting_information":
            raise EvidenceError("guidance_not_awaiting_information")
        self.last_reply = text
        self.dialogue.append({"role": "clinician", "text": text})
        return self._loop()

    def _advance(self):
        # Inherited confirm() has already validated/committed exactly one approved change.
        self.dialogue.append({"role": "host", "confirmed_change": self.confirmed[-1]})
        self.last_reply = ""
        return self._loop()

    def _verify_answer(self, draft, receipts):
        issues = binding_issues(draft, receipts, self.bound)
        if not self.sources:
            issues.append("No guideline source has been retrieved; search before answering this guidance request.")
        context = {"dialogue": self.dialogue, "confirmed_updates": self.confirmed,
            "commits": [t for t in self.trace if t.get("stage") == "commit"],
            "record_version": self.bound["record_version"]}
        return {"ok": False, "issues": issues} if issues else SemanticVerifier(
            GuidanceVerifierClient(self.client_factory, context)).verify(self.request, draft, receipts)

    def _answer_result(self, draft, receipts):
        self.phase = "answered"
        facts = fact_catalog(receipts); cited = set()
        for statement in draft.statements:
            for ref in statement.references:
                source_fact = facts[ref.fact_id]; pointer = source_fact["pointer"].split("/")
                receipt = next(r for r in receipts if r["id"] == source_fact["source_id"])
                if receipt["tool"] == "search_guidelines" and pointer[1:3] == ["data", "passages"]:
                    cited.add(receipt["data"]["passages"][int(pointer[3])]["id"])
        return self._result(text=display_answer(draft), evidence=receipts,
            retrieved_sources=list(self.sources), cited_sources=sorted(cited))

    def _loop(self):
        feedback = None
        while self.action_count < 24:
            self._save()
            receipts = self._receipts()
            payload = {"request": self.request, "conversation": self.dialogue,
                "evidence": [wire_receipt(r) for r in receipts], "previous_searches": self.searches,
                "storage_fields": list(CLINICIAN_ASSERTION_FACTS), "feedback": feedback}
            action = self._model(Action, SYSTEM, payload)
            self.action_count += 1
            self.trace.append({"stage": "guidance_action", "action": action.model_dump()})
            if action.kind == "search":
                if not action.query or len(self.searches) >= 6 or action.query in self.searches:
                    raise EvidenceError("guideline_search_empty_duplicate_or_budget")
                found, _ = self.retriever.search(action.query, k=5)
                self.searches.append(action.query)
                for row in found:
                    self.sources[row["id"]] = {k: row[k] for k in ("id", "title", "text", "source", "guideline_version")}
                self.trace.append({"stage": "guideline_search", "query": action.query, "retrieved_ids": [r["id"] for r in found]})
                continue
            if action.kind == "ask":
                if not action.question or (action.fact is not None and action.fact not in CLINICIAN_ASSERTION_FACTS):
                    raise EvidenceError("invalid_guidance_question")
                self.question = {"fact": action.fact, "question_text": action.question}
                self.dialogue.append({"role": "assistant", "question": action.question})
                self.phase = "awaiting_information"
                return self._result(question=action.question, fact=action.fact)
            if action.kind == "propose":
                valid = (action.fact in CLINICIAN_ASSERTION_FACTS and
                    (type(action.value) is int and action.value >= 0 if action.fact == "patient.total_nevus_count" else type(action.value) is bool))
                if not valid or not action.quote or action.quote not in self.last_reply:
                    raise EvidenceError("guidance_proposal_not_bound_to_explicit_reply")
                support = self._model(InterpretationCheck,
                    "Does the clinician's actual reply, in the context of the asked question, explicitly support this field/value? "
                    "Reject uncertainty, missed negation, or a value merely present in the assistant's question. This is an assertion-interpretation check, not a guideline decision.",
                    {"question": self.question, "reply": self.last_reply, "field": action.fact, "value": action.value}, verifier=True)
                if not support.supported:
                    feedback = support.reason
                    continue
                ledger, n0 = self.host.ledger_for_case(self.case_id)
                proposal = ledger.propose(self._current(), AmendmentProposal(kind="clinician_fact", field=action.fact, value=action.value),
                    actor=self.actor, session_id=self.session_id, proposed_at=now(), statement=self.last_reply)
                self.host.flush_ledger_events(self.case_id, ledger, n0)
                if proposal.status != "proposed":
                    raise EvidenceError("guidance_proposal_rejected:" + str(proposal.reason))
                self.pending, self.phase = proposal, "awaiting_confirmation"
                # Full consequence preview is shown to the clinician, but is not a model input.
                return self._result(proposal_id=proposal.proposal_id, preview_sha256=proposal.consequence_preview_sha256,
                    consequence_preview=proposal.consequence_preview, fact=action.fact, proposed_value=action.value,
                    text="Review and confirm this record change before it is saved.")
            # Source markers are rendered from structured references, not claim text.
            draft = Draft(kind="answer", statements=[s.model_copy(update={"text": re.sub(
                r"\s*\[e\d+(?:\.f\d+)?(?:\s*,\s*e\d+(?:\.f\d+)?)*\]", "", s.text).strip()})
                for s in action.statements])
            verdict = self._verify_answer(draft, receipts)
            self.trace.append({"stage": "guidance_verification", "draft": draft.model_dump(), "verdict": verdict})
            if verdict["ok"]:
                return self._answer_result(draft, receipts)
            if self.repairs >= 1:
                self.phase = "safe_fallback"
                return self._result(text="The guidance could not be verified; review the recorded evidence.")
            self.repairs += 1
            feedback = {"issues": verdict["issues"], "draft": draft.model_dump(),
                "instruction": "Minimally correct this draft; preserve supported answers, remove unsupported clauses, and add only evidence-backed content needed by the request."}
        self.phase = "safe_fallback"
        return self._result(text="The interaction budget was reached; review the saved clinical facts and sources.")
