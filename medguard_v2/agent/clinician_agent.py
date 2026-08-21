"""The clinician agent: an LLM that answers questions about a decision, and accepts corrections.

Where the LLM belongs, and where it does not
--------------------------------------------
G1 and G1c established that a vision-language model cannot read dermoscopic criteria: six models
including four frontier ones, best lift **-0.0059** against a per-criterion chance baseline, with
visual few-shot making it worse. So no LLM touches perception. That finding stands.

It says nothing about the rest of the system, and the rest of the system has a real hole. The
pipeline emits a contract and then goes silent. A clinician who wants to ask *"why did you call the
network atypical?"*, or who looks at the image and says *"criterion 3 is absent, you're wrong"*,
has nowhere to go. That is a genuine decision-support gap and it is exactly what a language model
is good at.

So the LLM sits **above** the decision, never inside it:

* it can read the contract and the hash-pinned guideline catalog;
* it decides **on its own** whether it needs more evidence, can answer, or must refuse;
* it can apply a clinician's override of any criterion and report the consequence -- but the
  consequence is computed by the **host's deterministic aggregator**, never by the model.

That last point is what makes the human-in-the-loop safe. The clinician changes a judgement; the
arithmetic that turns judgements into a verdict is not up for negotiation, and the model cannot
misreport it without the host catching the discrepancy.

Every claim it makes is checkable
---------------------------------
Because the aggregator is deterministic, the ground truth for any counterfactual the agent asserts
is computable exactly. If the agent says "removing that criterion drops the score to 2 and it no
longer meets the threshold", the host knows whether that is true. That yields a verification
signal most agent evaluations cannot get.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from medguard_v2.agent.annotation_store import AnnotationStore
from medguard_v2.agent.case_memory import CaseMemory
from medguard_v2.agent.verification import (
    citation_array_checks,
    deterministic_checks,
    override_answer_checks,
    uncited_specifics_check,
    verify_answer,
)
from medguard_v2.criteria.aggregate import MANAGEMENT_LADDER, recommend_management, score_criteria
from medguard_v2.criteria.schema import (
    CRITERIA_BY_NAME,
    CRITERION_NAMES,
    INDETERMINATE,
    MELANOMA_SUSPICION_THRESHOLD,
    validate_values,
)
from medguard_v2.evidence.contract import DecisionContract, contract_to_dict
from medguard_v2.evidence.guideline_binding import GROUP_TO_CONDITION, GuidelineCatalog
from medguard_v2.evidence.router import (
    QUESTION_TYPES_FOR_STATE,
    allowed_retrievals,
    classify_state,
)

MAX_TURNS = 6
#: v5.3: the loop allows a SECOND revision, but only when the draft that follows the first one
#: still fails EXCLUSIVELY host-mechanically-fixable checks -- see _second_revision_eligible.
#: Turns budget (MAX_TURNS) is unchanged; this caps how many of those turns may be spent on a
#: host-forced redraft rather than on tool calls.
MAX_REVISIONS = 2
#: v5.4: the agent's prompt/tool logic is unchanged from v5.3, but the promoted perception path
#: (run_cohort promoted=True) changes what get_decision serves -- different classifier diagnoses,
#: confidences, criteria readings, and the new criterion_margins field. The eval cache keys on
#: PROMPT_VERSION, so this bump is what forces the downstream agent eval to regenerate against the
#: promoted contracts instead of silently reusing v5.3 answers built on the pre-promotion inputs.
PROMPT_VERSION = "clinician-agent-v5.4"


# --------------------------------------------------------------------------- tools


@dataclass
class ToolResult:
    ok: bool
    payload: Dict[str, Any]
    note: str = ""


class AgentTools:
    """The agent's entire world. It cannot see the image, and it cannot invent evidence."""

    def __init__(
        self,
        contract: DecisionContract,
        catalog: GuidelineCatalog,
        case_memory: Optional[CaseMemory] = None,
        query_embedding: Optional[Any] = None,
        gated_retrieval: bool = True,
        dense_index: Optional[Any] = None,
        annotation_store: Optional[AnnotationStore] = None,
    ):
        self.contract = contract
        self.catalog = catalog
        self.case_memory = case_memory
        self.query_embedding = query_embedding
        #: Deployment-time annotation capture (ADDITIVE, default None = zero behaviour change).
        #: When present, every SUCCESSFUL override_criterion call is also recorded as one
        #: per-criterion annotation -- see medguard_v2/agent/annotation_store.py. No eval or agent
        #: code path passes it, so the default path stays byte-identical.
        self.annotation_store = annotation_store
        #: P1 dense-RAG control (ADDITIVE, default off). A DenseGuidelineIndex, or None. When
        #: present it powers the walled-off ``literature_lookup`` tool ONLY -- an unverified
        #: exploration channel whose dense-retrieved chunks can never populate the structured
        #: citations array (its chunk ids are not catalog record ids, so the existing in-loop
        #: citation gate rejects them) and are never recorded as a ``search_guidelines`` call, so
        #: they cannot enter the RAGAS guideline-context reconstruction or the fabrication check
        #: either. It never touches the decision, the contract, or the keyed retrieval path.
        self.dense_index = dense_index
        self.calls: List[Dict[str, Any]] = []
        self.overrides: Dict[str, str] = {}
        #: Chronological record of every successful override_criterion call, including repeats
        #: of the same criterion. ``self.overrides`` is kept too (flat, final-value-per-criterion)
        #: for callers that only need the end state (e.g. AgentAnswer.overrides_applied); this
        #: sequence exists because verification.override_answer_checks needs the TRUE last step,
        #: which a flat dict cannot recover once a criterion is overridden twice (dict iteration
        #: order reflects first insertion, not most recent update).
        self.override_sequence: List[Tuple[str, str]] = []

        # Router-governed retrieval (spec Part 2). Computed once: what kind of case this is, and
        # therefore what search_guidelines is allowed to actually serve. `gated_retrieval=False`
        # is the ablation arm -- the plan is still computed (the per-case prompt nudge and the
        # retrieval reporting both still want it) but search_guidelines stops enforcing it.
        self.gated_retrieval = gated_retrieval
        self.router_state, self.router_reason = classify_state(
            classifier_diagnosis=contract.classifier_diagnosis,
            classifier_confidence=contract.classifier_confidence,
            score=contract.score,
        )
        self.router_plan: Set[Tuple[str, str]] = allowed_retrievals(
            self.router_state, contract.classifier_diagnosis
        )
        self.retrieval_served = 0
        self.retrieval_skipped = 0
        self.retrieval_counterfactual_chars_skipped = 0

    # -- read the decision ---------------------------------------------------

    def get_decision(self) -> ToolResult:
        payload = contract_to_dict(self.contract)
        payload.pop("receipt", None)
        self.calls.append({"tool": "get_decision"})
        return ToolResult(True, payload)

    # -- search the pinned corpus -------------------------------------------

    def _in_router_plan(self, condition: str, question_type: str) -> bool:
        """Wildcards ok; an unspecified question_type on the call side also matches anything,
        mirroring catalog.bind's own default-to-management-priority behaviour for that case."""
        for group, qtype in self.router_plan:
            if group == condition and (qtype == "*" or question_type == "" or qtype == question_type):
                return True
        return False

    def search_guidelines(self, condition: str, question_type: str = "") -> ToolResult:
        self.calls.append(
            {"tool": "search_guidelines", "condition": condition, "question_type": question_type}
        )
        if condition not in GROUP_TO_CONDITION:
            return ToolResult(
                False, {}, f"unknown condition {condition!r}; valid: {sorted(GROUP_TO_CONDITION)}"
            )
        question_types = (question_type,) if question_type else ()
        # bind() is free and deterministic, so it is always computed -- even when the call turns
        # out to be gated -- which is what makes the forgone payload measurable without serving it.
        result = (
            self.catalog.bind(condition, question_types=question_types)
            if question_types
            else self.catalog.bind(condition)
        )

        if not self.gated_retrieval or self._in_router_plan(condition, question_type):
            self.retrieval_served += 1
            return ToolResult(result.covered, result.as_dict(), result.absence_reason or "")

        self.retrieval_skipped += 1
        self.retrieval_counterfactual_chars_skipped += len(json.dumps(result.as_dict(), default=str))
        gated_payload = {
            "router_gated": True,
            "router_state": self.router_state.value,
            "served": False,
            "condition_coverage": "covered" if result.covered else "not_covered",
            "reason": (
                f"This case is routed as {self.router_state.value} ({self.router_reason}); "
                f"{condition}/{question_type or 'any'} is outside the retrieval this case "
                f"indicates."
            ),
            # Absence disclosure is the honest answer for an uncovered condition and must not be
            # blocked by gating -- most visibly the committed condition itself in NO_COVERAGE.
            "absence_reason": result.absence_reason if not result.covered else None,
        }
        return ToolResult(True, gated_payload, "")

    def retrieval_summary(self) -> Dict[str, Any]:
        """Measured gating behaviour for this conversation, for AgentAnswer.retrieval."""
        return {
            "router_state": self.router_state.value,
            "served": self.retrieval_served,
            "skipped": self.retrieval_skipped,
            "counterfactual_chars_skipped": self.retrieval_counterfactual_chars_skipped,
        }

    # -- the human-in-the-loop step -----------------------------------------

    def override_criterion(self, criterion: str, value: str) -> ToolResult:
        """Apply a clinician's correction and recompute. The HOST does the arithmetic."""
        self.calls.append(
            {"tool": "override_criterion", "criterion": criterion, "value": value}
        )
        if criterion not in CRITERION_NAMES:
            return ToolResult(False, {}, f"unknown criterion {criterion!r}")
        spec = CRITERIA_BY_NAME[criterion]
        if value != INDETERMINATE and value not in spec.vocabulary:
            return ToolResult(
                False, {}, f"{criterion}: {value!r} is not a permitted value; "
                           f"allowed: {sorted(spec.vocabulary)}"
            )

        revised = dict(self.contract.criteria)
        revised.update(self.overrides)
        revised[criterion] = value
        validate_values(revised)

        effective_before = {**self.contract.criteria, **self.overrides}
        before = score_criteria(effective_before)
        after = score_criteria(revised)
        self.overrides[criterion] = value
        self.override_sequence.append((criterion, value))

        if self.annotation_store is not None:
            # A SUCCESSFUL override is one expert-verified (case, criterion, value) annotation;
            # capture it with the contract's provenance. Rejected overrides never reach here.
            self.annotation_store.record(
                case_id=self.contract.case_id,
                criterion=criterion,
                corrected_value=value,
                prior_value=effective_before.get(criterion),
                contract_receipt=self.contract.receipt or None,
            )

        action_before, _ = recommend_management(before)
        action_after, reason_after = recommend_management(after)

        return ToolResult(
            True,
            {
                "criterion": criterion,
                "was": effective_before.get(criterion),
                "now": value,
                "score_before": {"min": before.score_min, "max": before.score_max},
                "score_after": {"min": after.score_min, "max": after.score_max},
                "threshold": MELANOMA_SUSPICION_THRESHOLD,
                "verdict_before": action_before,
                "verdict_after": action_after,
                "verdict_reason": reason_after,
                "verdict_changed": action_before != action_after,
                "computed_by": "host deterministic aggregator, not the model",
            },
        )

    # -- structured reasoning chain -----------------------------------------

    def explain_reasoning(self) -> ToolResult:
        """Walk through every step of the decision chain with provenance."""
        self.calls.append({"tool": "explain_reasoning"})
        c = self.contract
        state, reason = classify_state(
            classifier_diagnosis=c.classifier_diagnosis,
            classifier_confidence=c.classifier_confidence,
            score=c.score,
        )
        indeterminate = [
            name for name in CRITERION_NAMES
            if c.criteria.get(name) == INDETERMINATE
        ]
        action, action_reason = recommend_management(c.score)
        steps = [
                {
                    "step": 1,
                    "name": "classifier_diagnosis",
                    "result": c.classifier_diagnosis,
                    "confidence": c.classifier_confidence,
                    "source": "DermLIP frozen probe (LogisticRegression, C=10, trained on Derm7pt train)",
                },
                {
                    "step": 2,
                    "name": "criterion_values",
                    "result": dict(c.criteria),
                    "indeterminate": indeterminate,
                    "source": "7 criterion heads with margin-based abstention",
                },
                {
                    "step": 3,
                    "name": "seven_point_score",
                    "score_min": c.score.score_min,
                    "score_max": c.score.score_max,
                    "decidable": c.score.decidable,
                    "meets_threshold": c.score.meets_threshold if c.score.decidable else None,
                    "threshold": MELANOMA_SUSPICION_THRESHOLD,
                    "source": "Argenziano 7-point checklist rule (deterministic)",
                },
                {
                    "step": 4,
                    "name": "router_state",
                    "state": state.value,
                    "reason": reason,
                    "source": "gated retrieval router (deterministic table)",
                },
                {
                    "step": 5,
                    "name": "conflict_analysis",
                    "present": c.conflict.present,
                    "escalate": c.conflict.escalate,
                    "description": c.conflict.description,
                },
                {
                    "step": 6,
                    "name": "guideline_binding",
                    "covered": c.guideline.covered,
                    "condition": c.guideline.catalog_condition,
                    "n_recommendations": len(c.guideline.recommendations),
                    "absence_reason": c.guideline.absence_reason,
                    "source": "hash-verified lookup against licensed CPG corpus",
                },
                {
                    "step": 7,
                    "name": "management_recommendation",
                    "action": action,
                    "reason": action_reason,
                    "source": "deterministic management ladder",
                },
                {
                    "step": 8,
                    "name": "contract_verification",
                    "safe_to_display": c.safe_to_display,
                    "failures": c.verification_failures(),
                },
        ]
        # Step 9: the ON-DEMAND, faithful-by-construction case-based explanation. explain_reasoning
        # is what the agent calls for a "why / what does this look like" question, so this is the
        # right place to surface the nearest confirmed TRAIN cases (identity in the model's own
        # feature space -- faithful by construction, not a post-hoc rationalisation) and the boundary
        # counterfactual, together with the saliency caveat. It is progressive disclosure (MG52): it
        # rides on an explicit reasoning request and is empty when no context was attached to this
        # case, rather than being pushed into the default contract view.
        steps.append({
            "step": 9,
            "name": "case_based_explanation",
            "on_demand": True,
            "rationale": (
                "MG52 progressive disclosure: surfaced on an explicit why/what-does-this-look-like "
                "request, never in the default 10-second view."
            ),
            "similar_cases_available": c.similar_cases is not None,
            "similar_cases": (c.similar_cases.as_dict() if c.similar_cases is not None else None),
            "faithful_by_construction": (
                "The neighbours are the query's nearest TRAIN cases by exact cosine in the frozen "
                "DermLIP feature space the reader itself reads -- identity in that space, not a "
                "post-hoc rationalisation; train-only and leave-one-out, so the case never retrieves "
                "itself."
            ),
            "saliency_caveat": (
                "A GradCAM saliency map over the frozen DermLIP backbone is faithful by "
                "deletion/insertion but spatially diffuse; not a substitute for clinician "
                "localization. It beats a random masking order at p<1e-5 (deletion and insertion, "
                "research/evidence/derm7pt_v1/attribution_faithfulness_v1.json) yet attends to "
                "artifacts, so it is offered only as a caveated pointer, never as primary evidence."
            ),
        })
        return ToolResult(True, {"steps": steps})

    # -- structural knowledge query -----------------------------------------

    def query_taxonomy(self, topic: str) -> ToolResult:
        """Query the system's structural knowledge."""
        self.calls.append({"tool": "query_taxonomy", "topic": topic})
        topic_lower = topic.lower().strip()

        if "criterion" in topic_lower or "criteria" in topic_lower:
            return ToolResult(True, {
                "criteria": {
                    name: {
                        "description": spec.description,
                        "vocabulary": sorted(spec.vocabulary),
                        "category": spec.category,
                        "points": spec.points,
                        "scoring_values": sorted(spec.scoring_values),
                    }
                    for name, spec in CRITERIA_BY_NAME.items()
                },
                "melanoma_suspicion_threshold": MELANOMA_SUSPICION_THRESHOLD,
            })

        if "condition" in topic_lower or "coverage" in topic_lower or "guideline" in topic_lower:
            return ToolResult(True, {
                "group_to_condition": dict(GROUP_TO_CONDITION),
                "covered_groups": [g for g, c in GROUP_TO_CONDITION.items() if c is not None],
                "uncovered_groups": [g for g, c in GROUP_TO_CONDITION.items() if c is None],
                "catalog_size": len(self.catalog),
                "catalog_conditions": list(self.catalog.conditions),
            })

        if "management" in topic_lower or "ladder" in topic_lower:
            return ToolResult(True, {
                "management_ladder": list(MANAGEMENT_LADDER),
                "threshold": MELANOMA_SUSPICION_THRESHOLD,
                "note": "Management action is determined by the 7-point score and conflict state",
            })

        if "router" in topic_lower or "retrieval" in topic_lower:
            return ToolResult(True, {
                "states": {
                    s.value: {
                        "question_types": list(qt),
                    }
                    for s, qt in QUESTION_TYPES_FOR_STATE.items()
                },
                "note": "The router classifies each case into a retrieval state based on "
                        "classifier diagnosis, confidence, and 7-point score decidability",
            })

        return ToolResult(False, {}, (
            f"Unknown topic {topic!r}. Try: 'criteria', 'conditions', "
            "'management', or 'router'."
        ))

    # -- case memory retrieval -------------------------------------------------

    def retrieve_similar_cases(self, k: int = 3) -> ToolResult:
        """Retrieve past system evaluations for visually similar lesions."""
        self.calls.append({"tool": "retrieve_similar_cases", "k": k})
        if self.case_memory is None or self.query_embedding is None:
            return ToolResult(False, {}, "Case memory not available for this session.")
        results = self.case_memory.retrieve(
            self.query_embedding,
            k=min(k, 5),
            exclude={self.contract.case_id},
        )
        return ToolResult(True, {
            "n_retrieved": len(results),
            "n_memory": len(self.case_memory),
            "cases": results,
            "note": "These are past SYSTEM assessments based on visual similarity, "
                    "not verified clinical outcomes.",
        })

    # -- walled-off unverified exploration (P1 dense-RAG control) -------------

    def literature_lookup(self, query: str, k: int = 3) -> ToolResult:
        """Dense top-k retrieval over chunked guideline TEXT -- UNVERIFIED exploration only.

        This is the deliberately walled-off counterpart to ``search_guidelines``. It returns
        dense-retrieved passages (spans of guideline source text, chosen by cosine similarity),
        each stamped ``unverified`` and ``citable: false``, behind a disclaimer that says these
        chunk ids are NOT guideline record ids and must never enter the citations array.

        The wall is structural, not merely advisory:

        * the chunk ids are not catalog record ids (and are not even citation-shaped), so a chunk
          id copied into the citations array is rejected in-loop by ``citation_array_checks`` and
          counted unresolvable by ``eval.agent_eval.score_citations`` -- neither of which this
          module touches;
        * this call is recorded under its OWN tool name, never as a ``search_guidelines`` call, so
          the RAGAS context reconstruction and the fabrication check (both of which key on
          ``search_guidelines``) never see dense chunks as guideline evidence.

        Nothing here can change a criterion, a score, a verdict, or a citation.
        """
        self.calls.append({"tool": "literature_lookup", "query": query})
        if self.dense_index is None:
            return ToolResult(False, {}, "literature_lookup is not available in this session.")
        from medguard_v2.evidence.dense_retrieval import format_unverified_chunks

        retrieved = self.dense_index.retrieve(query, k=max(1, min(int(k), 5)))
        payload = format_unverified_chunks(retrieved)
        return ToolResult(True, payload, payload["disclaimer"])


# --------------------------------------------------------------------------- agent


@dataclass
class AgentAnswer:
    kind: str                      # "answer" | "refusal"
    text: str
    citations: Tuple[str, ...] = ()
    tool_calls: Tuple[Dict[str, Any], ...] = ()
    turns: int = 0
    overrides_applied: Dict[str, str] = field(default_factory=dict)
    raw_error: Optional[str] = None
    #: {"issues_first_draft": [...], "revised": bool, "revisions": int, "issues_final": [...]}.
    #: Empty for anything that never reached the verification loop (refusals, parse failures).
    #: New in v5; defaults to {} so cached-dict reconstructions of AgentAnswer from pre-v5 rows
    #: keep working. "revised"/"issues_final" date from v5; "revisions" (0/1/2) is new in v5.3,
    #: when the loop grew a conditional second revision -- "revised" is kept as ``revisions > 0``
    #: so anything reading only the old key still gets the right bool.
    verification: Dict[str, Any] = field(default_factory=dict)
    #: {"router_state": ..., "served": n, "skipped": n, "counterfactual_chars_skipped": n}. New
    #: in v5; same backward-compatible empty default.
    retrieval: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "citations": list(self.citations),
            "tool_calls": [dict(c) for c in self.tool_calls],
            "turns": self.turns,
            "overrides_applied": dict(self.overrides_applied),
            "raw_error": self.raw_error,
            "verification": dict(self.verification),
            "retrieval": dict(self.retrieval),
        }


_TOOL_LIST_BASE = """  get_decision()                                  -- the case's criteria, 7-point score, conflict
                                                     state and any bound guideline
  search_guidelines(condition, question_type)     -- the licensed guideline corpus. Conditions:
                                                     BCC, MEL, NEV, SK, MISC. Question types:
                                                     initial_local_management, referral_triage,
                                                     follow_up_surveillance, differential_diagnosis,
                                                     clinical_visual_assessment,
                                                     biopsy_histopathology.{gated_clause}
  override_criterion(criterion, value)            -- apply the clinician's correction to one
                                                     criterion and get the recomputed score
  explain_reasoning()                             -- walk through the full decision chain step by
                                                     step: classifier → criteria → 7-point score →
                                                     router → conflict → guideline → management →
                                                     contract verification, each with its provenance
  query_taxonomy(topic)                           -- query structural knowledge. Topics: criteria
                                                     (names, vocabularies, weights), conditions
                                                     (coverage map), management (ladder), router
                                                     (retrieval states and question types)"""

#: Appended to search_guidelines' tool doc only when gated_retrieval is on (F6: the ablation arm
#: must not have its own system prompt claim a restriction that flag has just switched off).
_GATED_RETRIEVAL_CLAUSE = (
    " Results are\n"
    "                                                     router-governed: a call outside this case's\n"
    "                                                     indicated retrieval plan returns the\n"
    "                                                     router's reason instead of guideline text"
)

#: Per-case nudge (F3b): built only when this case's router plan is non-empty. This is the SOFT
#: half of the F3 amendment -- a prompt-level signal where the old blanket 1c host check was a
#: hard gate. Enforcement moves to F3a's structural check, which owns the full rationale (see
#: _ASSESSMENT_WITHOUT_SEARCH_ISSUE below).
_RETRIEVAL_NUDGE = (
    "\nGuidance exists in the corpus for this case's routed conditions; consult it with "
    "search_guidelines before making any management claim.\n"
)

_CASE_MEMORY_TOOL_TMPL = """
  retrieve_similar_cases(k)                       -- find k visually similar lesions from the
                                                     evaluated case library (NUM_MEMORY cases).
                                                     Returns each case's system assessment:
                                                     diagnosis, criteria, score, conflict state.
                                                     These are past system evaluations, not verified
                                                     clinical outcomes. Use to contextualise the
                                                     current case against similar presentations."""

#: Appended to the tool list only when a dense index is attached (P1). The tool is deliberately
#: described as UNVERIFIED and NON-citable, so the prompt itself carries the wall.
_LITERATURE_TOOL_TMPL = """
  literature_lookup(query)                        -- UNVERIFIED exploration only. Dense
                                                     (similarity) retrieval over chunked guideline
                                                     text. Returns background passages, NOT citable
                                                     guideline records. Their chunk_ids are NOT
                                                     record IDs and must NEVER go in the citations
                                                     array. Use only to orient yourself; ground
                                                     every clinical claim in search_guidelines."""

#: One-line prompt note (P1), appended only when literature_lookup is available.
_LITERATURE_NOTE = (
    "\nliterature_lookup is exploration-only: its dense-retrieved passages are unverified "
    "background reading and can never be cited; only search_guidelines records may enter the "
    "citations array.\n"
)

_VISION_PREAMBLE = """
You can see the dermoscopy image attached to the clinician's question.

IMPORTANT: Your visual observations are DESCRIPTIVE, never diagnostic.  You may describe
what you see ("I observe irregular dark globules in the lower-right quadrant") but you
must NOT contradict or override the pipeline's criterion predictions or classifier
diagnosis based on what you see.  The classification comes from the deterministic pipeline;
your visual impression does not change any score or verdict.

Use your visual capability to:
- Describe image features that are consistent or inconsistent with the criterion values
- Point out specific regions the clinician should verify themselves
- Contextualise the system's predictions with visual observations
- Help the clinician understand WHY a criterion might have been classified a certain way

Never state "I diagnose this as X" or "this looks like melanoma to me."  Visual
observations supplement the pipeline's output, they do not replace it.
"""

# Rule 5 assumes a text-only agent; when vision is on it contradicts the vision preamble,
# so _build_system_prompt swaps in the vision variant. Text-only prompt bytes are unchanged.
_RULE_5_TEXT_ONLY = """5. You do not see the image. You cannot verify a visual claim; the clinician can, and their
   override wins."""

_RULE_5_VISION = """5. You see the image, but your visual impression is descriptive only and cannot verify or
   overturn a criterion value; the clinician's in-person examination can, and their override wins."""

_RULES_AND_FORMAT = """
Rules you must follow:

1. Ground every clinical statement in what a tool returned. If it is not in the decision or the
   guideline corpus, you do not know it.
2. Never state a 7-point score, a total, or a verdict that you calculated yourself. Those come
   from the tools. Quote them.
3. Never invent management advice. If the corpus holds no recommendation for the condition, say
   so plainly. An honest gap is correct; a plausible-sounding recommendation is not.
4. If you cannot answer from the available evidence, refuse and say what is missing.
5. You do not see the image. You cannot verify a visual claim; the clinician can, and their
   override wins.
6. The citations array must contain ONLY record IDs returned by search_guidelines. Never put
   DOIs, case IDs, tool provenance labels, or any other identifier in citations.
7. When answering from guidelines, quote recommendation text VERBATIM from search_guidelines
   results. Preface each quoted passage with the record ID. Do NOT paraphrase, summarise, or
   rephrase guideline text -- copy it exactly as returned by the tool.
8. Clearly separate guideline-sourced facts (quote them) from system-computed facts (attribute
   them to "the system's deterministic computation"). Never present a computed score, criterion
   value, or verdict as if it came from a guideline.
9. When reporting an override result, explicitly contrast the BEFORE and AFTER states: state
   the previous score, the new score, whether the threshold verdict changed, and whether the
   previous score was incorrect given the clinician's correction. The clinician needs to see
   what changed and why.
10. A router-gated search_guidelines result for a COVERED condition is not an absence of
    guidance -- say retrieval was not indicated for this case's routing; never claim the corpus
    lacks guidance unless the result says not_covered or gives an absence_reason.

Answer format -- structure the text field with labelled sections:

  Diagnosis: <classifier result, confidence, 7-point score, conflict state -- anything
             computed by the deterministic pipeline>
  Assessment: <guideline-sourced recommendations ONLY. Quote each recommendation verbatim
              with its record ID prefix. Every statement in this section must come from
              search_guidelines results. Do not mix in computed facts.>

If you have no guideline content, omit the Assessment section entirely.
If you have no diagnosis context to report, omit the Diagnosis section.
For override answers, use only Diagnosis (the override result is computed, not guideline-sourced).

Reply with a single JSON object and nothing else.

To use a tool:
  {"action": "tool", "tool": "<name>", "args": {...}}

To answer:
  {"action": "answer", "text": "<answer>", "citations": ["<guideline record id>", ...]}

To refuse:
  {"action": "refuse", "text": "<what is missing and why you cannot answer>"}
"""


#: Tool-count words, indexed by count. Base tools = 5; +case_memory and +literature_lookup each
#: add one. The default (no case memory, no literature tool) stays "five", and case-memory-only
#: stays "six", so the default and case-memory prompt bytes are unchanged by the P1 addition.
_TOOL_COUNT_WORDS = {5: "five", 6: "six", 7: "seven"}


def _build_system_prompt(
    case_memory: Optional[CaseMemory] = None,
    vision: bool = False,
    gated_retrieval: bool = True,
    retrieval_plan_nonempty: bool = False,
    literature_tool: bool = False,
) -> str:
    header = "You support a dermatologist reviewing an automated dermoscopy assessment.\n\n"
    if vision:
        header += _VISION_PREAMBLE + "\n"
    base_tools = _TOOL_LIST_BASE.format(
        gated_clause=_GATED_RETRIEVAL_CLAUSE if gated_retrieval else ""
    )
    n_tools = 5 + (case_memory is not None) + bool(literature_tool)
    header += f"You have {_TOOL_COUNT_WORDS[n_tools]} tools:\n\n"
    tools = base_tools
    if case_memory is not None:
        tools += _CASE_MEMORY_TOOL_TMPL.replace("NUM_MEMORY", str(len(case_memory)))
    if literature_tool:
        tools += _LITERATURE_TOOL_TMPL
    rules = _RULES_AND_FORMAT
    if vision:
        rules = rules.replace(_RULE_5_TEXT_ONLY, _RULE_5_VISION)
    nudge = _RETRIEVAL_NUDGE if retrieval_plan_nonempty else ""
    lit_note = _LITERATURE_NOTE if literature_tool else ""
    return header + tools + nudge + lit_note + rules


SYSTEM_PROMPT = _build_system_prompt(None)


def _build_multimodal_user_message(question: str, image_path: Optional[str] = None) -> Dict[str, Any]:
    """Construct a user message, optionally including a dermoscopy image."""
    if image_path is None:
        return {"role": "user", "content": question}
    p = Path(image_path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": question},
        ],
    }


#: F3 design amendment (supersedes spec 1c). The original retrieval-completeness check bounced
#: every answer that made no search_guidelines call while the router plan was non-empty -- full
#: stop, whether the answer was override arithmetic, a "why" question, or anything else with no
#: reason to search. Measured: it fired on 100% of why-questions and 41.5% of override questions
#: on plan-armed cases. Worse, that bounce was unsatisfiable: rule 9 says override answers use
#: only Diagnosis, so a compliant override answer has no Assessment section and no revision of
#: the answer TEXT could ever clear a check only an actual search call could clear -- and the
#: revision message's own escape hatch ("state explicitly why it is not needed") was never
#: honoured by the code. The replacement is narrower and self-consistent: rule 7 requires an
#: Assessment section to quote search_guidelines results, so an Assessment section with ZERO
#: search_guidelines calls contradicts itself -- but only where quotable content exists (router
#: plan non-empty). On NO_COVERAGE cases an Assessment stating the absence of guidance is the
#: desired answer and needs no search: the v5.0 run bounced those 254 times and the model
#: responded by deleting the section -- taking the stated absence with it (absence disclosure
#: 0.9912 -> 0.177). Scoping the check to plan-armed cases and dropping the "remove the
#: section" escape reverses that. The broader routing signal remains _RETRIEVAL_NUDGE.
_ASSESSMENT_WITHOUT_SEARCH_ISSUE = (
    "The answer contains an Assessment section, which under rule 7 must quote search_guidelines "
    "results verbatim, but no search_guidelines call was made this conversation. Consult "
    "search_guidelines before making guideline claims."
)


def _draft_issues(
    *,
    contract: DecisionContract,
    catalog: GuidelineCatalog,
    question: str,
    text: str,
    citations: Tuple[str, ...],
    tools: "AgentTools",
    chat: Callable[[List[Dict[str, Any]]], str],
    llm_verifier: bool,
) -> Tuple[List[str], str]:
    """Check one draft answer. Returns ``(issues, source)``, ``source`` in
    ``{"deterministic", "llm_verifier", "none"}``.

    Called once per drafted answer -- so on a revised second draft, this runs again from
    scratch. That is by design (it is how the host confirms the revision actually satisfies the
    checks it was asked to fix), but when ``llm_verifier`` is on and the first draft was clean
    enough to reach it, a clean revised draft reaches it too: that is a second paid
    ``verify_answer`` call, not a cached re-check.

    Five checks share one free, always-on "deterministic" bucket -- the contract/citation checks
    (1a), the exact override check (1a), the uncited-specifics check (v5.2 -- deliberately shares
    ``verification.ADVICE_SPECIFICS`` with ``eval.agent_eval.score_fabrication``, the same
    disclosure pattern as the ``extract_score_claim_numbers`` mirror between
    ``override_answer_checks`` and ``score_counterfactual``), the citation-array check (every entry
    of the structured ``citations`` array must resolve in the catalog -- closes the red-team array
    channel, see ``verification.citation_array_checks``), and the hard structural check that an
    Assessment section implies at least one search_guidelines call this conversation (1c,
    redesigned per F3) -- and that bucket always takes precedence: the optional LLM verifier (1b)
    is only consulted, and only costs a call, when it is completely clean.
    """
    issues = deterministic_checks(contract, text, catalog)

    if tools.override_sequence:
        issues += override_answer_checks(contract, tools.override_sequence, text)

    # Traceability of quoted specifics, reconciled across BOTH citation channels: a specific is
    # safe if a resolvable record id backs it inline (rule 7) OR in the structured array (rule 6).
    # Passing the catalog is what lets it resolve the inline channel -- without it the check falls
    # back to the array-emptiness test that produced the 14/15 false positives (see the function).
    issues += uncited_specifics_check(text, citations, catalog)

    # Security complement to the traceability check: the structured citations array is the field
    # the citation-resolvability endpoint is computed over, yet the checks above never require a
    # NON-empty one to fully resolve -- a fabricated id placed only there, with no specifics quoted,
    # was invisible in-loop (redteam_v1.json, architectural_gap_confirmed). Resolve EVERY entry
    # against the same catalog the eval scorer uses; runs whether or not the array is empty.
    issues += citation_array_checks(citations, catalog)

    searched = any(c.get("tool") == "search_guidelines" for c in tools.calls)
    if tools.router_plan and "Assessment:" in text and not searched:
        issues.append(_ASSESSMENT_WITHOUT_SEARCH_ISSUE)

    if issues:
        return issues, "deterministic"

    # verify_answer only ever sees the pre-override contract, so on override answers it
    # systematically flags legitimate clinician corrections as violations: measured 0.278
    # precision / 0.980 recall on override answers in
    # research/evidence/derm7pt_v1/llm_verifier_study_v1.json. override_answer_checks above
    # already owns this surface exactly, so skip the verifier here rather than let it
    # re-litigate what the host already checked correctly.
    if llm_verifier and not tools.overrides:
        result = verify_answer(
            contract=contract, question=question, answer_text=text, chat=chat, catalog=catalog,
        )
        if not result.passed:
            return list(result.issues), "llm_verifier"

    return [], "none"


def _revision_message(issues: List[str], source: str) -> str:
    """The host's rejection of a draft answer. Never quotes the answer text back, so it cannot
    leak into what the model repeats; explicitly forbids the model from mentioning the check."""
    numbered = " ".join(f"{i + 1}. {issue}" for i, issue in enumerate(issues))
    basis = (
        "did not pass deterministic checks against the decision contract"
        if source == "deterministic"
        else "was flagged by an independent verification pass against the decision contract"
    )
    return (
        f"HOST VERIFICATION FAILED — your draft answer {basis}: {numbered} "
        "Revise the answer. Same JSON format. Do not apologise, do not mention this check in "
        "the answer text."
    )


#: v5.3: issue substrings the host can point the model at an EXACT fix for -- the override
#: check's missing-revised-score message states the number to quote (override_answer_checks);
#: the uncited-specifics check states which field to fill in (uncited_specifics_check). A second
#: bounce is worth the extra turn only when EVERY issue on the once-revised draft is one of
#: these. Matched by substring against verification.py's own issue text rather than duplicated
#: as a second copy of either message, so the two stay in sync by construction.
_MECHANICALLY_FIXABLE_MARKERS = (
    "does not state the revised score",
    "empty citations array",
)


def _second_revision_eligible(issues: List[str], issue_source: str) -> bool:
    """True only when the draft that followed the FIRST revision still fails exclusively
    host-mechanically-fixable checks. ``issue_source`` must be "deterministic": an LLM-verifier
    issue (1b) is free text against no fixed template and never earns the second bounce, same as
    a general deterministic issue that isn't one of the two named above -- a wrong score, a
    misstated criterion, or the Assessment-without-search structural check. Those mean the model
    is not converging on a mechanical correction, so the conversation's remaining turns are
    better spent returning what it has than on a second redraft unlikely to fix it either.
    """
    if issue_source != "deterministic" or not issues:
        return False
    return all(
        any(marker in issue for marker in _MECHANICALLY_FIXABLE_MARKERS) for issue in issues
    )


def _verification_state(
    issues_first_draft: Optional[List[str]], revisions: int
) -> Dict[str, Any]:
    """Snapshot ``AgentAnswer.verification`` for a terminal return that is NOT the checked
    answer path -- a parse failure, an explicit refuse, or turn exhaustion. If a draft was
    already checked earlier in this conversation before the model refused, failed to parse, or
    ran out of turns, that failure must still be visible on the final AgentAnswer rather than
    silently dropped because the conversation ended a different way (F5). No later draft was
    ever scored in these cases, so the first draft's issues are also the last known ones.
    """
    if issues_first_draft is None:
        return {}
    return {
        "issues_first_draft": issues_first_draft,
        "revised": revisions > 0,
        "revisions": revisions,
        "issues_final": issues_first_draft,
    }


#: Displayed instead of the model's output when that output cannot be parsed. Deliberately says
#: nothing clinical: a reply the host could not read is a reply the host cannot vouch for.
_UNPARSEABLE_TEXT = "The assistant produced output that could not be parsed, so nothing is shown."


def _answer_text(raw: Any) -> str:
    """Flatten an answer's ``text`` field to a string.

    Models sometimes emit the answer format's own section labels as a JSON object,
    ``{"Diagnosis": ..., "Assessment": ...}``, instead of the single labelled string the format
    asks for. Rendering those keys back into labelled lines keeps the downstream checks -- which
    look for a literal "Assessment:" -- working on that shape too, rather than being handed a
    stringified dict they cannot read.
    """
    if not isinstance(raw, dict):
        return str(raw)
    parts = [
        f"{section}: {raw[section]}"
        for section in ("Diagnosis", "Assessment")
        if raw.get(section)
    ]
    return "\n".join(parts) if parts else str(raw)


def _dispatch_tool(tools: AgentTools, name: str, args: Dict[str, Any]) -> Optional[ToolResult]:
    """Route one model-issued tool call onto ``tools``. ``None`` means the name is not a tool.

    Arguments arrive straight out of model JSON, so each is coerced here rather than trusted;
    the tools themselves then reject anything outside their own vocabularies.
    """
    if name == "get_decision":
        return tools.get_decision()
    if name == "search_guidelines":
        return tools.search_guidelines(
            str(args.get("condition", "")), str(args.get("question_type", ""))
        )
    if name == "override_criterion":
        return tools.override_criterion(
            str(args.get("criterion", "")), str(args.get("value", ""))
        )
    if name == "explain_reasoning":
        return tools.explain_reasoning()
    if name == "query_taxonomy":
        return tools.query_taxonomy(str(args.get("topic", "")))
    if name == "retrieve_similar_cases":
        try:
            k = int(args.get("k", 3))
        except (TypeError, ValueError):
            return ToolResult(False, {}, f"k must be an integer, got {args.get('k')!r}")
        return tools.retrieve_similar_cases(k=k)
    if name == "literature_lookup":
        try:
            k = int(args.get("k", 3))
        except (TypeError, ValueError):
            return ToolResult(False, {}, f"k must be an integer, got {args.get('k')!r}")
        return tools.literature_lookup(str(args.get("query", "")), k=k)
    return None


def run_agent(
    *,
    contract: DecisionContract,
    catalog: GuidelineCatalog,
    question: str,
    chat: Callable[[List[Dict[str, Any]]], str],
    model: str = "unknown",
    max_turns: int = MAX_TURNS,
    case_memory: Optional[CaseMemory] = None,
    query_embedding: Optional[Any] = None,
    image_path: Optional[str] = None,
    llm_verifier: bool = False,
    gated_retrieval: bool = True,
    dense_index: Optional[Any] = None,
) -> AgentAnswer:
    """Drive the agent loop. ``chat`` takes messages and returns the raw model string.

    When ``image_path`` is provided, the dermoscopy image is included in the
    first user message so the VLM can describe visual features.  The image is
    for descriptive support only — classification still comes from the pipeline.

    ``llm_verifier`` (default off; see research/evidence/derm7pt_v1/llm_verifier_study_v1.json
    for the catch-rate study that keeps it off) additionally consults ``verify_answer`` once the
    free deterministic checks pass -- except on override answers, where it is skipped entirely
    (the verifier only sees the pre-override contract and mismeasures those; see the comment in
    ``_draft_issues``). ``gated_retrieval`` (default on) RESTRICTS ``search_guidelines`` to the
    router's plan for this case -- it is a restriction, not an addition. The ablation arm
    (``gated_retrieval=False``) removes that restriction and reproduces clinician-agent-v4's
    ungated behaviour, where every call is served in full regardless of routing. The free
    host-verification checks (1a exact contract/override checks, 1c the Assessment-without-
    search structural check) are always on regardless of either flag.

    v5.3: a failing draft is revised once unconditionally, same as before. If the draft that
    follows STILL fails, a second revision fires only when ``_second_revision_eligible`` holds --
    every remaining issue is host-mechanically-fixable (the missing-revised-score or uncited-
    specifics checks). Capped at ``MAX_REVISIONS`` (2) regardless; MAX_TURNS is unchanged.
    """
    tools = AgentTools(
        contract, catalog, case_memory=case_memory, query_embedding=query_embedding,
        gated_retrieval=gated_retrieval, dense_index=dense_index,
    )
    vision = image_path is not None
    seen_calls: Set[str] = set()
    issues_first_draft: Optional[List[str]] = None
    revisions = 0

    def _refusal(text: str, *, turns: int, raw_error: Optional[str] = None) -> AgentAnswer:
        """Terminal refusal carrying whatever state the conversation had reached -- above all the
        verification snapshot, so a first-draft failure is not dropped just because the
        conversation ended by refusing rather than by answering (see ``_verification_state``).
        Reads the loop's own locals at call time, so it always reports the latest values."""
        return AgentAnswer(
            kind="refusal",
            text=text,
            tool_calls=tuple(tools.calls),
            turns=turns,
            overrides_applied=dict(tools.overrides),
            raw_error=raw_error,
            verification=_verification_state(issues_first_draft, revisions),
            retrieval=tools.retrieval_summary(),
        )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_system_prompt(
                case_memory, vision=vision, gated_retrieval=gated_retrieval,
                retrieval_plan_nonempty=bool(tools.router_plan),
                literature_tool=dense_index is not None,
            ),
        },
        _build_multimodal_user_message(question, image_path),
    ]

    for turn in range(1, max_turns + 1):
        raw = chat(messages)
        try:
            decision = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        except json.JSONDecodeError as exc:
            return _refusal(
                _UNPARSEABLE_TEXT, turns=turn, raw_error=f"JSONDecodeError: {exc}"
            )

        if not isinstance(decision, dict):
            return _refusal(
                _UNPARSEABLE_TEXT,
                turns=turn,
                raw_error=f"top-level JSON is {type(decision).__name__}, not an object",
            )

        action = str(decision.get("action", ""))
        remaining = max_turns - turn

        if action == "answer":
            text = _answer_text(decision.get("text", ""))
            citations = tuple(str(c) for c in decision.get("citations", []) or [])

            issues, issue_source = _draft_issues(
                contract=contract, catalog=catalog, question=question, text=text,
                citations=citations, tools=tools, chat=chat, llm_verifier=llm_verifier,
            )
            if issues_first_draft is None:
                issues_first_draft = list(issues)

            # v5.3: the first revision is unconditional (any issues at all, same as before). A
            # second is granted only when the draft that follows the first one STILL fails, and
            # fails EXCLUSIVELY host-mechanically-fixable checks -- see _second_revision_eligible.
            eligible = revisions == 0 or _second_revision_eligible(issues, issue_source)
            if issues and eligible and revisions < MAX_REVISIONS and remaining > 0:
                revisions += 1
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {"role": "user", "content": _revision_message(issues, issue_source)}
                )
                continue

            return AgentAnswer(
                kind="answer",
                text=text,
                citations=citations,
                tool_calls=tuple(tools.calls),
                turns=turn,
                overrides_applied=dict(tools.overrides),
                verification={
                    "issues_first_draft": issues_first_draft,
                    "revised": revisions > 0,
                    "revisions": revisions,
                    "issues_final": list(issues),
                },
                retrieval=tools.retrieval_summary(),
            )

        if action == "refuse":
            return _refusal(str(decision.get("text", "")), turns=turn)

        if action == "tool":
            name = str(decision.get("tool", ""))
            args = decision.get("args", {}) or {}

            # Repeated identical searches were the single largest failure mode: the agent would
            # call search_guidelines five times and exhaust its turn budget without concluding.
            # Returning the same payload again teaches it nothing, so say so explicitly and
            # remind it how many turns remain.
            signature = json.dumps({"tool": name, "args": args}, sort_keys=True)
            repeated = signature in seen_calls
            seen_calls.add(signature)

            if repeated:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": json.dumps({
                        "observation": {
                            "ok": False,
                            "note": (
                                f"You already made this exact call and the result has not changed. "
                                f"Use what you have. {remaining} turn(s) remain before the question "
                                f"is referred back to the clinician unanswered."
                            ),
                        }
                    }),
                })
                continue

            result = _dispatch_tool(tools, name, args)
            if result is None:
                observation = {"ok": False, "note": f"unknown tool {name!r}"}
            else:
                observation = {
                    "ok": result.ok,
                    "payload": result.payload,
                    "note": result.note,
                    "turns_remaining": remaining,
                }
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": json.dumps({"observation": observation}, default=str)}
            )
            continue

        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"observation": {"ok": False, "note": "unrecognised action; use tool/answer/refuse"}}
                ),
            }
        )

    return _refusal(
        f"No conclusion within {max_turns} turns; the question is referred back to the clinician.",
        turns=max_turns,
    )
