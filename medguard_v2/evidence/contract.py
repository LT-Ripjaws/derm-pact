"""The decision contract a clinician actually reads.

The design target is not a metric.  It is: **a dermatologist should be able to check this system's
work in about ten seconds, and disagree with any single line of it.**  That constraint drives
everything below.

- The criteria table is the reasoning, not a post-hoc explanation of it.  The score is computed
  from those seven rows by the published rule, so a reader who disputes one row can recompute the
  consequence themselves.
- The two independent views -- the diagnosis classifier and the criteria path -- are reported
  separately and **their disagreement is surfaced, never resolved silently**.  On Derm7pt they
  catch different melanomas (9 each of 61, 25 shared), so a hidden tie-break would discard the
  most clinically useful signal the system has.
- Guideline text is reproduced verbatim with its citation and content hash, or its absence is
  stated.  No management sentence is ever generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from medguard_v2.criteria.aggregate import (
    SevenPointScore,
    recommend_management,
    score_criteria,
)
from medguard_v2.criteria.schema import CRITERION_NAMES, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.evidence.guideline_binding import BindingResult, GuidelineCatalog


@dataclass(frozen=True)
class ConflictState:
    """Disagreement between the two views. The most decision-relevant field in the contract."""

    present: bool
    classifier_says: str
    criteria_path_flags_melanoma: bool
    description: str
    escalate: bool


@dataclass(frozen=True)
class SimilarCase:
    """One retrieved neighbour: a confirmed TRAIN case and its cosine similarity to the query."""

    #: Dataset case_num (string), the retrieval id-space -- NOT the display ``case_id`` used above.
    case_id: str
    #: The neighbour's CONFIRMED (ground-truth) diagnosis group.
    diagnosis: str
    #: Exact cosine similarity to the query in the frozen DermLIP feature space.
    similarity: float
    #: ``diagnosis == reference_diagnosis`` (see SimilarCasesContext). Precomputed so the render/agent
    #: never has to re-derive which label the retrieval is being compared against.
    same_label: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "diagnosis": self.diagnosis,
            "similarity": round(float(self.similarity), 4),
            "same_label": self.same_label,
        }


@dataclass(frozen=True)
class SimilarCasesContext:
    """Faithful-by-construction, ON-DEMAND case-based explanation for one decision.

    The neighbours are the query's own nearest points in the frozen DermLIP feature space the
    perception reader itself reads. That makes this explanation faithful *by construction*: it is
    the identity of the case in the model's representation, not a post-hoc rationalisation of a
    decision the model reached some other way. The bank is TRAIN-only and queried leave-one-out, so
    a case never retrieves itself (the leak-free property is host-verified in ``verification_failures``).

    Advisory metadata only: like ``criterion_margins`` it never touches the 7-point score, the
    conflict verdict, or ``safe_to_display`` except to reject a *malformed* context (a self-leak or an
    out-of-range cosine), which would only ever arise from a wiring bug.
    """

    #: The query's dataset case_num (the id-space the neighbours live in), so a self-leak is a
    #: format-consistent equality check rather than a cross-format comparison with the display case_id.
    query_case_id: str
    #: The label ``same_label`` is measured against. In production this is the COMMITTED classifier
    #: diagnosis (what the clinician is told); the retrieval-sanity eval measures it against the true
    #: label instead. Recorded so the meaning of ``same_label`` is never ambiguous.
    reference_diagnosis: str
    #: Top-k neighbours, descending similarity.
    neighbors: Tuple[SimilarCase, ...]
    #: The nearest confirmed case whose label differs from ``reference_diagnosis`` -- the boundary
    #: counterfactual ("the nearest confirmed melanoma is train case X at cosine 0.84"). ``None`` only
    #: in a degenerate single-label bank.
    nearest_different_label: Optional[SimilarCase]
    #: Human-readable description of the retrieval feature space, e.g.
    #: "dermlip (512-d frozen image embedding; image-only)".
    feature_space: str
    #: Provenance of the frozen bank the neighbours came from.
    bank_name: str
    bank_hash: str
    #: The k that was requested (``len(neighbors)`` may be smaller for a tiny bank).
    k: int

    def same_label_neighbor_rate(self) -> Optional[float]:
        if not self.neighbors:
            return None
        return sum(1 for n in self.neighbors if n.same_label) / len(self.neighbors)

    def as_dict(self) -> Dict[str, object]:
        rate = self.same_label_neighbor_rate()
        return {
            "query_case_id": self.query_case_id,
            "reference_diagnosis": self.reference_diagnosis,
            "neighbors": [n.as_dict() for n in self.neighbors],
            "nearest_different_label": (
                self.nearest_different_label.as_dict()
                if self.nearest_different_label is not None
                else None
            ),
            "same_label_neighbor_rate": (round(rate, 4) if rate is not None else None),
            "feature_space": self.feature_space,
            "bank_name": self.bank_name,
            "bank_hash": self.bank_hash,
            "k": self.k,
            "faithful_by_construction": (
                "Neighbours are the query's nearest TRAIN cases by exact cosine in the frozen "
                "DermLIP feature space the perception reader itself reads -- retrieval is identity in "
                "that space, not a post-hoc rationalisation. Train-only and leave-one-out, so a case "
                "never retrieves itself."
            ),
        }


def similar_cases_failures(sc: SimilarCasesContext) -> List[str]:
    """Host checks for one attached :class:`SimilarCasesContext`.

    Module-level (not a method) because the same obligations hold wherever the context is
    attached -- the dual-view :class:`DecisionContract` below and the single-view contract in
    ``evidence.single_view`` -- and two copies of the leak-free check would drift. Two things
    must hold or the context is a wiring bug, not something to show: (1) the leak-free property
    -- the query case must never appear among its own neighbours (the whole "train-only,
    leave-one-out" claim rests on this); (2) every reported similarity is a real cosine in
    [-1, 1]. Making these host-verified is what lets the render and the agent state "faithful
    by construction" without the reader having to take it on trust.
    """
    failures: List[str] = []
    neighbours = list(sc.neighbors)
    if sc.nearest_different_label is not None:
        neighbours.append(sc.nearest_different_label)
    if not sc.neighbors:
        failures.append("similar_cases is present but carries no neighbours")
    if any(nb.case_id == sc.query_case_id for nb in neighbours):
        failures.append(
            f"similar_cases leaks the query case {sc.query_case_id} into its own neighbours "
            "(train-only/leave-one-out violated)"
        )
    offending = [nb for nb in neighbours if not (-1.0001 <= nb.similarity <= 1.0001)]
    if offending:
        failures.append(
            "similar_cases carries a similarity outside the cosine range [-1,1]: "
            f"{[(nb.case_id, round(nb.similarity, 4)) for nb in offending]}"
        )
    return failures


@dataclass(frozen=True)
class DecisionContract:
    case_id: str
    classifier_diagnosis: str
    classifier_confidence: float
    criteria: Dict[str, str]
    score: SevenPointScore
    criteria_implied_action: str
    criteria_implied_reason: str
    conflict: ConflictState
    guideline: BindingResult
    unresolved_criteria: Tuple[str, ...]
    #: Guidance for the condition the CHECKLIST is raising, when that differs from the committed
    #: diagnosis. Contingent on the checklist's reading being right; never presented as settled.
    contingent_guideline: Optional[BindingResult] = None
    receipt: Dict[str, object] = field(default_factory=dict)
    #: Per-criterion model uncertainty (top-minus-second posterior of the fitted head), keyed by
    #: criterion name. Populated by the promoted perception path (run_cohort promoted=True); left
    #: ``None`` by the pre-promotion path, which keeps this contract byte-identical to what it
    #: serialised before the field existed. Advisory metadata only: it lets the rendered hinge line
    #: rank by ascending margin (most-uncertain criterion first) and never changes the 7-point
    #: score, the conflict verdict, or safe_to_display.
    criterion_margins: Optional[Dict[str, float]] = None
    #: Faithful-by-construction case-based explanation (nearest confirmed TRAIN cases + the boundary
    #: counterfactual), attached by the retrieval builder. ON-DEMAND: it is NOT rendered in the
    #: default 10-second view (MG52 progressive disclosure) and is left ``None`` by every existing
    #: path, so a contract without it serialises byte-identically to what it did before the field
    #: existed. Advisory metadata; see SimilarCasesContext.
    similar_cases: Optional[SimilarCasesContext] = None

    # -- the obligations the host checks before this contract may be shown -----

    def verification_failures(self) -> List[str]:
        """Everything that must hold. A non-empty list means do not display."""
        failures: List[str] = []

        if set(self.criteria) != set(CRITERION_NAMES):
            failures.append("criteria table is incomplete")

        recomputed = score_criteria(self.criteria)
        if (recomputed.score_min, recomputed.score_max) != (
            self.score.score_min,
            self.score.score_max,
        ):
            failures.append("score does not match the criteria table it claims to summarise")

        if self.guideline.covered:
            # Covered-but-empty is a wiring bug the render would crash on (it indexes
            # recommendations[0]); the single-view type has always checked this -- mirrored here.
            if not self.guideline.recommendations:
                failures.append("guideline claims coverage but carries no recommendations")
            for rec in self.guideline.recommendations:
                if not rec.verify():
                    failures.append(f"guideline record {rec.record_id} failed its content hash")
        elif self.guideline.absence_reason is None:
            failures.append("guideline is not covered but no reason was recorded")

        # contingent_guideline (the CHECKLIST's condition, attached when the checklist raises
        # melanoma against a non-MEL commitment, or the score is undecidable) carries its own
        # verbatim recommendation text just like the committed guideline does, and was not being
        # hash-reverified here -- a tampered contingent record would have passed safe_to_display.
        if self.contingent_guideline is not None and self.contingent_guideline.covered:
            for rec in self.contingent_guideline.recommendations:
                if not rec.verify():
                    failures.append(
                        f"contingent guideline record {rec.record_id} failed its content hash"
                    )

        if self.conflict.present and not self.conflict.description:
            failures.append("conflict flagged without a description")

        if tuple(self.score.indeterminate) != tuple(self.unresolved_criteria):
            failures.append("unresolved criteria disagree with the score's own record")

        # criterion_margins is optional, advisory metadata. Round-trip safely: when present, every
        # key must name a real criterion (a malformed margins map that referenced an unknown
        # criterion would be a wiring bug, not something to display). ``None`` -- the pre-promotion
        # default -- skips the check entirely, so no existing contract's verdict changes.
        if self.criterion_margins is not None:
            unknown = sorted(set(self.criterion_margins) - set(CRITERION_NAMES))
            if unknown:
                failures.append(f"criterion_margins references unknown criteria: {unknown}")

        # similar_cases is optional, advisory, ON-DEMAND metadata -- ``None`` (every existing path)
        # skips this entirely. When present, the shared module-level checks apply (leak-free
        # property + cosine range; see similar_cases_failures' docstring for why they are
        # host-verified and why they live at module level).
        if self.similar_cases is not None:
            failures.extend(similar_cases_failures(self.similar_cases))

        return failures

    @property
    def safe_to_display(self) -> bool:
        return not self.verification_failures()


def _build_conflict(classifier_diagnosis: str, score: SevenPointScore) -> ConflictState:
    criteria_flags = score.score_min >= MELANOMA_SUSPICION_THRESHOLD
    classifier_says_mel = classifier_diagnosis == "MEL"

    # Undecidable comes first, for every diagnosis. An unresolved checklist cannot agree OR
    # disagree with the classifier, and score_min >= threshold implies decidable, so no true
    # conflict is masked by this ordering. Mirrors router.classify_state.
    if not score.decidable:
        return ConflictState(
            present=False,
            classifier_says=classifier_diagnosis,
            criteria_path_flags_melanoma=False,
            description=(
                f"The pattern model reports {classifier_diagnosis}. The checklist score range "
                f"[{score.score_min},{score.score_max}] spans the threshold of "
                f"{MELANOMA_SUSPICION_THRESHOLD} due to unresolved criteria "
                f"{list(score.indeterminate)}: the checklist has not reached a verdict, so "
                f"agreement cannot be assessed. Resolve the unresolved criteria to decide."
            ),
            escalate=True,
        )

    if criteria_flags and not classifier_says_mel:
        return ConflictState(
            present=True,
            classifier_says=classifier_diagnosis,
            criteria_path_flags_melanoma=True,
            description=(
                f"The pattern model reports {classifier_diagnosis}, but the 7-point checklist "
                f"scores {score.score_min}, at or above the threshold of "
                f"{MELANOMA_SUSPICION_THRESHOLD}. The two views disagree in the direction that "
                f"matters: the checklist is raising melanoma suspicion the classifier is not."
            ),
            escalate=True,
        )

    if classifier_says_mel and not criteria_flags:
        return ConflictState(
            present=True,
            classifier_says=classifier_diagnosis,
            criteria_path_flags_melanoma=False,
            description=(
                f"The pattern model reports melanoma, but the checklist scores "
                f"{score.score_max}, below the threshold of {MELANOMA_SUSPICION_THRESHOLD}. "
                f"The classifier is raising suspicion the checklist does not support."
            ),
            escalate=True,
        )

    return ConflictState(
        present=False,
        classifier_says=classifier_diagnosis,
        criteria_path_flags_melanoma=criteria_flags,
        description="Both views agree.",
        escalate=False,
    )


def build_contract(
    *,
    case_id: str,
    classifier_diagnosis: str,
    classifier_confidence: float,
    criteria: Dict[str, str],
    catalog: GuidelineCatalog,
    receipt: Optional[Dict[str, object]] = None,
    criterion_margins: Optional[Dict[str, float]] = None,
    similar_cases: Optional[SimilarCasesContext] = None,
) -> DecisionContract:
    """Assemble the contract. Every field is derived; none is generated.

    ``criterion_margins`` (optional) carries the per-criterion model uncertainty (top-minus-second
    head posterior) from the promoted perception path. It is copied defensively and stored as
    advisory metadata; it never participates in scoring or the conflict verdict.

    ``similar_cases`` (optional) carries the faithful-by-construction case-based explanation built by
    the retrieval primitive (``agent.case_memory.build_similar_cases_context``). It is ON-DEMAND
    metadata: attached here so it round-trips and is host-verifiable, but never rendered in the
    default view. Left ``None`` by every existing caller, keeping their output byte-identical.
    """
    score = score_criteria(criteria)
    action, reason = recommend_management(score)
    conflict = _build_conflict(classifier_diagnosis, score)

    # Guideline binding follows the CLASSIFIER's committed condition, because that is what the
    # patient is being told they have. When the checklist disagrees, that is surfaced as a
    # conflict rather than silently switching which guideline is quoted.
    guideline = catalog.bind(classifier_diagnosis)

    # But a conflict case needs more than that. If the classifier says naevus and the checklist
    # is raising melanoma suspicion, quoting "no guidance exists for naevus" answers a question
    # nobody asked -- the clinician is deciding what to do about a possible melanoma. So the
    # guidance for the condition the CHECKLIST is raising is bound as well, clearly labelled as
    # contingent on that reading rather than on the committed diagnosis.
    #
    # Two triggers, not one. The explicit conflict above is the case where the checklist has
    # already tipped past the threshold while the classifier disagrees. But an undecidable score
    # -- unresolved criteria whose range spans the threshold -- is exactly as capable of turning
    # out to be melanoma; it just has not been asked to declare itself yet. Attaching contingent
    # guidance only to the resolved case would make its presence an artefact of criterion-scoring
    # order rather than of actual risk, so both triggers bind it, for every non-MEL diagnosis.
    checklist_may_raise_melanoma = (
        (conflict.present and conflict.criteria_path_flags_melanoma) or not score.decidable
    )
    contingent = None
    if checklist_may_raise_melanoma and classifier_diagnosis != "MEL":
        candidate = catalog.bind("MEL")
        if candidate.covered:
            contingent = candidate

    return DecisionContract(
        case_id=case_id,
        classifier_diagnosis=classifier_diagnosis,
        classifier_confidence=round(float(classifier_confidence), 4),
        criteria=dict(criteria),
        score=score,
        criteria_implied_action=action,
        criteria_implied_reason=reason,
        conflict=conflict,
        guideline=guideline,
        contingent_guideline=contingent,
        unresolved_criteria=tuple(score.indeterminate),
        receipt=dict(receipt or {}),
        criterion_margins=(dict(criterion_margins) if criterion_margins is not None else None),
        similar_cases=similar_cases,
    )


def contract_to_dict(contract: DecisionContract) -> Dict[str, object]:
    """Serialisable form, for the evidence record rather than the clinician.

    ``criterion_margins`` is emitted **only when present** so a pre-promotion contract (margins
    ``None``) serialises to exactly the bytes it did before the field existed -- the pinned
    ``cdss_contracts_valid.json`` digest depends on this.
    """
    payload: Dict[str, object] = {
        "case_id": contract.case_id,
        "classifier": {
            "diagnosis": contract.classifier_diagnosis,
            "confidence": contract.classifier_confidence,
        },
        "criteria": [
            {
                "name": c.name,
                "value": c.value,
                "points": c.points,
                "max_points": c.max_points,
                "determinate": c.determinate,
            }
            for c in contract.score.per_criterion
        ],
        "seven_point_score": {
            "min": contract.score.score_min,
            "max": contract.score.score_max,
            "determinate": contract.score.determinate,
            "decidable": contract.score.decidable,
            "threshold": MELANOMA_SUSPICION_THRESHOLD,
        },
        "criteria_implied_action": contract.criteria_implied_action,
        "criteria_implied_reason": contract.criteria_implied_reason,
        "conflict": {
            "present": contract.conflict.present,
            "escalate": contract.conflict.escalate,
            "description": contract.conflict.description,
        },
        "guideline": contract.guideline.as_dict(),
        # Contingent guidance exists precisely for the cases where the checklist may be raising
        # melanoma against the committed diagnosis; omitting it here would hide it from the
        # agent's get_decision view on exactly those cases.
        "contingent_guideline": (
            contract.contingent_guideline.as_dict()
            if contract.contingent_guideline is not None
            else None
        ),
        "unresolved_criteria": list(contract.unresolved_criteria),
        "verification_failures": contract.verification_failures(),
        "safe_to_display": contract.safe_to_display,
        "receipt": contract.receipt,
    }
    if contract.criterion_margins is not None:
        payload["criterion_margins"] = {
            name: round(float(margin), 6)
            for name, margin in contract.criterion_margins.items()
        }
    # ON-DEMAND case-based explanation, emitted only when present so a contract without it (every
    # pre-existing path) serialises to exactly the bytes it did before the field existed -- same
    # discipline as criterion_margins above, on which the pinned cdss_contracts_valid.json depends.
    if contract.similar_cases is not None:
        payload["similar_cases"] = contract.similar_cases.as_dict()
    return payload
