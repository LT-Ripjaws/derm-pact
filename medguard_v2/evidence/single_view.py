"""The classifier-only decision contract, for imaging sources with NO 7-point annotations.

HAM10000 and BCN20000 carry a diagnosis label but no dermoscopic-criteria annotations, so the
dual-view :class:`~medguard_v2.evidence.contract.DecisionContract` -- whose whole design is two
independent views whose disagreement is surfaced -- cannot exist for them. The wrong fix would be
to loosen that contract (Optional-ize criteria/score/conflict): every evaluated number, agent
tool and verification check rests on those invariants, and a contract that can silently lack its
checklist is a contract whose absence-of-conflict means nothing.

So this is a SEPARATE type, honest about what it is:

- **One view, and it says so.** ``checklist_unavailable_reason`` is a mandatory disclosure --
  the same philosophy as the guideline layer's ``absence_reason``: a silent gap reads as
  "nothing to say", an explicit one reads as "this source cannot support a cross-check", and
  those are different clinical facts.
- **No invented management.** The management surface is derived from the classifier's committed
  diagnosis via the hash-pinned guideline binding ONLY: covered means the next step is to follow
  the cited record (whose text is reproduced verbatim by the render), uncovered means the absence
  is stated. There is no checklist score to map onto the management ladder, so no ladder step is
  claimed.
- **Its own verification, mirroring what applies.** Guideline hash checks, absence-reason
  requirements and the similar-cases leak-free/cosine checks are shared with the dual-view
  contract; score/criteria checks are absent because there is nothing to check -- not because
  checking was relaxed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from medguard_v2.evidence.contract import SimilarCasesContext, similar_cases_failures
from medguard_v2.evidence.guideline_binding import (
    BindingResult,
    GROUP_FULL_NAME,
    GuidelineCatalog,
)


@dataclass(frozen=True)
class SingleViewContract:
    """A classifier-only decision. Everything the host can honestly say with one view."""

    case_id: str
    classifier_diagnosis: str
    classifier_confidence: float
    #: The mandatory disclosure: WHY no dermoscopic-criteria checklist exists for this case,
    #: e.g. "this imaging source carries no dermoscopic-criteria annotations; single-view
    #: decision". Rendered verbatim in the checklist's slot; an empty string fails verification.
    checklist_unavailable_reason: str
    #: Hash-pinned binding of the COMMITTED diagnosis, exactly as the dual-view contract binds it.
    guideline: BindingResult
    receipt: Dict[str, object] = field(default_factory=dict)
    #: ON-DEMAND case-based explanation (see SimilarCasesContext). Same discipline as the
    #: dual-view contract: advisory, never rendered in the default view, host-verified when present.
    similar_cases: Optional[SimilarCasesContext] = None

    # -- the obligations the host checks before this contract may be shown -----

    def verification_failures(self) -> List[str]:
        """Everything that must hold. A non-empty list means do not display.

        Mirrors the dual-view checks that APPLY to a single view (guideline hashes,
        absence-reason-required, similar-cases leak/cosine). There are deliberately no
        score/criteria checks: this contract carries no checklist, and its render must
        disclose that rather than pretend one was verified.
        """
        failures: List[str] = []

        if not self.checklist_unavailable_reason.strip():
            failures.append(
                "single-view contract carries no checklist-unavailable disclosure "
                "(the absence must be stated, not implied)"
            )

        if self.guideline.covered:
            if not self.guideline.recommendations:
                failures.append("guideline claims coverage but carries no recommendations")
            for rec in self.guideline.recommendations:
                if not rec.verify():
                    failures.append(f"guideline record {rec.record_id} failed its content hash")
        elif self.guideline.absence_reason is None:
            failures.append("guideline is not covered but no reason was recorded")

        if self.similar_cases is not None:
            failures.extend(similar_cases_failures(self.similar_cases))

        return failures

    @property
    def safe_to_display(self) -> bool:
        return not self.verification_failures()


def single_view_action(contract: SingleViewContract) -> Tuple[str, str]:
    """``(action, reason)`` derived from the classifier commitment + guideline binding ONLY.

    The single-view analogue of ``recommend_management`` -- except there is no checklist score
    to map onto the management ladder, so no ladder step is invented. Covered: the action is
    literally to follow the cited record (whose text the render reproduces verbatim).
    Uncovered: the action states the absence, same philosophy as the dual-view absence block.
    """
    full = GROUP_FULL_NAME.get(contract.classifier_diagnosis, contract.classifier_diagnosis)
    if contract.guideline.covered and contract.guideline.recommendations:
        citation = contract.guideline.recommendations[0].citation_id
        return (f"follow {citation}", f"pattern model reads {full}")
    return ("no citable management", f"pattern model reads {full}; the corpus holds no recommendation")


def build_single_view_contract(
    *,
    case_id: str,
    classifier_diagnosis: str,
    classifier_confidence: float,
    catalog: GuidelineCatalog,
    reason: str,
    receipt: Optional[Dict[str, object]] = None,
    similar_cases: Optional[SimilarCasesContext] = None,
) -> SingleViewContract:
    """Assemble a single-view contract. Every field is derived or disclosed; none is generated.

    ``reason`` is the mandatory checklist-unavailable disclosure (see the field docstring for
    the canonical HAM/BCN text). It is required rather than defaulted so a caller can never
    silently attach the wrong story to a source; an empty string fails verification.

    The guideline binding follows the classifier's committed condition through the same
    hash-pinned ``catalog.bind`` path the dual-view contract uses (an unknown diagnosis group
    raises ``KeyError`` there, before any contract exists). There is no contingent binding:
    contingent guidance exists to represent the CHECKLIST's dissent, and there is no checklist.
    """
    guideline = catalog.bind(classifier_diagnosis)
    return SingleViewContract(
        case_id=case_id,
        classifier_diagnosis=classifier_diagnosis,
        classifier_confidence=round(float(classifier_confidence), 4),
        checklist_unavailable_reason=reason,
        guideline=guideline,
        receipt=dict(receipt or {}),
        similar_cases=similar_cases,
    )


def single_view_contract_to_dict(contract: SingleViewContract) -> Dict[str, object]:
    """Serialisable form, for the evidence record rather than the clinician.

    ``similar_cases`` is emitted only when present, the same discipline as the dual-view
    serialiser. ``view: "single"`` makes the record class machine-distinguishable from a
    dual-view record without inspecting which keys are missing.
    """
    action, action_reason = single_view_action(contract)
    payload: Dict[str, object] = {
        "case_id": contract.case_id,
        "view": "single",
        "classifier": {
            "diagnosis": contract.classifier_diagnosis,
            "confidence": contract.classifier_confidence,
        },
        "checklist_unavailable_reason": contract.checklist_unavailable_reason,
        "derived_action": {
            "action": action,
            "reason": action_reason,
            "source": (
                "classifier diagnosis + hash-pinned guideline binding only; no checklist "
                "exists for this source and no management text is generated"
            ),
        },
        "guideline": contract.guideline.as_dict(),
        "verification_failures": contract.verification_failures(),
        "safe_to_display": contract.safe_to_display,
        "receipt": contract.receipt,
    }
    if contract.similar_cases is not None:
        payload["similar_cases"] = contract.similar_cases.as_dict()
    return payload
