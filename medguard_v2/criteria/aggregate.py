"""Deterministic 7-point aggregation, including under partial information.

Two properties matter here and neither is available to v1's architecture.

**The host does the arithmetic.**  The assessor decides whether a criterion is present; this
module turns those judgements into a score.  No language model is in the loop, so the score is
reproducible and a criterion can be removed and the case re-scored without regenerating anything.
That is what makes leave-one-criterion-out influence testing free.

**Indeterminate criteria do not force an abstention.**  If the criteria that *were* assessed
already put the total above or below the threshold no matter how the unknown ones resolve, the
case is still decidable.  v1 abstained on the whole case whenever the evidence was incomplete,
which is why it went silent on 41.7% of committed melanomas.  Here an unresolved criterion only
blocks a decision when it could actually change it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from medguard_v2.criteria.schema import (
    CRITERIA_BY_NAME,
    CRITERION_NAMES,
    INDETERMINATE,
    MAX_SCORE,
    MELANOMA_SUSPICION_THRESHOLD,
    validate_values,
)

#: Ordinal management ladder, matching Derm7pt's ``management`` column.  Ordered by escalation
#: so ordinal statistics (quadratic-weighted kappa, over/under-management) are well defined.
MANAGEMENT_LADDER: Tuple[str, ...] = (
    "no further examination",
    "clinical follow up",
    "excision",
)


@dataclass(frozen=True)
class CriterionScore:
    name: str
    value: str
    points: int
    max_points: int
    determinate: bool


@dataclass(frozen=True)
class SevenPointScore:
    """The aggregate, carrying its own uncertainty rather than collapsing it."""

    per_criterion: Tuple[CriterionScore, ...]
    score_min: int
    score_max: int
    indeterminate: Tuple[str, ...] = field(default=())

    @property
    def determinate(self) -> bool:
        return not self.indeterminate

    @property
    def score(self) -> int:
        """Point estimate. Only meaningful when every criterion was assessed."""
        if not self.determinate:
            raise ValueError(
                "score is undefined while criteria are indeterminate; "
                "use score_min/score_max or decide()"
            )
        return self.score_min

    @property
    def decidable(self) -> bool:
        """True when unresolved criteria cannot change the threshold verdict."""
        return (self.score_min >= MELANOMA_SUSPICION_THRESHOLD) or (
            self.score_max < MELANOMA_SUSPICION_THRESHOLD
        )

    @property
    def meets_threshold(self) -> bool:
        if not self.decidable:
            raise ValueError("verdict is undecidable; check .decidable first")
        return self.score_min >= MELANOMA_SUSPICION_THRESHOLD


def score_criteria(values: Mapping[str, str]) -> SevenPointScore:
    """Aggregate criterion judgements into a score, tolerating INDETERMINATE."""
    normalised = validate_values(values)

    per_criterion: List[CriterionScore] = []
    indeterminate: List[str] = []
    lo = 0
    hi = 0

    for name in CRITERION_NAMES:
        spec = CRITERIA_BY_NAME[name]
        value = normalised[name]
        if value == INDETERMINATE:
            indeterminate.append(name)
            hi += spec.points          # worst case: it scores
            per_criterion.append(
                CriterionScore(name, value, points=0, max_points=spec.points, determinate=False)
            )
            continue
        earned = spec.score_for(value)
        lo += earned
        hi += earned
        per_criterion.append(
            CriterionScore(name, value, points=earned, max_points=spec.points, determinate=True)
        )

    return SevenPointScore(
        per_criterion=tuple(per_criterion),
        score_min=lo,
        score_max=hi,
        indeterminate=tuple(indeterminate),
    )


def recommend_management(scored: SevenPointScore) -> Tuple[str, str]:
    """Map a score to a management step on the Derm7pt ladder.

    Returns ``(management, reason)``.  This is the checklist's own implication only -- binding it
    to an authoritative guideline recommendation happens downstream, against the hash-pinned
    catalog, so that the cited text and the derived action are separable.
    """
    if not scored.decidable:
        return (
            "clinical follow up",
            f"undecidable: score in [{scored.score_min},{scored.score_max}] spans the "
            f"threshold of {MELANOMA_SUSPICION_THRESHOLD}; unresolved "
            f"{list(scored.indeterminate)}",
        )
    if scored.meets_threshold:
        return (
            "excision",
            f"score {scored.score_min} >= {MELANOMA_SUSPICION_THRESHOLD}",
        )
    return (
        "clinical follow up",
        f"score {scored.score_max} < {MELANOMA_SUSPICION_THRESHOLD}",
    )


def leave_one_criterion_out(values: Mapping[str, str]) -> Dict[str, Dict[str, object]]:
    """Influence of each criterion, computed without any regeneration.

    Marks each criterion INDETERMINATE in turn and reports whether the threshold verdict
    survives.  This is the causal-influence measurement ProvenAI (MG140) says citation audits do
    not provide, and here it costs nothing because aggregation is deterministic.
    """
    base = score_criteria(values)
    out: Dict[str, Dict[str, object]] = {}

    for name in CRITERION_NAMES:
        perturbed = dict(values)
        perturbed[name] = INDETERMINATE
        alt = score_criteria(perturbed)
        # An undecidable baseline has no verdict to change, and removal can only widen the
        # range, so every perturbation would spuriously count as a flip without this guard.
        if not base.decidable:
            changed = False
        else:
            changed = (not alt.decidable) or (alt.meets_threshold != base.meets_threshold)
        out[name] = {
            "removed": name,
            "score_range": (alt.score_min, alt.score_max),
            "still_decidable": alt.decidable,
            "verdict_changed": bool(changed),
            "points_contributed": next(
                c.points for c in base.per_criterion if c.name == name
            ),
        }
    return out


def verdict_hinges(
    criteria: Mapping[str, str],
    margins: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, object]]:
    """Which single criteria, if read differently, would flip the melanoma-threshold verdict.

    Wraps :func:`leave_one_criterion_out`.  Only meaningful for a DECIDABLE base score -- an
    already-undecidable case has no settled verdict for a criterion to hinge on, so this returns
    ``[]`` for it (the resolution protocol, not this function, is the right surface for that
    case; see ``evidence.render``).  An empty result for a decidable base is a different, equally
    real answer: the verdict is robust to every single criterion, which is worth stating rather
    than leaving silent.

    Each entry carries ``points_contributed`` (0 for a criterion that currently scores nothing)
    and ``flipped_verdict_direction``:

    - ``"erodes"`` -- the criterion currently scores (``points_contributed > 0``); losing it
      erodes the score out from under a verdict that currently meets the threshold.
    - ``"elevates"`` -- the criterion currently scores nothing; if it turned out to score, it
      would elevate a currently-below-threshold score up to the threshold.

    These two directions never mix within one case: when the base meets the threshold, only a
    currently-scoring criterion's removal can widen the range across it from above; when the base
    is below the threshold, only a currently-non-scoring criterion's removal can widen it across
    from below (removing a non-contributor changes nothing about a "meets" verdict, and removing
    a contributor never raises the ceiling of a "does not meet" verdict beyond what it already
    was). Returned in canonical criterion order.

    ``margins`` (optional): per-criterion model uncertainty (top-minus-second head posterior). When
    supplied, each hinge carries its ``margin`` and the list is sorted by **ascending** margin --
    the criterion the reader is *least* certain about comes first, because that is where a second
    look is most likely to actually change the reading. A hinge whose criterion has no supplied
    margin sorts last (treated as maximally certain). The sort is stable, so canonical criterion
    order breaks equal-margin ties. When ``margins`` is ``None`` (the default) the result is
    returned in canonical order with no ``margin`` key -- byte-for-byte the pre-margin behaviour,
    which every existing caller and test relies on.
    """
    base = score_criteria(criteria)
    if not base.decidable:
        return []

    loco = leave_one_criterion_out(criteria)
    normalised = validate_values(criteria)
    hinges: List[Dict[str, object]] = []
    for name in CRITERION_NAMES:
        entry = loco[name]
        if not entry["verdict_changed"]:
            continue
        points = int(entry["points_contributed"])
        hinge: Dict[str, object] = {
            "name": name,
            "value": normalised[name],
            "points_contributed": points,
            "flipped_verdict_direction": "erodes" if points > 0 else "elevates",
        }
        if margins is not None and name in margins:
            hinge["margin"] = float(margins[name])
        hinges.append(hinge)

    if margins is not None:
        # Ascending model margin: most-uncertain hinge first. list.sort is stable, so criteria
        # with no supplied margin (inf) and equal-margin ties both keep canonical order.
        hinges.sort(key=lambda h: h.get("margin", float("inf")))
    return hinges


def max_possible_score() -> int:
    return MAX_SCORE
