"""Deterministic 7-point aggregation, including under partial information."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from dermpact.criteria.schema import (
    CRITERIA_BY_NAME,
    CRITERION_NAMES,
    INDETERMINATE,
    MAX_SCORE,
    MELANOMA_SUSPICION_THRESHOLD,
    validate_values,
)

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
    """Map a score to a management step on the Derm7pt ladder."""
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
    """Influence of each criterion, computed without any regeneration."""
    base = score_criteria(values)
    out: Dict[str, Dict[str, object]] = {}

    for name in CRITERION_NAMES:
        perturbed = dict(values)
        perturbed[name] = INDETERMINATE
        alt = score_criteria(perturbed)
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
    """Which single criteria, if read differently, would flip the melanoma-threshold verdict."""
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
        hinges.sort(key=lambda h: h.get("margin", float("inf")))
    return hinges


def max_possible_score() -> int:
    return MAX_SCORE
