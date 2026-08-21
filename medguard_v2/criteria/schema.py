"""The 7-point checklist: criterion vocabulary and point weights.

The weights here are not taken on faith from the literature -- they were solved against the
Derm7pt gold ``seven_point_score`` column and reproduce it on **1011/1011 cases**.  The one
genuinely ambiguous element, which values of ``vascular_structures`` count as an *atypical
vascular pattern*, resolved to exactly ``{dotted, linear irregular}``.  See
``medguard_v2/tests/test_criteria_scoring.py``, which re-derives the match and fails if a future
edit breaks it.

Structure of the classic (Argenziano) checklist:

* three **major** criteria worth 2 points each -- atypical pigment network, blue-whitish veil,
  atypical vascular pattern;
* four **minor** criteria worth 1 point each -- irregular streaks, irregular pigmentation,
  irregular dots/globules, regression structures.

A total of **3 or more** meets the threshold for melanoma suspicion.

Why the host owns this
----------------------
Aggregation is arithmetic over a published rule.  A language model must never perform it: the
agent's job is to decide *whether a criterion is present*, which is a perceptual judgement, and
the host's job is to combine those judgements, which is not.  Keeping the split here is what
makes the final score auditable and what lets a criterion be ablated without regeneration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Tuple

#: Emitted by the assessor when a criterion cannot be judged from the available images.
#: It is a first-class value, not a missing one: a case with indeterminate criteria yields a
#: score *range* rather than a point estimate, which is what criterion-level abstention acts on.
INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class Criterion:
    """One checklist row: its vocabulary, and which values earn its points."""

    name: str
    category: str                      # "major" (2 points) or "minor" (1 point)
    points: int
    scoring_values: FrozenSet[str]     # values that earn ``points``
    vocabulary: FrozenSet[str]         # every value the expert annotation may take
    description: str

    def score_for(self, value: str) -> int:
        if value == INDETERMINATE:
            raise ValueError(
                f"{self.name}: {INDETERMINATE!r} has no point value; use score_range()"
            )
        if value not in self.vocabulary:
            raise ValueError(f"{self.name}: unknown value {value!r}")
        return self.points if value in self.scoring_values else 0


#: Canonical order.  Major criteria first, then minor, matching how the checklist is taught.
CRITERION_SPECS: Tuple[Criterion, ...] = (
    Criterion(
        name="pigment_network",
        category="major",
        points=2,
        scoring_values=frozenset({"atypical"}),
        vocabulary=frozenset({"absent", "typical", "atypical"}),
        description="Atypical pigment network: irregular meshes and thickened lines.",
    ),
    Criterion(
        name="blue_whitish_veil",
        category="major",
        points=2,
        scoring_values=frozenset({"present"}),
        vocabulary=frozenset({"absent", "present"}),
        description="Blue-whitish veil: confluent blue pigment with an overlying white haze.",
    ),
    Criterion(
        name="vascular_structures",
        category="major",
        points=2,
        # Solved against gold, not assumed: only these two count as atypical.
        scoring_values=frozenset({"dotted", "linear irregular"}),
        vocabulary=frozenset(
            {
                "absent",
                "arborizing",
                "comma",
                "hairpin",
                "within regression",
                "wreath",
                "dotted",
                "linear irregular",
            }
        ),
        description="Atypical vascular pattern: dotted or linear-irregular vessels.",
    ),
    Criterion(
        name="streaks",
        category="minor",
        points=1,
        scoring_values=frozenset({"irregular"}),
        vocabulary=frozenset({"absent", "regular", "irregular"}),
        description="Irregular streaks: unevenly distributed radial lines or pseudopods.",
    ),
    Criterion(
        name="pigmentation",
        category="minor",
        points=1,
        scoring_values=frozenset({"diffuse irregular", "localized irregular"}),
        vocabulary=frozenset(
            {
                "absent",
                "diffuse regular",
                "localized regular",
                "diffuse irregular",
                "localized irregular",
            }
        ),
        description="Irregular pigmentation: asymmetric blotches of pigment.",
    ),
    Criterion(
        name="dots_and_globules",
        category="minor",
        points=1,
        scoring_values=frozenset({"irregular"}),
        vocabulary=frozenset({"absent", "regular", "irregular"}),
        description="Irregular dots/globules: varied size and uneven distribution.",
    ),
    Criterion(
        name="regression_structures",
        category="minor",
        points=1,
        # Any regression pattern counts; only true absence scores zero.
        scoring_values=frozenset({"blue areas", "white areas", "combinations"}),
        vocabulary=frozenset({"absent", "blue areas", "white areas", "combinations"}),
        description="Regression structures: white scar-like areas, blue peppering, or both.",
    ),
)

CRITERION_NAMES: Tuple[str, ...] = tuple(spec.name for spec in CRITERION_SPECS)

CRITERIA_BY_NAME: Mapping[str, Criterion] = {spec.name: spec for spec in CRITERION_SPECS}

#: Classic threshold.  A revised checklist using a threshold of 3 over a reduced criterion set
#: also exists; this project uses the classic form because it is what Derm7pt annotates.
MELANOMA_SUSPICION_THRESHOLD = 3

#: Maximum attainable score: 3 major x 2 + 4 minor x 1.
MAX_SCORE = 10


def validate_values(values: Mapping[str, str]) -> Dict[str, str]:
    """Check that a criterion mapping is complete and in-vocabulary.

    Returns a normalised dict in canonical order.  Raises on unknown criteria, missing
    criteria, or out-of-vocabulary values -- the assessor must not be able to invent a value
    that silently scores zero.
    """
    unknown = set(values) - set(CRITERION_NAMES)
    if unknown:
        raise ValueError(f"unknown criteria: {sorted(unknown)}")
    missing = set(CRITERION_NAMES) - set(values)
    if missing:
        raise ValueError(f"missing criteria: {sorted(missing)}")

    normalised: Dict[str, str] = {}
    for name in CRITERION_NAMES:
        value = values[name]
        if value != INDETERMINATE and value not in CRITERIA_BY_NAME[name].vocabulary:
            raise ValueError(f"{name}: unknown value {value!r}")
        normalised[name] = value
    return normalised
