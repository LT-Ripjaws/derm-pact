"""Coverage-adjusted utility: what abstention is worth, as a function of what a
wrong answer costs.

``coverage_adjusted_utility`` is named in three registered places -- the v5 supporting
endpoints, the v5 required-tables list, and the preregistration's supporting outputs --
and **defined in none of them**.  It has to be defined before the sealed run, and
defining it after the development numbers are visible is exactly the situation in which
a convenient definition writes itself.

The defence is to refuse to pick a point.  A system that abstains can only be scored
against an explicit price for being wrong, and nobody in this study is entitled to set
that price: it is a clinical judgement that varies by lesion, by setting and by who
carries the liability.  So this module reports utility as a **surface** over that price,
plus the **break-even point** at which one arm overtakes another.  A reader supplies
their own cost ratio and reads off the answer.

    U(lambda, mu) = ( n_correct - lambda * n_wrong - mu * n_abstained ) / n_cases

* ``lambda`` -- cost of an answered-but-wrong case, relative to a correct one.
* ``mu``     -- cost of an abstention, relative to a correct one: the deferral overhead
  of sending the case to a human.

Both are costs *relative to the value of one correct answer*, which is fixed at 1.0 so
the scale means something.  ``lambda = 0`` reduces to plain coverage-weighted accuracy
and is included precisely because it is the setting least favourable to abstention.

What this is not: a clinical utility model.  It prices decisions, not outcomes, and it
knows nothing about which errors are dangerous -- a missed melanoma and a mislabelled
benign keratosis both cost ``lambda``.  The actionable/malignant analysis is where
asymmetric harm is handled.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Reported sweep.  Deliberately spans the region where abstention LOSES (lambda < 1,
#: a wrong answer costing no more than a right one is worth) through the region where
#: it obviously wins, so the table cannot be read as advocacy.
DEFAULT_WRONG_COSTS: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)

#: Abstention overhead: free, light human review, heavy review.
DEFAULT_ABSTENTION_COSTS: Sequence[float] = (0.0, 0.1, 0.25, 0.5)


class CoverageUtilityError(ValueError):
    """Raised when an arm's case accounting does not add up."""


def arm_counts(
    *, n_cases: int, coverage: float, answered_accuracy: Optional[float],
) -> Dict[str, int]:
    """Turn coverage + answered accuracy into correct / wrong / abstained counts."""
    if n_cases <= 0:
        raise CoverageUtilityError("n_cases must be positive")
    if not 0.0 <= coverage <= 1.0:
        raise CoverageUtilityError(f"coverage {coverage} outside [0, 1]")
    answered = int(round(coverage * n_cases))
    if answered and answered_accuracy is None:
        raise CoverageUtilityError("an arm that answers must have an answered accuracy")
    correct = int(round((answered_accuracy or 0.0) * answered))
    return {"answered": answered, "correct": correct,
            "wrong": answered - correct, "abstained": n_cases - answered}


def utility(counts: Mapping[str, int], *, n_cases: int,
            wrong_cost: float, abstention_cost: float) -> float:
    """U = (correct - lambda*wrong - mu*abstained) / n_cases.

    Deliberately unrounded: this feeds the break-even algebra, and rounding inside a
    quantity you then solve equations with is how small inconsistencies get in.
    Presentation rounds; the arithmetic does not.
    """
    return ((counts["correct"] - wrong_cost * counts["wrong"]
             - abstention_cost * counts["abstained"]) / n_cases)


def break_even_wrong_cost(
    counts_a: Mapping[str, int], counts_b: Mapping[str, int], *, abstention_cost: float,
) -> Optional[float]:
    """The lambda at which arm A's utility overtakes arm B's.

    From ``U_a > U_b``::

        lambda * (wrong_b - wrong_a) > (correct_b - correct_a)
                                       + mu * (abstained_a - abstained_b)

    so the crossing is at::

        lambda* = [ (correct_b - correct_a) + mu*(abstained_a - abstained_b) ]
                  / (wrong_b - wrong_a)

    The ``abstained_a - abstained_b`` ordering matters and is easy to invert: the arm
    that abstains MORE carries the deferral cost, so charging for abstention must make
    its case harder, not easier. Inverting it made a higher deferral overhead look like
    an argument *for* abstaining.

    ``None`` when the arms make the same number of errors -- the lines are parallel and
    never cross, reported as absent rather than as an infinity.
    """
    denominator = counts_b["wrong"] - counts_a["wrong"]
    if denominator == 0:
        return None
    numerator = ((counts_b["correct"] - counts_a["correct"])
                 + abstention_cost * (counts_a["abstained"] - counts_b["abstained"]))
    return round(numerator / denominator, 4)


def build_utility_table(
    arms: Mapping[str, Mapping[str, Any]], *, n_cases: int,
    reference_arm: str,
    wrong_costs: Sequence[float] = DEFAULT_WRONG_COSTS,
    abstention_costs: Sequence[float] = DEFAULT_ABSTENTION_COSTS,
    tokens_per_answered_case: Optional[Mapping[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    """The registered coverage-adjusted-utility table, as a surface plus break-evens.

    ``arms`` maps arm id -> {"coverage": float, "answered_accuracy": float | None}.
    ``reference_arm`` is the comparator every break-even is computed against; it is the
    registered no-contract comparator (B1), not whichever arm makes the table look best.
    """
    if reference_arm not in arms:
        raise CoverageUtilityError(f"reference arm {reference_arm!r} not present")
    counts = {
        arm: arm_counts(n_cases=n_cases, coverage=float(spec["coverage"]),
                        answered_accuracy=spec.get("answered_accuracy"))
        for arm, spec in arms.items()
    }
    surface: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm, count in counts.items():
        surface[arm] = {
            f"mu={mu}": {
                f"lambda={lam}": round(utility(count, n_cases=n_cases,
                                               wrong_cost=lam, abstention_cost=mu), 4)
                for lam in wrong_costs
            }
            for mu in abstention_costs
        }
    break_evens = {
        arm: {
            f"mu={mu}": break_even_wrong_cost(
                counts[arm], counts[reference_arm], abstention_cost=mu)
            for mu in abstention_costs
        }
        for arm in counts if arm != reference_arm
    }
    per_token = None
    if tokens_per_answered_case:
        per_token = {}
        for arm, count in counts.items():
            tokens = tokens_per_answered_case.get(arm)
            if not tokens or not count["answered"]:
                per_token[arm] = None
                continue
            spend = tokens * count["answered"]
            per_token[arm] = {
                f"lambda={lam}": round(
                    utility(count, n_cases=n_cases, wrong_cost=lam,
                            abstention_cost=0.0) * n_cases / (spend / 1000.0), 4)
                for lam in wrong_costs
            }
    return {
        "endpoint": "coverage_adjusted_utility",
        "definition": "U(lambda, mu) = (correct - lambda*wrong - mu*abstained) / n_cases",
        "parameters": {
            "lambda": "cost of an answered-but-wrong case, relative to one correct answer",
            "mu": "cost of an abstention (human deferral overhead), relative to one correct answer",
        },
        "why_a_surface_not_a_number": (
            "The price of being wrong is a clinical judgement that varies by lesion, "
            "setting and liability, and no one in this study is entitled to fix it. "
            "Picking a single lambda after seeing the numbers would be choosing the "
            "answer. The sweep spans lambda=0 -- where abstention is worth nothing and "
            "MedGuard is at its worst -- through lambda=10."
        ),
        "reference_arm": reference_arm,
        "n_cases": n_cases,
        "counts": counts,
        "utility_surface": surface,
        "break_even_wrong_cost_vs_reference": break_evens,
        "break_even_reading": (
            "The lambda at which this arm's utility overtakes the reference. Below it "
            "the reference is preferable; above it this arm is. A null means the two "
            "arms make the same number of errors, so the lines never cross."
        ),
        "utility_per_1k_tokens": per_token,
        "not_a_clinical_utility_model": (
            "This prices decisions, not outcomes. It treats every error alike: a missed "
            "melanoma and a mislabelled keratosis both cost lambda. Asymmetric harm is "
            "handled by the actionable/malignant analysis, not here."
        ),
    }
