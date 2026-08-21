"""Bounded retrieve-evaluate-refine loop for the B6 comparator.

This is an Agentic-Graph-RAG-inspired *compatible adaptation* over the B5
evidence-lineage graph, never a reproduction.  It implements the source paper's
scalar sufficiency score

    Score_overall = w_sem * S_sem + w_cov * Norm(N_nodes) + w_den * Norm(D_avg)

with the paper's disclosed weights (0.4 / 0.3 / 0.3), its deterministic
rule-based strategy optimizer, and its maximum of three refinements.

Declared deviations from the source framework, all forced by this study's
boundaries and recorded on the B6 implementation card:

* The dual-model Text-to-Cypher stage and the three-votes-per-model
  self-consistency evaluator are omitted; this study authorizes no generation.
  The paper's own non-LLM vector-similarity failsafe is therefore promoted from
  tie-breaker to the primary sufficiency decision, which collapses the
  tau_high/tau_low adjudication band into one threshold.
* The optimizer's "switch the search strategy" branch is omitted: the
  concept<->evidence bipartite tier has no distinct DFS candidate set.  Only the
  paper's "increase the node limit" branch of that rule is taken.
* The paper's ``generate_llm_fallback`` terminal is replaced by ABSTAIN.  An
  ungrounded parametric answer is not permitted in this study.

This evaluator owns its scalar dimensions and must never reuse MedGuard's
failure-type vocabulary; ``assert_disjoint_from_medguard_failures`` enforces
that boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from medguard.eval.h2h_evidence_graph import GraphRetrieval


# Paper 104, Equation 7: "the weights, w_sem = 0.4, w_cov = 0.3, w_den = 0.3
# reflect the system's high emphasis on semantic accuracy."
WEIGHT_SEMANTIC_RELEVANCE = 0.4
WEIGHT_COVERAGE = 0.3
WEIGHT_DENSITY = 0.3
MAX_REFINEMENTS = 3
MAX_ATTEMPTS = MAX_REFINEMENTS + 1
# Paper 104 does not disclose tau_high/tau_low or its per-metric optimizer
# thresholds.  One neutral midpoint is pinned here as a declared study
# adaptation, fixed before any H2H output was viewed.
SUFFICIENCY_THRESHOLD = 0.5
METRIC_THRESHOLD = 0.5

# D4 resolution.  The weighted-overall >= 0.5 rule above is vacuous on this study's
# graph: every development lesion clears it on the first attempt, so the refinement
# and abstention branches never fire on real data (only in focused tests).  The
# H2H run therefore decides sufficiency on evidence *count* instead: the retrieved
# subgraph must yield at least ``final_k`` distinct evidence candidates -- the
# minimum needed to fill the returned answer set.  A committed-diagnosis concept
# whose directly linked corpus records number fewer than ``final_k`` fails on the
# first attempt and triggers the paper's own "increase depth on low coverage"
# refinement, which reaches co-linked concepts' evidence.  The target is
# structural (it equals ``final_k``), so it is not tuned on development outcomes;
# ``run_bounded_refinement`` takes it as ``min_evidence_count`` and the validate
# report keeps the old rule (``min_evidence_count=None``) so its bytes never move.
MIN_EVIDENCE_SUFFICIENCY_RULE = "candidate_evidence_count_ge_min_evidence_count"


class B6ConfigurationError(ValueError):
    """Raised when the bounded loop is configured outside its declared bounds."""


class LoopDecision(str, Enum):
    """Paper 104 Algorithm 1 decision labels plus this study's safe terminal."""

    USE_CURRENT = "USE_CURRENT"
    RETRY_SEARCH = "RETRY_SEARCH"
    ABSTAIN_INSUFFICIENT = "ABSTAIN_INSUFFICIENT"


class OptimizerRule(str, Enum):
    """Deterministic strategy-optimizer branches, in the paper's own priority."""

    INCREASE_DEPTH_ON_LOW_COVERAGE = "increase_depth_on_low_coverage"
    INCREASE_NODE_LIMIT_ON_LOW_RELEVANCE = "increase_node_limit_on_low_relevance"
    INCREASE_NODE_LIMIT_ON_LOW_DENSITY = "increase_node_limit_on_low_density"
    NO_APPLICABLE_REFINEMENT = "no_applicable_refinement"


def assert_disjoint_from_medguard_failures() -> None:
    """Fail closed if B6 ever borrows MedGuard's failure-typed vocabulary."""
    from medguard.evidence.verification import FailureType

    medguard = {value.value.strip().lower() for value in FailureType}
    mine = {value.value.strip().lower() for value in LoopDecision}
    mine |= {value.value.strip().lower() for value in OptimizerRule}
    mine |= {"semantic_relevance", "coverage", "density", "overall"}
    overlap = medguard & mine
    if overlap:
        raise B6ConfigurationError(
            f"B6 must not reuse MedGuard failure-type labels: {sorted(overlap)}"
        )


@dataclass(frozen=True)
class SearchStrategy:
    """The mutable parameters of one graph search, per the paper's optimizer."""

    depth: int
    node_limit: int

    def __post_init__(self) -> None:
        if self.depth < 1 or self.node_limit < 1:
            raise B6ConfigurationError("search depth and node limit must be positive")


@dataclass(frozen=True)
class RefinementBounds:
    """Declared, pre-registered bounds for the deterministic optimizer."""

    start_depth: int = 1
    start_node_limit: int = 5
    max_depth: int = 5
    max_node_limit: int = 20
    depth_step: int = 2
    node_limit_step: int = 5
    final_k: int = 5

    def __post_init__(self) -> None:
        if self.start_depth < 1 or self.start_depth > self.max_depth:
            raise B6ConfigurationError("require 1 <= start_depth <= max_depth")
        if self.start_node_limit < 1 or self.start_node_limit > self.max_node_limit:
            raise B6ConfigurationError("require 1 <= start_node_limit <= max_node_limit")
        if self.depth_step < 1 or self.node_limit_step < 1:
            raise B6ConfigurationError("optimizer steps must be positive")
        if self.final_k < 1 or self.final_k > self.start_node_limit:
            raise B6ConfigurationError("require final_k <= start_node_limit")

    def start(self) -> SearchStrategy:
        return SearchStrategy(depth=self.start_depth, node_limit=self.start_node_limit)


@dataclass(frozen=True)
class SufficiencyScore:
    """One evaluation of the paper's scalar sufficiency dimensions."""

    semantic_relevance: float
    coverage: float
    density: float
    overall: float
    sufficient: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_relevance": round(self.semantic_relevance, 9),
            "coverage": round(self.coverage, 9),
            "density": round(self.density, 9),
            "overall": round(self.overall, 9),
            "sufficient": self.sufficient,
            "weights": {
                "semantic_relevance": WEIGHT_SEMANTIC_RELEVANCE,
                "coverage": WEIGHT_COVERAGE,
                "density": WEIGHT_DENSITY,
            },
            "sufficiency_threshold": SUFFICIENCY_THRESHOLD,
        }


@dataclass(frozen=True)
class LoopAttempt:
    attempt: int
    strategy: SearchStrategy
    score: SufficiencyScore
    decision: LoopDecision
    optimizer_rule: OptimizerRule | None
    candidate_count: int
    returned_count: int
    subgraph_node_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "strategy": {
                "depth": self.strategy.depth,
                "node_limit": self.strategy.node_limit,
            },
            "sufficiency": self.score.to_dict(),
            "decision": self.decision.value,
            "optimizer_rule": (
                self.optimizer_rule.value if self.optimizer_rule is not None else None
            ),
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "subgraph_node_count": self.subgraph_node_count,
        }


@dataclass(frozen=True)
class B6Result:
    attempts: Tuple[LoopAttempt, ...]
    final_decision: LoopDecision
    retrieval: GraphRetrieval
    refinements_used: int

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def abstained(self) -> bool:
        return self.final_decision is LoopDecision.ABSTAIN_INSUFFICIENT


def _clamp_unit(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise B6ConfigurationError(f"{name} must be finite")
    return min(1.0, max(0.0, number))


def normalized_coverage(retrieval: GraphRetrieval, *, candidate_ceiling: int) -> float:
    """Norm(N_nodes): retrieved evidence-node count over the shared candidate pool."""
    if candidate_ceiling < 1:
        raise B6ConfigurationError("candidate_ceiling must be positive")
    return _clamp_unit(
        len(retrieval.candidate_evidence_ids) / candidate_ceiling, "coverage",
    )


def normalized_density(retrieval: GraphRetrieval, *, attribute_ceiling: int) -> float:
    """Norm(D_avg): mean populated attributes per returned node, normalized.

    This dimension is near-constant over the frozen corpus because every record
    carries the same attribute schema.  It is still computed and weighted
    faithfully rather than dropped.
    """
    if attribute_ceiling < 1:
        raise B6ConfigurationError("attribute_ceiling must be positive")
    return _clamp_unit(
        retrieval.mean_populated_attributes / attribute_ceiling, "density",
    )


def score_sufficiency(
    *, semantic_relevance: float, coverage: float, density: float,
) -> SufficiencyScore:
    """Evaluate paper 104's Equation 7 with its disclosed weights."""
    relevance = _clamp_unit(semantic_relevance, "semantic_relevance")
    cover = _clamp_unit(coverage, "coverage")
    dense = _clamp_unit(density, "density")
    overall = (
        WEIGHT_SEMANTIC_RELEVANCE * relevance
        + WEIGHT_COVERAGE * cover
        + WEIGHT_DENSITY * dense
    )
    return SufficiencyScore(
        semantic_relevance=relevance,
        coverage=cover,
        density=dense,
        overall=overall,
        sufficient=overall >= SUFFICIENCY_THRESHOLD,
    )


def optimize_strategy(
    strategy: SearchStrategy, score: SufficiencyScore, bounds: RefinementBounds,
) -> Tuple[SearchStrategy, OptimizerRule]:
    """Paper 104's rule-based decision tree, in its stated priority order.

    "If coverage is below its threshold and the search depth has not reached its
    maximum limit, the search depth is increased.  If semantic relevance is below
    its threshold, the system switches the search strategy or increases the node
    limit.  If information density is below its threshold, the node limit is
    increased."
    """
    if score.coverage < METRIC_THRESHOLD and strategy.depth < bounds.max_depth:
        return (
            SearchStrategy(
                depth=min(strategy.depth + bounds.depth_step, bounds.max_depth),
                node_limit=strategy.node_limit,
            ),
            OptimizerRule.INCREASE_DEPTH_ON_LOW_COVERAGE,
        )
    can_grow = strategy.node_limit < bounds.max_node_limit
    if score.semantic_relevance < METRIC_THRESHOLD and can_grow:
        return (
            SearchStrategy(
                depth=strategy.depth,
                node_limit=min(
                    strategy.node_limit + bounds.node_limit_step, bounds.max_node_limit,
                ),
            ),
            OptimizerRule.INCREASE_NODE_LIMIT_ON_LOW_RELEVANCE,
        )
    if score.density < METRIC_THRESHOLD and can_grow:
        return (
            SearchStrategy(
                depth=strategy.depth,
                node_limit=min(
                    strategy.node_limit + bounds.node_limit_step, bounds.max_node_limit,
                ),
            ),
            OptimizerRule.INCREASE_NODE_LIMIT_ON_LOW_DENSITY,
        )
    return strategy, OptimizerRule.NO_APPLICABLE_REFINEMENT


def run_bounded_refinement(
    *,
    retrieve: Callable[[SearchStrategy], GraphRetrieval],
    semantic_relevance: Callable[[GraphRetrieval], float],
    bounds: RefinementBounds = RefinementBounds(),
    candidate_ceiling: int = 20,
    attribute_ceiling: int = 8,
    min_evidence_count: int | None = None,
) -> B6Result:
    """Run the bounded retrieve-evaluate-refine loop over the B5 graph.

    The loop performs at most ``MAX_ATTEMPTS`` retrieve/evaluate cycles, i.e. one
    initial retrieval plus at most ``MAX_REFINEMENTS`` refinements.  Evidence that
    remains insufficient when the budget is exhausted abstains; it never falls
    back to an ungrounded parametric answer.

    ``min_evidence_count`` selects the sufficiency rule (see D4 above).  Left
    ``None``, sufficiency is the paper's weighted-overall threshold and the loop
    behaves exactly as it did for the frozen validate report.  Set to an integer
    (the H2H run passes ``bounds.final_k``), sufficiency instead requires at least
    that many distinct evidence candidates, which is what makes the refinement
    branch reachable on sparse-connectivity commitments.  The weighted score is
    still computed and recorded on every attempt as a reported diagnostic.
    """
    assert_disjoint_from_medguard_failures()
    if min_evidence_count is not None and min_evidence_count < 1:
        raise B6ConfigurationError("min_evidence_count must be positive when supplied")
    strategy = bounds.start()
    attempts: list[LoopAttempt] = []
    retrieval: GraphRetrieval | None = None
    refinements = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        retrieval = retrieve(strategy)
        score = score_sufficiency(
            semantic_relevance=semantic_relevance(retrieval),
            coverage=normalized_coverage(retrieval, candidate_ceiling=candidate_ceiling),
            density=normalized_density(retrieval, attribute_ceiling=attribute_ceiling),
        )
        is_sufficient = (
            score.sufficient if min_evidence_count is None
            else len(retrieval.candidate_evidence_ids) >= min_evidence_count
        )
        if is_sufficient:
            attempts.append(LoopAttempt(
                attempt=attempt, strategy=strategy, score=score,
                decision=LoopDecision.USE_CURRENT, optimizer_rule=None,
                candidate_count=len(retrieval.candidate_evidence_ids),
                returned_count=len(retrieval.returned_evidence_ids),
                subgraph_node_count=retrieval.subgraph_node_count,
            ))
            return B6Result(tuple(attempts), LoopDecision.USE_CURRENT, retrieval, refinements)

        if attempt == MAX_ATTEMPTS:
            attempts.append(LoopAttempt(
                attempt=attempt, strategy=strategy, score=score,
                decision=LoopDecision.ABSTAIN_INSUFFICIENT, optimizer_rule=None,
                candidate_count=len(retrieval.candidate_evidence_ids),
                returned_count=len(retrieval.returned_evidence_ids),
                subgraph_node_count=retrieval.subgraph_node_count,
            ))
            break

        proposed, rule = optimize_strategy(strategy, score, bounds)
        if rule is OptimizerRule.NO_APPLICABLE_REFINEMENT or proposed == strategy:
            # The optimizer is exhausted: nothing further can change the context,
            # so stop rather than burn the remaining attempts on identical work.
            attempts.append(LoopAttempt(
                attempt=attempt, strategy=strategy, score=score,
                decision=LoopDecision.ABSTAIN_INSUFFICIENT,
                optimizer_rule=OptimizerRule.NO_APPLICABLE_REFINEMENT,
                candidate_count=len(retrieval.candidate_evidence_ids),
                returned_count=len(retrieval.returned_evidence_ids),
                subgraph_node_count=retrieval.subgraph_node_count,
            ))
            break
        attempts.append(LoopAttempt(
            attempt=attempt, strategy=strategy, score=score,
            decision=LoopDecision.RETRY_SEARCH, optimizer_rule=rule,
            candidate_count=len(retrieval.candidate_evidence_ids),
            returned_count=len(retrieval.returned_evidence_ids),
            subgraph_node_count=retrieval.subgraph_node_count,
        ))
        strategy = proposed
        refinements += 1

    if retrieval is None:  # defensive; the loop always retrieves at least once
        raise B6ConfigurationError("bounded refinement produced no retrieval")
    if len(attempts) > MAX_ATTEMPTS or refinements > MAX_REFINEMENTS:
        raise B6ConfigurationError("bounded refinement exceeded its declared budget")
    return B6Result(
        tuple(attempts), LoopDecision.ABSTAIN_INSUFFICIENT, retrieval, refinements,
    )


__all__ = [
    "MAX_ATTEMPTS",
    "MAX_REFINEMENTS",
    "METRIC_THRESHOLD",
    "MIN_EVIDENCE_SUFFICIENCY_RULE",
    "SUFFICIENCY_THRESHOLD",
    "WEIGHT_COVERAGE",
    "WEIGHT_DENSITY",
    "WEIGHT_SEMANTIC_RELEVANCE",
    "B6ConfigurationError",
    "B6Result",
    "LoopAttempt",
    "LoopDecision",
    "OptimizerRule",
    "RefinementBounds",
    "SearchStrategy",
    "SufficiencyScore",
    "assert_disjoint_from_medguard_failures",
    "normalized_coverage",
    "normalized_density",
    "optimize_strategy",
    "run_bounded_refinement",
    "score_sufficiency",
]
