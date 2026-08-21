"""Focused tests for the B6 bounded retrieve-evaluate-refine loop.

The 160-lesion development run declares sufficiency on the first attempt for
every case, so its refinement and abstention branches are exercised here rather
than by development data.
"""
from __future__ import annotations

import pytest

from medguard.eval.h2h_b6_adapter import (
    MAX_ATTEMPTS,
    MAX_REFINEMENTS,
    METRIC_THRESHOLD,
    SUFFICIENCY_THRESHOLD,
    WEIGHT_COVERAGE,
    WEIGHT_DENSITY,
    WEIGHT_SEMANTIC_RELEVANCE,
    B6ConfigurationError,
    LoopDecision,
    OptimizerRule,
    RefinementBounds,
    SearchStrategy,
    assert_disjoint_from_medguard_failures,
    normalized_coverage,
    normalized_density,
    optimize_strategy,
    run_bounded_refinement,
    score_sufficiency,
)
from medguard.eval.h2h_evidence_graph import GraphRetrieval


def _retrieval(*, candidates: int, mean_attributes: float = 4.625) -> GraphRetrieval:
    ids = tuple(f"evidence:{index}" for index in range(candidates))
    return GraphRetrieval(
        seed_concept_ids=("concept:melanoma",),
        depth=1,
        node_limit=max(candidates, 1),
        candidate_evidence_ids=ids,
        returned_evidence_ids=ids[:5],
        paths=(),
        subgraph_node_ids=ids,
        subgraph_edge_ids=(),
        reached_evidence_count=candidates,
        frontier_sizes=(candidates,),
        mean_populated_attributes=mean_attributes,
    )


def test_weights_match_the_source_paper_and_sum_to_one():
    assert WEIGHT_SEMANTIC_RELEVANCE == 0.4
    assert WEIGHT_COVERAGE == 0.3
    assert WEIGHT_DENSITY == 0.3
    assert WEIGHT_SEMANTIC_RELEVANCE + WEIGHT_COVERAGE + WEIGHT_DENSITY == 1.0
    assert MAX_REFINEMENTS == 3
    assert MAX_ATTEMPTS == 4


def test_scalar_sufficiency_is_the_declared_weighted_sum():
    score = score_sufficiency(semantic_relevance=1.0, coverage=0.5, density=0.0)
    assert score.overall == pytest.approx(0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 0.0)
    assert score.sufficient is True
    low = score_sufficiency(semantic_relevance=0.1, coverage=0.1, density=0.1)
    assert low.overall == pytest.approx(0.1)
    assert low.sufficient is False
    edge = score_sufficiency(semantic_relevance=0.5, coverage=0.5, density=0.5)
    assert edge.overall == pytest.approx(SUFFICIENCY_THRESHOLD)
    assert edge.sufficient is True  # the threshold is inclusive


def test_dimensions_are_clamped_and_normalized_against_declared_ceilings():
    assert normalized_coverage(_retrieval(candidates=5), candidate_ceiling=20) == 0.25
    assert normalized_coverage(_retrieval(candidates=40), candidate_ceiling=20) == 1.0
    assert normalized_density(
        _retrieval(candidates=1, mean_attributes=4.0), attribute_ceiling=8,
    ) == 0.5
    assert score_sufficiency(
        semantic_relevance=-1.0, coverage=2.0, density=0.5,
    ).semantic_relevance == 0.0
    with pytest.raises(B6ConfigurationError):
        normalized_coverage(_retrieval(candidates=5), candidate_ceiling=0)


def test_b6_never_reuses_medguard_failure_type_labels():
    from medguard.evidence.verification import FailureType

    assert_disjoint_from_medguard_failures()
    medguard = {value.value for value in FailureType}
    mine = {value.value for value in LoopDecision} | {
        value.value for value in OptimizerRule
    }
    assert not (medguard & mine)
    # The evaluator's own vocabulary is scalar and process-shaped, not failure-typed.
    assert {value.value for value in LoopDecision} == {
        "USE_CURRENT", "RETRY_SEARCH", "ABSTAIN_INSUFFICIENT",
    }


def test_optimizer_follows_the_source_paper_priority_order():
    bounds = RefinementBounds()
    start = SearchStrategy(depth=1, node_limit=5)
    # Low coverage with depth headroom increases depth first.
    strategy, rule = optimize_strategy(
        start, score_sufficiency(semantic_relevance=0.9, coverage=0.1, density=0.9),
        bounds,
    )
    assert rule is OptimizerRule.INCREASE_DEPTH_ON_LOW_COVERAGE
    assert strategy == SearchStrategy(depth=3, node_limit=5)
    # With coverage satisfied, low relevance raises the node limit.
    strategy, rule = optimize_strategy(
        start, score_sufficiency(semantic_relevance=0.1, coverage=0.9, density=0.9),
        bounds,
    )
    assert rule is OptimizerRule.INCREASE_NODE_LIMIT_ON_LOW_RELEVANCE
    assert strategy == SearchStrategy(depth=1, node_limit=10)
    # With coverage and relevance satisfied, low density raises the node limit.
    strategy, rule = optimize_strategy(
        start, score_sufficiency(semantic_relevance=0.9, coverage=0.9, density=0.1),
        bounds,
    )
    assert rule is OptimizerRule.INCREASE_NODE_LIMIT_ON_LOW_DENSITY
    assert strategy == SearchStrategy(depth=1, node_limit=10)


def test_optimizer_respects_its_declared_ceilings():
    bounds = RefinementBounds()
    at_max = SearchStrategy(depth=bounds.max_depth, node_limit=bounds.max_node_limit)
    strategy, rule = optimize_strategy(
        at_max, score_sufficiency(semantic_relevance=0.0, coverage=0.0, density=0.0),
        bounds,
    )
    assert rule is OptimizerRule.NO_APPLICABLE_REFINEMENT
    assert strategy == at_max
    assert METRIC_THRESHOLD == 0.5


def test_loop_uses_current_without_refining_when_context_is_sufficient():
    calls: list[SearchStrategy] = []

    def retrieve(strategy):
        calls.append(strategy)
        return _retrieval(candidates=20)

    result = run_bounded_refinement(
        retrieve=retrieve, semantic_relevance=lambda _r: 0.9,
        candidate_ceiling=20, attribute_ceiling=8,
    )
    assert result.final_decision is LoopDecision.USE_CURRENT
    assert result.attempts_used == 1
    assert result.refinements_used == 0
    assert len(calls) == 1
    assert result.abstained is False


def test_loop_refines_then_succeeds_and_records_each_attempt():
    seen: list[SearchStrategy] = []

    def retrieve(strategy):
        seen.append(strategy)
        # Coverage only becomes adequate once the node limit has grown.
        return _retrieval(candidates=strategy.node_limit)

    def relevance(retrieval):
        return 0.3 if len(retrieval.candidate_evidence_ids) < 10 else 0.95

    result = run_bounded_refinement(
        retrieve=retrieve, semantic_relevance=relevance,
        candidate_ceiling=20, attribute_ceiling=8,
    )
    assert result.final_decision is LoopDecision.USE_CURRENT
    assert 1 < result.attempts_used <= MAX_ATTEMPTS
    assert result.refinements_used == result.attempts_used - 1
    assert [attempt.decision for attempt in result.attempts[:-1]] == [
        LoopDecision.RETRY_SEARCH
    ] * (result.attempts_used - 1)
    assert result.attempts[-1].decision is LoopDecision.USE_CURRENT
    # The optimizer changed the strategy on every refinement.
    assert len({(item.depth, item.node_limit) for item in seen}) == len(seen)


def test_depth_rule_cannot_raise_coverage_and_therefore_spends_refinements():
    """Document a real interaction between the paper's rule and this topology.

    The source paper's first optimizer rule raises search depth when coverage is
    low, assuming a deeper search retrieves more nodes.  Here the node limit caps
    the candidate set and the reachable pool already exceeds it, so extra depth
    cannot change coverage: the rule fires, consumes refinements, and leaves the
    score untouched until the depth ceiling hands control to the node-limit
    rules.  This is faithful to the source rule, not a defect to optimize away.
    """
    seen: list[SearchStrategy] = []

    def retrieve(strategy):
        seen.append(strategy)
        # A saturated pool: candidate count is bounded by node_limit, never depth.
        return _retrieval(candidates=min(strategy.node_limit, 50))

    result = run_bounded_refinement(
        retrieve=retrieve, semantic_relevance=lambda _r: 0.3,
        candidate_ceiling=20, attribute_ceiling=8,
    )
    depths = [item.depth for item in seen]
    node_limits = [item.node_limit for item in seen]
    assert depths == [1, 3, 5, 5]
    assert node_limits == [5, 5, 5, 10]
    # Two of the three refinements went to depth and moved coverage not at all.
    coverages = [attempt.score.coverage for attempt in result.attempts[:3]]
    assert coverages == [0.25, 0.25, 0.25]
    assert result.final_decision is LoopDecision.ABSTAIN_INSUFFICIENT
    assert result.refinements_used == MAX_REFINEMENTS


def test_unresolved_insufficiency_abstains_and_never_exceeds_its_budget():
    calls: list[SearchStrategy] = []

    def retrieve(strategy):
        calls.append(strategy)
        return _retrieval(candidates=1, mean_attributes=0.0)

    result = run_bounded_refinement(
        retrieve=retrieve, semantic_relevance=lambda _r: 0.0,
        candidate_ceiling=20, attribute_ceiling=8,
    )
    assert result.final_decision is LoopDecision.ABSTAIN_INSUFFICIENT
    assert result.abstained is True
    assert result.attempts_used == MAX_ATTEMPTS
    assert result.refinements_used == MAX_REFINEMENTS
    assert len(calls) == MAX_ATTEMPTS
    assert result.attempts[-1].decision is LoopDecision.ABSTAIN_INSUFFICIENT


def test_loop_stops_early_rather_than_repeat_an_exhausted_strategy():
    bounds = RefinementBounds(
        start_depth=5, start_node_limit=20, max_depth=5, max_node_limit=20,
    )
    calls: list[SearchStrategy] = []

    def retrieve(strategy):
        calls.append(strategy)
        return _retrieval(candidates=1, mean_attributes=0.0)

    result = run_bounded_refinement(
        retrieve=retrieve, semantic_relevance=lambda _r: 0.0, bounds=bounds,
        candidate_ceiling=20, attribute_ceiling=8,
    )
    assert result.final_decision is LoopDecision.ABSTAIN_INSUFFICIENT
    assert len(calls) == 1
    assert result.refinements_used == 0
    assert result.attempts[-1].optimizer_rule is OptimizerRule.NO_APPLICABLE_REFINEMENT


def test_declared_bounds_are_validated():
    with pytest.raises(B6ConfigurationError):
        RefinementBounds(start_depth=9, max_depth=5)
    with pytest.raises(B6ConfigurationError):
        RefinementBounds(start_node_limit=50, max_node_limit=20)
    with pytest.raises(B6ConfigurationError):
        RefinementBounds(final_k=10, start_node_limit=5)
    with pytest.raises(B6ConfigurationError):
        SearchStrategy(depth=0, node_limit=5)


# --- D4: evidence-count sufficiency rule for the H2H run -------------------------
def test_d4_evidence_count_rule_refines_where_the_weighted_rule_is_vacuous():
    """The weighted rule declares 3 candidates sufficient at once; the count rule
    treats it as too few (< final_k) and reaches the paper's depth refinement."""
    # Old rule: high relevance + full density clears 0.5 on the first attempt even
    # with only 3 candidates -- the vacuous behavior the validate report documents.
    old = run_bounded_refinement(
        retrieve=lambda _s: _retrieval(candidates=3, mean_attributes=8.0),
        semantic_relevance=lambda _r: 0.9, candidate_ceiling=20, attribute_ceiling=8,
    )
    assert old.final_decision is LoopDecision.USE_CURRENT
    assert old.attempts_used == 1

    # New rule (min_evidence_count = final_k = 5): 3 candidates is insufficient, low
    # coverage triggers INCREASE_DEPTH, and depth 3 reaches the 5-candidate target.
    seen: list[SearchStrategy] = []

    def retrieve(strategy):
        seen.append(strategy)
        return _retrieval(candidates=3 if strategy.depth < 3 else 5, mean_attributes=8.0)

    new = run_bounded_refinement(
        retrieve=retrieve, semantic_relevance=lambda _r: 0.9,
        candidate_ceiling=20, attribute_ceiling=8, min_evidence_count=5,
    )
    assert new.final_decision is LoopDecision.USE_CURRENT
    assert new.attempts_used == 2
    assert new.refinements_used == 1
    assert new.attempts[0].decision is LoopDecision.RETRY_SEARCH
    assert new.attempts[0].optimizer_rule is OptimizerRule.INCREASE_DEPTH_ON_LOW_COVERAGE
    assert [s.depth for s in seen] == [1, 3]


def test_d4_evidence_count_rule_abstains_when_final_k_is_never_reached():
    result = run_bounded_refinement(
        retrieve=lambda _s: _retrieval(candidates=2, mean_attributes=8.0),
        semantic_relevance=lambda _r: 0.9, candidate_ceiling=20, attribute_ceiling=8,
        min_evidence_count=5,
    )
    assert result.final_decision is LoopDecision.ABSTAIN_INSUFFICIENT
    assert result.abstained is True


def test_d4_min_evidence_count_must_be_positive():
    with pytest.raises(B6ConfigurationError):
        run_bounded_refinement(
            retrieve=lambda _s: _retrieval(candidates=5), semantic_relevance=lambda _r: 0.9,
            min_evidence_count=0,
        )
