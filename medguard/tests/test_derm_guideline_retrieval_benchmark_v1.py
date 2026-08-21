from medguard.eval.derm_guideline_retrieval_benchmark_v1 import (
    run_benchmark,
    score_query,
    verify_benchmark,
)


def test_score_query_computes_binary_relevance_metrics():
    metrics = score_query(["a", "b"], ["x", "a", "b"], 3)
    assert metrics["precision@3"] == 2 / 3
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 0.5
    assert 0.0 < metrics["ndcg@3"] < 1.0


def test_subset_benchmark_is_receipt_bound_and_claim_guarded():
    result = run_benchmark()
    receipt = verify_benchmark()
    assert result["query_count"] == 8
    assert result["covered_condition_question_cells"] == 8
    assert result["total_scoped_condition_question_cells"] == 29
    assert "not comprehensive" in result["claim_guard"]
    assert receipt["execution_guards"] == {
        "model_generation": False,
        "network_access": False,
        "sealed_test_access": False,
    }
