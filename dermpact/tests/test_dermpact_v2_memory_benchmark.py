from __future__ import annotations

from dermpact.core.eval import memory_benchmark as benchmark


def test_memory_benchmark_is_leave_one_out_and_honestly_labeled():
    result = benchmark.run()
    assert result["n_queries"] == result["n_bank_episodes"] == 50
    assert result["official_or_final_rows_loaded"] == 0
    assert "descriptive dataset-label proxies only" in result["interpretation"]
    assert result["correction_pattern_retrieval"].startswith("not_evaluable")
    assert all(row["n_matches"] <= result["k"] for row in result["rows"])
