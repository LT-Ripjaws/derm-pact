"""Selection-ledger seam for Phase 2.4A-T2-30 finalist refits."""
from dermpact.eval import dermpact_phase2_4a_t2_epoch30 as epoch30


def test_selection_ledger_reuses_exact_t2_epoch20_finalists_and_stage1_reference():
    ledger = epoch30.generate_selection_ledger()
    assert ledger["finalists"] == ["c10", "c05"]
    assert [row["candidate_id"] for row in ledger["candidates"]] == ["c10", "c05"]
    assert [row["config"]["attention_dim"] for row in ledger["candidates"]] == [64, 256]
    assert [row["t2_epoch20_rank"] for row in ledger["candidates"]] == [1, 2]
    assert ledger["stage1_reference"]["candidate_id"] == "c00@20"
    assert ledger["stage1_reference"]["retrained"] is False
    assert ledger["stage1_reference"]["immutable"] is True
    assert (
        ledger["source_t2_epoch20"]["ranking_receipt"]["sha256"]
        == "8ce62411150318dee82c950bf117f94fbfae0d1d89cf308b2c1a7a16496cf00b"
    )
    assert (
        ledger["source_t2_epoch20"]["ranking"]["sha256"]
        == "d6178cb23df8ab9fc7d329157653818f7b45684af36f2382a1e555b6d1213de8"
    )
