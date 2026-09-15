"""Selection-ledger seam for Phase 2.4A-T2 epoch-20."""
from dermpact.eval import dermpact_phase2_4a_t2_epoch20 as epoch20


def test_selection_ledger_reuses_exact_t1_configs_and_stage1_reference():
    ledger = epoch20.generate_selection_ledger()
    assert ledger["advanced_candidates"] == ["c05", "c10", "c01"]
    assert [row["candidate_id"] for row in ledger["candidates"]] == ["c05", "c10", "c01"]
    assert [row["config"]["attention_dim"] for row in ledger["candidates"]] == [256, 64, 256]
    assert ledger["stage1_reference"]["candidate_id"] == "c00@20"
    assert ledger["stage1_reference"]["retrained"] is False
    assert ledger["source_t1"]["ranking_receipt"]["sha256"] == "6d96e05fda8f5bbdac9f4c1c927289023b8dc8e6556f18ebdc16754e395001b0"
