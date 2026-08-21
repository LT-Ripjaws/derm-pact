"""Table 1 (corpus/gap) is deterministic and the Gap A zero-hit holds."""
from medguard.eval.corpus_gap_table_v1 import build_table


def test_corpus_gap_table_is_deterministic_and_covers_138_records():
    a = build_table()
    b = build_table()
    assert a == b  # pure function of the frozen workbook
    assert a["source"]["records"] == 138
    roles = a["corpus_composition"]["by_medguard_role"]
    assert roles["METHOD"] == 34 and roles["GOV"] == 34 and roles["THREAT"] == 16


def test_gap_a_provenance_composition_is_absent_from_prior_art():
    prov = build_table()["gap_a_evidence_contract_provenance_composition"]
    # The authority-class tier and source-version binding are absent from every
    # record, so the bound composition is 0 of 138 -- MedGuard's Gap A.
    assert prov["per_element"]["authority_class"]["records"] == 0
    assert prov["per_element"]["source_version"]["records"] == 0
    assert prov["bound_composition_records"] == 0
    # Every claimed count is auditable: matched MG# lists are present.
    for element in prov["per_element"].values():
        assert element["records"] == len(element["matched_mg"])
