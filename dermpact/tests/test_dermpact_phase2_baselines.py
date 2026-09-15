import json
from pathlib import Path

from dermpact.eval.dermpact_phase2_baselines import (
    CANDIDATE_KEYS,
    CRITERIA,
    OUTPUT_DIR,
    POSITIVE_VALUES,
    _criterion_labels,
    _derm7pt_records,
    _ham_records,
    sha256_file,
)


def _load(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def test_phase2_1_role_boundaries_are_exact() -> None:
    ham = _ham_records()
    derm = _derm7pt_records()
    assert len(ham) == 7008
    assert len({row["cluster_id"] for row in ham}) == 5228
    assert {row["split"] for row in ham} == {"train"}
    assert len(derm) == 616
    assert {row["phase1_role"] for row in derm} == {"development"}
    assert {row["source_split"] for row in derm} == {"train", "valid"}


def test_phase2_1_criterion_mapping_is_explicit_and_binary() -> None:
    derm = _derm7pt_records()
    for criterion in CRITERIA:
        labels = _criterion_labels(derm, criterion)
        assert set(labels.tolist()) == {0, 1}
        assert POSITIVE_VALUES[criterion]


def test_phase2_1_receipt_binds_passing_development_only_results() -> None:
    receipt = _load("dermpact_phase2_1_receipt_v1.json")
    assert receipt["status"] == "PASS"
    assert receipt["phase"] == "2.1"
    assert receipt["selected_backbone"] in CANDIDATE_KEYS
    assert all(value == 0 for value in receipt["hard_scope_checks"].values())
    assert receipt["runtime_audit"]["status"] == "PASS"
    for artifact in (
        receipt["protocol_artifact"], receipt["fold_artifact"],
        receipt["paired_comparisons"], receipt["decision"],
        receipt["runtime_audit"],
        *receipt["candidate_results"].values(),
    ):
        path = OUTPUT_DIR.parents[2] / artifact["path"]
        assert path.is_file()
        assert sha256_file(path) == artifact["sha256"]


def test_phase2_1_runtime_transition_is_explicit_and_locked() -> None:
    audit = _load("dermpact_phase2_1_runtime_audit_v1.json")
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["transition"]["locked_site_packages_changed"] is False
    assert audit["transition"]["phase2_0_receipt_remains_historical"] is True


def test_phase2_1_all_candidates_use_same_fixed_probe() -> None:
    protocol = _load("dermpact_phase2_1_protocol_v1.json")
    assert protocol["head"]["C"] == 1.0
    assert protocol["head"]["candidate_specific_hyperparameter_search"] is False
    assert [row["key"] for row in protocol["candidates_in_fixed_order"]] == list(CANDIDATE_KEYS)
    for key in CANDIDATE_KEYS:
        result = _load(f"dermpact_phase2_1_{key}_result_v1.json")
        assert result["head"] == {"type": "fixed logistic probe", "C": 1.0}
        assert result["forbidden_records_loaded_or_scored"] == 0


def test_phase2_1_selection_obeys_protection_rule() -> None:
    decision = _load("dermpact_phase2_1_backbone_decision_v1.json")
    selected = decision["selected_candidate"]
    assert selected in decision["eligible_candidates"]
    assert decision["protected_violations"][selected] == []
