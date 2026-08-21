import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CARD_PATH = ROOT / "research" / "implementation_cards" / "H2H_IMPLEMENTATION_CARDS_V1.json"


def _spec():
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


def test_roster_and_adaptive_slot_are_exact_and_preoutcome():
    value = _spec()
    cards = {card["id"]: card for card in value["cards"]}
    assert set(cards) == {"B0", "B1", "B2", "B3", "B4", "B5", "B6", "M", "A0", "A1", "A2", "A3", "A4"}
    assert value["primary_roster"] == ["B0", "B1", "B4", "B5", "B6", "M"]
    assert value["adaptive_slot"]["selected"] == "B4"
    assert cards["B4"]["h2h_role"] == "selected_primary_adaptive_slot"


def test_no_incomplete_paper_system_is_labeled_a_reproduction():
    value = _spec()
    paper_cards = [card for card in value["cards"] if card["source_paper_ids"]]
    assert paper_cards
    for card in paper_cards:
        fidelity = card["fidelity"]
        if fidelity["omitted_load_bearing"]:
            assert fidelity["faithful_reproduction"] is False
            assert fidelity["classification"] in {"style_arm", "component_proxy", "compatible_adaptation"}
            assert "reproduction" not in fidelity["reporting_label"].lower()


def test_ablation_matrix_changes_only_the_declared_component():
    cards = {card["id"]: card for card in _spec()["cards"]}
    assert cards["A0"]["alias_of"] == "M"
    assert "execution payload" in cards["A0"]["cache_rule"]
    assert "byte-for-byte" in cards["A0"]["cache_rule"]
    assert "alias_of=M" in cards["A0"]["cache_rule"]
    assert cards["A1"]["components"]["omitted"] == ["image_exemplar_retrieval"]
    assert cards["A2"]["components"]["omitted"] == ["graph_lineage"]
    assert cards["A3"]["components"]["omitted"] == ["bounded_remediation"]
    assert cards["A4"]["components"]["omitted"] == ["verification_gate"]
    for arm_id in ("A1", "A2", "A3", "A4"):
        assert cards[arm_id]["single_change_from_A0"]


def test_strawman_and_budget_guards_are_machine_visible():
    value = _spec()
    cards = {card["id"]: card for card in value["cards"]}
    assert value["common_conditions"]["generation"]["maximum_remote_attempts_per_case"] == 4
    assert value["common_conditions"]["generation"]["automatic_api_retries"] == 1
    assert value["common_conditions"]["generation"]["seed"]["status"].startswith("implemented")
    assert value["common_conditions"]["generation"]["seed"]["required_value"] == 0
    assert value["common_conditions"]["cache"]["cross_arm_reuse"].startswith("forbidden except M and A0")
    assert cards["B0"]["retrieval"]["strategy"] == "none"
    assert cards["B1"]["generation"]["output_contract"] is False
    assert cards["M"]["generation"]["output_contract"] is True
    assert cards["B4"]["retrieval"]["selector"]["gamma"] == 0.5
    assert cards["B6"]["generation"]["maximum_logical_calls"] <= 4
    assert cards["A4"]["restriction"].startswith("offline evaluation only")


def test_retired_components_and_provenance_as_truth_are_excluded():
    value = _spec()
    exclusions = set(value["common_conditions"]["hard_exclusions"])
    assert {"learned_depth_two_policy", "HANDOFF", "provenance_as_truth"} <= exclusions
    encoded = CARD_PATH.read_text(encoding="utf-8")
    assert '"HANDOFF"' in encoded
    assert "provenance is a lineage axis, never truth" in encoded


def test_source_evidence_files_are_hash_bound_and_present():
    import hashlib

    for paper in _spec()["source_papers"].values():
        path = ROOT / paper["full_text"]["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == paper["full_text"]["sha256"]
