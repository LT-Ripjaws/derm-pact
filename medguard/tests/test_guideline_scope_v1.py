import copy
import hashlib
import json
from pathlib import Path

import pytest

from medguard.data.freeze_guideline_scope_v1 import (
    OUTPUT_DIR,
    build_coverage_matrix,
    freeze_guideline_scope,
    validate_scope_and_inventory,
    verify_guideline_scope_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def _sources():
    scope = json.loads((ROOT / "research" / "GUIDELINE_SCOPE_V1.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (ROOT / "medguard" / "data" / "corpus" / "derm_guideline_inventory_v1.json").read_text(encoding="utf-8")
    )
    recommendations = [
        json.loads(line)
        for line in (ROOT / "medguard" / "data" / "corpus" / "derm_guideline_recommendations_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return scope, inventory, recommendations


def test_scope_maps_akiec_to_both_ak_and_in_situ_carcinoma():
    scope, _inventory, _recommendations = _sources()
    assert scope["clinical_boundary"]["label_mapping"]["akiec"] == [
        "actinic_keratosis",
        "cutaneous_squamous_cell_carcinoma_in_situ",
    ]


def test_inventory_covers_every_applicable_cell_with_primary_document():
    scope, inventory, recommendations = _sources()
    conditions, _questions, applicability = validate_scope_and_inventory(scope, inventory)
    coverage = build_coverage_matrix(inventory, conditions, applicability, recommendations)
    assert len(coverage) == 29
    assert all(row["primary_document_ids"] for row in coverage)


def test_supplemental_bad_inventory_includes_each_in_scope_condition_family():
    _scope, inventory, _recommendations = _sources()
    bad_conditions = {
        condition
        for row in inventory
        if row["issuer_family"] == "BAD"
        for condition in row["conditions"]
    }
    assert bad_conditions == {
        "melanoma",
        "basal_cell_carcinoma",
        "actinic_keratosis",
        "cutaneous_squamous_cell_carcinoma_in_situ",
        "invasive_cutaneous_squamous_cell_carcinoma",
    }


def test_inventory_document_sets_are_frozen_by_issuer():
    scope, inventory, _recommendations = _sources()
    expected = scope["issuer_policy"]["expected_document_ids_by_issuer_family"]
    actual = {
        issuer: sorted(row["document_id"] for row in inventory if row["issuer_family"] == issuer)
        for issuer in expected
    }
    assert actual == {issuer: sorted(document_ids) for issuer, document_ids in expected.items()}


def test_abstracts_do_not_count_and_licensed_subset_does_count_at_recommendation_level():
    freeze_guideline_scope()
    audit = json.loads((OUTPUT_DIR / "audit.json").read_text(encoding="utf-8"))
    assert audit["current_corpus_cross_check"]["current_guideline_abstract_count"] == 3
    assert audit["current_corpus_cross_check"]["recommendation_level_coverage_inferred_from_abstracts"] is False
    assert audit["claim_status"]["comprehensive_scoped_inventory_supported"] is True
    assert audit["claim_status"]["comprehensive_guideline_retrieval_supported"] is False
    assert audit["recommendation_retrieval"]["primary_documents_indexed_at_recommendation_level"] == 11
    assert audit["recommendation_retrieval"]["indexed_recommendation_record_count"] == 66
    assert audit["recommendation_retrieval"]["ready_condition_question_cells"] == 16


def test_unknown_indexing_status_fails_closed():
    scope, inventory, _recommendations = _sources()
    broken = copy.deepcopy(inventory)
    broken[0]["machine_indexing_status"] = "free_on_web_so_probably_ok"
    with pytest.raises(ValueError, match="invalid machine indexing status"):
        validate_scope_and_inventory(scope, broken)


def test_scope_receipt_binds_raw_artifact_bytes():
    freeze_guideline_scope()
    receipt = verify_guideline_scope_bundle()
    receipt_path = OUTPUT_DIR / "receipt.json"
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert receipt_path.with_name("receipt.json.sha256").read_text(encoding="ascii").strip() == digest
    assert receipt["execution_guards"] == {
        "model_generation": False,
        "network_access": False,
        "provenance_is_not_truth": True,
        "sealed_test_access": False,
    }
