from __future__ import annotations

import copy
import json

from medguard.evidence import (
    ClinicalClaim,
    ClaimType,
    EntailmentState,
    FindingState,
    ImageSupport,
    RelationOrigin,
    contract_sha256,
    score_contract,
    score_contract_mapping,
)
from medguard.tests.c1_fixtures import contract, diagnosis_claim


def _serialized(value):
    return json.loads(json.dumps(value.to_dict()))


def test_complete_contract_scores_full_without_trusting_model_prose():
    score = score_contract(contract())
    assert score.valid
    assert score.structural_completion == 1.0
    assert score.contract_complete and score.structured_pass_eligible
    assert score.free_text_semantics_assessed is False
    assert score.image_claims_supported == 1
    assert score.independent_evidence_relation_rate == 1.0


def test_missing_perception_fails_closed():
    value = _serialized(contract())
    del value["perception"]
    score = score_contract_mapping(value)
    assert not score.valid and not score.structured_pass_eligible
    assert "missing" in score.issues[0]


def test_unknown_field_fails_closed_instead_of_being_ignored():
    value = _serialized(contract())
    value["generator_says_safe"] = True
    score = score_contract_mapping(value)
    assert not score.valid and "unknown" in score.issues[0]


def test_unknown_nested_claim_field_also_fails_closed():
    value = _serialized(contract())
    value["claims"][0]["self_certified"] = True
    score = score_contract_mapping(value)
    assert not score.valid and "clinical claim fields mismatch" in score.issues[0]


def test_source_contradiction_is_incomplete_until_fail_closed_action():
    claim = diagnosis_claim(
        entailment=EntailmentState.CONTRADICTED, contradiction=True)
    score = score_contract(contract(claim=claim))
    assert score.valid and not score.contract_complete and not score.structured_pass_eligible
    assert not score.output_obligation_results["abstain_on_unresolved_conflict"]


def test_tampered_evidence_text_is_rejected_by_bound_content_hash():
    value = _serialized(contract())
    value["evidence"][0]["text"] = "Tampered text."
    score = score_contract_mapping(value)
    assert not score.valid and "hash mismatch" in score.issues[0]


def test_image_conflict_overrides_a_false_support_declaration():
    claim = diagnosis_claim(target="bcc", declared=ImageSupport.SUPPORTS)
    score = score_contract(contract(claim=claim))
    assert score.image_claims_contradicted == 1
    assert score.declared_image_relation_mismatches == 1
    assert not score.structured_pass_eligible


def test_generator_assessed_entailment_is_recorded_but_not_independent_support():
    claim = diagnosis_claim(origin=RelationOrigin.GENERATOR_DECLARED)
    score = score_contract(contract(claim=claim))
    assert score.independent_evidence_relation_rate == 0.0
    assert not score.contract_complete and not score.structured_pass_eligible


def test_unavailable_visual_finding_stays_uncertain_not_invented():
    claim = ClinicalClaim(
        claim_id="c1", claim_type=ClaimType.VISUAL_FINDING,
        text="The lesion is asymmetric.", target_concept="dermoscopy:asymmetry",
        target_state=FindingState.PRESENT, image_finding_ids=("f_commitment",),
        image_support=ImageSupport.SUPPORTS, evidence_ids=("e1",),
        entailment=EntailmentState.SUPPORTED,
        evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
        evidence_relation_assessor_id="c1-deterministic-fixture",
        evidence_relation_receipt_sha256="4" * 64,
    )
    score = score_contract(contract(claim=claim))
    assert score.image_claims_uncertain == 1
    assert score.declared_image_relation_mismatches == 1
    assert not score.structured_pass_eligible


def test_contract_content_hash_is_stable_and_sensitive_to_contract_state():
    first = contract()
    assert contract_sha256(first) == contract_sha256(copy.deepcopy(first))
    second = contract(claim=diagnosis_claim(target="bcc", declared=ImageSupport.CONTRADICTS))
    assert contract_sha256(first) != contract_sha256(second)
