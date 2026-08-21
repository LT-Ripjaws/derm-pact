from medguard.agents import score_process_trust
from medguard.evidence import (
    EntailmentState, RelationOrigin, VerificationState,
)
from medguard.remediation import Action, RemediationStep
from medguard.tests.c1_fixtures import contract, diagnosis_claim


def _contract(*, contradiction=False, action="PASS"):
    state = VerificationState.FAILED_CLOSED if action == "ABSTAIN" else VerificationState.SATISFIED
    claim = diagnosis_claim(
        entailment=(EntailmentState.CONTRADICTED if contradiction
                    else EntailmentState.SUPPORTED),
        contradiction=contradiction,
    )
    value = contract(claim=claim)
    value.verification_state = state
    value.proposed_action = action
    value.termination_reason = (
        "contract satisfied" if action == "PASS" else "contradiction unresolved")
    return value


def test_all_observable_dimensions_pass_for_complete_contract():
    contract = _contract()
    trajectory = [RemediationStep(1, (), Action.PASS)]
    profile = score_process_trust(contract, trajectory)
    assert set(profile.dimensions) == {
        "P1_image_claim_consistency", "P2_evidence_sufficiency",
        "P3_provenance_completeness", "P4_claim_evidence_binding",
        "P5_contradiction_handling", "P6_uncertainty_handling",
        "P7_bounded_action_appropriateness", "P8_receipt_completeness",
    }
    assert all(value == 1.0 for value in profile.dimensions.values())
    assert profile.composite == 1.0


def test_fail_closed_contradiction_is_handled_but_not_evidence_sufficient():
    contract = _contract(contradiction=True, action="ABSTAIN")
    trajectory = [RemediationStep(1, (), Action.ABSTAIN)]
    profile = score_process_trust(contract, trajectory)
    assert profile.dimensions["P2_evidence_sufficiency"] == 0.0
    assert profile.dimensions["P5_contradiction_handling"] == 1.0


def test_p2_credits_image_bound_diagnosis_without_text_entailment():
    # The production build_image_bound_claims diagnosis claim is NOT_ASSESSED with no
    # evidence -- it is discharged by the frozen patient image, not by retrieved text.
    # The old P2 demanded text-entailment on every claim, so it scored 0.0 for every
    # arm; the type-appropriate redefinition credits its image support.
    claim = diagnosis_claim(
        entailment=EntailmentState.NOT_ASSESSED,
        origin=RelationOrigin.NOT_ASSESSED, with_evidence=False,
    )
    value = contract(claim=claim)
    profile = score_process_trust(value, [RemediationStep(1, (), Action.PASS)])
    assert profile.dimensions["P2_evidence_sufficiency"] == 1.0
