from medguard.agents import ProcessTrustProfile
from medguard.evidence import VerificationState
from medguard.eval.harness import CaseResult, aggregate_run
from medguard.tests.c1_fixtures import contract


def _contract(case_id, action):
    value = contract()
    value.case_id = case_id
    value.verification_state = (VerificationState.SATISFIED if action == "PASS"
                                else VerificationState.FAILED_CLOSED)
    value.proposed_action = action
    value.termination_reason = "done"
    return value


def test_aggregate_keeps_coverage_accuracy_ragas_and_trust_separate():
    trust = ProcessTrustProfile({"P1": 1.0, "P2": 0.5}, 0.75)
    records = [
        CaseResult("a", "la", "melanoma", "melanoma", True, "PASS",
                   _contract("a", "PASS"), trust, calls=1, tokens=10,
                   ragas={"faithfulness": 0.8}),
        CaseResult("b", "lb", "nevus", "", False, "ABSTAIN",
                   _contract("b", "ABSTAIN"), trust,
                   ragas={"faithfulness": None}),
    ]
    report = aggregate_run(records, arm="candidate", labels=("melanoma", "nevus"))
    assert report.coverage == 0.5
    assert report.classification["accuracy"] == 1.0
    assert report.action_counts == {"PASS": 1, "ABSTAIN": 1}
    assert report.ragas_means["faithfulness"] == 0.8
    assert report.process_trust_dimensions["P2"] == 0.5


def test_record_rejects_prediction_on_abstention():
    trust = ProcessTrustProfile({}, 0.0)
    try:
        CaseResult("a", "la", "melanoma", "melanoma", False, "ABSTAIN",
                   _contract("a", "ABSTAIN"), trust).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("abstention with prediction was accepted")
