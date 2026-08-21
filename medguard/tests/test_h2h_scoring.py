"""The truth-joined H2H scorer wires every endpoint and never fakes a value."""
from medguard.eval.citation_integrity import AssertedCitation
from medguard.eval.h2h_scoring import (
    ArmCaseOutcome,
    paired_accuracy_difference,
    score_h2h,
)


TRUTH = {"c1": "melanoma", "c2": "bcc", "c3": "nevus", "c4": "bkl"}


class _FakeResolver:
    def __init__(self, ids):
        self.ids = set(ids)

    def resolve(self, document_id):
        if str(document_id) in self.ids:
            return {"eligible_claim_types": ("management",)}
        return None

    def offered_ids(self, committed_diagnosis, claim_type=None):
        return set(self.ids)


def _gen(calls=1, prompt=10, completion=20, latency=0.5, repairs=0, cache_hit=False):
    return {
        "remote_calls": calls, "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": prompt + completion, "latency_seconds": latency,
        "schema_repairs": repairs, "cache_hit": cache_hit,
    }


def _m_outcomes():
    # Probe arm: correct on c1/c2/c3, wrong on c4 (predicts melanoma, truth bkl).
    preds = {"c1": "melanoma", "c2": "bcc", "c3": "nevus", "c4": "melanoma"}
    conf = {"c1": 0.9, "c2": 0.85, "c3": 0.8, "c4": 0.4}
    act = {"c1": 0.9, "c2": 0.8, "c3": 0.2, "c4": 0.7}  # P(actionable)
    cite = {"c1": (AssertedCitation(cited_id="aad:1", claim_type="management"),),
            "c2": (AssertedCitation(cited_id="aad:2", claim_type="management"),)}
    outcomes = []
    for cluster in ("c1", "c2", "c3", "c4"):
        answered = cluster != "c4"  # abstain on the low-confidence wrong case
        outcomes.append(ArmCaseOutcome(
            arm_id="M", case_id=f"lesion:{cluster}", cluster_id=cluster,
            predicted_diagnosis=preds[cluster], answered=answered,
            abstained=not answered, final_action="PASS" if answered else "ABSTAIN",
            contract_applicable=True, calibrated_confidence=conf[cluster],
            actionable_score=act[cluster], contract_complete=answered,
            structured_pass_eligible=answered, structural_completion=1.0,
            citations=cite.get(cluster, ()), generation=_gen() if answered else _gen(calls=0, prompt=0, completion=0, latency=0.0),
            answer_text=f"Decision: {preds[cluster]}." if answered else "",
            context_texts=("guideline context",),
        ))
    return outcomes


def _b0_outcomes():
    # VLM arm: lower accuracy, no probe confidence, no contract, no citations.
    preds = {"c1": "melanoma", "c2": "nevus", "c3": "nevus", "c4": "bcc"}
    outcomes = []
    for cluster in ("c1", "c2", "c3", "c4"):
        outcomes.append(ArmCaseOutcome(
            arm_id="B0", case_id=f"lesion:{cluster}", cluster_id=cluster,
            predicted_diagnosis=preds[cluster], answered=True, abstained=False,
            final_action="ANSWER_UNGATED", contract_applicable=False,
            calibrated_confidence=None, actionable_score=None,
            generation=_gen(), answer_text=f"Diagnosis: {preds[cluster]}.",
        ))
    return outcomes


def test_process_trust_block_disaggregated_and_na_for_noncontract():
    from medguard.eval.h2h_scoring import _process_trust_block

    def _o(arm, pt):
        return ArmCaseOutcome(
            arm_id=arm, case_id="lesion:c1", cluster_id="c1",
            predicted_diagnosis="melanoma", answered=True, abstained=False,
            final_action="PASS", contract_applicable=pt is not None, process_trust=pt)

    contract_arm = [
        _o("M", {"P5_contradiction_handling": 1.0, "P8_receipt_completeness": None}),
        _o("M", {"P5_contradiction_handling": 0.0, "P8_receipt_completeness": None}),
    ]
    block = _process_trust_block(contract_arm)
    assert block["applicable"] is True and block["profiles_scored"] == 2
    assert block["dimensions"]["P5_contradiction_handling"] == 0.5   # mean(1.0, 0.0)
    assert block["dimensions"]["P8_receipt_completeness"] is None     # all None -> None, not 0
    # Non-contract arms carry no profile -> not_applicable, never a fabricated zero.
    assert _process_trust_block([_o("B0", None)]) == {
        "applicable": False, "status": "not_applicable"}


def test_score_h2h_wires_every_endpoint():
    outcomes = _m_outcomes() + _b0_outcomes()
    resolver = _FakeResolver({"aad:1", "aad:2"})
    report = score_h2h(
        outcomes, TRUTH, citation_resolver=resolver, question="What is the diagnosis?",
        paired_comparisons=(("M", "B0"),), n_boot=200,
    )
    m = report["per_arm"]["M"]
    b0 = report["per_arm"]["B0"]

    # Diagnosis: M correct on c1/c2/c3 (3/4); B0 correct on c1/c3 (2/4).
    assert m["diagnosis"]["accuracy"] == 0.75
    assert b0["diagnosis"]["accuracy"] == 0.5
    assert "balanced_accuracy" in m["diagnosis"] and "macro_f1" in m["diagnosis"]

    # Actionable: M has a usable score (AUROC + sensitivity@0.90 defined); B0 does not.
    assert m["actionable"]["actionable_score_available"] is True
    assert m["actionable"]["sensitivity_at_fixed_specificity"] is not None
    assert m["actionable"]["auroc"] is not None
    assert b0["actionable"]["actionable_score_available"] is False
    assert b0["actionable"]["auroc"] is None
    assert b0["actionable"]["sensitivity_at_fixed_specificity"] is None
    # The 2x2 itself is still produced for B0 from hard labels.
    assert b0["actionable"]["confusion_matrix"]

    # Selective: M has calibrated confidence -> AURC; B0 does not.
    assert m["selective"]["aurc"] is not None
    assert m["selective"]["coverage"] == 0.75  # abstained on c4
    assert b0["selective"]["aurc"] is None
    assert b0["selective"]["coverage"] == 1.0

    # Provenance: M cited two guideline records (both intact); B0 cited nothing.
    assert m["provenance_citation_integrity"]["integrity_rate"] == 1.0
    assert m["provenance_citation_integrity"]["total_citations"] == 2
    assert b0["provenance_citation_integrity"]["integrity_rate"] is None

    # Process: M applicable, B0 not_applicable (never zero).
    assert m["process_contract_completion"]["applicable"] is True
    assert b0["process_contract_completion"]["status"] == "not_applicable"

    # Efficiency + CIs + RAGAS supporting.
    assert m["efficiency"]["generation_calls"] == 3  # answered 3 of 4
    assert b0["efficiency"]["generation_calls"] == 4
    assert m["diagnostic_accuracy_ci"]["estimate"] == 0.75
    assert m["ragas_supporting"]["status"] == "mapped_not_executed"
    assert m["ragas_supporting"]["role"] == "supporting"

    # Paired M - B0 diagnostic accuracy difference is 0.75 - 0.5 = +0.25.
    assert report["paired_differences"][0]["estimate"] == 0.25
    assert report["endpoints_registered"]["bootstrap_seed"] == 20260712


def test_missing_truth_label_is_an_error_not_a_silent_zero():
    outcomes = _b0_outcomes()
    try:
        score_h2h(outcomes, {"c1": "melanoma"}, citation_resolver=_FakeResolver(()),
                  question="q", n_boot=50)
    except ValueError as exc:
        assert "label" in str(exc)
    else:
        raise AssertionError("scorer accepted cases with no joined truth label")


def test_paired_difference_is_lesion_clustered_and_directional():
    diff = paired_accuracy_difference(
        _m_outcomes(), _b0_outcomes(), TRUTH, n_boot=200,
    )
    assert diff["arm_a"] == "M" and diff["arm_b"] == "B0"
    assert diff["n_paired"] == 4
    assert diff["estimate"] == 0.25
