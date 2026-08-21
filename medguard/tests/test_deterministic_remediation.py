from __future__ import annotations

import copy

from medguard.evidence import (
    ClinicalClaim,
    ClaimType,
    EntailmentState,
    EvidenceContract,
    FindingState,
    ImageSupport,
    RelationOrigin,
    evidence_from_chunk,
)
from medguard.remediation import Action, DeterministicRemediationPolicy, run_remediation
from medguard.tests.c1_fixtures import perception


def _contract(*, entailment=EntailmentState.INSUFFICIENT, image=ImageSupport.NOT_APPLICABLE,
              contradiction=False, with_evidence=True, claim_type=ClaimType.ASSESSMENT):
    """Build a one-claim contract for the remediation hierarchy.

    Defaults to a source-bound ASSESSMENT claim because that is what "misaligned
    evidence" now means: a claim about retrieved evidence that the evidence does
    not support. An image-bound DIAGNOSIS claim is discharged by the patient image
    instead, so it cannot misalign on entailment -- pass claim_type explicitly to
    exercise the image-conflict branch.
    """
    evidence = ([evidence_from_chunk(
        {"text": "Prompt specialist assessment is indicated.", "source": "NICE",
         "version": "1", "authority": "guideline"},
        evidence_id="e1", retrieval_path=("dense", "guideline"))] if with_evidence else [])
    image_bound = claim_type in {ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL}
    return EvidenceContract(
        case_id="HAM_1", modality="dermoscopy", clinical_question="What is the diagnosis?",
        perception=perception(),
        claims=[ClinicalClaim(
            claim_id="c1", claim_type=claim_type,
            text="Prompt assessment.", image_support=image,
            target_concept=(
                ("bcc" if image is ImageSupport.CONTRADICTS else "melanoma")
                if image_bound else ""),
            target_state=FindingState.PRESENT,
            image_finding_ids=("f_commitment",) if image_bound else (),
            evidence_ids=("e1",) if with_evidence else (), entailment=entailment,
            contradiction=contradiction,
            evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
            evidence_relation_assessor_id="c1-deterministic-fixture",
            evidence_relation_receipt_sha256="4" * 64,
        )], evidence=evidence,
    )


class _Tools:
    def __init__(self):
        self.calls = []

    def _repair(self, contract, name):
        self.calls.append(name)
        value = copy.deepcopy(contract)
        old = value.claims[0]
        image_bound = old.claim_type in {ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL}
        value.claims[0] = ClinicalClaim(
            claim_id=old.claim_id, claim_type=old.claim_type, text=old.text,
            # Only an image-bound claim may declare image support at all.
            image_support=(
                (ImageSupport.CONTRADICTS if old.target_concept == "bcc"
                 else ImageSupport.SUPPORTS) if image_bound
                else ImageSupport.NOT_APPLICABLE),
            target_concept=old.target_concept, target_state=old.target_state,
            image_finding_ids=old.image_finding_ids,
            evidence_ids=("e1",), entailment=EntailmentState.SUPPORTED,
            evidence_relation_origin=RelationOrigin.HOST_DETERMINISTIC,
            evidence_relation_assessor_id="c1-deterministic-fixture",
            evidence_relation_receipt_sha256="4" * 64,
        )
        if not value.evidence:
            value.evidence = _contract().evidence
        return value

    def expand(self, contract):
        return self._repair(contract, "expand")

    def reformulate(self, contract):
        return self._repair(contract, "reformulate")

    def independent_source(self, contract):
        return self._repair(contract, "independent_source")


def test_missing_evidence_expands_once_then_passes():
    tools = _Tools()
    result = run_remediation(_contract(with_evidence=False), DeterministicRemediationPolicy(), tools)
    assert [step.action for step in result.trajectory] == [Action.EXPAND, Action.PASS]
    assert tools.calls == ["expand"]


def test_misalignment_reformulates_once_then_passes():
    tools = _Tools()
    result = run_remediation(_contract(), DeterministicRemediationPolicy(), tools)
    assert result.trajectory[0].action is Action.REFORMULATE
    assert result.final_action is Action.PASS


def test_source_contradiction_uses_independent_source():
    tools = _Tools()
    result = run_remediation(
        _contract(entailment=EntailmentState.CONTRADICTED, contradiction=True),
        DeterministicRemediationPolicy(), tools)
    assert result.trajectory[0].action is Action.INDEPENDENT_SOURCE
    assert tools.calls == ["independent_source"]


def test_image_conflict_abstains_without_tools():
    tools = _Tools()
    result = run_remediation(
        _contract(entailment=EntailmentState.SUPPORTED, image=ImageSupport.CONTRADICTS,
                  claim_type=ClaimType.DIAGNOSIS),
        DeterministicRemediationPolicy(), tools)
    assert result.final_action is Action.ABSTAIN
    assert tools.calls == []


def test_failed_repair_is_not_repeated_and_abstains():
    class Noop(_Tools):
        def _repair(self, contract, name):
            self.calls.append(name)
            return copy.deepcopy(contract)

    tools = Noop()
    result = run_remediation(_contract(), DeterministicRemediationPolicy(), tools)
    assert result.final_action is Action.ABSTAIN
    assert tools.calls == ["reformulate"]
