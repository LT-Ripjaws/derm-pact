import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile

from medguard.evidence import EntailmentState, RelationOrigin, VerificationState
from medguard.pipeline.generator import (
    VLMGenerator, bind_evidence_contract, render_evidence_contract,
)
from medguard.pipeline.llm import LLMClient
from medguard.errors import StructuredOutputError
from medguard.tests.c1_fixtures import perception
from medguard.tests.c1_fixtures import contract as fixture_contract, diagnosis_claim


class Client:
    def __init__(self, values):
        self.values = list(values)
        self.call_log = []
        self.vision_calls = []

    def vision(self, *_args, **_kwargs):
        self.vision_calls.append(dict(_kwargs))
        self.call_log.append({"succeeded": True, "usage_available": True,
                              "prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2, "latency_seconds": 0.1})
        return self.values.pop(0)


def _value(evidence_id="1"):
    return {
        "claims": [{"claim_id": "c1", "claim_type": "diagnosis",
                    "text": "The lesion is most consistent with melanoma.",
                    "image_support": "supports", "evidence_ids": [evidence_id],
                    "target_concept": "melanoma", "target_state": "present",
                    "image_finding_ids": ["f_commitment"],
                    "entailment": "supported", "contradiction": False}],
        "unresolved_uncertainties": [],
    }


def _kwargs():
    return dict(
        case_id="HAM_1", modality="dermoscopy", question="What is the diagnosis?",
        perception=perception(),
        chunks=[{"text": "Prompt assessment is indicated.", "source": "NICE",
                 "version": "1", "authority": "guideline",
                 "retrieval_path": ["dense", "guideline"]}],
    )


def test_host_binds_provenance_and_contract():
    contract = bind_evidence_contract(_value(), **_kwargs())
    assert contract.evidence[0].source_id == "NICE"
    assert contract.evidence[0].retrieval_path == ("dense", "guideline")
    assert contract.claims[0].entailment is EntailmentState.SUPPORTED
    assert contract.claims[0].evidence_relation_origin is RelationOrigin.GENERATOR_DECLARED


def test_empty_enum_value_defaults_like_missing_key():
    # The real VLM sometimes emits an enum field present-but-empty ("");
    # empty/null must default exactly like a missing key, not raise (which would
    # burn the one repair and force a STRUCTURED_OUTPUT_ERROR on a recoverable output).
    from medguard.evidence import FindingState, ImageSupport
    value = _value()
    value["claims"][0]["target_state"] = ""
    value["claims"][0]["image_support"] = ""
    contract = bind_evidence_contract(value, **_kwargs())
    assert contract.claims[0].target_state is FindingState.UNCERTAIN
    assert contract.claims[0].image_support is ImageSupport.UNCERTAIN


def test_unknown_evidence_id_is_dropped_not_rejected():
    # Graceful degradation (Fix 1): an image-bound claim that cites an evidence id the
    # host never supplied keeps standing on its host finding, with the bad id dropped
    # and its entailment downgraded -- no whole-contract rejection, no repair burned.
    # (The old reject-and-repair path was 117 of 187 development gen-errors.)
    client = Client([json.dumps(_value("99"))])
    generator = VLMGenerator(client, max_schema_repairs=1)
    contract = generator.generate_contract(**_kwargs())
    assert contract.claims[0].evidence_ids == ()
    assert contract.claims[0].entailment is EntailmentState.NOT_ASSESSED
    assert generator.last_metadata["schema_repairs"] == 0
    assert generator.last_metadata["remote_calls"] == 1
    assert [call["seed"] for call in client.vision_calls] == [0]


def test_source_bound_claim_with_only_unknown_evidence_is_dropped():
    # A non-image-bound claim whose only evidence id is unknown cannot stand, so it is
    # dropped; the image-bound diagnosis claim survives. The contract binds either way.
    value = {
        "claims": [
            _value()["claims"][0],
            {"claim_id": "c2", "claim_type": "assessment",
             "text": "Prompt specialist assessment is advised.",
             "image_support": "not_applicable", "evidence_ids": ["99"],
             "target_concept": "", "target_state": "uncertain",
             "image_finding_ids": [],
             "entailment": "supported", "contradiction": False},
        ],
        "unresolved_uncertainties": [
            {"uncertainty_id": "u1", "uncertainty_type": "evidence_support",
             "description": "The assessment lacks retrieved support.",
             "related_claim_ids": ["c2"]},
        ],
    }
    contract = bind_evidence_contract(value, **_kwargs())
    assert [claim.claim_id for claim in contract.claims] == ["c1"]
    # the uncertainty bound only to the dropped claim is dropped too (no dangling ref).
    assert contract.unresolved_uncertainties == []


def _exemplar_claim(**overrides):
    base = {
        "claim_id": "c2", "claim_type": "image_exemplar_support",
        "text": "A frozen train exemplar supports melanoma.",
        "image_support": "supports", "evidence_ids": ["1"],
        "target_concept": "melanoma", "target_state": "present",
        "image_finding_ids": ["f_commitment"],
        "entailment": "not_assessed", "contradiction": False,
    }
    base.update(overrides)
    return base


def _assessment_claim(**overrides):
    base = {
        "claim_id": "c2", "claim_type": "assessment",
        "text": "Prompt specialist assessment is advised.",
        "image_support": "uncertain", "evidence_ids": ["1"],
        "target_concept": "melanoma", "target_state": "uncertain",
        "image_finding_ids": ["f_commitment"],
        "entailment": "supported", "contradiction": False,
    }
    base.update(overrides)
    return base


def test_exemplar_claim_citing_a_patient_finding_is_repaired_not_rejected():
    # New residual failure: the model attaches image_finding_ids and a non-not-
    # applicable image_support to an IMAGE_EXEMPLAR_SUPPORT claim, both forbidden for
    # that type ("exemplar relation cannot cite patient findings" /
    # "exemplar relation requires not_applicable"). The host coerces both fields
    # instead of rejecting the whole contract.
    value = _value()
    value["claims"].append(_exemplar_claim())
    contract = bind_evidence_contract(value, **_kwargs())
    exemplar = next(c for c in contract.claims if c.claim_id == "c2")
    assert exemplar.image_finding_ids == ()
    from medguard.evidence import ImageSupport
    assert exemplar.image_support is ImageSupport.NOT_APPLICABLE
    assert exemplar.target_concept == "melanoma"


def test_assessment_claim_with_image_fields_is_repaired_not_rejected():
    # New residual failures: an ASSESSMENT claim (source-bound, never image-bound)
    # that declares a non-empty target_concept / image_finding_ids / a non-not-
    # applicable image_support ("type assessment is not image-bound" / "type
    # assessment requires not_applicable"). The host coerces all three.
    value = _value()
    value["claims"].append(_assessment_claim())
    contract = bind_evidence_contract(value, **_kwargs())
    assessment = next(c for c in contract.claims if c.claim_id == "c2")
    assert assessment.target_concept == ""
    assert assessment.image_finding_ids == ()
    from medguard.evidence import ImageSupport
    assert assessment.image_support is ImageSupport.NOT_APPLICABLE


def test_mis_formatted_target_concept_is_case_normalized_not_rejected():
    # A capitalized/spaced diagnosis label is a formatting slip, not new content;
    # mechanical normalization (lower + underscore) recovers it.
    value = _value()
    value["claims"][0]["target_concept"] = "  Melanoma "
    contract = bind_evidence_contract(value, **_kwargs())
    assert contract.claims[0].target_concept == "melanoma"


def test_visual_finding_concept_with_spaces_is_case_normalized():
    value = _value()
    value["claims"] = [{
        "claim_id": "c3", "claim_type": "visual_finding",
        "text": "Irregular border is present.", "image_support": "supports",
        "evidence_ids": [], "target_concept": "Irregular Border",
        "target_state": "present", "image_finding_ids": ["f_commitment"],
        "entailment": "not_assessed", "contradiction": False,
    }]
    contract = bind_evidence_contract(value, **_kwargs())
    assert contract.claims[0].target_concept == "irregular_border"


def test_exemplar_target_outside_ham7_is_dropped_not_rejected():
    # New residual failure: "exemplar target is outside HAM7" -- content the host
    # cannot safely guess, so the one claim is dropped; the contract still binds.
    value = _value()
    value["claims"].append(_exemplar_claim(target_concept="seborrheic_wart"))
    contract = bind_evidence_contract(value, **_kwargs())
    assert [c.claim_id for c in contract.claims] == ["c1"]


def test_diagnosis_target_outside_ham7_after_normalization_is_dropped():
    # New residual failure: "requires a canonical target_concept" / "diagnosis target
    # is outside HAM7" for free text that normalization cannot rescue into a HAM7
    # code. The claim is dropped, not the whole contract.
    value = _value()
    value["claims"][0]["target_concept"] = "invasive malignant melanoma"
    contract = bind_evidence_contract(value, **_kwargs())
    assert list(contract.claims) == []


def test_all_five_residual_failure_types_together_still_bind_a_contract():
    # A realistic mixed VLM output: one good diagnosis claim, one exemplar claim
    # misusing image fields (repaired), one assessment claim misusing image fields
    # (repaired), and one exemplar claim with an unrecoverable target (dropped).
    # None of this rejects the whole contract.
    value = _value()
    value["claims"].extend([
        _exemplar_claim(claim_id="c2"),
        _assessment_claim(claim_id="c3"),
        _exemplar_claim(claim_id="c4", target_concept="not_a_real_class"),
    ])
    contract = bind_evidence_contract(value, **_kwargs())
    assert [c.claim_id for c in contract.claims] == ["c1", "c2", "c3"]


def test_generator_cannot_promote_its_relation_origin_or_replace_perception():
    value = _value()
    value["claims"][0]["evidence_relation_origin"] = "host_deterministic"
    value["perception"] = {"committed_diagnosis": "bcc"}
    try:
        bind_evidence_contract(value, **_kwargs())
    except StructuredOutputError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("generator replaced host-owned relation/perception state")


def test_renderer_rejects_directly_stamped_pass_with_image_conflict():
    value = fixture_contract(claim=diagnosis_claim(target="bcc"))
    value.verification_state = VerificationState.SATISFIED
    value.proposed_action = "PASS"
    value.termination_reason = "caller stamped PASS"
    try:
        render_evidence_contract(value)
    except StructuredOutputError as exc:
        assert "unverified PASS" in str(exc)
    else:
        raise AssertionError("renderer exposed a caller-stamped image-conflict PASS")


def test_renderer_uses_checked_image_target_not_mismatched_generator_prose():
    claim = replace(
        diagnosis_claim(), text="The lesion is definitely BCC despite the target.")
    value = fixture_contract(claim=claim)
    value.verification_state = VerificationState.SATISFIED
    value.proposed_action = "PASS"
    value.termination_reason = "verified"
    rendered = render_evidence_contract(value)
    assert "Diagnosis: melanoma (present)." in rendered
    assert "definitely BCC" not in rendered


def test_vlm_client_preserves_ordered_multiview_input():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok")),
                ], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "a.jpg"
        second = Path(directory) / "b.jpg"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        client = LLMClient(api_key="injected-test-key", _client=fake, delay_sec=0)
        assert client.vision("question", [str(first), str(second)], seed=0) == "ok"
    content = captured["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"] != content[1]["image_url"]["url"]
    assert captured["seed"] == 0
