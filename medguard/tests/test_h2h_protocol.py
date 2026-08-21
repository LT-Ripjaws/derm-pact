from dataclasses import replace

import pytest

from medguard.eval.h2h_protocol import (
    ArmId,
    ArmProtocol,
    CaseLineage,
    EvidenceIdentity,
    ProtocolValidationError,
    UsageAccounting,
    build_request_key,
    enforce_usage,
    reject_truth_keys,
    summarize_scores,
    verify_identical_lineage,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def _protocol(arm: ArmId | str, **overrides):
    values = {
        "protocol_id": "medguard-h2h-development-v1",
        "arm_id": arm,
        "model_placeholder": "MODEL_TO_BE_LOCKED",
        "prompt_hash_placeholder": "PROMPT_SHA256_TO_BE_LOCKED",
        "arm_config": {"retrieval": arm.value if isinstance(arm, ArmId) else str(arm)},
        "decoding": {"temperature": 0, "seed": 7},
        "retry_policy": {"max_retries": 0},
        "corpus_receipt": "corpus-receipt-sha256-placeholder",
        "perception_receipt": "perception-receipt-sha256-placeholder",
        "perception_artifact": "perception-artifact-sha256-placeholder",
        "code_snapshot": "code-snapshot-placeholder",
    }
    values.update(overrides)
    return ArmProtocol(**values)


def _lineage(**overrides):
    values = {
        "case_id": "HAM_0001",
        "ordered_image_sha256s": (_A, _B),
        "frozen_perception": {
            "commitment": {"concept_code": "melanoma", "confidence": 0.91},
            "artifact_sha256": _C,
        },
    }
    values.update(overrides)
    return CaseLineage(**values)


def test_closed_arm_set_and_validate_only_are_fail_closed():
    assert _protocol("b0").arm_id is ArmId.B0
    # The enum is the frozen roster: primary comparators, the proposed method,
    # and the paired ablations.  B2/B3 are reference-only and must stay
    # unconstructible.
    assert [arm.value for arm in ArmId] == [
        "B0", "B1", "B4", "B5", "B6", "M", "A0", "A1", "A2", "A3", "A4",
    ]
    for absent in ("B2", "B3", "B7"):
        with pytest.raises(ProtocolValidationError, match="unknown H2H arm"):
            _protocol(absent)
    with pytest.raises(ProtocolValidationError, match="zero call and token ceilings"):
        _protocol(ArmId.B0, generation_call_ceiling=1)


def test_lineage_requires_the_declared_arm_set_and_exact_case_view_order():
    case = _lineage()
    verify_identical_lineage({arm: (case,) for arm in ArmId})

    trio = (ArmId.B0, ArmId.B1, ArmId.B4)
    reordered = replace(case, ordered_image_sha256s=(_B, _A))
    with pytest.raises(ProtocolValidationError, match="lineage differs"):
        verify_identical_lineage(
            {ArmId.B0: (case,), ArmId.B1: (reordered,), ArmId.B4: (case,)},
            expected=trio,
        )
    changed = replace(
        case,
        frozen_perception={"commitment": {"concept_code": "bcc", "confidence": 0.91}},
    )
    with pytest.raises(ProtocolValidationError, match="lineage differs"):
        verify_identical_lineage(
            {ArmId.B0: (case,), ArmId.B1: (case,), ArmId.B4: (changed,)},
            expected=trio,
        )
    # A runner cannot silently drop an arm it claims to compare.
    with pytest.raises(ProtocolValidationError, match="exactly B0/B1/B4"):
        verify_identical_lineage(
            {ArmId.B0: (case,), ArmId.B1: (case,)}, expected=trio,
        )
    with pytest.raises(ProtocolValidationError, match="at least two arms"):
        verify_identical_lineage({ArmId.M: (case,)})
    # Comparing an explicit subset stays legal when it is declared.
    verify_identical_lineage({arm: (case,) for arm in trio}, expected=trio)


def test_truth_keys_are_rejected_recursively_without_rejecting_commitments():
    _lineage()
    with pytest.raises(ProtocolValidationError, match="truth-bearing key"):
        _lineage(frozen_perception={"commitment": {"true_diagnosis": "melanoma"}})
    with pytest.raises(ProtocolValidationError, match="truth-bearing key"):
        reject_truth_keys({"nested": [{"ground-truth": "melanoma"}]})


def test_cache_namespace_and_request_key_bind_every_material_input():
    protocols = {arm: _protocol(arm) for arm in ArmId}
    assert len({protocol.cache_namespace for protocol in protocols.values()}) == len(ArmId)
    lineage = _lineage()
    evidence = (EvidenceIdentity("guideline-1", _C),)
    keys = {
        arm: build_request_key(
            protocol, lineage, question="What is the diagnosis?", retrieved_evidence=evidence
        )
        for arm, protocol in protocols.items()
    }
    assert len(set(keys.values())) == len(ArmId)
    original = keys[ArmId.B1]
    assert original != build_request_key(
        replace(protocols[ArmId.B1], code_snapshot="different-code"),
        lineage,
        question="What is the diagnosis?",
        retrieved_evidence=evidence,
    )
    changed_arm_config = replace(
        protocols[ArmId.B1], arm_config={"retrieval": "dense", "top_k": 9}
    )
    assert protocols[ArmId.B1].cache_namespace != changed_arm_config.cache_namespace
    assert original != build_request_key(
        changed_arm_config,
        lineage,
        question="What is the diagnosis?",
        retrieved_evidence=evidence,
    )
    assert original != build_request_key(
        protocols[ArmId.B1],
        lineage,
        question="What is the diagnosis?",
        retrieved_evidence=(EvidenceIdentity("guideline-1", _B),),
    )
    assert original != build_request_key(
        protocols[ArmId.B1], lineage, question="What is the management?", retrieved_evidence=evidence
    )
    with pytest.raises(TypeError):
        protocols[ArmId.B1].arm_config["retrieval"] = "mutated"
    with pytest.raises(TypeError):
        lineage.frozen_perception["commitment"]["concept_code"] = "mutated"


def test_usage_enforces_zero_mode_and_nonzero_ceiling_mode():
    validate = _protocol(ArmId.B0)
    enforce_usage(validate, UsageAccounting())
    with pytest.raises(ProtocolValidationError, match="call ceiling exceeded"):
        enforce_usage(validate, UsageAccounting(actual_generation_calls=1))
    with pytest.raises(ProtocolValidationError, match="actual or cached"):
        enforce_usage(validate, UsageAccounting(logical_generation_calls=1))

    live = _protocol(
        ArmId.B0,
        validate_only=False,
        generation_call_ceiling=2,
        token_ceiling=10,
    )
    enforce_usage(
        live,
        UsageAccounting(
            actual_generation_calls=2,
            actual_prompt_tokens=6,
            actual_completion_tokens=4,
        ),
    )
    with pytest.raises(ProtocolValidationError, match="token ceiling exceeded"):
        enforce_usage(
            live,
            UsageAccounting(
                actual_generation_calls=1,
                actual_prompt_tokens=8,
                actual_completion_tokens=3,
            ),
        )


def test_missing_scores_remain_missing_and_never_become_zero():
    summary = summarize_scores({"c1": 0.8, "c2": None, "c3": 0.4})
    assert summary.total_count == 3
    assert summary.observed_count == 2
    assert summary.missing_count == 1
    assert summary.missing_case_ids == ("c2",)
    assert summary.mean == pytest.approx(0.6)

    all_missing = summarize_scores({"c1": None})
    assert all_missing.mean is None
    assert all_missing.missing_count == 1
