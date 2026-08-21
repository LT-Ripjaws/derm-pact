"""Focused offline tests for the shared H2H generation foundation."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile

import pytest

from medguard.errors import CacheIntegrityError
from medguard.eval.h2h_generation import (
    H2H_GENERATION_SEED,
    H2HGenerationRequest,
    PromptAssembly,
    SharedClinicalTask,
    adapter_for_arm,
    effective_arm_id,
)
from medguard.eval.h2h_protocol import (
    ArmId,
    ArmProtocol,
    CaseLineage,
    EvidenceIdentity,
    ProtocolValidationError,
)
from medguard.pipeline.generation_cache import CachedStructuredGenerator, GenerationCache
from medguard.tests.c1_fixtures import contract


_A = hashlib.sha256(b"view-one").hexdigest()
_B = hashlib.sha256(b"view-two").hexdigest()
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64


def _common_patient_context():
    return {
        "commitment": {"concept_code": "melanoma", "confidence": 0.91},
        "artifact_sha256": _C,
    }


def _lineage():
    return CaseLineage(
        case_id="HAM_0001",
        ordered_image_sha256s=(_A, _B),
        frozen_perception=_common_patient_context(),
    )


def _evidence(arm: ArmId):
    if arm is ArmId.B0:
        return (), "(no external evidence retrieved)"
    return (
        (EvidenceIdentity(
            "guideline-1", _D, "NICE", "v1", ("dense", "guideline"), _E,
        ),),
        "[guideline-1] (guideline; v1) Supporting assessment evidence.",
    )


def _protocol(arm: ArmId, task: SharedClinicalTask, *, seed: int = H2H_GENERATION_SEED,
              model: str = "qwen/qwen3-vl-32b-instruct", code_snapshot: str = "code-v1",
              decoding=None):
    execution_arm = effective_arm_id(arm)
    adapter = adapter_for_arm(arm)
    template = PromptAssembly(
        task=task,
        adapter=adapter,
        common_patient_context=_common_patient_context(),
        evidence_context="adapter template only",
    ).template_sha256
    return ArmProtocol(
        protocol_id="medguard-derm-evidence-contract-h2h-v1",
        arm_id=execution_arm,
        model_placeholder=model,
        prompt_hash_placeholder=template,
        arm_config={"arm": execution_arm.value, "retrieval": execution_arm.value},
        decoding=(decoding if decoding is not None else {
            "temperature": 0.2,
            "max_completion_tokens": 768,
            "seed": seed,
        }),
        retry_policy={"automatic_api_retries": 1, "schema_repairs_maximum": 1},
        corpus_receipt="corpus-v1",
        perception_receipt="perception-receipt-v1",
        perception_artifact="perception-artifact-v1",
        code_snapshot=code_snapshot,
        validate_only=False,
        generation_call_ceiling=4,
        token_ceiling=3072,
    )


def _request(arm: ArmId, *, task: SharedClinicalTask | None = None,
             protocol: ArmProtocol | None = None, lineage: CaseLineage | None = None,
             evidence=None, evidence_context: str | None = None):
    task = task or SharedClinicalTask()
    evidence, default_context = _evidence(arm) if evidence is None else (evidence, evidence_context)
    if evidence_context is None:
        evidence_context = default_context
    protocol = protocol or _protocol(arm, task)
    return H2HGenerationRequest(
        arm_id=arm,
        protocol=protocol,
        lineage=lineage or _lineage(),
        task=task,
        adapter=adapter_for_arm(arm),
        common_patient_context=_common_patient_context(),
        evidence_context=evidence_context,
        retrieved_evidence=tuple(evidence),
        seed=H2H_GENERATION_SEED,
    )


def test_every_arm_shares_the_clinical_task_and_only_adapters_vary():
    requests = {arm: _request(arm) for arm in ArmId}
    shared = next(iter(requests.values())).prompt
    for request in requests.values():
        prompt = request.prompt
        assert prompt.task.clinical_question == shared.task.clinical_question
        assert prompt.task.diagnosis_vocabulary == shared.task.diagnosis_vocabulary
        assert prompt.task.ordered_image_rule == shared.task.ordered_image_rule
        assert prompt.task.core_prompt == shared.task.core_prompt
        assert prompt.common_patient_context == shared.common_patient_context
        assert set(prompt.adapter.declared_variation()) == {
            "evidence_representation", "obligations",
        }
        assert "ARM-SPECIFIC CONTEXT ADAPTER" in prompt.rendered_prompt

    # A0 is the sole exception: it is the M execution payload, prompt, and cache identity.
    assert requests[ArmId.A0].prompt.rendered_prompt == requests[ArmId.M].prompt.rendered_prompt
    assert requests[ArmId.A0].cache_identity.namespace == requests[ArmId.M].cache_identity.namespace
    assert requests[ArmId.A0].cache_identity.request_key == requests[ArmId.M].cache_identity.request_key
    assert requests[ArmId.A0].cache_identity.alias_of is ArmId.M


def test_common_prompt_context_rejects_truth_bearing_fields():
    with pytest.raises(ProtocolValidationError, match="truth-bearing key"):
        PromptAssembly(
            task=SharedClinicalTask(),
            adapter=adapter_for_arm(ArmId.B0),
            common_patient_context={"true_diagnosis": "melanoma"},
            evidence_context="(no external evidence retrieved)",
        )


def test_cache_identity_binds_arm_model_prompt_decoding_evidence_perception_and_code():
    base = _request(ArmId.B1)
    baseline = base.cache_identity
    task = base.task

    changed_model = _request(
        ArmId.B1, task=task,
        protocol=_protocol(ArmId.B1, task, model="another-model"),
    )
    changed_decoding = _request(
        ArmId.B1, task=task,
        protocol=_protocol(
            ArmId.B1, task,
            decoding={"temperature": 0.1, "max_completion_tokens": 768, "seed": 0},
        ),
    )
    changed_code = _request(
        ArmId.B1, task=task,
        protocol=_protocol(ArmId.B1, task, code_snapshot="code-v2"),
    )
    changed_prompt = replace(base, evidence_context="Different permitted evidence rendering.")
    changed_evidence = replace(
        base, retrieved_evidence=(EvidenceIdentity(
            "guideline-1", _C, "NICE", "v1", ("dense", "guideline"), _E,
        ),),
    )
    changed_evidence_lineage = replace(
        base, retrieved_evidence=(EvidenceIdentity(
            "guideline-1", _D, "NICE", "v2", ("graph", "version"), _E,
        ),),
    )
    changed_perception = replace(
        base,
        lineage=replace(
            base.lineage,
            frozen_perception={
                "commitment": {"concept_code": "bcc", "confidence": 0.91},
                "artifact_sha256": _C,
            },
        ),
        common_patient_context={
            "commitment": {"concept_code": "bcc", "confidence": 0.91},
            "artifact_sha256": _C,
        },
    )
    changed_arm = _request(ArmId.B4, task=task)

    for request in (
        changed_model, changed_decoding, changed_code, changed_prompt,
        changed_evidence, changed_evidence_lineage, changed_perception, changed_arm,
    ):
        assert request.cache_identity != baseline

    with pytest.raises(ProtocolValidationError, match="decoding seed"):
        replace(base, seed=1)
    with pytest.raises(ProtocolValidationError, match="rendered_prompt_sha256"):
        from medguard.eval.h2h_protocol import build_request_key

        build_request_key(
            base.protocol, base.lineage,
            question=base.task.clinical_question,
            retrieved_evidence=base.retrieved_evidence,
        )


class _Delegate:
    def __init__(self):
        self.calls = 0
        self.last_metadata = {}

    def generate_contract(self, **_kwargs):
        self.calls += 1
        self.last_metadata = {"remote_calls": 1, "prompt_tokens": 3, "completion_tokens": 5}
        return contract()


def _materialized_kwargs(request: H2HGenerationRequest, directory: Path):
    first = directory / "view-one.jpg"
    second = directory / "view-two.jpg"
    first.write_bytes(b"view-one")
    second.write_bytes(b"view-two")
    chunks = [
        {
            "document_id": item.evidence_id,
            "content_sha256": item.content_sha256,
            "record_sha256": item.record_sha256,
            "text": "Supporting assessment evidence.",
            "source": "NICE",
            "version": "v1",
            "authority": "guideline",
            "retrieval_path": ["dense", "guideline"],
        }
        for item in request.retrieved_evidence
    ]
    return request.structured_generator_kwargs(
        modality="dermoscopy",
        perception=object(),
        chunks=chunks,
        image_paths=(str(first), str(second)),
    )


def test_cache_only_replay_uses_zero_generation_calls_and_rejects_cross_arm_reuse():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        medguard = _request(ArmId.M)
        a0 = _request(ArmId.A0, protocol=medguard.protocol)
        medguard_kwargs = _materialized_kwargs(medguard, root)
        with pytest.raises(ProtocolValidationError, match="image order/content"):
            medguard.structured_generator_kwargs(
                modality="dermoscopy",
                perception=object(),
                chunks=medguard_kwargs["chunks"],
                image_paths=tuple(reversed(medguard_kwargs["image_path"])),
            )
        identity = medguard.cache_identity
        delegate = _Delegate()
        writer = CachedStructuredGenerator(
            delegate, GenerationCache(root / "cache", namespace=identity.namespace),
        )
        first = writer.generate_contract_with_key(
            identity.request_key, namespace=identity.namespace, **medguard_kwargs,
        )
        assert delegate.calls == 1

        replay_delegate = _Delegate()
        replay = CachedStructuredGenerator(
            replay_delegate,
            GenerationCache(root / "cache", namespace=identity.namespace),
            allow_remote=False,
        )
        second = replay.generate_contract_with_key(
            a0.cache_identity.request_key,
            namespace=a0.cache_identity.namespace,
            **medguard_kwargs,
        )
        assert first.to_dict() == second.to_dict()
        assert replay_delegate.calls == 0
        assert replay.last_metadata["cache_hit"] is True
        assert replay.last_metadata["remote_calls"] == 0

        b1 = _request(ArmId.B1)
        b1_gateway_kwargs = b1.gateway_kwargs(image_paths=medguard_kwargs["image_path"])
        wrong_namespace = CachedStructuredGenerator(
            _Delegate(), GenerationCache(root / "cache", namespace=identity.namespace),
            allow_remote=False,
        )
        with pytest.raises(CacheIntegrityError, match="namespace"):
            wrong_namespace.generate_contract_with_key(
                b1.cache_identity.request_key,
                namespace=b1.cache_identity.namespace,
                **b1_gateway_kwargs,
            )
        own_namespace = CachedStructuredGenerator(
            _Delegate(),
            GenerationCache(root / "cache", namespace=b1.cache_identity.namespace),
            allow_remote=False,
        )
        with pytest.raises(CacheIntegrityError, match="cache-only generation miss"):
            own_namespace.generate_contract_with_key(
                b1.cache_identity.request_key,
                namespace=b1.cache_identity.namespace,
                **b1_gateway_kwargs,
            )
