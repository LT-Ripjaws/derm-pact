"""Shared, offline H2H generation-request construction.

This is deliberately a request foundation, not a runner: it creates the exact
prompt and cache identity that a future authorized generation runner must hand to
the gateway/cache.  It has no API client, credentials, test manifest, or model
call path.  That keeps the current no-generation boundary intact while making the
eventual request materially auditable before it is allowed to run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from medguard.eval.h2h_protocol import (
    ArmId,
    ArmProtocol,
    CaseLineage,
    EvidenceIdentity,
    ProtocolValidationError,
    build_request_key,
    reject_truth_keys,
)


# The clinical question is split into an image-bound part and a source-bound part.
#
# It previously read: "What diagnosis is most consistent with the dermoscopic
# images, and what evidence supports that assessment?"  The guideline-catalog
# feasibility probe showed that second clause asks for something no source in the
# corpus can deliver.  Melanoma has zero assessment-eligible and zero
# visual_finding-eligible records; its diagnosis-tagged records are management
# text that presupposes the diagnosis ("When melanoma is suspected, the whole
# lesion should be completely excised").  Even BCC's strongest visual records are
# claims about method ("Dermatoscopy improves the accuracy of clinical diagnosis
# of BCC"), not about a patient.  A question that asks what external evidence
# supports this patient's assessment therefore penalises an arm for truthfully
# declining, and rewards an arm that conflates method evidence with patient
# evidence -- the exact conflation the EvidenceContract exists to prevent.
#
# The split below asks only what each evidence source can honestly discharge:
# the image carries the diagnosis; authoritative text says whether the diagnostic
# method is guideline-sanctioned and what the guideline directs given that
# commitment.  Changed before any generation ran.  See
# research/experiments/medguard_h2h_v1/guideline_catalog_feasibility_v1.json.
H2H_CLINICAL_QUESTION = (
    "Answer in two separate parts. "
    "(1) Diagnosis: which diagnosis is most consistent with the dermoscopic "
    "images themselves? Ground this in the images and the supplied image "
    "evidence only. "
    "(2) Authoritative guidance: given that committed diagnosis, what does the "
    "supplied authoritative source material state about the diagnostic method or "
    "the indicated management? Cite each such statement by its supplied "
    "identifier and do not restate it beyond what the source says. "
    "Do not present part (2) as proof of part (1): a source statement about a "
    "condition in general never establishes that this patient has it. If no "
    "authoritative statement is supplied for the committed diagnosis, say so and "
    "make no external claim."
)
H2H_DIAGNOSIS_VOCABULARY = (
    "akiec", "bcc", "bkl", "df", "melanoma", "nevus", "vasc",
)
H2H_ORDERED_IMAGE_RULE = (
    "Patient image views are supplied in the frozen first-to-last order. "
    "Interpret their order exactly as provided; do not reorder, drop, or infer "
    "additional views."
)
H2H_GENERATION_SEED = 0


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProtocolValidationError(
        f"unsupported common patient-context value: {type(value).__name__}"
    )


def _require_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ProtocolValidationError(f"{name} must be non-empty")
    return text


def _require_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ProtocolValidationError("seed must be a non-negative integer")
    return seed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SharedClinicalTask:
    """The byte-identical clinical task exposed to every H2H arm."""

    clinical_question: str = H2H_CLINICAL_QUESTION
    diagnosis_vocabulary: tuple[str, ...] = H2H_DIAGNOSIS_VOCABULARY
    ordered_image_rule: str = H2H_ORDERED_IMAGE_RULE

    def __post_init__(self) -> None:
        _require_text("clinical_question", self.clinical_question)
        _require_text("ordered_image_rule", self.ordered_image_rule)
        if not self.diagnosis_vocabulary:
            raise ProtocolValidationError("diagnosis_vocabulary must not be empty")
        vocabulary = tuple(_require_text("diagnosis vocabulary item", item)
                           for item in self.diagnosis_vocabulary)
        if len(vocabulary) != len(set(vocabulary)):
            raise ProtocolValidationError("diagnosis_vocabulary must be unique")
        object.__setattr__(self, "diagnosis_vocabulary", vocabulary)

    @property
    def core_prompt(self) -> str:
        vocabulary = ", ".join(self.diagnosis_vocabulary)
        return (
            "You are a careful dermatology evidence analyst.\n"
            f"Clinical question: {self.clinical_question}\n"
            f"Allowed diagnosis vocabulary: {vocabulary}.\n"
            f"Image order rule: {self.ordered_image_rule}\n"
            "Use the supplied patient-image context and the arm's declared evidence "
            "context only. State uncertainty explicitly; do not invent citations, "
            "provenance, findings, or diagnoses outside the allowed vocabulary."
        )

    @property
    def core_prompt_sha256(self) -> str:
        return hashlib.sha256(self.core_prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArmContextAdapter:
    """The only arm-specific prompt variation: evidence form and obligations."""

    arm_id: ArmId | str
    evidence_representation: str
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        arm = ArmId.coerce(self.arm_id)
        if arm is ArmId.A0:
            raise ProtocolValidationError("A0 must reuse M's prompt adapter")
        object.__setattr__(self, "arm_id", arm)
        _require_text("evidence_representation", self.evidence_representation)
        obligations = tuple(_require_text("adapter obligation", item)
                            for item in self.obligations)
        if not obligations:
            raise ProtocolValidationError("adapter obligations must not be empty")
        object.__setattr__(self, "obligations", obligations)

    def declared_variation(self) -> Mapping[str, Any]:
        """Only fields that an adapter is permitted to alter."""
        return MappingProxyType({
            "evidence_representation": self.evidence_representation,
            "obligations": self.obligations,
        })

    @property
    def declared_variation_sha256(self) -> str:
        return _sha256(dict(self.declared_variation()))

    def render(self, evidence_context: str) -> str:
        evidence_context = _require_text("evidence_context", evidence_context)
        obligations = "\n".join(f"- {item}" for item in self.obligations)
        return (
            "ARM-SPECIFIC CONTEXT ADAPTER (declared variation only)\n"
            f"Evidence representation: {self.evidence_representation}\n"
            f"Obligations:\n{obligations}\n"
            f"Evidence context:\n{evidence_context}"
        )


_CONTRACT_OBLIGATIONS = (
    "Return only the typed EvidenceContract proposal JSON required by the host.",
    "Use only supplied evidence and finding identifiers in contract references.",
    "Declare unresolved uncertainty rather than asserting unsupported evidence.",
)

_ADAPTERS = {
    ArmId.B0: ArmContextAdapter(
        ArmId.B0,
        "No retrieved external evidence; only the frozen patient-image context is available.",
        ("Do not imply that an external source was retrieved or cited.",),
    ),
    ArmId.B1: ArmContextAdapter(
        ArmId.B1,
        "Flat ranked retrieved evidence with source identifiers.",
        ("Cite supplied source identifiers for evidence-dependent statements.",),
    ),
    ArmId.B4: ArmContextAdapter(
        ArmId.B4,
        "Adaptively selected flat image-exemplar and text evidence under the shared ceiling.",
        ("Cite supplied source identifiers for evidence-dependent statements.",),
    ),
    ArmId.B5: ArmContextAdapter(
        ArmId.B5,
        "Graph-retrieved evidence with source/version lineage paths.",
        ("Cite supplied source identifiers and preserve the supplied lineage path.",),
    ),
    ArmId.B6: ArmContextAdapter(
        ArmId.B6,
        "Bounded graph-refinement evidence with the recorded sufficiency state.",
        (
            "Cite supplied source identifiers and preserve the supplied lineage path.",
            "State insufficient evidence rather than using an ungrounded fallback.",
        ),
    ),
    ArmId.M: ArmContextAdapter(
        ArmId.M,
        "EvidenceContract evidence with hybrid retrieval and provenance lineage.",
        _CONTRACT_OBLIGATIONS,
    ),
    ArmId.A1: ArmContextAdapter(
        ArmId.A1,
        "EvidenceContract evidence with authoritative text retrieval and no image-exemplar evidence.",
        _CONTRACT_OBLIGATIONS,
    ),
    ArmId.A2: ArmContextAdapter(
        ArmId.A2,
        "EvidenceContract evidence with flat source/version/hash provenance and no graph lineage.",
        _CONTRACT_OBLIGATIONS,
    ),
    ArmId.A3: ArmContextAdapter(
        ArmId.A3,
        "EvidenceContract evidence with bounded remediation disabled.",
        _CONTRACT_OBLIGATIONS,
    ),
    ArmId.A4: ArmContextAdapter(
        ArmId.A4,
        "EvidenceContract evidence with verifier observations recorded but ungated.",
        _CONTRACT_OBLIGATIONS,
    ),
}
H2H_ARM_ADAPTERS = MappingProxyType(_ADAPTERS)


def effective_arm_id(arm_id: ArmId | str) -> ArmId:
    """Return M for the one declared execution alias, otherwise the given arm."""
    arm = ArmId.coerce(arm_id)
    return ArmId.M if arm is ArmId.A0 else arm


def adapter_for_arm(arm_id: ArmId | str) -> ArmContextAdapter:
    return H2H_ARM_ADAPTERS[effective_arm_id(arm_id)]


@dataclass(frozen=True)
class PromptAssembly:
    """A prompt with an inspectable shared core and a bounded adapter section."""

    task: SharedClinicalTask
    adapter: ArmContextAdapter
    common_patient_context: Mapping[str, Any]
    evidence_context: str

    def __post_init__(self) -> None:
        if not isinstance(self.common_patient_context, Mapping):
            raise ProtocolValidationError("common_patient_context must be a mapping")
        reject_truth_keys(self.common_patient_context, path="common_patient_context")
        _json_value(self.common_patient_context)
        _require_text("evidence_context", self.evidence_context)

    @property
    def template_sha256(self) -> str:
        return _sha256({
            "core_prompt_sha256": self.task.core_prompt_sha256,
            "declared_adapter_variation": dict(self.adapter.declared_variation()),
        })

    @property
    def rendered_prompt(self) -> str:
        patient_context = json.dumps(
            _json_value(self.common_patient_context), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        )
        return (
            f"{self.task.core_prompt}\n\n"
            "COMMON PATIENT CONTEXT (identical across arms for this case)\n"
            f"{patient_context}\n\n"
            f"{self.adapter.render(self.evidence_context)}"
        )

    @property
    def rendered_prompt_sha256(self) -> str:
        return hashlib.sha256(self.rendered_prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class H2HCacheIdentity:
    """Replay identity for one arm/case generation request."""

    namespace: str
    request_key: str
    execution_arm_id: ArmId
    alias_of: ArmId | None = None

    def __post_init__(self) -> None:
        _require_text("cache namespace", self.namespace)
        if len(self.request_key) != 64 or any(
            char not in "0123456789abcdef" for char in self.request_key
        ):
            raise ProtocolValidationError("request_key must be a lowercase SHA-256 digest")
        if self.alias_of is not None and self.alias_of is not ArmId.M:
            raise ProtocolValidationError("the only H2H cache alias is A0 -> M")


@dataclass(frozen=True)
class H2HGenerationRequest:
    """One generation-capable, cache-addressable H2H request.

    The request never calls a model.  ``gateway_kwargs`` validates the ordered
    image material before exposing the exact ``prompt`` and ``seed`` for every
    arm.  ``structured_generator_kwargs`` additionally binds evidence chunks for
    the MedGuard EvidenceContract arms only.
    """

    arm_id: ArmId | str
    protocol: ArmProtocol
    lineage: CaseLineage
    task: SharedClinicalTask
    adapter: ArmContextAdapter
    common_patient_context: Mapping[str, Any]
    evidence_context: str
    retrieved_evidence: tuple[EvidenceIdentity, ...]
    seed: int = H2H_GENERATION_SEED

    def __post_init__(self) -> None:
        arm = ArmId.coerce(self.arm_id)
        object.__setattr__(self, "arm_id", arm)
        expected_execution_arm = effective_arm_id(arm)
        if self.protocol.arm_id is not expected_execution_arm:
            raise ProtocolValidationError(
                f"{arm.value} must use {expected_execution_arm.value}'s execution protocol"
            )
        if self.adapter.arm_id is not expected_execution_arm:
            raise ProtocolValidationError(
                f"{arm.value} must use {expected_execution_arm.value}'s prompt adapter"
            )
        if self.protocol.validate_only:
            raise ProtocolValidationError(
                "validate-only protocol cannot construct a generation request"
            )
        seed = _require_seed(self.seed)
        object.__setattr__(self, "seed", seed)
        if self.protocol.decoding.get("seed") != seed:
            raise ProtocolValidationError(
                "protocol decoding seed must equal the request seed"
            )
        if not self.retrieved_evidence:
            object.__setattr__(self, "retrieved_evidence", tuple())
        else:
            object.__setattr__(
                self, "retrieved_evidence", tuple(self.retrieved_evidence)
            )
        evidence_ids = [item.evidence_id for item in self.retrieved_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ProtocolValidationError("retrieved_evidence identities must be unique")
        if any(not item.has_complete_lineage for item in self.retrieved_evidence):
            raise ProtocolValidationError(
                "generation-capable evidence identity requires source/version/path/hash lineage"
            )
        if self.protocol.prompt_hash_placeholder != self.prompt.template_sha256:
            raise ProtocolValidationError(
                "protocol prompt hash must bind the shared core and this arm adapter"
            )
        if _json_value(self.common_patient_context) != _json_value(self.lineage.frozen_perception):
            raise ProtocolValidationError(
                "common patient context must equal the frozen perception lineage"
            )

    @property
    def prompt(self) -> PromptAssembly:
        return PromptAssembly(
            task=self.task,
            adapter=self.adapter,
            common_patient_context=self.common_patient_context,
            evidence_context=self.evidence_context,
        )

    @property
    def cache_identity(self) -> H2HCacheIdentity:
        return H2HCacheIdentity(
            namespace=self.protocol.cache_namespace,
            request_key=build_request_key(
                self.protocol,
                self.lineage,
                question=self.task.clinical_question,
                retrieved_evidence=self.retrieved_evidence,
                rendered_prompt_sha256=self.prompt.rendered_prompt_sha256,
            ),
            execution_arm_id=effective_arm_id(self.arm_id),
            alias_of=ArmId.M if self.arm_id is ArmId.A0 else None,
        )

    def gateway_kwargs(
        self,
        *,
        image_paths: str | Sequence[str],
    ) -> Mapping[str, Any]:
        """Return the common gateway request after frozen view-order validation."""
        paths = ((str(image_paths),) if isinstance(image_paths, str)
                 else tuple(str(path) for path in image_paths))
        if len(paths) != len(self.lineage.ordered_image_sha256s):
            raise ProtocolValidationError("generation image count differs from frozen lineage")
        observed_hashes = []
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"generation image not found: {path}")
            observed_hashes.append(_sha256_file(path))
        if tuple(observed_hashes) != tuple(self.lineage.ordered_image_sha256s):
            raise ProtocolValidationError("generation image order/content differs from frozen lineage")

        return MappingProxyType({
            "prompt": self.prompt.rendered_prompt,
            "image_path": paths,
            "seed": self.seed,
        })

    def structured_generator_kwargs(
        self,
        *,
        modality: str,
        perception: Any,
        chunks: Sequence[Mapping[str, Any]],
        image_paths: str | Sequence[str],
    ) -> Mapping[str, Any]:
        """Return EvidenceContract-generator inputs for M/A0-A4 only."""
        if effective_arm_id(self.arm_id) not in {
            ArmId.M, ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4,
        }:
            raise ProtocolValidationError(
                "structured generator kwargs are only valid for EvidenceContract arms"
            )
        gateway = self.gateway_kwargs(image_paths=image_paths)

        if len(chunks) != len(self.retrieved_evidence):
            raise ProtocolValidationError("generation chunks differ from cache-bound evidence")
        for expected, chunk in zip(self.retrieved_evidence, chunks):
            evidence_id = str(chunk.get("document_id") or chunk.get("evidence_id") or "")
            content_sha256 = str(chunk.get("content_sha256") or "")
            source_id = str(chunk.get("source_id") or chunk.get("source") or "")
            source_version = str(chunk.get("source_version") or chunk.get("version") or "")
            retrieval_path = tuple(str(value) for value in (chunk.get("retrieval_path") or ()))
            record_sha256 = str(chunk.get("record_sha256") or "")
            if (
                evidence_id, content_sha256, source_id, source_version,
                retrieval_path, record_sha256,
            ) != (
                expected.evidence_id, expected.content_sha256, expected.source_id,
                expected.source_version, tuple(expected.retrieval_path),
                expected.record_sha256,
            ):
                raise ProtocolValidationError(
                    "generation chunk differs from cache-bound evidence identity"
                )

        return MappingProxyType({
            "case_id": self.lineage.case_id,
            "modality": _require_text("modality", modality),
            "question": self.task.clinical_question,
            "perception": perception,
            "chunks": tuple(dict(chunk) for chunk in chunks),
            **dict(gateway),
        })
