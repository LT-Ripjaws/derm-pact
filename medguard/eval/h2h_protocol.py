"""Fail-closed protocol guards for matched H2H validation and accounting.

This module intentionally has no model, retrieval, or scientific-stack imports.  It
defines the host-owned identities that runners must bind before a generated result
can be cached or compared across arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class ProtocolValidationError(ValueError):
    """Raised when an H2H request violates a frozen comparison boundary."""


class ArmId(str, Enum):
    """Closed comparator set for the current H2H study.

    This is the frozen roster of `research/implementation_cards/`: the primary
    comparators B0/B1/B4/B5/B6, the proposed method M, and the paired
    leave-one-component-out ablations A0-A4.  B2 and B3 are deliberately absent:
    they are reference/appendix style arms and must not be constructible here.
    A0 is a declared alias of M and shares M's protocol identity byte-for-byte;
    it is never a separately generated arm.
    """

    B0 = "B0"
    B1 = "B1"
    B4 = "B4"
    B5 = "B5"
    B6 = "B6"
    M = "M"
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"

    @classmethod
    def coerce(cls, value: "ArmId | str") -> "ArmId":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ProtocolValidationError(
                f"unknown H2H arm {value!r}; expected one of "
                f"{', '.join(arm.value for arm in cls)}"
            ) from exc


# Truth is allowed only in a separately joined evaluator, never in an arm request.
# Deliberately avoid broad names such as ``target`` or ``diagnosis`` because they are
# legitimate fields in a frozen perception commitment.
_TRUTH_KEYS = frozenset(
    {
        "truth",
        "label",
        "labels",
        "ground_truth",
        "groundtruth",
        "gold",
        "gold_label",
        "gold_diagnosis",
        "true_label",
        "true_diagnosis",
        "diagnosis_truth",
        "y_true",
    }
)


def _require_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ProtocolValidationError(f"{name} must be non-empty")
    return text


def _require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolValidationError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(name: str, value: Any) -> str:
    digest = _require_text(name, value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ProtocolValidationError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def reject_truth_keys(value: Any, *, path: str = "case") -> None:
    """Recursively reject truth-bearing keys from arm-visible data."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _TRUTH_KEYS or normalized.startswith("ground_truth_"):
                raise ProtocolValidationError(f"truth-bearing key forbidden at {path}.{key}")
            reject_truth_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_truth_keys(item, path=f"{path}[{index}]")


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolValidationError("non-finite floats cannot enter protocol hashes")
        return value
    raise ProtocolValidationError(
        f"unsupported value in protocol hash: {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    """Copy nested hash inputs into immutable containers to prevent TOCTOU drift."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ArmProtocol:
    """Every host-owned input that separates one arm's cache namespace."""

    protocol_id: str
    arm_id: ArmId | str
    model_placeholder: str
    prompt_hash_placeholder: str
    arm_config: Mapping[str, Any]
    decoding: Mapping[str, Any]
    retry_policy: Mapping[str, Any]
    corpus_receipt: str
    perception_receipt: str
    perception_artifact: str
    code_snapshot: str
    validate_only: bool = True
    generation_call_ceiling: int = 0
    token_ceiling: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", ArmId.coerce(self.arm_id))
        for name in (
            "protocol_id",
            "model_placeholder",
            "prompt_hash_placeholder",
            "corpus_receipt",
            "perception_receipt",
            "perception_artifact",
            "code_snapshot",
        ):
            _require_text(name, getattr(self, name))
        if (
            not isinstance(self.arm_config, Mapping)
            or not isinstance(self.decoding, Mapping)
            or not isinstance(self.retry_policy, Mapping)
        ):
            raise ProtocolValidationError(
                "arm_config, decoding, and retry_policy must be mappings"
            )
        _canonical(self.arm_config)
        _canonical(self.decoding)
        _canonical(self.retry_policy)
        object.__setattr__(self, "arm_config", _freeze(self.arm_config))
        object.__setattr__(self, "decoding", _freeze(self.decoding))
        object.__setattr__(self, "retry_policy", _freeze(self.retry_policy))
        calls = _require_nonnegative_int(
            "generation_call_ceiling", self.generation_call_ceiling
        )
        tokens = _require_nonnegative_int("token_ceiling", self.token_ceiling)
        if not isinstance(self.validate_only, bool):
            raise ProtocolValidationError("validate_only must be boolean")
        if self.validate_only and (calls != 0 or tokens != 0):
            raise ProtocolValidationError(
                "validate-only mode requires zero call and token ceilings"
            )

    def namespace_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "medguard-h2h-cache-namespace-v1",
            "protocol_id": self.protocol_id,
            "arm_id": self.arm_id.value,
            "model_placeholder": self.model_placeholder,
            "prompt_hash_placeholder": self.prompt_hash_placeholder,
            "arm_config": self.arm_config,
            "decoding": self.decoding,
            "retry_policy": self.retry_policy,
            "corpus_receipt": self.corpus_receipt,
            "perception_receipt": self.perception_receipt,
            "perception_artifact": self.perception_artifact,
            "code_snapshot": self.code_snapshot,
        }

    @property
    def cache_namespace(self) -> str:
        return f"{self.protocol_id}:{self.arm_id.value}:{_sha256(self.namespace_payload())}"


@dataclass(frozen=True)
class CaseLineage:
    """Truth-free, ordered input lineage shared identically by every arm."""

    case_id: str
    ordered_image_sha256s: Sequence[str]
    frozen_perception: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text("case_id", self.case_id)
        if not self.ordered_image_sha256s:
            raise ProtocolValidationError("ordered_image_sha256s must not be empty")
        for index, digest in enumerate(self.ordered_image_sha256s):
            _require_sha256(f"ordered_image_sha256s[{index}]", digest)
        if not isinstance(self.frozen_perception, Mapping):
            raise ProtocolValidationError("frozen_perception must be a mapping")
        reject_truth_keys(self.frozen_perception, path=f"case[{self.case_id}].perception")
        _canonical(self.frozen_perception)
        object.__setattr__(
            self, "ordered_image_sha256s", tuple(self.ordered_image_sha256s)
        )
        object.__setattr__(self, "frozen_perception", _freeze(self.frozen_perception))

    def comparison_payload(self) -> Mapping[str, Any]:
        return {
            "case_id": self.case_id,
            "ordered_image_sha256s": list(self.ordered_image_sha256s),
            "frozen_perception": self.frozen_perception,
        }


def verify_identical_lineage(
    arm_cases: Mapping[ArmId | str, Sequence[CaseLineage]],
    *,
    expected: Sequence[ArmId | str] | None = None,
) -> None:
    """Require every compared arm to receive byte-identical ordered case lineage.

    ``expected`` pins the exact arm set a runner claims to compare, so a runner
    cannot silently drop an arm and still pass.  It defaults to the supplied
    keys, which is only appropriate for ad-hoc checks; production runners must
    pass their declared roster explicitly.
    """

    normalized: dict[ArmId, Sequence[CaseLineage]] = {}
    for raw_arm, cases in arm_cases.items():
        arm = ArmId.coerce(raw_arm)
        if arm in normalized:
            raise ProtocolValidationError(f"duplicate lineage supplied for {arm.value}")
        normalized[arm] = cases
    if expected is None:
        required = set(normalized)
    else:
        required = {ArmId.coerce(value) for value in expected}
    if len(required) < 2:
        raise ProtocolValidationError(
            "matched lineage verification requires at least two arms"
        )
    if set(normalized) != required:
        missing = sorted(arm.value for arm in required - set(normalized))
        extra = sorted(arm.value for arm in set(normalized) - required)
        roster = "/".join(arm.value for arm in ArmId if arm in required)
        raise ProtocolValidationError(
            f"lineage requires exactly {roster}; missing={missing}, extra={extra}"
        )
    # Compare against the first arm in declaration order so the reference is
    # deterministic regardless of mapping insertion order.
    reference_arm = next(arm for arm in ArmId if arm in normalized)
    reference = [
        _canonical_bytes(case.comparison_payload()) for case in normalized[reference_arm]
    ]
    if not reference:
        raise ProtocolValidationError("H2H lineage cannot be empty")
    reference_ids = [case.case_id for case in normalized[reference_arm]]
    if len(reference_ids) != len(set(reference_ids)):
        raise ProtocolValidationError(f"duplicate case_id in {reference_arm.value} lineage")
    for arm in (item for item in ArmId if item in normalized and item is not reference_arm):
        candidate = [_canonical_bytes(case.comparison_payload()) for case in normalized[arm]]
        if candidate != reference:
            raise ProtocolValidationError(
                f"{arm.value} case/view/perception lineage differs from {reference_arm.value}"
            )


@dataclass(frozen=True)
class EvidenceIdentity:
    evidence_id: str
    content_sha256: str
    source_id: str = ""
    source_version: str = ""
    retrieval_path: Sequence[str] = ()
    record_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text("evidence_id", self.evidence_id)
        _require_sha256("content_sha256", self.content_sha256)
        supplied_lineage = any((
            self.source_id, self.source_version, self.retrieval_path,
            self.record_sha256,
        ))
        if not supplied_lineage:
            return
        _require_text("evidence source_id", self.source_id)
        _require_text("evidence source_version", self.source_version)
        if not self.retrieval_path:
            raise ProtocolValidationError("evidence retrieval_path must not be empty")
        path = tuple(
            _require_text("evidence retrieval_path item", value)
            for value in self.retrieval_path
        )
        _require_sha256("evidence record_sha256", self.record_sha256)
        object.__setattr__(self, "retrieval_path", path)

    @property
    def has_complete_lineage(self) -> bool:
        return bool(
            self.source_id and self.source_version and self.retrieval_path
            and self.record_sha256
        )

    def cache_payload(self) -> Mapping[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content_sha256": self.content_sha256,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "retrieval_path": list(self.retrieval_path),
            "record_sha256": self.record_sha256,
        }


def build_request_key(
    protocol: ArmProtocol,
    lineage: CaseLineage,
    *,
    question: str,
    retrieved_evidence: Sequence[EvidenceIdentity] = (),
    rendered_prompt_sha256: str | None = None,
) -> str:
    """Hash every material generation input, including the rendered prompt.

    Validate-only callers may omit ``rendered_prompt_sha256`` because they do not
    construct a prompt or make a generation request.  A generation-capable
    protocol must provide the byte hash of the fully rendered prompt: the
    namespace binds the static task/adapter template while this key binds the
    case-specific prompt instance.
    """

    question = _require_text("question", question)
    if rendered_prompt_sha256 is None:
        if not protocol.validate_only:
            raise ProtocolValidationError(
                "generation-capable request requires rendered_prompt_sha256"
            )
        prompt_sha256 = protocol.prompt_hash_placeholder
    else:
        prompt_sha256 = _require_sha256(
            "rendered_prompt_sha256", rendered_prompt_sha256
        )
    evidence_ids = [item.evidence_id for item in retrieved_evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ProtocolValidationError("retrieved evidence identities must be unique")
    payload = {
        "schema": "medguard-h2h-request-key-v1",
        "cache_namespace": protocol.cache_namespace,
        "arm_config": protocol.namespace_payload(),
        "case_id": lineage.case_id,
        "ordered_image_sha256s": list(lineage.ordered_image_sha256s),
        "frozen_perception": lineage.frozen_perception,
        "question": question,
        "rendered_prompt_sha256": prompt_sha256,
        "retrieved_evidence": [
            item.cache_payload() for item in retrieved_evidence
        ],
    }
    reject_truth_keys(payload, path="request")
    return _sha256(payload)


@dataclass(frozen=True)
class UsageAccounting:
    actual_generation_calls: int = 0
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    logical_generation_calls: int = 0
    logical_prompt_tokens: int = 0
    logical_completion_tokens: int = 0

    @property
    def actual_total_tokens(self) -> int:
        return self.actual_prompt_tokens + self.actual_completion_tokens

    @property
    def logical_total_tokens(self) -> int:
        return self.logical_prompt_tokens + self.logical_completion_tokens


def enforce_usage(protocol: ArmProtocol, usage: UsageAccounting) -> None:
    """Enforce both protocol ceilings and strict validate-only zero usage."""

    calls = _require_nonnegative_int(
        "actual_generation_calls", usage.actual_generation_calls
    )
    prompt = _require_nonnegative_int(
        "actual_prompt_tokens", usage.actual_prompt_tokens
    )
    completion = _require_nonnegative_int(
        "actual_completion_tokens", usage.actual_completion_tokens
    )
    logical_calls = _require_nonnegative_int(
        "logical_generation_calls", usage.logical_generation_calls
    )
    logical_prompt = _require_nonnegative_int(
        "logical_prompt_tokens", usage.logical_prompt_tokens
    )
    logical_completion = _require_nonnegative_int(
        "logical_completion_tokens", usage.logical_completion_tokens
    )
    total = prompt + completion
    logical_total = logical_prompt + logical_completion
    if calls > protocol.generation_call_ceiling:
        raise ProtocolValidationError(
            f"generation call ceiling exceeded: {calls}>{protocol.generation_call_ceiling}"
        )
    if total > protocol.token_ceiling:
        raise ProtocolValidationError(
            f"token ceiling exceeded: {total}>{protocol.token_ceiling}"
        )
    if protocol.validate_only and (
        calls != 0 or total != 0 or logical_calls != 0 or logical_total != 0
    ):
        raise ProtocolValidationError(
            "validate-only execution used actual or cached generation resources"
        )


@dataclass(frozen=True)
class ScoreSummary:
    total_count: int
    observed_count: int
    missing_count: int
    missing_case_ids: tuple[str, ...]
    mean: float | None


def summarize_scores(scores: Mapping[str, float | int | None]) -> ScoreSummary:
    """Summarize observed scores without turning missing values into zeroes."""

    observed: list[float] = []
    missing: list[str] = []
    for raw_case_id, raw_score in scores.items():
        case_id = _require_text("score case_id", raw_case_id)
        if raw_score is None:
            missing.append(case_id)
            continue
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ProtocolValidationError(f"score for {case_id} must be numeric or None")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ProtocolValidationError(f"score for {case_id} must be finite")
        observed.append(score)
    mean = sum(observed) / len(observed) if observed else None
    return ScoreSummary(
        total_count=len(scores),
        observed_count=len(observed),
        missing_count=len(missing),
        missing_case_ids=tuple(missing),
        mean=mean,
    )
